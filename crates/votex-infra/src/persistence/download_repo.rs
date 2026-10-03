use rusqlite::params;
use std::sync::{Arc, Mutex};
use votex_domain::error::DomainError;

/// 下载记录
///
/// 追踪单个模型文件的下载进度，支持断点续传。
/// 每个文件一条记录，由 (model_id, file_name) 唯一标识。
#[derive(Debug, Clone)]
pub struct DownloadRecord {
    /// 所属模型 ID（如 "kokoro-82m"）
    pub model_id: String,
    /// 文件名（如 "model.onnx"）
    pub file_name: String,
    /// 下载源 URL
    pub url: String,
    /// 已下载字节数
    pub bytes_downloaded: u64,
    /// 文件总大小（字节），0 表示未知
    pub total_bytes: u64,
    /// 状态：Pending / Downloading / Paused / Completed / Failed
    pub status: String,
    /// 错误信息
    pub error_message: Option<String>,
    /// 创建时间（Unix 时间戳）
    pub created_at: String,
    /// 更新时间（Unix 时间戳）
    pub updated_at: String,
}

/// SQLite 下载记录仓储
///
/// 表结构：
///   downloads(
///     model_id        TEXT NOT NULL,
///     file_name       TEXT NOT NULL,
///     url             TEXT NOT NULL,
///     bytes_downloaded INTEGER NOT NULL DEFAULT 0,
///     total_bytes     INTEGER NOT NULL DEFAULT 0,
///     status          TEXT NOT NULL DEFAULT 'Pending',
///     error_message   TEXT,
///     created_at      TEXT NOT NULL,
///     updated_at      TEXT NOT NULL,
///     PRIMARY KEY (model_id, file_name)
///   )
#[derive(Clone)]
pub struct SqliteDownloadRepository {
    conn: Arc<Mutex<rusqlite::Connection>>,
}

impl SqliteDownloadRepository {
    /// 创建共享连接的下载记录仓储
    pub fn new(conn: Arc<Mutex<rusqlite::Connection>>) -> Self {
        Self { conn }
    }

    /// 当前时间戳（Unix 秒）
    fn now() -> String {
        format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        )
    }

    /// 插入或更新下载记录
    pub fn upsert(
        &self,
        model_id: &str,
        file_name: &str,
        url: &str,
        bytes_downloaded: u64,
        total_bytes: u64,
        status: &str,
        error_message: Option<&str>,
    ) -> Result<(), DomainError> {
        let conn = self.conn.lock().map_err(|e| {
            DomainError::Config(crate::config_err(&format!("加锁失败: {}", e)))
        })?;

        let now = Self::now();

        // 获取已存在的创建时间
        let created_at: String = conn
            .query_row(
                "SELECT created_at FROM downloads WHERE model_id = ?1 AND file_name = ?2",
                params![model_id, file_name],
                |row| row.get(0),
            )
            .unwrap_or_else(|_| now.clone());

        conn.execute(
            "INSERT OR REPLACE INTO downloads
                (model_id, file_name, url, bytes_downloaded, total_bytes, status, error_message, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                model_id,
                file_name,
                url,
                bytes_downloaded as i64,
                total_bytes as i64,
                status,
                error_message,
                created_at,
                now,
            ],
        )
        .map_err(|e| DomainError::Config(crate::config_err(&format!("保存下载记录失败: {}", e))))?;

        tracing::debug!(
            "下载记录已保存: {}/{} ({}%)",
            model_id,
            file_name,
            if total_bytes > 0 {
                bytes_downloaded * 100 / total_bytes
            } else {
                0
            }
        );
        Ok(())
    }

    /// 按模型 ID 查找所有下载记录
    pub fn find_by_model(&self, model_id: &str) -> Vec<DownloadRecord> {
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        let mut stmt = match conn.prepare(
            "SELECT model_id, file_name, url, bytes_downloaded, total_bytes, status,
                    error_message, created_at, updated_at
             FROM downloads WHERE model_id = ?1
             ORDER BY file_name",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let rows = match stmt.query_map(params![model_id], |row| {
            Ok(DownloadRecord {
                model_id: row.get(0)?,
                file_name: row.get(1)?,
                url: row.get(2)?,
                bytes_downloaded: row.get::<_, i64>(3)? as u64,
                total_bytes: row.get::<_, i64>(4)? as u64,
                status: row.get(5)?,
                error_message: row.get(6)?,
                created_at: row.get(7)?,
                updated_at: row.get(8)?,
            })
        }) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        rows.filter_map(|r| r.ok()).collect()
    }

    /// 查找所有未完成的下载（用于启动恢复）
    pub fn find_incomplete(&self) -> Vec<DownloadRecord> {
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        let mut stmt = match conn.prepare(
            "SELECT model_id, file_name, url, bytes_downloaded, total_bytes, status,
                    error_message, created_at, updated_at
             FROM downloads
             WHERE status IN ('Pending', 'Downloading', 'Paused')
             ORDER BY created_at",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let rows = match stmt.query_map([], |row| {
            Ok(DownloadRecord {
                model_id: row.get(0)?,
                file_name: row.get(1)?,
                url: row.get(2)?,
                bytes_downloaded: row.get::<_, i64>(3)? as u64,
                total_bytes: row.get::<_, i64>(4)? as u64,
                status: row.get(5)?,
                error_message: row.get(6)?,
                created_at: row.get(7)?,
                updated_at: row.get(8)?,
            })
        }) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        rows.filter_map(|r| r.ok()).collect()
    }

    /// 查找所有下载记录（按模型分组）
    pub fn list_all(&self) -> Vec<DownloadRecord> {
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        let mut stmt = match conn.prepare(
            "SELECT model_id, file_name, url, bytes_downloaded, total_bytes, status,
                    error_message, created_at, updated_at
             FROM downloads
             ORDER BY model_id, file_name",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let rows = match stmt.query_map([], |row| {
            Ok(DownloadRecord {
                model_id: row.get(0)?,
                file_name: row.get(1)?,
                url: row.get(2)?,
                bytes_downloaded: row.get::<_, i64>(3)? as u64,
                total_bytes: row.get::<_, i64>(4)? as u64,
                status: row.get(5)?,
                error_message: row.get(6)?,
                created_at: row.get(7)?,
                updated_at: row.get(8)?,
            })
        }) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        rows.filter_map(|r| r.ok()).collect()
    }

    /// 删除下载记录
    pub fn delete(&self, model_id: &str, file_name: &str) -> Result<(), DomainError> {
        let conn = self.conn.lock().map_err(|e| {
            DomainError::Config(crate::config_err(&format!("加锁失败: {}", e)))
        })?;

        conn.execute(
            "DELETE FROM downloads WHERE model_id = ?1 AND file_name = ?2",
            params![model_id, file_name],
        )
        .map_err(|e| DomainError::Config(crate::config_err(&format!("删除下载记录失败: {}", e))))?;

        Ok(())
    }

    /// 删除某个模型的所有下载记录
    pub fn delete_by_model(&self, model_id: &str) -> Result<(), DomainError> {
        let conn = self.conn.lock().map_err(|e| {
            DomainError::Config(crate::config_err(&format!("加锁失败: {}", e)))
        })?;

        conn.execute("DELETE FROM downloads WHERE model_id = ?1", params![model_id])
            .map_err(|e| DomainError::Config(crate::config_err(&format!("删除模型下载记录失败: {}", e))))?;

        Ok(())
    }

    /// 更新下载进度
    pub fn update_progress(
        &self,
        model_id: &str,
        file_name: &str,
        bytes_downloaded: u64,
        total_bytes: u64,
    ) -> Result<(), DomainError> {
        let conn = self.conn.lock().map_err(|e| {
            DomainError::Config(crate::config_err(&format!("加锁失败: {}", e)))
        })?;

        let now = Self::now();
        conn.execute(
            "UPDATE downloads SET bytes_downloaded = ?1, total_bytes = ?2, updated_at = ?3
             WHERE model_id = ?4 AND file_name = ?5",
            params![bytes_downloaded as i64, total_bytes as i64, now, model_id, file_name],
        )
        .map_err(|e| DomainError::Config(crate::config_err(&format!("更新下载进度失败: {}", e))))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_repo() -> SqliteDownloadRepository {
        SqliteDownloadRepository::new(crate::persistence::db::open_in_memory_database().unwrap())
    }

    #[test]
    fn sqlite_插入和查找() {
        let repo = setup_repo();
        repo.upsert("kokoro-82m", "model.onnx", "https://example.com/model.onnx", 0, 100, "Pending", None)
            .unwrap();

        let records = repo.find_by_model("kokoro-82m");
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].file_name, "model.onnx");
        assert_eq!(records[0].status, "Pending");
    }

    #[test]
    fn sqlite_更新进度() {
        let repo = setup_repo();
        repo.upsert("kokoro-82m", "model.onnx", "https://example.com/model.onnx", 0, 100, "Downloading", None)
            .unwrap();

        repo.update_progress("kokoro-82m", "model.onnx", 50, 100).unwrap();

        let records = repo.find_by_model("kokoro-82m");
        assert_eq!(records[0].bytes_downloaded, 50);
        assert_eq!(records[0].status, "Downloading");
    }

    #[test]
    fn sqlite_标记完成() {
        let repo = setup_repo();
        repo.upsert("kokoro-82m", "model.onnx", "https://example.com/model.onnx", 100, 100, "Completed", None)
            .unwrap();

        let records = repo.find_incomplete();
        assert!(records.is_empty());
    }

    #[test]
    fn sqlite_查找未完成() {
        let repo = setup_repo();
        repo.upsert("kokoro-82m", "model.onnx", "https://example.com/model.onnx", 0, 100, "Downloading", None)
            .unwrap();
        repo.upsert("whisper-base", "model.onnx", "https://example.com/whisper.onnx", 0, 200, "Pending", None)
            .unwrap();
        repo.upsert("sensevoice", "model.onnx", "https://example.com/sensevoice.onnx", 200, 200, "Completed", None)
            .unwrap();

        let incomplete = repo.find_incomplete();
        assert_eq!(incomplete.len(), 2);
    }

    #[test]
    fn sqlite_删除() {
        let repo = setup_repo();
        repo.upsert("kokoro-82m", "model.onnx", "https://example.com/model.onnx", 0, 100, "Pending", None)
            .unwrap();
        repo.delete("kokoro-82m", "model.onnx").unwrap();

        let records = repo.find_by_model("kokoro-82m");
        assert!(records.is_empty());
    }

    #[test]
    fn sqlite_删除模型所有记录() {
        let repo = setup_repo();
        repo.upsert("kokoro-82m", "file1.onnx", "https://example.com/1", 0, 100, "Pending", None).unwrap();
        repo.upsert("kokoro-82m", "file2.onnx", "https://example.com/2", 0, 100, "Pending", None).unwrap();
        repo.upsert("whisper-base", "model.onnx", "https://example.com/3", 0, 100, "Completed", None).unwrap();

        repo.delete_by_model("kokoro-82m").unwrap();

        assert_eq!(repo.find_by_model("kokoro-82m").len(), 0);
        assert_eq!(repo.find_by_model("whisper-base").len(), 1);
    }

    #[test]
    fn sqlite_重复写入保留创建时间() {
        let repo = setup_repo();
        repo.upsert("kokoro-82m", "model.onnx", "https://example.com/1", 0, 100, "Pending", None).unwrap();

        // 等待至少 1 秒以确保时间戳变化（实际上时间精度是秒）
        std::thread::sleep(std::time::Duration::from_millis(10));

        // 再次 upsert，创建时间应保持不变
        repo.upsert("kokoro-82m", "model.onnx", "https://example.com/2", 50, 100, "Downloading", None).unwrap();

        let records = repo.find_by_model("kokoro-82m");
        assert_eq!(records.len(), 1);
        // updated_at 应大于 created_at
        assert!(records[0].updated_at >= records[0].created_at);
    }
}
