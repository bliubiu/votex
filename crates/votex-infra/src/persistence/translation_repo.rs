//! 翻译任务与翻译历史仓储
//!
//! 两张表：
//! - `translation_tasks`：批量翻译任务（与 TTS/ASR/OCR 任务表同构，JSON 存储）
//! - `translation_history`：单条翻译记录，用于**跨会话复用译文**
//!   （内存缓存只在进程内有效，落库后重启也能命中）

use rusqlite::params;
use std::sync::{Arc, Mutex};

/// 一条翻译历史记录
#[derive(Debug, Clone)]
pub struct TranslationRecord {
    /// 自增主键
    pub id: i64,
    pub engine: String,
    pub direction: String,
    pub source_text: String,
    pub target_text: String,
    /// 术语表指纹（0 表示无术语表）
    pub glossary_fp: i64,
    pub created_at: String,
}

/// SQLite 翻译仓储
pub struct SqliteTranslationRepository {
    conn: Arc<Mutex<rusqlite::Connection>>,
}

impl SqliteTranslationRepository {
    /// 创建仓储
    pub fn new(conn: Arc<Mutex<rusqlite::Connection>>) -> Self {
        Self { conn }
    }

    /// 保存一条翻译记录
    pub fn save_record(
        &self,
        engine: &str,
        direction: &str,
        source_text: &str,
        target_text: &str,
        glossary_fp: u64,
    ) -> Result<(), String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("获取数据库连接失败: {}", e))?;

        conn.execute(
            "INSERT INTO translation_history
             (engine, direction, source_text, target_text, glossary_fp, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, datetime('now'))",
            params![
                engine,
                direction,
                source_text,
                target_text,
                glossary_fp as i64
            ],
        )
        .map_err(|e| format!("保存翻译记录失败: {}", e))?;

        Ok(())
    }

    /// 查询最近一条匹配的翻译记录
    ///
    /// 按 `(engine, direction, glossary_fp, source_text)` 精确匹配。
    pub fn find_recent(
        &self,
        engine: &str,
        direction: &str,
        source_text: &str,
        glossary_fp: u64,
    ) -> Option<String> {
        let conn = self.conn.lock().ok()?;

        let mut stmt = conn
            .prepare(
                "SELECT target_text FROM translation_history
                 WHERE engine = ?1 AND direction = ?2
                   AND source_text = ?3 AND glossary_fp = ?4
                 ORDER BY id DESC LIMIT 1",
            )
            .ok()?;

        let result: rusqlite::Result<String> = stmt.query_row(
            params![engine, direction, source_text, glossary_fp as i64],
            |row| row.get(0),
        );

        result.ok()
    }

    /// 列出最近的翻译记录
    pub fn list_recent(&self, limit: usize) -> Vec<TranslationRecord> {
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        let mut stmt = match conn.prepare(
            "SELECT id, engine, direction, source_text, target_text, glossary_fp, created_at
             FROM translation_history ORDER BY id DESC LIMIT ?1",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let rows = stmt.query_map(params![limit as i64], |row| {
            Ok(TranslationRecord {
                id: row.get(0)?,
                engine: row.get(1)?,
                direction: row.get(2)?,
                source_text: row.get(3)?,
                target_text: row.get(4)?,
                glossary_fp: row.get(5)?,
                created_at: row.get(6)?,
            })
        });

        match rows {
            Ok(r) => r.filter_map(|x| x.ok()).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// 清空翻译历史
    pub fn clear_history(&self) -> Result<(), String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("获取数据库连接失败: {}", e))?;
        conn.execute("DELETE FROM translation_history", [])
            .map_err(|e| format!("清空翻译历史失败: {}", e))?;
        Ok(())
    }

    /// 翻译历史条数
    pub fn history_count(&self) -> usize {
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return 0,
        };
        conn.query_row("SELECT COUNT(*) FROM translation_history", [], |r| r.get(0))
            .unwrap_or(0)
    }

    // ============ 批量翻译任务 ============

    /// 保存翻译任务（JSON 序列化）
    pub fn save_task(
        &self,
        id: &str,
        data_json: &str,
        status: &str,
    ) -> Result<(), String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("获取数据库连接失败: {}", e))?;

        conn.execute(
            "INSERT INTO translation_tasks (id, data_json, status, created_at, updated_at)
             VALUES (?1, ?2, ?3, datetime('now'), datetime('now'))
             ON CONFLICT(id) DO UPDATE SET
                data_json = excluded.data_json,
                status = excluded.status,
                updated_at = datetime('now')",
            params![id, data_json, status],
        )
        .map_err(|e| format!("保存翻译任务失败: {}", e))?;

        Ok(())
    }

    /// 读取翻译任务
    pub fn load_task(&self, id: &str) -> Option<(String, String)> {
        let conn = self.conn.lock().ok()?;
        let mut stmt = conn
            .prepare("SELECT data_json, status FROM translation_tasks WHERE id = ?1")
            .ok()?;
        stmt.query_row(params![id], |row| Ok((row.get(0)?, row.get(1)?)))
            .ok()
    }

    /// 列出翻译任务
    pub fn list_tasks(&self) -> Vec<(String, String, String)> {
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };
        let mut stmt = match conn.prepare(
            "SELECT id, data_json, status FROM translation_tasks ORDER BY created_at DESC",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };
        let rows = stmt.query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        });
        match rows {
            Ok(r) => r.filter_map(|x| x.ok()).collect(),
            Err(_) => Vec::new(),
        }
    }

    /// 更新任务状态
    pub fn update_task_status(&self, id: &str, status: &str) -> Result<(), String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("获取数据库连接失败: {}", e))?;
        conn.execute(
            "UPDATE translation_tasks SET status = ?1, updated_at = datetime('now') WHERE id = ?2",
            params![status, id],
        )
        .map_err(|e| format!("更新翻译任务状态失败: {}", e))?;
        Ok(())
    }

    /// 删除翻译任务
    pub fn delete_task(&self, id: &str) -> Result<(), String> {
        let conn = self
            .conn
            .lock()
            .map_err(|e| format!("获取数据库连接失败: {}", e))?;
        conn.execute("DELETE FROM translation_tasks WHERE id = ?1", params![id])
            .map_err(|e| format!("删除翻译任务失败: {}", e))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn open_db() -> Arc<Mutex<rusqlite::Connection>> {
        crate::persistence::db::open_in_memory_database().expect("初始化内存数据库失败")
    }

    #[test]
    fn 翻译记录_保存与查询() {
        let conn = open_db();
        let repo = SqliteTranslationRepository::new(conn);
        repo.save_record("dict", "zh-en", "你好", "hello", 0).unwrap();
        assert_eq!(repo.find_recent("dict", "zh-en", "你好", 0).as_deref(), Some("hello"));
    }

    #[test]
    fn 翻译记录_术语表不同则不命中() {
        let conn = open_db();
        let repo = SqliteTranslationRepository::new(conn);
        repo.save_record("dict", "zh-en", "你好", "hello", 0).unwrap();
        assert!(repo.find_recent("dict", "zh-en", "你好", 123).is_none());
    }

    #[test]
    fn 翻译记录_引擎不同则不命中() {
        let conn = open_db();
        let repo = SqliteTranslationRepository::new(conn);
        repo.save_record("dict", "zh-en", "你好", "hello", 0).unwrap();
        assert!(repo.find_recent("opus-mt", "zh-en", "你好", 0).is_none());
    }

    #[test]
    fn 翻译记录_列出最近记录() {
        let conn = open_db();
        let repo = SqliteTranslationRepository::new(conn);
        repo.save_record("dict", "zh-en", "a", "A", 0).unwrap();
        repo.save_record("dict", "zh-en", "b", "B", 0).unwrap();
        let list = repo.list_recent(10);
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].source_text, "b", "应按 id 倒序，最新在前");
        assert_eq!(repo.history_count(), 2);
    }

    #[test]
    fn 翻译记录_清空() {
        let conn = open_db();
        let repo = SqliteTranslationRepository::new(conn);
        repo.save_record("dict", "zh-en", "a", "A", 0).unwrap();
        repo.clear_history().unwrap();
        assert_eq!(repo.history_count(), 0);
    }

    #[test]
    fn 翻译任务_保存读取与更新() {
        let conn = open_db();
        let repo = SqliteTranslationRepository::new(conn);
        repo.save_task("t1", "{\"a\":1}", "Queued").unwrap();

        let (data, status) = repo.load_task("t1").unwrap();
        assert_eq!(data, "{\"a\":1}");
        assert_eq!(status, "Queued");

        repo.update_task_status("t1", "Completed").unwrap();
        assert_eq!(repo.load_task("t1").unwrap().1, "Completed");
    }

    #[test]
    fn 翻译任务_重复保存为更新() {
        let conn = open_db();
        let repo = SqliteTranslationRepository::new(conn);
        repo.save_task("t1", "{\"a\":1}", "Queued").unwrap();
        repo.save_task("t1", "{\"a\":2}", "Running").unwrap();
        assert_eq!(repo.list_tasks().len(), 1, "同 id 应覆盖而非新增");
        assert_eq!(repo.load_task("t1").unwrap().0, "{\"a\":2}");
    }

    #[test]
    fn 翻译任务_删除() {
        let conn = open_db();
        let repo = SqliteTranslationRepository::new(conn);
        repo.save_task("t1", "{}", "Queued").unwrap();
        repo.delete_task("t1").unwrap();
        assert!(repo.load_task("t1").is_none());
    }

    #[test]
    fn 翻译任务_列出() {
        let conn = open_db();
        let repo = SqliteTranslationRepository::new(conn);
        repo.save_task("t1", "{}", "Queued").unwrap();
        repo.save_task("t2", "{}", "Completed").unwrap();
        assert_eq!(repo.list_tasks().len(), 2);
    }
}
