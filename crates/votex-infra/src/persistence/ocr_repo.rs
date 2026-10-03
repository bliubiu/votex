use rusqlite::params;
use std::sync::{Arc, Mutex};
use votex_domain::error::DomainError;
use votex_domain::ocr::entity::OcrTask;
use votex_domain::ocr::value_object::{OcrTextBlock, PageStatus};
use votex_domain::repository::OcrTaskRepository;
use votex_domain::shared::value_object::TaskId;

/// SQLite OCR 任务仓储
///
/// 将 OcrTask 序列化为 JSON 存入 SQLite，支持跨会话持久化。
/// 表结构：
///   ocr_tasks(id TEXT PK, data_json TEXT, created_at TEXT, updated_at TEXT)
pub struct SqliteOcrTaskRepository {
    conn: Arc<Mutex<rusqlite::Connection>>,
}

impl SqliteOcrTaskRepository {
    /// 创建共享连接的 OCR 任务仓储
    pub fn new(conn: Arc<Mutex<rusqlite::Connection>>) -> Self {
        Self { conn }
    }
}

impl OcrTaskRepository for SqliteOcrTaskRepository {
    fn find_by_id(&self, id: &TaskId) -> Option<OcrTask> {
        let conn = self.conn.lock().ok()?;
        let id_str = id.as_str();
        conn.query_row(
            "SELECT data_json FROM ocr_tasks WHERE id = ?1",
            params![id_str],
            |row| {
                let json: String = row.get(0)?;
                serde_json::from_str::<OcrTask>(&json)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
            },
        )
        .ok()
    }

    fn save(&self, task: &OcrTask) -> Result<(), DomainError> {
        let conn = self.conn.lock().map_err(|e| {
            DomainError::Config(crate::config_err(&format!("加锁失败: {}", e)))
        })?;

        let id = task.id.as_str();
        let json = serde_json::to_string(task).map_err(|e| {
            DomainError::Config(crate::config_err(&format!("序列化失败: {}", e)))
        })?;
        let created = &task.created_at;
        let updated = &task.updated_at;

        let status_str = match &task.status {
            votex_domain::shared::value_object::TaskStatus::Queued => "Queued",
            votex_domain::shared::value_object::TaskStatus::Running => "Running",
            votex_domain::shared::value_object::TaskStatus::Completed => "Completed",
            votex_domain::shared::value_object::TaskStatus::Failed(_) => "Failed",
            votex_domain::shared::value_object::TaskStatus::Cancelled => "Cancelled",
            votex_domain::shared::value_object::TaskStatus::Paused => "Paused",
        };

        conn.execute(
            "INSERT OR REPLACE INTO ocr_tasks (id, data_json, status, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, json, status_str, created, updated],
        )
        .map_err(|e| DomainError::Config(crate::config_err(&format!("保存任务失败: {}", e))))?;

        // 同步写入页面表，方便单页更新
        for page in &task.pages {
            let blocks_json = page.blocks.as_ref().map(|b| serde_json::to_string(b).unwrap_or_default());
            let status_str = match &page.status {
                PageStatus::Pending => "Pending",
                PageStatus::Processing => "Processing",
                PageStatus::Completed => "Completed",
                PageStatus::Failed(_) => "Failed",
                PageStatus::Skipped => "Skipped",
            };
            let error_msg = match &page.status {
                PageStatus::Failed(msg) => Some(msg.as_str()),
                _ => None,
            };

            conn.execute(
                "INSERT OR REPLACE INTO ocr_pages
                    (task_id, page_index, status, blocks_json, confidence, retry_count, error_msg)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    id,
                    page.index as i64,
                    status_str,
                    blocks_json,
                    page.confidence,
                    page.retry_count as i64,
                    error_msg,
                ],
            )
            .map_err(|e| {
                DomainError::Config(crate::config_err(&format!("保存页面失败: {}", e)))
            })?;
        }

        tracing::debug!("OCR 任务已保存: {}", id);
        Ok(())
    }

    fn delete(&self, id: &TaskId) -> Result<(), DomainError> {
        let conn = self.conn.lock().map_err(|e| {
            DomainError::Config(crate::config_err(&format!("加锁失败: {}", e)))
        })?;

        let id_str = id.as_str();
        conn.execute("DELETE FROM ocr_pages WHERE task_id = ?1", params![id_str])
            .map_err(|e| DomainError::Config(crate::config_err(&format!("删除页面失败: {}", e))))?;
        conn.execute("DELETE FROM ocr_tasks WHERE id = ?1", params![id_str])
            .map_err(|e| DomainError::Config(crate::config_err(&format!("删除任务失败: {}", e))))?;

        tracing::debug!("OCR 任务已删除: {}", id);
        Ok(())
    }

    fn find_incomplete(&self) -> Vec<OcrTask> {
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        let mut stmt = match conn.prepare(
            "SELECT data_json FROM ocr_tasks WHERE status IN ('Queued', 'Running', 'Paused')",
        ) {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let rows = match stmt.query_map([], |row| {
            let json: String = row.get(0)?;
            serde_json::from_str::<OcrTask>(&json)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        }) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        rows.filter_map(|r| r.ok()).collect()
    }

    fn list_all(&self) -> Vec<OcrTask> {
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        let mut stmt = match conn.prepare("SELECT data_json FROM ocr_tasks ORDER BY created_at DESC") {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let rows = match stmt.query_map([], |row| {
            let json: String = row.get(0)?;
            serde_json::from_str::<OcrTask>(&json)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        }) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        rows.filter_map(|r| r.ok()).collect()
    }

    fn update_status(
        &self,
        id: &TaskId,
        status: &votex_domain::shared::value_object::TaskStatus,
    ) -> Result<(), DomainError> {
        let conn = self.conn.lock().map_err(|e| {
            DomainError::Config(crate::config_err(&format!("加锁失败: {}", e)))
        })?;

        let id_str = id.as_str();
        let now = format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );
        let status_str = match status {
            votex_domain::shared::value_object::TaskStatus::Queued => "Queued",
            votex_domain::shared::value_object::TaskStatus::Running => "Running",
            votex_domain::shared::value_object::TaskStatus::Completed => "Completed",
            votex_domain::shared::value_object::TaskStatus::Failed(_) => "Failed",
            votex_domain::shared::value_object::TaskStatus::Cancelled => "Cancelled",
            votex_domain::shared::value_object::TaskStatus::Paused => "Paused",
        };
        let status_json = serde_json::to_string(status).map_err(|e| {
            DomainError::Config(crate::config_err(&format!("状态序列化失败: {}", e)))
        })?;

        let affected = conn
            .execute(
                "UPDATE ocr_tasks
                 SET data_json = json_set(data_json, '$.status', json(?1), '$.updated_at', ?2),
                     status = ?3, updated_at = ?2
                 WHERE id = ?4",
                params![status_json, now, status_str, id_str],
            )
            .map_err(|e| DomainError::Config(crate::config_err(&format!("更新失败: {}", e))))?;

        if affected == 0 {
            return Err(DomainError::Config(crate::config_err(&format!(
                "任务不存在: {}",
                id_str
            ))));
        }

        Ok(())
    }

    fn update_page_result(
        &self,
        task_id: &TaskId,
        page_index: usize,
        blocks: &[OcrTextBlock],
        status: &PageStatus,
    ) -> Result<(), DomainError> {
        let conn = self.conn.lock().map_err(|e| {
            DomainError::Config(crate::config_err(&format!("加锁失败: {}", e)))
        })?;

        let id_str = task_id.as_str();

        // 读取完整任务
        let json: String = conn
            .query_row(
                "SELECT data_json FROM ocr_tasks WHERE id = ?1",
                params![id_str],
                |row| row.get(0),
            )
            .map_err(|_| DomainError::Config(crate::config_err(&format!("任务不存在: {}", id_str))))?;

        let mut task: OcrTask = serde_json::from_str(&json).map_err(|e| {
            DomainError::Config(crate::config_err(&format!("反序列化失败: {}", e)))
        })?;

        // 更新页面
        if let Some(page) = task.pages.get_mut(page_index) {
            page.status = status.clone();
            page.blocks = Some(blocks.to_vec());
            page.confidence = Some(if blocks.is_empty() {
                0.0
            } else {
                blocks.iter().map(|b| b.confidence).sum::<f32>() / blocks.len() as f32
            });
        }

        task.updated_at = format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );

        let new_json = serde_json::to_string(&task).map_err(|e| {
            DomainError::Config(crate::config_err(&format!("序列化失败: {}", e)))
        })?;

        // 更新主表
        conn.execute(
            "UPDATE ocr_tasks SET data_json = ?1, updated_at = ?2 WHERE id = ?3",
            params![new_json, task.updated_at, id_str],
        )
        .map_err(|e| DomainError::Config(crate::config_err(&format!("更新任务失败: {}", e))))?;

        // 更新页面表
        let status_str = match status {
            PageStatus::Pending => "Pending",
            PageStatus::Processing => "Processing",
            PageStatus::Completed => "Completed",
            PageStatus::Failed(_) => "Failed",
            PageStatus::Skipped => "Skipped",
        };
        let error_msg = match status {
            PageStatus::Failed(msg) => Some(msg.as_str()),
            _ => None,
        };
        let blocks_json = serde_json::to_string(blocks).ok();
        let confidence = if blocks.is_empty() {
            None
        } else {
            Some(blocks.iter().map(|b| b.confidence).sum::<f32>() / blocks.len() as f32)
        };

        conn.execute(
            "INSERT OR REPLACE INTO ocr_pages
                (task_id, page_index, status, blocks_json, confidence, retry_count, error_msg)
             VALUES (?1, ?2, ?3, ?4, ?5, (SELECT retry_count FROM ocr_pages WHERE task_id=?1 AND page_index=?2), ?6)",
            params![id_str, page_index as i64, status_str, blocks_json, confidence, error_msg],
        )
        .map_err(|e| DomainError::Config(crate::config_err(&format!("更新页面失败: {}", e))))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use votex_domain::ocr::value_object::{OcrParams, OcrTextBlock, TextBox};
    use std::path::PathBuf;

    fn create_test_task() -> OcrTask {
        OcrTask::new_single(PathBuf::from("test.png"), OcrParams::default())
    }

    #[test]
    fn sqlite_保存和读取() {
        let repo = SqliteOcrTaskRepository::new(crate::persistence::db::open_in_memory_database().unwrap());
        let task = create_test_task();
        let id = task.id.clone();

        repo.save(&task).unwrap();
        let loaded = repo.find_by_id(&id).unwrap();
        assert_eq!(loaded.id, task.id);
        assert_eq!(loaded.pages.len(), 1);
        assert_eq!(loaded.pages[0].image_path, PathBuf::from("test.png"));
    }

    #[test]
    fn sqlite_更新页面结果() {
        let repo = SqliteOcrTaskRepository::new(crate::persistence::db::open_in_memory_database().unwrap());
        let task = create_test_task();
        let id = task.id.clone();
        repo.save(&task).unwrap();

        let blocks = vec![OcrTextBlock {
            text: "识别结果".to_string(),
            confidence: 0.95,
            box_points: TextBox {
                x0: 0.0, y0: 0.0, x1: 100.0, y1: 0.0,
                x2: 100.0, y2: 30.0, x3: 0.0, y3: 30.0,
            },
        }];
        repo.update_page_result(&id, 0, &blocks, &PageStatus::Completed)
            .unwrap();

        let loaded = repo.find_by_id(&id).unwrap();
        assert_eq!(loaded.pages[0].status, PageStatus::Completed);
        assert!(loaded.pages[0].blocks.is_some());
    }

    #[test]
    fn sqlite_查找未完成任务() {
        let repo = SqliteOcrTaskRepository::new(crate::persistence::db::open_in_memory_database().unwrap());

        let mut task1 = create_test_task();
        task1.name = "未完成".to_string();
        repo.save(&task1).unwrap();

        let incomplete = repo.find_incomplete();
        assert!(!incomplete.is_empty());
    }

    #[test]
    fn sqlite_删除任务() {
        let repo = SqliteOcrTaskRepository::new(crate::persistence::db::open_in_memory_database().unwrap());
        let task = create_test_task();
        let id = task.id.clone();
        repo.save(&task).unwrap();
        repo.delete(&id).unwrap();
        assert!(repo.find_by_id(&id).is_none());
    }

    #[test]
    fn sqlite_列表顺序() {
        let repo = SqliteOcrTaskRepository::new(crate::persistence::db::open_in_memory_database().unwrap());

        let task1 = create_test_task();
        let task2 = create_test_task();
        repo.save(&task1).unwrap();
        repo.save(&task2).unwrap();

        let all = repo.list_all();
        assert_eq!(all.len(), 2);
    }
}
