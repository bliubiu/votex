use rusqlite::params;
use std::sync::{Arc, Mutex};
use votex_domain::asr::entity::AsrTask;
use votex_domain::error::DomainError;
use votex_domain::repository::AsrTaskRepository;
use votex_domain::shared::value_object::TaskId;

/// SQLite ASR 任务仓储
///
/// 将 AsrTask 序列化为 JSON 存入 SQLite，支持跨会话持久化。
/// 表结构：
///   asr_tasks(id TEXT PK, data_json TEXT, status TEXT NOT NULL,
///             created_at TEXT NOT NULL, updated_at TEXT NOT NULL)
pub struct SqliteAsrTaskRepository {
    conn: Arc<Mutex<rusqlite::Connection>>,
}

impl SqliteAsrTaskRepository {
    /// 创建共享连接的 ASR 任务仓储
    pub fn new(conn: Arc<Mutex<rusqlite::Connection>>) -> Self {
        Self { conn }
    }

    fn status_to_str(status: &votex_domain::shared::value_object::TaskStatus) -> &'static str {
        match status {
            votex_domain::shared::value_object::TaskStatus::Queued => "Queued",
            votex_domain::shared::value_object::TaskStatus::Running => "Running",
            votex_domain::shared::value_object::TaskStatus::Completed => "Completed",
            votex_domain::shared::value_object::TaskStatus::Failed(_) => "Failed",
            votex_domain::shared::value_object::TaskStatus::Cancelled => "Cancelled",
            votex_domain::shared::value_object::TaskStatus::Paused => "Paused",
        }
    }

    /// 列出所有任务（按创建时间降序）
    pub fn list_all(&self) -> Vec<AsrTask> {
        let conn = crate::persistence::guard::lock_conn(&self.conn);

        let mut stmt = match conn.prepare("SELECT data_json FROM asr_tasks ORDER BY created_at DESC",
        ) {
            Ok(s) => s,
            Err(e) => {
                crate::persistence::guard::log_err("asr_task_repo", &e);
                return Vec::new();
            }
        };

        let rows = match stmt.query_map([], |row| {
            let json: String = row.get(0)?;
            serde_json::from_str::<AsrTask>(&json)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        }) {
            Ok(r) => r,
            Err(e) => {
                crate::persistence::guard::log_err("asr_task_repo", &e);
                return Vec::new();
            }
        };

        rows.filter_map(|r| match r {
            Ok(v) => Some(v),
            Err(e) => {
                crate::persistence::guard::log_err("asr_task_repo 行解析", &e);
                None
            }
        }).collect()
    }

    /// 更新任务状态
    ///
    /// 使用 SQLite 内置 json_set() 原地更新状态字段，避免 JSON 全量读改写。
    pub fn update_status(
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
        let status_str = Self::status_to_str(status);
        let status_json = serde_json::to_string(status).map_err(|e| {
            DomainError::Config(crate::config_err(&format!("状态序列化失败: {}", e)))
        })?;

        let affected = conn
            .execute(
                "UPDATE asr_tasks
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
}

impl AsrTaskRepository for SqliteAsrTaskRepository {
    fn find_by_id(&self, id: &TaskId) -> Option<AsrTask> {
        let conn = crate::persistence::guard::lock_conn(&self.conn);
        let id_str = id.as_str();
        match conn.query_row(
            "SELECT data_json FROM asr_tasks WHERE id = ?1",
            params![id_str],
            |row| {
                let json: String = row.get(0)?;
                serde_json::from_str::<AsrTask>(&json)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
            },
        ) {
            // 区分「查不到行」与「查询出错」——后者必须留痕，
            // 否则调用方会误判为任务不存在而重复创建
            Ok(v) => Some(v),
            Err(rusqlite::Error::QueryReturnedNoRows) => None,
            Err(e) => {
                crate::persistence::guard::log_err("asr_tasks.find_by_id", &e);
                None
            }
        }
    }

    fn save(&self, task: &AsrTask) -> Result<(), DomainError> {
        let conn = self.conn.lock().map_err(|e| {
            DomainError::Config(crate::config_err(&format!("加锁失败: {}", e)))
        })?;

        let id = task.id.as_str();
        let json = serde_json::to_string(task).map_err(|e| {
            DomainError::Config(crate::config_err(&format!("序列化失败: {}", e)))
        })?;
        let created = &task.created_at;
        let updated = &task.updated_at;
        let status_str = Self::status_to_str(&task.status);

        conn.execute(
            "INSERT OR REPLACE INTO asr_tasks (id, data_json, status, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, json, status_str, created, updated],
        )
        .map_err(|e| DomainError::Config(crate::config_err(&format!("保存任务失败: {}", e))))?;

        tracing::debug!("ASR 任务已保存: {}", id);
        Ok(())
    }

    fn delete(&self, id: &TaskId) -> Result<(), DomainError> {
        let conn = self.conn.lock().map_err(|e| {
            DomainError::Config(crate::config_err(&format!("加锁失败: {}", e)))
        })?;

        let id_str = id.as_str();
        conn.execute("DELETE FROM asr_tasks WHERE id = ?1", params![id_str])
            .map_err(|e| DomainError::Config(crate::config_err(&format!("删除任务失败: {}", e))))?;

        tracing::debug!("ASR 任务已删除: {}", id);
        Ok(())
    }

    fn find_pending(&self) -> Vec<AsrTask> {
        let conn = crate::persistence::guard::lock_conn(&self.conn);

        let mut stmt = match conn.prepare("SELECT data_json FROM asr_tasks WHERE status IN ('Queued', 'Running', 'Paused')",
        ) {
            Ok(s) => s,
            Err(e) => {
                crate::persistence::guard::log_err("asr_task_repo", &e);
                return Vec::new();
            }
        };

        let rows = match stmt.query_map([], |row| {
            let json: String = row.get(0)?;
            serde_json::from_str::<AsrTask>(&json)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        }) {
            Ok(r) => r,
            Err(e) => {
                crate::persistence::guard::log_err("asr_task_repo", &e);
                return Vec::new();
            }
        };

        rows.filter_map(|r| match r {
            Ok(v) => Some(v),
            Err(e) => {
                crate::persistence::guard::log_err("asr_task_repo 行解析", &e);
                None
            }
        }).collect()
    }

    fn find_all(&self) -> Vec<AsrTask> {
        let conn = crate::persistence::guard::lock_conn(&self.conn);

        let mut stmt = match conn.prepare(
            "SELECT data_json FROM asr_tasks ORDER BY created_at DESC",
        ) {
            Ok(s) => s,
            Err(e) => {
                crate::persistence::guard::log_err("asr_task_repo.find_all", &e);
                return Vec::new();
            }
        };

        let rows = match stmt.query_map([], |row| {
            let json: String = row.get(0)?;
            serde_json::from_str::<AsrTask>(&json)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        }) {
            Ok(r) => r,
            Err(e) => {
                crate::persistence::guard::log_err("asr_task_repo.find_all", &e);
                return Vec::new();
            }
        };

        rows.filter_map(|r| match r {
            Ok(v) => Some(v),
            Err(e) => {
                crate::persistence::guard::log_err("asr_task_repo.find_all 行解析", &e);
                None
            }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use votex_domain::asr::value_object::{
        AsrInput, AsrParams, AsrPhase, AsrProgress, AsrResult, AudioFormatKind, DenoiseLevel,
        Language, MediaKind, SliceLength, SubtitleFormat,
    };
    use votex_domain::model::value_object::ModelId;
    use votex_domain::shared::value_object::TaskStatus;

    fn create_test_task() -> AsrTask {
        let now = format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );
        AsrTask {
            id: TaskId::new(),
            input: AsrInput {
                file_path: PathBuf::from("test.mp3"),
                file_kind: MediaKind::Audio(AudioFormatKind::Mp3),
            },
            params: AsrParams {
                model: ModelId::new("whisper-base"),
                language: Language::Zh,
                auto_punctuation: true,
                auto_slice: true,
                slice_length: SliceLength::S30,
                denoise: false,
                denoise_level: DenoiseLevel::Low,
                output_format: SubtitleFormat::Srt,
            },
            slices: Vec::new(),
            status: TaskStatus::Queued,
            result: Some(AsrResult {
                text: "测试语音内容".to_string(),
                subtitles: Vec::new(),
                output_path: PathBuf::from("output.srt"),
            }),
            progress: AsrProgress {
                current_slice: 0,
                total_slices: 0,
                phase: AsrPhase::Idle,
            },
            output_path: Some(PathBuf::from("output.srt")),
            created_at: now.clone(),
            updated_at: now,
        }
    }

    #[test]
    fn sqlite_保存和读取() {
        let repo = SqliteAsrTaskRepository::new(crate::persistence::db::open_in_memory_database().unwrap());
        let task = create_test_task();
        let id = task.id.clone();

        repo.save(&task).unwrap();
        let loaded = repo.find_by_id(&id).unwrap();
        assert_eq!(loaded.id, task.id);
        assert_eq!(loaded.input.file_path, PathBuf::from("test.mp3"));
    }

    #[test]
    fn sqlite_更新状态() {
        let repo = SqliteAsrTaskRepository::new(crate::persistence::db::open_in_memory_database().unwrap());
        let task = create_test_task();
        let id = task.id.clone();
        repo.save(&task).unwrap();

        repo.update_status(&id, &TaskStatus::Running).unwrap();

        let loaded = repo.find_by_id(&id).unwrap();
        assert_eq!(loaded.status, TaskStatus::Running);
    }

    #[test]
    fn sqlite_查找未完成任务() {
        let repo = SqliteAsrTaskRepository::new(crate::persistence::db::open_in_memory_database().unwrap());
        let mut task = create_test_task();
        task.status = TaskStatus::Queued;
        repo.save(&task).unwrap();

        let pending = repo.find_pending();
        assert!(!pending.is_empty());
    }

    #[test]
    fn sqlite_删除任务() {
        let repo = SqliteAsrTaskRepository::new(crate::persistence::db::open_in_memory_database().unwrap());
        let task = create_test_task();
        let id = task.id.clone();
        repo.save(&task).unwrap();
        repo.delete(&id).unwrap();
        assert!(repo.find_by_id(&id).is_none());
    }

    #[test]
    fn sqlite_列表顺序() {
        let repo = SqliteAsrTaskRepository::new(crate::persistence::db::open_in_memory_database().unwrap());
        let task1 = create_test_task();
        let task2 = create_test_task();
        repo.save(&task1).unwrap();
        repo.save(&task2).unwrap();

        let all = repo.list_all();
        assert_eq!(all.len(), 2);
    }
}
