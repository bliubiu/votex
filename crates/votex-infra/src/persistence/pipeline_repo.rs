use rusqlite::params;
use std::sync::{Arc, Mutex};
use votex_domain::error::DomainError;
use votex_domain::pipeline::entity::Pipeline;
use votex_domain::pipeline::value_object::PipelineStatus;
use votex_domain::repository::PipelineRepository;
use votex_domain::shared::value_object::PipelineId;

/// SQLite 流水线仓储
///
/// 将 Pipeline 序列化为 JSON 存入 SQLite，支持跨会话持久化。
/// 表结构：
///   pipelines(id TEXT PK, data_json TEXT, status TEXT NOT NULL,
///             created_at TEXT NOT NULL, updated_at TEXT NOT NULL)
pub struct SqlitePipelineRepository {
    conn: Arc<Mutex<rusqlite::Connection>>,
}

impl SqlitePipelineRepository {
    /// 创建共享连接的流水线仓储
    pub fn new(conn: Arc<Mutex<rusqlite::Connection>>) -> Self {
        Self { conn }
    }

    fn status_to_str(status: &PipelineStatus) -> &'static str {
        match status {
            PipelineStatus::Idle => "Idle",
            PipelineStatus::Running(_) => "Running",
            PipelineStatus::Paused(_) => "Paused",
            PipelineStatus::Completed => "Completed",
            PipelineStatus::Failed(_, _) => "Failed",
            PipelineStatus::Cancelled => "Cancelled",
        }
    }

    /// 列出所有流水线（按创建时间降序）
    pub fn list_all(&self) -> Vec<Pipeline> {
        let conn = crate::persistence::guard::lock_conn(&self.conn);

        let mut stmt = match conn.prepare("SELECT data_json FROM pipelines ORDER BY created_at DESC",
        ) {
            Ok(s) => s,
            Err(e) => {
                crate::persistence::guard::log_err("pipeline_repo", &e);
                return Vec::new();
            }
        };

        let rows = match stmt.query_map([], |row| {
            let json: String = row.get(0)?;
            serde_json::from_str::<Pipeline>(&json)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        }) {
            Ok(r) => r,
            Err(e) => {
                crate::persistence::guard::log_err("pipeline_repo", &e);
                return Vec::new();
            }
        };

        rows.filter_map(|r| match r {
            Ok(v) => Some(v),
            Err(e) => {
                crate::persistence::guard::log_err("pipeline_repo 行解析", &e);
                None
            }
        }).collect()
    }

    /// 更新流水线状态
    ///
    /// 使用 SQLite 内置 json_set() 原地更新状态字段，避免 JSON 全量读改写。
    pub fn update_status(
        &self,
        id: &PipelineId,
        status: &PipelineStatus,
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
                "UPDATE pipelines
                 SET data_json = json_set(data_json, '$.status', json(?1), '$.updated_at', ?2),
                     status = ?3, updated_at = ?2
                 WHERE id = ?4",
                params![status_json, now, status_str, id_str],
            )
            .map_err(|e| DomainError::Config(crate::config_err(&format!("更新失败: {}", e))))?;

        if affected == 0 {
            return Err(DomainError::Config(crate::config_err(&format!(
                "流水线不存在: {}",
                id_str
            ))));
        }

        Ok(())
    }
}

impl PipelineRepository for SqlitePipelineRepository {
    fn find_by_id(&self, id: &PipelineId) -> Option<Pipeline> {
        let conn = crate::persistence::guard::lock_conn(&self.conn);
        let id_str = id.as_str();
        match conn.query_row(
            "SELECT data_json FROM pipelines WHERE id = ?1",
            params![id_str],
            |row| {
                let json: String = row.get(0)?;
                serde_json::from_str::<Pipeline>(&json)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
            },
        ) {
            // 区分「查不到行」与「查询出错」——后者必须留痕，
            // 否则调用方会误判为任务不存在而重复创建
            Ok(v) => Some(v),
            Err(rusqlite::Error::QueryReturnedNoRows) => None,
            Err(e) => {
                crate::persistence::guard::log_err("pipelines.find_by_id", &e);
                None
            }
        }
    }

    fn save(&self, pipeline: &Pipeline) -> Result<(), DomainError> {
        let conn = self.conn.lock().map_err(|e| {
            DomainError::Config(crate::config_err(&format!("加锁失败: {}", e)))
        })?;

        let id = pipeline.id.as_str();
        let json = serde_json::to_string(pipeline).map_err(|e| {
            DomainError::Config(crate::config_err(&format!("序列化失败: {}", e)))
        })?;
        let created = &pipeline.created_at;
        let status_str = Self::status_to_str(&pipeline.status);
        let updated_at = format!(
            "{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        );

        conn.execute(
            "INSERT OR REPLACE INTO pipelines (id, data_json, status, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, json, status_str, created, updated_at],
        )
        .map_err(|e| DomainError::Config(crate::config_err(&format!("保存流水线失败: {}", e))))?;

        tracing::debug!("流水线已保存: {}", id);
        Ok(())
    }

    fn delete(&self, id: &PipelineId) -> Result<(), DomainError> {
        let conn = self.conn.lock().map_err(|e| {
            DomainError::Config(crate::config_err(&format!("加锁失败: {}", e)))
        })?;

        let id_str = id.as_str();
        conn.execute("DELETE FROM pipelines WHERE id = ?1", params![id_str])
            .map_err(|e| DomainError::Config(crate::config_err(&format!("删除流水线失败: {}", e))))?;

        tracing::debug!("流水线已删除: {}", id);
        Ok(())
    }

    fn find_incomplete(&self) -> Vec<Pipeline> {
        let conn = crate::persistence::guard::lock_conn(&self.conn);

        let mut stmt = match conn.prepare("SELECT data_json FROM pipelines WHERE status IN ('Idle', 'Running', 'Paused')",
        ) {
            Ok(s) => s,
            Err(e) => {
                crate::persistence::guard::log_err("pipeline_repo", &e);
                return Vec::new();
            }
        };

        let rows = match stmt.query_map([], |row| {
            let json: String = row.get(0)?;
            serde_json::from_str::<Pipeline>(&json)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        }) {
            Ok(r) => r,
            Err(e) => {
                crate::persistence::guard::log_err("pipeline_repo", &e);
                return Vec::new();
            }
        };

        rows.filter_map(|r| match r {
            Ok(v) => Some(v),
            Err(e) => {
                crate::persistence::guard::log_err("pipeline_repo 行解析", &e);
                None
            }
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use votex_domain::pipeline::entity::PipelineInput;
    use votex_domain::pipeline::value_object::PipelineKind;

    fn create_test_pipeline() -> Pipeline {
        Pipeline::new(
            PipelineKind::Audiobook,
            PipelineInput {
                text_file: None,
                media_file: None,
                tts_params: None,
                asr_params: None,
            },
        )
    }

    #[test]
    fn sqlite_保存和读取() {
        let repo = SqlitePipelineRepository::new(crate::persistence::db::open_in_memory_database().unwrap());
        let pipeline = create_test_pipeline();
        let id = pipeline.id.clone();

        repo.save(&pipeline).unwrap();
        let loaded = repo.find_by_id(&id).unwrap();
        assert_eq!(loaded.id, pipeline.id);
        assert_eq!(loaded.kind, PipelineKind::Audiobook);
        assert_eq!(loaded.stages.len(), 5);
    }

    #[test]
    fn sqlite_更新状态() {
        let repo = SqlitePipelineRepository::new(crate::persistence::db::open_in_memory_database().unwrap());
        let pipeline = create_test_pipeline();
        let id = pipeline.id.clone();
        repo.save(&pipeline).unwrap();

        repo.update_status(&id, &PipelineStatus::Running(0))
            .unwrap();

        let loaded = repo.find_by_id(&id).unwrap();
        assert_eq!(loaded.status, PipelineStatus::Running(0));
    }

    #[test]
    fn sqlite_查找未完成流水线() {
        let repo = SqlitePipelineRepository::new(crate::persistence::db::open_in_memory_database().unwrap());

        let pipeline = create_test_pipeline();
        repo.save(&pipeline).unwrap();

        let incomplete = repo.find_incomplete();
        assert!(!incomplete.is_empty());
    }

    #[test]
    fn sqlite_删除流水线() {
        let repo = SqlitePipelineRepository::new(crate::persistence::db::open_in_memory_database().unwrap());
        let pipeline = create_test_pipeline();
        let id = pipeline.id.clone();
        repo.save(&pipeline).unwrap();
        repo.delete(&id).unwrap();
        assert!(repo.find_by_id(&id).is_none());
    }

    #[test]
    fn sqlite_列表顺序() {
        let repo = SqlitePipelineRepository::new(crate::persistence::db::open_in_memory_database().unwrap());
        let p1 = create_test_pipeline();
        let p2 = create_test_pipeline();
        repo.save(&p1).unwrap();
        repo.save(&p2).unwrap();

        let all = repo.list_all();
        assert_eq!(all.len(), 2);
    }
}
