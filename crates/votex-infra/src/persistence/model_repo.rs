use votex_domain::error::DomainError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{ModelId, ModelKind};
use votex_domain::repository::ModelRepository;
use std::collections::HashMap;
use std::sync::Mutex;

/// 模型仓储实现（文件系统）
pub struct FileModelRepo {
    models: Mutex<HashMap<ModelId, Model>>,
}

impl FileModelRepo {
    pub fn new() -> Self {
        Self {
            models: Mutex::new(HashMap::new()),
        }
    }
}

impl ModelRepository for FileModelRepo {
    fn find_by_id(&self, id: &ModelId) -> Option<Model> {
        self.models.lock().ok()?.get(id).cloned()
    }

    fn find_by_kind(&self, kind: ModelKind) -> Vec<Model> {
        self.models
            .lock()
            .map(|models| {
                models
                    .values()
                    .filter(|m| m.kind == kind)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    fn save(&self, model: &Model) -> Result<(), DomainError> {
        let mut models = self.models.lock().map_err(|e| {
            DomainError::Config(votex_domain::error::ConfigError::ReadFailed(e.to_string()))
        })?;
        models.insert(model.id.clone(), model.clone());
        Ok(())
    }

    fn delete(&self, id: &ModelId) -> Result<(), DomainError> {
        let mut models = self.models.lock().map_err(|e| {
            DomainError::Config(votex_domain::error::ConfigError::ReadFailed(e.to_string()))
        })?;
        models.remove(id);
        Ok(())
    }

    fn exists(&self, id: &ModelId) -> bool {
        self.models
            .lock()
            .map(|models| models.contains_key(id))
            .unwrap_or(false)
    }
}

/// SQLite 模型仓储
///
/// 将 Model 序列化为 JSON 存入 SQLite，支持跨会话持久化。
/// 表结构：
///   models(id TEXT PK, data_json TEXT NOT NULL, kind TEXT NOT NULL, status TEXT NOT NULL)
pub struct SqliteModelRepository {
    conn: std::sync::Arc<std::sync::Mutex<rusqlite::Connection>>,
}

impl SqliteModelRepository {
    /// 创建共享连接的模型仓储
    pub fn new(conn: std::sync::Arc<std::sync::Mutex<rusqlite::Connection>>) -> Self {
        Self { conn }
    }

    fn kind_to_str(kind: &ModelKind) -> &'static str {
        match kind {
            ModelKind::Tts => "Tts",
            ModelKind::Asr => "Asr",
            ModelKind::Ocr => "Ocr",
            ModelKind::Translation => "Translation",
        }
    }

    fn status_to_str(status: &votex_domain::model::value_object::ModelStatus) -> String {
        // Use debug/display representation for the enum variants with data
        match status {
            votex_domain::model::value_object::ModelStatus::NotDownloaded => "NotDownloaded".to_string(),
            votex_domain::model::value_object::ModelStatus::Downloading(_) => "Downloading".to_string(),
            votex_domain::model::value_object::ModelStatus::DownloadPaused => "DownloadPaused".to_string(),
            votex_domain::model::value_object::ModelStatus::Verifying => "Verifying".to_string(),
            votex_domain::model::value_object::ModelStatus::VerifyFailed => "VerifyFailed".to_string(),
            votex_domain::model::value_object::ModelStatus::Ready => "Ready".to_string(),
            votex_domain::model::value_object::ModelStatus::Loading => "Loading".to_string(),
            votex_domain::model::value_object::ModelStatus::Loaded => "Loaded".to_string(),
            votex_domain::model::value_object::ModelStatus::LoadFailed(_) => "LoadFailed".to_string(),
        }
    }

    /// 列出所有模型
    pub fn list_all(&self) -> Vec<Model> {
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        let mut stmt = match conn.prepare("SELECT data_json FROM models") {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let rows = match stmt.query_map([], |row| {
            let json: String = row.get(0)?;
            serde_json::from_str::<Model>(&json)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        }) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        rows.filter_map(|r| r.ok()).collect()
    }
}

impl ModelRepository for SqliteModelRepository {
    fn find_by_id(&self, id: &ModelId) -> Option<Model> {
        let conn = self.conn.lock().ok()?;
        let id_str = id.as_str();
        conn.query_row(
            "SELECT data_json FROM models WHERE id = ?1",
            rusqlite::params![id_str],
            |row| {
                let json: String = row.get(0)?;
                serde_json::from_str::<Model>(&json)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
            },
        )
        .ok()
    }

    fn find_by_kind(&self, kind: ModelKind) -> Vec<Model> {
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        let kind_str = Self::kind_to_str(&kind);
        let mut stmt = match conn.prepare("SELECT data_json FROM models WHERE kind = ?1") {
            Ok(s) => s,
            Err(_) => return Vec::new(),
        };

        let rows = match stmt.query_map(rusqlite::params![kind_str], |row| {
            let json: String = row.get(0)?;
            serde_json::from_str::<Model>(&json)
                .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))
        }) {
            Ok(r) => r,
            Err(_) => return Vec::new(),
        };

        rows.filter_map(|r| r.ok()).collect()
    }

    fn save(&self, model: &Model) -> Result<(), DomainError> {
        let conn = self.conn.lock().map_err(|e| {
            DomainError::Config(crate::config_err(&format!("加锁失败: {}", e)))
        })?;

        let id = model.id.as_str();
        let json = serde_json::to_string(model).map_err(|e| {
            DomainError::Config(crate::config_err(&format!("序列化失败: {}", e)))
        })?;
        let kind_str = Self::kind_to_str(&model.kind);
        let status_str = Self::status_to_str(&model.status);

        conn.execute(
            "INSERT OR REPLACE INTO models (id, data_json, kind, status)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![id, json, kind_str, status_str],
        )
        .map_err(|e| DomainError::Config(crate::config_err(&format!("保存模型失败: {}", e))))?;

        tracing::debug!("模型已保存: {}", id);
        Ok(())
    }

    fn delete(&self, id: &ModelId) -> Result<(), DomainError> {
        let conn = self.conn.lock().map_err(|e| {
            DomainError::Config(crate::config_err(&format!("加锁失败: {}", e)))
        })?;

        let id_str = id.as_str();
        conn.execute("DELETE FROM models WHERE id = ?1", rusqlite::params![id_str])
            .map_err(|e| DomainError::Config(crate::config_err(&format!("删除模型失败: {}", e))))?;

        tracing::debug!("模型已删除: {}", id);
        Ok(())
    }

    fn exists(&self, id: &ModelId) -> bool {
        let conn = match self.conn.lock() {
            Ok(c) => c,
            Err(_) => return false,
        };

        let id_str = id.as_str();
        conn.query_row(
            "SELECT 1 FROM models WHERE id = ?1",
            rusqlite::params![id_str],
            |_| Ok(()),
        )
        .is_ok()
    }
}

#[cfg(test)]
mod model_tests {
    use super::*;
    use votex_domain::model::value_object::EngineKind;

    fn create_test_model() -> Model {
        Model::new(
            ModelId::new("test-model"),
            "测试模型",
            ModelKind::Tts,
            EngineKind::Kokoro,
        )
    }

    #[test]
    fn sqlite_保存和读取() {
        let repo = SqliteModelRepository::new(crate::persistence::db::open_in_memory_database().unwrap());
        let model = create_test_model();
        let id = model.id.clone();

        repo.save(&model).unwrap();
        let loaded = repo.find_by_id(&id).unwrap();
        assert_eq!(loaded.id, model.id);
        assert_eq!(loaded.name, "测试模型");
    }

    #[test]
    fn sqlite_按类型查找() {
        let repo = SqliteModelRepository::new(crate::persistence::db::open_in_memory_database().unwrap());
        let model = create_test_model();
        repo.save(&model).unwrap();

        let tts_models = repo.find_by_kind(ModelKind::Tts);
        assert!(!tts_models.is_empty());

        let asr_models = repo.find_by_kind(ModelKind::Asr);
        assert!(asr_models.is_empty());
    }

    #[test]
    fn sqlite_删除模型() {
        let repo = SqliteModelRepository::new(crate::persistence::db::open_in_memory_database().unwrap());
        let model = create_test_model();
        let id = model.id.clone();
        repo.save(&model).unwrap();
        repo.delete(&id).unwrap();
        assert!(!repo.exists(&id));
    }

    #[test]
    fn sqlite_存在性检查() {
        let repo = SqliteModelRepository::new(crate::persistence::db::open_in_memory_database().unwrap());
        let model = create_test_model();
        let id = model.id.clone();

        assert!(!repo.exists(&id));
        repo.save(&model).unwrap();
        assert!(repo.exists(&id));
    }
}
