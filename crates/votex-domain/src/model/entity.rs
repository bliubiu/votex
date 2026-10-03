use crate::model::value_object::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 模型实体（聚合根）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: ModelId,
    pub name: String,
    pub kind: ModelKind,
    pub engine: EngineKind,
    pub status: ModelStatus,
    pub file_paths: Vec<FilePath>,
    pub checksums: HashMap<String, Hash>,
    pub size_bytes: u64,
    pub download_source: Option<DownloadSource>,
}

use crate::shared::value_object::{FilePath, Hash};

impl Model {
    pub fn new(id: ModelId, name: &str, kind: ModelKind, engine: EngineKind) -> Self {
        Self {
            id,
            name: name.to_string(),
            kind,
            engine,
            status: ModelStatus::NotDownloaded,
            file_paths: Vec::new(),
            checksums: HashMap::new(),
            size_bytes: 0,
            download_source: None,
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self.status, ModelStatus::Ready | ModelStatus::Loaded)
    }

    pub fn is_loaded(&self) -> bool {
        matches!(self.status, ModelStatus::Loaded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_创建与状态检查() {
        let model = Model::new(
            ModelId::new("kokoro-82m"),
            "Kokoro-82M",
            ModelKind::Tts,
            EngineKind::Kokoro,
        );
        assert_eq!(model.name, "Kokoro-82M");
        assert!(!model.is_ready());
        assert!(!model.is_loaded());
    }

    #[test]
    fn model_就绪状态() {
        let mut model = Model::new(
            ModelId::new("kokoro-82m"),
            "Kokoro-82M",
            ModelKind::Tts,
            EngineKind::Kokoro,
        );
        model.status = ModelStatus::Ready;
        assert!(model.is_ready());
        assert!(!model.is_loaded());
    }
}
