use crate::error::ModelError;
use crate::model::entity::Model;
use crate::model::value_object::ModelId;

/// 模型管理领域服务
pub struct ModelService;

impl ModelService {
    pub fn new() -> Self {
        Self
    }

    /// 下载模型
    pub fn download(&self, _model_id: &ModelId) -> Result<(), ModelError> {
        // 实际实现在应用层编排，领域层仅定义业务规则
        Ok(())
    }

    /// 从本地路径导入模型
    pub fn import(&self, _model_id: &ModelId, _path: &std::path::Path) -> Result<(), ModelError> {
        Ok(())
    }

    /// 校验模型文件完整性
    pub fn verify(&self, model: &Model) -> Result<(), ModelError> {
        if !model.is_ready() && !matches!(model.status, crate::model::value_object::ModelStatus::VerifyFailed) {
            return Err(ModelError::InvalidStatus {
                current: format!("{:?}", model.status),
                expected: "Ready 或 VerifyFailed".to_string(),
            });
        }
        Ok(())
    }

    /// 删除模型
    pub fn remove(&self, model: &Model) -> Result<(), ModelError> {
        if matches!(model.status, crate::model::value_object::ModelStatus::Loaded) {
            return Err(ModelError::InvalidStatus {
                current: "Loaded".to_string(),
                expected: "非 Loaded 状态".to_string(),
            });
        }
        Ok(())
    }
}

/// 模型加载领域服务
pub struct ModelLoader;

impl ModelLoader {
    pub fn new() -> Self {
        Self
    }

    /// 加载模型到内存
    pub fn load(&self, model: &Model) -> Result<(), ModelError> {
        if !model.is_ready() {
            return Err(ModelError::InvalidStatus {
                current: format!("{:?}", model.status),
                expected: "Ready".to_string(),
            });
        }
        Ok(())
    }

    /// 释放模型资源
    pub fn unload(&self, model: &Model) -> Result<(), ModelError> {
        if !model.is_loaded() {
            return Err(ModelError::InvalidStatus {
                current: format!("{:?}", model.status),
                expected: "Loaded".to_string(),
            });
        }
        Ok(())
    }

    /// 检查模型是否已加载
    pub fn is_loaded(&self, model: &Model) -> bool {
        model.is_loaded()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::value_object::ModelKind;

    #[test]
    fn model_service_校验状态检查() {
        let model = Model::new(
            ModelId::new("kokoro-82m"),
            "Kokoro-82M",
            ModelKind::Tts,
            crate::model::value_object::EngineKind::Kokoro,
        );
        let service = ModelService::new();
        // NotDownloaded 状态不允许校验
        assert!(service.verify(&model).is_err());
    }

    #[test]
    fn model_service_删除已加载模型应失败() {
        let mut model = Model::new(
            ModelId::new("kokoro-82m"),
            "Kokoro-82M",
            ModelKind::Tts,
            crate::model::value_object::EngineKind::Kokoro,
        );
        model.status = crate::model::value_object::ModelStatus::Loaded;
        let service = ModelService::new();
        assert!(service.remove(&model).is_err());
    }

    #[test]
    fn model_loader_加载未就绪模型应失败() {
        let model = Model::new(
            ModelId::new("kokoro-82m"),
            "Kokoro-82M",
            ModelKind::Tts,
            crate::model::value_object::EngineKind::Kokoro,
        );
        let loader = ModelLoader::new();
        assert!(loader.load(&model).is_err());
    }
}
