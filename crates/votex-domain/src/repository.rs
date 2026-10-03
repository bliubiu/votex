use crate::error::DomainError;
use crate::model::entity::Model;
use crate::model::value_object::ModelId;
use crate::model::value_object::ModelKind;
use crate::shared::value_object::TaskId;
use crate::tts::entity::TtsTask;
use crate::asr::entity::AsrTask;
use crate::ocr::entity::OcrTask;
use crate::pipeline::entity::Pipeline;
use crate::shared::value_object::PipelineId;

/// 模型仓储接口
pub trait ModelRepository: Send + Sync {
    fn find_by_id(&self, id: &ModelId) -> Option<Model>;
    fn find_by_kind(&self, kind: ModelKind) -> Vec<Model>;
    fn save(&self, model: &Model) -> Result<(), DomainError>;
    fn delete(&self, id: &ModelId) -> Result<(), DomainError>;
    fn exists(&self, id: &ModelId) -> bool;
}

/// TTS 任务仓储接口
pub trait TtsTaskRepository: Send + Sync {
    fn find_by_id(&self, id: &TaskId) -> Option<TtsTask>;
    fn save(&self, task: &TtsTask) -> Result<(), DomainError>;
    fn delete(&self, id: &TaskId) -> Result<(), DomainError>;
    fn find_pending(&self) -> Vec<TtsTask>;
}

/// ASR 任务仓储接口
pub trait AsrTaskRepository: Send + Sync {
    fn find_by_id(&self, id: &TaskId) -> Option<AsrTask>;
    fn save(&self, task: &AsrTask) -> Result<(), DomainError>;
    fn delete(&self, id: &TaskId) -> Result<(), DomainError>;
    fn find_pending(&self) -> Vec<AsrTask>;
}

/// OCR 任务仓储接口
pub trait OcrTaskRepository: Send + Sync {
    /// 按 ID 查找
    fn find_by_id(&self, id: &TaskId) -> Option<OcrTask>;

    /// 保存任务
    fn save(&self, task: &OcrTask) -> Result<(), DomainError>;

    /// 删除任务
    fn delete(&self, id: &TaskId) -> Result<(), DomainError>;

    /// 查找所有未完成的任务（用于恢复）
    fn find_incomplete(&self) -> Vec<OcrTask>;

    /// 列出所有任务
    fn list_all(&self) -> Vec<OcrTask>;

    /// 更新任务状态
    fn update_status(&self, id: &TaskId, status: &crate::shared::value_object::TaskStatus) -> Result<(), DomainError>;

    /// 更新页面结果
    fn update_page_result(
        &self,
        task_id: &TaskId,
        page_index: usize,
        blocks: &[crate::ocr::value_object::OcrTextBlock],
        status: &crate::ocr::value_object::PageStatus,
    ) -> Result<(), DomainError>;
}

/// 流水线仓储接口
pub trait PipelineRepository: Send + Sync {
    fn find_by_id(&self, id: &PipelineId) -> Option<Pipeline>;
    fn save(&self, pipeline: &Pipeline) -> Result<(), DomainError>;
    fn delete(&self, id: &PipelineId) -> Result<(), DomainError>;
    fn find_incomplete(&self) -> Vec<Pipeline>;
}
