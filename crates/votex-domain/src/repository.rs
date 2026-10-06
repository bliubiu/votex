use crate::error::DomainError;
use crate::model::entity::{DownloadRecord, Model, PlaybackProgress};
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

/// 下载进度仓储接口
///
/// 记录模型文件的下载进度，是断点续传的**进度可观测**手段
/// （真正的字节级续传由 infra 的下载器负责，这里只管状态）。
///
/// 表现层（CLI / GUI）通过本 trait 上报进度，
/// 无需知道底层是 SQLite 还是内存实现。
pub trait DownloadRepository: Send + Sync {
    /// 插入或更新一条下载记录
    fn upsert(&self, record: &DownloadRecord) -> Result<(), DomainError>;

    /// 仅更新进度字节数（高频调用，避免整行重写）
    fn update_progress(
        &self,
        model_id: &str,
        file_name: &str,
        bytes_downloaded: u64,
        total_bytes: u64,
    ) -> Result<(), DomainError>;

    /// 按模型 ID 查找全部文件记录
    fn find_by_model(&self, model_id: &str) -> Vec<DownloadRecord>;

    /// 查找所有未完成记录（启动时恢复下载）
    fn find_incomplete(&self) -> Vec<DownloadRecord>;

    /// 列出全部记录
    fn list_all(&self) -> Vec<DownloadRecord>;

    /// 删除单条记录
    fn delete(&self, model_id: &str, file_name: &str) -> Result<(), DomainError>;

    /// 删除某个模型的所有记录
    fn delete_by_model(&self, model_id: &str) -> Result<(), DomainError>;
}

/// 播放进度仓储接口（断点续听）
pub trait PlaybackRepository: Send + Sync {
    /// 读取某文件的播放进度（无记录返回 None）
    fn get(&self, file_path: &str) -> Result<Option<PlaybackProgress>, String>;

    /// 保存（upsert）播放进度
    fn save(&self, file_path: &str, position_sec: f32, duration_sec: f32) -> Result<(), String>;

    /// 清除播放进度
    fn clear(&self, file_path: &str) -> Result<(), String>;
}

/// TTS 任务仓储接口
pub trait TtsTaskRepository: Send + Sync {
    fn find_by_id(&self, id: &TaskId) -> Option<TtsTask>;
    fn save(&self, task: &TtsTask) -> Result<(), DomainError>;
    fn delete(&self, id: &TaskId) -> Result<(), DomainError>;
    fn find_pending(&self) -> Vec<TtsTask>;
    /// 全量任务列表（按创建时间倒序）
    fn find_all(&self) -> Vec<TtsTask>;
}

/// ASR 任务仓储接口
pub trait AsrTaskRepository: Send + Sync {
    fn find_by_id(&self, id: &TaskId) -> Option<AsrTask>;
    fn save(&self, task: &AsrTask) -> Result<(), DomainError>;
    fn delete(&self, id: &TaskId) -> Result<(), DomainError>;
    fn find_pending(&self) -> Vec<AsrTask>;
    /// 全量任务列表（按创建时间倒序）
    fn find_all(&self) -> Vec<AsrTask>;
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
