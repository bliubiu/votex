use crate::model::value_object::ModelId;
use crate::shared::value_object::TaskId;
use crate::shared::value_object::PipelineId;
use serde::{Deserialize, Serialize};

/// 领域事件
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DomainEvent {
    // 模型事件
    ModelDownloadStarted { model_id: ModelId },
    ModelDownloadProgress { model_id: ModelId, downloaded_bytes: u64, total_bytes: u64 },
    ModelDownloadCompleted { model_id: ModelId },
    ModelVerifyCompleted { model_id: ModelId, success: bool },
    ModelLoaded { model_id: ModelId },

    // TTS 事件
    TtsSegmentCompleted { task_id: TaskId, segment_index: usize, total_segments: usize },
    TtsTaskCompleted { task_id: TaskId },

    // ASR 事件
    AsrSliceCompleted { task_id: TaskId, slice_index: usize, total_slices: usize },
    AsrTaskCompleted { task_id: TaskId },

    // 流水线事件
    PipelineStageCompleted { pipeline_id: PipelineId, stage_index: usize },
    PipelineCompleted { pipeline_id: PipelineId },

    // 通用事件
    TaskFailed { task_id: TaskId, reason: String },
}

/// 事件总线接口
pub trait EventBus: Send + Sync {
    fn publish(&self, event: DomainEvent);
}
