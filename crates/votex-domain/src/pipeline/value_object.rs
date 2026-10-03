use serde::{Deserialize, Serialize};

/// 流水线类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineKind {
    Audiobook,
    Subtitle,
    AudiobookWithSubtitle,
}

/// 阶段类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StageKind {
    TextPreprocess,
    TextSegment,
    TtsSynthesize,
    AudioConcat,
    AudioExtract,
    AudioSlice,
    AsrRecognize,
    OcrRecognize,
    SubtitleAlign,
    Export,
}

/// 阶段状态
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageStatus {
    Waiting,
    Running,
    Completed,
    Failed(String),
    Skipped,
}

/// 流水线状态
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineStatus {
    Idle,
    Running(usize),
    Paused(usize),
    Completed,
    Failed(usize, String),
    Cancelled,
}

/// 质量升级建议
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualityUpgrade {
    pub current_engine: String,
    pub suggested_engine: String,
    pub reason: String,
}
