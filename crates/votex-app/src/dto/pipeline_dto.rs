use serde::{Deserialize, Serialize};

/// 流水线请求 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineRequest {
    pub kind: String,
    pub input_path: String,
    pub output_dir: String,
}

/// 流水线响应 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineResponse {
    pub pipeline_id: String,
    pub audio_path: Option<String>,
    pub subtitle_path: Option<String>,
}
