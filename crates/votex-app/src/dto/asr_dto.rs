use serde::{Deserialize, Serialize};

/// ASR 请求 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrRequest {
    pub input_path: String,
    pub output_path: String,
    pub model: String,
    pub language: String,
    pub output_format: String,
}

/// ASR 响应 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrResponse {
    pub task_id: String,
    pub output_path: String,
}
