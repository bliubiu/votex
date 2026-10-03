use serde::{Deserialize, Serialize};

/// TTS 请求 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsRequest {
    pub input_path: String,
    pub output_path: String,
    pub engine: String,
    pub voice: String,
    pub speed: f32,
}

/// TTS 响应 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsResponse {
    pub task_id: String,
    pub output_path: String,
}
