use serde::{Deserialize, Serialize};

/// 翻译请求 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationRequest {
    pub text: String,
    pub direction: String,
    pub engine: String,
}

/// 翻译响应 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResponse {
    pub translated_text: String,
    pub engine: String,
}
