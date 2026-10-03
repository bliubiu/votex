use serde::{Deserialize, Serialize};

/// LLM 对话消息角色
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

impl MessageRole {
    pub fn as_str(&self) -> &str {
        match self {
            MessageRole::System => "system",
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
        }
    }
}

/// LLM 对话消息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmMessage {
    pub role: MessageRole,
    pub content: String,
}

impl LlmMessage {
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        Self {
            role,
            content: content.into(),
        }
    }

    pub fn system(content: impl Into<String>) -> Self {
        Self::new(MessageRole::System, content)
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self::new(MessageRole::User, content)
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self::new(MessageRole::Assistant, content)
    }
}

/// LLM 推理参数配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmConfig {
    pub temperature: f64,
    pub max_tokens: u32,
    pub top_p: f64,
    pub top_k: u32,
    pub repetition_penalty: f64,
    pub stream: bool,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            max_tokens: 2048,
            top_p: 0.9,
            top_k: 50,
            repetition_penalty: 1.0,
            stream: false,
        }
    }
}

/// 支持的 LLM 引擎
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LlmEngine {
    DeepSeekChat,
    DeepSeekReasoner,
}

impl LlmEngine {
    pub fn as_str(&self) -> &str {
        match self {
            LlmEngine::DeepSeekChat => "deepseek-chat",
            LlmEngine::DeepSeekReasoner => "deepseek-reasoner",
        }
    }
}

impl std::fmt::Display for LlmEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
