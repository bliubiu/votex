use crate::error::LlmError;
use crate::llm::value_object::{LlmConfig, LlmMessage};

/// LLM 提供者接口
pub trait LlmProvider: Send + Sync {
    /// 模型名称
    fn name(&self) -> &str;

    /// 是否支持流式输出
    fn supports_streaming(&self) -> bool {
        false
    }

    /// 对话补全
    fn chat(&self, messages: &[LlmMessage], config: Option<&LlmConfig>) -> Result<String, LlmError>;

    /// 单轮生成（简化接口）
    fn generate(&self, prompt: &str, system_prompt: Option<&str>, config: Option<&LlmConfig>) -> Result<String, LlmError> {
        let mut messages = Vec::new();
        if let Some(sp) = system_prompt {
            messages.push(LlmMessage::system(sp));
        }
        messages.push(LlmMessage::user(prompt));
        self.chat(&messages, config)
    }
}
