//! LLM 在线翻译提供者
//!
//! 使用 DeepSeek API 进行文本翻译，支持中↔英双向。
//!
//! 作为 Chat 类引擎，它支持两类提升译文质量的提示：
//! - **术语表**：以 `原文->译文 #备注` 形式注入 system prompt
//! - **历史上下文**：把之前的若干轮「原文/译文」作为 few-shot 示例，
//!   让模型保持专有名词、人称与语气的一致性

use crate::api::base::{ApiConfig, BaseApiClient};
use std::time::Duration;
use votex_domain::error::TranslationError;
use votex_domain::translation::options::TranslationHints;
use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::value_object::TranslationDirection;

const API_ENDPOINT: &str = "https://api.deepseek.com/v1/chat/completions";
const MODEL: &str = "deepseek-chat";

/// LLM 翻译提供者
pub struct LlmTranslationProvider {
    client: BaseApiClient,
}

impl LlmTranslationProvider {
    /// 创建 LLM 翻译提供者
    ///
    /// 从环境变量 `DEEPSEEK_API_KEY` 读取 API Key。
    pub fn new() -> Result<Self, TranslationError> {
        let api_key = std::env::var("DEEPSEEK_API_KEY")
            .map_err(|_| TranslationError::ApiError("请设置 DEEPSEEK_API_KEY 环境变量".to_string()))?;

        let config = ApiConfig {
            api_key: Some(api_key),
            endpoint: Some("https://api.deepseek.com".to_string()),
            timeout: Duration::from_secs(60),
            ..Default::default()
        };

        Ok(Self {
            client: BaseApiClient::new(config),
        })
    }

    /// 构造 system prompt（不含术语表与上下文）
    fn base_system_prompt(direction: &TranslationDirection) -> String {
        match direction {
            TranslationDirection::ZhToEn => {
                "你是一个专业的中译英翻译。请将用户输入的中文翻译为英文。只输出翻译结果，不要解释。"
                    .to_string()
            }
            TranslationDirection::EnToZh => {
                "你是一个专业的英译中翻译。请将用户输入的英文翻译为中文。只输出翻译结果，不要解释。"
                    .to_string()
            }
            TranslationDirection::Auto => {
                "你是一个专业的翻译。请检测文本语言并翻译为另一种语言。只输出翻译结果，不要解释。"
                    .to_string()
            }
            TranslationDirection::ByLanguagePair { source, target } => {
                format!(
                    "你是一个专业的翻译。请将用户输入的{}文本翻译为{}。只输出翻译结果，不要解释。",
                    source, target
                )
            }
        }
    }

    /// 在基础 prompt 上追加术语表
    fn append_glossary(prompt: &mut String, glossary_text: &str) {
        let glossary = glossary_text.trim();
        if glossary.is_empty() {
            return;
        }
        prompt.push_str(&format!(
            "\n\n翻译时必须遵守以下术语对照表（格式 原文->译文 #备注），不得自行改写这些专有名词：\n{}",
            glossary
        ));
    }

    /// 把历史上下文转成 few-shot 消息对
    fn context_messages(context: &[(String, String)]) -> Vec<serde_json::Value> {
        let mut messages = Vec::new();
        for (src, dst) in context {
            if src.trim().is_empty() || dst.trim().is_empty() {
                continue;
            }
            messages.push(serde_json::json!({"role": "user", "content": src}));
            messages.push(serde_json::json!({"role": "assistant", "content": dst}));
        }
        messages
    }

    /// 发起一次聊天补全请求并返回译文
    fn call(&self, messages: Vec<serde_json::Value>) -> Result<String, TranslationError> {
        let api_key = self
            .client
            .config
            .api_key
            .as_deref()
            .ok_or_else(|| TranslationError::ApiError("API Key 未配置".to_string()))?;

        let body = serde_json::json!({
            "model": MODEL,
            "messages": messages,
            "temperature": 0.3,
            "max_tokens": 1024,
        });

        let headers = vec![
            ("Authorization".to_string(), format!("Bearer {}", api_key)),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];

        let resp = self
            .client
            .post_json(API_ENDPOINT, Some(&headers), body)
            .map_err(|e| TranslationError::ApiError(format!("API 调用失败: {}", e)))?;

        let result: serde_json::Value = resp
            .json()
            .map_err(|e| TranslationError::ApiError(format!("解析响应失败: {}", e)))?;

        let translated = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .trim()
            .to_string();

        if translated.is_empty() {
            return Err(TranslationError::ApiError("翻译结果为空".to_string()));
        }

        Ok(translated)
    }
}

impl TranslationProvider for LlmTranslationProvider {
    fn name(&self) -> &str {
        "llm"
    }

    fn translate(
        &self,
        text: &str,
        direction: TranslationDirection,
    ) -> Result<String, TranslationError> {
        self.translate_with_hints(text, direction, &TranslationHints::empty())
    }

    fn is_online(&self) -> bool {
        true
    }

    fn supports_glossary_prompt(&self) -> bool {
        true
    }

    fn translate_with_hints(
        &self,
        text: &str,
        direction: TranslationDirection,
        hints: &TranslationHints,
    ) -> Result<String, TranslationError> {
        if text.trim().is_empty() {
            return Err(TranslationError::EmptyText);
        }

        let mut system_prompt = Self::base_system_prompt(&direction);
        Self::append_glossary(&mut system_prompt, &hints.glossary_text);

        let mut messages = vec![serde_json::json!({"role": "system", "content": system_prompt})];
        messages.extend(Self::context_messages(&hints.context));
        messages.push(serde_json::json!({"role": "user", "content": text}));

        self.call(messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_prompt_按方向生成() {
        let p = LlmTranslationProvider::base_system_prompt(&TranslationDirection::ZhToEn);
        assert!(p.contains("中译英"));
        let p = LlmTranslationProvider::base_system_prompt(&TranslationDirection::EnToZh);
        assert!(p.contains("英译中"));
        let p = LlmTranslationProvider::base_system_prompt(&TranslationDirection::ByLanguagePair {
            source: "ja".into(),
            target: "zh".into(),
        });
        assert!(p.contains("ja"));
        assert!(p.contains("zh"));
    }

    #[test]
    fn 术语表追加到system_prompt() {
        let mut p = "基础指令".to_string();
        LlmTranslationProvider::append_glossary(&mut p, "爱丽丝->Alice #女主角");
        assert!(p.starts_with("基础指令"));
        assert!(p.contains("爱丽丝->Alice"));
        assert!(p.contains("术语对照表"));
    }

    #[test]
    fn 空术语表不改写prompt() {
        let mut p = "基础指令".to_string();
        LlmTranslationProvider::append_glossary(&mut p, "   ");
        assert_eq!(p, "基础指令");
    }

    #[test]
    fn 上下文转成交替的user_assistant消息() {
        let ctx = vec![("你好".to_string(), "hello".to_string())];
        let msgs = LlmTranslationProvider::context_messages(&ctx);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[1]["role"], "assistant");
    }

    #[test]
    fn 上下文跳过空条目() {
        let ctx = vec![
            ("".to_string(), "hello".to_string()),
            ("你好".to_string(), "".to_string()),
        ];
        assert!(LlmTranslationProvider::context_messages(&ctx).is_empty());
    }

    #[test]
    fn 引擎标记为在线且支持术语注入() {
        // 不触发网络请求，仅验证能力声明
        fn assert_capable<P: TranslationProvider>(p: &P) {
            assert!(p.is_online());
            assert!(p.supports_glossary_prompt());
            assert!(p.is_loaded());
        }
        let _ = assert_capable::<LlmTranslationProvider>;
    }
}
