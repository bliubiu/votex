//! Qwen-MT 在线翻译引擎（DashScope API）
//!
//! 使用阿里云百炼（DashScope）的 Qwen-MT API 进行翻译。
//! 基于 Qwen 大模型，支持流畅的中英互译和多语言翻译。
//!
//! # 环境变量
//! - `DASHSCOPE_API_KEY`：阿里云百炼的 API Key
//!
//! # API 说明
//! 端点：`https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions`
//! 模型：`qwen-mt-turbo`
//! 认证方式：Bearer Token
//!
//! 参考：https://help.aliyun.com/zh/model-studio/developer-reference/use-qwen-mt

use crate::api::base::{ApiConfig, BaseApiClient};
use std::borrow::Cow;
use std::time::Duration;
use votex_domain::error::TranslationError;
use votex_domain::translation::options::TranslationHints;
use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::value_object::TranslationDirection;

/// Qwen-MT API 端点（OpenAI 兼容格式）
const QWEN_MT_ENDPOINT: &str =
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions";

/// Qwen-MT 在线翻译提供者
pub struct QwenMtProvider {
    client: BaseApiClient,
}

impl QwenMtProvider {
    /// 创建 Qwen-MT 翻译提供者
    ///
    /// 从环境变量 `DASHSCOPE_API_KEY` 读取 API Key。
    pub fn new() -> Result<Self, TranslationError> {
        let api_key = std::env::var("DASHSCOPE_API_KEY").map_err(|_| {
            TranslationError::ApiError(
                "请设置 DASHSCOPE_API_KEY 环境变量（阿里云百炼 API Key，\
                 可从 https://help.aliyun.com/zh/model-studio/getting-started/first-api-call-to-qwen \
                 获取）"
                    .to_string(),
            )
        })?;

        let config = ApiConfig {
            api_key: Some(api_key),
            endpoint: Some("https://dashscope-intl.aliyuncs.com".to_string()),
            timeout: Duration::from_secs(60),
            ..Default::default()
        };

        Ok(Self {
            client: BaseApiClient::new(config),
        })
    }

    /// 将语言代码映射为 Qwen-MT API 使用的语言名称
    fn lang_code_to_name(code: &str) -> Cow<'static, str> {
        // Qwen-MT 使用完整语言名称而非缩写
        match code.to_lowercase().as_str() {
            "zh" | "zh-cn" | "zh-chs" | "chinese" => Cow::Borrowed("Chinese"),
            "en" | "en-us" | "english" => Cow::Borrowed("English"),
            "ja" | "japanese" => Cow::Borrowed("Japanese"),
            "ko" | "korean" => Cow::Borrowed("Korean"),
            "fr" | "french" => Cow::Borrowed("French"),
            "de" | "german" => Cow::Borrowed("German"),
            "es" | "spanish" => Cow::Borrowed("Spanish"),
            "pt" | "portuguese" => Cow::Borrowed("Portuguese"),
            "ru" | "russian" => Cow::Borrowed("Russian"),
            "ar" | "arabic" => Cow::Borrowed("Arabic"),
            "it" | "italian" => Cow::Borrowed("Italian"),
            "nl" | "dutch" => Cow::Borrowed("Dutch"),
            "pl" | "polish" => Cow::Borrowed("Polish"),
            "tr" | "turkish" => Cow::Borrowed("Turkish"),
            "vi" | "vietnamese" => Cow::Borrowed("Vietnamese"),
            "th" | "thai" => Cow::Borrowed("Thai"),
            "id" | "indonesian" => Cow::Borrowed("Indonesian"),
            // 不支持的语言保持原样传递
            _ => Cow::Owned(code.to_string()),
        }
    }

    /// 将 TranslationDirection 映射为 Qwen-MT 的 (source_lang, target_lang)
    fn direction_to_lang(direction: &TranslationDirection) -> (Cow<'static, str>, Cow<'static, str>) {
        match direction {
            TranslationDirection::ZhToEn => (Cow::Borrowed("Chinese"), Cow::Borrowed("English")),
            TranslationDirection::EnToZh => (Cow::Borrowed("English"), Cow::Borrowed("Chinese")),
            TranslationDirection::Auto => (Cow::Borrowed("auto"), Cow::Borrowed("Chinese")),
            TranslationDirection::ByLanguagePair { source, target } => {
                (Self::lang_code_to_name(source), Self::lang_code_to_name(target))
            }
        }
    }
}

impl QwenMtProvider {
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

    /// 把术语表拼到待译文本前，作为翻译参考
    fn with_glossary_prefix(text: &str, glossary_text: &str) -> String {
        let glossary = glossary_text.trim();
        if glossary.is_empty() {
            return text.to_string();
        }
        format!(
            "参考以下术语对照表（格式 原文->译文 #备注），翻译时必须采用表中译法：\n{}\n\n待翻译文本：\n{}",
            glossary, text
        )
    }
}

impl TranslationProvider for QwenMtProvider {
    fn name(&self) -> &str {
        "qwen-mt"
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

        let (source_lang, target_lang) = Self::direction_to_lang(&direction);
        let source_lang = source_lang.as_ref();
        let target_lang = target_lang.as_ref();

        let api_key = self.client.config.api_key.as_deref().ok_or_else(|| {
            TranslationError::ApiError("API Key 未配置".to_string())
        })?;

        let mut messages = Self::context_messages(&hints.context);
        messages.push(serde_json::json!({
            "role": "user",
            "content": Self::with_glossary_prefix(text, &hints.glossary_text),
        }));

        // Qwen-MT 使用 OpenAI 兼容格式
        // translation_options 放在顶级字段（非嵌套 extra_body）
        let body = serde_json::json!({
            "model": "qwen-mt-turbo",
            "messages": messages,
            "translation_options": {
                "source_lang": source_lang,
                "target_lang": target_lang
            }
        });

        let headers = vec![
            ("Authorization".to_string(), format!("Bearer {}", api_key)),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];

        let resp = self
            .client
            .post_json(QWEN_MT_ENDPOINT, Some(&headers), body)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 语言代码映射为完整名称() {
        assert_eq!(QwenMtProvider::lang_code_to_name("zh"), "Chinese");
        assert_eq!(QwenMtProvider::lang_code_to_name("en"), "English");
        assert_eq!(QwenMtProvider::lang_code_to_name("ja"), "Japanese");
        assert_eq!(QwenMtProvider::lang_code_to_name("xx"), "xx");
    }

    #[test]
    fn 方向转语言对() {
        let (s, t) = QwenMtProvider::direction_to_lang(&TranslationDirection::ZhToEn);
        assert_eq!((s.as_ref(), t.as_ref()), ("Chinese", "English"));
        let (s, t) = QwenMtProvider::direction_to_lang(&TranslationDirection::EnToZh);
        assert_eq!((s.as_ref(), t.as_ref()), ("English", "Chinese"));
    }

    #[test]
    fn 空术语表时不改写待译文本() {
        assert_eq!(QwenMtProvider::with_glossary_prefix("你好", "  "), "你好");
    }

    #[test]
    fn 术语表作为前缀拼入文本() {
        let out = QwenMtProvider::with_glossary_prefix("你好", "甲->A");
        assert!(out.contains("甲->A"));
        assert!(out.ends_with("你好"));
    }

    #[test]
    fn 上下文生成交替消息且跳过空条目() {
        let ctx = vec![
            ("a".to_string(), "b".to_string()),
            ("".to_string(), "x".to_string()),
        ];
        let msgs = QwenMtProvider::context_messages(&ctx);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0]["role"], "user");
    }
}
