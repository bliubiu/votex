//! DeepSeek LLM API 适配器
//!
//! 通过 DeepSeek 在线 API 实现 LLM 对话补全。
//! API 文档: https://api-docs.deepseek.com/

use crate::api::base::{ApiConfig, ApiError, BaseApiClient};
use std::time::Duration;
use votex_domain::error::LlmError;
use votex_domain::llm::provider::LlmProvider;
use votex_domain::llm::value_object::{LlmConfig, LlmMessage};

/// DeepSeek API 基础地址
const DEEPSEEK_BASE_URL: &str = "https://api.deepseek.com";

/// DeepSeek LLM 提供者
pub struct DeepSeekProvider {
    name: String,
    model: String,
    client: BaseApiClient,
}

impl DeepSeekProvider {
    /// 创建 DeepSeek 提供者
    ///
    /// 从环境变量 `DEEPSEEK_API_KEY` 读取 API Key。
    pub fn new(model: &str) -> Result<Self, LlmError> {
        let api_key = std::env::var("DEEPSEEK_API_KEY")
            .map_err(|_| LlmError::ApiKeyNotConfigured)?;

        let config = ApiConfig {
            api_key: Some(api_key),
            endpoint: Some(DEEPSEEK_BASE_URL.to_string()),
            timeout: Duration::from_secs(120),
            retry_max: 3,
            ..Default::default()
        };

        Ok(Self {
            name: format!("deepseek-{}", model),
            model: model.to_string(),
            client: BaseApiClient::new(config),
        })
    }
}

impl LlmProvider for DeepSeekProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn chat(&self, messages: &[LlmMessage], config: Option<&LlmConfig>) -> Result<String, LlmError> {
        let cfg = config.cloned().unwrap_or_default();
        let api_key = self.client.config.api_key.as_deref()
            .ok_or(LlmError::ApiKeyNotConfigured)?;

        // 构建消息数组
        let msgs: Vec<serde_json::Value> = messages
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role.as_str(),
                    "content": m.content
                })
            })
            .collect();

        let body = serde_json::json!({
            "model": self.model,
            "messages": msgs,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "top_p": cfg.top_p,
            "stream": false,
        });

        let headers = vec![
            ("Authorization".to_string(), format!("Bearer {}", api_key)),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];

        let url = format!("{}/v1/chat/completions", DEEPSEEK_BASE_URL);
        let resp = self
            .client
            .post_json(&url, Some(&headers), body)
            .map_err(|e| match e {
                ApiError::ApiKeyNotConfigured(_) => LlmError::ApiKeyNotConfigured,
                ApiError::AuthenticationFailed => {
                    LlmError::ApiError("认证失败，请检查 API Key".to_string())
                }
                ApiError::RateLimited => LlmError::NetworkError("请求被限流".to_string()),
                ApiError::NetworkError(s) => LlmError::NetworkError(s),
                ApiError::RequestFailed(s) => LlmError::ApiError(s),
                ApiError::ParseFailed(s) => LlmError::ApiError(s),
            })?;

        let result: serde_json::Value =
            resp.json().map_err(|e| LlmError::ApiError(format!("解析响应失败: {}", e)))?;

        let content = result["choices"][0]["message"]["content"]
            .as_str()
            .ok_or(LlmError::EmptyResponse)?
            .to_string();

        Ok(content)
    }
}
