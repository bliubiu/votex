//! 文案生成用例
//!
//! 通过 LLM API 根据主题和风格生成短视频文案。

use anyhow::{Context, Result};
use std::time::Duration;
use votex_infra::api::base::{ApiConfig, BaseApiClient};

/// 文案生成请求
#[derive(Debug, Clone)]
pub struct ScriptGenerateRequest {
    /// 文案主题
    pub topic: String,
    /// 文案风格（科普/故事/营销/教程）
    pub style: String,
    /// 目标时长（秒）
    pub duration_seconds: Option<u32>,
    /// LLM 引擎
    pub engine: String,
    /// 额外要求
    pub extra_instructions: Option<String>,
}

/// 文案生成响应
#[derive(Debug, Clone)]
pub struct ScriptGenerateResponse {
    /// 标题（第一行）
    pub title: String,
    /// 完整文案内容
    pub content: String,
}

/// 文案生成用例
pub struct ScriptGenerateUseCase;

impl ScriptGenerateUseCase {
    /// 执行文案生成
    pub fn execute(&self, req: ScriptGenerateRequest) -> Result<ScriptGenerateResponse> {
        let api_key = std::env::var("DEEPSEEK_API_KEY")
            .context("请设置 DEEPSEEK_API_KEY 环境变量")?;

        let client = BaseApiClient::new(ApiConfig {
            api_key: Some(api_key.clone()),
            endpoint: Some("https://api.deepseek.com".to_string()),
            timeout: Duration::from_secs(120),
            ..Default::default()
        });

        let system_prompt = match req.style.as_str() {
            "故事" => "你是一个故事创作专家。请根据给定的主题生成一段吸引人的故事短视频文案。开头制造悬念，中间展开情节，结尾有感悟。语言生动形象，适合朗读。",
            "营销" => "你是一个营销文案专家。请根据给定的产品/主题生成一段有说服力的营销短视频文案。开头直击痛点，中间展示价值，结尾引导行动。语言简洁有力。",
            "教程" => "你是一个教学视频文案专家。请根据给定的主题生成一段步骤清晰的教学短视频文案。开头说明目标，分步骤讲解，结尾总结要点。",
            _ => "你是一个科普短视频文案专家。请根据给定的主题生成一段约60秒的科普短视频文案。开头用问题吸引注意力，中间用通俗语言解释，结尾总结核心观点。语言口语化，适合朗读。",
        };

        let mut user_prompt = format!(
            "请为主题「{}」生成一段{}风格的短视频文案。",
            req.topic, req.style
        );
        if let Some(d) = req.duration_seconds {
            user_prompt.push_str(&format!("\n目标时长：约{}秒。", d));
        }
        if let Some(ref extra) = req.extra_instructions {
            user_prompt.push_str(&format!("\n额外要求：{}", extra));
        }
        user_prompt.push_str("\n\n直接输出文案内容，不要包含额外的解释。第一行作为标题。");

        let body = serde_json::json!({
            "model": req.engine,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2048,
        });

        let headers = vec![
            ("Authorization".to_string(), format!("Bearer {}", api_key)),
            ("Content-Type".to_string(), "application/json".to_string()),
        ];

        let resp = client
            .post_json(
                "https://api.deepseek.com/v1/chat/completions",
                Some(&headers),
                body,
            )
            .context("DeepSeek API 调用失败")?;

        let result: serde_json::Value = resp.json().context("解析响应失败")?;
        let content = result["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        // 第一行为标题
        let title = content.lines().next().unwrap_or("").to_string();

        Ok(ScriptGenerateResponse { title, content })
    }
}
