use clap::Args;
use anyhow::{Context, Result};
use std::path::Path;
use std::time::Duration;

/// 生成短视频文案
#[derive(Args, Debug)]
pub struct ScriptCommand {
    /// 文案主题
    #[arg(short, long)]
    pub topic: String,

    /// 文案风格 (科普 / 故事 / 营销 / 教程)
    #[arg(short, long, default_value = "科普")]
    pub style: String,

    /// 目标时长（秒）
    #[arg(short, long)]
    pub duration: Option<u32>,

    /// LLM 引擎 (deepseek-chat / deepseek-reasoner)
    #[arg(long, default_value = "deepseek-chat")]
    pub engine: String,

    /// 额外要求
    #[arg(long)]
    pub extra: Option<String>,

    /// 输出文件路径（可选，默认输出到终端）
    #[arg(short, long)]
    pub output: Option<String>,
}

/// 处理文案生成命令
pub fn handle(cmd: &ScriptCommand) -> Result<()> {
    println!("📝 正在生成文案...");
    println!("   主题: {}", cmd.topic);
    println!("   风格: {}", cmd.style);
    println!("   引擎: {}", cmd.engine);

    let api_key = std::env::var("DEEPSEEK_API_KEY")
        .context("请设置 DEEPSEEK_API_KEY 环境变量")?;

    let client = votex_infra::api::base::BaseApiClient::new(
        votex_infra::api::base::ApiConfig {
            api_key: Some(api_key),
            endpoint: Some("https://api.deepseek.com".to_string()),
            timeout: Duration::from_secs(60),
            ..Default::default()
        }
    );

    let system_prompt = match cmd.style.as_str() {
        "故事" => "你是一个故事创作专家。请根据给定的主题生成一段吸引人的故事短视频文案。开头制造悬念，中间展开情节，结尾有感悟。语言生动形象，适合朗读。",
        "营销" => "你是一个营销文案专家。请根据给定的产品/主题生成一段有说服力的营销短视频文案。开头直击痛点，中间展示价值，结尾引导行动。语言简洁有力。",
        "教程" => "你是一个教学视频文案专家。请根据给定的主题生成一段步骤清晰的教学短视频文案。开头说明目标，分步骤讲解，结尾总结要点。",
        _ => "你是一个科普短视频文案专家。请根据给定的主题生成一段约60秒的科普短视频文案。开头用问题吸引注意力，中间用通俗语言解释，结尾总结核心观点。语言口语化，适合朗读。",
    };

    let mut user_prompt = format!("请为主题「{}」生成一段{}风格的短视频文案。", cmd.topic, cmd.style);
    if let Some(d) = cmd.duration {
        user_prompt.push_str(&format!("\n目标时长：约{}秒。", d));
    }
    if let Some(ref extra) = cmd.extra {
        user_prompt.push_str(&format!("\n额外要求：{}", extra));
    }
    user_prompt.push_str("\n\n直接输出文案内容，不要包含额外的解释。第一行作为标题。");

    let body = serde_json::json!({
        "model": cmd.engine,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 2048,
    });

    let headers = vec![
        ("Authorization".to_string(), format!("Bearer {}", std::env::var("DEEPSEEK_API_KEY").unwrap_or_default())),
        ("Content-Type".to_string(), "application/json".to_string()),
    ];

    let resp = client.post_json(
        "https://api.deepseek.com/v1/chat/completions",
        Some(&headers),
        body,
    ).context("DeepSeek API 调用失败")?;

    let result: serde_json::Value = resp.json().context("解析响应失败")?;
    let text = result["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    if let Some(ref out_path) = cmd.output {
        std::fs::write(Path::new(out_path), &text)
            .context("写入输出文件失败")?;
        println!("\n✅ 文案已保存到: {}", out_path);
    } else {
        println!("\n✅ 生成结果:\n\n{}", text);
    }

    Ok(())
}
