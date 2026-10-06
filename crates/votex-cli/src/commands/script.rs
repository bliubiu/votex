//! 文案生成子命令
//!
//! 只做参数映射与输出格式化，
//! 业务逻辑（提示词拼装、API 调用、响应解析）全部在
//! `votex_app::use_case::script_generate`。
//!
//! 旧实现在本文件里复制了一份完整的提示词与 HTTP 调用，
//! 与用例实现漂移（超时 60s vs 120s、缺少标题解析），已统一到用例。

use anyhow::{Context, Result};
use clap::Args;
use std::path::Path;
use votex_app::use_case::script_generate::{ScriptGenerateRequest, ScriptGenerateUseCase};

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
    println!("正在生成文案...");
    println!("   主题: {}", cmd.topic);
    println!("   风格: {}", cmd.style);
    println!("   引擎: {}", cmd.engine);

    let response = ScriptGenerateUseCase.execute(ScriptGenerateRequest {
        topic: cmd.topic.clone(),
        style: cmd.style.clone(),
        duration_seconds: cmd.duration,
        engine: cmd.engine.clone(),
        extra_instructions: cmd.extra.clone(),
    })?;

    if let Some(ref out_path) = cmd.output {
        std::fs::write(Path::new(out_path), &response.content)
            .context("写入输出文件失败")?;
        println!("\n文案已保存到: {}", out_path);
    } else {
        if !response.title.is_empty() {
            println!("\n标题: {}", response.title);
        }
        println!("\n生成结果:\n\n{}", response.content);
    }

    Ok(())
}
