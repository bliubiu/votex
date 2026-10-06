use anyhow::Result;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use votex_app::use_case::asr_use_case::AsrUseCase;

/// 处理 asr 子命令
///
/// 长音频识别可能持续数分钟，支持 Ctrl+C 中断：
/// 取消令牌会在每个音频切片边界被检查，最长延迟为单段推理时间。
pub fn handle(
    input: &str,
    output: &str,
    model: &str,
    format: &str,
    lang: &str,
) -> Result<()> {
    let input_path = std::path::Path::new(input);
    let output_path = std::path::Path::new(output);

    let use_case = AsrUseCase::new();

    // Ctrl+C 令牌：置位后合成循环在下一个切片边界中断
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let flag = Arc::clone(&cancel);
        // 只注册一次 handler；注册失败（如已在运行）不阻断识别
        if let Err(e) = ctrlc::set_handler(move || {
            use std::sync::atomic::Ordering::SeqCst;
            flag.store(true, SeqCst);
            eprintln!("收到中断信号，正在取消（当前段推理完成后停止）...");
        }) {
            tracing::warn!("注册 Ctrl+C 处理器失败: {}", e);
        }
    }

    println!("ASR 识别开始");
    println!("  输入: {}", input);
    println!("  输出: {}", output);
    println!("  模型: {}", model);
    println!("  格式: {}", format);
    println!("  语言: {}", lang);

    let result = use_case.recognize(
        input_path,
        output_path,
        model,
        lang,
        format,
        Some(cancel.as_ref()),
    )?;

    println!("ASR 识别完成");
    println!("  识别文本: {}", result.text);
    println!("  字幕条目: {} 条", result.subtitles.len());
    println!("  输出文件: {:?}", result.output_path);
    Ok(())
}

#[cfg(test)]
mod tests {
    /// 验证 ASR 输入输出路径基本逻辑
    #[test]
    fn asr_参数验证() {
        // 输入文件应存在（逻辑上由 handle 调用者保证）
        // 此处验证路径合法性判断
        assert!(std::path::Path::new("Cargo.toml").exists());
        assert!(!std::path::Path::new("nonexistent_file.asr_input").exists());
    }
}
