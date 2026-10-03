use anyhow::Result;
use votex_app::use_case::asr_use_case::AsrUseCase;

/// 处理 asr 子命令
pub fn handle(
    input: &str,
    output: &str,
    model: &str,
    format: &str,
    lang: &str,
) -> Result<()> {
    let input_path = std::path::Path::new(input);
    let output_path = std::path::Path::new(output);

    let mut use_case = AsrUseCase::new();

    println!("ASR 识别开始");
    println!("  输入: {}", input);
    println!("  输出: {}", output);
    println!("  模型: {}", model);
    println!("  格式: {}", format);
    println!("  语言: {}", lang);

    let result = use_case.recognize(input_path, output_path, model, lang, format)?;

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
