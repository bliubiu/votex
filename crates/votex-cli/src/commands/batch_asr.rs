use anyhow::Result;
use std::path::Path;
use votex_app::use_case::batch_asr_use_case::{BatchAsrConfig, BatchAsrUseCase};
use votex_domain::tts::value_object::DenoiseLevel;

/// 处理 batch asr 子命令
pub fn handle(
    input_dir: &str,
    output_dir: &str,
    model: &str,
    format: &str,
    lang: &str,
    recursive: bool,
    concurrency: usize,
    denoise: bool,
    denoise_level: &str,
) -> Result<()> {
    let dl = match denoise_level {
        "low" => DenoiseLevel::Low,
        "medium" => DenoiseLevel::Medium,
        "high" => DenoiseLevel::High,
        _ => DenoiseLevel::Low,
    };

    let extensions = vec!["wav".to_string(), "mp3".to_string(), "flac".to_string()];

    println!("批量 ASR 识别开始");
    println!("  输入目录: {}", input_dir);
    println!("  输出目录: {}", output_dir);
    println!("  模型: {}", model);
    println!("  格式: {}", format);
    println!("  语言: {}", lang);
    println!("  递归: {}", if recursive { "是" } else { "否" });
    println!("  并发: {}", concurrency);
    if denoise {
        println!("  降噪: {:?}", dl);
    }

    let config = BatchAsrConfig {
        input_dir: Path::new(input_dir),
        output_dir: Path::new(output_dir),
        model,
        language: lang,
        format,
        recursive,
        extensions,
        concurrency,
        denoise,
        denoise_level: dl,
    };

    let results = BatchAsrUseCase::execute(config)?;

    let success_count = results.iter().filter(|r| r.success).count();
    let fail_count = results.len() - success_count;

    println!("\n批量 ASR 识别完成");
    println!("  总计: {} 个文件", results.len());
    println!("  成功: {}", success_count);
    println!("  失败: {}", fail_count);

    if fail_count > 0 {
        for r in results.iter().filter(|r| !r.success) {
            println!("  ❌ {:?}: {}", r.input_path, r.error.as_deref().unwrap_or("未知错误"));
        }
    }

    Ok(())
}
