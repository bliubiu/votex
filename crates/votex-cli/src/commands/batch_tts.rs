use anyhow::Result;
use std::path::Path;
use votex_app::use_case::batch_tts_use_case::{BatchTtsConfig, BatchTtsUseCase};
use votex_domain::tts::value_object::{
    parse_tts_engine, AudioFormat, DenoiseLevel, TTS_ENGINE_HINT,
};

/// 处理 batch tts 子命令
pub fn handle(
    input: &str,
    output_dir: &str,
    engine: &str,
    voice: &str,
    speed: f32,
    lang: Option<String>,
    model: Option<String>,
    format: &str,
    concurrency: usize,
    denoise: bool,
    denoise_level: &str,
) -> Result<()> {
    let engine_kind = parse_tts_engine(engine).ok_or_else(|| {
        anyhow::anyhow!(
            "不支持的 TTS 引擎: {}，可选: {}",
            engine,
            TTS_ENGINE_HINT
        )
    })?;

    let audio_format = match format {
        "mp3" => AudioFormat::Mp3,
        _ => AudioFormat::Wav,
    };

    let dl = match denoise_level {
        "low" => DenoiseLevel::Low,
        "medium" => DenoiseLevel::Medium,
        "high" => DenoiseLevel::High,
        _ => DenoiseLevel::Low,
    };

    let input_path = Path::new(input);
    let output_path = Path::new(output_dir);

    println!("批量 TTS 合成开始");
    println!("  输入: {}", input);
    println!("  输出目录: {}", output_dir);
    println!("  引擎: {}", engine);
    println!("  音色: {}", voice);
    println!("  语速: {}", speed);
    println!("  格式: {}", format);
    println!("  并发: {}", concurrency);
    if denoise {
        println!("  降噪: {:?}", dl);
    }

    // 模型变体映射
    let model_override = model.as_deref().map(|m| match m {
        "0.6b" => "qwen3-tts-0.6b",
        "1.7b" => "qwen3-tts-1.7b",
        _ => m,
    });

    let config = BatchTtsConfig::basic(
        input_path,
        output_path,
        engine_kind,
        voice,
        speed,
        audio_format,
        lang.as_deref(),
        model_override,
        concurrency,
        denoise,
        dl,
    );

    let results = BatchTtsUseCase::execute(config)?;

    let success_count = results.iter().filter(|r| r.success).count();
    let fail_count = results.len() - success_count;

    println!("\n批量 TTS 合成完成");
    println!("  总计: {} 条", results.len());
    println!("  成功: {}", success_count);
    println!("  失败: {}", fail_count);

    if fail_count > 0 {
        for r in results.iter().filter(|r| !r.success) {
            println!("  ❌ {:?}: {}", r.output_path, r.error.as_deref().unwrap_or("未知错误"));
        }
    }

    Ok(())
}
