use anyhow::Result;
use std::sync::Arc;
use votex_app::use_case::pipeline_use_case::PipelineUseCase;
use votex_domain::pipeline::value_object::PipelineKind;
use votex_domain::repository::PipelineRepository;

/// 处理 pipeline 子命令
pub fn handle(
    kind: &str,
    input: &str,
    output: &str,
    engine: &str,
    voice: &str,
    speed: f32,
    asr_model: &str,
    subtitle_format: &str,
    repo: Option<Arc<dyn PipelineRepository>>,
) -> Result<()> {
    let pipeline_kind = match kind {
        "audiobook" => PipelineKind::Audiobook,
        "subtitle" => PipelineKind::Subtitle,
        "audiobook-sub" => PipelineKind::AudiobookWithSubtitle,
        _ => anyhow::bail!("不支持的流水线类型: {}，可选: audiobook, subtitle, audiobook-sub", kind),
    };

    let input_path = std::path::Path::new(input);
    let output_dir = std::path::Path::new(output);

    println!("流水线执行开始");
    println!("  类型: {}", kind);
    println!("  输入: {}", input);
    println!("  输出: {}", output);
    println!("  引擎: {}", engine);
    println!("  音色: {}", voice);
    println!("  语速: {}", speed);
    println!("  ASR 模型: {}", asr_model);
    println!("  字幕格式: {}", subtitle_format);

    let mut use_case = PipelineUseCase::new();
    if let Some(r) = repo {
        use_case = use_case.with_repo(r);
    }
    use_case.execute(
        pipeline_kind,
        input_path,
        output_dir,
        engine,
        voice,
        speed,
        asr_model,
        subtitle_format,
        None, // 无进度回调
    )?;

    println!("流水线执行完成");
    Ok(())
}
