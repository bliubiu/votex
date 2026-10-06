use clap::Args;
use anyhow::{Context, Result};
use std::path::Path;
use votex_domain::model::value_object::EngineKind;
use votex_domain::tts::value_object::{parse_tts_engine, AudioFormat};

/// 解析视频流水线的引擎名。
///
/// TTS 分支委托 domain 的 `parse_tts_engine`（与 `tts` / `batch tts` / `pipeline`
/// 共用同一份映射，docs/20 F62）；本命令额外接受 ASR 引擎串（视频流水线自带
/// ASR 字幕步骤），未知值仍回落到 kokoro（保持既有行为，不静默改变默认引擎）。
fn parse_engine(s: &str) -> EngineKind {
    match s.to_lowercase().as_str() {
        "whisper" => EngineKind::Whisper,
        "sensevoice" => EngineKind::SenseVoice,
        other => parse_tts_engine(other).unwrap_or(EngineKind::Kokoro),
    }
}

/// 一键生成短视频（LLM 文案 + TTS 配音 + ASR 字幕 + 素材合成）
#[derive(Args, Debug)]
pub struct VideoCommand {
    /// 文案内容（直接传入文本）
    pub script: String,

    /// 输出文件路径
    #[arg(short, long, default_value = "output.mp4")]
    pub output: String,

    /// TTS 引擎 (kokoro / indextts2 / indextts25 / qwen3 / cosyvoice3)
    #[arg(long, default_value = "kokoro")]
    pub tts_engine: String,

    /// 音色
    #[arg(long, default_value = "zf_001")]
    pub voice: String,

    /// 语速 (0.5~2.0)
    #[arg(long, default_value = "1.0")]
    pub speed: f32,

    /// 是否生成字幕
    #[arg(long)]
    pub subtitle: bool,

    /// 字幕格式 (srt / lrc / txt)
    #[arg(long, default_value = "srt")]
    pub subtitle_format: String,

    /// ASR 引擎 (whisper-base / whisper-small / sensevoice)
    #[arg(long, default_value = "whisper-base")]
    pub asr_engine: String,

    /// 输出分辨率 (1920x1080 / 1080x1920 / ...)
    #[arg(long, default_value = "1920x1080")]
    pub resolution: String,

    /// 合成引擎 (ffmpeg / moviepy)
    #[arg(long, default_value = "ffmpeg")]
    pub composer: String,

    /// 背景图片路径（多个用逗号分隔）
    #[arg(long)]
    pub bg_images: Option<String>,

    /// 背景音乐路径
    #[arg(long)]
    pub bg_music: Option<String>,
}

/// 处理短视频生成命令
pub fn handle(cmd: &VideoCommand) -> Result<()> {
    println!("🎬 正在生成短视频...");
    println!("   文案长度: {} 字", cmd.script.len());
    println!("   TTS 引擎: {}", cmd.tts_engine);
    println!("   音色: {}", cmd.voice);

    // 1. TTS 合成配音
    let audio_path = Path::new("_temp_audio.wav");
    {
        let tts = votex_app::use_case::tts_use_case::TtsUseCase::new();
        tts.synthesize(
            &cmd.script,
            audio_path,
            parse_engine(&cmd.tts_engine),
            &cmd.voice,
            cmd.speed,
            AudioFormat::Wav,
            None,
            None,
            None,
        ).context("TTS 合成失败")?;
        println!("   ✅ 配音合成完成");
    }

    // 2. ASR 字幕（可选）
    let subtitle_path = if cmd.subtitle {
        let srt_path = Path::new("_temp_subtitle.srt");
        let asr = votex_app::use_case::asr_use_case::AsrUseCase::new();
        asr.recognize(
            audio_path,
            srt_path,
            &cmd.asr_engine,
            "zh",
            &cmd.subtitle_format,
            None,
        ).context("ASR 识别失败")?;
        println!("   ✅ 字幕生成完成");
        Some(srt_path.to_path_buf())
    } else {
        None
    };

    // 3. 合成视频
    let output_path = Path::new(&cmd.output);
    let bg_image_list: Option<Vec<std::path::PathBuf>> = cmd.bg_images.as_ref().map(|s| {
        s.split(',').map(|p| std::path::PathBuf::from(p.trim())).collect()
    });

    let mut ffmpeg_cmd = std::process::Command::new("ffmpeg");
    ffmpeg_cmd.arg("-y");

    if let Some(ref images) = bg_image_list {
        let list_path = Path::new("_images.txt");
        let mut list_content = String::new();
        for img in images {
            list_content.push_str(&format!("file '{}'\nduration 5.0\n", img.display()));
        }
        std::fs::write(list_path, &list_content).context("写入图片列表失败")?;
        ffmpeg_cmd.args(["-f", "concat", "-safe", "0", "-i", list_path.to_str().unwrap()]);
    } else {
        ffmpeg_cmd.args(["-f", "lavfi", "-i", "color=c=#2C3E50:s=1920x1080:d=30"]);
    }

    ffmpeg_cmd.args(["-i", &audio_path.as_os_str().to_string_lossy()]);

    let mut vf_parts = vec![format!("scale={},setsar=1", cmd.resolution)];
    if let Some(ref sub) = subtitle_path {
        let escaped = sub.as_os_str().to_string_lossy().replace(":", "\\:");
        vf_parts.push(format!("subtitles='{}'", escaped));
    }
    ffmpeg_cmd.args(["-vf", &vf_parts.join(",")]);
    ffmpeg_cmd.args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"]);
    ffmpeg_cmd.arg(output_path.as_os_str());

    let status = ffmpeg_cmd.status().context("ffmpeg 执行失败")?;
    if !status.success() {
        anyhow::bail!("ffmpeg 合成返回非零退出码");
    }

    // 清理临时文件
    let _ = std::fs::remove_file(audio_path);
    if let Some(ref sub) = subtitle_path {
        let _ = std::fs::remove_file(sub);
    }

    println!("\n✅ 视频生成完成: {}", cmd.output);
    Ok(())
}
