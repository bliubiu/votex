//! `votex dub` —— 视频配音命令
//!
//! 薄壳：参数转发给 `votex_app::use_case::video_dub_use_case`，只做输出格式化。

use anyhow::Result;
use std::sync::Arc;

use votex_app::use_case::video_dub_use_case::{VideoDubRequest, VideoDubUseCase};

/// 处理 dub 子命令
#[allow(clippy::too_many_arguments)]
pub fn handle(
    video: &str,
    output: &str,
    asr_engine: &str,
    lang: &str,
    engine: &str,
    voice: &str,
    speed: f32,
    translate: Option<&str>,
    translate_engine: &str,
    max_tempo: f32,
    verify: bool,
    keep_background: bool,
) -> Result<()> {
    let req = VideoDubRequest {
        video_path: video.to_string(),
        asr_engine: asr_engine.to_string(),
        language: lang.to_string(),
        tts_engine: engine.to_string(),
        voice: voice.to_string(),
        speed,
        translate_direction: translate.map(|s| s.to_string()),
        translate_engine: translate_engine.to_string(),
        max_tempo,
        verify,
        keep_background,
    };

    println!(
        "视频配音: {}（ASR: {}，TTS: {}/{}，翻译: {}）",
        video,
        asr_engine,
        engine,
        voice,
        req.translate_direction.as_deref().unwrap_or("不翻译")
    );

    let use_case = VideoDubUseCase::new();
    let result = use_case.execute(
        &req,
        std::path::Path::new(output),
        Some(Box::new(|done, total, msg| {
            if total > 0 {
                println!("  [{}/{}] {}", done, total, msg);
            } else {
                println!("  {}", msg);
            }
        })),
        Some(Arc::new(std::sync::atomic::AtomicBool::new(false))),
    );

    match result {
        Ok(resp) => {
            println!("配音完成: {} 段", resp.segments);
            println!("  配音视频: {}", resp.output_video);
            println!("  配音音轨: {}", resp.dubbed_audio);
            println!("  配音字幕: {}", resp.subtitle_path);
            if let Some(score) = resp.verify_score {
                println!("  质检得分: {:.3}（>0.7 一般可接受）", score);
                if let Some(p) = resp.verify_subtitle_path {
                    println!("  质检字幕: {}", p);
                }
            }
            Ok(())
        }
        Err(e) => Err(e),
    }
}
