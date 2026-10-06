use anyhow::Result;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use votex_app::use_case::tts_use_case::{SynthesisOptions, TtsUseCase};
use votex_domain::tts::value_object::{parse_tts_engine, AudioFormat, TTS_ENGINE_HINT};

/// 处理 tts 子命令
#[allow(clippy::too_many_arguments)]
pub fn handle(
    input: &str,
    output: &str,
    engine: &str,
    voice: &str,
    speed: f32,
    lang: Option<String>,
    model: Option<String>,
    session_dir: Option<String>,
    role_map: Option<String>,
    loudnorm: bool,
    atempo: Option<f32>,
) -> Result<()> {
    let engine_kind = parse_tts_engine(engine).ok_or_else(|| {
        anyhow::anyhow!(
            "不支持的 TTS 引擎: {}，可选: {}",
            engine,
            TTS_ENGINE_HINT
        )
    })?;

    // 读取输入：TXT/MD（编码自动检测）或 EPUB/DOCX/PDF（结构解析）
    let text = load_input_text(input)?;

    // 多角色映射表加载
    let role_voice_map = match &role_map {
        Some(path) => {
            let json = std::fs::read_to_string(path)
                .map_err(|e| anyhow::anyhow!("读取角色映射表失败 {:?}: {}", path, e))?;
            Some(votex_domain::tts::role::RoleVoiceMap::from_json_str(&json)
                .map_err(|e| anyhow::anyhow!("角色映射表 JSON 解析失败: {}", e))?)
        }
        None => None,
    };
    let opts = SynthesisOptions {
        role_map: role_voice_map,
        loudnorm,
        output_tempo: atempo,
    };

    // 根据输出文件扩展名判断格式
    let format = detect_format(output);

    // 模型变体映射
    let model_override = model.as_deref().map(|m| match m {
        "0.6b" => "qwen3-tts-0.6b",
        "1.7b" => "qwen3-tts-1.7b",
        _ => m,
    });

    let output_path = std::path::Path::new(output);
    let use_case = TtsUseCase::new();

    println!("TTS 合成开始");
    println!("  引擎: {}", engine);
    println!("  音色: {}", voice);
    println!("  语速: {}", speed);
    println!("  格式: {}", format.extension());
    if let Some(ref lang_val) = lang {
        println!("  语言: {}", lang_val);
    }
    if let Some(ref m) = model_override {
        println!("  模型: {}", m);
    }
    if let Some(ref dir) = session_dir {
        println!("  断点续转: {}", dir);
    }
    println!("  输出: {}", output);

    // CLI 取消：Ctrl+C 由上层注册（此处先支持令牌占位）
    let cancel = Arc::new(AtomicBool::new(false));

    // 进度上报到终端（\r 单行刷新）
    let cancel_for_cb = Arc::clone(&cancel);
    let on_progress: votex_app::use_case::tts_use_case::TtsProgressCallback =
        Some(Box::new(move |done, total, msg| {
            if cancel_for_cb.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            if total > 0 {
                print!("\r  进度: {}/{} 段 - {}        ", done, total, msg);
                use std::io::Write;
                let _ = std::io::stdout().flush();
            } else {
                println!("  {}", msg);
            }
        }));

    use_case.synthesize_ext(
        &text,
        output_path,
        engine_kind,
        voice,
        speed,
        format,
        lang.as_deref(),
        on_progress,
        model_override,
        Some(Arc::clone(&cancel)),
        session_dir.as_deref().map(std::path::Path::new),
        &opts,
    )?;

    println!("\nTTS 合成完成: {}", output);
    Ok(())
}

/// 读取输入文本：存在的路径走文档提取（TXT/MD/EPUB/DOCX/PDF），否则按内联文本
fn load_input_text(input: &str) -> Result<String> {
    if std::path::Path::new(input).exists() {
        votex_app::platform::text::extract_document_text(std::path::Path::new(input))
    } else {
        Ok(input.to_string())
    }
}

/// 根据输出扩展名检测音频格式
pub fn detect_format(output: &str) -> AudioFormat {
    let lower = output.to_ascii_lowercase();
    if lower.ends_with(".mp3") {
        AudioFormat::Mp3
    } else if lower.ends_with(".m4a") {
        AudioFormat::M4A
    } else if lower.ends_with(".m4b") {
        AudioFormat::M4B
    } else if lower.ends_with(".flac") {
        AudioFormat::Flac
    } else {
        AudioFormat::Wav
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use votex_domain::model::value_object::EngineKind;

    #[test]
    fn tts_engine_解析() {
        // docs/20 F62：本命令不再内联引擎串映射，统一委托 domain 层
        // `votex_domain::tts::value_object::parse_tts_engine`（单一权威，返回 Option）。
        // `cosyvoice3` 是 F62 的回归点：此前 CLI 缺该分支会报「不支持的 TTS 引擎」。
        assert_eq!(parse_tts_engine("kokoro"), Some(EngineKind::Kokoro));
        // 迁移别名：indextts2 归一到 IndexTTS25
        assert_eq!(parse_tts_engine("indextts2"), Some(EngineKind::IndexTTS25));
        assert_eq!(parse_tts_engine("qwen3"), Some(EngineKind::Qwen3Tts));
        assert_eq!(parse_tts_engine("cosyvoice3"), Some(EngineKind::CosyVoice3));
        // 未知引擎必须返回 None 而非静默回落到默认引擎（避免用错引擎产出错误音频）
        assert_eq!(parse_tts_engine("unknown"), None);
        // 不接受 `qwen` 简写：它会被 FromStr 误解析成 QwenLlm 而静默用错引擎
        assert_eq!(parse_tts_engine("qwen"), None);
    }

    #[test]
    fn tts_format_检测() {
        assert_eq!(detect_format("output.mp3"), AudioFormat::Mp3);
        assert_eq!(detect_format("output.wav"), AudioFormat::Wav);
        assert_eq!(detect_format("output.WAV"), AudioFormat::Wav); // 不区分大小写
        assert_eq!(detect_format("audiobook.m4b"), AudioFormat::M4B);
        assert_eq!(detect_format("audiobook.M4B"), AudioFormat::M4B);
        assert_eq!(detect_format("song.m4a"), AudioFormat::M4A);
        assert_eq!(detect_format("lossless.flac"), AudioFormat::Flac);
        assert_eq!(detect_format("output"), AudioFormat::Wav); // 无扩展名默认 Wav
    }
}
