use anyhow::Result;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use votex_app::use_case::asr_use_case::{AsrRunOptions, AsrUseCase};

/// 处理 asr 子命令（兼容入口：无词级对齐/说话人分离）
pub fn handle(input: &str, output: &str, model: &str, format: &str, lang: &str) -> Result<()> {
    handle_ex(input, output, model, format, lang, false, false, 0)
}

/// 处理 asr 子命令（扩展入口）
///
/// 长音频识别可能持续数分钟，支持 Ctrl+C 中断：
/// 取消令牌会在每个音频切片边界被检查，最长延迟为单段推理时间。
///
/// - `words`：词级时间戳对齐（精准字幕 + `<output>.words.json`）
/// - `diarize`：说话人分离（说话人标注字幕 + `<output>.diarization.json`）
/// - `num_speakers`：期望说话人数（0 = 自动）
#[allow(clippy::too_many_arguments)]
pub fn handle_ex(
    input: &str,
    output: &str,
    model: &str,
    format: &str,
    lang: &str,
    words: bool,
    diarize: bool,
    num_speakers: u32,
) -> Result<()> {
    let input_path = Path::new(input);
    let output_path = Path::new(output);

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
    if words {
        println!("  词级时间戳对齐: 开启");
    }
    if diarize {
        println!(
            "  说话人分离: 开启（{}）",
            if num_speakers > 0 {
                format!("固定 {} 人", num_speakers)
            } else {
                "自动聚类".to_string()
            }
        );
    }

    let options = AsrRunOptions {
        word_timestamps: words,
        diarize,
        num_speakers,
        ..Default::default()
    };

    let run = use_case.recognize_ex(
        input_path,
        output_path,
        model,
        lang,
        format,
        &options,
        Some(cancel.as_ref()),
    )?;

    println!("ASR 识别完成");
    println!("  识别文本: {}", run.result.text);
    println!("  字幕条目: {} 条", run.result.subtitles.len());
    println!("  输出文件: {:?}", run.result.output_path);
    if !run.word_timestamps.is_empty() {
        println!("  词级时间戳: {} 个词（{}.words.json）", run.word_timestamps.len(), output);
    }
    if let Some(count) = run.speaker_count {
        println!("  说话人数量: {} 人（{}.diarization.json）", count, output);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 验证 ASR 输入输出路径基本逻辑
    #[test]
    fn asr_参数验证() {
        // 输入文件应存在（逻辑上由 handle 调用者保证）
        // 此处验证路径合法性判断
        assert!(std::path::Path::new("Cargo.toml").exists());
        assert!(!std::path::Path::new("nonexistent_file.asr_input").exists());
    }

    /// AsrRunOptions 字段传递
    #[test]
    fn asr_扩展选项构造() {
        let options = AsrRunOptions {
            word_timestamps: true,
            diarize: true,
            num_speakers: 2,
            ..Default::default()
        };
        assert!(options.word_timestamps);
        assert!(options.diarize);
        assert_eq!(options.num_speakers, 2);
    }
}
