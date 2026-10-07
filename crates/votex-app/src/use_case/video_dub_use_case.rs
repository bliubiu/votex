//! 视频配音用例（借鉴 VoiceStudio 配音链的功能规格，代码自主实现）
//!
//! 链路：视频 → 提取音轨 → ASR 转写 →（可选）翻译 → 逐段 TTS →
//! **时长拟合**（起点对齐原时间轴 + atempo 变速）→ 拼装配音轨 →
//! 用**实测段时长**重建字幕时间轴 →（可选）二次 ASR 质检 → 音轨替换/混入视频。
//!
//! 与现有 `PipelineUseCase` 的 StageFn 注册表（文件粒度）不同，配音是
//! 段粒度操作（每段独立时间轴），故实现为独立用例，阶段进度经
//! `ProgressEvent` 上报，取消令牌在段边界生效。
//!
//! 时长拟合策略（第一版）：段起点对齐原时间轴，段音频超长时按
//! `max_tempo` 上限变速压缩；段音频短于原段时自然留白。视频减速的
//! 50/50 分摊策略见 docs/27 §5.1，暂未实现。

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use votex_domain::asr::value_object::{SubtitleEntry, SubtitleFormat, Timestamp};
use votex_domain::tts::value_object::parse_tts_engine;

/// 配音进度回调：(当前段, 总段数或 0, 阶段描述)
pub type DubProgressCallback = Option<Box<dyn Fn(usize, usize, &str) + Send + Sync>>;

/// 视频配音请求
#[derive(Debug, Clone)]
pub struct VideoDubRequest {
    /// 源视频路径（mp4/mkv/mov 等 ffmpeg 可读格式）
    pub video_path: String,
    /// ASR 引擎（识别原声，如 "paraformer"）
    pub asr_engine: String,
    /// ASR 识别语言
    pub language: String,
    /// 配音 TTS 引擎
    pub tts_engine: String,
    /// 配音音色
    pub voice: String,
    /// 配音语速（引擎级，区别于时长拟合的 atempo）
    pub speed: f32,
    /// 翻译方向（如 "zh-en"）；None/空 = 不翻译，用原声文本配音
    pub translate_direction: Option<String>,
    /// 翻译引擎（仅 translate_direction 存在时使用，默认 "dict"）
    pub translate_engine: String,
    /// 单段音频最大加速倍率（时长拟合上限，1.0 ~ 4.0，默认 1.5）
    pub max_tempo: f32,
    /// 二次 ASR 质检：对配音结果再识别，验证合成可懂度
    pub verify: bool,
    /// 保留背景音：原音轨按 0.25 音量与配音混音；false = 替换原音轨
    pub keep_background: bool,
}

impl Default for VideoDubRequest {
    fn default() -> Self {
        Self {
            video_path: String::new(),
            asr_engine: "paraformer".to_string(),
            language: "zh".to_string(),
            tts_engine: "kokoro".to_string(),
            voice: "zf_001".to_string(),
            speed: 1.0,
            translate_direction: None,
            translate_engine: "dict".to_string(),
            max_tempo: 1.5,
            verify: true,
            keep_background: false,
        }
    }
}

/// 视频配音结果
#[derive(Debug, Clone)]
pub struct VideoDubResponse {
    /// 配音后视频路径
    pub output_video: String,
    /// 拼装后的配音音轨（WAV）
    pub dubbed_audio: String,
    /// 依据实测段时长重建的字幕文件
    pub subtitle_path: String,
    /// 二次质检字幕（对配音结果再识别的时间轴）
    pub verify_subtitle_path: Option<String>,
    /// 二次质检得分（0.0 ~ 1.0，配音文本与再识别文本的字符二元组相似度）
    pub verify_score: Option<f32>,
    /// 配音段数
    pub segments: usize,
}

/// 视频配音用例
pub struct VideoDubUseCase;

impl VideoDubUseCase {
    pub fn new() -> Self {
        Self
    }

    /// 执行视频配音
    ///
    /// `output_dir`：产物目录（配音视频/音轨/字幕均落在此处）
    pub fn execute(
        &self,
        req: &VideoDubRequest,
        output_dir: &Path,
        progress: DubProgressCallback,
        cancel: Option<std::sync::Arc<AtomicBool>>,
    ) -> Result<VideoDubResponse> {
        let report = |done: usize, total: usize, msg: &str| {
            if let Some(ref cb) = progress {
                cb(done, total, msg);
            }
        };
        let check_cancel = |stage: &str| -> Result<()> {
            if let Some(ref token) = cancel {
                if token.load(Ordering::SeqCst) {
                    anyhow::bail!("任务已取消（{}）", stage);
                }
            }
            Ok(())
        };

        let video = PathBuf::from(&req.video_path);
        if !video.is_file() {
            anyhow::bail!("视频文件不存在: {:?}", video);
        }
        if !(1.0..=4.0).contains(&req.max_tempo) {
            anyhow::bail!("max_tempo 应在 1.0 ~ 4.0 之间，当前 {}", req.max_tempo);
        }
        std::fs::create_dir_all(output_dir).context("创建输出目录失败")?;
        let base_name = video
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("dub")
            .to_string();

        // TTS 引擎解析（提前失败，避免 ASR 跑完才发现引擎名写错）
        let tts_engine = parse_tts_engine(&req.tts_engine)
            .ok_or_else(|| anyhow::anyhow!("不支持的 TTS 引擎: {}", req.tts_engine))?;

        // 阶段一：提取音轨（归一 24kHz 单声道，供 ASR 与总时长测量）
        report(0, 0, "正在提取视频音轨...");
        check_cancel("提取音轨前")?;
        let temp_dir = std::env::temp_dir().join(format!(
            "votex_dub_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&temp_dir).context("创建配音临时目录失败")?;
        let run = || -> Result<VideoDubResponse> {
            let source_wav = temp_dir.join("source.wav");
            votex_infra::video::dub_ops::extract_audio_to_wav(&video, &source_wav)
                .context("从视频提取音轨失败")?;
            let source_audio = votex_infra::audio::wav::read_wav(&source_wav)?;
            let total_ms = source_audio.duration_ms() as u64;
            drop(source_audio);
            if total_ms < 500 {
                anyhow::bail!("视频音轨时长过短（{}ms），无需配音", total_ms);
            }
            tracing::info!("配音: 音轨提取完成，总时长 {}ms", total_ms);

            // 阶段二：ASR 转写原声
            report(0, 0, "正在识别原声（ASR）...");
            check_cancel("识别原声前")?;
            let asr = crate::services::shared_cases::shared_asr();
            let asr_result = asr.recognize(
                &source_wav,
                &output_dir.join(format!("{}.original.srt", base_name)),
                &req.asr_engine,
                &req.language,
                "srt",
                cancel.as_deref(),
            )?;
            if asr_result.text.trim().is_empty() {
                anyhow::bail!("未从视频音轨中识别到任何语音，无法配音");
            }
            tracing::info!("配音: 原声识别完成，{} 字", asr_result.text.chars().count());

            // 阶段三：分句 + 按字数比例分配原时间轴（段级对齐）
            let sentences = split_sentences(&asr_result.text);
            if sentences.is_empty() {
                anyhow::bail!("原声文本分句后为空");
            }
            let timeline = plan_segment_timeline(&sentences, total_ms);

            // 阶段四（可选）：翻译
            let dub_texts: Vec<String> = match req.translate_direction.as_deref().filter(|d| !d.trim().is_empty()) {
                Some(direction) => {
                    report(0, sentences.len(), "正在翻译（逐段）...");
                    check_cancel("翻译前")?;
                    let translator = crate::use_case::translation_use_case::TranslationUseCase::new(
                        &req.translate_engine,
                    )
                    .context("创建翻译引擎失败")?;
                    let translated = translator.translate_batch(&sentences, direction)?;
                    tracing::info!("配音: 翻译完成 {} 段", translated.len());
                    translated
                }
                None => sentences.clone(),
            };

            // 阶段五：逐段 TTS 合成
            let seg_dir = temp_dir.join("segments");
            let tts = crate::services::shared_cases::shared_tts();
            let total = dub_texts.len();
            let mut seg_paths: Vec<Option<PathBuf>> = Vec::with_capacity(total);
            for (i, text) in dub_texts.iter().enumerate() {
                check_cancel(&format!("合成第 {}/{} 段时", i + 1, total))?;
                report(i, total, &format!("正在合成配音 {}/{} 段...", i + 1, total));
                if text.trim().is_empty() {
                    // 翻译可能产生空段（如纯标点原文），留空处理
                    seg_paths.push(None);
                    continue;
                }
                let seg_path = seg_dir.join(format!("seg_{:04}.wav", i));
                tts.synthesize(
                    text,
                    &seg_path,
                    tts_engine,
                    &req.voice,
                    req.speed,
                    votex_domain::tts::value_object::AudioFormat::Wav,
                    None,
                    None,
                    None,
                )
                .with_context(|| format!("第 {} 段配音合成失败: {}", i + 1, text))?;
                seg_paths.push(Some(seg_path));
            }

            // 阶段六：时长拟合 + 按原时间轴拼装配音轨
            report(total, total, "正在时长拟合与拼装...");
            check_cancel("拼装前")?;
            let dubbed_path = output_dir.join(format!("{}.dub.wav", base_name));
            let mut fitted: Vec<FittedSegment> = Vec::with_capacity(total);
            {
                // 首段确定采样率/声道；段间不一致直接报错（同引擎应恒定）
                let mut sample_rate: Option<u32> = None;
                let mut channels: Option<u16> = None;
                let mut writer_ready = false;
                let mut writer = None;
                let mut cursor_ms: u64 = 0;

                for (i, seg_path) in seg_paths.iter().enumerate() {
                    let Some(seg_path) = seg_path else { continue };
                    let seg_audio = votex_infra::audio::wav::read_wav(seg_path)?;
                    let seg_ms = seg_audio.duration_ms() as u64;
                    if seg_audio.samples.is_empty() {
                        continue;
                    }
                    match (sample_rate, channels) {
                        (Some(sr), Some(ch)) => {
                            if sr != seg_audio.sample_rate || ch != seg_audio.channels {
                                anyhow::bail!(
                                    "第 {} 段音频参数不一致（{}Hz/{}ch，期望 {}Hz/{}ch）",
                                    i + 1,
                                    seg_audio.sample_rate,
                                    seg_audio.channels,
                                    sr,
                                    ch
                                );
                            }
                        }
                        _ => {
                            sample_rate = Some(seg_audio.sample_rate);
                            channels = Some(seg_audio.channels);
                        }
                    }

                    let (start_ms, tempo) = fit_segment(
                        cursor_ms,
                        timeline[i].0,
                        seg_ms,
                        timeline[i].1,
                        req.max_tempo,
                    );

                    // 变速（需要时经 ffmpeg atempo，产物落到临时目录）
                    let (final_path, final_ms) = if let Some(t) = tempo {
                        let fitted_path = temp_dir.join(format!("fit_{:04}.wav", i));
                        votex_infra::video::dub_ops::apply_tempo(seg_path, &fitted_path, t)
                            .with_context(|| format!("第 {} 段变速失败（{}x）", i + 1, t))?;
                        let fitted_audio = votex_infra::audio::wav::read_wav(&fitted_path)?;
                        let ms = fitted_audio.duration_ms() as u64;
                        drop(fitted_audio);
                        (fitted_path, ms)
                    } else {
                        (seg_path.clone(), seg_ms)
                    };

                    if !writer_ready {
                        writer = Some(votex_infra::audio::wav::StreamingWav::create(
                            &dubbed_path,
                            sample_rate.unwrap(),
                            channels.unwrap(),
                        )?);
                        writer_ready = true;
                    }
                    let w = writer.as_mut().expect("writer 已创建");
                    if start_ms > cursor_ms {
                        w.append_silence_ms((start_ms - cursor_ms).min(u32::MAX as u64) as u32)?;
                    }
                    let fitted_audio = votex_infra::audio::wav::read_wav(&final_path)?;
                    w.append_f32(&fitted_audio.samples)?;
                    cursor_ms = start_ms + final_ms;
                    fitted.push(FittedSegment {
                        index: i,
                        text: dub_texts[i].clone(),
                        start_ms,
                        end_ms: cursor_ms,
                    });
                }
                if let Some(w) = writer {
                    w.finish()?;
                }
                if fitted.is_empty() {
                    anyhow::bail!("配音段全部为空，无法拼装音轨");
                }
            }
            tracing::info!("配音: 音轨拼装完成 → {:?}", dubbed_path);

            // 阶段七：用实测段时长重建字幕时间轴（非字数估算）
            let subtitle_format = SubtitleFormat::Srt;
            let entries: Vec<SubtitleEntry> = fitted
                .iter()
                .enumerate()
                .map(|(i, s)| SubtitleEntry {
                    index: i + 1,
                    start_time: Timestamp::from_millis(s.start_ms),
                    end_time: Timestamp::from_millis(s.end_ms),
                    text: s.text.clone(),
                    speaker: None,
                })
                .collect();
            let subtitle_path = output_dir.join(format!("{}.dub.srt", base_name));
            votex_infra::subtitle::SubtitleWriter::write(&entries, subtitle_format, &subtitle_path)?;

            // 阶段八（可选）：二次 ASR 质检
            let mut verify_subtitle_path = None;
            let mut verify_score = None;
            if req.verify {
                report(0, 0, "正在二次 ASR 质检...");
                check_cancel("质检前")?;
                let verify_srt = output_dir.join(format!("{}.dub.verify.srt", base_name));
                let verify = asr.recognize(
                    &dubbed_path,
                    &verify_srt,
                    &req.asr_engine,
                    &req.language,
                    "srt",
                    cancel.as_deref(),
                )?;
                let expected: String = fitted.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join("");
                let score = char_bigram_similarity(&expected, &verify.text);
                // 质检报告：可懂度得分 + 再识别全文（与配音文本对照）
                let report_path = output_dir.join(format!("{}.dub.verify.txt", base_name));
                std::fs::write(
                    &report_path,
                    format!(
                        "配音可懂度得分: {:.3}\n（配音文本与对配音结果二次识别文本的字符二元组相似度，>0.7 一般可接受）\n\n\
                         ---- 配音文本 ----\n{}\n\n---- 二次识别文本 ----\n{}\n",
                        score, expected, verify.text
                    ),
                )?;
                verify_subtitle_path = Some(verify_srt.display().to_string());
                verify_score = Some(score);
                tracing::info!("配音: 质检完成，可懂度得分 {:.3}", score);
            }

            // 阶段九：音轨替换/混入视频（视频流直拷，不重编码）
            report(0, 0, "正在合成配音视频...");
            check_cancel("合成视频前")?;
            let out_video = output_dir.join(format!("{}.dubbed.mp4", base_name));
            votex_infra::video::dub_ops::mux_audio_into_video(
                &video,
                &dubbed_path,
                req.keep_background.then_some(0.25),
                &out_video,
            )?;

            Ok(VideoDubResponse {
                output_video: out_video.display().to_string(),
                dubbed_audio: dubbed_path.display().to_string(),
                subtitle_path: subtitle_path.display().to_string(),
                verify_subtitle_path,
                verify_score,
                segments: fitted.len(),
            })
        };

        let result = run();
        // 临时目录清理（成败都清；失败时保留诊断价值有限，跳过复杂保留策略）
        let _ = std::fs::remove_dir_all(&temp_dir);
        result
    }
}

impl Default for VideoDubUseCase {
    fn default() -> Self {
        Self::new()
    }
}

/// 拟合后的段（实测时间轴）
struct FittedSegment {
    #[allow(dead_code)] // 保留段序号便于调试与后续增量重配
    index: usize,
    text: String,
    start_ms: u64,
    end_ms: u64,
}

/// 中文分句：按句末标点切分，超长句再按逗号/顿号二级切分
///
/// 纯函数，供时间轴规划与测试使用。空句/纯标点句被过滤。
pub fn split_sentences(text: &str) -> Vec<String> {
    const MAX_LEN: usize = 60;
    let mut out: Vec<String> = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if matches!(ch, '。' | '！' | '？' | '!' | '?' | '…' | ';' | '；' | '\n') {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.trim().is_empty() {
        out.push(current);
    }

    // 二级切分：长句按逗号/顿号/冒号再断（TTS 单段不宜过长，时间轴粒度也更细）
    let mut refined: Vec<String> = Vec::new();
    for s in out {
        let s = s.trim().to_string();
        if s.is_empty() {
            continue;
        }
        if s.chars().count() <= MAX_LEN {
            refined.push(s);
            continue;
        }
        let mut part = String::new();
        for ch in s.chars() {
            part.push(ch);
            let len = part.chars().count();
            // 到达硬上限强制切段（无标点的长句也要断，避免单段过长）
            if len >= MAX_LEN
                || (matches!(ch, '，' | ',' | '、' | '：' | ':') && len >= MAX_LEN / 2)
            {
                refined.push(part.trim().to_string());
                part.clear();
            }
        }
        if !part.trim().is_empty() {
            refined.push(part.trim().to_string());
        }
    }
    refined.into_iter().filter(|s| !s.is_empty()).collect()
}

/// 段时间轴规划：按字数比例把总时长分配到各句
///
/// 与 infra `FastSubtitleGenerator` 同思路；返回 (start_ms, end_ms)。
/// 纯函数。末段结束时间对齐 `total_ms`（比例取整误差全部归末段）。
pub fn plan_segment_timeline(sentences: &[String], total_ms: u64) -> Vec<(u64, u64)> {
    if sentences.is_empty() {
        return Vec::new();
    }
    let char_counts: Vec<u64> = sentences
        .iter()
        .map(|s| s.chars().count().max(1) as u64)
        .collect();
    let total_chars: u64 = char_counts.iter().sum();
    let mut timeline = Vec::with_capacity(sentences.len());
    let mut cursor = 0u64;
    for (i, count) in char_counts.iter().enumerate() {
        if i + 1 == sentences.len() {
            timeline.push((cursor, total_ms.max(cursor)));
        } else {
            let dur = total_ms * count / total_chars;
            timeline.push((cursor, cursor + dur));
            cursor += dur;
        }
    }
    timeline
}

/// 单段时长拟合决策（纯函数）
///
/// - 段起点：目标槽起点，但若前段溢出已越过它，则紧跟前段（避免重叠）
/// - 段音频短于等于剩余槽空间：不变速，自然留白
/// - 段音频超长：tempo = seg/剩余槽空间，clamp 到 `max_tempo`
/// - 起点已完全越过目标槽（前段溢出挤压）：不再变速追赶（越追越失真），
///   按自然语速顺延，溢出交给后续段
///
/// 返回 (实际起点, 变速倍率 Option)
pub fn fit_segment(
    cursor_ms: u64,
    slot_start_ms: u64,
    seg_ms: u64,
    slot_end_ms: u64,
    max_tempo: f32,
) -> (u64, Option<f32>) {
    let start = slot_start_ms.max(cursor_ms);
    // 已溢出目标槽：变速追赶只会加剧失真且追不上，按自然语速顺延
    if start >= slot_end_ms {
        return (start, None);
    }
    let available = slot_end_ms - start;
    if seg_ms <= available {
        (start, None)
    } else {
        let tempo = (seg_ms as f32 / available as f32).min(max_tempo.max(1.0));
        (start, Some(tempo))
    }
}

/// 字符二元组相似度（Dice 系数），用于二次质检的可懂度打分
///
/// 先剥离空白与常见标点再比较；纯函数。
pub fn char_bigram_similarity(a: &str, b: &str) -> f32 {
    fn normalize(s: &str) -> String {
        s.chars()
            .filter(|c| !c.is_whitespace() && !matches!(c, '，' | '。' | '！' | '？' | '、' | '；' | '：' | ',' | '.' | '!' | '?' | ';' | ':' | '"' | '“' | '”' | '…' | '（' | '）' | '(' | ')'))
            .collect()
    }
    fn bigrams(s: &str) -> std::collections::HashMap<(char, char), usize> {
        let chars: Vec<char> = s.chars().collect();
        let mut map = std::collections::HashMap::new();
        for w in chars.windows(2) {
            *map.entry((w[0], w[1])).or_insert(0usize) += 1;
        }
        map
    }
    let (na, nb) = (normalize(a), normalize(b));
    if na.chars().count() < 2 || nb.chars().count() < 2 {
        return if na == nb { 1.0 } else { 0.0 };
    }
    let (ba, bb) = (bigrams(&na), bigrams(&nb));
    let inter: usize = ba
        .iter()
        .map(|(k, v)| (*v).min(*bb.get(k).unwrap_or(&0)))
        .sum();
    let total = ba.values().sum::<usize>() + bb.values().sum::<usize>();
    if total == 0 {
        return 0.0;
    }
    2.0 * inter as f32 / total as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 分句_句末标点与长句二级切分() {
        let text = "你好，世界！今天天气不错。我们一起去公园散步，然后吃午饭，好吗？";
        let sentences = split_sentences(text);
        assert!(sentences.len() >= 3, "应按句末标点切分: {:?}", sentences);
        assert!(sentences[0].starts_with("你好"));
        assert!(sentences.iter().all(|s| !s.is_empty()));
    }

    #[test]
    fn 分句_超长句按逗号断开() {
        let long = format!("{}，{}。", "非常长的一句话内容".repeat(10), "结束");
        let sentences = split_sentences(&long);
        assert!(
            sentences.iter().all(|s| s.chars().count() <= 60),
            "长句应被二级切分: {:?}",
            sentences.iter().map(|s| s.chars().count()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn 时间轴_按字数比例分配且末段对齐总长() {
        let sentences = vec!["一二三".to_string(), "一二三四五六七".to_string()];
        let tl = plan_segment_timeline(&sentences, 10_000);
        // 3 字 : 7 字 → 3000ms : 7000ms
        assert_eq!(tl[0], (0, 3000));
        assert_eq!(tl[1], (3000, 10_000));
    }

    #[test]
    fn 时间轴_空输入() {
        assert!(plan_segment_timeline(&[], 1000).is_empty());
    }

    #[test]
    fn 拟合_短段不变速且起点对齐() {
        let (start, tempo) = fit_segment(0, 2000, 1500, 5000, 1.5);
        assert_eq!(start, 2000);
        assert_eq!(tempo, None);
    }

    #[test]
    fn 拟合_超长段变速且受上限约束() {
        // 段 6000ms，槽 4000ms → 1.5x；上限 1.2 → clamp
        let (_, tempo) = fit_segment(0, 0, 6000, 4000, 1.2);
        assert_eq!(tempo, Some(1.2));
        let (_, tempo) = fit_segment(0, 0, 6000, 4000, 1.5);
        assert_eq!(tempo, Some(1.5));
    }

    #[test]
    fn 拟合_前段溢出时起点顺延() {
        // cursor 已到 5000，目标槽起点 3000 → 实际起点 5000
        let (start, tempo) = fit_segment(5000, 3000, 500, 4000, 1.5);
        assert_eq!(start, 5000);
        assert_eq!(tempo, None);
    }

    #[test]
    fn 相似度_一致与无关() {
        let s = char_bigram_similarity("今天天气不错", "今天天气不错");
        assert!(s > 0.99, "完全一致应接近 1.0，实际 {s}");
        let d = char_bigram_similarity("今天天气不错", " completely unrelated");
        assert!(d < 0.1, "无关文本应接近 0，实际 {d}");
        // 标点/空白不影响
        let p = char_bigram_similarity("你好，世界。", "你好 世界");
        assert!(p > 0.99, "标点应被剥离，实际 {p}");
    }
}
