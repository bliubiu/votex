use anyhow::Result;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{
    AsrParams, AsrResult, DenoiseLevel, Language, SliceLength, SubtitleEntry, SubtitleFormat,
    WordTimestamp,
};
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{EngineKind, ModelId, ModelKind};
use votex_domain::shared::value_object::AudioData;
use votex_infra::asr::sensevoice::SenseVoiceProvider;
use votex_infra::asr::whisper::WhisperProvider;
use votex_infra::asr::paraformer::ParaformerProvider;
use votex_infra::asr::qwen3_asr::Qwen3AsrProvider;
use votex_infra::asr::firered_asr::FireRedAsrCtcProvider;
use votex_infra::asr::wenet::WeNetProvider;
use votex_infra::asr::speaker_diarization::{DiarizationSegment, DiarizeOptions, SpeakerDiarizationProvider};
use votex_infra::subtitle::SubtitleWriter;

/// ASR 扩展运行选项
///
/// `recognize`（兼容入口）等价于全部默认值；
/// 词级时间戳对齐与说话人分离通过 `recognize_ex` 显式开启。
#[derive(Debug, Clone, PartialEq)]
pub struct AsrRunOptions {
    /// 词级时间戳对齐：引擎返回词级时间戳时按词组句生成精准字幕，
    /// 并导出 `<output>.words.json`（绝对时间轴词列表）
    pub word_timestamps: bool,
    /// 说话人分离：识别后对全音频做分离，为每条字幕标注说话人，
    /// 并导出 `<output>.diarization.json`（说话人时间轴片段）
    pub diarize: bool,
    /// 期望说话人数（0 = 由聚类阈值自动决定）
    pub num_speakers: u32,
    /// 说话人聚类相似度阈值（num_speakers=0 时生效）
    pub diarize_threshold: f32,
}

impl Default for AsrRunOptions {
    fn default() -> Self {
        Self {
            word_timestamps: false,
            diarize: false,
            num_speakers: 0,
            diarize_threshold: 0.5,
        }
    }
}

/// ASR 扩展运行结果
#[derive(Debug, Clone)]
pub struct AsrRunResult {
    /// 与旧接口一致的基础结果（文本 / 字幕 / 输出路径）
    pub result: AsrResult,
    /// 绝对时间轴词级时间戳（未启用对齐或引擎不支持时为空）
    pub word_timestamps: Vec<WordTimestamp>,
    /// 说话人数量（未启用分离时为 None）
    pub speaker_count: Option<usize>,
}

/// 单切片识别记录（偏移推进 + 文本 + 词级时间戳）
struct SliceOutput {
    offset_ms: u64,
    duration_ms: u64,
    text: String,
    words: Vec<WordTimestamp>,
}

/// ASR 用例——语音识别
pub struct AsrUseCase {
    whisper: WhisperProvider,
    sensevoice: SenseVoiceProvider,
    paraformer: ParaformerProvider,
    qwen3_asr: Qwen3AsrProvider,
    firered_asr: FireRedAsrCtcProvider,
    wenet: WeNetProvider,
    speaker_diarization: SpeakerDiarizationProvider,
}

impl AsrUseCase {
    pub fn new() -> Self {
        Self {
            whisper: WhisperProvider::new(),
            sensevoice: SenseVoiceProvider::new(),
            paraformer: ParaformerProvider::new(),
            qwen3_asr: Qwen3AsrProvider::new(),
            firered_asr: FireRedAsrCtcProvider::new(),
            wenet: WeNetProvider::new(),
            speaker_diarization: SpeakerDiarizationProvider::new(),
        }
    }

    /// 列出各本地 ASR 引擎的能力描述（不加载模型）
    ///
    /// 供 `votex model list --json` 与 `votex serve` 的程序化发现复用。
    pub fn capabilities(&self) -> Vec<votex_domain::model::capability::EngineCapability> {
        vec![
            self.whisper.capability(),
            self.sensevoice.capability(),
            self.paraformer.capability(),
            self.qwen3_asr.capability(),
            self.firered_asr.capability(),
            self.wenet.capability(),
        ]
    }

    /// 根据引擎类型获取对应 Provider 的不可变引用
    fn get_provider(&self, engine: EngineKind) -> Result<&dyn AsrProvider> {
        match engine {
            EngineKind::Whisper => Ok(&self.whisper),
            EngineKind::SenseVoice => Ok(&self.sensevoice),
            EngineKind::Paraformer => Ok(&self.paraformer),
            EngineKind::Qwen3Asr => Ok(&self.qwen3_asr),
            EngineKind::FireRedAsr => Ok(&self.firered_asr),
            EngineKind::WeNet => Ok(&self.wenet),
            _ => anyhow::bail!("不支持的 ASR 引擎: {:?}", engine),
        }
    }

    /// 执行 ASR 识别（兼容入口，无词级对齐/说话人分离）
    ///
    /// `cancel` — 可选取消令牌。长音频会被切成多段逐段识别，
    /// 每个切片边界都会检查该标志；置 true 即中断并返回错误。
    /// 传 `None` 表示不可取消（CLI 一次性调用场景）。
    pub fn recognize(
        &self,
        input_path: &Path,
        output_path: &Path,
        model_id: &str,
        language: &str,
        format: &str,
        cancel: Option<&AtomicBool>,
    ) -> Result<AsrResult> {
        self.recognize_ex(
            input_path,
            output_path,
            model_id,
            language,
            format,
            &AsrRunOptions::default(),
            cancel,
        )
        .map(|r| r.result)
    }

    /// 执行 ASR 识别（扩展入口）
    ///
    /// - `options.word_timestamps`：按词级时间戳组句生成精准字幕，
    ///   并导出 `<output>.words.json`；引擎不支持词级时间戳时自动
    ///   回退切片级字幕（与旧行为一致）。
    /// - `options.diarize`：识别后对全音频执行说话人分离，
    ///   每条字幕标注 `speaker`，并导出 `<output>.diarization.json`。
    pub fn recognize_ex(
        &self,
        input_path: &Path,
        output_path: &Path,
        model_id: &str,
        language: &str,
        format: &str,
        options: &AsrRunOptions,
        cancel: Option<&AtomicBool>,
    ) -> Result<AsrRunResult> {
        if !input_path.exists() {
            anyhow::bail!("输入文件不存在: {:?}", input_path);
        }

        // 解析模型和参数
        let model = Self::parse_model(model_id)?;
        let engine = model.engine;
        let lang = Self::parse_language(language);
        let subtitle_format = Self::parse_format(format);

        let params = AsrParams {
            model: model.id.clone(),
            language: lang,
            auto_punctuation: true,
            auto_slice: true,
            slice_length: SliceLength::S30,
            denoise: false,
            denoise_level: DenoiseLevel::Low,
            output_format: subtitle_format,
        };

        // 加载模型
        self.ensure_loaded(&model)?;

        // 读取音频
        let audio = Self::load_audio(input_path)?;

        // 音频分段
        let slices = Self::slice_audio(&audio, params.slice_length.seconds());
        tracing::info!("音频分段完成: {} 段", slices.len());

        let provider = self.get_provider(engine)?;
        let mut slice_outputs: Vec<SliceOutput> = Vec::new();

        for (i, slice) in slices.iter().enumerate() {
            // 切片边界检查取消：长音频识别可能持续数分钟，
            // 没有这个检查用户点「取消」后仍要等全程结束。
            if let Some(flag) = cancel {
                if flag.load(Ordering::SeqCst) {
                    anyhow::bail!("任务已取消（已完成 {}/{} 段）", i, slices.len());
                }
            }
            tracing::info!("识别第 {}/{} 段", i + 1, slices.len());
            let output = provider.recognize(slice, &params)?;

            // 时间轴推进必须无条件执行：静音/未检出文字的切片不累计偏移，
            // 其后所有字幕时间戳会整体提前一个切片时长（docs/23 A2）
            let duration_ms = slice.duration_ms() as u64;
            let offset_ms = slice_outputs
                .last()
                .map(|s| s.offset_ms + s.duration_ms)
                .unwrap_or(0);
            slice_outputs.push(SliceOutput {
                offset_ms,
                duration_ms,
                text: output.text,
                words: output.word_timestamps,
            });
        }

        // 组装绝对时间轴的词级时间戳（仅保留真实对齐的词）
        let all_words: Vec<WordTimestamp> = slice_outputs
            .iter()
            .flat_map(|s| {
                s.words.iter().filter(|w| is_aligned_word(w)).map(|w| WordTimestamp {
                    word: w.word.clone(),
                    start_ms: w.start_ms + s.offset_ms as f64,
                    end_ms: w.end_ms + s.offset_ms as f64,
                })
            })
            .collect();

        // 字幕条目：词级对齐开启且有真实词时间戳 → 组句精准字幕；否则切片级
        let mut all_entries =
            if options.word_timestamps && !all_words.is_empty() {
                tracing::info!("词级时间戳对齐: {} 个词，组句生成字幕", all_words.len());
                group_words_into_entries(&all_words)
            } else {
                slice_outputs
                    .iter()
                    .filter(|s| !s.text.is_empty())
                    .enumerate()
                    .map(|(i, s)| {
                        SubtitleEntry::new(i + 1, s.offset_ms, s.offset_ms + s.duration_ms, s.text.clone())
                    })
                    .collect::<Vec<_>>()
            };

        // 说话人分离：为每条字幕标注说话人
        let mut speaker_count = None;
        if options.diarize {
            self.ensure_diarization_loaded()?;
            let mono_16k = AudioData {
                samples: audio.to_mono_f32_16k(),
                sample_rate: 16000,
                channels: 1,
            };
            let diar_options = DiarizeOptions {
                num_speakers: options.num_speakers,
                threshold: options.diarize_threshold,
            };
            let segments = self.speaker_diarization.diarize_with(&mono_16k, &diar_options)?;
            let count = segments.iter().map(|s| s.speaker).max().map(|m| m + 1).unwrap_or(0);
            tracing::info!("说话人分离完成: {} 个片段，{} 个说话人", segments.len(), count);

            assign_speakers(&mut all_entries, &segments);
            speaker_count = Some(count);

            // 导出说话人时间轴 JSON
            let diar_path = output_path.with_extension("diarization.json");
            write_diarization_json(&diar_path, &segments, speaker_count)?;
            tracing::info!("说话人时间轴已保存: {:?}", diar_path);
        }

        // 词级时间戳 JSON 导出
        if options.word_timestamps && !all_words.is_empty() {
            let words_path = output_path.with_extension("words.json");
            write_words_json(&words_path, &all_words)?;
            tracing::info!("词级时间戳已保存: {:?}", words_path);
        }

        // 写入字幕文件
        SubtitleWriter::write(&all_entries, subtitle_format, output_path)?;
        tracing::info!("字幕文件已保存: {:?}", output_path);

        let full_text: String = all_entries
            .iter()
            .map(|e| e.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        Ok(AsrRunResult {
            result: AsrResult {
                text: full_text,
                subtitles: all_entries,
                output_path: output_path.to_path_buf(),
            },
            word_timestamps: all_words,
            speaker_count,
        })
    }

    /// 释放指定 ASR 引擎已加载的模型会话（真实释放内存）
    pub fn unload(&self, engine: EngineKind) -> Result<()> {
        match engine {
            EngineKind::SpeakerDiarization => self.speaker_diarization.unload()?,
            other => self.get_provider(other)?.unload()?,
        }
        tracing::info!("ASR 引擎 {:?} 会话已释放", engine);
        Ok(())
    }

    /// 确保说话人分离模型已加载
    fn ensure_diarization_loaded(&self) -> Result<()> {
        if !self.speaker_diarization.is_loaded() {
            let model = Model::new(
                ModelId::new("speaker-diarization"),
                "speaker-diarization",
                ModelKind::Asr,
                EngineKind::SpeakerDiarization,
            );
            self.speaker_diarization.load(&model)?;
        }
        Ok(())
    }

    /// 解析模型 ID 为 (Model, EngineKind)
    fn parse_model(model_id: &str) -> Result<Model> {
        let (id, engine) = match model_id {
            "whisper-base" => (ModelId::new("whisper-base"), EngineKind::Whisper),
            "whisper-small" => (ModelId::new("whisper-small"), EngineKind::Whisper),
            "sensevoice" => (ModelId::new("sensevoice"), EngineKind::SenseVoice),
            "paraformer" => (ModelId::new("paraformer"), EngineKind::Paraformer),
            "qwen3-asr" => (ModelId::new("qwen3-asr"), EngineKind::Qwen3Asr),
            "firered-asr" => (ModelId::new("firered-asr"), EngineKind::FireRedAsr),
            "wenet" => (ModelId::new("wenet"), EngineKind::WeNet),
            _ => anyhow::bail!(
                "不支持的 ASR 模型: {}，可选: whisper-base, whisper-small, sensevoice, paraformer, qwen3-asr, firered-asr, wenet",
                model_id
            ),
        };
        Ok(Model::new(id, model_id, ModelKind::Asr, engine))
    }

    /// 解析语言参数
    fn parse_language(lang: &str) -> Language {
        match lang {
            "zh" => Language::Zh,
            "zhen" => Language::ZhEn,
            "en" => Language::En,
            _ => Language::Zh,
        }
    }

    /// 解析输出格式
    fn parse_format(format: &str) -> SubtitleFormat {
        match format {
            "srt" => SubtitleFormat::Srt,
            "lrc" => SubtitleFormat::Lrc,
            _ => SubtitleFormat::Txt,
        }
    }

    /// ASR 可经 ffmpeg 转码后识别的压缩音频格式
    const FFMPEG_DECODABLE: &[&str] = &[
        "mp3", "m4a", "m4b", "mp4", "flac", "aac", "ogg", "opus", "wma", "webm", "amr", "3gp",
    ];

    /// 加载音频文件
    ///
    /// WAV 直接读取；MP3/M4A/FLAC 等压缩格式经 ffmpeg 转码为 WAV 后读取
    /// （AGENTS.md 声明支持这些格式）；其余格式返回 VA004 错误——
    /// 此前此处静默返回 5 秒静音并继续识别，产出与输入无关的垃圾结果。
    fn load_audio(path: &Path) -> Result<AudioData> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();
        tracing::info!("加载音频文件: {:?} (格式: {})", path, ext);

        let audio = match ext.as_str() {
            "wav" => votex_infra::audio::wav::read_wav(path)?,
            e if Self::FFMPEG_DECODABLE.contains(&e) => Self::load_audio_via_ffmpeg(path)?,
            other => {
                return Err(votex_domain::error::AsrError::UnsupportedFormat(other.to_string())
                    .into());
            }
        };
        tracing::info!(
            "音频信息: {}Hz, {}声道, {}ms",
            audio.sample_rate,
            audio.channels,
            audio.duration_ms()
        );
        Ok(audio)
    }

    /// 经 ffmpeg 转码加载压缩音频：解码到临时 WAV，读取后即删
    fn load_audio_via_ffmpeg(path: &Path) -> Result<AudioData> {
        let encoder = votex_infra::audio::mp3::FfmpegEncoder::new()
            .map_err(|e| votex_domain::error::AsrError::AudioExtractFailed(e.to_string()))?;

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let temp_wav = std::env::temp_dir().join(format!(
            "votex_asr_decode_{}_{}.wav",
            std::process::id(),
            unique
        ));
        let result = (|| {
            encoder.decode_to_wav(path, &temp_wav)?;
            votex_infra::audio::wav::read_wav(&temp_wav)
        })();
        let _ = std::fs::remove_file(&temp_wav);
        result.map_err(|e| votex_domain::error::AsrError::AudioExtractFailed(e.to_string()).into())
    }

    /// 将音频按指定秒数分片
    fn slice_audio(audio: &AudioData, slice_seconds: u32) -> Vec<AudioData> {
        let samples_per_slice =
            (audio.sample_rate as usize * slice_seconds as usize) * audio.channels as usize;
        if audio.samples.len() <= samples_per_slice {
            return vec![audio.clone()];
        }

        audio
            .samples
            .chunks(samples_per_slice)
            .map(|chunk| AudioData {
                samples: chunk.to_vec(),
                sample_rate: audio.sample_rate,
                channels: audio.channels,
            })
            .collect()
    }

    /// 确保模型已加载
    fn ensure_loaded(&self, model: &Model) -> Result<()> {
        let provider = self.get_provider(model.engine)?;
        if !provider.is_loaded() {
            provider.load(model)?;
        }
        Ok(())
    }
}

// ============================================================
// 词级时间戳对齐
// ============================================================

/// 是否为真实词级时间戳（伪时间戳约定为 start=end=0）
fn is_aligned_word(w: &WordTimestamp) -> bool {
    w.end_ms > w.start_ms
}

/// 词级时间戳组句规则常量
const MAX_ENTRY_CHARS: usize = 24;
const MAX_ENTRY_DURATION_MS: f64 = 6000.0;
/// 词间静音间隔达到该值时断句（长停顿 = 新句）
const GAP_BREAK_MS: f64 = 600.0;

/// 将绝对时间轴的词级时间戳组装为字幕条目
///
/// 断句条件（任一命中即断）：
/// - 句尾标点（。！？；.!?;）
/// - 累计字符数超上限
/// - 累计时长超上限
/// - 与前一词的静音间隔超阈值
///
/// 文本拼接：CJK 连写；相邻英文词/数字之间插入空格。
pub fn group_words_into_entries(words: &[WordTimestamp]) -> Vec<SubtitleEntry> {
    if words.is_empty() {
        return Vec::new();
    }

    let mut entries = Vec::new();
    let mut seg_start = 0usize;

    while seg_start < words.len() {
        let mut seg_end = seg_start; // 当前句最后一个词索引（含）
        let mut chars = 0usize;
        let mut text = String::new();

        for i in seg_start..words.len() {
            let w = &words[i];
            let word_chars = w.word.chars().count();
            let start_ms = words[seg_start].start_ms;

            // 超限且已有内容 → 在当前词前断句
            if i > seg_start
                && (chars + word_chars > MAX_ENTRY_CHARS
                    || w.end_ms - start_ms > MAX_ENTRY_DURATION_MS
                    || w.start_ms - words[i - 1].end_ms > GAP_BREAK_MS)
            {
                break;
            }
            join_word(&mut text, &w.word);
            chars += word_chars;
            seg_end = i;

            // 句尾标点 → 包含该词后断句
            if ends_sentence(&w.word) {
                break;
            }
        }

        let entry_start = words[seg_start].start_ms.max(0.0) as u64;
        let entry_end = words[seg_end].end_ms.max(entry_start as f64) as u64;
        entries.push(SubtitleEntry::new(entries.len() + 1, entry_start, entry_end, text));
        seg_start = seg_end + 1;
    }

    entries
}

/// 是否为句尾标点（中英文）
fn ends_sentence(word: &str) -> bool {
    word.chars()
        .last()
        .map(|c| matches!(c, '。' | '！' | '？' | '；' | '.' | '!' | '?' | ';'))
        .unwrap_or(false)
}

/// 拼接词到文本：相邻 ASCII 字母/数字之间补空格，CJK 连写
fn join_word(text: &mut String, word: &str) {
    if word.is_empty() {
        return;
    }
    if let Some(prev) = text.chars().last() {
        let next = word.chars().next().unwrap();
        if is_ascii_word_char(prev) && is_ascii_word_char(next) {
            text.push(' ');
        }
    }
    text.push_str(word);
}

fn is_ascii_word_char(c: char) -> bool {
    c.is_ascii_alphanumeric()
}

// ============================================================
// 说话人标注
// ============================================================

/// 按时间重叠为字幕条目分配说话人（最大重叠片段的说话人）
pub fn assign_speakers(entries: &mut [SubtitleEntry], segments: &[DiarizationSegment]) {
    for entry in entries.iter_mut() {
        let e_start = entry.start_time.to_millis();
        let e_end = entry.end_time.to_millis();
        let mut best_speaker: Option<usize> = None;
        let mut best_overlap: u64 = 0;
        for seg in segments {
            let overlap = e_end.min(seg.end_ms).saturating_sub(e_start.max(seg.start_ms));
            if overlap > best_overlap {
                best_overlap = overlap;
                best_speaker = Some(seg.speaker);
            }
        }
        entry.speaker = best_speaker;
    }
}

/// 说话人时间轴 JSON 导出
fn write_diarization_json(path: &Path, segments: &[DiarizationSegment], speaker_count: Option<usize>) -> Result<()> {
    #[derive(serde::Serialize)]
    struct SegmentJson {
        speaker: usize,
        speaker_label: String,
        start_ms: u64,
        end_ms: u64,
    }
    #[derive(serde::Serialize)]
    struct DiarJson {
        num_speakers: Option<usize>,
        segments: Vec<SegmentJson>,
    }
    let doc = DiarJson {
        num_speakers: speaker_count,
        segments: segments
            .iter()
            .map(|s| SegmentJson {
                speaker: s.speaker,
                speaker_label: format!("说话人{}", s.speaker + 1),
                start_ms: s.start_ms,
                end_ms: s.end_ms,
            })
            .collect(),
    };
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(&doc)?)?;
    Ok(())
}

/// 词级时间戳 JSON 导出
fn write_words_json(path: &Path, words: &[WordTimestamp]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec_pretty(words)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(w: &str, start: f64, end: f64) -> WordTimestamp {
        WordTimestamp { word: w.into(), start_ms: start, end_ms: end }
    }

    #[test]
    fn 词级对齐_按标点断句() {
        let words = vec![
            word("今天", 0.0, 300.0),
            word("天气", 300.0, 600.0),
            word("不错。", 600.0, 900.0),
            word("出门", 900.0, 1200.0),
            word("走走。", 1200.0, 1500.0),
        ];
        let entries = group_words_into_entries(&words);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "今天天气不错。");
        assert_eq!(entries[1].text, "出门走走。");
        assert_eq!(entries[0].start_time.to_millis(), 0);
        assert_eq!(entries[0].end_time.to_millis(), 900);
        assert_eq!(entries[1].start_time.to_millis(), 900);
        assert_eq!(entries[1].end_time.to_millis(), 1500);
    }

    #[test]
    fn 词级对齐_长停顿断句() {
        // 两个词之间隔了 1 秒（> 600ms 阈值）→ 断句
        let words = vec![
            word("你好", 0.0, 300.0),
            word("喂喂", 1300.0, 1600.0),
        ];
        let entries = group_words_into_entries(&words);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "你好");
        assert_eq!(entries[1].text, "喂喂");
    }

    #[test]
    fn 词级对齐_英文词间补空格() {
        let words = vec![
            word("hello", 0.0, 300.0),
            word("world", 300.0, 600.0),
        ];
        let entries = group_words_into_entries(&words);
        assert_eq!(entries[0].text, "hello world");
    }

    #[test]
    fn 词级对齐_条目数上限编号连续() {
        let words: Vec<WordTimestamp> = (0..100)
            .map(|i| word(&format!("词{}", i), i as f64 * 100.0, i as f64 * 100.0 + 90.0))
            .collect();
        let entries = group_words_into_entries(&words);
        assert!(entries.len() > 1);
        for (i, e) in entries.iter().enumerate() {
            assert_eq!(e.index, i + 1);
            assert!(e.start_time.to_millis() <= e.end_time.to_millis());
        }
    }

    #[test]
    fn 词级对齐_空列表返回空() {
        assert!(group_words_into_entries(&[]).is_empty());
    }

    #[test]
    fn 伪时间戳不被视为对齐() {
        let pseudo = WordTimestamp { word: "整段".into(), start_ms: 0.0, end_ms: 0.0 };
        assert!(!is_aligned_word(&pseudo));
        assert!(is_aligned_word(&word("真", 0.0, 100.0)));
    }

    #[test]
    fn 说话人标注_按最大重叠分配() {
        let mut entries = vec![
            SubtitleEntry::new(1, 0, 1000, "甲说的话"),
            SubtitleEntry::new(2, 4000, 6000, "乙说的话"),
        ];
        let segments = vec![
            DiarizationSegment { start_ms: 0, end_ms: 3000, speaker: 0 },
            DiarizationSegment { start_ms: 3500, end_ms: 8000, speaker: 1 },
        ];
        assign_speakers(&mut entries, &segments);
        assert_eq!(entries[0].speaker, Some(0));
        assert_eq!(entries[1].speaker, Some(1));
        assert_eq!(entries[0].speaker_label().as_deref(), Some("说话人1"));
    }

    #[test]
    fn 说话人标注_无重叠时保持未标注() {
        let mut entries = vec![SubtitleEntry::new(1, 10000, 12000, "静音后的话")];
        let segments = vec![DiarizationSegment { start_ms: 0, end_ms: 3000, speaker: 0 }];
        assign_speakers(&mut entries, &segments);
        assert_eq!(entries[0].speaker, None);
    }

    #[test]
    fn 字幕条目_默认无说话人() {
        let e = SubtitleEntry::new(1, 0, 1000, "文本");
        assert_eq!(e.speaker, None);
        assert!(e.speaker_label().is_none());
    }
}
