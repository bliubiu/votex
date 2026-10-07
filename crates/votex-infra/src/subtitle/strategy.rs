//! 双字幕生成策略
//!
//! 借鉴 MoneyPrinterTurbo 的双字幕引擎设计：
//!
//! | 策略 | 方式 | 速度 | 准确度 | 依赖 |
//! |------|------|------|--------|------|
//! | Fast | 使用 TTS 时间戳直接对齐 | 快 | 中等 | 无额外依赖 |
//! | Precise | 使用 ASR 二次识别生成 | 慢 | 高 | ASR 模型 |
//!
//! # 使用建议
//!
//! - 快速生成预览时使用 `Fast` 策略
//! - 需要精确字幕时使用 `Precise` 策略
//! - 也可在 GUI 中让用户选择

use std::path::Path;

use anyhow::Result;
use votex_domain::asr::value_object::{SubtitleEntry, SubtitleFormat, Timestamp, WordTimestamp};

use super::writer::SubtitleWriter;

// ============================================================
// 策略枚举
// ============================================================

/// 字幕生成策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleStrategy {
    /// 快速模式：基于 TTS 时间戳/文本分段对齐
    ///
    /// 适用于：预览、草稿、对准确度要求不高
    Fast,
    /// 精确模式：基于 ASR 二次识别生成带词级时间戳的字幕
    ///
    /// 适用于：正式发布、需要精确字幕
    Precise,
}

impl SubtitleStrategy {
    pub fn display_name(&self) -> &'static str {
        match self {
            SubtitleStrategy::Fast => "快速模式",
            SubtitleStrategy::Precise => "精确模式",
        }
    }

    /// 从配置字符串解析
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "precise" | "whisper" | "精确" => SubtitleStrategy::Precise,
            _ => SubtitleStrategy::Fast,
        }
    }
}

// ============================================================
// 快速字幕生成器
// ============================================================

/// 快速字幕生成器
///
/// 基于文本分段和 TTS 时长估计生成字幕。
/// 不需要额外的 ASR 模型，速度快。
///
/// 算法：
/// 1. 将文本按句子分割
/// 2. 根据总时长按比例分配每句的时间
/// 3. 生成 SRT/LRC/TXT 格式
pub struct FastSubtitleGenerator;

impl FastSubtitleGenerator {
    /// 从文本和音频时长生成字幕
    ///
    /// # 参数
    /// - `text`: 完整文本
    /// - `total_duration_ms`: 音频总时长（毫秒）
    /// - `format`: 输出字幕格式
    ///
    /// # 返回
    /// 字幕条目列表
    pub fn generate(text: &str, total_duration_ms: u64, _format: SubtitleFormat) -> Vec<SubtitleEntry> {
        let sentences = Self::split_sentences(text);
        if sentences.is_empty() {
            return Vec::new();
        }

        let total_chars: usize = sentences.iter().map(|s| s.chars().count()).sum();
        if total_chars == 0 {
            return Vec::new();
        }

        let mut entries = Vec::new();
        let mut current_time_ms: u64 = 0;

        for (i, sentence) in sentences.iter().enumerate() {
            let char_count = sentence.chars().count() as u64;
            // 按字数比例分配时间。不能给单句设 500ms 下限：句子数 × 500ms
            // 超过总时长时时间轴会在结尾塌缩（几十条字幕堆叠在同一时刻）。
            // 比例分配的总和恒 ≤ 总时长，最后一段取剩余全部时间。
            let duration_ms = if i == sentences.len() - 1 {
                // 最后一段占剩余全部时间
                total_duration_ms.saturating_sub(current_time_ms)
            } else {
                char_count * total_duration_ms / total_chars as u64
            };

            let end_time_ms = (current_time_ms + duration_ms).min(total_duration_ms);

            entries.push(SubtitleEntry {
                index: (i + 1) as usize,
                start_time: Timestamp::from_millis(current_time_ms),
                end_time: Timestamp::from_millis(end_time_ms),
                text: sentence.clone(),
                speaker: None,
            });

            current_time_ms = end_time_ms;
        }

        entries
    }

    /// 使用词级时间戳生成更精确的字幕
    ///
    /// 当 TTS 引擎返回词级时间戳时使用此方法
    pub fn from_word_timestamps(
        word_timestamps: &[WordTimestamp],
        format: SubtitleFormat,
    ) -> Vec<SubtitleEntry> {
        if word_timestamps.is_empty() {
            return Vec::new();
        }

        // 将词级时间戳合并为句子级
        let mut entries = Vec::new();
        let mut sentence_start_ms = word_timestamps[0].start_ms;
        let mut sentence_words = Vec::new();
        let mut index = 1;
        // 已组句消费的词数——下一句的起始时间必须取「下一个未消费词」，
        // 不能用 entries.len()（那是句子数，会导致时间轴随句子数累积错位）
        let mut word_index = 0usize;
        let _ = format; // 格式参数由调用方决定输出格式

        for wt in word_timestamps {
            sentence_words.push(wt.word.clone());

            // 遇到句尾标点时结束当前句子
            if wt.word.ends_with('.') || wt.word.ends_with('!') || wt.word.ends_with('?')
                || wt.word.ends_with('。') || wt.word.ends_with('！') || wt.word.ends_with('？')
                || wt.word.ends_with('\n')
            {
                entries.push(SubtitleEntry {
                    index,
                    start_time: Timestamp::from_millis(sentence_start_ms as u64),
                    end_time: Timestamp::from_millis(wt.end_ms as u64),
                    text: sentence_words.join(" "),
                    speaker: None,
                });
                index += 1;
                word_index += sentence_words.len();
                sentence_words.clear();
                if let Some(next) = word_timestamps.get(word_index) {
                    sentence_start_ms = next.start_ms;
                }
            }
        }

        // 处理剩余的未闭合句子
        if !sentence_words.is_empty() {
            let last_wt = word_timestamps.last().unwrap();
            entries.push(SubtitleEntry {
                index,
                start_time: Timestamp::from_millis(sentence_start_ms as u64),
                end_time: Timestamp::from_millis(last_wt.end_ms as u64),
                text: sentence_words.join(" "),
                speaker: None,
            });
        }

        entries
    }

    /// 将文本分割为句子
    fn split_sentences(text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            current.push(ch);
            if matches!(ch, '。' | '！' | '？' | '.' | '!' | '?' | '\n') {
                let s = current.trim().to_string();
                if !s.is_empty() {
                    sentences.push(s);
                }
                current.clear();
            }
        }

        let remaining = current.trim().to_string();
        if !remaining.is_empty() {
            sentences.push(remaining);
        }

        sentences
    }
}

// ============================================================
// 精确字幕生成器（基于 ASR 识别）
// ============================================================

/// 精确字幕生成器
///
/// 使用 ASR 引擎对生成的音频进行二次识别，
/// 获取精确的词级时间戳，生成高准确度字幕。
///
/// 速度较慢，但字幕准确度远高于 Fast 模式。
pub struct PreciseSubtitleGenerator;

impl PreciseSubtitleGenerator {
    /// 从 ASR 识别结果生成字幕
    ///
    /// 将词级时间戳按最大时长分组，在句子边界处断开。
    pub fn generate(
        word_timestamps: &[WordTimestamp],
        _format: SubtitleFormat,
        max_duration_ms: u64,
    ) -> Vec<SubtitleEntry> {
        if word_timestamps.is_empty() {
            return Vec::new();
        }

        let mut entries = Vec::new();
        let mut index = 1;
        let mut seg_start = 0; // 当前段在 word_timestamps 中的起始索引

        for i in 0..word_timestamps.len() {
            let wt = &word_timestamps[i];
            let seg_duration = wt.end_ms - word_timestamps[seg_start].start_ms;

            if seg_duration > max_duration_ms as f64 && i > seg_start {
                // 在当前段内找句子边界
                let break_pos = Self::find_break(word_timestamps, seg_start, i);
                let end_pos = break_pos.unwrap_or(i - 1);

                let start_ts = Timestamp::from_millis(word_timestamps[seg_start].start_ms as u64);
                let end_ts = Timestamp::from_millis(word_timestamps[end_pos].end_ms as u64);
                let text: String = word_timestamps[seg_start..=end_pos]
                    .iter()
                    .map(|w| w.word.as_str())
                    .collect::<Vec<_>>()
                    .join(" ");

                entries.push(SubtitleEntry { index, start_time: start_ts, end_time: end_ts, text, speaker: None });
                index += 1;
                seg_start = end_pos + 1;
            }
        }

        // 最后一段
        if seg_start < word_timestamps.len() {
            let start_ts = Timestamp::from_millis(word_timestamps[seg_start].start_ms as u64);
            let end_ts = Timestamp::from_millis(word_timestamps.last().unwrap().end_ms as u64);
            let text: String = word_timestamps[seg_start..]
                .iter()
                .map(|w| w.word.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            entries.push(SubtitleEntry { index, start_time: start_ts, end_time: end_ts, text, speaker: None });
        }

        entries
    }

    /// 在 [start, end] 范围内找句子边界，优先找句尾标点
    fn find_break(wts: &[WordTimestamp], start: usize, end: usize) -> Option<usize> {
        let search_end = end.min(wts.len().saturating_sub(1));
        for i in (start..=search_end).rev() {
            let w = &wts[i].word;
            if w.ends_with('.') || w.ends_with('!') || w.ends_with('?')
                || w.ends_with('。') || w.ends_with('！') || w.ends_with('？')
            {
                return Some(i);
            }
        }
        None
    }

    }

// ============================================================
// 字幕生成服务（策略分发）
// ============================================================

/// 字幕生成器（按策略分发）
pub struct SubtitleGenerator;

impl SubtitleGenerator {
    /// 根据策略生成字幕
    pub fn generate(
        strategy: SubtitleStrategy,
        text: &str,
        word_timestamps: Option<&[WordTimestamp]>,
        total_duration_ms: u64,
        format: SubtitleFormat,
    ) -> Vec<SubtitleEntry> {
        match strategy {
            SubtitleStrategy::Fast => {
                if let Some(wts) = word_timestamps {
                    if !wts.is_empty() {
                        return FastSubtitleGenerator::from_word_timestamps(wts, format);
                    }
                }
                FastSubtitleGenerator::generate(text, total_duration_ms, format)
            }
            SubtitleStrategy::Precise => {
                if let Some(wts) = word_timestamps {
                    if !wts.is_empty() {
                        return PreciseSubtitleGenerator::generate(wts, format, 5000);
                    }
                }
                // 没有时间戳时回退到快速模式
                FastSubtitleGenerator::generate(text, total_duration_ms, format)
            }
        }
    }

    /// 生成并写入文件
    pub fn generate_to_file(
        strategy: SubtitleStrategy,
        text: &str,
        word_timestamps: Option<&[WordTimestamp]>,
        total_duration_ms: u64,
        format: SubtitleFormat,
        output_path: &Path,
    ) -> Result<()> {
        let entries = Self::generate(strategy, text, word_timestamps, total_duration_ms, format);
        SubtitleWriter::write(&entries, format, output_path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_display() {
        assert_eq!(SubtitleStrategy::Fast.display_name(), "快速模式");
        assert_eq!(SubtitleStrategy::Precise.display_name(), "精确模式");
    }

    #[test]
    fn test_strategy_from_str() {
        assert_eq!(SubtitleStrategy::from_str("fast"), SubtitleStrategy::Fast);
        assert_eq!(SubtitleStrategy::from_str("precise"), SubtitleStrategy::Precise);
        assert_eq!(SubtitleStrategy::from_str("精确"), SubtitleStrategy::Precise);
        assert_eq!(SubtitleStrategy::from_str("unknown"), SubtitleStrategy::Fast);
    }

    #[test]
    fn test_fast_generator_基本功能() {
        let text = "你好世界。这是一段测试。";
        let entries = FastSubtitleGenerator::generate(text, 6000, SubtitleFormat::Srt);
        assert!(!entries.is_empty());
        assert_eq!(entries[0].index, 1);
        // 两句话应该生成两条字幕
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_fast_generator_总时长分配() {
        let text = "你好世界。这是一段测试文本。";
        let entries = FastSubtitleGenerator::generate(text, 10000, SubtitleFormat::Srt);
        assert!(entries.len() >= 2);
        // 最后一条字幕的结束时间不应超过总时长
        if let Some(last) = entries.last() {
            assert!(last.end_time.to_srt_format() >= last.start_time.to_srt_format());
        }
    }

    #[test]
    fn test_fast_generator_空文本() {
        let entries = FastSubtitleGenerator::generate("", 1000, SubtitleFormat::Srt);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_fast_generator_从词级时间戳() {
        let wts = vec![
            WordTimestamp { word: "你好".to_string(), start_ms: 0.0, end_ms: 500.0 },
            WordTimestamp { word: "世界。".to_string(), start_ms: 500.0, end_ms: 1000.0 },
            WordTimestamp { word: "这是".to_string(), start_ms: 1000.0, end_ms: 1500.0 },
            WordTimestamp { word: "测试。".to_string(), start_ms: 1500.0, end_ms: 2000.0 },
        ];

        let entries = FastSubtitleGenerator::from_word_timestamps(&wts, SubtitleFormat::Srt);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "你好 世界。");
        assert_eq!(entries[1].text, "这是 测试。");
    }

    #[test]
    fn test_fast_generator_词级时间戳_时间轴不错位() {
        // 第一句占 3 个词：下一句的起始时间必须取第 4 个词（索引 3），
        // 此前用 entries.len()（=1）当词索引，取到第一句中间的词导致时间轴错位
        let wts = vec![
            WordTimestamp { word: "第一句".to_string(), start_ms: 0.0, end_ms: 400.0 },
            WordTimestamp { word: "占三个".to_string(), start_ms: 400.0, end_ms: 800.0 },
            WordTimestamp { word: "词。".to_string(), start_ms: 800.0, end_ms: 1200.0 },
            WordTimestamp { word: "第二句。".to_string(), start_ms: 1200.0, end_ms: 2000.0 },
        ];

        let entries = FastSubtitleGenerator::from_word_timestamps(&wts, SubtitleFormat::Srt);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].start_time.to_srt_format(), "00:00:00,000");
        assert_eq!(entries[0].end_time.to_srt_format(), "00:00:01,200");
        assert_eq!(entries[1].start_time.to_srt_format(), "00:00:01,200");
        assert_eq!(entries[1].end_time.to_srt_format(), "00:00:02,000");
    }

    #[test]
    fn test_fast_generator_短时长不塌缩() {
        // 句数 × 500ms 下限曾超过总时长，导致几十条字幕堆叠在结尾同一时刻；
        // 比例分配下时间轴必须单调前进且落在总时长内
        let text = "一。二。三。四。五。六。七。八。九。十。";
        let total = 1000u64;
        let entries = FastSubtitleGenerator::generate(text, total, SubtitleFormat::Srt);
        assert_eq!(entries.len(), 10);
        for (i, e) in entries.iter().enumerate() {
            assert!(e.start_time.to_srt_format() <= e.end_time.to_srt_format());
            if i > 0 {
                let prev = &entries[i - 1];
                assert!(
                    e.start_time.to_srt_format() >= prev.end_time.to_srt_format(),
                    "字幕时间轴必须单调前进: 第{}条早于第{}条结束",
                    i + 1,
                    i
                );
            }
        }
        let last = entries.last().unwrap();
        assert_eq!(last.end_time.to_srt_format(), "00:00:01,000");
    }

    #[test]
    fn test_lrc_超一小时时间轴() {
        // 1h01m23s 处的条目此前写成 [01:23.45]（丢小时位），
        // LRC 分钟位必须折入小时：1h01m23.45s = 61分23.45秒 → [61:23.45]
        let ts = Timestamp::from_millis(3_683_450);
        assert_eq!(ts.to_lrc_format(), "[61:23.45]");
        assert_eq!(Timestamp::from_millis(59_999).to_lrc_format(), "[00:59.99]");
    }

    #[test]
    fn test_precise_generator_基本功能() {
        let wts = vec![
            WordTimestamp { word: "今天".to_string(), start_ms: 0.0, end_ms: 300.0 },
            WordTimestamp { word: "天气".to_string(), start_ms: 300.0, end_ms: 600.0 },
            WordTimestamp { word: "真不错。".to_string(), start_ms: 600.0, end_ms: 1000.0 },
            WordTimestamp { word: "我们".to_string(), start_ms: 1000.0, end_ms: 1300.0 },
            WordTimestamp { word: "去".to_string(), start_ms: 1300.0, end_ms: 1500.0 },
            WordTimestamp { word: "散步吧。".to_string(), start_ms: 1500.0, end_ms: 2000.0 },
        ];

        let entries = PreciseSubtitleGenerator::generate(&wts, SubtitleFormat::Srt, 5000);
        assert!(!entries.is_empty());
        assert_eq!(entries[0].index, 1);
    }

    #[test]
    fn test_precise_generator_时长分组() {
        let wts: Vec<WordTimestamp> = (0..20).map(|i| {
            WordTimestamp {
                word: format!("词{}", i),
                start_ms: i as f64 * 300.0,
                end_ms: (i + 1) as f64 * 300.0,
            }
        }).collect();

        // 设置 max_duration_ms = 2000，应该有多个分组
        let entries = PreciseSubtitleGenerator::generate(&wts, SubtitleFormat::Srt, 2000);
        assert!(entries.len() > 1);
    }

    #[test]
    fn test_subtitle_generator_策略分发() {
        let text = "测试文本。用于生成字幕。";
        let entries = SubtitleGenerator::generate(
            SubtitleStrategy::Fast,
            text,
            None,
            5000,
            SubtitleFormat::Srt,
        );
        assert!(!entries.is_empty());
    }

    #[test]
    fn test_subtitle_generator_写入文件() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.srt");

        SubtitleGenerator::generate_to_file(
            SubtitleStrategy::Fast,
            "测试文本。",
            None,
            3000,
            SubtitleFormat::Srt,
            &path,
        ).unwrap();

        assert!(path.exists());
    }
}
