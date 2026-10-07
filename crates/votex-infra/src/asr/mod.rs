//! ASR 引擎适配器：SenseVoice / Whisper / FireRed / WeNet / Qwen3-ASR 等。
//!
//! 每个适配器负责自己的分词、特征提取与解码，差异较大，暂不抽象公共基类。

pub mod whisper;
pub mod sensevoice;
pub mod paraformer;
pub mod qwen3_asr;
pub mod firered_asr;
pub mod wenet;
pub mod azure_speech;
pub mod aliyun;
pub mod speaker_diarization;
pub mod streaming;

use votex_domain::asr::value_object::{RecognizeOutput, WordTimestamp};

/// 将 sherpa-onnx 离线识别结果组装为领域输出
///
/// - 模型提供 token 对齐时间戳（whisper / paraformer / CTC 类）时，
///   产出**真实词级时间戳**，供字幕精准对齐与说话人分离复用；
/// - 模型不提供（如 SenseVoice）时回退为整段伪时间戳
///   （`start_ms = end_ms = 0`），下游据此识别为「未对齐」并走切片级字幕。
///
/// `tokens` 与 `timestamps` 长度必须一致且非空才视为有效对齐。
/// `tokens` 中的 sentencepiece 占位符（`▁`）与特殊标记（`<|...|>`）会被清理。
pub fn recognize_output_from(
    text: String,
    tokens: &[String],
    timestamps: Option<&[f32]>,
) -> RecognizeOutput {
    let word_timestamps = match (tokens, timestamps) {
        ([first, ..], Some(ts)) if ts.len() == tokens.len() && !ts.is_empty() => {
            let mut words = Vec::with_capacity(tokens.len());
            for (token, &sec) in tokens.iter().zip(ts.iter()) {
                // 清理 sentencepiece 下划线占位与特殊标记
                let word = token.replace('▁', " ");
                let word = word.trim();
                if word.is_empty() || word.starts_with("<|") {
                    continue;
                }
                words.push(WordTimestamp {
                    word: word.to_string(),
                    start_ms: (sec.max(0.0) * 1000.0) as f64,
                    end_ms: (sec.max(0.0) * 1000.0) as f64,
                });
            }
            // 时间戳是帧粒度（单点），给每个词补一个递增的终点：
            // 用下一词起点作为当前词终点，末词保持单点
            for i in 0..words.len().saturating_sub(1) {
                if words[i + 1].start_ms > words[i].start_ms {
                    words[i].end_ms = words[i + 1].start_ms;
                } else {
                    words[i].end_ms = words[i].start_ms;
                }
            }
            // sherpa 结果异常（token 多而时间戳全零）时视为未对齐
            if words.iter().all(|w| w.end_ms <= 0.0) {
                pseudo_span(text.clone())
            } else {
                words.shrink_to_fit();
                words
            }
        }
        _ => pseudo_span(text.clone()),
    };

    RecognizeOutput {
        text,
        word_timestamps,
    }
}

/// 伪时间戳：整段一个条目、零时长——语义为「引擎未提供词级时间戳」
fn pseudo_span(text: String) -> Vec<WordTimestamp> {
    if text.is_empty() {
        Vec::new()
    } else {
        vec![WordTimestamp {
            word: text,
            start_ms: 0.0,
            end_ms: 0.0,
        }]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 结果组装_有对齐时间戳时产出词级() {
        let tokens = vec!["你".to_string(), "好".to_string(), "▁world".to_string()];
        let ts = vec![0.0f32, 0.2, 0.5];
        let out = recognize_output_from("你好 world".into(), &tokens, Some(&ts));
        assert_eq!(out.word_timestamps.len(), 3);
        assert_eq!(out.word_timestamps[0].word, "你");
        assert_eq!(out.word_timestamps[1].word, "好");
        // ▁ 清理 + 词终点 = 下一词起点
        assert_eq!(out.word_timestamps[2].word, "world");
        assert_eq!(out.word_timestamps[0].end_ms, 200.0);
        assert_eq!(out.word_timestamps[1].end_ms, 500.0);
        assert_eq!(out.word_timestamps[2].end_ms, 500.0);
    }

    #[test]
    fn 结果组装_特殊标记被过滤() {
        let tokens = vec!["<|zh|>".to_string(), "你".to_string(), "<|EMO|>".to_string(), "好".to_string()];
        let ts = vec![0.0f32, 0.1, 0.2, 0.3];
        let out = recognize_output_from("你好".into(), &tokens, Some(&ts));
        assert_eq!(out.word_timestamps.len(), 2);
        assert_eq!(out.word_timestamps[0].word, "你");
        assert_eq!(out.word_timestamps[1].word, "好");
    }

    #[test]
    fn 结果组装_无时间戳回退伪整段() {
        let out = recognize_output_from("你好".into(), &[], None);
        assert_eq!(out.word_timestamps.len(), 1);
        assert_eq!(out.word_timestamps[0].start_ms, 0.0);
        assert_eq!(out.word_timestamps[0].end_ms, 0.0);

        let empty = recognize_output_from(String::new(), &[], None);
        assert!(empty.word_timestamps.is_empty());
    }

    #[test]
    fn 结果组装_长度不匹配回退() {
        let tokens = vec!["你".to_string(), "好".to_string()];
        let ts = vec![0.1f32];
        let out = recognize_output_from("你好".into(), &tokens, Some(&ts));
        assert_eq!(out.word_timestamps.len(), 1, "长度不匹配应回退伪整段");
    }
}
