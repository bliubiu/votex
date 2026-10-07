use crate::asr::value_object::*;
use crate::asr::entity::AudioSlice;

/// 音频切片领域服务
pub struct AudioSlicer;

impl AudioSlicer {
    pub fn new() -> Self {
        Self
    }

    /// 按时长计算切片列表
    pub fn plan_slices(duration_ms: u64, slice_length: SliceLength) -> Vec<AudioSlice> {
        let slice_ms = slice_length.seconds() as u64 * 1000;
        let mut slices = Vec::new();
        let mut start = 0u64;
        let mut index = 0;

        while start < duration_ms {
            let remaining = duration_ms - start;
            let this_duration = remaining.min(slice_ms);
            slices.push(AudioSlice::new(index, start, this_duration));
            start += this_duration;
            index += 1;
        }

        slices
    }
}

/// 字幕格式化领域服务
pub struct SubtitleFormatter;

impl SubtitleFormatter {
    /// 格式化为纯文本
    pub fn format_txt(result: &AsrResult) -> String {
        result.text.clone()
    }

    /// 格式化为 SRT
    pub fn format_srt(result: &AsrResult) -> String {
        result
            .subtitles
            .iter()
            .map(|entry| {
                format!(
                    "{}\n{} --> {}\n{}\n",
                    entry.index + 1,
                    entry.start_time.to_srt_format(),
                    entry.end_time.to_srt_format(),
                    entry.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// 格式化为 LRC
    pub fn format_lrc(result: &AsrResult) -> String {
        result
            .subtitles
            .iter()
            .map(|entry| {
                format!("{}{}", entry.start_time.to_lrc_format(), entry.text)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn audio_slicer_切片计划() {
        let slices = AudioSlicer::plan_slices(60000, SliceLength::S30);
        assert_eq!(slices.len(), 2);
        assert_eq!(slices[0].start_ms, 0);
        assert_eq!(slices[0].duration_ms, 30000);
        assert_eq!(slices[1].start_ms, 30000);
        assert_eq!(slices[1].duration_ms, 30000);
    }

    #[test]
    fn audio_slicer_不足一切片() {
        let slices = AudioSlicer::plan_slices(10000, SliceLength::S30);
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].duration_ms, 10000);
    }

    #[test]
    fn subtitle_formatter_srt格式() {
        let result = AsrResult {
            text: "你好世界".to_string(),
            subtitles: vec![SubtitleEntry {
                index: 0,
                start_time: Timestamp::from_millis(0),
                end_time: Timestamp::from_millis(3000),
                text: "你好世界".to_string(),
                speaker: None,
            }],
            output_path: PathBuf::from("test.srt"),
        };
        let srt = SubtitleFormatter::format_srt(&result);
        assert!(srt.contains("1\n"));
        assert!(srt.contains("00:00:00,000 --> 00:00:03,000"));
        assert!(srt.contains("你好世界"));
    }

    #[test]
    fn subtitle_formatter_lrc格式() {
        let result = AsrResult {
            text: "你好".to_string(),
            subtitles: vec![SubtitleEntry {
                index: 0,
                start_time: Timestamp {
                    hours: 0,
                    minutes: 1,
                    seconds: 5,
                    millis: 300,
                },
                end_time: Timestamp::from_millis(70000),
                text: "你好".to_string(),
                speaker: None,
            }],
            output_path: PathBuf::from("test.lrc"),
        };
        let lrc = SubtitleFormatter::format_lrc(&result);
        assert!(lrc.contains("[01:05.30]你好"));
    }
}
