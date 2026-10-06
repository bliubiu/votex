use crate::model::value_object::ModelId;
use serde::{Deserialize, Serialize};

/// 识别语种
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Language {
    /// 中文
    Zh,
    /// 中英混合
    ZhEn,
    /// 英文
    En,
}

/// 切片时长
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SliceLength {
    S15,
    S30,
    S60,
}

impl SliceLength {
    pub fn seconds(&self) -> u32 {
        match self {
            SliceLength::S15 => 15,
            SliceLength::S30 => 30,
            SliceLength::S60 => 60,
        }
    }
}

impl Default for SliceLength {
    fn default() -> Self {
        SliceLength::S30
    }
}

/// 字幕格式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubtitleFormat {
    Txt,
    Srt,
    Lrc,
}

/// 降噪强度（复用 TTS 域定义）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DenoiseLevel {
    Low,
    Medium,
    High,
}

impl Default for DenoiseLevel {
    fn default() -> Self {
        DenoiseLevel::Low
    }
}

/// 媒体类型
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MediaKind {
    Audio(AudioFormatKind),
    Video,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormatKind {
    Wav,
    Mp3,
    Flac,
    Other,
}

/// ASR 输入
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrInput {
    pub file_path: std::path::PathBuf,
    pub file_kind: MediaKind,
}

/// ASR 识别参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrParams {
    pub model: ModelId,
    pub language: Language,
    pub auto_punctuation: bool,
    pub auto_slice: bool,
    pub slice_length: SliceLength,
    pub denoise: bool,
    pub denoise_level: DenoiseLevel,
    pub output_format: SubtitleFormat,
}

/// 时间戳
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Timestamp {
    pub hours: u32,
    pub minutes: u32,
    pub seconds: u32,
    pub millis: u32,
}

impl Timestamp {
    pub fn from_millis(ms: u64) -> Self {
        Self {
            hours: (ms / 3_600_000) as u32,
            minutes: ((ms % 3_600_000) / 60_000) as u32,
            seconds: ((ms % 60_000) / 1000) as u32,
            millis: (ms % 1000) as u32,
        }
    }

    /// SRT 时间格式：HH:MM:SS,mmm
    pub fn to_srt_format(&self) -> String {
        format!(
            "{:02}:{:02}:{:02},{:03}",
            self.hours, self.minutes, self.seconds, self.millis
        )
    }

    /// LRC 时间格式：[mm:ss.xx]（超 1 小时折入分钟位，如 1h02m → [62:00.00]，
    /// LRC 标准只有分钟位，直接丢弃小时位会让整条时间轴错位）
    pub fn to_lrc_format(&self) -> String {
        format!(
            "[{:02}:{:02}.{:02}]",
            self.hours * 60 + self.minutes,
            self.seconds,
            self.millis / 10
        )
    }
}

/// 字幕条目
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubtitleEntry {
    pub index: usize,
    pub start_time: Timestamp,
    pub end_time: Timestamp,
    pub text: String,
}

/// ASR 识别结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrResult {
    pub text: String,
    pub subtitles: Vec<SubtitleEntry>,
    pub output_path: std::path::PathBuf,
}

/// 词级时间戳
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WordTimestamp {
    pub word: String,
    pub start_ms: f64,
    pub end_ms: f64,
}

/// 识别输出
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognizeOutput {
    pub text: String,
    pub word_timestamps: Vec<WordTimestamp>,
}

/// 切片状态
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SliceStatus {
    Pending,
    Recognizing,
    Completed,
    Failed(String),
}

/// ASR 阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AsrPhase {
    Idle,
    ExtractingAudio,
    Denoising,
    Slicing,
    Recognizing,
    Exporting,
    Completed,
    Failed,
    Paused,
    Cancelled,
}

/// ASR 进度
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AsrProgress {
    pub current_slice: usize,
    pub total_slices: usize,
    pub phase: AsrPhase,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_length_秒数映射() {
        assert_eq!(SliceLength::S15.seconds(), 15);
        assert_eq!(SliceLength::S30.seconds(), 30);
        assert_eq!(SliceLength::S60.seconds(), 60);
    }

    #[test]
    fn timestamp_从毫秒创建() {
        let ts = Timestamp::from_millis(3661500); // 1h1m1s500ms
        assert_eq!(ts.hours, 1);
        assert_eq!(ts.minutes, 1);
        assert_eq!(ts.seconds, 1);
        assert_eq!(ts.millis, 500);
    }

    #[test]
    fn timestamp_srt格式() {
        let ts = Timestamp::from_millis(3661500);
        assert_eq!(ts.to_srt_format(), "01:01:01,500");
    }

    #[test]
    fn timestamp_lrc格式() {
        let ts = Timestamp {
            hours: 0,
            minutes: 1,
            seconds: 23,
            millis: 450,
        };
        assert_eq!(ts.to_lrc_format(), "[01:23.45]");
    }
}
