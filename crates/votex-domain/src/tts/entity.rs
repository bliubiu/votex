use crate::shared::value_object::TaskId;
use crate::tts::value_object::*;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// 段落
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub index: usize,
    pub text: String,
    pub status: SegmentStatus,
    pub audio_path: Option<PathBuf>,
    pub duration_ms: Option<u32>,
}

impl Segment {
    pub fn new(index: usize, text: String) -> Self {
        Self {
            index,
            text,
            status: SegmentStatus::Pending,
            audio_path: None,
            duration_ms: None,
        }
    }
}

/// TTS 任务（聚合根）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsTask {
    pub id: TaskId,
    pub input: TtsInput,
    pub params: TtsParams,
    pub segments: Vec<Segment>,
    pub status: crate::shared::value_object::TaskStatus,
    pub output: Option<AudioOutput>,
    pub progress: TtsProgress,
    pub created_at: String,
    pub updated_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_创建() {
        let seg = Segment::new(0, "你好世界".to_string());
        assert_eq!(seg.index, 0);
        assert_eq!(seg.text, "你好世界");
        assert_eq!(seg.status, SegmentStatus::Pending);
        assert!(seg.audio_path.is_none());
    }
}
