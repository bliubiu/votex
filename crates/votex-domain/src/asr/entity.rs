use crate::asr::value_object::*;
use crate::shared::value_object::TaskId;
use serde::{Deserialize, Serialize};

/// 音频切片
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSlice {
    pub index: usize,
    pub start_ms: u64,
    pub duration_ms: u64,
    pub status: SliceStatus,
    pub text: Option<String>,
}

impl AudioSlice {
    pub fn new(index: usize, start_ms: u64, duration_ms: u64) -> Self {
        Self {
            index,
            start_ms,
            duration_ms,
            status: SliceStatus::Pending,
            text: None,
        }
    }
}

/// ASR 任务（聚合根）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrTask {
    pub id: TaskId,
    pub input: AsrInput,
    pub params: AsrParams,
    pub slices: Vec<AudioSlice>,
    pub status: crate::shared::value_object::TaskStatus,
    pub result: Option<AsrResult>,
    pub progress: AsrProgress,
    /// 用户指定的字幕输出路径（None = 执行时按输入文件名推导）
    #[serde(default)]
    pub output_path: Option<std::path::PathBuf>,
    pub created_at: String,
    pub updated_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_slice_创建() {
        let slice = AudioSlice::new(0, 0, 30000);
        assert_eq!(slice.index, 0);
        assert_eq!(slice.start_ms, 0);
        assert_eq!(slice.duration_ms, 30000);
        assert_eq!(slice.status, SliceStatus::Pending);
        assert!(slice.text.is_none());
    }
}
