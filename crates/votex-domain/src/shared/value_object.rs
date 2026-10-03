use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::PathBuf;

/// 任务唯一标识
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(String);

impl TaskId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    pub fn from_string(s: String) -> Self {
        Self(s)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// 任务状态
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    Queued,
    Running,
    Completed,
    Failed(String),
    Cancelled,
    Paused,
}

/// 流水线唯一标识
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PipelineId(String);

impl PipelineId {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for PipelineId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for PipelineId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// 内存中的音频数据（PCM）
#[derive(Debug, Clone)]
pub struct AudioData {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioData {
    /// 创建静音音频数据
    pub fn silence(sample_rate: u32, duration_ms: u32) -> Self {
        let num_samples = (sample_rate as u64 * duration_ms as u64 / 1000) as usize;
        Self {
            samples: vec![0.0f32; num_samples],
            sample_rate,
            channels: 1,
        }
    }

    /// 音频时长（毫秒）
    pub fn duration_ms(&self) -> u32 {
        if self.sample_rate == 0 {
            return 0;
        }
        (self.samples.len() as u64 * 1000 / self.sample_rate as u64 / self.channels as u64) as u32
    }

    /// 转换为 16kHz 单声道 f32 PCM（Whisper 所需格式）
    pub fn to_mono_f32_16k(&self) -> Vec<f32> {
        // 快速路径：已经是 16kHz 单声道
        if self.channels == 1 && self.sample_rate == 16000 {
            return self.samples.clone();
        }

        let mono: Vec<f32> = if self.channels > 1 {
            let inv_channels = 1.0 / self.channels as f32;
            self.samples
                .chunks(self.channels as usize)
                .map(|ch| ch.iter().sum::<f32>() * inv_channels)
                .collect()
        } else {
            self.samples.clone()
        };

        if self.sample_rate == 16000 {
            mono
        } else {
            // 简单线性重采样
            let ratio = 16000.0 / self.sample_rate as f64;
            let new_len = (mono.len() as f64 * ratio) as usize;
            (0..new_len)
                .map(|i| {
                    let src_idx = i as f64 / ratio;
                    let idx = src_idx as usize;
                    let frac = src_idx - idx as f64;
                    let a = mono.get(idx).copied().unwrap_or(0.0);
                    let b = mono.get(idx + 1).copied().unwrap_or(0.0);
                    a + (b - a) * frac as f32
                })
                .collect()
        }
    }
}

/// 哈希算法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HashAlgorithm {
    Sha256,
}

/// 哈希值
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hash {
    pub algorithm: HashAlgorithm,
    pub value: String,
}

/// 文件路径（类型别名）
pub type FilePath = PathBuf;

/// 取消令牌 - 用于支持任务取消
/// 简单包装 AtomicBool，不依赖 tokio，保持 domain 层纯净
#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            cancelled: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// 请求取消
    pub fn cancel(&self) {
        self.cancelled.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// 是否已取消
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// 重置取消状态
    pub fn reset(&self) {
        self.cancelled.store(false, std::sync::atomic::Ordering::SeqCst);
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

/// 进度回调 - 用于实时报告任务进度
pub type ProgressCallback = Option<Box<dyn Fn(ProgressEvent) + Send + 'static>>;

/// 进度事件
#[derive(Debug, Clone)]
pub enum ProgressEvent {
    /// 阶段变更
    PhaseChanged { phase: String },
    /// 页面进度
    PageProgress { current: usize, total: usize, page_index: usize },
    /// 识别块进度
    BlockProgress { current: usize, total: usize },
    /// 消息
    Message { text: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_id_唯一性() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn task_id_从字符串创建() {
        let id = TaskId::from_string("test-id".to_string());
        assert_eq!(id.as_str(), "test-id");
    }

    #[test]
    fn task_id_显示格式() {
        let id = TaskId::from_string("abc-123".to_string());
        assert_eq!(format!("{}", id), "abc-123");
    }

    #[test]
    fn pipeline_id_唯一性() {
        let id1 = PipelineId::new();
        let id2 = PipelineId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn audio_data_静音生成() {
        let audio = AudioData::silence(16000, 1000);
        assert_eq!(audio.sample_rate, 16000);
        assert_eq!(audio.samples.len(), 16000);
        assert!(audio.samples.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn audio_data_时长计算() {
        let audio = AudioData::silence(16000, 2000);
        assert_eq!(audio.duration_ms(), 2000);
    }

    #[test]
    fn task_status_序列化() {
        let status = TaskStatus::Failed("模型加载失败".to_string());
        let json = serde_json::to_string(&status).unwrap();
        let deserialized: TaskStatus = serde_json::from_str(&json).unwrap();
        assert_eq!(status, deserialized);
    }
}
