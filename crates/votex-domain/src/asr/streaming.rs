//! 流式（实时）语音识别领域接口
//!
//! 与 [`crate::asr::provider::AsrProvider`]（整段识别）不同，
//! 流式识别按增量送入音频、随时吐出中间结果，用于实时听写场景。
//!
//! 线程模型：`create_session` 返回的会话对象可整体移动（`Send`）到
//! 工作线程使用；同一会话内的所有方法只在单一线程调用，
//! 引擎并发安全由 Provider 侧保证。

use crate::error::AsrError;
use crate::model::entity::Model;
use crate::model::value_object::EngineKind;

/// 流式识别增量更新
///
/// 一次 `accept_waveform` 调用返回一次更新：
/// - `partial`：当前尚未定稿的文本（会被后续更新覆盖，展示用）
/// - `final_text`：端点检测命中的定稿文本（不会重复给出，非 None 时应追加到记录）
/// - `elapsed_ms`：自会话开始累计的音频时长
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StreamingUpdate {
    pub partial: String,
    pub final_text: Option<String>,
    pub elapsed_ms: u64,
}

/// 流式识别会话
///
/// 生命周期：`accept_waveform`（任意次）→ `finish`（一次，终结）。
/// `finish` 之后会话作废，继续调用返回 `AsrError::RecognizeFailed`。
pub trait StreamingAsrSession: Send {
    /// 送入一段 16kHz 单声道 f32 PCM 音频
    fn accept_waveform(&mut self, samples: &[f32]) -> Result<StreamingUpdate, AsrError>;

    /// 结束输入，冲刷解码缓冲，返回最后一段未定稿文本
    fn finish(&mut self) -> Result<String, AsrError>;
}

/// 流式 ASR 引擎 Provider 接口
pub trait StreamingAsrProvider: Send + Sync {
    /// 返回引擎类型
    fn engine_kind(&self) -> EngineKind;

    /// 加载模型到内存（解析并校验模型文件，构建识别器配置）
    fn load(&self, model: &Model) -> Result<(), AsrError>;

    /// 释放模型资源
    fn unload(&self) -> Result<(), AsrError>;

    /// 是否已加载
    fn is_loaded(&self) -> bool;

    /// 期望输入采样率
    fn sample_rate(&self) -> u32;

    /// 创建一个新的流式识别会话
    ///
    /// 会话持有独立的解码上下文；多个会话可并存，但底层引擎解码
    /// 会串行化（同一模型只加载一份）。
    fn create_session(&self) -> Result<Box<dyn StreamingAsrSession + Send>, AsrError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 假会话：验证 trait 对象可用性与默认更新值
    struct FakeSession;

    impl StreamingAsrSession for FakeSession {
        fn accept_waveform(&mut self, _samples: &[f32]) -> Result<StreamingUpdate, AsrError> {
            Ok(StreamingUpdate::default())
        }

        fn finish(&mut self) -> Result<String, AsrError> {
            Ok(String::new())
        }
    }

    #[test]
    fn streaming_update_默认值为空() {
        let u = StreamingUpdate::default();
        assert!(u.partial.is_empty());
        assert!(u.final_text.is_none());
        assert_eq!(u.elapsed_ms, 0);
    }

    #[test]
    fn streaming_session_可作为trait对象() {
        let mut s: Box<dyn StreamingAsrSession + Send> = Box::new(FakeSession);
        let u = s.accept_waveform(&[0.0; 160]).unwrap();
        assert_eq!(u, StreamingUpdate::default());
        assert_eq!(s.finish().unwrap(), "");
    }
}
