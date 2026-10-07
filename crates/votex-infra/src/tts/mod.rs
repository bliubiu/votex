//! TTS 引擎适配器：Kokoro / IndexTTS-2.5 / CosyVoice / Qwen3-TTS。
//!
//! 采样逻辑已统一到 `crate::shared::sampling`，引擎内不应再有各自的
//! softmax / top-k 实现。

pub mod kokoro;
pub mod indextts25;
pub mod kokoro_g2p;
pub mod azure_speech;
pub mod aliyun;
pub mod qwen3_tts;
pub mod cosyvoice;
pub mod qwen3_model_selector;
pub mod voice_library;

/// 情感参数不支持的一次性告警器（每个引擎实例化一个，进程内只提示一次）
///
/// 长文本合成会把文本切成几十上百段，逐段告警会刷屏；
/// 情感通道缺失属于能力边界而非逐段故障，提示一次即可。
pub(crate) struct EmotionWarnOnce(std::sync::Once);

impl EmotionWarnOnce {
    pub(crate) const fn new() -> Self {
        Self(std::sync::Once::new())
    }

    /// `params.emotion` 为非 Neutral 且引擎不支持时告警一次
    pub(crate) fn warn_if_unsupported(&self, engine: &str, params: &votex_domain::tts::value_object::TtsParams) {
        let has_emotion = params
            .emotion
            .as_ref()
            .is_some_and(|e| e.emotion != votex_domain::tts::value_object::Emotion::Neutral);
        if has_emotion {
            self.0.call_once(|| {
                tracing::warn!("{engine} 不支持情感合成参数（无情感控制通道），emotion 将被忽略");
            });
        }
    }
}
