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
