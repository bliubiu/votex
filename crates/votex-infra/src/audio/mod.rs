//! 音频处理：编解码、重采样、降噪、VAD、音量分析。
//!
//! 重采样与降噪为纯函数，不依赖 ONNX，便于独立测试。

#[cfg(feature = "ffmpeg")]
pub mod mp3;
pub mod wav;
pub mod denoiser;
pub mod ref_audio;
/// 麦克风实时采集（实时听写输入）
pub mod capture;
