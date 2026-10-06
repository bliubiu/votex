//! IndexTTS-2.5（ONNX）Rust 移植模块。
//!
//! 蓝本：`docs/25-IndexTTS25-Rust移植蓝本.md`（源自 index-tts-2.5-onnx 0.1.0 参考实现，
//! 粤语试听决策门 2026-10-05 通过后启动移植）。
//!
//! 模块进度（对应蓝本 §6 规划）：
//! - `bpe` / `frontend`：文本前端（词表加载、BPE 编码、清洗与分段）✅
//! - `features`：w2v-bert 输入特征（SeamlessM4T fbank + stride 堆叠）✅
//! - `engines`：8 个 ONNX session 封装 ✅
//! - `sampler` / `gpt`：GPT 采样链与自回归主循环 ✅
//! - `dsp`：sinc 重采样 / s2mel 频谱 / kaldi fbank / campplus fbank ✅
//! - `pipeline`：build_speaker / cfm_solve / synthesize ✅（Provider 接入见 #47）

pub mod bpe;
pub mod dsp;
pub mod engines;
pub mod features;
pub mod frontend;
pub mod gpt;
pub mod normalizer;
pub mod pipeline;
pub mod provider;
pub mod sampler;

pub use bpe::{lang_id, Tiktoken, EXPECTED_N_VOCAB};
pub use engines::IndexTts25Engines;
pub use frontend::Frontend;
pub use gpt::generate_codes;
pub use pipeline::{build_speaker, cfm_solve, synthesize, SpeakerContext, SpkProj, SynthParams};
pub use provider::IndexTts25Provider;
pub use sampler::{GptParams, GptRng};
