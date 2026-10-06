//! TTS 领域：音色、音频参数、合成任务与 `TtsProvider` trait。
//!
//! `TtsProvider::load` 接收 `&mut self` 而非 `&self`，导致每个适配器都要
//! 手写 `Mutex<Option<Session>>`；这是 infra 层重复代码的主要根因。

pub mod value_object;
pub mod entity;
pub mod service;
pub mod provider;
pub mod dialect;
pub mod tokenizer;
pub mod number;
pub mod chapter;
pub mod pause;
pub mod role;

pub use tokenizer::{TextTokenizer, TokenizerError};
