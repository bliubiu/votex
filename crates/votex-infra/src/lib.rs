pub mod config;
pub mod persistence;
pub mod download;
pub mod encoding;
pub mod event;
pub mod logging;
pub mod inference;
pub mod shared;
pub mod tts;
pub mod asr;
pub mod ocr;
pub mod audio;
pub mod document;
pub mod subtitle;
pub mod api;
pub mod security;
pub mod llm;
pub mod video;
pub mod translation;
pub mod bpe_tokenizer;
pub mod provider;
pub mod pipeline;
pub mod tokenizer;

/// 辅助：将字符串转为 DomainError::Config
pub fn config_err(msg: &str) -> votex_domain::error::ConfigError {
    votex_domain::error::ConfigError::ReadFailed(msg.to_string())
}
