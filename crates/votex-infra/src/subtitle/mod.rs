//! 字幕基础设施：SRT / LRC 生成与断句策略。

pub mod writer;
pub mod strategy;

pub use writer::SubtitleWriter;
pub use strategy::{SubtitleStrategy, SubtitleGenerator, FastSubtitleGenerator, PreciseSubtitleGenerator};
