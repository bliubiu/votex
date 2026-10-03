pub mod writer;
pub mod strategy;

pub use writer::SubtitleWriter;
pub use strategy::{SubtitleStrategy, SubtitleGenerator, FastSubtitleGenerator, PreciseSubtitleGenerator};
