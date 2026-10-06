//! 数据传输对象：表现层与用例层之间的数据契约。
//!
//! 与领域实体分离，避免 egui / clap 类型渗入领域层。

pub mod tts_dto;
pub mod asr_dto;
pub mod ocr_dto;
pub mod pipeline_dto;
pub mod translation_dto;
