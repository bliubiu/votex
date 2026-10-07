//! 应用用例：编排领域对象与仓储 trait，承载业务规则。
//!
//! 本层**不直接做 IO** —— 一切读写通过注入的仓储 trait 完成，
//! 这是 CLI 与 GUI 能共用同一套业务逻辑的前提。

pub mod model_use_case;
pub mod tts_use_case;
pub mod asr_use_case;
pub mod ocr_use_case;
pub mod pipeline_use_case;
pub mod config_use_case;
pub mod batch_tts_use_case;
pub mod batch_asr_use_case;
pub mod script_generate;
pub mod video_generate;
pub mod video_dub_use_case;
pub mod translation_use_case;
pub mod role_scan_use_case;
pub mod capability_use_case;
/// 实时/流式听写（麦克风 → 流式识别 → 增量文本）
pub mod dictation_use_case;
