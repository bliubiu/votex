//! CLI 子命令实现：`clap` 子命令与领域用例的桥接层。
//!
//! 只做参数解析与结果格式化，业务逻辑全部下沉到 `votex-app`。

pub mod root;
pub mod model;
pub mod tts;
pub mod asr;
/// 实时听写（麦克风 → 流式识别）
pub mod dictate;
pub mod ocr;
pub mod pipeline;
pub mod batch_tts;
pub mod batch_asr;
pub mod script;
pub mod video;
pub mod dub;
pub mod translate;
pub mod config;
pub mod voice;
pub mod role;
pub mod serve;
