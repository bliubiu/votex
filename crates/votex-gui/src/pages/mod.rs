//! GUI 页面：每个页面对应一个功能区。
//!
//! egui 为立即模式：**禁止在 UI 线程做重型推理或 IO**，
//! 所有耗时任务经 `task_runner` 派发到 `std::thread`（不引入 tokio，
//! 避免异步运行时与 ort 的线程模型冲突）。

pub mod dashboard;
pub mod tts_page;
pub mod asr_page;
/// 实时听写（麦克风 → 流式识别）
pub mod dictation_page;
pub mod voices_page;
pub mod ocr_page;
pub mod translation_page;
pub mod batch_page;
pub mod pipeline_page;
pub mod video_page;
pub mod settings_page;
pub mod onboarding;
pub mod page_template;
