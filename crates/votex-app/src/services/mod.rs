//! 应用服务层
//!
//! 提供文件管理、关键词翻译、进程级共享用例等跨用例服务。

pub mod file_management;
pub mod keyword_translator;
pub mod shared_cases;
pub mod task_manager;
pub mod translation_pipeline;
pub mod translation_runtime;
