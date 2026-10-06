//! 领域层（DDD 最内层）
//!
//! # 硬性约束
//!
//! 本层**必须**保持纯净，违反任一条都属于架构缺陷：
//!
//! - ❌ 零 `std::fs` / `std::net` / `std::process` / 任何 IO
//! - ❌ 零 `async` / 异步运行时
//! - ❌ 不依赖 `votex-infra` / `votex-app` / ort / rusqlite / reqwest
//! - ❌ 不含 `unsafe`
//!
//! 只允许依赖：`uuid` / `thiserror` / `serde` / `chrono` / `ndarray`。
//!
//! # 结构
//!
//! | 模块 | 职责 |
//! | :--- | :--- |
//! | `error` | 统一错误体系（`thiserror`），其他领域错误由此扩展 |
//! | `repository` | 仓储 trait 定义，实现交给 infra |
//! | `provider` | 引擎适配器 trait（TTS / ASR / OCR / 翻译） |
//! | `event` | 领域事件与事件总线 trait |
//! | `model` | 模型实体、值对象、清单（registry）定义 |
//! | `tts` / `asr` / `ocr` / `translation` / `llm` | 各业务领域模型 |
//! | `pipeline` | 流水线阶段与数据契约 |
//! | `config` | `AppConfig` 八段配置值对象 |
//! | `video_material` | 视频素材与时间轴 |
//!
//! # 为什么重要
//!
//! 领域层纯净是「CLI 与 GUI 功能完全对等」的技术前提 ——
//! 两个表现层复用同一套用例与领域逻辑，差异只在入口与展示。

// 测试函数使用中文语义命名（如 `解析_非法输入报错`），
// 有助于表达断言意图；仅在测试编译时豁免 snake_case 检查。
#![cfg_attr(test, allow(non_snake_case))]

pub mod shared;
pub mod model;
pub mod tts;
pub mod asr;
pub mod ocr;
pub mod pipeline;
pub mod config;
pub mod event;
pub mod error;
pub mod repository;
pub mod provider;
pub mod llm;
pub mod video_material;
pub mod translation;
