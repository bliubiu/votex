//! 应用层（用例编排）
//!
//! 编排领域对象与仓储 trait，承载跨实体的业务规则。
//! CLI 与 GUI 共用本层，是「两套表现层功能对等」的复用点。
//!
//! # 分层规则
//!
//! - ✅ 依赖 `votex-domain`（实体、值对象、trait）
//! - ❌ **不直接做 IO** —— 文件 / 网络 / 数据库一律通过注入的仓储 trait 完成
//! - ❌ 不依赖 `votex-cli` / `votex-gui`
//!
//! # 结构
//!
//! | 模块 | 职责 |
//! | :--- | :--- |
//! | `use_case` | 用例：TTS / ASR / OCR / 翻译 / 模型管理 / 批量任务 / 配置 |
//! | `dto` | 数据传输对象，隔离 egui / clap 类型 |
//! | `services` | 进程级共享运行时（翻译会话池、关键词翻译、脚本生成） |
//! | `bootstrap` | 装配根：`AppContext` 统一 CLI / GUI 启动流程 |
//! | `platform` | 基础设施门面：唯一允许直接依赖 `votex-infra` 的地方 |
//!
//! # 依赖注入
//!
//! 用例通过构造函数接收仓储 trait 的实现，由 [`bootstrap::AppContext`] 在启动时装配。
//! 这是本层不产生 IO 依赖的前提。
//!
//! # 分层约束（重要）
//!
//! - ✅ `platform` / `bootstrap` 可以依赖 `votex-infra`
//! - ❌ `use_case` / `services` / `dto` 中的**业务逻辑**不得直接触碰 infra
//! - ❌ 表现层（`votex-cli` / `votex-gui`）不得出现 `votex_infra::` / `rusqlite`
//!
//! 换 infra 实现时，只需改 `platform` 内部，表现层零改动。

// 测试函数使用中文语义命名（如 `解析_非法输入报错`），
// 有助于表达断言意图；仅在测试编译时豁免 snake_case 检查。
#![cfg_attr(test, allow(non_snake_case))]

pub mod use_case;
pub mod dto;
pub mod services;
pub mod bootstrap;
pub mod platform;

pub use bootstrap::{AppContext, BootstrapOptions};

