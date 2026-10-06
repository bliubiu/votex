//! 配置领域：`AppConfig` 及其八段配置的值对象定义。
//!
//! 全部字段带 `#[serde(default)]`，避免单个字段错误导致整份配置失效。

pub mod value_object;
pub mod service;
