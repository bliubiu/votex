//! 模型领域：实体、值对象、清单定义与仓储 trait。
//!
//! 本层**零 IO、零异步、零基础设施依赖**，清单只描述「模型是什么」，
//! 「模型在哪、怎么下载」由 infra 层负责。

pub mod value_object;
pub mod entity;
pub mod service;
pub mod registry;
pub mod capability;
