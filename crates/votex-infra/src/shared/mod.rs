//! 共享基础设施模块
//!
//! 提供跨 Provider 的通用工具，消除重复代码：
//! - `OrtSessionFactory`: 统一 ONNX Session 创建
//! - `ModelFileLocator`: 统一模型文件查找
//! - `EngineState`: 泛型引擎生命周期管理
//! - `ResourceMonitor`: 系统资源监控与内存压力分级
//! - `InferenceGate`: 推理并发闸门（防 OOM/宕机）

mod engine_state;
mod inference_gate;
mod model_locator;
mod ort_factory;
mod resource_monitor;

pub use engine_state::EngineState;
pub use inference_gate::{derive_max_permits, GatePermit, InferenceGate};
pub use model_locator::ModelFileLocator;
pub use ort_factory::{ensure_ort_dylib_path, ExecutionProvider, OrtSessionFactory};
pub use resource_monitor::{MemoryPressure, ResourceMonitor, ResourceSnapshot};
