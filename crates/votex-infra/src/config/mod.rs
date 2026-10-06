//! 配置基础设施：YAML 加载/保存、模型清单加载。
//!
//! `ConfigLoader::load_optional` 是「配置缺失静默、解析失败报错」的入口，
//! 禁止在调用方再套 `.ok()` 把错误吞掉。

pub mod loader;
pub mod model_registry;
