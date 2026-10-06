//! 模型状态检测工具
//!
//! 以 `models/registry/*.yaml` 为数据源（模型清单），
//! 自动检测本地模型文件是否存在，无需硬编码模型列表和校验规则。
//! 新增模型只需在 `models/registry/` 下添加 YAML 文件即可。
//!
//! # 分层
//!
//! 本模块负责「读一次全局清单 + 解析路径」，判定规则本身在
//! [`crate::model_ready`]——那里是纯函数，可被任意清单驱动测试，
//! 不受本模块 `OnceLock` 全局单例的限制。

use std::path::Path;
use std::sync::OnceLock;
use votex_domain::model::registry::ModelRegistryEntry;
use votex_app::platform::registry;

use crate::model_ready::is_ready_with;

fn load_registry() -> &'static Vec<ModelRegistryEntry> {
    static REGISTRY: OnceLock<Vec<ModelRegistryEntry>> = OnceLock::new();
    REGISTRY.get_or_init(registry::load_registry_entries)
}

/// 检测指定模型是否已下载就绪
pub fn is_model_ready(models_dir: &Path, _model_kind: &str, model_id: &str) -> bool {
    is_ready_with(load_registry(), models_dir, model_id)
}

/// 获取默认模型目录
pub fn default_models_dir() -> std::path::PathBuf {
    votex_app::platform::paths::models_dir()
}

/// 模型定义信息（从 registry 动态加载，非硬编码）
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub engine: String,
}

/// 从模型清单加载所有已知模型列表
pub fn all_models() -> Vec<ModelInfo> {
    load_registry()
        .iter()
        .map(|e| ModelInfo {
            id: e.id.clone(),
            name: e.name.clone(),
            kind: e.kind.clone(),
            engine: e.engine.clone(),
        })
        .collect()
}

/// 创建关键词翻译服务
pub fn create_keyword_translator() -> votex_app::services::keyword_translator::KeywordTranslator {
    votex_app::services::keyword_translator::KeywordTranslator::new(None)
}
