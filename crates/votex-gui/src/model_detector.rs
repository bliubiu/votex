//! 模型状态检测工具
//!
//! 以 `models/registry/*.yaml` 为数据源（模型清单），
//! 自动检测本地模型文件是否存在，无需硬编码模型列表和校验规则。
//! 新增模型只需在 `models/registry/` 下添加 YAML 文件即可。

use std::path::Path;
use std::sync::OnceLock;
use votex_domain::model::registry::ModelRegistryEntry;
use votex_infra::config::model_registry::ModelRegistryLoader;

fn load_registry() -> &'static Vec<ModelRegistryEntry> {
    static REGISTRY: OnceLock<Vec<ModelRegistryEntry>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let registry_dir = std::env::current_dir()
            .unwrap_or_default()
            .join("models")
            .join("registry");
        match ModelRegistryLoader::load(&registry_dir) {
            Ok(entries) => entries,
            Err(e) => {
                tracing::warn!("加载模型清单失败: {}", e);
                Vec::new()
            }
        }
    })
}

/// 检测指定模型是否已下载就绪
pub fn is_model_ready(models_dir: &Path, _model_kind: &str, model_id: &str) -> bool {
    let entry = load_registry().iter().find(|e| e.id == model_id);
    let entry = match entry {
        Some(e) => e,
        None => return false,
    };

    let storage_path = models_dir.join(entry.storage_dir());
    if !storage_path.exists() {
        return false;
    }

    // 如果 registry 中声明了 required 文件，要求所有 required 文件必须存在；
    // 否则回退到宽松策略（任意一个文件存在即视为就绪）
    let required: Vec<_> = entry.files.iter().filter(|f| f.required).collect();
    if !required.is_empty() {
        required.iter().all(|f| storage_path.join(&f.name).exists())
    } else {
        entry.files.iter().any(|f| storage_path.join(&f.name).exists())
    }
}

/// 获取默认模型目录
pub fn default_models_dir() -> std::path::PathBuf {
    std::env::current_dir()
        .unwrap_or_default()
        .join("models")
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
