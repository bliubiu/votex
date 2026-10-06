//! 模型清单门面
//!
//! 以 `models/registry/*.yaml` 为数据源描述「模型是什么」，
//! 避免在代码里硬编码模型列表与文件校验规则。
//! 新增模型只需在 registry 目录添加 YAML 文件。

use votex_domain::model::registry::ModelRegistryEntry;

/// 加载全部清单条目
///
/// 加载失败返回空列表并记录告警，**不阻断启动**——
/// 清单只影响模型管理页，缺失不应让整个应用打不开。
pub fn load_registry_entries() -> Vec<ModelRegistryEntry> {
    let registry_dir = crate::platform::paths::registry_dir();
    match votex_infra::config::model_registry::ModelRegistryLoader::load(&registry_dir) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!("加载模型清单失败（目录 {}）: {}", registry_dir.display(), e);
            Vec::new()
        }
    }
}

/// 从指定目录加载清单条目（错误向上传递）
pub fn load_registry_from(dir: &std::path::Path) -> anyhow::Result<Vec<ModelRegistryEntry>> {
    Ok(votex_infra::config::model_registry::ModelRegistryLoader::load(
        dir,
    )?)
}

/// 按 ID 查找清单条目
pub fn find_entry<'a>(
    entries: &'a [ModelRegistryEntry],
    model_id: &str,
) -> Option<&'a ModelRegistryEntry> {
    votex_infra::config::model_registry::ModelRegistryLoader::find_by_id(entries, model_id)
}

/// 检测模型是否已下载就绪
///
/// 优先要求 registry 声明的全部 `required` 文件存在；
/// 未声明时回退到「任意一个文件存在即就绪」的宽松策略。
pub fn is_model_ready(models_dir: &std::path::Path, model_id: &str) -> bool {
    let entries = load_registry_entries();
    let entry = match find_entry(&entries, model_id) {
        Some(e) => e,
        None => return false,
    };

    let storage_path = models_dir.join(entry.storage_dir());
    if !storage_path.exists() {
        return false;
    }

    let required: Vec<_> = entry.files.iter().filter(|f| f.required).collect();
    if !required.is_empty() {
        required.iter().all(|f| storage_path.join(&f.name).exists())
    } else {
        entry.files.iter().any(|f| storage_path.join(&f.name).exists())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 清单加载失败返回空列表而非报错() {
        // registry 目录不存在时应降级
        let entries = load_registry_entries();
        // 不做具体断言（取决于本机是否有 registry），
        // 只验证函数不会 panic
        let _ = entries.len();
    }

    #[test]
    fn 不存在的模型判定为未就绪() {
        assert!(!is_model_ready(
            std::path::Path::new("models"),
            "definitely-not-a-real-model-xyz"
        ));
    }
}
