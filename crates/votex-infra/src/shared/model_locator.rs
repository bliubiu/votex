//! 统一模型文件查找器
//!
//! 基于 Model 的 engine kind 和 name 查找模型目录和文件，
//! 消除各 Provider 中重复的 find_model_dir() 逻辑。

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::ModelKind;

/// 模型文件查找器
///
/// 统一处理模型目录和文件的查找逻辑，支持多种查找策略：
/// 1. 基于 registry 的标准路径（models/<kind_dir>/<sub_dir>/）
/// 2. 基于 model.name 的路径
/// 3. 绝对路径
pub struct ModelFileLocator;

impl ModelFileLocator {
    /// 根据 Model 查找模型目录
    ///
    /// 查找顺序：
    /// 1. `models/<kind_dir>/<model_id>/` （如 models/asr/whisper-base/）
    /// 2. `models/<kind_dir>/<engine_name>/` （如 models/asr/sensevoice/）
    /// 3. **版本变体目录**：`models/<kind_dir>/<前缀>-<版本>/`（如请求 `qwen3-tts`
    ///    而本地只有 `qwen3-tts-0.6b` / `qwen3-tts-1.7b` 时命中）
    /// 4. `models/<model.name>/`
    /// 5. `model.name` 作为绝对路径
    ///
    /// 模型根目录由 `WorkspacePaths` 统一解析，不依赖进程工作目录。
    ///
    /// # 参数
    /// - `model`: Model 实体
    ///
    /// # 返回
    /// 成功返回模型目录路径，失败返回错误
    pub fn locate_model_dir(model: &Model) -> Result<PathBuf> {
        let kind_dir = Self::kind_to_dir(model.kind);
        let engine_name = model.engine.as_str();
        let model_id = model.id.as_str();

        // 模型根目录统一解析：环境变量 → application.yml → 可执行文件目录 → cwd 向上回溯
        let models_root = crate::shared::workspace_paths::WorkspacePaths::models_dir();
        let category = models_root.join(kind_dir);

        // 1. 尝试按 model ID 查找：models/<kind_dir>/<model_id>/
        let id_path = category.join(model_id);
        if id_path.exists() && id_path.is_dir() {
            tracing::debug!("找到模型目录（ID 路径）: {:?}", id_path);
            return Ok(id_path);
        }

        // 2. 尝试按 engine 名查找：models/<kind_dir>/<engine_name>/
        let engine_path = category.join(engine_name);
        if engine_path.exists() && engine_path.is_dir() {
            tracing::debug!("找到模型目录（engine 路径）: {:?}", engine_path);
            return Ok(engine_path);
        }

        // 3. 版本变体回退：清单按版本分目录存放（qwen3-tts-0.6b / qwen3-tts-1.7b），
        //    而调用方常用无版本后缀的泛化名（qwen3-tts）。此时扫描同分类下
        //    以该名称为前缀的兄弟目录，避免要求调用方硬编码具体版本号。
        for prefix in [model_id, engine_name] {
            if let Some(dir) = Self::find_versioned_dir(&category, prefix) {
                tracing::info!(
                    "未找到精确目录 {}，回退到已下载的版本变体: {:?}",
                    prefix,
                    dir
                );
                return Ok(dir);
            }
        }

        // 4. 尝试 model.name 路径：models/<model.name>/
        let name_path = models_root.join(&model.name);
        if name_path.exists() && name_path.is_dir() {
            tracing::debug!("找到模型目录（name 路径）: {:?}", name_path);
            return Ok(name_path);
        }

        // 5. 尝试 model.name 作为绝对路径
        let abs_path = Path::new(&model.name);
        if abs_path.exists() && abs_path.is_dir() {
            tracing::debug!("找到模型目录（绝对路径）: {:?}", abs_path);
            return Ok(abs_path.to_path_buf());
        }

        anyhow::bail!(
            "未找到模型目录: {}。请将模型文件放在 models/{}/{}/ 目录下",
            model.id,
            kind_dir,
            engine_name
        )
    }

    /// 在分类目录下查找 `<前缀>-<后缀>` 形式的版本变体目录
    ///
    /// 排序后取第一个，保证结果稳定可复现；字符串排序天然让
    /// `qwen3-tts-0.6b` 优先于 `qwen3-tts-1.7b`（小模型占用更少）。
    fn find_versioned_dir(category: &Path, prefix: &str) -> Option<PathBuf> {
        // 前缀本身已不存在，无需再找变体
        if category.join(prefix).is_dir() {
            return None;
        }
        let needle = format!("{}-", prefix);
        let mut candidates: Vec<PathBuf> = std::fs::read_dir(category)
            .ok()?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_dir())
            .filter_map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                if name.starts_with(&needle) {
                    Some(e.path())
                } else {
                    None
                }
            })
            .collect();
        candidates.sort();
        candidates.into_iter().next()
    }

    /// 在模型目录中查找 ONNX 文件
    ///
    /// 按候选文件名优先级查找，如果都不存在则回退到扫描目录中任意 .onnx 文件。
    ///
    /// # 参数
    /// - `model_dir`: 模型目录路径
    /// - `candidates`: 候选文件名列表（按优先级排序）
    ///
    /// # 返回
    /// 成功返回 ONNX 文件路径，失败返回错误
    pub fn find_onnx_file(model_dir: &Path, candidates: &[&str]) -> Result<PathBuf> {
        if !model_dir.exists() {
            anyhow::bail!("模型目录不存在: {:?}", model_dir);
        }

        // 1. 按候选名优先级查找
        for name in candidates {
            let path = model_dir.join(name);
            if path.exists() {
                tracing::debug!("找到 ONNX 文件（候选名）: {:?}", path);
                return Ok(path);
            }
        }

        // 2. 回退：扫描目录中任意 .onnx 文件
        for entry in std::fs::read_dir(model_dir)
            .with_context(|| format!("读取模型目录失败: {:?}", model_dir))?
        {
            let entry = entry?;
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".onnx") {
                    let path = entry.path();
                    tracing::debug!("找到 ONNX 文件（扫描）: {:?}", path);
                    return Ok(path);
                }
            }
        }

        anyhow::bail!(
            "未找到 ONNX 模型文件: {:?}。候选文件名: {:?}",
            model_dir,
            candidates
        )
    }

    /// 在模型目录中查找多个 ONNX 文件
    ///
    /// 用于需要加载多个模型的引擎（如 PaddleOCR 的 det + cls + rec）。
    ///
    /// # 参数
    /// - `model_dir`: 模型目录路径
    /// - `required_files`: 必需的文件名列表
    ///
    /// # 返回
    /// 成功返回所有文件路径的 Vec，失败返回错误（任一文件不存在）
    pub fn find_multiple_files(model_dir: &Path, required_files: &[&str]) -> Result<Vec<PathBuf>> {
        if !model_dir.exists() {
            anyhow::bail!("模型目录不存在: {:?}", model_dir);
        }

        let mut paths = Vec::with_capacity(required_files.len());
        for name in required_files {
            let path = model_dir.join(name);
            if !path.exists() {
                anyhow::bail!("必需文件不存在: {:?}", path);
            }
            paths.push(path);
        }

        Ok(paths)
    }

    /// 查找辅助文件（如 tokens.txt、config.json 等）
    ///
    /// # 参数
    /// - `model_dir`: 模型目录路径
    /// - `filename`: 文件名
    ///
    /// # 返回
    /// 成功返回文件路径，失败返回错误
    pub fn find_auxiliary_file(model_dir: &Path, filename: &str) -> Result<PathBuf> {
        let path = model_dir.join(filename);
        if !path.exists() {
            anyhow::bail!("辅助文件不存在: {:?}（期望: {}）", model_dir, filename);
        }
        Ok(path)
    }

    /// 按候选名顺序查找第一个存在的文件
    ///
    /// 同一个模型在不同导出工具下文件名并不一致（如 Paraformer 的
    /// `model.onnx`（规范）与 `model_quant.onnx`（FunASR）），
    /// 单文件名断言会导致引擎在资产齐备的情况下仍报"模型不存在"。
    ///
    /// # 参数
    /// - `model_dir`: 模型目录路径
    /// - `candidates`: 按优先级排列的候选文件名
    ///
    /// # 返回
    /// 成功返回第一个存在的文件路径；全部缺失时错误信息会列出目录实况
    pub fn find_first_existing(model_dir: &Path, candidates: &[&str]) -> Result<PathBuf> {
        for name in candidates {
            let path = model_dir.join(name);
            if path.exists() {
                return Ok(path);
            }
        }
        anyhow::bail!(
            "在 {:?} 下未找到任何候选文件: {:?}；目录实况: {}",
            model_dir,
            candidates,
            Self::list_dir(model_dir)
        )
    }

    /// 列出目录下的文件名（供错误信息定位资产缺失）
    pub fn list_dir(dir: &Path) -> String {
        match std::fs::read_dir(dir) {
            Ok(entries) => {
                let mut names: Vec<String> = entries
                    .filter_map(|e| e.ok())
                    .filter_map(|e| e.file_name().to_str().map(|s| s.to_string()))
                    .collect();
                names.sort();
                if names.is_empty() {
                    "(空目录或不可读)".to_string()
                } else {
                    names.join(", ")
                }
            }
            Err(e) => format!("(读取失败: {})", e),
        }
    }

    /// 将 ModelKind 转换为目录名
    fn kind_to_dir(kind: ModelKind) -> &'static str {
        match kind {
            ModelKind::Tts => "tts",
            ModelKind::Asr => "asr",
            ModelKind::Ocr => "ocr",
            ModelKind::Translation => "translation",
            // 运行时依赖统一放 models/runtime/，与 WorkspacePaths::runtime_dir() 对应
            ModelKind::Runtime => "runtime",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;
    use votex_domain::model::value_object::{EngineKind, ModelId};

    fn create_test_model(id: &str, name: &str, kind: ModelKind, engine: EngineKind) -> Model {
        Model::new(ModelId::new(id), name, kind, engine)
    }

    #[test]
    fn locate_model_dir_标准路径() {
        let dir = tempdir().unwrap();
        let model_dir = dir.path().join("models").join("asr").join("sensevoice");
        fs::create_dir_all(&model_dir).unwrap();

        // 临时改变工作目录
        let _guard = crate::shared::workspace_paths::tests_support::EnvGuard::set_workspace(dir.path());

        let model = create_test_model(
            "sensevoice",
            "SenseVoice",
            ModelKind::Asr,
            EngineKind::SenseVoice,
        );
        let result = ModelFileLocator::locate_model_dir(&model);

        drop(_guard);

        assert!(result.is_ok());
        assert!(result.unwrap().ends_with("models/asr/sensevoice"));
    }

    #[test]
    fn locate_model_dir_name路径() {
        let dir = tempdir().unwrap();
        let model_dir = dir.path().join("models").join("custom-model");
        fs::create_dir_all(&model_dir).unwrap();

        let _guard = crate::shared::workspace_paths::tests_support::EnvGuard::set_workspace(dir.path());

        let model = create_test_model(
            "custom",
            "custom-model",
            ModelKind::Asr,
            EngineKind::SenseVoice,
        );
        let result = ModelFileLocator::locate_model_dir(&model);

        drop(_guard);

        assert!(result.is_ok());
        assert!(result.unwrap().ends_with("models/custom-model"));
    }

    #[test]
    fn locate_model_dir_版本变体回退() {
        // 复现 qwen3-tts 场景：清单按版本分目录，调用方用无后缀泛化名
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("models").join("tts").join("qwen3-tts-1.7b")).unwrap();
        fs::create_dir_all(dir.path().join("models").join("tts").join("qwen3-tts-0.6b")).unwrap();

        let _guard = crate::shared::workspace_paths::tests_support::EnvGuard::set_workspace(dir.path());

        let model = create_test_model("qwen3-tts", "Qwen3-TTS", ModelKind::Tts, EngineKind::Qwen3Tts);
        let result = ModelFileLocator::locate_model_dir(&model);

        drop(_guard);

        let found = result.expect("应回退到已下载的版本变体目录");
        // 排序取第一个 → 0.6b 小模型优先，行为确定
        assert!(
            found.ends_with("qwen3-tts-0.6b"),
            "应选中排序最前的变体，实际: {:?}",
            found
        );
    }

    #[test]
    fn locate_model_dir_精确目录优先于变体() {
        let dir = tempdir().unwrap();
        fs::create_dir_all(dir.path().join("models").join("tts").join("qwen3-tts-0.6b")).unwrap();
        fs::create_dir_all(dir.path().join("models").join("tts").join("qwen3-tts")).unwrap();

        let _guard = crate::shared::workspace_paths::tests_support::EnvGuard::set_workspace(dir.path());

        let model = create_test_model("qwen3-tts", "Qwen3-TTS", ModelKind::Tts, EngineKind::Qwen3Tts);
        let result = ModelFileLocator::locate_model_dir(&model);

        drop(_guard);

        let found = result.expect("应命中精确目录");
        assert!(found.ends_with("qwen3-tts"), "实际: {:?}", found);
        assert!(!found.ends_with("qwen3-tts-0.6b"));
    }

    #[test]
    fn locate_model_dir_不存在应报错() {
        let dir = tempdir().unwrap();
        // 建空的 models 分类目录：确保回溯找不到任何真实模型
        fs::create_dir_all(dir.path().join("models").join("asr")).unwrap();
        fs::create_dir_all(dir.path().join("models").join("tts")).unwrap();
        let _guard = crate::shared::workspace_paths::tests_support::EnvGuard::set_workspace(dir.path());

        // 引擎名用不存在的值，避免命中真实工作区里的同名模型
        // （如 EngineKind::SenseVoice 对应 models/asr/sensevoice）
        let model = create_test_model(
            "zzz-not-installed-engine",
            "zzz-not-installed-engine",
            ModelKind::Asr,
            EngineKind::SenseVoice,
        );
        let result = ModelFileLocator::locate_model_dir(&model);

        drop(_guard);

        let err = result.expect_err("不存在的模型目录应报错");
        // 错误信息必须给出可操作提示（模型 id + 目标路径）
        let msg = err.to_string();
        assert!(msg.contains("zzz-not-installed-engine"), "错误信息缺模型 id: {msg}");
        assert!(msg.contains("models"), "错误信息缺目标路径: {msg}");
    }

    #[test]
    fn find_onnx_file_候选名优先级() {
        let dir = tempdir().unwrap();
        // 创建两个 onnx 文件
        fs::write(dir.path().join("model.onnx"), b"fake").unwrap();
        fs::write(dir.path().join("model_quant.onnx"), b"fake").unwrap();

        // 应该优先返回 model_quant.onnx（在候选列表中排前面）
        let result = ModelFileLocator::find_onnx_file(
            dir.path(),
            &["model_quant.onnx", "model.onnx"],
        );
        assert!(result.is_ok());
        assert!(result.unwrap().ends_with("model_quant.onnx"));
    }

    #[test]
    fn find_onnx_file_回退扫描() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("custom_model.onnx"), b"fake").unwrap();

        // 候选名都不存在，应该回退到扫描
        let result = ModelFileLocator::find_onnx_file(
            dir.path(),
            &["model.onnx", "model_quant.onnx"],
        );
        assert!(result.is_ok());
        assert!(result.unwrap().ends_with("custom_model.onnx"));
    }

    #[test]
    fn find_onnx_file_目录不存在应报错() {
        let result = ModelFileLocator::find_onnx_file(
            Path::new("/nonexistent"),
            &["model.onnx"],
        );
        assert!(result.is_err());
    }

    #[test]
    fn find_multiple_files_全部存在() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("det.onnx"), b"fake").unwrap();
        fs::write(dir.path().join("cls.onnx"), b"fake").unwrap();
        fs::write(dir.path().join("rec.onnx"), b"fake").unwrap();

        let result = ModelFileLocator::find_multiple_files(
            dir.path(),
            &["det.onnx", "cls.onnx", "rec.onnx"],
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    fn find_multiple_files_缺少文件应报错() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("det.onnx"), b"fake").unwrap();
        // 缺少 cls.onnx

        let result = ModelFileLocator::find_multiple_files(
            dir.path(),
            &["det.onnx", "cls.onnx"],
        );
        assert!(result.is_err());
    }

    #[test]
    fn find_auxiliary_file_存在() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("tokens.txt"), b"fake").unwrap();

        let result = ModelFileLocator::find_auxiliary_file(dir.path(), "tokens.txt");
        assert!(result.is_ok());
    }

    #[test]
    fn find_auxiliary_file_不存在应报错() {
        let dir = tempdir().unwrap();

        let result = ModelFileLocator::find_auxiliary_file(dir.path(), "tokens.txt");
        assert!(result.is_err());
    }

    #[test]
    fn kind_to_dir_映射正确() {
        assert_eq!(ModelFileLocator::kind_to_dir(ModelKind::Tts), "tts");
        assert_eq!(ModelFileLocator::kind_to_dir(ModelKind::Asr), "asr");
        assert_eq!(ModelFileLocator::kind_to_dir(ModelKind::Ocr), "ocr");
        assert_eq!(
            ModelFileLocator::kind_to_dir(ModelKind::Translation),
            "translation"
        );
    }
}
