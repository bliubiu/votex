use anyhow::Result;
use std::path::{Path, PathBuf};
use votex_domain::config::value_object::DownloadConfig;
use votex_domain::model::entity::Model;
use votex_domain::model::registry::ModelRegistryEntry;
use votex_domain::model::value_object::{
    DownloadProgress, EngineKind, ModelId, ModelKind, ModelStatus,
};
use votex_infra::download::downloader::Downloader;
use votex_infra::download::mirror_resolver::MirrorResolver;

/// 模型管理用例
///
/// 基于模型清单（registry）驱动，替代硬编码的 match 块。
pub struct ModelUseCase {
    models_dir: PathBuf,
    registry: Vec<ModelRegistryEntry>,
    /// 下载配置（超时 / 断点续传）
    ///
    /// 旧实现下载时用 `Downloader` 的硬编码超时，
    /// `application.yml` 的 `models.download.timeout` 与 `resume` 均无消费者。
    download_config: DownloadConfig,
}

/// 进程级默认下载配置
///
/// 由 CLI / GUI 启动时从 `application.yml` 写入，供未显式传配置的
/// `ModelUseCase::new` 使用。避免为传递两项配置而改动全部 10 处构造点。
static GLOBAL_DOWNLOAD_CONFIG: std::sync::OnceLock<DownloadConfig> = std::sync::OnceLock::new();

/// 设置进程级默认下载配置
///
/// 应在应用启动时（读取 `application.yml` 之后）调用一次。
/// 后设置者不生效（`OnceLock` 语义），符合「配置只读一次」的预期。
pub fn set_global_download_config(config: DownloadConfig) {
    let timeout = config.timeout;
    if GLOBAL_DOWNLOAD_CONFIG.set(config).is_err() {
        tracing::debug!("下载配置已初始化，忽略后续设置（timeout={timeout}）");
    } else {
        tracing::info!("下载配置已生效: timeout={}s, resume={}", timeout, GLOBAL_DOWNLOAD_CONFIG.get().map(|c| c.resume).unwrap_or(true));
    }
}

/// 获取进程级默认下载配置
pub fn global_download_config() -> DownloadConfig {
    GLOBAL_DOWNLOAD_CONFIG
        .get()
        .cloned()
        .unwrap_or_default()
}

impl ModelUseCase {
    /// 以默认下载配置构造
    ///
    /// 若此前调用过 [`set_global_download_config`]，则使用该配置；
    /// 否则退回 `DownloadConfig::default()`。
    pub fn new(models_dir: &Path, registry: Vec<ModelRegistryEntry>) -> Self {
        Self::with_config(models_dir, registry, global_download_config())
    }

    /// 指定下载配置构造
    pub fn with_config(
        models_dir: &Path,
        registry: Vec<ModelRegistryEntry>,
        download_config: DownloadConfig,
    ) -> Self {
        Self {
            models_dir: models_dir.to_path_buf(),
            registry,
            download_config,
        }
    }

    /// 获取当前注册表引用
    pub fn registry(&self) -> &[ModelRegistryEntry] {
        &self.registry
    }

    /// 下载模型：解析 URL → 下载 → 校验
    pub fn download_model(
        &self,
        model_id: &ModelId,
        priority: &[String],
        on_progress: Option<votex_infra::download::downloader::ProgressFn>,
    ) -> Result<Model> {
        let entry = self.find_entry(model_id)?;
        let mut model = Self::create_model_from_entry(entry)?;

        let files = MirrorResolver::resolve(entry, priority);
        if files.is_empty() {
            anyhow::bail!("未找到模型 {} 的下载源", model_id);
        }

        model.status = ModelStatus::Downloading(DownloadProgress {
            downloaded_bytes: 0,
            total_bytes: 0,
            source_name: String::new(),
        });

        // 按配置构造下载器：超时与断点续传开关来自 application.yml
        let downloader = Downloader::with_config(&self.download_config)?;
        let downloaded_files = downloader.download_files(&files, &self.models_dir, on_progress)?;

        for path in &downloaded_files {
            model.file_paths.push(path.clone());
        }

        model.status = ModelStatus::Ready;
        tracing::info!("模型 {} 下载完成", model_id);
        Ok(model)
    }

    /// 从本地路径导入模型
    pub fn import_model(&self, model_id: &ModelId, path: &Path) -> Result<Model> {
        if !path.exists() {
            anyhow::bail!("文件不存在: {:?}", path);
        }

        let entry = self.find_entry(model_id)?;
        let mut model = Self::create_model_from_entry(entry)?;

        let dest = self.model_dest_dir(entry).join(
            path.file_name()
                .unwrap_or_else(|| std::ffi::OsStr::new("model.onnx")),
        );
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(path, &dest)?;

        model.file_paths.push(dest);
        model.status = ModelStatus::Ready;
        tracing::info!("模型 {} 导入成功", model_id);
        Ok(model)
    }

    /// 加载模型到内存
    pub fn load_model(&self, model: &mut Model) -> Result<()> {
        if !model.is_ready() {
            anyhow::bail!("模型未就绪，无法加载");
        }

        model.status = ModelStatus::Loading;

        let model_dir = self.models_dir.join(model.id.as_str());

        let result = match model.engine {
            EngineKind::Kokoro
            | EngineKind::IndexTTS25
            | EngineKind::PaddleOCR
            | EngineKind::SenseVoice
            | EngineKind::Paraformer
            | EngineKind::Qwen3Asr
            | EngineKind::EasyOcr
            | EngineKind::Whisper => {
                votex_infra::inference::EngineLoader::load(&model_dir, model.engine)
            }
            _ => {
                anyhow::bail!("不支持的引擎: {:?}", model.engine)
            }
        };

        match result {
            Ok(()) => {
                model.status = ModelStatus::Loaded;
                tracing::info!("模型 {} 加载成功", model.id);
                Ok(())
            }
            Err(e) => {
                model.status = ModelStatus::LoadFailed(e.to_string());
                anyhow::bail!("模型 {} 加载失败: {}", model.id, e);
            }
        }
    }

    /// 释放模型资源（真实释放引擎会话）
    ///
    /// 引擎会话懒加载于进程级单例（shared_tts/shared_asr/shared_ocr），
    /// 此前只把状态改回 Ready、不释放任何会话——GUI 显示「已释放」但内存不降。
    /// unload 后下次合成/识别会自动重新加载模型。
    pub fn unload_model(&self, model: &mut Model) -> Result<()> {
        if !model.is_loaded() {
            anyhow::bail!("模型未加载");
        }

        let release = match model.engine {
            EngineKind::Kokoro
            | EngineKind::Qwen3Tts
            | EngineKind::CosyVoice3
            | EngineKind::IndexTTS25 => {
                crate::services::shared_cases::shared_tts().unload(model.engine)
            }
            EngineKind::Whisper
            | EngineKind::SenseVoice
            | EngineKind::Paraformer
            | EngineKind::Qwen3Asr
            | EngineKind::FireRedAsr
            | EngineKind::WeNet => {
                crate::services::shared_cases::shared_asr().unload(model.engine)
            }
            EngineKind::PaddleOCR | EngineKind::EasyOcr => {
                crate::services::shared_cases::shared_ocr().unload_engine()
            }
            // 翻译/LLM 等引擎由 translation_runtime::release_all 统一管理
            _ => Ok(()),
        };
        release?;

        model.status = ModelStatus::Ready;
        tracing::info!("模型 {} 已释放（引擎会话已卸载）", model.id);
        Ok(())
    }

    /// 列出所有可用模型，自动检测本地文件状态
    ///
    /// 注意：registry 条目若 `kind`/`engine` 字段无法解析，此前会被 `filter_map` 的
    /// `.ok()?` **静默丢弃**——`model list` 退出码仍为 0，只是少列几行，用户无从察觉。
    /// 现改为记录 WARN 后跳过，让配置错误在日志中可见（docs/20 F54）。
    pub fn list_models(&self) -> Vec<Model> {
        self.registry
            .iter()
            .filter_map(|entry| {
                let mut model = match Self::create_model_from_entry(entry) {
                    Ok(m) => m,
                    Err(e) => {
                        tracing::warn!(
                            "模型清单条目无法解析，已从列表中跳过: id={} 原因={}",
                            entry.id,
                            e
                        );
                        return None;
                    }
                };
                let model_dir = self.models_dir.join(entry.storage_dir());
                if Self::check_model_files_exist(&model_dir, entry) {
                    model.status = ModelStatus::Ready;
                }
                Some(model)
            })
            .collect()
    }

    // ─── 内部方法 ─────────────────────────────────────────

    /// 根据 ModelId 查找注册表条目
    fn find_entry(&self, model_id: &ModelId) -> Result<&ModelRegistryEntry> {
        self.registry
            .iter()
            .find(|e| e.id == model_id.as_str())
            .ok_or_else(|| anyhow::anyhow!("未知模型: {}", model_id))
    }

    /// 检测模型文件是否已存在于本地
    fn check_model_files_exist(model_dir: &Path, entry: &ModelRegistryEntry) -> bool {
        if !model_dir.exists() {
            return false;
        }
        entry
            .files
            .iter()
            .all(|f| model_dir.join(&f.name).exists())
    }

    /// 从注册表条目创建模型实体
    fn create_model_from_entry(entry: &ModelRegistryEntry) -> Result<Model> {
        let kind = parse_model_kind(&entry.kind)?;
        let engine = parse_engine_kind(&entry.engine)?;
        Ok(Model::new(ModelId::new(&entry.id), &entry.name, kind, engine))
    }

    /// 获取模型存储目录（sub_dir 感知）
    fn model_dest_dir(&self, entry: &ModelRegistryEntry) -> PathBuf {
        self.models_dir.join(entry.storage_dir())
    }
}

// ─── 辅助函数 ─────────────────────────────────────────

/// 将字符串转为 ModelKind
fn parse_model_kind(s: &str) -> Result<ModelKind> {
    match s {
        "Tts" => Ok(ModelKind::Tts),
        "Asr" => Ok(ModelKind::Asr),
        "Ocr" => Ok(ModelKind::Ocr),
        "Translation" => Ok(ModelKind::Translation),
        // 运行时依赖（ONNX Runtime 动态库）由 registry 统一管理下载与校验，
        // 但它不是可 `load()` 的模型，因此只出现在列表中而不参与引擎分发
        "Runtime" => Ok(ModelKind::Runtime),
        _ => anyhow::bail!("无效的模型类型: {}", s),
    }
}

/// 将字符串转为 EngineKind
///
/// 注意：这里硬编码了一份 `EngineKind` 变体名 → 变体的映射表，
/// 与 `EngineKind::from_str`（`votex_domain::model::value_object`）**重复且曾长期不完整**——
/// 原先漏掉 `M2m100` / `HyMt1_5` 等 8 个已存在的变体，导致 registry 中
/// `m2m-100-418m` / `hy-mt-1.5` / `cosyvoice` 三条条目被 `list_models` 的
/// `filter_map(... .ok()?)` **静默丢弃**：`model list` 退出码仍为 0、只少列 3 行，
/// 用户完全无从察觉（见 docs/20 F54）。
///
/// 保留本函数而非直接调 `from_str`：本表用 PascalCase 变体名（`CosyVoice3`），
/// 而 `from_str` 收的是 CLI 用的 kebab-case（`cosyvoice3`），两者输入域不同。
/// 新增 `EngineKind` 变体时**必须同步这里**。
fn parse_engine_kind(s: &str) -> Result<EngineKind> {
    match s {
        "Kokoro" => Ok(EngineKind::Kokoro),
        // 迁移别名：IndexTTS2 已移除，历史持久化串归一到 IndexTTS25
        "IndexTTS2" => Ok(EngineKind::IndexTTS25),
        "IndexTTS25" => Ok(EngineKind::IndexTTS25),
        "Whisper" => Ok(EngineKind::Whisper),
        "SenseVoice" => Ok(EngineKind::SenseVoice),
        "Paraformer" => Ok(EngineKind::Paraformer),
        "Qwen3Asr" => Ok(EngineKind::Qwen3Asr),
        "PaddleOCR" => Ok(EngineKind::PaddleOCR),
        "EasyOcr" => Ok(EngineKind::EasyOcr),
        "WeNet" => Ok(EngineKind::WeNet),
        "CosyVoice3" => Ok(EngineKind::CosyVoice3),
        "FireRedAsr" => Ok(EngineKind::FireRedAsr),
        "Qwen3Tts" => Ok(EngineKind::Qwen3Tts),
        "OpusMt" => Ok(EngineKind::OpusMt),
        "QwenMt" => Ok(EngineKind::QwenMt),
        "Nllb" => Ok(EngineKind::Nllb),
        // ===== 以下为 v1.2 补齐（F54）：变体早已存在，但本表遗漏导致条目被静默丢弃 =====
        "M2m100" => Ok(EngineKind::M2m100),
        "HyMt1_5" => Ok(EngineKind::HyMt1_5),
        "CTranslate2" => Ok(EngineKind::CTranslate2),
        "OpenAi" => Ok(EngineKind::OpenAi),
        "Ollama" => Ok(EngineKind::Ollama),
        "Pexels" => Ok(EngineKind::Pexels),
        "Pixabay" => Ok(EngineKind::Pixabay),
        "Coverr" => Ok(EngineKind::Coverr),
        // ===== 推理运行时 =====
        "OnnxRuntime" => Ok(EngineKind::OnnxRuntime),
        _ => anyhow::bail!("无效的引擎类型: {}", s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_registry() -> Vec<ModelRegistryEntry> {
        vec![
            ModelRegistryEntry {
                id: "kokoro-82m".into(),
                name: "Kokoro-82M".into(),
                kind: "Tts".into(),
                engine: "Kokoro".into(),
                sub_dir: None,
                tokenizer: None,
                files: vec![
                    file_entry("kokoro-v1.0.int8.onnx", &[("github", "https://github.com/test")]),
                    file_entry("config.json", &[("huggingface", "https://hf.co/test")]),
                ],
                    ..Default::default()
            },
            ModelRegistryEntry {
                id: "paddleocr-v6-tiny".into(),
                name: "PaddleOCR v6 Tiny".into(),
                kind: "Ocr".into(),
                engine: "PaddleOCR".into(),
                sub_dir: Some("paddleocr-v6".into()),
                tokenizer: None,
                files: vec![
                    file_entry("ch_PP-OCRv6_det_tiny.onnx", &[("huggingface", "https://hf.co/test")]),
                ],
                    ..Default::default()
            },
        ]
    }

    fn file_entry(name: &str, sources: &[(&str, &str)]) -> votex_domain::model::registry::ModelFileEntry {
        let mut map = HashMap::new();
        for (k, v) in sources {
            map.insert(k.to_string(), v.to_string());
        }
        votex_domain::model::registry::ModelFileEntry {
            name: name.into(),
            sha256: None,
            size: None,
            required: false,
            sources: map,
                ..Default::default()
        }
    }

    #[test]
    fn model_use_case_list_models() {
        let dir = tempfile::tempdir().unwrap();
        let registry = make_registry();
        let use_case = ModelUseCase::new(dir.path(), registry);

        let models = use_case.list_models();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].id.as_str(), "kokoro-82m");
        assert_eq!(models[1].id.as_str(), "paddleocr-v6-tiny");
        // 文件不存在，状态应为 NotDownloaded
        assert!(!models[0].is_ready());
    }

    #[test]
    fn model_use_case_list_models_检测已下载文件() {
        let dir = tempfile::tempdir().unwrap();
        // 注意：storage_dir 会按 kind 分目录（Tts → tts/）
        let model_dir = dir.path().join("tts").join("kokoro-82m");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("kokoro-v1.0.int8.onnx"), "data").unwrap();
        std::fs::write(model_dir.join("config.json"), "{}").unwrap();

        let registry = make_registry();
        let use_case = ModelUseCase::new(dir.path(), registry);

        let models = use_case.list_models();
        let kokoro = models.iter().find(|m| m.id.as_str() == "kokoro-82m").unwrap();
        assert!(kokoro.is_ready());
    }

    #[test]
    fn model_use_case_list_models_sub_dir检测() {
        let dir = tempfile::tempdir().unwrap();
        // 文件在 ocr/paddleocr-v6 子目录下（sub_dir 生效，kind 为 Ocr）
        let model_dir = dir.path().join("ocr").join("paddleocr-v6");
        std::fs::create_dir_all(&model_dir).unwrap();
        std::fs::write(model_dir.join("ch_PP-OCRv6_det_tiny.onnx"), "data").unwrap();

        let registry = make_registry();
        let use_case = ModelUseCase::new(dir.path(), registry);

        let models = use_case.list_models();
        let ocr = models.iter().find(|m| m.id.as_str() == "paddleocr-v6-tiny").unwrap();
        assert!(ocr.is_ready());
    }

    #[test]
    fn model_use_case_find_entry_未知模型() {
        let dir = tempfile::tempdir().unwrap();
        let registry = make_registry();
        let use_case = ModelUseCase::new(dir.path(), registry);

        let result = use_case.find_entry(&ModelId::new("non-existent"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("未知模型"));
    }

    #[test]
    fn model_use_case_find_entry_存在() {
        let dir = tempfile::tempdir().unwrap();
        let registry = make_registry();
        let use_case = ModelUseCase::new(dir.path(), registry);

        let entry = use_case.find_entry(&ModelId::new("kokoro-82m")).unwrap();
        assert_eq!(entry.name, "Kokoro-82M");
        assert_eq!(entry.storage_dir(), "tts/kokoro-82m");
    }

    #[test]
    fn model_use_case_create_model_from_entry() {
        let entry = ModelRegistryEntry {
            id: "sensevoice".into(),
            name: "SenseVoice".into(),
            kind: "Asr".into(),
            engine: "SenseVoice".into(),
            sub_dir: None,
                tokenizer: None,
            files: vec![],
                ..Default::default()
        };

        let model = ModelUseCase::create_model_from_entry(&entry).unwrap();
        assert_eq!(model.id.as_str(), "sensevoice");
        assert_eq!(model.name, "SenseVoice");
        assert_eq!(model.kind, ModelKind::Asr);
        assert_eq!(model.engine, EngineKind::SenseVoice);
    }

    #[test]
    fn model_use_case_invalid_kind_should_fail() {
        let entry = ModelRegistryEntry {
            id: "bad".into(),
            name: "Bad".into(),
            kind: "InvalidKind".into(),
            engine: "Kokoro".into(),
            sub_dir: None,
                tokenizer: None,
            files: vec![],
                ..Default::default()
        };

        let result = ModelUseCase::create_model_from_entry(&entry);
        assert!(result.is_err());
    }

    #[test]
    fn model_use_case_invalid_engine_should_fail() {
        let entry = ModelRegistryEntry {
            id: "bad".into(),
            name: "Bad".into(),
            kind: "Tts".into(),
            engine: "NonExistentEngine".into(),
            sub_dir: None,
                tokenizer: None,
            files: vec![],
                ..Default::default()
        };

        let result = ModelUseCase::create_model_from_entry(&entry);
        assert!(result.is_err());
    }

    #[test]
    fn model_use_case_import_model_uses_storage_dir() {
        let dir = tempfile::tempdir().unwrap();
        let registry = make_registry();
        let use_case = ModelUseCase::new(dir.path(), registry);

        // 创建源文件
        let src = dir.path().join("source_model.onnx");
        std::fs::write(&src, "model data").unwrap();

        let model = use_case
            .import_model(&ModelId::new("paddleocr-v6-tiny"), &src)
            .unwrap();
        assert!(model.is_ready());

        // 文件应导入到 ocr/paddleocr-v6 子目录下
        let expected = dir
            .path()
            .join("ocr")
            .join("paddleocr-v6")
            .join("source_model.onnx");
        assert!(expected.exists());
    }
}
