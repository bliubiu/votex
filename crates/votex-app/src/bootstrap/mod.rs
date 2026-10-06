//! 应用装配入口（Composition Root）
//!
//! # 为什么需要这个模块
//!
//! 此前 CLI 与 GUI 各自写了一遍启动流程：
//!
//! - 定位 ONNX Runtime 动态库
//! - 加载 `application.yml`
//! - 初始化日志、执行提供器、推理资源限制
//! - 下发下载配置（超时 / 断点续传）
//! - 配置资源治理阈值与推理并发闸门
//! - 初始化翻译运行时
//! - 打开 SQLite 并构造各仓储
//!
//! 两份实现逐渐漂移：GUI 侧长期缺下载配置与 ORT 初始化时机，
//! 且两个 crate 都被迫直接依赖 `votex-infra` 与 `rusqlite`，
//! 表现层里到处是 `votex_infra::persistence::xxx::SqliteXxxRepository` 这类
//! **基础设施类型泄漏**。
//!
//! 本模块把这些装配动作收敛到一处，并让表现层只持有
//! **领域仓储 trait object**，从而：
//!
//! - ✅ 装配逻辑单点维护，CLI / GUI 天然对等
//! - ✅ 表现层不再依赖 `votex-infra` / `rusqlite`
//! - ✅ 数据库不可用时降级为 `None`，功能自动关闭而非崩溃
//!
//! # 使用方式
//!
//! ```no_run
//! use votex_app::bootstrap::{AppContext, BootstrapOptions};
//!
//! # fn main() -> anyhow::Result<()> {
//! let opts = BootstrapOptions::new("application.yml");
//! let ctx = AppContext::bootstrap(opts)?;
//! println!("模型目录: {}", ctx.models_dir().display());
//! # Ok(())
//! # }
//! ```
//!
//! # 顺序约束（重要）
//!
//! [`AppContext::bootstrap`] 内部顺序不可随意调整：
//!
//! 1. **ORT 动态库定位**必须最先执行。`ort` 会在首次调用时尝试按默认名
//!    `onnxruntime.dll` 加载，并把失败结果缓存进内部 `OnceLock`。
//!    一旦缓存失败，后续再设置 `ORT_DYLIB_PATH` 也无效。
//! 2. **配置加载**先于日志初始化，因为日志级别本身来自配置。
//! 3. **翻译运行时**先于任何翻译调用，否则退化为每次重新加载数 GB 模型。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use votex_domain::config::value_object::AppConfig;
use votex_domain::repository::{
    DownloadRepository, OcrTaskRepository, PipelineRepository, PlaybackRepository,
};

/// 装配选项
#[derive(Debug, Clone)]
pub struct BootstrapOptions {
    /// `application.yml` 路径
    pub config_path: PathBuf,
    /// 模型根目录覆盖值。
    ///
    /// - `None`：走统一解析（环境变量 → 配置 → 可执行文件目录 → cwd 回溯）
    /// - `Some(p)`：显式指定，`--models-dir` 参数走这里
    pub models_dir_override: Option<PathBuf>,
    /// 日志级别覆盖（`--verbose` → `debug`）
    pub log_level_override: Option<String>,
    /// 数据库文件路径；`None` 时不打开数据库（纯内存运行）
    pub db_path: Option<PathBuf>,
}

impl BootstrapOptions {
    /// 以配置文件路径构造，其余项使用默认值
    pub fn new(config_path: impl Into<PathBuf>) -> Self {
        Self {
            config_path: config_path.into(),
            models_dir_override: None,
            log_level_override: None,
            db_path: Some(PathBuf::from("data/votex.db")),
        }
    }

    /// 覆盖模型目录
    pub fn with_models_dir(mut self, dir: Option<PathBuf>) -> Self {
        self.models_dir_override = dir;
        self
    }

    /// 覆盖日志级别
    pub fn with_log_level(mut self, level: Option<String>) -> Self {
        self.log_level_override = level;
        self
    }

    /// 覆盖数据库路径
    pub fn with_db_path(mut self, path: Option<PathBuf>) -> Self {
        self.db_path = path;
        self
    }
}

/// 应用上下文 —— 进程级装配结果
///
/// 持有配置、路径与各仓储的 trait object。
/// CLI 与 GUI 共享同一份装配代码，保证「同一套领域层，两套表现层」。
#[derive(Clone)]
pub struct AppContext {
    config: Option<AppConfig>,
    config_path: PathBuf,
    models_dir: PathBuf,
    download_repo: Option<Arc<dyn DownloadRepository>>,
    pipeline_repo: Option<Arc<dyn PipelineRepository>>,
    ocr_repo: Option<Arc<dyn OcrTaskRepository>>,
    playback_repo: Option<Arc<dyn PlaybackRepository>>,
    /// 数据库是否可用。`false` 时所有仓储均为 `None`
    pub database_available: bool,
}

impl AppContext {
    /// 执行完整装配流程
    ///
    /// 该函数**幂等前提**：进程内只能安全调用一次
    /// （ORT 路径与各全局单例只生效一次）。
    pub fn bootstrap(opts: BootstrapOptions) -> Result<Self> {
        // ── 1. ONNX Runtime 动态库（必须最先）────────────────────
        crate::platform::runtime::ensure_runtime_library();

        // ── 2. 加载配置 ────────────────────────────────────────
        let config = crate::platform::config::load_config_optional(&opts.config_path);

        // 配置中的明文敏感字段落盘加密（密钥初始化失败不阻断启动）
        if config.is_some() {
            crate::platform::config::encrypt_config_secrets(&opts.config_path);
        }

        // ── 3. 日志 ────────────────────────────────────────────
        let log_level = opts
            .log_level_override
            .clone()
            .or_else(|| config.as_ref().map(|c| c.log.level.clone()))
            .unwrap_or_else(|| "info".to_string());
        let log_dir = config
            .as_ref()
            .map(|c| c.log.dir.clone())
            .unwrap_or_else(|| "logs".to_string());
        let log_file_enabled = config.as_ref().map(|c| c.log.file_enabled).unwrap_or(true);
        crate::platform::runtime::init_logging(&log_dir, &log_level, log_file_enabled)?;

        // ── 4. 执行提供器与推理资源限制 ──────────────────────────
        if let Some(cfg) = config.as_ref() {
            crate::platform::runtime::configure_execution_provider(&cfg.inference.execution_provider);
            crate::platform::runtime::apply_inference_config(&cfg.inference);
        }

        // ── 5. 资源治理（内存阈值 + 推理并发闸门）──────────────────
        if let Some(cfg) = config.as_ref() {
            crate::platform::runtime::configure_resource_governance(&cfg.resource);
        }

        // ── 6. 模型目录 ────────────────────────────────────────
        let models_dir = resolve_models_dir(opts.models_dir_override.as_deref(), config.as_ref());

        // ── 7. 下载配置（超时 / 断点续传）────────────────────────
        // 旧实现这两项配置在 yml 中存在但无消费者，行为完全由硬编码决定
        if let Some(cfg) = config.as_ref() {
            crate::use_case::model_use_case::set_global_download_config(cfg.models.download.clone());
        }

        // ── 8. 翻译运行时（会话池 + 缓存）────────────────────────
        crate::services::translation_runtime::init(models_dir.clone());
        if let Some(cfg) = config.as_ref() {
            crate::services::translation_runtime::init_cache(
                if cfg.translation.enable_cache {
                    cfg.translation.cache_capacity
                } else {
                    0
                },
            );
        }

        // ── 9. 数据库与仓储 ────────────────────────────────────
        let (db_available, download_repo, pipeline_repo, ocr_repo, playback_repo) =
            match opts.db_path.as_ref() {
                Some(db_path) => open_repositories(db_path),
                None => (false, None, None, None, None),
            };

        Ok(Self {
            config,
            config_path: opts.config_path,
            models_dir,
            download_repo,
            pipeline_repo,
            ocr_repo,
            playback_repo,
            database_available: db_available,
        })
    }

    /// 已加载的配置（配置文件不存在或解析失败时为 `None`）
    pub fn config(&self) -> Option<&AppConfig> {
        self.config.as_ref()
    }

    /// 配置文件路径
    pub fn config_path(&self) -> &Path {
        &self.config_path
    }

    /// 模型根目录
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// 下载进度仓储（数据库不可用时为 `None`）
    pub fn download_repo(&self) -> Option<&Arc<dyn DownloadRepository>> {
        self.download_repo.as_ref()
    }

    /// 流水线仓储（数据库不可用时为 `None`）
    pub fn pipeline_repo(&self) -> Option<&Arc<dyn PipelineRepository>> {
        self.pipeline_repo.as_ref()
    }

    /// OCR 任务仓储（数据库不可用时为 `None`）
    pub fn ocr_repo(&self) -> Option<&Arc<dyn OcrTaskRepository>> {
        self.ocr_repo.as_ref()
    }

    /// 播放进度仓储（数据库不可用时为 `None`）
    pub fn playback_repo(&self) -> Option<&Arc<dyn PlaybackRepository>> {
        self.playback_repo.as_ref()
    }

    /// 模型清单（registry）条目
    ///
    /// 加载失败时返回空列表并记录告警，不阻断启动 ——
    /// 模型清单只影响「模型管理」页面，缺失不应让整个应用打不开。
    pub fn registry_entries(&self) -> Vec<votex_domain::model::registry::ModelRegistryEntry> {
        crate::platform::registry::load_registry_entries()
    }

    /// 供 GUI 直接落地的上下文（表现层不再看到 infra 类型）
    pub fn gui_handles(&self) -> GuiHandles {
        GuiHandles {
            models_dir: self.models_dir.display().to_string(),
            pipeline_repo: self.pipeline_repo.clone(),
            download_repo: self.download_repo.clone(),
            config_path: self.config_path.display().to_string(),
        }
    }
}

/// GUI 启动所需的句柄集合
///
/// 从 [`AppContext`] 派生，字段类型全部是 domain trait object，
/// 因此 `votex-gui` 无需 `use votex_infra::...`。
#[derive(Clone)]
pub struct GuiHandles {
    /// 模型根目录（字符串形式，便于放入 GUI 状态）
    pub models_dir: String,
    /// 流水线仓储
    pub pipeline_repo: Option<Arc<dyn PipelineRepository>>,
    /// 下载进度仓储
    pub download_repo: Option<Arc<dyn DownloadRepository>>,
    /// 配置文件路径（GUI 设置页读写 application.yml 用；
    /// 此前 GUI 写死相对路径，双击启动时 cwd 任意会写错位置）
    pub config_path: String,
}

/// 打开数据库并构造全部仓储
///
/// 数据库不可用属于**可降级故障**：记录告警后返回全 `None`，
/// 依赖持久化的功能自动关闭，其余功能照常工作。
fn open_repositories(
    db_path: &Path,
) -> (
    bool,
    Option<Arc<dyn DownloadRepository>>,
    Option<Arc<dyn PipelineRepository>>,
    Option<Arc<dyn OcrTaskRepository>>,
    Option<Arc<dyn PlaybackRepository>>,
) {
    let conn = match crate::platform::persistence::open_database(db_path) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("打开数据库失败，持久化功能降级: {}", e);
            return (false, None, None, None, None);
        }
    };

    let download_repo: Arc<dyn DownloadRepository> =
        Arc::new(crate::platform::persistence::sqlite_download_repository(conn.clone()));
    let pipeline_repo: Arc<dyn PipelineRepository> =
        Arc::new(crate::platform::persistence::sqlite_pipeline_repository(conn.clone()));
    let ocr_repo: Arc<dyn OcrTaskRepository> =
        Arc::new(crate::platform::persistence::sqlite_ocr_repository(conn.clone()));
    let playback_repo: Arc<dyn PlaybackRepository> =
        Arc::new(crate::platform::persistence::sqlite_playback_repository(conn));

    (
        true,
        Some(download_repo),
        Some(pipeline_repo),
        Some(ocr_repo),
        Some(playback_repo),
    )
}

/// 解析模型根目录
///
/// 优先级：显式覆盖 → `WorkspacePaths` 统一解析。
/// 统一解析内部顺序为：环境变量 → 配置绝对路径 → 可执行文件目录 → cwd 回溯。
fn resolve_models_dir(override_dir: Option<&Path>, config: Option<&AppConfig>) -> PathBuf {
    if let Some(dir) = override_dir {
        // `--models-dir` 传了占位默认值 "models" 时视为未指定，
        // 交给统一解析，避免双击启动时 cwd 任意导致找不到模型
        let is_placeholder = dir.as_os_str().is_empty() || dir == Path::new("models");
        if !is_placeholder {
            return dir.to_path_buf();
        }
    }

    let resolved = crate::platform::paths::models_dir();
    if resolved.as_os_str().is_empty() {
        // 配置里显式给了相对/绝对路径时优先尊重配置
        if let Some(cfg) = config {
            let from_config = PathBuf::from(&cfg.models.storage_path);
            if !from_config.as_os_str().is_empty() {
                return from_config;
            }
        }
        return PathBuf::from("models");
    }
    resolved
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 模型目录_占位值走统一解析() {
        // "models" 是 CLI 默认占位值，不应被当成用户显式指定
        let dir = resolve_models_dir(Some(Path::new("models")), None);
        assert!(!dir.as_os_str().is_empty());
    }

    #[test]
    fn 模型目录_显式路径被尊重() {
        let dir = resolve_models_dir(Some(Path::new("/tmp/custom-models")), None);
        assert_eq!(dir, PathBuf::from("/tmp/custom-models"));
    }

    #[test]
    fn 模型目录_空路径视为未指定() {
        let dir = resolve_models_dir(Some(Path::new("")), None);
        assert!(!dir.as_os_str().is_empty());
    }

    #[test]
    fn 装配选项_默认值() {
        let opts = BootstrapOptions::new("application.yml");
        assert_eq!(opts.config_path, PathBuf::from("application.yml"));
        assert!(opts.models_dir_override.is_none());
        assert!(opts.log_level_override.is_none());
        assert_eq!(opts.db_path, Some(PathBuf::from("data/votex.db")));
    }

    #[test]
    fn 装配选项_链式覆盖() {
        let opts = BootstrapOptions::new("a.yml")
            .with_models_dir(Some(PathBuf::from("m")))
            .with_log_level(Some("debug".into()))
            .with_db_path(None);
        assert_eq!(opts.models_dir_override, Some(PathBuf::from("m")));
        assert_eq!(opts.log_level_override.as_deref(), Some("debug"));
        assert!(opts.db_path.is_none());
    }
}
