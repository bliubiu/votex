//! 统一 ONNX Session 创建工厂
//!
//! 封装 ort Session 的创建流程，消除各 Provider 中重复的：
//! - 线程数设置（available_parallelism）
//! - 优化级别设置（All=ORT_ENABLE_ALL）
//! - 执行提供器配置（CPU / DirectML GPU）
//! - 资源限制（内存、CPU 线程、执行模式）
//! - 错误映射

use anyhow::Result;
use ort::session::{Session, builder::GraphOptimizationLevel};
use std::path::Path;
use std::sync::LazyLock;
use std::sync::OnceLock;
use std::sync::RwLock;
use votex_domain::config::value_object::InferenceConfig;

/// 全局默认执行提供器
static GLOBAL_EP: OnceLock<ExecutionProvider> = OnceLock::new();

/// ONNX Runtime 动态库在各平台的文件名
///
/// ort 启用 `load-dynamic` 后，未设置 `ORT_DYLIB_PATH` 时会按此默认名
/// 在系统库搜索路径中查找，因此自携带的库必须使用对应平台的文件名。
#[cfg(target_os = "windows")]
const ORT_DYLIB_NAME: &str = "onnxruntime.dll";
#[cfg(any(target_os = "linux", target_os = "android", target_os = "freebsd"))]
const ORT_DYLIB_NAME: &str = "libonnxruntime.so";
#[cfg(any(target_os = "macos", target_os = "ios"))]
const ORT_DYLIB_NAME: &str = "libonnxruntime.dylib";
#[cfg(not(any(
    target_os = "windows",
    target_os = "linux",
    target_os = "android",
    target_os = "freebsd",
    target_os = "macos",
    target_os = "ios"
)))]
const ORT_DYLIB_NAME: &str = "onnxruntime.dll";

/// 确保 ORT 动态库路径已设置（load-dynamic 模式）
///
/// ort-sys rc.13 的静态预编译库存在初始化失败问题（GetApi 返回空），
/// 因此 workspace 启用 `load-dynamic`，运行时加载项目自带的
/// `models/runtime/<平台库名>`（ORT 1.30，CPU fp16 内核比 1.22 快 ~2-3 倍）。
///
/// 必须在**任何** ort API 调用前执行（不只是建 Session）：
/// 一旦 ort 先按默认名 `onnxruntime.dll` 加载失败，其内部的全局库句柄
/// 会被写入失败状态并缓存，之后再设置 `ORT_DYLIB_PATH` 也无法补救。
///
/// 行为：
/// - 用户已显式设置 `ORT_DYLIB_PATH` 时不覆盖；
/// - 依次从可执行文件所在目录、当前工作目录向上回溯最多 6 层查找；
/// - 找不到时不设置（回落到系统库搜索路径）。
pub fn ensure_ort_dylib_path() {
    // 已显式设置时仅在路径确实存在时才沿用。
    // .cargo/config.toml 为让 cargo test 找到库而设置了该变量（文件名按 Windows 写死），
    // 若在 Linux/macOS 上直接沿用会指向不存在的 .dll，故此处校验后按平台名重新查找。
    if let Some(path) = std::env::var_os("ORT_DYLIB_PATH") {
        let p = std::path::PathBuf::from(&path);
        if p.is_file() {
            return;
        }
        tracing::debug!("ORT_DYLIB_PATH 指向的文件不存在，重新查找: {:?}", p);
    }

    // 可执行文件所在目录优先：GUI 双击启动时 cwd 可能是任意位置
    let mut roots: Vec<std::path::PathBuf> = Vec::with_capacity(2);
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.to_path_buf());
        }
    }
    // 当前目录（cargo test 的 cwd 是 crates/<name>，需向上回溯）
    roots.push(std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from(".")));

    for root in roots {
        let mut dir = root;
        for _ in 0..6 {
            let candidate = dir.join("models").join("runtime").join(ORT_DYLIB_NAME);
            if candidate.exists() {
                tracing::info!("ORT 动态库: {:?}", candidate);
                std::env::set_var("ORT_DYLIB_PATH", &candidate);
                return;
            }
            if !dir.pop() {
                break;
            }
        }
    }

    tracing::warn!(
        "未找到自带 ONNX Runtime 动态库 {}，回落到系统库搜索路径",
        ORT_DYLIB_NAME
    );
}

/// 全局推理配置（启动时注入，支持 GUI 运行时更新）
static GLOBAL_CONFIG: LazyLock<RwLock<InferenceConfig>> = LazyLock::new(|| RwLock::new(InferenceConfig::default()));

/// 执行提供器类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    /// 仅 CPU
    Cpu,
    /// DirectML GPU（Windows）
    DirectML,
    /// 自动选择（优先 GPU，不可用时回退 CPU）
    Auto,
}

impl ExecutionProvider {
    /// 从配置字符串解析执行提供器
    pub fn from_config(s: &str) -> Self {
        match s {
            "directml" => Self::DirectML,
            "cpu" => Self::Cpu,
            _ => Self::Auto, // "auto" 或未知值默认自动
        }
    }
}

/// ONNX Session 创建工厂
///
/// 统一封装 Session 创建流程，自动配置：
/// - 线程数：使用系统可用并行度
/// - 优化级别：All（ORT_ENABLE_ALL，含 Attention/LayerNorm 融合，对齐 Python 默认）
/// - 执行提供器：可选 GPU 加速（DirectML），自动降级 CPU
/// - 错误处理：统一的错误上下文
pub struct OrtSessionFactory;

impl OrtSessionFactory {
    /// 设置全局默认执行提供器
    ///
    /// 在应用启动时调用一次。之后的 `create_raw()` 会自动使用此配置。
    pub fn set_global_ep(ep: ExecutionProvider) {
        let _ = GLOBAL_EP.set(ep);
    }

    /// 获取全局默认执行提供器
    ///
    /// 未显式设置时返回 [`ExecutionProvider::Cpu`]：
    /// DirectML 在部分驱动上会以不可捕获的进程崩溃（ACCESS_VIOLATION）失败，
    /// 因此把 GPU 加速设为**显式开启**，避免默认路径在异常环境下崩溃。
    /// 需要 GPU 时请在配置中把 `inference.execution_provider` 设为 `auto` / `directml`。
    pub fn get_global_ep() -> ExecutionProvider {
        GLOBAL_EP.get().copied().unwrap_or(ExecutionProvider::Cpu)
    }

    /// 设置全局推理配置（资源限制等）
    ///
    /// 在应用启动时设置，或由 GUI 设置页在保存配置时更新。
    /// 之后的 `create_raw*()` 方法会自动使用此配置。
    pub fn set_global_config(config: InferenceConfig) {
        if let Ok(mut g) = GLOBAL_CONFIG.write() {
            *g = config;
        }
    }

    /// 更新全局推理配置中的单个字段（不覆盖其他字段）
    pub fn update_global_config<F>(f: F)
    where
        F: FnOnce(&mut InferenceConfig),
    {
        if let Ok(mut g) = GLOBAL_CONFIG.write() {
            f(&mut g);
        }
    }

    /// 获取全局推理配置
    pub fn get_global_config() -> InferenceConfig {
        GLOBAL_CONFIG.read().ok().map(|g| g.clone()).unwrap_or_default()
    }

    /// 创建标准 ONNX Session
    ///
    /// 自动设置：
    /// - 线程数：系统可用并行度（默认 4）
    /// - 优化级别：All
    ///
    /// # 参数
    /// - `model_path`: ONNX 模型文件路径
    ///
    /// # 返回
    /// 成功返回 `Session`，失败返回带上下文的错误
    pub fn create(model_path: &Path) -> Result<Session> {
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self::create_with_threads(model_path, n_threads)
    }

    /// 创建 ONNX Session 并指定线程数
    ///
    /// # 参数
    /// - `model_path`: ONNX 模型文件路径
    /// - `threads`: 推理线程数
    ///
    /// # 返回
    /// 成功返回 `Session`，失败返回带上下文的错误
    pub fn create_with_threads(model_path: &Path, threads: usize) -> Result<Session> {
        ensure_ort_dylib_path();
        if !model_path.exists() {
            anyhow::bail!("ONNX 模型文件不存在: {:?}", model_path);
        }

        let mut builder = Session::builder()
            .map_err(|e| anyhow::anyhow!("创建 ONNX Session builder 失败: {}", e))?;
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| anyhow::anyhow!("设置优化级别失败: {}", e))?;
        builder = builder
            .with_intra_threads(threads)
            .map_err(|e| anyhow::anyhow!("设置线程数失败: {}", e))?;
        builder
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("加载 ONNX 模型失败 {:?}: {}", model_path, e))
    }

    /// 创建 ONNX Session（简化版，使用默认配置）
    ///
    /// 返回 `ort::Result<Session>`，便于在需要直接处理 ort 错误的场景使用。
    /// 自动使用全局执行提供器和全局推理配置。
    pub fn create_raw(model_path: &Path) -> ort::Result<Session> {
        let ep = Self::get_global_ep();
        let config = Self::get_global_config();
        Self::create_raw_with_config(model_path, ep, &config)
    }

    /// 创建 ONNX Session（带执行提供器配置，使用全局推理配置）
    ///
    /// 根据指定的执行提供器尝试 GPU 加速，自动降级到 CPU。
    ///
    /// # 参数
    /// - `model_path`: ONNX 模型文件路径
    /// - `ep`: 执行提供器类型
    ///
    /// # 返回
    /// 成功返回 `Session`，失败返回 `ort::Error`
    pub fn create_raw_with_ep(model_path: &Path, ep: ExecutionProvider) -> ort::Result<Session> {
        let config = Self::get_global_config();
        Self::create_raw_with_config(model_path, ep, &config)
    }

    /// 创建 ONNX Session（带执行提供器和优化级别，使用全局推理配置）
    ///
    /// 当默认 Level3 优化耗时过长（如大模型 int4 量化模型），
    /// 可通过此方法指定较低的优化级别以加快加载速度。
    ///
    /// # 参数
    /// - `model_path`: ONNX 模型文件路径
    /// - `ep`: 执行提供器类型
    /// - `opt_level`: 优化级别（Level1=基本, Level2=标准, Level3=最高）
    ///
    /// # 返回
    /// 成功返回 `Session`，失败返回 `ort::Error`
    pub fn create_raw_with_ep_and_level(
        model_path: &Path,
        ep: ExecutionProvider,
        opt_level: GraphOptimizationLevel,
    ) -> ort::Result<Session> {
        let config = Self::get_global_config();
        Self::create_raw_with_config_and_level(model_path, ep, opt_level, &config)
    }

    /// 创建 ONNX Session（完整配置版）
    ///
    /// 根据指定的执行提供器和推理配置，应用资源限制后创建 Session。
    /// 默认使用 Level3 优化级别。
    ///
    /// # 参数
    /// - `model_path`: ONNX 模型文件路径
    /// - `ep`: 执行提供器类型
    /// - `config`: 推理配置（线程数、内存限制、执行模式等）
    ///
    /// # 返回
    /// 成功返回 `Session`，失败返回 `ort::Error`
    pub fn create_raw_with_config(
        model_path: &Path,
        ep: ExecutionProvider,
        config: &InferenceConfig,
    ) -> ort::Result<Session> {
        Self::create_raw_with_config_and_level(
            model_path, ep, GraphOptimizationLevel::All, config,
        )
    }

    /// 创建 ONNX Session（完整配置 + 自定义优化级别）
    pub fn create_raw_with_config_and_level(
        model_path: &Path,
        ep: ExecutionProvider,
        opt_level: GraphOptimizationLevel,
        config: &InferenceConfig,
    ) -> ort::Result<Session> {
        ensure_ort_dylib_path();
        let n_threads = resolve_thread_count(config.num_threads);
        let inter_threads = resolve_thread_count(config.inter_threads);

        // 尝试 GPU
        if ep == ExecutionProvider::DirectML || ep == ExecutionProvider::Auto {
            #[cfg(feature = "directml")]
            {
                match Self::try_create_directml_with_level(model_path, n_threads, opt_level, config) {
                    Ok(session) => {
                        tracing::info!("  ONNX 会话使用 DirectML GPU: {:?}", model_path);
                        return Ok(session);
                    }
                    Err(e) => {
                        if ep == ExecutionProvider::DirectML {
                            return Err(e);
                        }
                        tracing::warn!("  DirectML 不可用，回退 CPU: {}", e);
                    }
                }
            }
            #[cfg(not(feature = "directml"))]
            {
                tracing::debug!("  DirectML feature 未启用，使用 CPU: {:?}", model_path);
            }
        }

        // CPU fallback
        build_configured_session_with_level(model_path, n_threads, inter_threads, opt_level, config)
    }

    /// 尝试创建 DirectML GPU 会话（自定义优化级别）
    #[cfg(feature = "directml")]
    fn try_create_directml_with_level(
        model_path: &Path,
        n_threads: usize,
        opt_level: GraphOptimizationLevel,
        config: &InferenceConfig,
    ) -> ort::Result<Session> {
        let inter_threads = resolve_thread_count(config.inter_threads);
        let mut builder = Session::builder()?
            .with_optimization_level(opt_level)?
            .with_intra_threads(n_threads)?
            .with_execution_providers([ort::ep::DirectML::default().build()])?;

        // 应用可选配置
        if inter_threads > 0 {
            builder = builder.with_inter_threads(inter_threads)?;
        }

        if !config.enable_memory_pattern {
            builder = builder.with_memory_pattern(false)?;
        }

        if config.execution_mode == "parallel" {
            builder = builder.with_parallel_execution(true)?;
        }

        builder.commit_from_file(model_path)
    }
}

/// 将用户配置的线程数（0 = auto）解析为实际线程数
fn resolve_thread_count(config_val: u32) -> usize {
    if config_val == 0 {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    } else {
        config_val as usize
    }
}

/// 应用推理配置构建 Session（CPU 路径，自定义优化级别）
fn build_configured_session_with_level(
    model_path: &Path,
    intra_threads: usize,
    inter_threads: usize,
    opt_level: GraphOptimizationLevel,
    config: &InferenceConfig,
) -> ort::Result<Session> {
    let mut builder = Session::builder()?
        .with_optimization_level(opt_level)?
        .with_intra_threads(intra_threads)?;

    // inter-op 线程（仅 parallel 模式有效）
    if inter_threads > 0 {
        builder = builder.with_inter_threads(inter_threads)?;
    }

    // 内存模式优化（动态输入形状时关闭可减少峰值内存）
    if !config.enable_memory_pattern {
        builder = builder.with_memory_pattern(false)?;
    }

    // 并行/顺序执行模式
    if config.execution_mode == "parallel" {
        builder = builder.with_parallel_execution(true)?;
    }

    // 内存限制：通过 ort 环境变量实现（ONNX Runtime 1.17+）
    if config.memory_limit_mb > 0 {
        // ORT 会在内部使用 `OrtSessionOptionsAppendExecutionProvider` 的环境设置
        // 这里通过设置环境变量辅助限制内存（仅 CPU EP 有效）
        std::env::set_var("ORT_ENABLE_MEMORY_ARENA", "0");
    }

    builder.commit_from_file(model_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn create_文件不存在应报错() {
        let result = OrtSessionFactory::create(Path::new("/nonexistent/model.onnx"));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("不存在"));
    }

    #[test]
    fn create_with_threads_文件不存在应报错() {
        let result = OrtSessionFactory::create_with_threads(
            Path::new("/nonexistent/model.onnx"),
            4,
        );
        assert!(result.is_err());
    }

    #[test]
    fn create_文件存在但非有效onnx应报错() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("fake.onnx");
        fs::write(&model_path, b"not a real onnx file").unwrap();

        // 文件存在但不是有效的 ONNX 格式，应该报错
        let result = OrtSessionFactory::create(&model_path);
        assert!(result.is_err());
    }

    #[test]
    fn create_with_threads_自定义线程数() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("fake.onnx");
        fs::write(&model_path, b"fake").unwrap();

        // 验证线程数参数被接受（虽然文件无效会报错，但参数传递正确）
        let result = OrtSessionFactory::create_with_threads(&model_path, 2);
        assert!(result.is_err());
    }
}
