//! 运行时门面：ONNX Runtime、日志、执行提供器、资源治理
//!
//! 收敛 CLI / GUI 中重复的 5 段初始化代码。
//! 这几段历史上在两个 crate 各写一遍，导致 GUI 侧长期遗漏
//! 「下载配置下发」与「ORT 初始化时机」两处关键步骤。

use votex_domain::config::value_object::{InferenceConfig, ResourceConfig};
use votex_infra::shared::MemoryPressure;

/// 定位 ONNX Runtime 动态库
///
/// # 为什么必须最先调用
///
/// `ort` 会在首次 API 调用时按默认名 `onnxruntime.dll` 加载动态库，
/// 并把**加载失败的结果缓存**进内部 `OnceLock`。
/// 一旦缓存了失败，后续再设置 `ORT_DYLIB_PATH` 也无效，
/// 表现为「所有推理都报找不到 DLL」且无法自愈。
///
/// 跨平台文件名由 infra 内部按 `cfg!` 选择：
/// Windows `onnxruntime.dll` / Linux `libonnxruntime.so` / macOS `libonnxruntime.dylib`。
pub fn ensure_runtime_library() {
    votex_infra::shared::ensure_ort_dylib_path();
}

/// 初始化日志系统
///
/// 必须在配置加载之后调用（日志级别本身来自配置）。
pub fn init_logging(log_dir: &str, level: &str, file_enabled: bool) -> anyhow::Result<()> {
    votex_infra::logging::init::init(log_dir, level, file_enabled)
}

/// 配置 ONNX Runtime 执行提供器（GPU / CPU）
pub fn configure_execution_provider(provider: &str) {
    let ep = votex_infra::shared::ExecutionProvider::from_config(provider);
    votex_infra::shared::OrtSessionFactory::set_global_ep(ep);
    tracing::info!("ONNX Runtime 执行提供器: {:?}", ep);
}

/// 下发推理资源配置（线程数、内存上限、执行模式）
pub fn apply_inference_config(config: &InferenceConfig) {
    let mut resource_info = String::new();
    if config.num_threads > 0 {
        resource_info.push_str(&format!("CPU 线程={}", config.num_threads));
    }
    if config.memory_limit_mb > 0 {
        if !resource_info.is_empty() {
            resource_info.push_str(", ");
        }
        resource_info.push_str(&format!("内存上限={}MB", config.memory_limit_mb));
    }
    if resource_info.is_empty() {
        resource_info = "无限（使用全部可用资源）".to_string();
    }

    votex_infra::shared::OrtSessionFactory::set_global_config(config.clone());
    tracing::info!("推理资源限制: {}", resource_info);
}

/// 就地修改全局推理配置
///
/// GUI 设置页保存配置时使用：改完落盘同时立即作用于已运行的会话工厂，
/// 避免「重启才生效」。
pub fn update_inference_config<F>(f: F)
where
    F: FnOnce(&mut InferenceConfig),
{
    votex_infra::shared::OrtSessionFactory::update_global_config(f);
}

/// 配置资源治理：内存压力阈值 + 推理并发闸门
///
/// `enforce_limits=false` 时只告警不设限，
/// 便于用户在受控环境（如集显机器）下临时关闭限制。
pub fn configure_resource_governance(config: &ResourceConfig) {
    if !config.enforce_limits {
        tracing::warn!("资源治理已关闭（resource.enforce_limits=false），推理不再受内存限制");
        return;
    }

    let monitor = votex_infra::shared::ResourceMonitor::global();
    monitor.set_thresholds(config.memory_yellow_used_pct, config.memory_red_used_pct);
    votex_infra::shared::InferenceGate::global()
        .configure_max_permits(config.max_inference_concurrency);

    tracing::info!(
        "资源治理已启用: 推理并发上限 {}, 内存压力阈值 Yellow {:.0}% / Red {:.0}%",
        config.max_inference_concurrency,
        config.memory_yellow_used_pct,
        config.memory_red_used_pct
    );
}

/// 获取当前内存压力等级
///
/// GUI 状态栏展示用。
pub fn current_memory_pressure() -> MemoryPressure {
    votex_infra::shared::ResourceMonitor::global().pressure()
}

/// 当前推理并发上限
pub fn max_inference_permits() -> usize {
    votex_infra::shared::InferenceGate::global().max_permits()
}

/// 当前已占用的推理许可数
pub fn held_inference_permits() -> usize {
    votex_infra::shared::InferenceGate::global().held_permits()
}
