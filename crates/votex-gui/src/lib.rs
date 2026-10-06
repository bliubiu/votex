//! GUI 表现层（egui / eframe）
//!
//! 声阅的图形界面入口。
//!
//! # 线程模型（重要）
//!
//! egui 是**立即模式**且在 UI 线程渲染，因此：
//!
//! - ❌ **禁止**在 UI 线程做推理、文件 IO、数据库查询
//! - ✅ 耗时任务一律经 `task_runner` 派发到 `std::thread`
//! - ✅ **不引入 tokio** —— 异步运行时会与 ONNX Runtime 的线程模型冲突
//! - ✅ 跨线程通信用 `std::sync::mpsc`，事件在 `update()` 中消费
//!
//! # 分层规则
//!
//! - ✅ 依赖 `votex-app` / `votex-domain`
//! - ❌ **不得出现 `votex_infra::` / `rusqlite`** —— 基础设施一律走
//!   `votex-app::platform` 门面
//!
//! # 启动流程
//!
//! 装配全部由 [`votex_app::bootstrap::AppContext`] 在 CLI 侧完成，
//! GUI 只负责把装配结果灌入 UI 状态：
//!
//! 1. 接收 `AppContext`（已含 ORT 初始化、配置、日志、资源治理、仓储）
//! 2. 加载中文字体（避免中文显示为方块）
//! 3. 进入 `eframe::run_native`
//!
//! # 结构
//!
//! | 模块 | 职责 |
//! | :--- | :--- |
//! | `app` | 应用根组件与页面路由 |
//! | `pages` | 各功能页面（仪表盘 / TTS / ASR / OCR / 翻译 / 设置 / 引导） |
//! | `widgets` | 复用组件（音频播放器等） |
//! | `state` | 全局 UI 状态 |
//! | `task_runner` | 后台任务派发与事件回传 |
//! | `model_detector` | 模型就绪状态探测（读 registry 清单） |
//! | `model_ready` | 模型就绪判定纯逻辑（有单元测试） |
//! | `theme` | 配色与样式 |

// 测试函数使用中文语义命名（如 `required_缺任一即不就绪`），
// 有助于表达断言意图；仅在测试编译时豁免 snake_case 检查。
#![cfg_attr(test, allow(non_snake_case))]

pub mod app;
pub mod pages;
pub mod widgets;
pub mod state;
pub mod task_runner;
pub mod model_detector;
pub mod model_ready;
pub mod theme;

#[cfg(test)]
mod cancel_contract;

use eframe;
use egui::{FontDefinitions, FontFamily};
use votex_app::bootstrap::AppContext;
use votex_app::platform::persistence;

use crate::state::AppState;

/// 加载自定义中文字体，解决 GUI 中文方块问题
fn load_chinese_font(cc: &eframe::CreationContext) {
    // 走统一路径解析：GUI 双击启动时 cwd 可能是任意目录
    let font_path =
        votex_app::platform::paths::models_dir().join("SourceHanSansCN-Regular.otf");
    if !font_path.exists() {
        tracing::warn!("中文字体文件不存在: {:?}，中文可能显示为方块", font_path);
        return;
    }
    let font_data = match std::fs::read(&font_path) {
        Ok(data) => data,
        Err(e) => {
            tracing::warn!("读取字体文件失败: {}，中文可能显示为方块", e);
            return;
        }
    };

    let mut fonts = FontDefinitions::default();
    fonts.font_data.insert(
        "SourceHanSansCN".to_owned(),
        std::sync::Arc::new(egui::FontData::from_owned(font_data)),
    );
    if let Some(proportional) = fonts.families.get_mut(&FontFamily::Proportional) {
        proportional.insert(0, "SourceHanSansCN".to_owned());
    }
    if let Some(monospace) = fonts.families.get_mut(&FontFamily::Monospace) {
        monospace.insert(0, "SourceHanSansCN".to_owned());
    }
    cc.egui_ctx.set_fonts(fonts);
    tracing::info!("中文字体已加载: {:?}", font_path);
}

/// 启动 GUI 应用
///
/// `ctx` 由 CLI 侧的 [`AppContext::bootstrap`] 装配完成后传入，
/// 因此这里**不需要**再做 ORT 初始化、配置加载或日志初始化。
pub fn run(ctx: AppContext) -> anyhow::Result<()> {
    let handles = ctx.gui_handles();

    // 播放进度仓储：GUI 侧的断点续听依赖它。
    // 优先复用 AppContext 已装配的仓储；CLI 未开数据库时按需单独打开。
    let playback_repo = ctx
        .playback_repo()
        .cloned()
        .or_else(|| open_playback_repo_fallback());

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1024.0, 700.0])
            .with_min_inner_size([800.0, 500.0])
            .with_title("声阅 (votex)"),
        ..Default::default()
    };

    eframe::run_native(
        "声阅",
        options,
        Box::new(move |cc| {
            load_chinese_font(cc);
            crate::theme::configure_style(&cc.egui_ctx);
            let mut state = AppState::new(handles.models_dir);
            state.config_path = handles.config_path;
            state.pipeline_repo = handles.pipeline_repo;
            state.download_repo = handles.download_repo;
            state.audio_player.playback_repo = playback_repo;
            Ok(Box::new(app::VotexApp::new(state)))
        }),
    )
    .map_err(|e| anyhow::anyhow!("GUI 启动失败: {}", e))
}

/// 播放进度仓储兜底装配
///
/// 仅在 `AppContext` 未持有播放仓储时触发（例如以库方式嵌入 GUI）。
/// 失败时返回 `None`，播放器降级为「不续播」，不影响其他功能。
fn open_playback_repo_fallback() -> Option<std::sync::Arc<dyn votex_domain::repository::PlaybackRepository>> {
    let conn = match persistence::open_database(&votex_app::platform::paths::default_db_path()) {
        Ok(c) => c,
        Err(e) => {
            tracing::warn!("打开数据库失败，播放进度不持久化: {}", e);
            return None;
        }
    };
    Some(std::sync::Arc::new(persistence::sqlite_playback_repository(
        conn,
    )))
}
