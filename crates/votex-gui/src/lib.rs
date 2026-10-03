pub mod app;
pub mod pages;
pub mod widgets;
pub mod state;
pub mod task_runner;
pub mod model_detector;
pub mod theme;

use std::sync::Arc;
use eframe;
use egui::{FontDefinitions, FontFamily};
use votex_domain::repository::PipelineRepository;
use votex_infra::persistence::download_repo::SqliteDownloadRepository;

use crate::state::AppState;

/// 加载自定义中文字体，解决 GUI 中文方块问题
fn load_chinese_font(cc: &eframe::CreationContext) {
    let font_path = std::path::Path::new("models").join("SourceHanSansCN-Regular.otf");
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
pub fn run(
    models_dir: String,
    pipeline_repo: Option<Arc<dyn PipelineRepository>>,
    download_repo: Option<Arc<SqliteDownloadRepository>>,
) -> anyhow::Result<()> {
    // 初始化翻译运行时（进程级共享会话池 + 缓存），
    // 否则 GUI 每次点「翻译」都会重新加载数 GB 的离线模型
    votex_app::services::translation_runtime::init(std::path::PathBuf::from(&models_dir));

    // 初始化资源治理（GUI 读 application.yml 的 resource 段；缺省用安全默认值）
    {
        let rc = std::fs::read_to_string("application.yml")
            .ok()
            .and_then(|s| serde_yml::from_str::<votex_domain::config::value_object::AppConfig>(&s).ok())
            .map(|c| c.resource)
            .unwrap_or_default();
        let monitor = votex_infra::shared::ResourceMonitor::global();
        monitor.set_thresholds(rc.memory_yellow_used_pct, rc.memory_red_used_pct);
        votex_infra::shared::InferenceGate::global()
            .configure_max_permits(rc.max_inference_concurrency);
        tracing::info!(
            "资源治理已启用: 推理并发上限 {}, 内存压力阈值 Yellow {:.0}% / Red {:.0}%",
            rc.max_inference_concurrency,
            rc.memory_yellow_used_pct,
            rc.memory_red_used_pct
        );
    }

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
            let mut state = AppState::new(models_dir);
            state.pipeline_repo = pipeline_repo;
            state.download_repo = download_repo;
            Ok(Box::new(app::VotexApp::new(state)))
        }),
    ).map_err(|e| anyhow::anyhow!("GUI 启动失败: {}", e))
}
