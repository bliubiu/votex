//! CLI 表现层
//!
//! 命令行入口。仅负责参数解析（`clap`）、调用应用层用例、格式化输出。
//!
//! # 分层规则
//!
//! - ✅ 依赖 `votex-app` / `votex-domain`
//! - ❌ **业务逻辑不得写在命令里** —— 一律下沉到 `votex-app::use_case`
//! - ❌ **不得出现 `votex_infra::` / `rusqlite`** —— 基础设施一律走
//!   `votex-app::platform` 门面，装配走 `votex-app::bootstrap::AppContext`
//!
//! # 与 GUI 的对等性
//!
//! AGENTS.md 要求「CLI 与 GUI 功能完全对等（同一套领域层，两套表现层）」。
//! `commands/` 下的每个子命令都应在 `gui/pages/` 有对应页面；
//! 两者的差异只能是交互形式（参数 vs 表单），不能是能力。
//!
//! # 启动流程
//!
//! 全部收敛到 [`votex_app::bootstrap::AppContext::bootstrap`]，
//! 与 GUI 走**同一份**装配代码：
//!
//! 1. 定位 ONNX Runtime 动态库（`ensure_runtime_library`，必须在任何 ort 调用前）
//! 2. 加载 `application.yml`（解析失败显式报错，不静默降级）
//! 3. 初始化日志、执行提供器、推理资源限制
//! 4. 下发下载配置（超时 / 断点续传）
//! 5. 资源治理、翻译运行时
//! 6. 打开 SQLite（启用 WAL 与 `foreign_keys`），构造各仓储
//! 7. 解析子命令并执行

pub mod commands;

use anyhow::Result;
use clap::Parser;
use commands::root::{BatchAction, Cli, Commands, OcrAction};
use votex_app::bootstrap::{AppContext, BootstrapOptions};
use votex_domain::repository::DownloadRepository;

/// 运行模式 —— CLI 执行完初始化后主入口根据此值决定下一步
pub enum RunMode {
    /// 纯 CLI 模式，命令已执行完毕
    Cli,
    /// 需要启动 GUI
    Gui(Box<AppContext>),
}

/// CLI 初始化流程（与 GUI 共享同一份 `AppContext` 装配），返回下一步的运行模式
pub fn run() -> Result<RunMode> {
    let cli = Cli::parse();

    // 装配：ORT 动态库 → 配置 → 日志 → EP/推理配置 → 资源治理 →
    //      下载配置 → 翻译运行时 → 数据库与仓储
    let ctx = AppContext::bootstrap(
        BootstrapOptions::new(&cli.config)
            .with_models_dir(Some(std::path::PathBuf::from(&cli.models_dir)))
            .with_log_level(if cli.verbose { Some("debug".into()) } else { None })
            .with_db_path(Some(votex_app::platform::paths::default_db_path())),
    )?;

    // 注册 Ctrl+C 信号处理器，确保优雅退出
    ctrlc::set_handler(|| {
        tracing::warn!("收到中断信号（Ctrl+C），正在退出...");
        std::process::exit(0);
    })?;

    let models_dir = ctx.models_dir().to_path_buf();

    // GUI 模式 → 返回 RunMode::Gui，由主入口启动 GUI
    if cli.gui {
        return Ok(RunMode::Gui(Box::new(ctx)));
    }

    if let Some(command) = cli.command {
        match command {
            Commands::Model { action } => {
                commands::model::handle(&action, &models_dir, ctx.download_repo().cloned())?;
            }
            Commands::Voice { action } => {
                commands::voice::handle(&action)?;
            }
            Commands::Tts { input, output, engine, voice, speed, lang, model, session_dir, role_map, loudnorm, atempo } => {
                commands::tts::handle(&input, &output, &engine, &voice, speed, lang, model, session_dir, role_map, loudnorm, atempo)?;
            }
            Commands::Asr { input, output, model, format, lang } => {
                commands::asr::handle(&input, &output, &model, &format, &lang)?;
            }
            Commands::Ocr { action } => {
                handle_ocr(action, &models_dir, ctx.ocr_repo().cloned())?;
            }
            Commands::Pipeline { kind, input, output, engine, voice, speed, asr_model, subtitle_format } => {
                commands::pipeline::handle(
                    &kind, &input, &output, &engine, &voice, speed, &asr_model, &subtitle_format,
                    ctx.pipeline_repo().cloned(),
                )?;
            }
            Commands::Batch { action } => {
                handle_batch(action)?;
            }
            Commands::Config { action } => {
                commands::config::handle(&action, ctx.config_path())?;
            }
            Commands::Script(cmd) => {
                commands::script::handle(&cmd)?;
            }
            Commands::Video(cmd) => {
                commands::video::handle(&cmd)?;
            }
            Commands::Translate {
                text,
                engine,
                direction,
                glossary,
                models_dir,
                target_script,
                max_segment_chars,
                no_cache,
            } => {
                commands::translate::handle_translate_with_args(
                    commands::translate::TranslateArgs {
                        text: &text,
                        engine: &engine,
                        direction: &direction,
                        glossary: glossary.as_deref(),
                        models_dir: models_dir.as_deref(),
                        target_script: target_script.as_deref(),
                        max_segment_chars,
                        no_cache,
                    },
                )?;
            }
            Commands::GlossaryInit { output } => {
                commands::translate::write_glossary_template(&output)?;
            }
        }
    } else {
        Cli::parse_from(["votex", "--help"]);
    }

    Ok(RunMode::Cli)
}

/// 处理 OCR 子命令
///
/// `repo` 为 `None` 表示数据库不可用，此时只执行不持久化。
fn handle_ocr(
    action: OcrAction,
    models_dir: &std::path::Path,
    repo: Option<std::sync::Arc<dyn votex_domain::repository::OcrTaskRepository>>,
) -> Result<()> {
    use commands::ocr;
    match action {
        OcrAction::Single { input, output, format, no_cls, engine } => {
            ocr::run_ocr(
                &input,
                output.as_deref(),
                &format,
                no_cls,
                &engine,
                models_dir,
                repo,
            )
        }
        OcrAction::Batch { inputs, output, format, no_cls, engine, concurrency } => {
            ocr::run_batch_ocr(
                &inputs, &output, &format, no_cls, &engine, concurrency, models_dir, repo,
            )
        }
        OcrAction::List => {
            ocr::list_ocr_tasks(repo)
        }
        OcrAction::Show { task_id } => {
            ocr::show_ocr_task(repo, &task_id)
        }
        OcrAction::Delete { task_id } => {
            ocr::delete_ocr_task(repo, &task_id)
        }
    }
}

/// 处理 batch 子命令
fn handle_batch(action: BatchAction) -> Result<()> {
    match action {
        BatchAction::Tts { input, output, engine, voice, speed, lang, model, format, concurrency, denoise, denoise_level } => {
            commands::batch_tts::handle(
                &input, &output, &engine, &voice, speed, lang, model, &format, concurrency, denoise, &denoise_level,
            )
        }
        BatchAction::Asr { input, output, model, format, lang, recursive, concurrency, denoise, denoise_level } => {
            commands::batch_asr::handle(
                &input, &output, &model, &format, &lang, recursive, concurrency, denoise, &denoise_level,
            )
        }
        BatchAction::Translate { input, output, engine, direction, glossary, models_dir } => {
            commands::translate::handle_batch_translate(
                &input, &output, &engine, &direction,
                glossary.as_deref(), models_dir.as_deref(),
            )
        }
    }
}

/// 供 `commands::model` 复用的下载进度上报类型
pub type SharedDownloadRepo = Option<std::sync::Arc<dyn DownloadRepository>>;
