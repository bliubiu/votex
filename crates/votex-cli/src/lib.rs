pub mod commands;

use anyhow::Result;
use clap::Parser;
use commands::root::{BatchAction, Cli, Commands, OcrAction};
use std::sync::{Arc, Mutex};
use rusqlite::Connection;
use votex_domain::config::value_object::AppConfig;

/// 运行模式 —— CLI 执行完初始化后主入口根据此值决定下一步
pub enum RunMode {
    /// 纯 CLI 模式，命令已执行完毕
    Cli,
    /// 需要启动 GUI
    Gui {
        models_dir: String,
        pipeline_repo: Option<Arc<dyn votex_domain::repository::PipelineRepository>>,
        download_repo: Option<Arc<votex_infra::persistence::download_repo::SqliteDownloadRepository>>,
    },
}

/// CLI 初始化流程（与 GUI 共享），返回下一步的运行模式
pub fn run() -> Result<RunMode> {
    let cli = Cli::parse();

    // 加载配置文件（可选）
    let config_path = std::path::PathBuf::from(&cli.config);
    let app_config: Option<AppConfig> = std::fs::read_to_string(&config_path)
        .ok()
        .and_then(|c| serde_yml::from_str(&c).ok());

    // 加密配置文件中的敏感字段
    if app_config.is_some() {
        let crypto = votex_infra::security::config_crypto::ConfigCrypto::new(None);
        if crypto.initialize().is_ok() {
            let _ = crypto.encrypt_config_file(&config_path);
        }
    }

    // 从配置读取日志参数，--verbose 覆盖为 debug
    let log_level = app_config.as_ref()
        .map(|c| c.log.level.as_str())
        .unwrap_or("info");
    let log_level = if cli.verbose { "debug" } else { log_level };
    let log_dir = app_config.as_ref()
        .map(|c| c.log.dir.as_str())
        .unwrap_or("logs");
    let log_file_enabled = app_config.as_ref()
        .map(|c| c.log.file_enabled)
        .unwrap_or(true);

    // 初始化日志
    votex_infra::logging::init::init(log_dir, log_level, log_file_enabled)?;

    // 配置 ONNX Runtime 执行提供器（GPU / CPU）
    let ep_name = app_config.as_ref()
        .map(|c| c.inference.execution_provider.as_str())
        .unwrap_or("auto");
    let ep = votex_infra::shared::ExecutionProvider::from_config(ep_name);
    votex_infra::shared::OrtSessionFactory::set_global_ep(ep);
    tracing::info!("ONNX Runtime 执行提供器: {:?}", ep);

    // 注入全局推理配置（资源限制等）
    if let Some(ref cfg) = app_config {
        let inference = &cfg.inference;
        let mut resource_info = String::new();
        if inference.num_threads > 0 {
            resource_info.push_str(&format!("CPU 线程={}", inference.num_threads));
        }
        if inference.memory_limit_mb > 0 {
            if !resource_info.is_empty() { resource_info.push_str(", "); }
            resource_info.push_str(&format!("内存上限={}MB", inference.memory_limit_mb));
        }
        if resource_info.is_empty() {
            resource_info = "无限（使用全部可用资源）".to_string();
        }
        votex_infra::shared::OrtSessionFactory::set_global_config(cfg.inference.clone());
        tracing::info!("推理资源限制: {}", resource_info);
    }

    // 注册 Ctrl+C 信号处理器，确保优雅退出
    ctrlc::set_handler(move || {
        tracing::warn!("收到中断信号（Ctrl+C），正在退出...");
        std::process::exit(0);
    })?;

    // 默认目录（优先 CLI 参数，其次配置文件）
    let models_dir = std::path::PathBuf::from(
        app_config.as_ref()
            .map(|c| c.models.storage_path.as_str())
            .unwrap_or("models")
    );
    let models_dir_str = cli.models_dir.clone(); // CLI 参数（有默认值 "models"），优先级最高
    let db_path = std::path::PathBuf::from("data/votex.db");

    // 初始化翻译运行时：进程级共享的模型会话池 + 翻译缓存
    // 必须在任何翻译调用之前完成，否则会退化成「每次翻译重新加载模型」
    let effective_models_dir = std::path::PathBuf::from(&models_dir_str);
    votex_app::services::translation_runtime::init(effective_models_dir.clone());
    if let Some(cfg) = app_config.as_ref() {
        if cfg.translation.enable_cache {
            votex_app::services::translation_runtime::init_cache(cfg.translation.cache_capacity);
        } else {
            votex_app::services::translation_runtime::init_cache(0);
        }
    }

    // 初始化资源治理：内存压力阈值 + 推理并发闸门
    {
        let rc = app_config
            .as_ref()
            .map(|c| c.resource.clone())
            .unwrap_or_default();
        if rc.enforce_limits {
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
        } else {
            tracing::warn!("资源治理已关闭（resource.enforce_limits=false），推理不再受内存限制");
        }
    }

    // 打开共享数据库连接，各 repo 共用同一连接
    let db_conn = match votex_infra::persistence::db::open_database(&db_path) {
        Ok(conn) => Some(conn),
        Err(e) => {
            tracing::warn!("打开数据库失败，部分功能降级: {}", e);
            None
        }
    };

    // 创建下载记录 repo
    let download_repo = db_conn.clone().map(|c| {
        Arc::new(votex_infra::persistence::download_repo::SqliteDownloadRepository::new(c))
    });

    // GUI 模式 → 返回 RunMode::Gui，由主入口启动 GUI
    if cli.gui {
        let pipeline_repo = db_conn.clone().map(|c| {
            Arc::new(votex_infra::persistence::pipeline_repo::SqlitePipelineRepository::new(c))
                as Arc<dyn votex_domain::repository::PipelineRepository>
        });
        return Ok(RunMode::Gui {
            models_dir: models_dir_str,
            pipeline_repo,
            download_repo,
        });
    }

    if let Some(command) = cli.command {
        match command {
            Commands::Model { action } => {
                commands::model::handle(&action, &models_dir, download_repo.clone())?;
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
                handle_ocr(action, &models_dir, db_conn.clone())?;
            }
            Commands::Pipeline { kind, input, output, engine, voice, speed, asr_model, subtitle_format } => {
                let pipeline_repo = db_conn.clone().map(|c| {
                    Arc::new(votex_infra::persistence::pipeline_repo::SqlitePipelineRepository::new(c))
                        as Arc<dyn votex_domain::repository::PipelineRepository>
                });
                commands::pipeline::handle(
                    &kind, &input, &output, &engine, &voice, speed, &asr_model, &subtitle_format,
                    pipeline_repo,
                )?;
            }
            Commands::Batch { action } => {
                handle_batch(action)?;
            }
            Commands::Config { action } => {
                let config_path = std::path::PathBuf::from(&cli.config);
                commands::config::handle(&action, &config_path)?;
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
fn handle_ocr(
    action: OcrAction,
    models_dir: &std::path::Path,
    db_conn: Option<Arc<Mutex<Connection>>>,
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
                db_conn.clone(),
            )
        }
        OcrAction::Batch { inputs, output, format, no_cls, engine, concurrency } => {
            if let Some(conn) = db_conn.clone() {
                let repo = votex_infra::persistence::ocr_repo::SqliteOcrTaskRepository::new(conn);
                ocr::run_batch_ocr_with_repo(
                    &inputs, &output, &format, no_cls, &engine, concurrency, models_dir, repo,
                )
            } else {
                ocr::run_batch_ocr(
                    &inputs, &output, &format, no_cls, &engine, concurrency, models_dir,
                )
            }
        }
        OcrAction::List => {
            if let Some(conn) = db_conn {
                ocr::list_ocr_tasks(conn)
            } else {
                tracing::warn!("数据库不可用，无法列出 OCR 任务");
                Ok(())
            }
        }
        OcrAction::Show { task_id } => {
            if let Some(conn) = db_conn {
                ocr::show_ocr_task(conn, &task_id)
            } else {
                tracing::warn!("数据库不可用，无法查看 OCR 任务");
                Ok(())
            }
        }
        OcrAction::Delete { task_id } => {
            if let Some(conn) = db_conn {
                ocr::delete_ocr_task(conn, &task_id)
            } else {
                tracing::warn!("数据库不可用，无法删除 OCR 任务");
                Ok(())
            }
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
