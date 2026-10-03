//! 后台任务执行器
//!
//! 管理 UseCase 在后台线程中的执行，通过 Channel 将进度和结果传回 GUI 线程。

use std::sync::mpsc;
use std::path::Path;
use std::sync::Arc;

use crate::state::Page;
use votex_domain::ocr::value_object::PageStatus;
use votex_domain::repository::PipelineRepository;
use votex_infra::persistence::download_repo::SqliteDownloadRepository;

/// 后台任务事件
#[derive(Debug, Clone)]
pub enum TaskEvent {
    /// 进度更新
    Progress {
        current: usize,
        total: usize,
        message: String,
    },
    /// 执行成功
    Success {
        message: String,
        data: Option<String>,
    },
    /// 执行失败
    Error {
        message: String,
    },
}

/// 启动 TTS 合成任务
///
/// `model_override` — Qwen3-TTS 模型变体覆盖（如 "qwen3-tts-0.6b" 或 "qwen3-tts-1.7b"），
/// 由 GUI 弹窗选择后传入，引擎内部据此加载对应变体。
pub fn spawn_tts(
    tx: mpsc::Sender<(Page, TaskEvent)>,
    text: String,
    output_path: String,
    engine: String,
    voice: String,
    speed: f32,
    format: String,
    lang: Option<String>,
    model_override: Option<String>,
    cancel_token: Arc<std::sync::atomic::AtomicBool>,
) {
    std::thread::spawn(move || {
        let page = Page::Tts;

        let _ = tx.send((page, TaskEvent::Progress {
            current: 0, total: 1,
            message: "准备中...".to_string(),
        }));

        // 检查取消
        if cancel_token.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = tx.send((page, TaskEvent::Error {
                message: "任务已取消".to_string(),
            }));
            return;
        }

        let mut tts = votex_app::use_case::tts_use_case::TtsUseCase::new();

        let engine_kind = match engine.as_str() {
            "kokoro" => votex_domain::model::value_object::EngineKind::Kokoro,
            "indextts2" => votex_domain::model::value_object::EngineKind::IndexTTS2,
            "qwen3" => votex_domain::model::value_object::EngineKind::Qwen3Tts,
            "cosyvoice3" => votex_domain::model::value_object::EngineKind::CosyVoice3,
            _ => {
                let _ = tx.send((page, TaskEvent::Error {
                    message: format!("不支持的引擎: {}", engine),
                }));
                return;
            }
        };

        let audio_format = match format.as_str() {
            "mp3" => votex_domain::tts::value_object::AudioFormat::Mp3,
            "m4b" => votex_domain::tts::value_object::AudioFormat::M4B,
            _ => votex_domain::tts::value_object::AudioFormat::Wav,
        };

        let _ = tx.send((page, TaskEvent::Progress {
            current: 0, total: 1,
            message: "正在合成...".to_string(),
        }));

        // 创建进度回调解包，逐段回传进度
        let tx_progress = tx.clone();
        let on_progress: votex_app::use_case::tts_use_case::TtsProgressCallback = Some(Box::new(move |current, total, msg| {
            let _ = tx_progress.send((page, TaskEvent::Progress {
                current,
                total,
                message: msg.to_string(),
            }));
        }));

        let result = tts.synthesize_ext(
            &text,
            Path::new(&output_path),
            engine_kind,
            &voice,
            speed,
            audio_format,
            lang.as_deref(),
            on_progress,
            model_override.as_deref(),
            Some(cancel_token.clone()),
            None,
            &votex_app::use_case::tts_use_case::SynthesisOptions::default(),
        );

        match result {
            Ok(()) => {
                let _ = tx.send((page, TaskEvent::Success {
                    message: format!("合成成功: {}", output_path),
                    data: None,
                }));
            }
            Err(e) => {
                let _ = tx.send((page, TaskEvent::Error {
                    message: format!("合成失败: {}", e),
                }));
            }
        }
    });
}

/// 启动 ASR 识别任务
pub fn spawn_asr(
    tx: mpsc::Sender<(Page, TaskEvent)>,
    input_path: String,
    output_path: String,
    model: String,
    language: String,
    format: String,
    cancel_token: Arc<std::sync::atomic::AtomicBool>,
) {
    std::thread::spawn(move || {
        let page = Page::Asr;

        let _ = tx.send((page, TaskEvent::Progress {
            current: 0, total: 1,
            message: "准备中...".to_string(),
        }));

        if cancel_token.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = tx.send((page, TaskEvent::Error {
                message: "任务已取消".to_string(),
            }));
            return;
        }

        let mut asr = votex_app::use_case::asr_use_case::AsrUseCase::new();

        let _ = tx.send((page, TaskEvent::Progress {
            current: 0, total: 1,
            message: "正在识别...".to_string(),
        }));

        let result = asr.recognize(
            Path::new(&input_path),
            Path::new(&output_path),
            &model,
            &language,
            &format,
        );

        match result {
            Ok(asr_result) => {
                let _ = tx.send((page, TaskEvent::Success {
                    message: format!("识别成功: {} 条字幕", asr_result.subtitles.len()),
                    data: None,
                }));
            }
            Err(e) => {
                let _ = tx.send((page, TaskEvent::Error {
                    message: format!("识别失败: {}", e),
                }));
            }
        }
    });
}

/// 启动 OCR 识别任务
pub fn spawn_ocr(
    tx: mpsc::Sender<(Page, TaskEvent)>,
    input_path: String,
    output_path: String,
    output_format: String,
    no_cls: bool,
    engine: String,
    cancel_token: Arc<std::sync::atomic::AtomicBool>,
) {
    std::thread::spawn(move || {
        let page = Page::Ocr;

        let send_progress = |current: usize, total: usize, msg: &str| {
            let _ = tx.send((page, TaskEvent::Progress {
                current, total, message: msg.to_string(),
            }));
        };

        send_progress(0, 1, "准备中...");

        if cancel_token.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = tx.send((page, TaskEvent::Error { message: "任务已取消".to_string() }));
            return;
        }

        let mut ocr = votex_app::use_case::ocr_use_case::OcrUseCase::new();

        // 加载 OCR 引擎
        let models_dir = std::env::current_dir()
            .unwrap_or_default()
            .join("models");

        if let Err(e) = ocr.load_engine(&models_dir, &engine) {
            let _ = tx.send((page, TaskEvent::Error {
                message: format!("OCR 引擎加载失败: {}，请先在设置中下载模型", e),
            }));
            return;
        }

        send_progress(0, 1, "正在识别...");

        let output_path_buf = Path::new(&output_path).to_path_buf();
        let tx_for_progress = tx.clone();
        let on_progress: votex_domain::shared::value_object::ProgressCallback = Some(Box::new(move |evt| {
            use votex_domain::shared::value_object::ProgressEvent;
            match evt {
                ProgressEvent::PhaseChanged { phase } => {
                    let phase_name = match phase.as_str() {
                        "loading_image" => "加载图片",
                        "detecting" => "文本检测",
                        "classifying" => "方向分类",
                        "recognizing" => "文字识别",
                        "batch_start" => "批量开始",
                        "batch_completed" => "批量完成",
                        _ => &phase,
                    };
                    let _ = tx_for_progress.send((page, TaskEvent::Progress {
                        current: 0, total: 1, message: format!("{}...", phase_name),
                    }));
                }
                ProgressEvent::BlockProgress { current, total } => {
                    let _ = tx_for_progress.send((page, TaskEvent::Progress {
                        current, total,
                        message: if total > 0 {
                            format!("识别进度 {}/{}", current, total)
                        } else {
                            "识别中...".to_string()
                        },
                    }));
                }
                ProgressEvent::PageProgress { current, total, .. } => {
                    let _ = tx_for_progress.send((page, TaskEvent::Progress {
                        current, total,
                        message: format!("识别第 {}/{} 页", current, total),
                    }));
                }
                ProgressEvent::Message { text } => {
                    let _ = tx_for_progress.send((page, TaskEvent::Progress {
                        current: 0, total: 1, message: text,
                    }));
                }
            }
        }));

        let result = ocr.recognize_single(
            Path::new(&input_path),
            &output_path_buf,
            &output_format,
            no_cls,
            on_progress,
        );

        match result {
            Ok(task) => {
                let block_count = task.pages.len();
                let _ = tx.send((page, TaskEvent::Success {
                    message: format!("识别完成: {} 页 → {:?}", block_count, output_path),
                    data: None,
                }));
            }
            Err(e) => {
                let _ = tx.send((page, TaskEvent::Error {
                    message: format!("识别失败: {}", e),
                }));
            }
        }
    });
}

/// 启动批量 OCR 识别任务
pub fn spawn_ocr_batch(
    tx: mpsc::Sender<(Page, TaskEvent)>,
    image_paths: Vec<String>,
    output_dir: String,
    output_format: String,
    no_cls: bool,
    engine: String,
    max_concurrency: usize,
    cancel_token: Arc<std::sync::atomic::AtomicBool>,
) {
    std::thread::spawn(move || {
        let page = Page::Ocr;

        let send_progress = |current: usize, total: usize, msg: &str| {
            let _ = tx.send((page, TaskEvent::Progress {
                current, total, message: msg.to_string(),
            }));
        };

        send_progress(0, 1, "准备中...");

        if cancel_token.load(std::sync::atomic::Ordering::SeqCst) {
            let _ = tx.send((page, TaskEvent::Error { message: "任务已取消".to_string() }));
            return;
        }

        if image_paths.is_empty() {
            let _ = tx.send((page, TaskEvent::Error { message: "图片列表为空".to_string() }));
            return;
        }

        let mut ocr = votex_app::use_case::ocr_use_case::OcrUseCase::new();

        let models_dir = std::env::current_dir()
            .unwrap_or_default()
            .join("models");

        if let Err(e) = ocr.load_engine(&models_dir, &engine) {
            let _ = tx.send((page, TaskEvent::Error {
                message: format!("OCR 引擎加载失败: {}，请先在设置中下载模型", e),
            }));
            return;
        }

        send_progress(0, image_paths.len(), "正在批量识别...");

        let path_bufs: Vec<std::path::PathBuf> = image_paths.iter().map(std::path::PathBuf::from).collect();
        let output_dir_buf = std::path::PathBuf::from(&output_dir);

        let tx_for_progress = tx.clone();
        let on_progress: votex_domain::shared::value_object::ProgressCallback = Some(Box::new(move |evt| {
            use votex_domain::shared::value_object::ProgressEvent;
            match evt {
                ProgressEvent::PhaseChanged { phase } => {
                    let phase_name = match phase.as_str() {
                        "loading_image" => "加载图片",
                        "detecting" => "文本检测",
                        "classifying" => "方向分类",
                        "recognizing" => "文字识别",
                        "batch_start" => "批量开始",
                        "batch_completed" => "批量完成",
                        _ => &phase,
                    };
                    let _ = tx_for_progress.send((page, TaskEvent::Progress {
                        current: 0, total: 1, message: format!("{}...", phase_name),
                    }));
                }
                ProgressEvent::BlockProgress { current, total } => {
                    let _ = tx_for_progress.send((page, TaskEvent::Progress {
                        current, total,
                        message: if total > 0 {
                            format!("识别进度 {}/{}", current, total)
                        } else {
                            "识别中...".to_string()
                        },
                    }));
                }
                ProgressEvent::PageProgress { current, total, .. } => {
                    let _ = tx_for_progress.send((page, TaskEvent::Progress {
                        current, total,
                        message: format!("识别第 {}/{} 页", current, total),
                    }));
                }
                ProgressEvent::Message { text } => {
                    let _ = tx_for_progress.send((page, TaskEvent::Progress {
                        current: 0, total: 1, message: text,
                    }));
                }
            }
        }));

        let result = ocr.recognize_batch(
            &path_bufs,
            &output_dir_buf,
            &output_format,
            no_cls,
            max_concurrency,
            on_progress,
        );

        match result {
            Ok(task) => {
                let total = task.pages.len();
                let ok_count = task.pages.iter().filter(|p| matches!(p.status, PageStatus::Completed)).count();
                let _ = tx.send((page, TaskEvent::Success {
                    message: format!("批量识别完成: {}/{} 页 → {:?}", ok_count, total, output_dir),
                    data: None,
                }));
            }
            Err(e) => {
                let _ = tx.send((page, TaskEvent::Error {
                    message: format!("批量识别失败: {}", e),
                }));
            }
        }
    });
}

/// 下载模型（用于设置页面）
pub fn spawn_model_download(
    tx: mpsc::Sender<(Page, TaskEvent)>,
    model_id: String,
    mirror: String,
    models_dir: String,
    download_repo: Option<Arc<SqliteDownloadRepository>>,
) {
    std::thread::spawn(move || {
        let page = Page::Settings;

        let _ = tx.send((page, TaskEvent::Progress {
            current: 0, total: 1,
            message: format!("正在下载 {}...", model_id),
        }));

        // 加载模型清单
        let registry_dir = std::path::Path::new(&models_dir)
            .parent()
            .map_or_else(|| std::path::PathBuf::from("models/registry"), |p| p.join("models/registry"));
        let registry = votex_infra::config::model_registry::ModelRegistryLoader::load(&registry_dir)
            .unwrap_or_default();

        let usecase = votex_app::use_case::model_use_case::ModelUseCase::new(
            std::path::Path::new(&models_dir),
            registry,
        );

        let model = votex_domain::model::value_object::ModelId::new(&model_id);

        // 镜像优先级：--mirror cn 走国内源
        let priority: Vec<String> = if mirror == "cn" {
            vec!["modelscope".into(), "hf-mirror".into(), "gitee".into()]
        } else {
            vec![
                "modelscope".into(),
                "hf-mirror".into(),
                "github".into(),
                "huggingface".into(),
            ]
        };

        let tx_progress = tx.clone();
        let repo_for_dl = download_repo.clone();
        let dl_model_id = model_id.clone();
        let dl_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let dl_started_clone = dl_started.clone();
        let on_progress: Option<Box<dyn Fn(u64, u64, &str, &str) + Send>> = Some(Box::new(
            move |downloaded, total, file_name, _status| {
                let _ = tx_progress.send((Page::Settings, TaskEvent::Progress {
                    current: downloaded as usize,
                    total: total as usize,
                    message: format!("{}. {}/{}", file_name, downloaded, total),
                }));
                // 将进度写入 SQLite 下载记录
                if let Some(ref r) = repo_for_dl {
                    if !dl_started_clone.swap(true, std::sync::atomic::Ordering::SeqCst) {
                        let _ = r.upsert(&dl_model_id, file_name, "unknown", downloaded, total, "Downloading", None);
                    } else {
                        let _ = r.update_progress(&dl_model_id, file_name, downloaded, total);
                    }
                }
            }
        ));

        let result = usecase.download_model(&model, &priority, on_progress);

        // 标记下载完成或失败
        if let Some(ref r) = download_repo {
            if result.is_ok() {
                let _ = r.upsert(&model_id, "completed", "unknown", 0, 0, "Completed", None);
            } else {
                let _ = r.upsert(&model_id, "failed", "unknown", 0, 0, "Failed", Some(&format!("{}", result.as_ref().unwrap_err())));
            }
        }

        match result {
            Ok(_) => {
                let _ = tx.send((page, TaskEvent::Success {
                    message: format!("模型 {} 下载完成", model_id),
                    data: None,
                }));
            }
            Err(e) => {
                let _ = tx.send((page, TaskEvent::Error {
                    message: format!("模型 {} 下载失败: {}", model_id, e),
                }));
            }
        }
    });
}

/// 启动流水线任务
pub fn spawn_pipeline(
    tx: mpsc::Sender<(Page, TaskEvent)>,
    kind: String,
    input_path: String,
    output_dir: String,
    engine: String,
    voice: String,
    speed: f32,
    pipeline_repo: Option<Arc<dyn PipelineRepository>>,
) {
    std::thread::spawn(move || {
        let page = Page::Pipeline;

        let send_event = |evt: TaskEvent| {
            let _ = tx.send((page, evt));
        };

        send_event(TaskEvent::Progress { current: 0, total: 1, message: "流水线准备中...".to_string() });

        let kind = match kind.as_str() {
            "audiobook" => votex_domain::pipeline::value_object::PipelineKind::Audiobook,
            "subtitle" => votex_domain::pipeline::value_object::PipelineKind::Subtitle,
            _ => {
                send_event(TaskEvent::Error { message: format!("不支持的流水线类型: {}", kind) });
                return;
            }
        };

        let input = std::path::Path::new(&input_path);
        let output = std::path::Path::new(&output_dir);

        send_event(TaskEvent::Progress { current: 0, total: 1, message: "流水线执行中...".to_string() });

        let mut use_case = votex_app::use_case::pipeline_use_case::PipelineUseCase::new();
        if let Some(repo) = pipeline_repo {
            use_case = use_case.with_repo(repo);
        }
        let result = use_case.execute(
            kind,
            input,
            output,
            &engine,
            &voice,
            speed,
            "whisper-base",
            "srt",
            None,
        );

        match result {
            Ok(_) => {
                send_event(TaskEvent::Success {
                    message: format!("流水线完成: {} → {}", input_path, output_dir),
                    data: None,
                });
            }
            Err(e) => {
                send_event(TaskEvent::Error {
                    message: format!("流水线执行失败: {}", e),
                });
            }
        }
    });
}

/// 启动视频生成任务
pub fn spawn_video(
    tx: mpsc::Sender<(Page, TaskEvent)>,
    script: String,
    output_path: String,
    tts_engine: String,
    voice: String,
    speed: f32,
    subtitle: bool,
    subtitle_format: String,
    asr_engine: String,
    resolution: String,
    bg_images: Option<Vec<String>>,
    bg_music: Option<String>,
) {
    std::thread::spawn(move || {
        let page = Page::Video;

        let send_event = |evt: TaskEvent| {
            let _ = tx.send((page, evt));
        };

        send_event(TaskEvent::Progress { current: 0, total: 1, message: "视频生成准备中...".to_string() });

        let req = votex_app::use_case::video_generate::VideoGenerateRequest {
            script,
            tts_engine,
            voice,
            speed,
            subtitle,
            subtitle_format,
            asr_engine,
            resolution,
            bg_images,
            bg_music,
        };

        send_event(TaskEvent::Progress { current: 0, total: 1, message: "视频生成中...".to_string() });

        let usecase = votex_app::use_case::video_generate::VideoGenerateUseCase;
        let result = usecase.execute(req, &output_path);

        match result {
            Ok(resp) => {
                send_event(TaskEvent::Success {
                    message: format!("视频生成完成: {} (时长: {:.1}秒)", resp.output_path, resp.duration_secs),
                    data: None,
                });
            }
            Err(e) => {
                send_event(TaskEvent::Error {
                    message: format!("视频生成失败: {}", e),
                });
            }
        }
    });
}

/// 启动翻译任务
pub fn spawn_translate(
    tx: mpsc::Sender<(Page, TaskEvent)>,
    source_text: String,
    engine: String,
    direction: String,
) {
    std::thread::spawn(move || {
        let page = Page::Translation;

        let _ = tx.send((page, TaskEvent::Progress {
            current: 0, total: 1,
            message: "翻译中...".to_string(),
        }));

        let use_case = match votex_app::use_case::translation_use_case::TranslationUseCase::new(&engine) {
            Ok(uc) => uc,
            Err(e) => {
                let _ = tx.send((page, TaskEvent::Error {
                    message: format!("创建翻译引擎失败: {}", e),
                }));
                return;
            }
        };

        let options = match votex_app::use_case::translation_use_case::TranslationUseCase::parse_direction(&direction) {
            Ok(dir) => votex_domain::translation::options::TranslationOptions::new(dir),
            Err(e) => {
                let _ = tx.send((page, TaskEvent::Error {
                    message: format!("翻译方向解析失败: {}", e),
                }));
                return;
            }
        };

        // 按片段汇报真实进度（长文本分段时可见）
        let result = use_case.pipeline().translate_with_progress(
            &source_text,
            &options,
            &mut |done, total| {
                let _ = tx.send((page, TaskEvent::Progress {
                    current: done,
                    total,
                    message: if total > 1 {
                        format!("翻译中... {}/{} 段", done, total)
                    } else {
                        "翻译中...".to_string()
                    },
                }));
            },
        );

        match result {
            Ok(outcome) => {
                let mut message = format!("翻译成功: {} → {}", source_text.len(), outcome.text.len());
                if outcome.segments > 1 {
                    message.push_str(&format!("（{} 段）", outcome.segments));
                }
                if outcome.cached_segments > 0 {
                    message.push_str(&format!("，命中缓存 {}", outcome.cached_segments));
                }
                let _ = tx.send((page, TaskEvent::Success {
                    message,
                    data: Some(outcome.text),
                }));
            }
            Err(e) => {
                let _ = tx.send((page, TaskEvent::Error {
                    message: format!("翻译失败: {}", e),
                }));
            }
        }
    });
}

/// 启动批量 TTS 合成任务
pub fn spawn_batch_tts(
    tx: mpsc::Sender<(Page, TaskEvent)>,
    input_dir: String,
    output_dir: String,
    engine: String,
    voice: String,
    speed: f32,
    format: String,
    concurrency: usize,
    cancel_token: Arc<std::sync::atomic::AtomicBool>,
) {
    std::thread::spawn(move || {
        let page = Page::Batch;

        let _ = tx.send((page, TaskEvent::Progress {
            current: 0, total: 1,
            message: "批量 TTS 准备中...".to_string(),
        }));

        let engine_kind = match engine.as_str() {
            "kokoro" => votex_domain::model::value_object::EngineKind::Kokoro,
            "indextts2" => votex_domain::model::value_object::EngineKind::IndexTTS2,
            "qwen3" => votex_domain::model::value_object::EngineKind::Qwen3Tts,
            "cosyvoice3" => votex_domain::model::value_object::EngineKind::CosyVoice3,
            _ => {
                let _ = tx.send((page, TaskEvent::Error {
                    message: format!("不支持的引擎: {}", engine),
                }));
                return;
            }
        };

        let audio_format = match format.as_str() {
            "mp3" => votex_domain::tts::value_object::AudioFormat::Mp3,
            "m4b" => votex_domain::tts::value_object::AudioFormat::M4B,
            _ => votex_domain::tts::value_object::AudioFormat::Wav,
        };

        let input_path = std::path::Path::new(&input_dir);
        let output_path = std::path::Path::new(&output_dir);

        use votex_app::use_case::batch_tts_use_case::{BatchTtsConfig, BatchTtsUseCase};

        // 条目级进度回调：转发给 GUI
        let tx_progress = tx.clone();
        let on_progress: votex_app::use_case::batch_tts_use_case::BatchProgressCallback =
            Some(Arc::new(move |done: usize, total: usize, msg: &str| {
                let _ = tx_progress.send((page, TaskEvent::Progress {
                    current: done,
                    total,
                    message: msg.to_string(),
                }));
            }));

        let config = BatchTtsConfig {
            input: input_path,
            output_dir: output_path,
            engine: engine_kind,
            voice: &voice,
            speed,
            format: audio_format,
            lang: None,
            model_override: None,
            concurrency,
            denoise: false,
            denoise_level: votex_domain::tts::value_object::DenoiseLevel::Low,
            cancel: Some(cancel_token),
            on_progress,
        };

        let _ = tx.send((page, TaskEvent::Progress {
            current: 0, total: 1,
            message: "正在执行批量 TTS...".to_string(),
        }));

        match BatchTtsUseCase::execute(config) {
            Ok(results) => {
                let success_count = results.iter().filter(|r| r.success).count();
                let _ = tx.send((page, TaskEvent::Success {
                    message: format!("批量 TTS 完成: {}/{} 成功", success_count, results.len()),
                    data: None,
                }));
            }
            Err(e) => {
                let _ = tx.send((page, TaskEvent::Error {
                    message: format!("批量 TTS 失败: {}", e),
                }));
            }
        }
    });
}

/// 启动批量 ASR 识别任务
pub fn spawn_batch_asr(
    tx: mpsc::Sender<(Page, TaskEvent)>,
    input_dir: String,
    output_dir: String,
    model: String,
    language: String,
    format: String,
    recursive: bool,
    concurrency: usize,
) {
    std::thread::spawn(move || {
        let page = Page::Batch;

        let _ = tx.send((page, TaskEvent::Progress {
            current: 0, total: 1,
            message: "批量 ASR 准备中...".to_string(),
        }));

        let input_path = std::path::Path::new(&input_dir);
        let output_path = std::path::Path::new(&output_dir);

        use votex_app::use_case::batch_asr_use_case::{BatchAsrConfig, BatchAsrUseCase};
        let config = BatchAsrConfig {
            input_dir: input_path,
            output_dir: output_path,
            model: &model,
            language: &language,
            format: &format,
            recursive,
            extensions: vec!["wav".into(), "mp3".into(), "m4b".into(), "m4a".into(), "flac".into()],
            concurrency,
            denoise: false,
            denoise_level: votex_domain::tts::value_object::DenoiseLevel::Low,
        };

        let _ = tx.send((page, TaskEvent::Progress {
            current: 0, total: 1,
            message: "正在执行批量 ASR...".to_string(),
        }));

        match BatchAsrUseCase::execute(config) {
            Ok(results) => {
                let success_count = results.iter().filter(|r| r.success).count();
                let _ = tx.send((page, TaskEvent::Success {
                    message: format!("批量 ASR 完成: {}/{} 成功", success_count, results.len()),
                    data: None,
                }));
            }
            Err(e) => {
                let _ = tx.send((page, TaskEvent::Error {
                    message: format!("批量 ASR 失败: {}", e),
                }));
            }
        }
    });
}
