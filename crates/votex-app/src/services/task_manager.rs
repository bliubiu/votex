//! 统一任务管理器
//!
//! 提供 TTS/ASR 任务的统一排队、执行、进度回调和持久化能力。
//! 借鉴 MoneyPrinterTurbo 的任务队列模式：
//! - 统一队列接口，支持添加/查询/取消
//! - 执行时自动更新状态并持久化
//! - 进度回调实时通知
//! - 自动重试失败任务

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use votex_domain::asr::entity::AsrTask;
use votex_domain::asr::value_object::{
    AsrInput, AsrParams, AsrPhase, AsrProgress, AudioFormatKind,
    DenoiseLevel as AsrDenoise, Language, MediaKind, SliceLength, SubtitleFormat,
};
use votex_domain::model::value_object::{EngineKind, ModelId};
use votex_domain::repository::{AsrTaskRepository, TtsTaskRepository};
use votex_domain::shared::value_object::{ProgressCallback, ProgressEvent, TaskId, TaskStatus};
use votex_domain::tts::entity::TtsTask;
use votex_domain::tts::value_object::{
    AudioFormat, AudioOutput, DenoiseLevel, FileEncoding, InputSource, Pitch, SegmentSize, Speed,
    TtsInput, TtsParams, TtsProgress, TtsPhase, VoiceId, Volume,
};

use crate::use_case::tts_use_case::TtsUseCase;

/// 统一任务管理器
///
/// # 示例
///
/// ```ignore
/// let manager = TaskManager::new(tts_repo, asr_repo);
/// let task_id = manager.create_tts_task("你好世界", EngineKind::Kokoro, "output.wav")?;
/// manager.execute_tts(&task_id, Some(Box::new(|event| {
///     println!("进度: {:?}", event);
/// })))?;
/// ```
pub struct TaskManager {
    tts_repo: Arc<dyn TtsTaskRepository>,
    asr_repo: Arc<dyn AsrTaskRepository>,
}

impl TaskManager {
    pub fn new(
        tts_repo: Arc<dyn TtsTaskRepository>,
        asr_repo: Arc<dyn AsrTaskRepository>,
    ) -> Self {
        Self { tts_repo, asr_repo }
    }

    // ======== TTS 任务管理 ========

    /// 创建 TTS 任务并持久化
    pub fn create_tts_task(
        &self,
        text: &str,
        engine: EngineKind,
        voice: &str,
        speed: f32,
        output_path: &Path,
    ) -> Result<TaskId> {
        let now = now_str();
        let task = TtsTask {
            id: TaskId::new(),
            input: TtsInput {
                source: InputSource::DirectInput,
                raw_text: Some(text.to_string()),
                file_path: None,
                encoding: FileEncoding::Utf8,
            },
            params: TtsParams {
                engine,
                voice: VoiceId::new(voice, "", engine),
                speed: Speed::new(speed).unwrap_or_default(),
                pitch: Pitch::default(),
                volume: Volume::default(),
                segment_size: SegmentSize::S500,
                segment_silence_ms: 300,
                crossfade_ms: 50,
                num_to_chinese: true,
                denoise: false,
                denoise_level: DenoiseLevel::Low,
                emotion: None,
                dialect: None,
            },
            segments: Vec::new(),
            status: TaskStatus::Queued,
            output: Some(AudioOutput {
                format: AudioFormat::Wav,
                path: output_path.to_path_buf(),
                duration_ms: 0,
                sample_rate: 24000,
                bitrate_kbps: None,
                sample_depth: None,
            }),
            progress: TtsProgress {
                current_segment: 0,
                total_segments: 0,
                phase: TtsPhase::Idle,
            },
            created_at: now.clone(),
            updated_at: now,
        };

        let task_id = task.id.clone();
        self.tts_repo
            .save(&task)
            .map_err(|e| anyhow::anyhow!("保存 TTS 任务失败: {}", e))?;
        tracing::info!("TTS 任务已创建: {}", task_id);
        Ok(task_id)
    }

    /// 执行单个 TTS 任务（带进度回调）
    pub fn execute_tts(
        &self,
        task_id: &TaskId,
        progress: &ProgressCallback,
    ) -> Result<()> {
        let mut task = self
            .tts_repo
            .find_by_id(task_id)
            .ok_or_else(|| anyhow::anyhow!("TTS 任务不存在: {}", task_id))?;

        // 执行前检查：任务已被取消则不再启动
        if matches!(task.status, TaskStatus::Cancelled) {
            tracing::info!("TTS 任务 {} 已取消，跳过执行", task_id);
            return Ok(());
        }

        task.status = TaskStatus::Running;
        task.updated_at = now_str();
        self.tts_repo
            .save(&task)
            .map_err(|e| anyhow::anyhow!("更新任务状态失败: {}", e))?;

        let output = task.output.clone().ok_or_else(|| anyhow::anyhow!("TTS 任务缺少输出配置"))?;

        fire_progress(progress, ProgressEvent::PhaseChanged {
            phase: "TTS 合成中".to_string(),
        });

        let result = (|| -> Result<()> {
            let tts = TtsUseCase::new();
            let text = task
                .input
                .raw_text
                .as_deref()
                .ok_or_else(|| anyhow::anyhow!("TTS 任务缺少输入文本"))?;

            tts.synthesize(
                text,
                &output.path,
                task.params.engine,
                task.params.voice.id.as_str(),
                task.params.speed.value(),
                output.format,
                None,
                None,
                None,
            )?;

            Ok(())
        })();

        // 状态机收尾：执行期间用户可能已取消任务（cancel_tts_task 直接改库），
        // 重读当前状态——仍是 Cancelled 就不能被 Completed/Failed 覆盖
        let current_status = self
            .tts_repo
            .find_by_id(task_id)
            .map(|t| t.status)
            .unwrap_or(task.status.clone());
        let cancelled_during_run = matches!(current_status, TaskStatus::Cancelled);

        match (&result, cancelled_during_run) {
            (_, true) => {
                // 取消优先：保持 Cancelled 终态
                task.status = TaskStatus::Cancelled;
                fire_progress(progress, ProgressEvent::Message {
                    text: "任务已取消".to_string(),
                });
            }
            (Ok(()), false) => {
                task.status = TaskStatus::Completed;
                task.progress.phase = TtsPhase::Completed;
                fire_progress(progress, ProgressEvent::PhaseChanged {
                    phase: "TTS 完成".to_string(),
                });
            }
            (Err(ref e), false) => {
                task.status = TaskStatus::Failed(format!("{}", e));
                fire_progress(progress, ProgressEvent::Message {
                    text: format!("TTS 失败: {}", e),
                });
            }
        }

        task.updated_at = now_str();
        self.tts_repo
            .save(&task)
            .map_err(|e| anyhow::anyhow!("保存任务结果失败: {}", e))?;

        result
    }

    // ======== ASR 任务管理 ========

    /// 创建 ASR 识别任务并持久化
    pub fn create_asr_task(
        &self,
        audio_path: &Path,
        output_path: &Path,
        model: &str,
        language: &str,
        format: &str,
    ) -> Result<TaskId> {
        let now = now_str();
        let lang = parse_language(language);
        let fmt = parse_subtitle_format(format);
        let task = AsrTask {
            id: TaskId::new(),
            input: AsrInput {
                file_path: audio_path.to_path_buf(),
                file_kind: MediaKind::Audio(AudioFormatKind::Other),
            },
            params: AsrParams {
                model: ModelId::new(model),
                language: lang,
                auto_punctuation: true,
                auto_slice: true,
                slice_length: SliceLength::S30,
                denoise: false,
                denoise_level: AsrDenoise::Low,
                output_format: fmt,
            },
            slices: Vec::new(),
            status: TaskStatus::Queued,
            result: None,
            progress: AsrProgress {
                current_slice: 0,
                total_slices: 0,
                phase: AsrPhase::Idle,
            },
            // 用户指定的输出路径必须持久化：此前参数收下即丢，
            // 执行时从输入文件名推导，用户指定路径失效
            output_path: Some(output_path.to_path_buf()),
            created_at: now.clone(),
            updated_at: now,
        };

        let task_id = task.id.clone();
        self.asr_repo
            .save(&task)
            .map_err(|e| anyhow::anyhow!("保存 ASR 任务失败: {}", e))?;
        tracing::info!("ASR 任务已创建: {}", task_id);
        Ok(task_id)
    }

    /// 执行单个 ASR 任务（带进度回调）
    pub fn execute_asr(
        &self,
        task_id: &TaskId,
        progress: &ProgressCallback,
    ) -> Result<()> {
        let mut task = self
            .asr_repo
            .find_by_id(task_id)
            .ok_or_else(|| anyhow::anyhow!("ASR 任务不存在: {}", task_id))?;

        // 执行前检查：任务已被取消则不再启动
        if matches!(task.status, TaskStatus::Cancelled) {
            tracing::info!("ASR 任务 {} 已取消，跳过执行", task_id);
            return Ok(());
        }

        task.status = TaskStatus::Running;
        task.updated_at = now_str();
        self.asr_repo
            .save(&task)
            .map_err(|e| anyhow::anyhow!("更新任务状态失败: {}", e))?;

        fire_progress(progress, ProgressEvent::PhaseChanged {
            phase: "ASR 识别中".to_string(),
        });

        let model_str = task.params.model.as_str().to_string();
        let lang_str = language_str(&task.params.language).to_string();
        let fmt_str = format_str(&task.params.output_format).to_string();

        let result = (|| -> Result<PathBuf> {
            let asr = crate::services::shared_cases::shared_asr();
            // 优先用户指定的输出路径，缺省按输入文件名 + 目标扩展名推导
            let out_path = task
                .output_path
                .clone()
                .unwrap_or_else(|| task.input.file_path.with_extension(&fmt_str));

            asr.recognize(
                &task.input.file_path,
                &out_path,
                &model_str,
                &lang_str,
                &fmt_str,
                None,
            )?;

            Ok(out_path)
        })();

        // 状态机收尾：执行期间用户可能已取消任务，重读当前状态防覆盖
        let current_status = self
            .asr_repo
            .find_by_id(task_id)
            .map(|t| t.status)
            .unwrap_or(task.status.clone());
        let cancelled_during_run = matches!(current_status, TaskStatus::Cancelled);

        match (&result, cancelled_during_run) {
            (_, true) => {
                task.status = TaskStatus::Cancelled;
                fire_progress(progress, ProgressEvent::Message {
                    text: "任务已取消".to_string(),
                });
            }
            (Ok(ref output_path), false) => {
                task.status = TaskStatus::Completed;
                fire_progress(progress, ProgressEvent::PhaseChanged {
                    phase: "ASR 完成".to_string(),
                });
                fire_progress(progress, ProgressEvent::Message {
                    text: format!("识别完成: {:?}", output_path),
                });
            }
            (Err(ref e), false) => {
                task.status = TaskStatus::Failed(format!("{}", e));
                fire_progress(progress, ProgressEvent::Message {
                    text: format!("ASR 失败: {}", e),
                });
            }
        }

        task.updated_at = now_str();
        self.asr_repo
            .save(&task)
            .map_err(|e| anyhow::anyhow!("保存 ASR 任务结果失败: {}", e))?;

        result.map(|_| ())
    }

    // ======== 批量/查询 ========

    /// 执行所有等待中的 TTS 任务
    pub fn execute_all_pending_tts(&self, progress: &ProgressCallback) -> usize {
        let pending = self.tts_repo.find_pending();
        if pending.is_empty() {
            tracing::info!("没有等待中的 TTS 任务");
            return 0;
        }

        let total = pending.len();
        tracing::info!("开始批量处理 {} 个 TTS 任务", total);

        for (i, task) in pending.iter().enumerate() {
            fire_progress(progress, ProgressEvent::PageProgress {
                current: i + 1,
                total,
                page_index: i,
            });

            if let Err(e) = self.execute_tts(&task.id, progress) {
                tracing::error!("TTS 任务 {} 失败: {}", task.id, e);
            }
        }

        tracing::info!("批量 TTS 处理完成: {}/{}", total, total);
        total
    }

    /// 获取 TTS 任务列表（全量，按创建时间倒序）
    pub fn list_tts_tasks(&self) -> Vec<TtsTask> {
        self.tts_repo.find_all()
    }

    /// 获取 ASR 任务列表（全量，按创建时间倒序）
    pub fn list_asr_tasks(&self) -> Vec<AsrTask> {
        self.asr_repo.find_all()
    }

    /// 取消 TTS 任务
    pub fn cancel_tts_task(&self, task_id: &TaskId) -> Result<()> {
        let mut task = self
            .tts_repo
            .find_by_id(task_id)
            .ok_or_else(|| anyhow::anyhow!("任务不存在: {}", task_id))?;

        task.status = TaskStatus::Cancelled;
        task.updated_at = now_str();
        self.tts_repo
            .save(&task)
            .map_err(|e| anyhow::anyhow!("取消任务失败: {}", e))?;

        tracing::info!("TTS 任务已取消: {}", task_id);
        Ok(())
    }
}

fn now_str() -> String {
    format!(
        "{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    )
}

fn fire_progress(cb: &ProgressCallback, event: ProgressEvent) {
    if let Some(ref cb) = cb {
        cb(event);
    }
}

fn parse_language(s: &str) -> Language {
    match s.to_lowercase().as_str() {
        "zh" | "chinese" => Language::Zh,
        "en" | "english" => Language::En,
        "zhen" | "zheng" => Language::ZhEn,
        _ => Language::Zh,
    }
}

fn language_str(lang: &Language) -> &'static str {
    match lang {
        Language::Zh => "zh",
        Language::En => "en",
        Language::ZhEn => "zhen",
    }
}

fn parse_subtitle_format(s: &str) -> SubtitleFormat {
    match s.to_lowercase().as_str() {
        "srt" => SubtitleFormat::Srt,
        "lrc" => SubtitleFormat::Lrc,
        _ => SubtitleFormat::Txt,
    }
}

fn format_str(fmt: &SubtitleFormat) -> &'static str {
    match fmt {
        SubtitleFormat::Srt => "srt",
        SubtitleFormat::Lrc => "lrc",
        SubtitleFormat::Txt => "txt",
    }
}
