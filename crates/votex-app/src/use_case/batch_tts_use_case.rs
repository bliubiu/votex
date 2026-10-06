//! 批量 TTS 合成用例
//!
//! 支持输入模式：
//! 1. 纯文本文件（每行一段文本，自动分配输出文件名）
//! 2. Tab 分隔文件（每行: `文本\t输出文件名\t音色\t语速`）
//! 3. 目录（收集目录下全部 .txt 文件，按文件分组避免文件名冲突）
//!
//! 工程保障：
//! - **单模型实例**：串行/并行均共享同一 TtsUseCase（引擎 Session 自带互斥，
//!   多线程 × N 份模型副本纯属内存浪费，并行收益来自降噪/IO 重叠）
//! - **并发闸门**：真实并发数受 `InferenceGate` 限制（内存压力自动收缩）
//! - **断点续转**：输出目录记录 `.votex_batch_progress.txt` 清单，
//!   已完成条目自动跳过
//! - **取消与进度**：支持 cancel_token 与条目级进度回调

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::TaskId;
use votex_domain::tts::value_object::{AudioFormat, DenoiseLevel};
use votex_infra::audio::denoiser;
use votex_infra::encoding::detector::EncodingDetector;
use votex_infra::shared::{GateCancel, InferenceGate};

use crate::use_case::tts_use_case::{engine_memory_estimate_mb, TtsUseCase};

/// 批量任务结果
#[derive(Debug, Clone)]
pub struct BatchTaskResult {
    /// 任务 ID
    pub task_id: TaskId,
    /// 索引
    pub index: usize,
    /// 输入文本
    pub text: String,
    /// 输出文件路径
    pub output_path: PathBuf,
    /// 是否成功
    pub success: bool,
    /// 错误信息
    pub error: Option<String>,
    /// 音频时长（毫秒；批量模式下不单独统计，恒为 0）
    pub duration_ms: u32,
}

/// 批量任务条目
#[derive(Debug, Clone)]
pub struct BatchTtsEntry {
    pub text: String,
    pub output_filename: String,
    pub voice: Option<String>,
    pub speed: Option<f32>,
}

/// 条目级进度回调：参数 (已完成条目数, 总条目数, 描述)
pub type BatchProgressFn = dyn Fn(usize, usize, &str) + Send + Sync;
pub type BatchProgressCallback = Option<Arc<BatchProgressFn>>;

/// 批量 TTS 配置
pub struct BatchTtsConfig<'a> {
    /// 输入路径（文件或目录）
    pub input: &'a Path,
    /// 输出目录
    pub output_dir: &'a Path,
    /// TTS 引擎
    pub engine: EngineKind,
    /// 默认音色
    pub voice: &'a str,
    /// 默认语速
    pub speed: f32,
    /// 输出格式
    pub format: AudioFormat,
    /// 语言
    pub lang: Option<&'a str>,
    /// 模型变体覆盖（qwen3 专用: "qwen3-tts-0.6b" / "qwen3-tts"）
    pub model_override: Option<&'a str>,
    /// 并发数（受资源闸门自动收缩）
    pub concurrency: usize,
    /// 是否启用降噪
    pub denoise: bool,
    /// 降噪级别
    pub denoise_level: DenoiseLevel,
    /// 取消令牌
    pub cancel: Option<Arc<AtomicBool>>,
    /// 条目级进度回调
    pub on_progress: BatchProgressCallback,
}

/// 断点续转清单文件名（位于输出目录）
const RESUME_MANIFEST: &str = ".votex_batch_progress.txt";

impl<'a> BatchTtsConfig<'a> {
    /// 兼容旧调用方的最小配置（无取消/进度/断点续转）
    pub fn basic(
        input: &'a Path,
        output_dir: &'a Path,
        engine: EngineKind,
        voice: &'a str,
        speed: f32,
        format: AudioFormat,
        lang: Option<&'a str>,
        model_override: Option<&'a str>,
        concurrency: usize,
        denoise: bool,
        denoise_level: DenoiseLevel,
    ) -> Self {
        Self {
            input,
            output_dir,
            engine,
            voice,
            speed,
            format,
            lang,
            model_override,
            concurrency,
            denoise,
            denoise_level,
            cancel: None,
            on_progress: None,
        }
    }
}

/// 批量 TTS 用例
pub struct BatchTtsUseCase;

impl BatchTtsUseCase {
    pub fn new() -> Self {
        Self
    }

    /// 执行批量 TTS 合成
    pub fn execute(config: BatchTtsConfig) -> Result<Vec<BatchTaskResult>> {
        // 确保输出目录存在
        std::fs::create_dir_all(config.output_dir)?;

        // 解析输入（文件或目录），获取条目列表
        let entries = Self::parse_input(config.input)?;
        if entries.is_empty() {
            anyhow::bail!("输入中没有有效的条目");
        }

        // 断点续转：读取已完成清单，跳过输出已存在的条目
        let manifest_path = config.output_dir.join(RESUME_MANIFEST);
        let done_set: std::collections::HashSet<String> =
            std::fs::read_to_string(&manifest_path)
                .map(|s| s.lines().map(|l| l.trim().to_string()).collect())
                .unwrap_or_default();
        let skipped_before = done_set.len();

        let pending: Vec<usize> = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| !done_set.contains(&e.output_filename))
            .map(|(i, _)| i)
            .collect();

        tracing::info!(
            "批量 TTS 启动: 共 {} 条, 断点续转跳过 {}, 待处理 {}, 引擎 {:?}, 请求并发 {}",
            entries.len(),
            skipped_before,
            pending.len(),
            config.engine,
            config.concurrency,
        );

        // 共享单模型实例：串行/并行都只用一份模型内存。
        // 引擎推理在 Session 互斥下天然串行，多线程的收益来自
        // 降噪（ffmpeg 子进程）与文件 IO 的重叠执行。
        let shared_tts = Arc::new(Mutex::new(TtsUseCase::new()));

        // 并发闸门：请求数受内存压力限制（Red=0 排队, Yellow=1, Green=配置值）
        let gate = InferenceGate::global();
        // 闸门取消令牌复用 config.cancel（Option<Arc<AtomicBool>>），
        // 使「排队等许可」阶段也能被 GUI 的停止操作中断
        let gate_cancel = GateCancel::from_atomic(config.cancel.as_ref().map(|c| Arc::clone(c)));
        let estimate_mb = engine_memory_estimate_mb(config.engine, config.model_override);
        let requested = config.concurrency.max(1);
        let workers = requested.min(gate.max_permits()).max(1);
        if workers < requested {
            tracing::info!(
                "并发数受资源闸门限制: 请求 {} → 实际 {}",
                requested,
                workers
            );
        }

        let next_index = AtomicUsize::new(0);
        let results = Mutex::new(Vec::with_capacity(entries.len()));
        let manifest_lock = Mutex::new(());
        let done_count = AtomicUsize::new(0);
        let total_count = pending.len();

        // 预计算各条目输出路径
        let ext = match config.format {
            AudioFormat::Wav => "wav",
            AudioFormat::Mp3 => "mp3",
            AudioFormat::M4A => "m4a",
            AudioFormat::Flac => "flac",
            AudioFormat::M4B => "m4b",
        };
        let output_paths: Vec<PathBuf> = entries
            .iter()
            .map(|e| {
                config
                    .output_dir
                    .join(&e.output_filename)
                    .with_extension(ext)
            })
            .collect();

        std::thread::scope(|scope| {
            let mut handles = Vec::new();
            for _worker in 0..workers {
                let config = &config;
                let entries = &entries;
                let output_paths = &output_paths;
                let pending = &pending;
                let next_index = &next_index;
                let results = &results;
                let shared_tts = &shared_tts;
                let manifest_lock = &manifest_lock;
                let manifest_path = &manifest_path;
                let done_count = &done_count;
                // 令牌按 worker 克隆（内部 Arc，克隆代价低且共享同一标志）
                let gate_cancel = &gate_cancel;

                handles.push(scope.spawn(move || {
                    loop {
                        // 取消检查
                        if let Some(ref token) = config.cancel {
                            if token.load(Ordering::SeqCst) {
                                break;
                            }
                        }

                        // 抢占下一个待处理条目
                        let slot = next_index.fetch_add(1, Ordering::SeqCst);
                        if slot >= pending.len() {
                            break;
                        }
                        let entry_idx = pending[slot];
                        let entry = &entries[entry_idx];
                        let output_path = &output_paths[entry_idx];

                        let started = std::time::Instant::now();
                        let synth_result = {
                            // 推理闸门：内存预检 + 并发限流（可取消）
                            let _permit = match gate.acquire(&gate_cancel, estimate_mb) {
                                Ok(p) => p,
                                // 被取消：跳出 worker 循环，剩余条目不再处理
                                Err(_) => {
                                    tracing::info!("批量合成已被取消，跳过剩余条目");
                                    break;
                                }
                            };
                            let tts = shared_tts.lock().unwrap_or_else(|e| e.into_inner());
                            tts.synthesize(
                                &entry.text,
                                output_path,
                                config.engine,
                                entry.voice.as_deref().unwrap_or(config.voice),
                                entry.speed.unwrap_or(config.speed),
                                config.format,
                                config.lang,
                                None,
                                config.model_override,
                            )
                        };

                        let result = match synth_result {
                            Ok(()) => {
                                // 降噪在锁外执行（ffmpeg 子进程，不占推理资源）
                                if config.denoise {
                                    if let Err(e) =
                                        Self::apply_denoise(output_path, config.denoise_level)
                                    {
                                        tracing::warn!("降噪失败 (跳过): {}", e);
                                    }
                                }
                                // 记入断点续转清单
                                {
                                    let _g = manifest_lock.lock().unwrap_or_else(|e| e.into_inner());
                                    use std::io::Write;
                                    if let Ok(mut f) = std::fs::OpenOptions::new()
                                        .create(true)
                                        .append(true)
                                        .open(&manifest_path)
                                    {
                                        let _ = writeln!(f, "{}", entry.output_filename);
                                    }
                                }
                                tracing::info!(
                                    "条目 {}/{} 完成: {} ({:.1}s)",
                                    slot + 1,
                                    total_count,
                                    entry.output_filename,
                                    started.elapsed().as_secs_f32()
                                );
                                BatchTaskResult {
                                    task_id: TaskId::new(),
                                    index: entry_idx,
                                    text: entry.text.chars().take(50).collect(),
                                    output_path: output_path.clone(),
                                    success: true,
                                    error: None,
                                    duration_ms: 0,
                                }
                            }
                            Err(e) => {
                                tracing::error!("条目 {} 合成失败: {}", entry.output_filename, e);
                                BatchTaskResult {
                                    task_id: TaskId::new(),
                                    index: entry_idx,
                                    text: entry.text.chars().take(50).collect(),
                                    output_path: output_path.clone(),
                                    success: false,
                                    error: Some(format!("{}", e)),
                                    duration_ms: 0,
                                }
                            }
                        };

                        let finished = done_count.fetch_add(1, Ordering::SeqCst) + 1;
                        if let Some(cb) = &config.on_progress {
                            cb(
                                finished,
                                total_count,
                                &format!("已完成 {} ({} 字符)", entry.output_filename, entry.text.len()),
                            );
                        }

                        results.lock().unwrap_or_else(|e| e.into_inner()).push(result);
                    }
                }));
            }
            for h in handles {
                let _ = h.join();
            }
        });

        let results = results.into_inner().unwrap();
        // 按条目索引排序，输出顺序稳定
        let mut all_results = results;
        all_results.sort_by_key(|r| r.index);
        let success_count = all_results.iter().filter(|r| r.success).count();
        let fail_count = all_results.len() - success_count;

        tracing::info!(
            "批量 TTS 完成: 本次成功 {}, 失败 {}, 历史累计完成 {}",
            success_count,
            fail_count,
            skipped_before + success_count
        );

        Ok(all_results)
    }

    /// 解析输入路径
    ///
    /// - 单文件：按行解析
    /// - 目录：收集目录下全部 .txt 文件（文件名排序），输出文件名
    ///   加源文件名前缀，避免多文件之间的名字冲突
    fn parse_input(input: &Path) -> Result<Vec<BatchTtsEntry>> {
        if input.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(input)?
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.extension()
                        .and_then(|e| e.to_str())
                        .map(|e| e.eq_ignore_ascii_case("txt"))
                        .unwrap_or(false)
                })
                .collect();
            files.sort();
            if files.is_empty() {
                anyhow::bail!("目录中没有 .txt 输入文件: {}", input.display());
            }
            let mut entries = Vec::new();
            for file in &files {
                let stem = file
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("book")
                    .to_string();
                for (i, entry) in Self::parse_file(file)?.into_iter().enumerate() {
                    entries.push(BatchTtsEntry {
                        text: entry.text,
                        // 目录模式：文件名加前缀保证唯一
                        output_filename: format!("{}_{}", stem, entry.output_filename),
                        voice: entry.voice,
                        speed: entry.speed,
                    });
                    let _ = i;
                }
            }
            Ok(entries)
        } else {
            Self::parse_file(input)
        }
    }

    /// 解析单个输入文件（自动检测 UTF-8/GBK 编码）
    fn parse_file(input: &Path) -> Result<Vec<BatchTtsEntry>> {
        let content = EncodingDetector::read_text_file(input)?;
        let mut entries = Vec::new();

        for (i, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }

            let parts: Vec<&str> = line.split('\t').collect();
            match parts.len() {
                1 => {
                    // 纯文本，自动生成文件名
                    let filename = format!("output_{:04}", i + 1);
                    entries.push(BatchTtsEntry {
                        text: parts[0].to_string(),
                        output_filename: filename,
                        voice: None,
                        speed: None,
                    });
                }
                2 => {
                    // text \t output_filename
                    entries.push(BatchTtsEntry {
                        text: parts[0].to_string(),
                        output_filename: parts[1].to_string(),
                        voice: None,
                        speed: None,
                    });
                }
                3 => {
                    // text \t output_filename \t voice
                    entries.push(BatchTtsEntry {
                        text: parts[0].to_string(),
                        output_filename: parts[1].to_string(),
                        voice: Some(parts[2].to_string()),
                        speed: None,
                    });
                }
                4 => {
                    // text \t output_filename \t voice \t speed
                    let speed = parts[3].parse::<f32>().ok();
                    entries.push(BatchTtsEntry {
                        text: parts[0].to_string(),
                        output_filename: parts[1].to_string(),
                        voice: Some(parts[2].to_string()),
                        speed,
                    });
                }
                _ => {
                    tracing::warn!("第 {} 行格式错误, 跳过: {}", i + 1, line);
                }
            }
        }

        Ok(entries)
    }

    /// 对 WAV 文件应用降噪
    fn apply_denoise(path: &Path, level: DenoiseLevel) -> Result<()> {
        let audio = votex_infra::audio::wav::read_wav(path)?;
        let denoised = denoiser::denoise(&audio, level)?;
        votex_infra::audio::wav::WavWriter::write(&denoised, path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_input_纯文本() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("input.txt");
        std::fs::write(&path, "第一段文本\n第二段文本\n# 注释行\n// 也注释\n第三段文本").unwrap();

        let entries = BatchTtsUseCase::parse_input(&path).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].text, "第一段文本");
        assert_eq!(entries[1].text, "第二段文本");
        assert_eq!(entries[2].text, "第三段文本");
    }

    #[test]
    fn test_parse_input_tab分隔() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("batch.txt");
        std::fs::write(
            &path,
            "文本1\tout1\tvoice1\t1.2\n文本2\tout2\n文本3\tout3\tvoice3\n",
        )
        .unwrap();

        let entries = BatchTtsUseCase::parse_input(&path).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].output_filename, "out1");
        assert_eq!(entries[0].voice, Some("voice1".to_string()));
        assert_eq!(entries[0].speed, Some(1.2));
        assert_eq!(entries[1].voice, None);
        assert_eq!(entries[2].voice, Some("voice3".to_string()));
    }

    #[test]
    fn test_parse_input_空文件() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.txt");
        std::fs::write(&path, "").unwrap();

        let entries = BatchTtsUseCase::parse_input(&path).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_parse_input_仅注释() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("comments.txt");
        std::fs::write(&path, "# 注释行\n// 也注释\n").unwrap();

        let entries = BatchTtsUseCase::parse_input(&path).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn test_parse_input_gbk编码自动转换() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gbk.txt");
        let (bytes, _, _) = encoding_rs::GBK.encode("第一章 开端\n主角登场");
        std::fs::write(&path, &bytes).unwrap();

        let entries = BatchTtsUseCase::parse_input(&path).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].text, "第一章 开端");
    }

    #[test]
    fn test_parse_input_目录模式加前缀() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "文本甲\n文本乙").unwrap();
        std::fs::write(dir.path().join("b.txt"), "文本丙").unwrap();

        let entries = BatchTtsUseCase::parse_input(dir.path()).unwrap();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].output_filename, "a_output_0001");
        assert_eq!(entries[1].output_filename, "a_output_0002");
        assert_eq!(entries[2].output_filename, "b_output_0001");
    }
}
