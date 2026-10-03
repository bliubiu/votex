//! 批量 ASR 识别用例
//!
//! 递归扫描音频目录，对每个音频文件执行 ASR 识别。
//! 支持 WAV 格式，其他格式需先转换（预留接口）。

use anyhow::Result;
use std::path::{Path, PathBuf};

use votex_domain::shared::value_object::TaskId;
use votex_domain::tts::value_object::DenoiseLevel as DenoiseLevelTts;
use crate::use_case::asr_use_case::AsrUseCase;

/// ASR 批量任务结果
#[derive(Debug, Clone)]
pub struct BatchAsrResult {
    pub task_id: TaskId,
    pub index: usize,
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub success: bool,
    pub error: Option<String>,
    pub duration_secs: f64,
    pub text_length: usize,
}

/// 批量 ASR 配置
#[derive(Debug)]
pub struct BatchAsrConfig<'a> {
    /// 输入目录
    pub input_dir: &'a Path,
    /// 输出目录
    pub output_dir: &'a Path,
    /// ASR 模型 (whisper-base / whisper-small / sensevoice / paraformer / qwen3-asr / firered-asr / wenet)
    pub model: &'a str,
    /// 语言 (zh / en / zhen)
    pub language: &'a str,
    /// 字幕格式 (txt / srt / lrc)
    pub format: &'a str,
    /// 是否递归扫描子目录
    pub recursive: bool,
    /// 文件扩展名过滤
    pub extensions: Vec<String>,
    /// 并发数
    pub concurrency: usize,
    /// 是否启用降噪
    pub denoise: bool,
    /// 降噪级别
    pub denoise_level: DenoiseLevelTts,
}

impl Default for BatchAsrConfig<'_> {
    fn default() -> Self {
        Self {
            input_dir: Path::new("."),
            output_dir: Path::new("asr_output"),
            model: "whisper-base",
            language: "zh",
            format: "srt",
            recursive: false,
            extensions: vec!["wav".to_string(), "mp3".to_string()],
            concurrency: 1,
            denoise: false,
            denoise_level: DenoiseLevelTts::Low,
        }
    }
}

/// 批量 ASR 用例
pub struct BatchAsrUseCase;

impl BatchAsrUseCase {
    pub fn new() -> Self {
        Self
    }

    /// 执行批量 ASR 识别
    pub fn execute(config: BatchAsrConfig) -> Result<Vec<BatchAsrResult>> {
        std::fs::create_dir_all(config.output_dir)?;

        // 扫描输入目录
        let files = Self::scan_directory(config.input_dir, config.recursive, &config.extensions)?;
        if files.is_empty() {
            anyhow::bail!("输入目录中没有找到音频文件（支持格式: {:?}）", config.extensions);
        }

        tracing::info!(
            "批量 ASR 启动: {} 个音频文件, 模型 {}, 格式 {}, 并发 {}",
            files.len(),
            config.model,
            config.format,
            config.concurrency,
        );

        let file_count = files.len();
        let mut results = Vec::with_capacity(file_count);

        // 串行执行（ASR 引擎通常是单线程的）
        let mut asr = AsrUseCase::new();

        for (i, file_path) in files.iter().enumerate() {
            tracing::info!("处理第 {}/{}: {:?}", i + 1, file_count, file_path);

            // 构建输出路径：保留相对于输入目录的路径结构
            let rel_path = file_path.strip_prefix(config.input_dir)
                .unwrap_or(file_path);
            let output_rel = rel_path.with_extension(config.format);
            let output_path = config.output_dir.join(&output_rel);

            // 确保输出子目录存在
            if let Some(parent) = output_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            // 可选的降噪处理
            let process_path = if config.denoise {
                match Self::denoise_file(file_path, config.denoise_level) {
                    Ok(denoised_path) => denoised_path,
                    Err(e) => {
                        tracing::warn!("降噪失败, 使用原始文件: {}", e);
                        file_path.clone()
                    }
                }
            } else {
                file_path.clone()
            };

            match asr.recognize(
                &process_path,
                &output_path,
                config.model,
                config.language,
                config.format,
            ) {
                Ok(result) => {
                    let duration_secs = result.subtitles.last()
                        .map(|s| {
                            let end = &s.end_time;
                            end.hours as f64 * 3600.0
                                + end.minutes as f64 * 60.0
                                + end.seconds as f64
                                + end.millis as f64 / 1000.0
                        })
                        .unwrap_or(0.0);

                    results.push(BatchAsrResult {
                        task_id: TaskId::new(),
                        index: i,
                        input_path: file_path.clone(),
                        output_path,
                        success: true,
                        error: None,
                        duration_secs,
                        text_length: result.text.len(),
                    });
                }
                Err(e) => {
                    tracing::error!("文件 {:?} 识别失败: {}", file_path, e);
                    results.push(BatchAsrResult {
                        task_id: TaskId::new(),
                        index: i,
                        input_path: file_path.clone(),
                        output_path,
                        success: false,
                        error: Some(format!("{}", e)),
                        duration_secs: 0.0,
                        text_length: 0,
                    });
                }
            }

            // 清理临时降噪文件
            if config.denoise {
                let temp_path = file_path.with_extension("denoised.wav");
                if temp_path.exists() {
                    let _ = std::fs::remove_file(&temp_path);
                }
            }
        }

        let success_count = results.iter().filter(|r| r.success).count();
        let fail_count = results.len() - success_count;

        tracing::info!(
            "批量 ASR 完成: 成功 {}, 失败 {}",
            success_count, fail_count,
        );

        Ok(results)
    }

    /// 扫描目录获取音频文件列表
    fn scan_directory(dir: &Path, recursive: bool, extensions: &[String]) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        if !dir.is_dir() {
            anyhow::bail!("输入路径不是目录: {:?}", dir);
        }

        Self::scan_dir_recursive(dir, dir, recursive, extensions, &mut files)?;

        // 按文件名排序
        files.sort();

        Ok(files)
    }

    fn scan_dir_recursive(
        base_dir: &Path,
        dir: &Path,
        recursive: bool,
        extensions: &[String],
        files: &mut Vec<PathBuf>,
    ) -> Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() && recursive {
                Self::scan_dir_recursive(base_dir, &path, recursive, extensions, files)?;
            } else if path.is_file() {
                if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                    if extensions.iter().any(|e| e == ext) {
                        files.push(path);
                    }
                }
            }
        }
        Ok(())
    }

    /// 对音频文件进行降噪，返回降噪后的文件路径
    fn denoise_file(input_path: &Path, level: DenoiseLevelTts) -> Result<PathBuf> {
        let audio = votex_infra::audio::wav::read_wav(input_path)?;
        let denoised = votex_infra::audio::denoiser::denoise(&audio, level)?;
        let output_path = input_path.with_extension("denoised.wav");
        votex_infra::audio::wav::WavWriter::write(&denoised, &output_path)?;
        Ok(output_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_scan_directory_查找wav文件() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("test1.wav"), "fake wav").unwrap();
        std::fs::write(dir.path().join("test2.wav"), "fake wav").unwrap();
        std::fs::write(dir.path().join("readme.txt"), "text").unwrap();

        let files = BatchAsrUseCase::scan_directory(
            dir.path(),
            false,
            &["wav".to_string()],
        ).unwrap();

        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_scan_directory_递归查找() {
        let dir = tempdir().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(dir.path().join("root.wav"), "fake").unwrap();
        std::fs::write(sub.join("sub.wav"), "fake").unwrap();

        let files = BatchAsrUseCase::scan_directory(
            dir.path(),
            true,
            &["wav".to_string()],
        ).unwrap();
        assert_eq!(files.len(), 2);
    }

    #[test]
    fn test_scan_directory_非递归() {
        let dir = tempdir().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();
        std::fs::write(dir.path().join("root.wav"), "fake").unwrap();
        std::fs::write(sub.join("sub.wav"), "fake").unwrap();

        let files = BatchAsrUseCase::scan_directory(
            dir.path(),
            false,
            &["wav".to_string()],
        ).unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_scan_directory_多格式过滤() {
        let dir = tempdir().unwrap();
        std::fs::write(dir.path().join("test.wav"), "fake").unwrap();
        std::fs::write(dir.path().join("test.mp3"), "fake").unwrap();
        std::fs::write(dir.path().join("test.flac"), "fake").unwrap();

        let files = BatchAsrUseCase::scan_directory(
            dir.path(),
            false,
            &["wav".to_string(), "mp3".to_string()],
        ).unwrap();
        assert_eq!(files.len(), 2);
    }
}
