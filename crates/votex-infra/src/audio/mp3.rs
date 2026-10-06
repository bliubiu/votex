//! MP3 编码器（基于 ffmpeg.exe 进程调用）
//!
//! 使用系统安装的 ffmpeg.exe 将 WAV 文件转换为 MP3 格式。
//! 当前通过 std::process::Command 调用 ffmpeg CLI，
//! 预留接口以便后续切换到 ffmpeg-next crate（当 Windows 开发 SDK 可用时）。
//!
//! ffmpeg 路径搜索优先级：
//! 1. 环境变量 FFMPEG_PATH
//! 2. 默认路径 C:\Apps\ffmpeg\bin\ffmpeg.exe
//! 3. PATH 环境变量中的 ffmpeg

use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;

/// ffmpeg 默认安装路径
const DEFAULT_FFMPEG_PATH: &str = r"C:\Apps\ffmpeg\bin\ffmpeg.exe";

/// 查找系统中可用的 ffmpeg 可执行文件路径
fn find_ffmpeg() -> Option<String> {
    // 1. 环境变量 FFMPEG_PATH
    if let Ok(path) = std::env::var("FFMPEG_PATH") {
        if Path::new(&path).exists() {
            return Some(path);
        }
    }

    // 2. 默认路径
    if Path::new(DEFAULT_FFMPEG_PATH).exists() {
        return Some(DEFAULT_FFMPEG_PATH.to_string());
    }

    // 3. PATH 中查找 ffmpeg
    if let Ok(path) = which::which("ffmpeg") {
        return Some(path.to_string_lossy().to_string());
    }

    None
}

/// 检查 ffmpeg 是否可用
pub fn is_ffmpeg_available() -> bool {
    find_ffmpeg().is_some()
}

/// 音频后处理选项（T5：响度归一 + 输出语速）
///
/// - `loudnorm`: EBU R128 响度归一化（I=-16 LUFS, TP=-1.5 dBTP）——
///   消除不同引擎/不同次合成之间的响度波动，全书听感一致
/// - `tempo`: atempo 输出变速（0.5~2.0，保音调），与引擎级 speed 参数独立
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EncodeOpts {
    pub loudnorm: bool,
    pub tempo: Option<f32>,
}

impl EncodeOpts {
    /// 构建 ffmpeg `-af` 滤镜链（无滤镜返回 None）
    pub fn filter_chain(&self) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        if let Some(t) = self.tempo {
            let t = t.clamp(0.5, 2.0);
            if (t - 1.0).abs() > 1e-6 {
                parts.push(format!("atempo={:.3}", t));
            }
        }
        if self.loudnorm {
            parts.push("loudnorm=I=-16:TP=-1.5:LRA=11".to_string());
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(","))
        }
    }
}

/// ffmpeg 编码器结构体
pub struct FfmpegEncoder {
    ffmpeg_path: String,
}

impl FfmpegEncoder {
    /// 创建编码器实例，自动查找 ffmpeg
    pub fn new() -> Result<Self> {
        let ffmpeg_path = find_ffmpeg().context(
            "未找到 ffmpeg，请安装 ffmpeg 并将其所在目录添加到 PATH 环境变量。\n\
             下载地址: https://ffmpeg.org/download.html\n\
             或设置环境变量 FFMPEG_PATH 指向 ffmpeg.exe 的完整路径"
        )?;

        Ok(Self { ffmpeg_path })
    }

    /// 使用指定路径创建编码器
    pub fn new_with_path(ffmpeg_path: impl Into<String>) -> Self {
        Self {
            ffmpeg_path: ffmpeg_path.into(),
        }
    }

    /// 将 WAV 文件转换为 MP3 文件
    ///
    /// # 参数
    /// - `input_path`: 输入的 WAV 文件路径
    /// - `output_path`: 输出的 MP3 文件路径
    /// - `bitrate`: MP3 比特率，如 "192k"、"256k"、"320k"，默认 "192k"
    pub fn wav_to_mp3(
        &self,
        input_path: &Path,
        output_path: &Path,
        bitrate: Option<&str>,
    ) -> Result<()> {
        self.wav_to_mp3_opts(input_path, output_path, bitrate, EncodeOpts::default())
    }

    /// WAV → MP3（含后处理选项）
    pub fn wav_to_mp3_opts(
        &self,
        input_path: &Path,
        output_path: &Path,
        bitrate: Option<&str>,
        opts: EncodeOpts,
    ) -> Result<()> {
        if !input_path.exists() {
            anyhow::bail!("输入 WAV 文件不存在: {:?}", input_path);
        }

        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .context("创建输出目录失败")?;
        }

        let bitrate = bitrate.unwrap_or("192k");
        let filter = opts.filter_chain();

        tracing::debug!(
            "ffmpeg 转换: {:?} → {:?} (比特率: {}, 滤镜: {:?})",
            input_path, output_path, bitrate, filter
        );

        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.arg("-y")
            .arg("-i")
            .arg(input_path);
        if let Some(f) = &filter {
            cmd.args(["-af", f]);
        }
        cmd.args(["-codec:a", "libmp3lame"])
            .args(["-b:a", bitrate])
            .args(["-ar", "24000"]) // 保持 24kHz 采样率
            .args(["-ac", "1"]) // 单声道
            .arg(output_path);
        let output = cmd.output().context("执行 ffmpeg 命令失败")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::error!("ffmpeg 转换失败: {}", stderr);
            anyhow::bail!("ffmpeg 转换失败: {}", stderr);
        }

        tracing::info!(
            "MP3 编码完成: {:?} → {:?} (比特率: {})",
            input_path, output_path, bitrate
        );
        Ok(())
    }

    /// 将 WAV 文件按目标扩展名编码为对应音频格式
    ///
    /// 支持 m4a (AAC) / flac / ogg / mp3——修复此前「M4A/FLAC 写出
    /// 实为 WAV 内容」的假格式问题。找不到 ffmpeg 时明确报错（不静默降级）。
    pub fn wav_to_format(&self, input_path: &Path, output_path: &Path) -> Result<()> {
        self.wav_to_format_opts(input_path, output_path, EncodeOpts::default())
    }

    /// WAV → 目标格式（含后处理选项）
    pub fn wav_to_format_opts(
        &self,
        input_path: &Path,
        output_path: &Path,
        opts: EncodeOpts,
    ) -> Result<()> {
        if !input_path.exists() {
            anyhow::bail!("输入 WAV 文件不存在: {:?}", input_path);
        }
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).context("创建输出目录失败")?;
        }

        let ext = output_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();

        let (codec, extra_args): (&str, Vec<&str>) = match ext.as_str() {
            "m4a" | "mp4" | "aac" => ("aac", vec!["-b:a", "192k"]),
            "flac" => ("flac", vec![]),
            "ogg" | "opus" => ("libopus", vec!["-b:a", "96k"]),
            "mp3" => ("libmp3lame", vec!["-b:a", "192k"]),
            other => anyhow::bail!("不支持的音频输出格式: .{}", other),
        };

        let filter = opts.filter_chain();
        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.arg("-y")
            .arg("-i")
            .arg(input_path);
        if let Some(f) = &filter {
            cmd.args(["-af", f]);
        }
        cmd.args(["-codec:a", codec])
            .args(&extra_args)
            .arg(output_path);
        let output = cmd.output().context("执行 ffmpeg 命令失败")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::error!("ffmpeg 编码({})失败: {}", ext, stderr);
            anyhow::bail!("ffmpeg 编码 .{} 失败: {}", ext, stderr);
        }

        tracing::info!("音频编码完成: {:?} → {} ({:?})", input_path, ext, codec);
        Ok(())
    }

    /// 将任意 ffmpeg 支持的音频文件解码为标准 WAV（PCM）
    ///
    /// ASR 只能直接读取 WAV，MP3/M4A/FLAC 等格式经此转码后再喂给识别引擎。
    /// `-vn` 丢弃 m4b/mp4 等容器中的视频轨；不强制采样率/声道，
    /// 由调用方的重采样路径统一处理。
    pub fn decode_to_wav(&self, input_path: &Path, output_path: &Path) -> Result<()> {
        if !input_path.exists() {
            anyhow::bail!("输入音频文件不存在: {:?}", input_path);
        }
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).context("创建转码输出目录失败")?;
        }

        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.arg("-y")
            .arg("-i")
            .arg(input_path)
            .args(["-vn", "-map", "a:0?", "-f", "wav"])
            .arg(output_path);
        let output = cmd.output().context("执行 ffmpeg 命令失败")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::error!("ffmpeg 解码失败: {}", stderr);
            anyhow::bail!("ffmpeg 解码 {:?} 失败: {}", input_path, stderr);
        }

        tracing::info!("音频解码完成: {:?} → {:?}", input_path, output_path);
        Ok(())
    }

    /// 将 WAV 编码为带章节元数据的 m4b 有声书
    ///
    /// `chapters`: (标题, 起始毫秒) 序列，按起始时间升序；
    /// 最后一章的结束时间为 `total_ms`，其余章节结束于下一章起始。
    /// 章节表写入 ffmpeg FFMETADATA 文件后一次性封装（AAC 编码 + 章节映射）。
    /// 空章节列表则产出无章节的纯 m4b。
    pub fn wav_to_m4b(
        &self,
        input_path: &Path,
        output_path: &Path,
        chapters: &[(String, u64)],
        total_ms: u64,
    ) -> Result<()> {
        self.wav_to_m4b_opts(input_path, output_path, chapters, total_ms, EncodeOpts::default())
    }

    /// WAV → m4b（含后处理选项）
    pub fn wav_to_m4b_opts(
        &self,
        input_path: &Path,
        output_path: &Path,
        chapters: &[(String, u64)],
        total_ms: u64,
        opts: EncodeOpts,
    ) -> Result<()> {
        if !input_path.exists() {
            anyhow::bail!("输入 WAV 文件不存在: {:?}", input_path);
        }
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent).context("创建输出目录失败")?;
        }

        // 写临时 FFMETADATA 文件（章节表）
        let meta_path = output_path.with_extension("ffmeta.tmp");
        let mut meta = String::from(";FFMETADATA1\ntitle=Audiobook\n");
        for (i, (title, start_ms)) in chapters.iter().enumerate() {
            let end_ms = chapters.get(i + 1).map(|(_, s)| *s).unwrap_or(total_ms);
            if end_ms <= *start_ms {
                continue; // 零时长章节跳过
            }
            meta.push_str(&format!(
                "[CHAPTER]\nTIMEBASE=1/1000\nSTART={}\nEND={}\ntitle={}\n",
                start_ms,
                end_ms,
                escape_ffmetadata(title),
            ));
        }
        std::fs::write(&meta_path, &meta).context("写入章节元数据文件失败")?;

        let filter = opts.filter_chain();
        let mut cmd = Command::new(&self.ffmpeg_path);
        cmd.arg("-y")
            .arg("-i")
            .arg(input_path)
            .arg("-i")
            .arg(&meta_path)
            .args(["-map", "0:a"])
            .args(["-map_metadata", "1"]);
        if let Some(f) = &filter {
            cmd.args(["-af", f]);
        }
        cmd.args(["-codec:a", "aac", "-b:a", "192k"])
            .arg(output_path);
        let output = cmd
            .output()
            .context("执行 ffmpeg 命令失败");

        let _ = std::fs::remove_file(&meta_path);
        let output = output?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::error!("ffmpeg m4b 封装失败: {}", stderr);
            anyhow::bail!("ffmpeg 封装 m4b 失败: {}", stderr);
        }

        tracing::info!(
            "m4b 有声书封装完成: {:?} ({} 章节, 总时长 {:.1}s)",
            output_path,
            chapters.len(),
            total_ms as f64 / 1000.0
        );
        Ok(())
    }

    /// 获取 ffmpeg 版本信息
    pub fn version(&self) -> Result<String> {
        let output = Command::new(&self.ffmpeg_path)
            .arg("-version")
            .output()
            .context("获取 ffmpeg 版本失败")?;

        let version_info = String::from_utf8_lossy(&output.stdout).to_string();
        // 只取第一行
        Ok(version_info
            .lines()
            .next()
            .unwrap_or("未知版本")
            .to_string())
    }
}

/// 便捷函数：直接将 WAV 转换为 MP3（使用默认编码器）
pub fn wav_to_mp3(input_path: &Path, output_path: &Path) -> Result<()> {
    let encoder = FfmpegEncoder::new()?;
    encoder.wav_to_mp3(input_path, output_path, None)
}

/// FFMETADATA 值转义（`;` `#` `=` `\` 与换行为保留字符）
fn escape_ffmetadata(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('=', "\\=")
        .replace(';', "\\;")
        .replace('#', "\\#")
        .replace('\n', " ")
        .replace('\r', "")
}

/// 便捷函数：将 AudioData 直接编码为 MP3 文件
///
/// 内部流程：AudioData → 临时 WAV 文件 → ffmpeg 转为 MP3
pub fn encode_audio_to_mp3(
    audio: &votex_domain::shared::value_object::AudioData,
    output_path: &Path,
) -> Result<()> {
    let encoder = FfmpegEncoder::new()?;

    // 写入临时 WAV 文件
    let temp_dir = tempfile::tempdir().context("创建临时目录失败")?;
    let temp_wav = temp_dir.path().join("temp_audio.wav");
    super::wav::WavWriter::write(audio, &temp_wav)?;

    // 转换为 MP3
    encoder.wav_to_mp3(&temp_wav, output_path, None)?;

    Ok(())
}

/// 便捷函数：将多个 AudioData 拼接后编码为 MP3 文件
///
/// 内部流程：拼接 AudioData → 临时 WAV 文件 → ffmpeg 转为 MP3
/// 使用 votex-infra 内部的 tempfile 依赖，避免调用方引入额外依赖
pub fn concat_and_encode_to_mp3(
    audio_segments: &[votex_domain::shared::value_object::AudioData],
    output_path: &Path,
) -> Result<()> {
    if audio_segments.is_empty() {
        anyhow::bail!("音频段列表为空");
    }

    let encoder = FfmpegEncoder::new()?;

    // 拼接音频数据
    let sample_rate = audio_segments[0].sample_rate;
    let channels = audio_segments[0].channels;

    let mut all_samples = Vec::new();
    for seg in audio_segments {
        if seg.sample_rate != sample_rate || seg.channels != channels {
            anyhow::bail!("音频段采样率或声道数不一致");
        }
        all_samples.extend_from_slice(&seg.samples);
    }

    let combined = votex_domain::shared::value_object::AudioData {
        samples: all_samples,
        sample_rate,
        channels,
    };

    // 写入临时 WAV 文件并转换为 MP3
    let temp_dir = tempfile::tempdir().context("创建临时目录失败")?;
    let temp_wav = temp_dir.path().join("temp_audio.wav");
    super::wav::WavWriter::write(&combined, &temp_wav)?;

    encoder.wav_to_mp3(&temp_wav, output_path, None)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use votex_domain::shared::value_object::AudioData;

    #[test]
    fn encode_opts_滤镜链() {
        assert_eq!(EncodeOpts::default().filter_chain(), None);
        let o = EncodeOpts { loudnorm: true, tempo: None };
        assert_eq!(o.filter_chain().as_deref(), Some("loudnorm=I=-16:TP=-1.5:LRA=11"));
        let o = EncodeOpts { loudnorm: false, tempo: Some(1.5) };
        assert_eq!(o.filter_chain().as_deref(), Some("atempo=1.500"));
        let o = EncodeOpts { loudnorm: true, tempo: Some(0.75) };
        assert_eq!(
            o.filter_chain().as_deref(),
            Some("atempo=0.750,loudnorm=I=-16:TP=-1.5:LRA=11")
        );
        // 变速 1.0 与越界值处理
        let o = EncodeOpts { loudnorm: false, tempo: Some(1.0) };
        assert_eq!(o.filter_chain(), None);
        let o = EncodeOpts { loudnorm: false, tempo: Some(3.0) };
        assert_eq!(o.filter_chain().as_deref(), Some("atempo=2.000"));
    }

    #[test]
    fn test_ffmpeg_查找() {
        // 不验证具体路径，只确认方法可执行
        let result = find_ffmpeg();
        // 如果系统有 ffmpeg，则应有返回值
        if result.is_some() {
            println!("找到 ffmpeg: {:?}", result.unwrap());
        }
    }

    #[test]
    fn test_编码器创建() {
        let result = FfmpegEncoder::new();
        // 取决于系统是否安装 ffmpeg
        if result.is_ok() {
            assert!(is_ffmpeg_available());
        }
    }

    #[test]
    fn test_简易wav转mp3() {
        // 创建测试音频
        let dir = tempdir().unwrap();
        let wav_path = dir.path().join("test.wav");
        let mp3_path = dir.path().join("test.mp3");

        let audio = AudioData::silence(24000, 500);
        crate::audio::wav::WavWriter::write(&audio, &wav_path).unwrap();

        // 如果 ffmpeg 可用则测试转换
        if is_ffmpeg_available() {
            let encoder = FfmpegEncoder::new().unwrap();
            let result = encoder.wav_to_mp3(&wav_path, &mp3_path, Some("128k"));
            assert!(result.is_ok(), "MP3 转换失败: {:?}", result.err().unwrap());
            assert!(mp3_path.exists(), "MP3 文件未生成");
            assert!(mp3_path.metadata().unwrap().len() > 0, "MP3 文件为空");

            tracing::info!("测试 MP3 文件大小: {} 字节", mp3_path.metadata().unwrap().len());
        } else {
            tracing::warn!("跳过 MP3 转换测试: ffmpeg 不可用");
        }
    }

    #[test]
    fn test_encode_audio_to_mp3() {
        let dir = tempdir().unwrap();
        let mp3_path = dir.path().join("output.mp3");
        let audio = AudioData::silence(24000, 1000);

        if is_ffmpeg_available() {
            let result = encode_audio_to_mp3(&audio, &mp3_path);
            assert!(result.is_ok(), "编码失败: {:?}", result.err().unwrap());
            assert!(mp3_path.exists());
        }
    }
}
