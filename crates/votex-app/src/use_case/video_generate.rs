//! 短视频生成用例
//!
//! 编排 TTS 配音、ASR 字幕和视频合成流程，一键生成短视频。

use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use votex_domain::tts::value_object::AudioFormat;

/// 视频生成请求
#[derive(Debug, Clone)]
pub struct VideoGenerateRequest {
    /// 文案内容
    pub script: String,
    /// TTS 引擎
    pub tts_engine: String,
    /// 音色
    pub voice: String,
    /// 语速
    pub speed: f32,
    /// 是否生成字幕
    pub subtitle: bool,
    /// 字幕格式
    pub subtitle_format: String,
    /// ASR 引擎
    pub asr_engine: String,
    /// 输出分辨率
    pub resolution: String,
    /// 背景图片路径列表
    pub bg_images: Option<Vec<String>>,
    /// 背景音乐路径
    pub bg_music: Option<String>,
}

/// 视频生成响应
#[derive(Debug, Clone)]
pub struct VideoGenerateResponse {
    /// 输出文件路径
    pub output_path: String,
    /// 视频时长（秒）
    pub duration_secs: f64,
}

/// 短视频生成用例
pub struct VideoGenerateUseCase;

impl VideoGenerateUseCase {
    /// 执行视频生成
    ///
    /// `cancel` 置 true 后在各阶段边界中断（返回错误，临时文件照常清理）。
    pub fn execute(
        &self,
        req: VideoGenerateRequest,
        output_path: &str,
        cancel: Option<std::sync::Arc<AtomicBool>>,
    ) -> Result<VideoGenerateResponse> {
        // 临时文件放系统临时目录并带进程 ID + 时间戳：
        // 此前写死在进程 cwd（GUI 双击启动时 cwd 任意）且无唯一性，
        // 并发两次合成互相覆盖
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let audio_path = std::env::temp_dir().join(format!(
            "votex_video_audio_{}_{}.wav",
            std::process::id(),
            unique
        ));
        let subtitle_path = std::env::temp_dir().join(format!(
            "votex_video_subtitle_{}_{}.{}",
            std::process::id(),
            unique,
            if req.subtitle_format.is_empty() { "srt" } else { &req.subtitle_format }
        ));

        let result = self.run(&req, &audio_path, &subtitle_path, output_path, cancel);

        // 无论成败都清理临时文件（此前失败路径会残留）
        let _ = std::fs::remove_file(&audio_path);
        let _ = std::fs::remove_file(&subtitle_path);

        result
    }

    fn run(
        &self,
        req: &VideoGenerateRequest,
        audio_path: &Path,
        subtitle_path: &Path,
        output_path: &str,
        cancel: Option<std::sync::Arc<AtomicBool>>,
    ) -> Result<VideoGenerateResponse> {
        let cancel_flag = cancel.clone();
        let cancelled = move |stage: &str| -> Result<()> {
            if let Some(c) = cancel_flag.as_deref() {
                if c.load(Ordering::SeqCst) {
                    anyhow::bail!("任务已取消（{} 阶段）", stage);
                }
            }
            Ok(())
        };

        // 1. TTS 合成配音
        cancelled("TTS 合成")?;
        let engine = votex_domain::tts::value_object::parse_tts_engine(&req.tts_engine)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "不支持的 TTS 引擎: {}，可选: kokoro / indextts25 / qwen3-tts / cosyvoice3",
                    req.tts_engine
                )
            })?;
        {
            let tts = crate::services::shared_cases::shared_tts();
            tts.synthesize_ext(
                &req.script,
                audio_path,
                engine,
                &req.voice,
                req.speed,
                AudioFormat::Wav,
                None,
                None,
                None,
                cancel.clone(),
                None,
                &Default::default(),
            )
            .context("TTS 合成失败")?;
        }

        // 2. ASR 字幕（可选）
        let subtitle = if req.subtitle {
            cancelled("ASR 字幕")?;
            let asr = crate::services::shared_cases::shared_asr();
            asr.recognize(
                audio_path,
                subtitle_path,
                &req.asr_engine,
                "zh",
                &req.subtitle_format,
                cancel.as_deref(),
            )
            .context("ASR 识别失败")?;
            Some(subtitle_path)
        } else {
            None
        };

        // 3. 合成视频
        cancelled("视频合成")?;
        let output = Path::new(output_path);
        self.compose_video(
            audio_path,
            subtitle.as_deref(),
            req.bg_images.as_ref(),
            req.bg_music.as_ref(),
            output,
            &req.resolution,
        )?;

        // 估算时长（按语速 4字/秒）
        let duration_secs = req.script.chars().count() as f64 / 4.0 / req.speed as f64;

        Ok(VideoGenerateResponse {
            output_path: output_path.to_string(),
            duration_secs,
        })
    }

    /// 使用 ffmpeg 合成视频
    fn compose_video(
        &self,
        audio_path: &Path,
        subtitle_path: Option<&Path>,
        bg_images: Option<&Vec<String>>,
        bg_music: Option<&String>,
        output_path: &Path,
        resolution: &str,
    ) -> Result<()> {
        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-y");

        // 背景
        if let Some(images) = bg_images {
            if !images.is_empty() {
                let unique = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or(0);
                let list_path = std::env::temp_dir().join(format!(
                    "votex_video_images_{}_{}.txt",
                    std::process::id(),
                    unique
                ));
                let mut list_content = String::new();
                for img in images {
                    list_content.push_str(&format!("file '{}'\nduration 5.0\n", img));
                }
                std::fs::write(&list_path, &list_content).context("写入图片列表失败")?;
                cmd.args(["-f", "concat", "-safe", "0", "-i"]);
                cmd.arg(&list_path);
            } else {
                cmd.args(["-f", "lavfi", "-i", "color=c=#2C3E50:s=1920x1080:d=30"]);
            }
        } else {
            cmd.args(["-f", "lavfi", "-i", "color=c=#2C3E50:s=1920x1080:d=30"]);
        }

        cmd.args(["-i", &audio_path.to_string_lossy()]);

        let mut vf_parts = vec![format!("scale={},setsar=1", resolution)];
        if let Some(sub) = subtitle_path {
            let escaped = sub.to_string_lossy().replace(":", "\\:");
            vf_parts.push(format!("subtitles='{}'", escaped));
        }

        cmd.args(["-vf", &vf_parts.join(",")]);
        cmd.args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest"]);

        if let Some(music) = bg_music {
            cmd.args(["-i", music]);
            cmd.args(["-filter_complex", "[1:a][2:a]amix=inputs=2:duration=first[aout]", "-map", "0:v", "-map", "[aout]"]);
        }

        cmd.arg(output_path.as_os_str());

        let status = cmd.status().context("ffmpeg 执行失败")?;
        if !status.success() {
            anyhow::bail!("ffmpeg 合成返回非零退出码");
        }

        Ok(())
    }
}
