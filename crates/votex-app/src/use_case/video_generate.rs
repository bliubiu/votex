//! 短视频生成用例
//!
//! 编排 TTS 配音、ASR 字幕和视频合成流程，一键生成短视频。

use anyhow::{Context, Result};
use std::path::Path;
use std::process::Command;
use votex_domain::model::value_object::EngineKind;
use votex_domain::tts::value_object::AudioFormat;
use crate::use_case::tts_use_case::TtsUseCase;
use crate::use_case::asr_use_case::AsrUseCase;

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
    pub fn execute(&self, req: VideoGenerateRequest, output_path: &str) -> Result<VideoGenerateResponse> {
        let audio_path = Path::new("_temp_audio.wav");

        // 1. TTS 合成配音
        {
            let mut tts = TtsUseCase::new();
            let engine = parse_engine(&req.tts_engine);
            tts.synthesize(
                &req.script,
                audio_path,
                engine,
                &req.voice,
                req.speed,
                AudioFormat::Wav,
                None,
                None,
                None,
            )
            .context("TTS 合成失败")?;
        }

        // 2. ASR 字幕（可选）
        let subtitle_path = if req.subtitle {
            let srt_path = Path::new("_temp_subtitle.srt");
            let mut asr = AsrUseCase::new();
            asr.recognize(
                audio_path,
                srt_path,
                &req.asr_engine,
                "zh",
                &req.subtitle_format,
            )
            .context("ASR 识别失败")?;
            Some(srt_path.to_path_buf())
        } else {
            None
        };

        // 3. 合成视频
        let output = Path::new(output_path);
        self.compose_video(
            audio_path,
            subtitle_path.as_deref(),
            req.bg_images.as_ref(),
            req.bg_music.as_ref(),
            output,
            &req.resolution,
        )?;

        // 4. 清理临时文件
        let _ = std::fs::remove_file(audio_path);
        if let Some(ref sub) = subtitle_path {
            let _ = std::fs::remove_file(sub);
        }

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
                let list_path = Path::new("_votex_images.txt");
                let mut list_content = String::new();
                for img in images {
                    list_content.push_str(&format!("file '{}'\nduration 5.0\n", img));
                }
                std::fs::write(list_path, &list_content).context("写入图片列表失败")?;
                cmd.args(["-f", "concat", "-safe", "0", "-i", list_path.to_str().unwrap()]);
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

/// 解析引擎名称
fn parse_engine(s: &str) -> EngineKind {
    match s.to_lowercase().as_str() {
        "kokoro" => EngineKind::Kokoro,
        "indextts2" | "indextts" => EngineKind::IndexTTS2,
        "whisper" => EngineKind::Whisper,
        "sensevoice" => EngineKind::SenseVoice,
        _ => EngineKind::Kokoro,
    }
}
