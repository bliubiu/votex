//! 视频合成器
//!
//! 基于 FFmpeg CLI 实现视频合成：背景画面 + 配音音频 + 可选字幕烧录。

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// FFmpeg 视频合成器
pub struct VideoComposer;

impl VideoComposer {
    /// 合成视频
    ///
    /// * `audio_path` - TTS 配音音频文件路径
    /// * `subtitle_path` - 可选字幕文件路径（SRT 格式）
    /// * `bg_images` - 可选背景图片列表（循环播放）
    /// * `bg_music` - 可选背景音乐路径
    /// * `output_path` - 输出视频文件路径
    /// * `resolution` - 输出分辨率，如 "1920x1080"
    pub fn compose(
        audio_path: &Path,
        subtitle_path: Option<&Path>,
        bg_images: Option<&[PathBuf]>,
        bg_music: Option<&Path>,
        output_path: &Path,
        resolution: &str,
    ) -> Result<()> {
        let mut cmd = Command::new("ffmpeg");
        cmd.arg("-y");

        // 输入 1：背景画面
        if let Some(images) = bg_images {
            if !images.is_empty() {
                let list_path = Path::new("_votex_images.txt");
                let mut list_content = String::new();
                for img in images {
                    list_content.push_str(&format!(
                        "file '{}'\nduration 5.0\n",
                        img.display()
                    ));
                }
                std::fs::write(&list_path, &list_content)
                    .context("写入图片列表失败")?;
                cmd.args([
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_path.to_str().unwrap(),
                ]);
            } else {
                // 纯色背景
                cmd.args(["-f", "lavfi", "-i", "color=c=#2C3E50:s=1920x1080:d=30"]);
            }
        } else {
            cmd.args(["-f", "lavfi", "-i", "color=c=#2C3E50:s=1920x1080:d=30"]);
        }

        // 输入 2：配音音频
        cmd.args(["-i", &audio_path.to_string_lossy()]);

        // 滤镜链
        let mut vf_parts = vec![format!("scale={},setsar=1", resolution)];
        if let Some(sub) = subtitle_path {
            let escaped = sub.to_string_lossy().replace(":", "\\:");
            vf_parts.push(format!("subtitles='{}'", escaped));
        }

        cmd.args(["-vf", &vf_parts.join(",")]);
        cmd.args([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
        ]);

        // 可选背景音乐混音
        if let Some(music) = bg_music {
            cmd.args(["-i", &music.to_string_lossy()]);
            cmd.args([
                "-filter_complex",
                "[1:a][2:a]amix=inputs=2:duration=first[aout]",
                "-map", "0:v",
                "-map", "[aout]",
            ]);
        }

        cmd.arg(output_path.as_os_str());

        let status = cmd.status().context("ffmpeg 执行失败")?;
        if !status.success() {
            anyhow::bail!("ffmpeg 合成返回非零退出码: {:?}", status.code());
        }

        Ok(())
    }
}
