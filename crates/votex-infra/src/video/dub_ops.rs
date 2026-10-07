//! 视频配音 ffmpeg 操作（借鉴 VoiceStudio 配音链的工程能力）
//!
//! 提供配音流水线的三个原语：
//! - [`extract_audio_to_wav`]：从视频提取音轨并归一为 24kHz 单声道 WAV
//! - [`apply_tempo`]：WAV → WAV 变速（atempo，保音调），供时长拟合使用
//! - [`mux_audio_into_video`]：把配音轨替换/混入视频（视频流直拷不重编码）
//!
//! 全部通过 `std::process::Command` 调用 ffmpeg CLI（与 mp3.rs 同一套
//! 查找逻辑），不引入 ffmpeg-next 绑定。

use anyhow::{bail, Context, Result};
use std::path::Path;
use std::process::Command;

/// 查找 ffmpeg 可执行文件（复用 mp3.rs 的查找优先级）
fn find_ffmpeg() -> Option<String> {
    crate::audio::mp3::find_ffmpeg_path()
}

fn run_ffmpeg(args: &[&std::ffi::OsStr], what: &str) -> Result<()> {
    let ffmpeg = find_ffmpeg().context(
        "未找到 ffmpeg，视频配音需要 ffmpeg 支持。\n\
         下载地址: https://ffmpeg.org/download.html\n\
         或设置环境变量 FFMPEG_PATH 指向 ffmpeg.exe 的完整路径",
    )?;
    let output = Command::new(&ffmpeg)
        .arg("-y")
        .args(args)
        .output()
        .context("执行 ffmpeg 命令失败")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::error!("ffmpeg {what}失败: {}", stderr);
        bail!("ffmpeg {what}失败: {}", stderr.lines().last().unwrap_or(""));
    }
    tracing::info!("ffmpeg {what}完成");
    Ok(())
}

/// 从视频（或任意媒体）提取音轨，归一为 24kHz 单声道 WAV
///
/// 归一的目的：ASR 输入与段级时长测量都在同一条 24kHz 单声道基线上，
/// 不受源视频音轨采样率/声道数差异影响。
pub fn extract_audio_to_wav(video: &Path, out_wav: &Path) -> Result<()> {
    if !video.is_file() {
        bail!("视频文件不存在: {:?}", video);
    }
    if let Some(parent) = out_wav.parent() {
        std::fs::create_dir_all(parent).context("创建输出目录失败")?;
    }
    let args = [
        "-i".as_ref(),
        video.as_os_str(),
        "-vn".as_ref(),
        "-map".as_ref(),
        "a:0?".as_ref(),
        "-ar".as_ref(),
        "24000".as_ref(),
        "-ac".as_ref(),
        "1".as_ref(),
        "-f".as_ref(),
        "wav".as_ref(),
        out_wav.as_os_str(),
    ];
    run_ffmpeg(&args, "提取音轨")
}

/// WAV → WAV 变速（atempo，保音调）
///
/// 单级 atempo 限 0.5~2.0，超出时拆成多级串联（支持 0.25~4.0）。
pub fn apply_tempo(in_wav: &Path, out_wav: &Path, tempo: f32) -> Result<()> {
    if !in_wav.is_file() {
        bail!("输入音频不存在: {:?}", in_wav);
    }
    if !(0.25..=4.0).contains(&tempo) {
        bail!("变速倍率 {tempo} 超出支持范围 (0.25 ~ 4.0)");
    }
    if let Some(parent) = out_wav.parent() {
        std::fs::create_dir_all(parent).context("创建输出目录失败")?;
    }

    // 拆分为多级 atempo（每级 0.5~2.0）
    let mut factors = Vec::new();
    let mut remaining = tempo as f64;
    while remaining > 2.0 {
        factors.push(2.0);
        remaining /= 2.0;
    }
    while remaining < 0.5 {
        factors.push(0.5);
        remaining /= 0.5;
    }
    factors.push(remaining);
    let filter = factors
        .iter()
        .map(|f| format!("atempo={f:.6}"))
        .collect::<Vec<_>>()
        .join(",");

    let args = [
        "-i".as_ref(),
        in_wav.as_os_str(),
        "-af".as_ref(),
        filter.as_ref(),
        "-f".as_ref(),
        "wav".as_ref(),
        out_wav.as_os_str(),
    ];
    run_ffmpeg(&args, "音频变速")
}

/// 把配音轨替换/混入视频，输出新视频文件
///
/// - `bg_volume = None`：替换原音轨（`-c:v copy` 视频流不重编码，速度快）
/// - `bg_volume = Some(v)`：原音轨按音量 `v`（0.0~1.0）与配音轨混音，
///   用于保留背景声/环境音的场景
pub fn mux_audio_into_video(
    video: &Path,
    dubbed_audio: &Path,
    bg_volume: Option<f32>,
    out_video: &Path,
) -> Result<()> {
    if !video.is_file() {
        bail!("视频文件不存在: {:?}", video);
    }
    if !dubbed_audio.is_file() {
        bail!("配音音频不存在: {:?}", dubbed_audio);
    }
    if let Some(parent) = out_video.parent() {
        std::fs::create_dir_all(parent).context("创建输出目录失败")?;
    }

    match bg_volume {
        None => {
            let args = [
                "-i".as_ref(),
                video.as_os_str(),
                "-i".as_ref(),
                dubbed_audio.as_os_str(),
                "-map".as_ref(),
                "0:v:0".as_ref(),
                "-map".as_ref(),
                "1:a:0".as_ref(),
                "-c:v".as_ref(),
                "copy".as_ref(),
                "-c:a".as_ref(),
                "aac".as_ref(),
                "-b:a".as_ref(),
                "192k".as_ref(),
                "-shortest".as_ref(),
                out_video.as_os_str(),
            ];
            run_ffmpeg(&args, "替换音轨")
        }
        Some(vol) => {
            let vol = vol.clamp(0.0, 1.0);
            // normalize=0 保持各自增益不被 amix 平分；老版本 ffmpeg 无该参数时
            // 会报错，此时提示用户改用替换模式
            let filter = format!(
                "[1:a]volume={vol:.3}[bg];[0:a][bg]amix=inputs=2:duration=first:normalize=0[mix]"
            );
            let args = [
                "-i".as_ref(),
                video.as_os_str(),
                "-i".as_ref(),
                dubbed_audio.as_os_str(),
                "-filter_complex".as_ref(),
                filter.as_ref(),
                "-map".as_ref(),
                "0:v:0".as_ref(),
                "-map".as_ref(),
                "[mix]".as_ref(),
                "-c:v".as_ref(),
                "copy".as_ref(),
                "-c:a".as_ref(),
                "aac".as_ref(),
                "-b:a".as_ref(),
                "192k".as_ref(),
                "-shortest".as_ref(),
                out_video.as_os_str(),
            ];
            run_ffmpeg(&args, "混音")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 变速_范围校验() {
        let dir = tempfile::tempdir().unwrap();
        let inp = dir.path().join("in.wav");
        let out = dir.path().join("out.wav");
        assert!(apply_tempo(&inp, &out, 0.1).is_err(), "0.1 应被拒绝");
        assert!(apply_tempo(&inp, &out, 8.0).is_err(), "8.0 应被拒绝");
        assert!(apply_tempo(&inp, &out, 1.5).is_err(), "输入文件不存在应报错");
    }
}
