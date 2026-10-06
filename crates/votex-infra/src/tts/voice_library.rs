//! 声音克隆音色库（T4）
//!
//! 管理用户克隆音色的参考音频：入库 → 校验 → 自动降噪 → 存档 + 元数据。
//!
//! 目录结构（`models/voices/refs/`）：
//! ```text
//! {name}.wav       # 降噪后的参考音频（IndexTTS-2.5/CosyVoice 零样本克隆输入）
//! {name}.meta.json # 元数据（原始时长、降噪标记、创建时间）
//! ```
//!
//! 安全约束（AGENTS.md）：禁止自动删除音色文件——`remove` 必须显式
//! 传入 `confirm = true`，且不允许路径穿越字符。

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// 音色元数据
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloneVoiceMeta {
    /// 音色名（同时是文件名主体）
    pub name: String,
    /// 参考音频文件名
    pub ref_file: String,
    /// 参考音频时长（毫秒）
    pub duration_ms: u64,
    /// 入库时是否已降噪
    pub denoised: bool,
    /// 采样率
    pub sample_rate: u32,
    /// 创建时间（ISO 8601）
    pub created_at: String,
}

/// 音色库根目录（models/voices/refs/）
pub fn base_dir() -> PathBuf {
    crate::shared::workspace_paths::WorkspacePaths::models_dir()
        .join("voices")
        .join("refs")
}

/// 校验音色名（禁止路径穿越/危险字符）
fn validate_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 64
        || name.contains('/')
        || name.contains('\\')
        || name.contains("..")
        || name.contains('\0')
        || name.chars().any(|c| c.is_control())
    {
        return Err(anyhow!(
            "非法音色名: 只允许 1~64 个不含路径分隔符/控制字符的字符"
        ));
    }
    Ok(())
}

/// 把参考音频入库（校验 → 可选降噪 → 存档 + 元数据）
///
/// - 参考音频需为 WAV，时长 3 秒 ~ 10 分钟（IndexTTS-2.5 建议不超过 15 秒）
/// - `denoise = true` 时自动降噪（复用项目 denoiser，Low 档）
/// - 已存在同名音色会报错（不静默覆盖）
pub fn add(name: &str, ref_wav: &Path, denoise: bool) -> Result<CloneVoiceMeta> {
    validate_name(name)?;
    if !ref_wav.exists() {
        return Err(anyhow!("参考音频不存在: {:?}", ref_wav));
    }

    // 读取 WAV（采样率/声道随源文件保留）
    let mut audio = crate::audio::wav::read_wav(ref_wav)
        .with_context(|| format!("读取参考音频失败（需 WAV 格式）: {:?}", ref_wav))?;
    let duration_ms = audio.duration_ms() as u64;
    if !(3000..=600_000).contains(&duration_ms) {
        return Err(anyhow!(
            "参考音频时长 {}ms 不在 3 秒 ~ 10 分钟范围内",
            duration_ms
        ));
    }

    // 可选降噪
    if denoise {
        audio = crate::audio::denoiser::denoise(&audio, votex_domain::tts::value_object::DenoiseLevel::Low)
            .context("参考音频降噪失败")?;
    }

    // 存档
    let dir = base_dir();
    std::fs::create_dir_all(&dir).context("创建音色库目录失败")?;
    let wav_path = dir.join(format!("{}.wav", name));
    if wav_path.exists() {
        return Err(anyhow!("音色已存在: {}（如需覆盖请先删除）", name));
    }
    votex_infra_wav_write(&audio, &wav_path)?;

    let meta = CloneVoiceMeta {
        name: name.to_string(),
        ref_file: wav_path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default(),
        duration_ms,
        denoised: denoise,
        sample_rate: audio.sample_rate,
        created_at: chrono::Utc::now().to_rfc3339(),
    };
    std::fs::write(meta_path(name), serde_json::to_string_pretty(&meta)?)?;
    Ok(meta)
}

/// 列出全部克隆音色
pub fn list() -> Vec<CloneVoiceMeta> {
    std::fs::read_dir(base_dir())
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| e.path().extension().map(|x| x == "json").unwrap_or(false))
                .filter_map(|e| {
                    let s = std::fs::read_to_string(e.path()).ok()?;
                    serde_json::from_str(&s).ok()
                })
                .collect()
        })
        .unwrap_or_default()
}

/// 删除音色（⚠️ 破坏性操作：必须显式确认）
///
/// 返回 Ok(false) 表示音色不存在。
pub fn remove(name: &str, confirm: bool) -> Result<bool> {
    validate_name(name)?;
    if !confirm {
        return Err(anyhow!(
            "删除音色 {} 需要显式确认（禁止自动删除音色文件）",
            name
        ));
    }
    let wav = base_dir().join(format!("{}.wav", name));
    let meta = meta_path(name);
    let existed = wav.exists() || meta.exists();
    if wav.exists() {
        std::fs::remove_file(&wav).context("删除参考音频失败")?;
    }
    if meta.exists() {
        let _ = std::fs::remove_file(&meta);
    }
    Ok(existed)
}

/// 音色的参考音频路径（供引擎做零样本克隆输入）
pub fn reference_path(name: &str) -> Result<PathBuf> {
    validate_name(name)?;
    let p = base_dir().join(format!("{}.wav", name));
    if !p.exists() {
        return Err(anyhow!("音色不存在: {}（请先添加）", name));
    }
    Ok(p)
}

fn meta_path(name: &str) -> PathBuf {
    base_dir().join(format!("{}.meta.json", name))
}

fn votex_infra_wav_write(
    audio: &votex_domain::shared::value_object::AudioData,
    path: &Path,
) -> Result<()> {
    crate::audio::wav::WavWriter::write(audio, path)
        .map_err(|e| anyhow!("写参考音频失败: {:?}", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_wav(dir: &Path, name: &str, sample_rate: u32, ms: u64) -> PathBuf {
        let audio = votex_domain::shared::value_object::AudioData::silence(sample_rate, ms as _);
        let p = dir.join(name);
        crate::audio::wav::WavWriter::write(&audio, &p).unwrap();
        p
    }

    #[test]
    fn 音色名_路径穿越拒绝() {
        assert!(validate_name("../etc").is_err());
        assert!(validate_name("a/b").is_err());
        assert!(validate_name("").is_err());
        assert!(validate_name("我的音色").is_ok());
    }

    #[test]
    fn 音色入库_列表_删除() {
        let dir = tempfile::tempdir().unwrap();
        let wav = make_wav(dir.path(), "ref.wav", 24000, 4000);
        assert!(wav.exists());
        let audio = crate::audio::wav::read_wav(&wav).unwrap();
        assert_eq!(audio.sample_rate, 24000);
        assert_eq!(audio.duration_ms(), 4000);
    }

    #[test]
    fn 删除未确认拒绝() {
        assert!(remove("任意名", false).is_err());
    }
}
