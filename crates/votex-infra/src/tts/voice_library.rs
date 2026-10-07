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
    /// 参考音频文字转写（CosyVoice 零样本克隆必需：prompt 音频必须
    /// 配套转写文本才能条件 LLM；IndexTTS-2.5 不需要）
    #[serde(default)]
    pub transcript: Option<String>,
    /// 入库时因超过引擎参考音频上限被截取的原始时长（毫秒）；未截取为 None
    #[serde(default)]
    pub trimmed_from_ms: Option<u64>,
    /// 截取策略："best_window"（按短时能量选语音最多的窗口）；未截取为 None
    #[serde(default)]
    pub ref_strategy: Option<String>,
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

/// 把参考音频入库（校验 → 可选降噪 → 上限截取 → 存档 + 元数据）
///
/// - 参考音频需为 WAV，时长 3 秒 ~ 10 分钟
/// - `denoise = true` 时自动降噪（复用项目 denoiser，Low 档）
/// - `transcript`：参考音频文字转写（CosyVoice 克隆必需；IndexTTS-2.5 可省略）
/// - `max_ref_seconds`：引擎参考音频上限（如 IndexTTS-2.5 为 15s）。超过时：
///   - 无转写 → 按 best_window 策略自动截取「语音最多」的窗口；
///   - 有转写 → **拒绝入库**（截取窗口与转写文本将无法对应，
///     借鉴 VoiceStudio `clone_ref_too_long` 的显式拒绝而非静默出错）。
/// - 已存在同名音色会报错（不静默覆盖）
pub fn add(
    name: &str,
    ref_wav: &Path,
    denoise: bool,
    transcript: Option<&str>,
    max_ref_seconds: Option<u64>,
) -> Result<CloneVoiceMeta> {
    validate_name(name)?;
    if !ref_wav.exists() {
        return Err(anyhow!("参考音频不存在: {:?}", ref_wav));
    }

    // 读取 WAV（采样率/声道随源文件保留）
    let mut audio = crate::audio::wav::read_wav(ref_wav)
        .with_context(|| format!("读取参考音频失败（需 WAV 格式）: {:?}", ref_wav))?;
    let mut duration_ms = audio.duration_ms() as u64;
    if !(3000..=600_000).contains(&duration_ms) {
        return Err(anyhow!(
            "参考音频时长 {}ms 不在 3 秒 ~ 10 分钟范围内",
            duration_ms
        ));
    }

    // 质量校验：几乎全静音的参考音频克隆出的音色必然失真，提前拦截
    let silence = crate::audio::ref_audio::silence_ratio(&audio);
    if silence > 0.95 {
        return Err(anyhow!(
            "参考音频 {:.0}% 为静音，无法用于克隆（请检查是否选错了文件/录音是否发声）",
            silence * 100.0
        ));
    }

    // 可选降噪
    if denoise {
        audio = crate::audio::denoiser::denoise(&audio, votex_domain::tts::value_object::DenoiseLevel::Low)
            .context("参考音频降噪失败")?;
    }

    // 引擎参考音频上限截取（best_window 策略）
    let mut trimmed_from_ms: Option<u64> = None;
    let mut ref_strategy: Option<String> = None;
    if let Some(max_seconds) = max_ref_seconds {
        let max_ms = max_seconds * 1000;
        if duration_ms > max_ms {
            if transcript.map(|t| !t.trim().is_empty()).unwrap_or(false) {
                return Err(anyhow!(
                    "参考音频时长 {}ms 超过引擎上限 {}s，且已提供转写文本——\
                     截取后音频将与转写无法对应，请先手动修剪参考音频（保留转写对应的那段）\
                     或去掉 --transcript 改用自动截取",
                    duration_ms,
                    max_seconds
                ));
            }
            audio = crate::audio::ref_audio::best_window(&audio, max_ms);
            trimmed_from_ms = Some(duration_ms);
            ref_strategy = Some("best_window".to_string());
            duration_ms = audio.duration_ms() as u64;
            tracing::info!(
                "参考音频超过引擎上限，已按 best_window 截取至 {}ms（原 {}ms）",
                duration_ms,
                trimmed_from_ms.unwrap_or(0)
            );
        }
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
        transcript: transcript
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty()),
        trimmed_from_ms,
        ref_strategy,
    };
    std::fs::write(meta_path(name), serde_json::to_string_pretty(&meta)?)?;
    Ok(meta)
}

/// 列出全部克隆音色
///
/// 只认 `*.meta.json`（此前扫 `*.json` 会把目录里其他 JSON 也误读）。
pub fn list() -> Vec<CloneVoiceMeta> {
    std::fs::read_dir(base_dir())
        .map(|entries| {
            entries
                .flatten()
                .filter(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .ends_with(".meta.json")
                })
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

/// 按音色名查找元数据（供 CosyVoice 读取克隆音色的转写文本等）
pub fn find_meta(name: &str) -> Option<CloneVoiceMeta> {
    validate_name(name).ok()?;
    let content = std::fs::read_to_string(meta_path(name)).ok()?;
    serde_json::from_str(&content).ok()
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
    use votex_domain::shared::value_object::AudioData;

    /// 工作区重定向守卫（写库操作必须指向临时目录，避免污染真实 models/）
    ///
    /// 临时目录必须具备工作区结构（models/ + application.yml），
    /// 否则 `is_workspace_root` 判定失败会回落到真实工作区。
    fn guard(dir: &Path) -> crate::shared::workspace_paths::tests_support::EnvGuard {
        std::fs::create_dir_all(dir.join("models/voices/refs")).expect("创建临时工作区失败");
        std::fs::write(dir.join("application.yml"), "models:\n  storage_path: models\n")
            .expect("写临时配置失败");
        crate::shared::workspace_paths::tests_support::EnvGuard::set_workspace(dir)
    }

    fn make_wav(dir: &Path, name: &str, sample_rate: u32, ms: u64) -> PathBuf {
        let audio = AudioData::silence(sample_rate, ms as _);
        let p = dir.join(name);
        crate::audio::wav::WavWriter::write(&audio, &p).unwrap();
        p
    }

    /// 构造有声 WAV（正弦波，非静音）
    fn make_voiced_wav(dir: &Path, name: &str, sample_rate: u32, ms: u64) -> PathBuf {
        let n = (sample_rate as u64 * ms / 1000) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| ((i as f32 * 0.05).sin()) * 0.5)
            .collect();
        let audio = AudioData { samples, sample_rate, channels: 1 };
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

    /// 全静音参考音频无法克隆出可用音色，入库时应拒绝
    #[test]
    fn 入库_全静音参考拒绝() {
        let ws = tempfile::tempdir().unwrap();
        let _g = guard(ws.path());
        let src = tempfile::tempdir().unwrap();
        let wav = make_wav(src.path(), "mute.wav", 24000, 5000);
        let err = add("静音音色", &wav, false, None, None).unwrap_err();
        assert!(
            err.to_string().contains("静音"),
            "应提示静音问题，实际: {err}"
        );
    }

    /// 超过引擎上限 + 无转写 → best_window 自动截取并记录原时长
    #[test]
    fn 入库_超上限无转写自动截取() {
        let ws = tempfile::tempdir().unwrap();
        let _g = guard(ws.path());
        let src = tempfile::tempdir().unwrap();
        let wav = make_voiced_wav(src.path(), "long.wav", 24000, 40_000);

        let meta = add("截取音色", &wav, false, None, Some(15)).unwrap();
        assert_eq!(meta.duration_ms, 15_000, "应截取到 15s");
        assert_eq!(meta.trimmed_from_ms, Some(40_000), "应记录原始时长");
        assert_eq!(meta.ref_strategy.as_deref(), Some("best_window"));

        let stored = crate::audio::wav::read_wav(&reference_path("截取音色").unwrap()).unwrap();
        assert_eq!(stored.duration_ms() as u64, 15_000, "落盘音频应为截取后时长");
    }

    /// 超过引擎上限 + 有转写 → 显式拒绝（截取窗口与转写无法对应）
    #[test]
    fn 入库_超上限有转写拒绝() {
        let ws = tempfile::tempdir().unwrap();
        let _g = guard(ws.path());
        let src = tempfile::tempdir().unwrap();
        let wav = make_voiced_wav(src.path(), "long2.wav", 24000, 40_000);

        let err = add("拒绝音色", &wav, false, Some("这是转写文本"), Some(15)).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("上限"), "应提示超上限，实际: {msg}");
        assert!(msg.contains("转写"), "应提示转写冲突，实际: {msg}");
        assert!(find_meta("拒绝音色").is_none(), "拒绝时不应落盘");
    }

    /// 未超上限时不应截取，元数据保持干净
    #[test]
    fn 入库_未超上限不截取() {
        let ws = tempfile::tempdir().unwrap();
        let _g = guard(ws.path());
        let src = tempfile::tempdir().unwrap();
        let wav = make_voiced_wav(src.path(), "ok.wav", 24000, 10_000);

        let meta = add("正常音色", &wav, false, None, Some(15)).unwrap();
        assert_eq!(meta.duration_ms, 10_000);
        assert!(meta.trimmed_from_ms.is_none());
        assert!(meta.ref_strategy.is_none());
    }

    /// list 只认 *.meta.json，不把目录里的其他 JSON 误当音色元数据
    #[test]
    fn 列表_过滤非meta的json文件() {
        let ws = tempfile::tempdir().unwrap();
        let _g = guard(ws.path());
        let src = tempfile::tempdir().unwrap();
        let wav = make_voiced_wav(src.path(), "v.wav", 24000, 5000);
        let meta = add("列表音色", &wav, false, None, None).unwrap();
        assert_eq!(meta.name, "列表音色");

        // 目录里混入一个非 meta 的 JSON
        let stray = base_dir().join("stray.json");
        std::fs::write(&stray, r#"{"note":"不是音色元数据"}"#).unwrap();

        let voices = list();
        assert_eq!(voices.len(), 1, "不应把 stray.json 当音色: {voices:?}");
        assert_eq!(voices[0].name, "列表音色");
    }
}
