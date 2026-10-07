//! TTS 门面：音色库与设备条件探测

use std::path::{Path, PathBuf};

pub use votex_infra::tts::qwen3_model_selector;
pub use votex_infra::tts::qwen3_model_selector::{DeviceConditions, ModelRecommendation};
pub use votex_infra::tts::voice_library::CloneVoiceMeta;

/// 音色库根目录
pub fn voice_library_dir() -> PathBuf {
    votex_infra::tts::voice_library::base_dir()
}

/// 添加克隆音色
///
/// - `transcript`：参考音频文字转写（CosyVoice 零样本克隆必需，IndexTTS-2.5 可省略）
/// - `max_ref_seconds`：引擎参考音频上限（秒）。超过时无转写自动按 best_window
///   截取，有转写拒绝入库；传 `None` 不截取
pub fn add_voice(
    name: &str,
    reference_wav: &Path,
    denoise: bool,
    transcript: Option<&str>,
    max_ref_seconds: Option<u64>,
) -> anyhow::Result<votex_infra::tts::voice_library::CloneVoiceMeta> {
    Ok(votex_infra::tts::voice_library::add(
        name,
        reference_wav,
        denoise,
        transcript,
        max_ref_seconds,
    )?)
}

/// 列出全部克隆音色
pub fn list_voices() -> Vec<votex_infra::tts::voice_library::CloneVoiceMeta> {
    votex_infra::tts::voice_library::list()
}

/// 删除克隆音色（`confirm=false` 时仅查询不删除）
pub fn remove_voice(name: &str, confirm: bool) -> anyhow::Result<bool> {
    Ok(votex_infra::tts::voice_library::remove(name, confirm)?)
}

/// 查询克隆音色参考音频路径
pub fn voice_reference_path(name: &str) -> anyhow::Result<PathBuf> {
    Ok(votex_infra::tts::voice_library::reference_path(name)?)
}

/// 探测设备条件（是否支持 GPU / 显存大小）
pub fn detect_device(models_base: &Path) -> DeviceConditions {
    votex_infra::tts::qwen3_model_selector::detect_device(models_base)
}

/// 枚举引擎内置音色池（**不加载模型**，仅扫描音色文件名）
///
/// `models_dir` 传模型根目录（如 `models/`）。用于多角色配音的
/// 「有哪些音色可选」查询——角色表生成是纯文本操作，不该被迫加载模型。
pub fn list_engine_voices(models_dir: &Path) -> Vec<votex_domain::tts::value_object::VoiceMeta> {
    votex_infra::tts::kokoro::list_voice_pool(&models_dir.join("tts"))
}

/// 依据设备条件推荐模型规格
pub fn recommend_model(conditions: &DeviceConditions) -> ModelRecommendation {
    votex_infra::tts::qwen3_model_selector::recommend(conditions)
}
