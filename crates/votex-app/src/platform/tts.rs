//! TTS 门面：音色库与设备条件探测

use std::path::{Path, PathBuf};

pub use votex_infra::tts::qwen3_model_selector;
pub use votex_infra::tts::qwen3_model_selector::{DeviceConditions, ModelRecommendation};

/// 音色库根目录
pub fn voice_library_dir() -> PathBuf {
    votex_infra::tts::voice_library::base_dir()
}

/// 添加克隆音色
pub fn add_voice(name: &str, reference_wav: &Path, denoise: bool) -> anyhow::Result<votex_infra::tts::voice_library::CloneVoiceMeta> {
    Ok(votex_infra::tts::voice_library::add(
        name,
        reference_wav,
        denoise,
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

/// 依据设备条件推荐模型规格
pub fn recommend_model(conditions: &DeviceConditions) -> ModelRecommendation {
    votex_infra::tts::qwen3_model_selector::recommend(conditions)
}
