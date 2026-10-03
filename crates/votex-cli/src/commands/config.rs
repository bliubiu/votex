use anyhow::Result;
use std::path::Path;
use votex_infra::config::loader::ConfigLoader;
use votex_domain::config::value_object::AppConfig;

use crate::commands::root::ConfigAction;

/// 处理 config 子命令
pub fn handle(action: &ConfigAction, config_path: &Path) -> Result<()> {
    match action {
        ConfigAction::Show => show(config_path),
        ConfigAction::Set { key, value } => set(config_path, key, value),
        ConfigAction::Reset => reset(config_path),
    }
}

/// 显示当前配置
fn show(config_path: &Path) -> Result<()> {
    let config = ConfigLoader::from_file(config_path)?;
    let yaml = serde_yml::to_string(&config)
        .map_err(|e| anyhow::anyhow!("序列化配置失败: {}", e))?;
    println!("配置文件: {}", config_path.display());
    println!("---");
    print!("{}", yaml);
    Ok(())
}

/// 设置配置项
fn set(config_path: &Path, key: &str, value: &str) -> Result<()> {
    let mut config = ConfigLoader::from_file(config_path)?;
    apply_set(&mut config, key, value)?;
    ConfigLoader::save_to_file(&config, config_path)?;
    println!("已设置 {} = {}", key, value);
    println!("配置文件: {}", config_path.display());
    Ok(())
}

/// 重置为默认配置
fn reset(config_path: &Path) -> Result<()> {
    let config = AppConfig::default();
    ConfigLoader::save_to_file(&config, config_path)?;
    println!("已重置为默认配置");
    println!("配置文件: {}", config_path.display());
    Ok(())
}

/// 根据键路径设置配置项
fn apply_set(config: &mut AppConfig, key: &str, value: &str) -> Result<()> {
    let parts: Vec<&str> = key.split('.').collect();
    match parts.as_slice() {
        ["models", "storage_path"] => config.models.storage_path = value.to_string(),
        ["models", "mirror"] => {
            config.models.download.priority = match value.to_lowercase().as_str() {
                "default" => vec!["modelscope".into(), "hf-mirror".into(), "github".into(), "huggingface".into()],
                "cn" | "china" => vec!["modelscope".into(), "hf-mirror".into(), "gitee".into()],
                _ => return Err(anyhow::anyhow!("无效的 mirror 值: {} (支持 default/cn)", value)),
            };
        }
        ["models", "download", "priority"] => {
            config.models.download.priority = value.split(',').map(|s| s.trim().to_string()).collect();
        }
        ["models", "download", "resume"] => {
            config.models.download.resume = parse_bool(value)?;
        }
        ["models", "download", "timeout"] => {
            config.models.download.timeout = value.parse()
                .map_err(|e| anyhow::anyhow!("解析 timeout 失败: {}", e))?;
        }
        ["tts", "default_engine"] => config.tts.default_engine = value.to_string(),
        ["tts", "default_voice"] => config.tts.default_voice = value.to_string(),
        ["tts", "default_speed"] => {
            config.tts.default_speed = value.parse()
                .map_err(|e| anyhow::anyhow!("解析 speed 失败: {}", e))?;
        }
        ["tts", "default_pitch"] => {
            config.tts.default_pitch = value.parse()
                .map_err(|e| anyhow::anyhow!("解析 pitch 失败: {}", e))?;
        }
        ["tts", "default_volume"] => {
            config.tts.default_volume = value.parse()
                .map_err(|e| anyhow::anyhow!("解析 volume 失败: {}", e))?;
        }
        ["tts", "default_segment_size"] => config.tts.default_segment_size = value.to_string(),
        ["tts", "default_segment_silence_ms"] => {
            config.tts.default_segment_silence_ms = value.parse()
                .map_err(|e| anyhow::anyhow!("解析失败: {}", e))?;
        }
        ["tts", "default_crossfade_ms"] => {
            config.tts.default_crossfade_ms = value.parse()
                .map_err(|e| anyhow::anyhow!("解析失败: {}", e))?;
        }
        ["tts", "num_to_chinese"] => config.tts.num_to_chinese = parse_bool(value)?,
        ["tts", "denoise"] => config.tts.denoise = parse_bool(value)?,
        ["tts", "denoise_level"] => config.tts.denoise_level = value.to_string(),
        ["asr", "default_model"] => config.asr.default_model = value.to_string(),
        ["asr", "default_language"] => config.asr.default_language = value.to_string(),
        ["asr", "auto_punctuation"] => config.asr.auto_punctuation = parse_bool(value)?,
        ["asr", "auto_slice"] => config.asr.auto_slice = parse_bool(value)?,
        ["asr", "default_slice_length"] => config.asr.default_slice_length = value.to_string(),
        ["asr", "denoise"] => config.asr.denoise = parse_bool(value)?,
        ["asr", "denoise_level"] => config.asr.denoise_level = value.to_string(),
        ["asr", "default_output_format"] => config.asr.default_output_format = value.to_string(),
        ["output", "dir"] => config.output.dir = value.to_string(),
        ["output", "default_format"] => config.output.default_format = value.to_string(),
        ["output", "mp3_bitrate"] => config.output.mp3_bitrate = value.to_string(),
        ["output", "wav_sample_depth"] => config.output.wav_sample_depth = value.to_string(),
        ["log", "level"] => config.log.level = value.to_string(),
        ["log", "file_enabled"] => config.log.file_enabled = parse_bool(value)?,
        ["log", "dir"] => config.log.dir = value.to_string(),
        ["log", "max_days"] => {
            config.log.max_days = value.parse()
                .map_err(|e| anyhow::anyhow!("解析失败: {}", e))?;
        }
        ["ui", "theme"] => config.ui.theme = value.to_string(),
        ["ui", "language"] => config.ui.language = value.to_string(),
        ["ui", "font_size"] => {
            config.ui.font_size = value.parse()
                .map_err(|e| anyhow::anyhow!("解析失败: {}", e))?;
        }
        ["task", "max_concurrent_tts"] => {
            config.task.max_concurrent_tts = value.parse()
                .map_err(|e| anyhow::anyhow!("解析失败: {}", e))?;
        }
        ["task", "max_concurrent_asr"] => {
            config.task.max_concurrent_asr = value.parse()
                .map_err(|e| anyhow::anyhow!("解析失败: {}", e))?;
        }
        ["task", "retry_on_failure"] => config.task.retry_on_failure = parse_bool(value)?,
        ["task", "max_retries"] => {
            config.task.max_retries = value.parse()
                .map_err(|e| anyhow::anyhow!("解析失败: {}", e))?;
        }
        ["task", "persist_file"] => config.task.persist_file = value.to_string(),
        ["inference", "num_threads"] => {
            config.inference.num_threads = value.parse()
                .map_err(|e| anyhow::anyhow!("解析失败: {}", e))?;
        }
        ["pipeline", "progressive_quality"] => config.pipeline.progressive_quality = parse_bool(value)?,
        _ => return Err(anyhow::anyhow!("不支持的配置项: {}", key)),
    }
    Ok(())
}

fn parse_bool(value: &str) -> Result<bool> {
    match value.to_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err(anyhow::anyhow!("无效的 bool 值: {} (支持 true/false/1/0/yes/no)", value)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ==================== parse_bool ====================

    #[test]
    fn parse_bool_true_variants() {
        assert!(parse_bool("true").unwrap());
        assert!(parse_bool("TRUE").unwrap());
        assert!(parse_bool("1").unwrap());
        assert!(parse_bool("yes").unwrap());
        assert!(parse_bool("on").unwrap());
    }

    #[test]
    fn parse_bool_false_variants() {
        assert!(!parse_bool("false").unwrap());
        assert!(!parse_bool("FALSE").unwrap());
        assert!(!parse_bool("0").unwrap());
        assert!(!parse_bool("no").unwrap());
        assert!(!parse_bool("off").unwrap());
    }

    #[test]
    fn parse_bool_invalid() {
        assert!(parse_bool("invalid").is_err());
        assert!(parse_bool("").is_err());
        assert!(parse_bool("2").is_err());
    }

    // ==================== apply_set ====================

    fn default_config() -> AppConfig {
        AppConfig::default()
    }

    #[test]
    fn apply_set_models_storage_path() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "models.storage_path", "custom_models").unwrap();
        assert_eq!(cfg.models.storage_path, "custom_models");
    }

    #[test]
    fn apply_set_tts_default_engine() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "tts.default_engine", "qwen3").unwrap();
        assert_eq!(cfg.tts.default_engine, "qwen3");
    }

    #[test]
    fn apply_set_tts_default_speed() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "tts.default_speed", "1.5").unwrap();
        assert!((cfg.tts.default_speed - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn apply_set_tts_default_pitch() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "tts.default_pitch", "5").unwrap();
        assert_eq!(cfg.tts.default_pitch, 5);
    }

    #[test]
    fn apply_set_tts_default_volume() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "tts.default_volume", "90").unwrap();
        assert_eq!(cfg.tts.default_volume, 90);
    }

    #[test]
    fn apply_set_asr_default_model() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "asr.default_model", "sensevoice").unwrap();
        assert_eq!(cfg.asr.default_model, "sensevoice");
    }

    #[test]
    fn apply_set_asr_default_language() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "asr.default_language", "en").unwrap();
        assert_eq!(cfg.asr.default_language, "en");
    }

    #[test]
    fn apply_set_log_level() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "log.level", "debug").unwrap();
        assert_eq!(cfg.log.level, "debug");
    }

    #[test]
    fn apply_set_log_file_enabled() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "log.file_enabled", "false").unwrap();
        assert!(!cfg.log.file_enabled);
    }

    #[test]
    fn apply_set_ui_theme() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "ui.theme", "dark").unwrap();
        assert_eq!(cfg.ui.theme, "dark");
    }

    #[test]
    fn apply_set_ui_language() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "ui.language", "en").unwrap();
        assert_eq!(cfg.ui.language, "en");
    }

    #[test]
    fn apply_set_task_max_retries() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "task.max_retries", "5").unwrap();
        assert_eq!(cfg.task.max_retries, 5);
    }

    #[test]
    fn apply_set_inference_num_threads() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "inference.num_threads", "8").unwrap();
        assert_eq!(cfg.inference.num_threads, 8);
    }

    #[test]
    fn apply_set_output_dir() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "output.dir", "custom_output").unwrap();
        assert_eq!(cfg.output.dir, "custom_output");
    }

    #[test]
    fn apply_set_output_format() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "output.default_format", "mp3").unwrap();
        assert_eq!(cfg.output.default_format, "mp3");
    }

    #[test]
    fn apply_set_unknown_key() {
        let mut cfg = default_config();
        let result = apply_set(&mut cfg, "nonexistent.key", "value");
        assert!(result.is_err());
    }

    #[test]
    fn apply_set_models_mirror_cn() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "models.mirror", "cn").unwrap();
        assert_eq!(cfg.models.download.priority, vec!["modelscope", "hf-mirror", "gitee"]);
    }

    #[test]
    fn apply_set_models_mirror_default() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "models.mirror", "default").unwrap();
        assert_eq!(cfg.models.download.priority, vec!["modelscope", "hf-mirror", "github", "huggingface"]);
    }

    #[test]
    fn apply_set_models_mirror_invalid() {
        let mut cfg = default_config();
        let result = apply_set(&mut cfg, "models.mirror", "invalid");
        assert!(result.is_err());
    }

    #[test]
    fn apply_set_download_priority() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "models.download.priority", "github,hf-mirror").unwrap();
        assert_eq!(cfg.models.download.priority, vec!["github", "hf-mirror"]);
    }

    #[test]
    fn apply_set_download_resume() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "models.download.resume", "false").unwrap();
        assert!(!cfg.models.download.resume);
    }

    #[test]
    fn apply_set_download_timeout() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "models.download.timeout", "600").unwrap();
        assert_eq!(cfg.models.download.timeout, 600);
    }

    #[test]
    fn apply_set_segment_silence() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "tts.default_segment_silence_ms", "500").unwrap();
        assert_eq!(cfg.tts.default_segment_silence_ms, 500);
    }

    #[test]
    fn apply_set_num_to_chinese() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "tts.num_to_chinese", "false").unwrap();
        assert!(!cfg.tts.num_to_chinese);
    }

    #[test]
    fn apply_set_pipeline_progressive_quality() {
        let mut cfg = default_config();
        apply_set(&mut cfg, "pipeline.progressive_quality", "false").unwrap();
        assert!(!cfg.pipeline.progressive_quality);
    }

    // ==================== apply_set — 错误输入 ====================

    #[test]
    fn apply_set_speed_invalid() {
        let mut cfg = default_config();
        let result = apply_set(&mut cfg, "tts.default_speed", "not_a_number");
        assert!(result.is_err());
    }

    #[test]
    fn apply_set_timeout_invalid() {
        let mut cfg = default_config();
        let result = apply_set(&mut cfg, "models.download.timeout", "not_a_number");
        assert!(result.is_err());
    }

    #[test]
    fn apply_set_bool_invalid() {
        let mut cfg = default_config();
        let result = apply_set(&mut cfg, "log.file_enabled", "maybe");
        assert!(result.is_err());
    }
}
