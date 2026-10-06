use anyhow::Result;
use std::path::Path;
use votex_domain::config::value_object::AppConfig;

/// 配置加载器
///
/// 读取 `application.yml` 并反序列化为 `AppConfig`。
/// 文件不存在或解析失败时，返回包含详细错误信息的 `anyhow::Error`。
pub struct ConfigLoader;

impl ConfigLoader {
    /// 从 YAML 文件加载配置
    ///
    /// # 错误
    /// - 文件不存在
    /// - 读取失败
    /// - 格式不匹配 `AppConfig` 结构
    pub fn from_file(path: &Path) -> Result<AppConfig> {
        if !path.exists() {
            tracing::warn!("配置文件不存在: {:?}, 使用默认配置", path);
            return Ok(AppConfig::default());
        }

        let content = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("读取配置文件失败: {} - {}", path.display(), e))?;

        // 敏感字段解密：加密落盘的 ENC(...) 密钥必须在此还原为明文，
        // 否则保存路径加密、读取路径却拿密文当 key 用（闭环缺失）
        let content = crate::security::config_crypto::ConfigCrypto::decrypt_config_content(&content, None);

        let config: AppConfig = serde_yml::from_str(&content)
            .map_err(|e| anyhow::anyhow!("解析配置文件失败: {} - {}", path.display(), e))?;

        tracing::info!("配置文件已加载: {:?}", path);
        Ok(config)
    }

    /// 从字符串加载配置（用于测试）
    pub fn from_str(yaml: &str) -> Result<AppConfig> {
        let config: AppConfig = serde_yml::from_str(yaml)
            .map_err(|e| anyhow::anyhow!("解析 YAML 字符串失败: {}", e))?;
        Ok(config)
    }

    /// 可选地加载配置文件：文件不存在属正常，解析失败必须显式暴露
    ///
    /// 设计要点（修复"配置静默失效"缺陷）：
    /// - **文件不存在**：首次启动的正常场景，静默返回 `None`（由调用方用内置默认值）
    /// - **解析失败**：配置文件存在但内容有误，此时**绝不静默降级**。
    ///   因为日志系统尚未初始化，直接写 stderr 保证用户一定看到，
    ///   并返回 `None` 让程序继续用默认值运行（不阻断启动）
    ///
    /// 调用方禁止再写 `.ok()` 把本方法的告警吞掉——历史上正是因为
    /// `read_to_string().ok().and_then(|c| serde_yml::from_str(&c).ok())`
    /// 这种双重静默吞错，导致 `application.yml` 里一行 YAML 语法错误
    /// 就让整份配置（含 DirectML 崩溃保护开关）失效且毫无提示。
    pub fn load_optional(path: &Path) -> Option<AppConfig> {
        // 文件不存在是正常情况（首次启动），不打扰用户
        if !path.exists() {
            return None;
        }

        let content = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!(
                    "[配置错误] 读取配置文件失败: {} - {}",
                    path.display(),
                    e
                );
                eprintln!("[配置错误] 程序将使用内置默认配置运行。");
                return None;
            }
        };

        // 敏感字段解密（与 from_file 一致；解密失败保留密文原值）
        let content = crate::security::config_crypto::ConfigCrypto::decrypt_config_content(&content, None);

        match serde_yml::from_str::<AppConfig>(&content) {
            Ok(c) => Some(c),
            Err(e) => {
                // YAML 语法错误（如注释 `#` 前缺空格、缩进错误、类型不匹配）
                // 会导致整份配置反序列化失败，必须让用户知道
                eprintln!(
                    "[配置错误] 解析配置文件失败: {} - {}",
                    path.display(),
                    e
                );
                eprintln!(
                    "[配置错误] 常见原因：注释 `#` 前缺少空格、缩进不一致、字段类型不匹配。"
                );
                eprintln!("[配置错误] 程序将使用内置默认配置运行，请修正配置文件后重启。");
                None
            }
        }
    }

    /// 生成默认配置的 YAML 文件
    ///
    /// 将 `AppConfig::default()` 序列化为 YAML 写入指定路径。
    /// 如果文件已存在，不会覆盖。
    pub fn generate_default(path: &Path) -> Result<()> {
        if path.exists() {
            tracing::debug!("配置文件已存在，跳过生成: {:?}", path);
            return Ok(());
        }

        let config = AppConfig::default();
        let yaml = serde_yml::to_string(&config)
            .map_err(|e| anyhow::anyhow!("序列化默认配置失败: {}", e))?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, &yaml)?;

        tracing::info!("默认配置文件已生成: {:?}", path);
        Ok(())
    }

    /// 生成默认配置 YAML 字符串
        pub fn default_yaml() -> String {
        serde_yml::to_string(&AppConfig::default())
            .expect("序列化默认配置失败")
    }

    /// 保存配置到 YAML 文件
    ///
    /// 将 `AppConfig` 序列化为 YAML 写入指定路径（覆盖已有文件）。
    pub fn save_to_file(config: &AppConfig, path: &Path) -> Result<()> {
        let yaml = serde_yml::to_string(config)
            .map_err(|e| anyhow::anyhow!("序列化配置失败: {}", e))?;

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, &yaml)?;

        tracing::info!("配置文件已保存: {:?}", path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===============================================================
    // 共用测试配置常量（提取重复 YAML，消除三份拷贝）
    // ===============================================================

    /// 标准默认配置
    const DEFAULT_CONFIG_YAML: &str = r#"
models:
  storage_path: "models"
  download:
    priority:
      - modelscope
      - hf-mirror
      - github
      - huggingface
    resume: true
    timeout: 300
tts:
  default_engine: "kokoro"
  default_voice: "zf_001"
  default_speed: 1.0
  default_pitch: 0
  default_volume: 80
  default_segment_size: "500"
  default_segment_silence_ms: 300
  default_crossfade_ms: 50
  num_to_chinese: true
  denoise: false
  denoise_level: "low"
asr:
  default_model: "whisper-base"
  default_language: "zh"
  auto_punctuation: true
  auto_slice: true
  default_slice_length: "30"
  denoise: false
  denoise_level: "low"
  default_output_format: "srt"
output:
  dir: "output"
  default_format: "wav"
  mp3_bitrate: "192"
  wav_sample_depth: "16"
log:
  level: "info"
  file_enabled: true
  dir: "logs"
  max_days: 32
ui:
  theme: "light"
  language: "zh"
  font_size: 14
task:
  max_concurrent_tts: 1
  max_concurrent_asr: 1
  retry_on_failure: true
  max_retries: 1
  persist_file: "tasks.json"
inference:
  num_threads: 0
pipeline:
  progressive_quality: true
"#;

    /// 自定义下载优先级配置（覆盖 priority 和 timeout）
    const CUSTOM_PRIORITY_CONFIG_YAML: &str = r#"
models:
  storage_path: "models"
  download:
    priority:
      - hf-mirror
      - huggingface
    resume: true
    timeout: 120
tts:
  default_engine: "kokoro"
  default_voice: "zf_001"
  default_speed: 1.0
  default_pitch: 0
  default_volume: 80
  default_segment_size: "500"
  default_segment_silence_ms: 300
  default_crossfade_ms: 50
  num_to_chinese: true
  denoise: false
  denoise_level: "low"
asr:
  default_model: "whisper-base"
  default_language: "zh"
  auto_punctuation: true
  auto_slice: true
  default_slice_length: "30"
  denoise: false
  denoise_level: "low"
  default_output_format: "srt"
output:
  dir: "output"
  default_format: "wav"
  mp3_bitrate: "192"
  wav_sample_depth: "16"
log:
  level: "info"
  file_enabled: true
  dir: "logs"
  max_days: 32
ui:
  theme: "light"
  language: "zh"
  font_size: 14
task:
  max_concurrent_tts: 1
  max_concurrent_asr: 1
  retry_on_failure: true
  max_retries: 1
  persist_file: "tasks.json"
inference:
  num_threads: 0
pipeline:
  progressive_quality: true
"#;

    /// 自定义字段覆盖配置（验证缺失字段回退默认值）
    const CUSTOM_FIELDS_CONFIG_YAML: &str = r#"
models:
  storage_path: "custom_models"
  download:
    priority:
      - modelscope
      - hf-mirror
      - github
      - huggingface
    resume: true
    timeout: 300
tts:
  default_engine: "kokoro"
  default_voice: "zf_001"
  default_speed: 1.0
  default_pitch: 0
  default_volume: 80
  default_segment_size: "500"
  default_segment_silence_ms: 300
  default_crossfade_ms: 50
  num_to_chinese: true
  denoise: false
  denoise_level: "low"
asr:
  default_model: "whisper-small"
  default_language: "zh"
  auto_punctuation: true
  auto_slice: true
  default_slice_length: "30"
  denoise: false
  denoise_level: "low"
  default_output_format: "srt"
output:
  dir: "output"
  default_format: "wav"
  mp3_bitrate: "192"
  wav_sample_depth: "16"
log:
  level: "debug"
  file_enabled: true
  dir: "logs"
  max_days: 32
ui:
  theme: "dark"
  language: "en"
  font_size: 14
task:
  max_concurrent_tts: 2
  max_concurrent_asr: 2
  retry_on_failure: true
  max_retries: 3
  persist_file: "tasks.json"
inference:
  num_threads: 4
pipeline:
  progressive_quality: true
"#;

    #[test]
    fn config_从字符串加载() {
        let config = ConfigLoader::from_str(DEFAULT_CONFIG_YAML).unwrap();
        assert_eq!(config.models.storage_path, "models");
        assert_eq!(config.tts.default_engine, "kokoro");
        assert_eq!(config.asr.default_model, "whisper-base");
        assert_eq!(config.log.level, "info");
        assert_eq!(config.ui.theme, "light");
        assert_eq!(config.task.max_concurrent_tts, 1);
        assert!(config.pipeline.progressive_quality);
    }

    #[test]
    fn config_下载优先级解析() {
        let config = ConfigLoader::from_str(CUSTOM_PRIORITY_CONFIG_YAML).unwrap();
        assert_eq!(config.models.storage_path, "models");
        assert_eq!(config.models.download.priority, vec!["hf-mirror", "huggingface"]);
        assert!(config.models.download.resume);
        assert_eq!(config.models.download.timeout, 120);
    }

    #[test]
    fn config_缺失字段回退默认值() {
        let config = ConfigLoader::from_str(CUSTOM_FIELDS_CONFIG_YAML).unwrap();
        assert_eq!(config.models.storage_path, "custom_models");
        assert_eq!(config.asr.default_model, "whisper-small");
        assert_eq!(config.log.level, "debug");
        assert_eq!(config.ui.theme, "dark");
        assert_eq!(config.task.max_retries, 3);
        assert_eq!(config.inference.num_threads, 4);
    }

    #[test]
    fn config_无效YAML应报错() {
        let result = ConfigLoader::from_str("这不是有效的 YAML: [}");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("解析"));
    }

    #[test]
    fn config_默认配置序列化() {
        let yaml = ConfigLoader::default_yaml();
        assert!(!yaml.is_empty());
        assert!(yaml.contains("kokoro"));
        assert!(yaml.contains("whisper-base"));
        assert!(yaml.contains("info"));
    }

    #[test]
    fn config_保存到文件并重新加载() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("application.yml");

        // 保存
        let config = AppConfig {
            models: votex_domain::config::value_object::ModelsConfig {
                storage_path: "test_models".to_string(),
                ..Default::default()
            },
            log: votex_domain::config::value_object::LogConfig {
                level: "debug".to_string(),
                ..Default::default()
            },
            ..Default::default()
        };
        ConfigLoader::save_to_file(&config, &path).unwrap();
        assert!(path.exists());

        // 重新加载
        let loaded = ConfigLoader::from_file(&path).unwrap();
        assert_eq!(loaded.models.storage_path, "test_models");
        assert_eq!(loaded.log.level, "debug");
        assert_eq!(loaded.tts.default_engine, "kokoro");
    }
}
