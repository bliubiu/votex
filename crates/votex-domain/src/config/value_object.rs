use serde::{Deserialize, Serialize};

/// 应用配置（聚合根）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub models: ModelsConfig,
    pub tts: TtsConfig,
    pub asr: AsrConfig,
    /// 翻译配置（`translation` 段缺省时使用默认值）
    #[serde(default)]
    pub translation: TranslationConfig,
    pub output: OutputConfig,
    pub log: LogConfig,
    pub ui: UiConfig,
    pub task: TaskConfig,
    pub inference: InferenceConfig,
    pub pipeline: PipelineConfig,
    #[serde(default)]
    pub resource: ResourceConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            models: ModelsConfig::default(),
            tts: TtsConfig::default(),
            asr: AsrConfig::default(),
            translation: TranslationConfig::default(),
            output: OutputConfig::default(),
            log: LogConfig::default(),
            ui: UiConfig::default(),
            task: TaskConfig::default(),
            inference: InferenceConfig::default(),
            pipeline: PipelineConfig::default(),
            resource: ResourceConfig::default(),
        }
    }
}

/// 资源治理配置（防止推理任务占满系统资源）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    /// 最大推理并发许可数（Green 状态；Yellow 自动收缩为 1，Red 暂停派发）
    #[serde(default = "default_max_inference_concurrency")]
    pub max_inference_concurrency: usize,
    /// 内存已用百分比达到该值 → Yellow（收缩并发）
    #[serde(default = "default_memory_yellow_pct")]
    pub memory_yellow_used_pct: f32,
    /// 内存已用百分比达到该值 → Red（暂停派发新任务）
    #[serde(default = "default_memory_red_pct")]
    pub memory_red_used_pct: f32,
    /// 是否强制执行资源限制（false 仅供调试）
    #[serde(default = "default_true2")]
    pub enforce_limits: bool,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        Self {
            max_inference_concurrency: default_max_inference_concurrency(),
            memory_yellow_used_pct: default_memory_yellow_pct(),
            memory_red_used_pct: default_memory_red_pct(),
            enforce_limits: true,
        }
    }
}

fn default_max_inference_concurrency() -> usize {
    2
}
fn default_memory_yellow_pct() -> f32 {
    70.0
}
fn default_memory_red_pct() -> f32 {
    85.0
}
fn default_true2() -> bool {
    true
}

/// 翻译配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationConfig {
    /// 默认翻译引擎
    #[serde(default = "default_translation_engine")]
    pub default_engine: String,
    /// 默认翻译方向
    #[serde(default = "default_translation_direction")]
    pub default_direction: String,
    /// 是否启用翻译结果缓存
    #[serde(default = "default_true")]
    pub enable_cache: bool,
    /// 缓存容量（条目数）
    #[serde(default = "default_cache_capacity")]
    pub cache_capacity: usize,
    /// 长文本分段的单段最大字符数（0 表示按引擎能力自动决定）
    #[serde(default)]
    pub max_segment_chars: usize,
    /// 术语表文件路径（JSON 数组，可选）
    #[serde(default)]
    pub glossary_path: Option<String>,
    /// 目标中文书写系统：auto / simplified / traditional / none
    #[serde(default = "default_target_script")]
    pub target_script: String,
    /// LLM 引擎携带的历史上下文轮数
    #[serde(default = "default_context_turns")]
    pub context_turns: usize,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            default_engine: default_translation_engine(),
            default_direction: default_translation_direction(),
            enable_cache: true,
            cache_capacity: default_cache_capacity(),
            max_segment_chars: 0,
            glossary_path: None,
            target_script: default_target_script(),
            context_turns: default_context_turns(),
        }
    }
}

fn default_translation_engine() -> String {
    "dict".to_string()
}

fn default_translation_direction() -> String {
    "zh-en".to_string()
}

fn default_true() -> bool {
    true
}

fn default_cache_capacity() -> usize {
    2000
}

fn default_target_script() -> String {
    "none".to_string()
}

fn default_context_turns() -> usize {
    3
}

/// 模型配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsConfig {
    /// 模型文件存储根目录
    #[serde(default = "default_storage_path")]
    pub storage_path: String,
    /// 模型清单目录（默认为 `{storage_path}/registry`）
    #[serde(default)]
    pub registry_path: Option<String>,
    /// 下载配置
    #[serde(default)]
    pub download: DownloadConfig,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            storage_path: default_storage_path(),
            registry_path: None,
            download: DownloadConfig::default(),
        }
    }
}

fn default_storage_path() -> String {
    "models".to_string()
}

/// 下载配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadConfig {
    /// 镜像源优先级（按顺序尝试），如 `["modelscope", "hf-mirror", "github", "huggingface"]`
    #[serde(default = "default_priority")]
    pub priority: Vec<String>,
    /// 是否启用断点续传
    #[serde(default = "default_resume")]
    pub resume: bool,
    /// 单次下载超时（秒）
    #[serde(default = "default_timeout")]
    pub timeout: u64,
}

fn default_priority() -> Vec<String> {
    vec![
        "modelscope".into(),
        "hf-mirror".into(),
        "github".into(),
        "huggingface".into(),
    ]
}

fn default_resume() -> bool {
    true
}

fn default_timeout() -> u64 {
    300
}

impl Default for DownloadConfig {
    fn default() -> Self {
        Self {
            priority: default_priority(),
            resume: default_resume(),
            timeout: default_timeout(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsConfig {
    pub default_engine: String,
    pub default_voice: String,
    pub default_speed: f32,
    pub default_pitch: i32,
    pub default_volume: u8,
    pub default_segment_size: String,
    pub default_segment_silence_ms: u32,
    pub default_crossfade_ms: u32,
    pub num_to_chinese: bool,
    pub denoise: bool,
    pub denoise_level: String,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            default_engine: "kokoro".to_string(),
            default_voice: "zf_001".to_string(),
            default_speed: 1.0,
            default_pitch: 0,
            default_volume: 80,
            default_segment_size: "500".to_string(),
            default_segment_silence_ms: 300,
            default_crossfade_ms: 50,
            num_to_chinese: true,
            denoise: false,
            denoise_level: "low".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrConfig {
    pub default_model: String,
    pub default_language: String,
    pub auto_punctuation: bool,
    pub auto_slice: bool,
    pub default_slice_length: String,
    pub denoise: bool,
    pub denoise_level: String,
    pub default_output_format: String,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            default_model: "whisper-base".to_string(),
            default_language: "zh".to_string(),
            auto_punctuation: true,
            auto_slice: true,
            default_slice_length: "30".to_string(),
            denoise: false,
            denoise_level: "low".to_string(),
            default_output_format: "srt".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub dir: String,
    pub default_format: String,
    pub mp3_bitrate: String,
    pub wav_sample_depth: String,
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            dir: "output".to_string(),
            default_format: "wav".to_string(),
            mp3_bitrate: "192".to_string(),
            wav_sample_depth: "16".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogConfig {
    pub level: String,
    pub file_enabled: bool,
    pub dir: String,
    pub max_days: u32,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            file_enabled: true,
            dir: "logs".to_string(),
            max_days: 32,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    pub theme: String,
    pub language: String,
    pub font_size: u32,
}

impl Default for UiConfig {
    fn default() -> Self {
        Self {
            theme: "light".to_string(),
            language: "zh".to_string(),
            font_size: 14,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    pub max_concurrent_tts: u32,
    pub max_concurrent_asr: u32,
    pub retry_on_failure: bool,
    pub max_retries: u32,
    pub persist_file: String,
}

impl Default for TaskConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tts: 1,
            max_concurrent_asr: 1,
            retry_on_failure: true,
            max_retries: 1,
            persist_file: "tasks.json".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// intra-op 线程数（0 = 自动使用全部可用核心）
    pub num_threads: u32,
    /// inter-op 线程数（0 = 自动），仅 execution_mode=parallel 时生效
    #[serde(default = "default_inter_threads")]
    pub inter_threads: u32,
    /// 量化策略：none（不量化）、int8（全量化）、partial（部分量化，关键层保留 fp32）
    #[serde(default = "default_quantization")]
    pub quantization: String,
    /// 是否启用 KV cache 优化（适用于 NLLB 等生成式模型）
    #[serde(default = "default_kv_cache")]
    pub kv_cache: bool,
    /// 执行提供器：cpu（仅 CPU）、directml（DirectML GPU）、auto（自动选择）
    #[serde(default = "default_execution_provider")]
    pub execution_provider: String,
    /// 内存限制（MB），0 = 不限制（ORT 默认行为）
    #[serde(default = "default_memory_limit")]
    pub memory_limit_mb: u32,
    /// 是否启用内存模式优化（关闭可减少动态输入时的内存峰值）
    #[serde(default = "default_memory_pattern")]
    pub enable_memory_pattern: bool,
    /// 执行模式：sequential（顺序执行，省内存）/ parallel（并行执行，高性能）
    #[serde(default = "default_execution_mode")]
    pub execution_mode: String,
}

fn default_quantization() -> String {
    "none".to_string()
}

fn default_kv_cache() -> bool {
    true
}

fn default_inter_threads() -> u32 {
    0
}

fn default_memory_limit() -> u32 {
    0
}

fn default_memory_pattern() -> bool {
    true
}

fn default_execution_mode() -> String {
    "sequential".to_string()
}

fn default_execution_provider() -> String {
    "auto".to_string()
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            num_threads: 0,
            inter_threads: default_inter_threads(),
            quantization: default_quantization(),
            kv_cache: default_kv_cache(),
            execution_provider: default_execution_provider(),
            memory_limit_mb: default_memory_limit(),
            enable_memory_pattern: default_memory_pattern(),
            execution_mode: default_execution_mode(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub progressive_quality: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            progressive_quality: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_config_默认值生成() {
        let config = AppConfig::default();
        assert_eq!(config.models.storage_path, "models");
        assert_eq!(config.tts.default_engine, "kokoro");
        assert_eq!(config.asr.default_model, "whisper-base");
        assert_eq!(config.log.level, "info");
        assert_eq!(config.task.max_concurrent_tts, 1);
        assert!(config.pipeline.progressive_quality);
    }
}
