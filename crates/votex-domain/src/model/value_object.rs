use serde::{Deserialize, Serialize};
use std::fmt;

/// 模型唯一标识
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelId(String);

impl ModelId {
    pub fn new(key: &str) -> Self {
        Self(key.to_string())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// 模型类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelKind {
    Tts,
    Asr,
    Ocr,
    Translation,
    /// 推理运行时依赖（如 ONNX Runtime 动态库）
    ///
    /// 这类条目由 registry 统一管理下载与校验，
    /// 但**不是可加载的模型**——没有会话、没有 `load()`。
    /// 单独建变体而不是塞进上面四类，是为了让
    /// 「列出 TTS 模型」这类操作天然排除它们。
    Runtime,
}

impl ModelKind {
    /// 是否为可加载的模型（排除运行时依赖）
    pub fn is_loadable_model(&self) -> bool {
        !matches!(self, ModelKind::Runtime)
    }

    /// 中文显示名
    pub fn display_name(&self) -> &'static str {
        match self {
            ModelKind::Tts => "TTS",
            ModelKind::Asr => "ASR",
            ModelKind::Ocr => "OCR",
            ModelKind::Translation => "Translation",
            ModelKind::Runtime => "Runtime",
        }
    }
}

/// 模型存储格式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelFormat {
    /// ONNX 格式，可直接用于 ort 推理
    Onnx,
}

/// 引擎类型
///
/// 涵盖项目中所有 TTS/ASR/OCR/LLM/翻译/视频素材引擎。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EngineKind {
    // ===== TTS 引擎 =====
    /// Kokoro-82M (ONNX)
    Kokoro,
    /// IndexTTS-2.5 (ONNX，yunfengwang fp32 分图布局，粤语原生支持；已取代 IndexTTS2)
    IndexTTS25,
    /// CosyVoice 3 (ONNX)
    CosyVoice3,
    /// Qwen3-TTS
    Qwen3Tts,
    /// Azure Speech TTS（在线 API）
    AzureTts,
    /// 阿里云 TTS（在线 API）
    AliyunTts,

    // ===== ASR 引擎 =====
    /// Whisper (ONNX)
    Whisper,
    /// SenseVoice (ONNX)
    SenseVoice,
    /// Paraformer (ONNX)
    Paraformer,
    /// Qwen3-ASR
    Qwen3Asr,
    /// Azure Speech ASR（在线 API）
    AzureAsr,
    /// 阿里云 ASR（在线 API）
    AliyunAsr,
    /// FireRedASR (ONNX)
    FireRedAsr,
    /// WeNet Conformer (ONNX)
    WeNet,

    // ===== OCR 引擎 =====
    /// PaddleOCR (PP-OCRv6)
    PaddleOCR,
    /// EasyOCR
    EasyOcr,

    // ===== LLM 引擎 =====
    /// DeepSeek（在线 API）
    DeepSeek,
    /// OpenAI 兼容 API
    OpenAi,
    /// 通义千问 LLM（在线 API）
    QwenLlm,
    /// Google Gemini（在线 API）
    Gemini,
    /// Ollama 本地模型
    Ollama,
    /// Azure OpenAI（在线 API）
    AzureLlm,

    // ===== 翻译引擎 =====
    /// OPUS-MT (ONNX)
    OpusMt,
    /// Qwen-MT（在线 API）
    QwenMt,
    /// NLLB-200 (ONNX, Meta)
    Nllb,
    /// M2M-100 (ONNX, Meta)
    M2m100,
    /// HY-MT1.5 (Candle/ONNX, Tencent)
    HyMt1_5,
    /// CTranslate2 优化引擎
    CTranslate2,

    // ===== 推理运行时 =====
    /// ONNX Runtime 动态库本身（`load-dynamic` 运行时依赖，非模型）
    OnnxRuntime,

    // ===== 视频素材 =====
    /// Pexels
    Pexels,
    /// Pixabay
    Pixabay,
    /// Coverr
    Coverr,
}

impl EngineKind {
    /// 引擎所属模型类型
    pub fn model_kind(&self) -> ModelKind {
        match self {
            EngineKind::Kokoro | EngineKind::IndexTTS25
                | EngineKind::CosyVoice3 | EngineKind::Qwen3Tts
                | EngineKind::AzureTts | EngineKind::AliyunTts => ModelKind::Tts,

            EngineKind::Whisper | EngineKind::SenseVoice
                | EngineKind::Paraformer | EngineKind::Qwen3Asr
                | EngineKind::AzureAsr | EngineKind::AliyunAsr
                | EngineKind::FireRedAsr | EngineKind::WeNet => ModelKind::Asr,

            EngineKind::PaddleOCR | EngineKind::EasyOcr => ModelKind::Ocr,

            // 推理运行时：不是可 load() 的模型，但需要下载与完整性校验
            EngineKind::OnnxRuntime => ModelKind::Runtime,

            // LLM/翻译/素材不需要模型文件
            EngineKind::DeepSeek | EngineKind::OpenAi
                | EngineKind::QwenLlm | EngineKind::Gemini
                | EngineKind::Ollama | EngineKind::AzureLlm
                | EngineKind::OpusMt | EngineKind::QwenMt | EngineKind::Nllb
                | EngineKind::M2m100 | EngineKind::HyMt1_5 | EngineKind::CTranslate2
                | EngineKind::Pexels | EngineKind::Pixabay
                | EngineKind::Coverr => ModelKind::Tts, // placeholder, 不用于模型管理
        }
    }

    /// 引擎使用的推理格式（统一使用 ONNX）
    pub fn model_format(&self) -> ModelFormat {
        ModelFormat::Onnx
    }

    /// 引擎名称（用于配置和 CLI）
    pub fn as_str(&self) -> &'static str {
        match self {
            EngineKind::Kokoro => "kokoro",
            EngineKind::IndexTTS25 => "indextts25",
            EngineKind::CosyVoice3 => "cosyvoice3",
            EngineKind::Qwen3Tts => "qwen3-tts",
            EngineKind::AzureTts => "azure-tts",
            EngineKind::AliyunTts => "aliyun-tts",
            EngineKind::Whisper => "whisper",
            EngineKind::SenseVoice => "sensevoice",
            EngineKind::Paraformer => "paraformer",
            EngineKind::Qwen3Asr => "qwen3-asr",
            EngineKind::AzureAsr => "azure-asr",
            EngineKind::AliyunAsr => "aliyun-asr",
            EngineKind::FireRedAsr => "firered-asr",
            EngineKind::WeNet => "wenet",
            EngineKind::PaddleOCR => "paddleocr",
            EngineKind::EasyOcr => "easyocr",
            EngineKind::DeepSeek => "deepseek",
            EngineKind::OpenAi => "openai",
            EngineKind::QwenLlm => "qwen-llm",
            EngineKind::Gemini => "gemini",
            EngineKind::Ollama => "ollama",
            EngineKind::AzureLlm => "azure-llm",
            EngineKind::OpusMt => "opus-mt",
            EngineKind::QwenMt => "qwen-mt",
            EngineKind::Nllb => "nllb",
            EngineKind::M2m100 => "m2m-100",
            EngineKind::HyMt1_5 => "hy-mt-1.5",
            EngineKind::CTranslate2 => "ctranslate2",
            EngineKind::Pexels => "pexels",
            EngineKind::Pixabay => "pixabay",
            EngineKind::Coverr => "coverr",
            EngineKind::OnnxRuntime => "onnxruntime",
        }
    }

    /// 从字符串解析引擎类型
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "kokoro" => Some(EngineKind::Kokoro),
            // 迁移别名：IndexTTS2 已移除，历史配置/持久化中的 indextts2 归一到 IndexTTS25
            "indextts2" | "indextts" => Some(EngineKind::IndexTTS25),
            "indextts25" | "indextts-2.5" | "indextts2.5" => Some(EngineKind::IndexTTS25),
            "cosyvoice3" | "cosyvoice" => Some(EngineKind::CosyVoice3),
            "qwen3-tts" | "qwen3tts" => Some(EngineKind::Qwen3Tts),
            "azure-tts" | "azuretts" => Some(EngineKind::AzureTts),
            "aliyun-tts" | "aliyuntts" => Some(EngineKind::AliyunTts),
            "whisper" => Some(EngineKind::Whisper),
            "sensevoice" => Some(EngineKind::SenseVoice),
            "paraformer" => Some(EngineKind::Paraformer),
            "qwen3-asr" | "qwen3asr" => Some(EngineKind::Qwen3Asr),
            "azure-asr" | "azureasr" => Some(EngineKind::AzureAsr),
            "aliyun-asr" | "aliyunasr" => Some(EngineKind::AliyunAsr),
            "firered-asr" | "fireredasr" => Some(EngineKind::FireRedAsr),
            "wenet" => Some(EngineKind::WeNet),
            "paddleocr" | "paddle" => Some(EngineKind::PaddleOCR),
            "easyocr" => Some(EngineKind::EasyOcr),
            "deepseek" => Some(EngineKind::DeepSeek),
            "openai" => Some(EngineKind::OpenAi),
            "qwen-llm" | "qwenllm" | "qwen" => Some(EngineKind::QwenLlm),
            "gemini" => Some(EngineKind::Gemini),
            "ollama" => Some(EngineKind::Ollama),
            "azure-llm" | "azurellm" => Some(EngineKind::AzureLlm),
            "opus-mt" | "opusmt" => Some(EngineKind::OpusMt),
            "qwen-mt" | "qwenmt" => Some(EngineKind::QwenMt),
            "nllb" | "nllb-200" => Some(EngineKind::Nllb),
            "m2m-100" | "m2m100" => Some(EngineKind::M2m100),
            "hy-mt-1.5" | "hymt1.5" | "hy-mt" => Some(EngineKind::HyMt1_5),
            "ctranslate2" | "ct2" => Some(EngineKind::CTranslate2),
            "pexels" => Some(EngineKind::Pexels),
            "pixabay" => Some(EngineKind::Pixabay),
            "coverr" => Some(EngineKind::Coverr),
            _ => None,
        }
    }
}

/// 下载进度
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DownloadProgress {
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub source_name: String,
}

impl DownloadProgress {
    pub fn percentage(&self) -> f64 {
        if self.total_bytes == 0 {
            return 0.0;
        }
        (self.downloaded_bytes as f64 / self.total_bytes as f64) * 100.0
    }
}

/// 模型状态
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelStatus {
    NotDownloaded,
    Downloading(DownloadProgress),
    DownloadPaused,
    Verifying,
    VerifyFailed,
    Ready,
    Loading,
    Loaded,
    LoadFailed(String),
}

/// 下载源信息
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadSource {
    pub mirror: String,
    pub urls: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_id_new_and_as_str() {
        let id = ModelId::new("test-model");
        assert_eq!(id.as_str(), "test-model");
    }

    #[test]
    fn model_id_display() {
        let id = ModelId::new("kokoro-82m");
        assert_eq!(format!("{}", id), "kokoro-82m");
    }

    #[test]
    fn engine_kind_模型类型映射() {
        assert_eq!(EngineKind::Kokoro.model_kind(), ModelKind::Tts);
        assert_eq!(EngineKind::IndexTTS25.model_kind(), ModelKind::Tts);
        assert_eq!(EngineKind::Whisper.model_kind(), ModelKind::Asr);
        assert_eq!(EngineKind::SenseVoice.model_kind(), ModelKind::Asr);
    }

    #[test]
    fn download_progress_百分比计算() {
        let progress = DownloadProgress {
            downloaded_bytes: 50,
            total_bytes: 100,
            source_name: "hf-mirror".to_string(),
        };
        assert!((progress.percentage() - 50.0).abs() < 0.001);
    }

    #[test]
    fn download_progress_零总字节() {
        let progress = DownloadProgress {
            downloaded_bytes: 0,
            total_bytes: 0,
            source_name: "test".to_string(),
        };
        assert_eq!(progress.percentage(), 0.0);
    }

    #[test]
    fn model_kind_Runtime不是可加载模型() {
        // ONNX Runtime 动态库由 registry 管理下载与校验，但不是可 load() 的模型。
        // 这条契约保证「列出 TTS 模型」之类的操作天然排除运行时依赖。
        assert!(!ModelKind::Runtime.is_loadable_model());
        assert!(ModelKind::Tts.is_loadable_model());
        assert!(ModelKind::Asr.is_loadable_model());
        assert!(ModelKind::Ocr.is_loadable_model());
        assert!(ModelKind::Translation.is_loadable_model());
    }

    #[test]
    fn engine_kind_映射到Runtime类型() {
        assert_eq!(EngineKind::OnnxRuntime.model_kind(), ModelKind::Runtime);
        assert_eq!(EngineKind::OnnxRuntime.as_str(), "onnxruntime");
    }

    #[test]
    fn model_kind_中文显示名() {
        assert_eq!(ModelKind::Runtime.display_name(), "Runtime");
        assert_eq!(ModelKind::Tts.display_name(), "TTS");
        assert_eq!(ModelKind::Translation.display_name(), "Translation");
    }
}
