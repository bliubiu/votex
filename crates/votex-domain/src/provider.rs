//! Provider 注册表接口
//!
//! 定义统一的 Provider 注册、发现和工厂模式接口。
//! 借鉴 MoneyPrinterTurbo 的配置驱动 Provider 架构设计。
//!
//! # 设计思路
//!
//! 各领域（TTS/ASR/OCR/LLM/翻译）都有自己的 Provider trait，
//! `ProviderRegistry` 提供一个统一的注册中心，支持：
//!
//! - 按类型和引擎名注册 Provider
//! - 按类型和引擎名获取/创建 Provider
//! - 列出所有已注册的 Provider

use crate::error::DomainError;
use crate::model::value_object::EngineKind;

// ============================================================
// 注册表接口
// ============================================================

/// Provider 能力标记
///
/// 描述一个 Provider 属于哪个领域。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProviderCapability {
    Tts,
    Asr,
    Ocr,
    Llm,
    Translation,
    VideoMaterial,
    /// 推理运行时（ONNX Runtime 动态库本身）
    ///
    /// 不是业务能力，而是所有 ONNX 能力的**前置依赖**。
    /// 单列变体而非塞进某一类，避免 GUI 按能力分组时把 ORT 显示成某种业务引擎。
    Runtime,
}

impl ProviderCapability {
    pub fn display_name(&self) -> &'static str {
        match self {
            ProviderCapability::Tts => "TTS",
            ProviderCapability::Asr => "ASR",
            ProviderCapability::Ocr => "OCR",
            ProviderCapability::Llm => "LLM",
            ProviderCapability::Translation => "翻译",
            ProviderCapability::VideoMaterial => "视频素材",
            ProviderCapability::Runtime => "运行时",
        }
    }
}

/// Provider 元信息
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    /// 能力类型
    pub capability: ProviderCapability,
    /// 引擎名（如 "kokoro", "whisper"）
    pub engine_name: String,
    /// 显示名称
    pub display_name: String,
    /// 引擎类型枚举
    pub engine_kind: EngineKind,
    /// 版本信息
    pub version: Option<String>,
}

/// Provider 注册表接口
///
/// 所有 Provider 的注册、发现和获取都通过此接口进行。
pub trait ProviderRegistry: Send + Sync {
    /// 注册一个 Provider
    ///
    /// 同一个 engine_kind 多次注册会覆盖上次的。
    fn register(
        &mut self,
        engine_kind: EngineKind,
        info: ProviderInfo,
        factory: Box<dyn ProviderFactory>,
    ) -> Result<(), DomainError>;

    /// 获取指定引擎的 Provider 工厂
    fn get_factory(&self, engine_kind: &EngineKind) -> Option<std::sync::Arc<dyn ProviderFactory>>;

    /// 按能力类型列出所有已注册的 Provider
    fn list_by_capability(&self, capability: ProviderCapability) -> Vec<ProviderInfo>;

    /// 列出所有已注册的 Provider
    fn list_all(&self) -> Vec<ProviderInfo>;

    /// 检查指定引擎是否已注册
    fn has_engine(&self, engine_kind: &EngineKind) -> bool;

    /// 获取已注册的引擎数量
    fn count(&self) -> usize;
}

/// Provider 工厂接口
///
/// 每个 Provider 引擎实现此接口以便运行时动态创建。
pub trait ProviderFactory: Send + Sync {
    /// 返回此工厂创建的 Provider 元信息
    fn info(&self) -> &ProviderInfo;

    /// 创建新的 Provider 实例（动态分发）
    ///
    /// 返回的 `Box<dyn Any>` 需要调用者 downcast 为具体 Provider trait。
    fn create(&self) -> Box<dyn std::any::Any>;
}

// ============================================================
// 辅助函数
// ============================================================

/// 从 EngineKind 的字符串表示获取对应的 ProviderCapability
pub fn engine_kind_to_capability(kind: &EngineKind) -> ProviderCapability {
    match kind {
        EngineKind::Kokoro
        | EngineKind::IndexTTS25
        | EngineKind::CosyVoice3
        | EngineKind::AzureTts
        | EngineKind::AliyunTts
        | EngineKind::Qwen3Tts => ProviderCapability::Tts,

        EngineKind::Whisper
        | EngineKind::SenseVoice
        | EngineKind::Paraformer
        | EngineKind::Qwen3Asr
        | EngineKind::AzureAsr
        | EngineKind::AliyunAsr
        | EngineKind::FireRedAsr
        | EngineKind::WeNet => ProviderCapability::Asr,

        EngineKind::PaddleOCR
        | EngineKind::EasyOcr => ProviderCapability::Ocr,

        EngineKind::DeepSeek
        | EngineKind::OpenAi
        | EngineKind::QwenLlm
        | EngineKind::Gemini
        | EngineKind::Ollama
        | EngineKind::AzureLlm => ProviderCapability::Llm,

        EngineKind::OpusMt
        | EngineKind::QwenMt
        | EngineKind::Nllb
        | EngineKind::M2m100
        | EngineKind::HyMt1_5
        | EngineKind::CTranslate2 => ProviderCapability::Translation,

        EngineKind::Pexels
        | EngineKind::Pixabay
        | EngineKind::Coverr => ProviderCapability::VideoMaterial,

        EngineKind::OnnxRuntime => ProviderCapability::Runtime,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_display() {
        assert_eq!(ProviderCapability::Tts.display_name(), "TTS");
        assert_eq!(ProviderCapability::Asr.display_name(), "ASR");
        assert_eq!(ProviderCapability::Llm.display_name(), "LLM");
    }

    #[test]
    fn test_engine_kind_to_capability() {
        assert_eq!(
            engine_kind_to_capability(&EngineKind::Kokoro),
            ProviderCapability::Tts
        );
        assert_eq!(
            engine_kind_to_capability(&EngineKind::Whisper),
            ProviderCapability::Asr
        );
        assert_eq!(
            engine_kind_to_capability(&EngineKind::PaddleOCR),
            ProviderCapability::Ocr
        );
        assert_eq!(
            engine_kind_to_capability(&EngineKind::DeepSeek),
            ProviderCapability::Llm
        );
        assert_eq!(
            engine_kind_to_capability(&EngineKind::OpusMt),
            ProviderCapability::Translation
        );
        assert_eq!(
            engine_kind_to_capability(&EngineKind::Pexels),
            ProviderCapability::VideoMaterial
        );
    }
}
