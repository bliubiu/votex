use thiserror::Error;

/// LLM 错误
#[derive(Debug, Error)]
pub enum LlmError {
    #[error("API 调用失败: {0}")]
    ApiError(String),

    #[error("API 返回空结果")]
    EmptyResponse,

    #[error("API Key 未配置")]
    ApiKeyNotConfigured,

    #[error("不支持的模型: {0}")]
    UnsupportedModel(String),

    #[error("网络错误: {0}")]
    NetworkError(String),

    #[error("请求超时")]
    Timeout,
}

impl LlmError {
    pub fn error_code(&self) -> &str {
        match self {
            LlmError::ApiError(_) => "VL001",
            LlmError::EmptyResponse => "VL002",
            LlmError::ApiKeyNotConfigured => "VL003",
            LlmError::UnsupportedModel(_) => "VL004",
            LlmError::NetworkError(_) => "VL005",
            LlmError::Timeout => "VL006",
        }
    }
}

/// 视频素材错误
#[derive(Debug, Error)]
pub enum MaterialError {
    #[error("API 调用失败: {0}")]
    ApiError(String),

    #[error("API Key 未配置")]
    ApiKeyNotConfigured,

    #[error("下载失败: {0}")]
    DownloadFailed(String),

    #[error("不支持的素材源: {0}")]
    UnsupportedSource(String),
}

impl MaterialError {
    pub fn error_code(&self) -> &str {
        match self {
            MaterialError::ApiError(_) => "VMAT001",
            MaterialError::ApiKeyNotConfigured => "VMAT002",
            MaterialError::DownloadFailed(_) => "VMAT003",
            MaterialError::UnsupportedSource(_) => "VMAT004",
        }
    }
}

/// 翻译错误
#[derive(Debug, Error)]
pub enum TranslationError {
    #[error("API 调用失败: {0}")]
    ApiError(String),

    #[error("翻译模型未加载")]
    ModelNotLoaded,

    #[error("不支持的语言方向")]
    UnsupportedDirection,

    #[error("文本为空")]
    EmptyText,

    #[error("不支持的翻译引擎: {0}")]
    UnsupportedEngine(String),

    #[error("翻译方向解析失败: {0}")]
    InvalidDirection(String),

    #[error("长文本分段失败: {0}")]
    SegmentFailed(String),

    #[error("术语表处理失败: {0}")]
    GlossaryFailed(String),

    #[error("翻译已被取消")]
    Cancelled,
}

impl TranslationError {
    pub fn error_code(&self) -> &str {
        match self {
            TranslationError::ApiError(_) => "VTR001",
            TranslationError::ModelNotLoaded => "VTR002",
            TranslationError::UnsupportedDirection => "VTR003",
            TranslationError::EmptyText => "VTR004",
            TranslationError::UnsupportedEngine(_) => "VTR005",
            TranslationError::InvalidDirection(_) => "VTR006",
            TranslationError::SegmentFailed(_) => "VTR007",
            TranslationError::GlossaryFailed(_) => "VTR008",
            TranslationError::Cancelled => "VTR009",
        }
    }
}

/// 视频合成错误
#[derive(Debug, Error)]
pub enum VideoError {
    #[error("视频合成失败: {0}")]
    ComposeFailed(String),

    #[error("ffmpeg 未安装")]
    FFmpegNotFound,

    #[error("合成超时")]
    Timeout,

    #[error("不支持的合成引擎: {0}")]
    UnsupportedEngine(String),

    #[error("不支持的参数: {0}")]
    InvalidParameter(String),
}

impl VideoError {
    pub fn error_code(&self) -> &str {
        match self {
            VideoError::ComposeFailed(_) => "VV001",
            VideoError::FFmpegNotFound => "VV002",
            VideoError::Timeout => "VV003",
            VideoError::UnsupportedEngine(_) => "VV004",
            VideoError::InvalidParameter(_) => "VV005",
        }
    }
}

/// 加密错误
#[derive(Debug, Error)]
pub enum CryptoError {
    #[error("加密引擎未初始化")]
    NotInitialized,

    #[error("密钥长度无效")]
    InvalidKeyLength,

    #[error("密钥文件不存在: {0}")]
    KeyFileNotFound(String),

    #[error("解密失败: {0}")]
    DecryptFailed(String),

    #[error("加密失败: {0}")]
    EncryptFailed(String),

    #[error("无效的加密数据格式")]
    InvalidFormat,
}

impl CryptoError {
    pub fn error_code(&self) -> &str {
        match self {
            CryptoError::NotInitialized => "VCR001",
            CryptoError::InvalidKeyLength => "VCR002",
            CryptoError::KeyFileNotFound(_) => "VCR003",
            CryptoError::DecryptFailed(_) => "VCR004",
            CryptoError::EncryptFailed(_) => "VCR005",
            CryptoError::InvalidFormat => "VCR006",
        }
    }
}

/// 模型管理错误
#[derive(Debug, Error)]
pub enum ModelError {
    #[error("模型不存在: {0}")]
    NotFound(String),

    #[error("模型已存在: {0}")]
    AlreadyExists(String),

    #[error("模型状态不允许此操作: 当前状态 {current}, 期望 {expected}")]
    InvalidStatus { current: String, expected: String },

    #[error("模型下载失败: {0}")]
    DownloadFailed(String),

    #[error("模型校验失败: {0}")]
    VerifyFailed(String),

    #[error("模型加载失败: {0}")]
    LoadFailed(String),

    #[error("模型文件不存在: {0}")]
    FileNotFound(String),

    #[error("网络错误: {0}")]
    NetworkError(String),

    #[error("磁盘空间不足")]
    InsufficientDiskSpace,

    #[error("IO 错误: {0}")]
    IoError(String),
}

impl ModelError {
    pub fn error_code(&self) -> &str {
        match self {
            ModelError::NotFound(_) => "VM001",
            ModelError::AlreadyExists(_) => "VM002",
            ModelError::InvalidStatus { .. } => "VM003",
            ModelError::DownloadFailed(_) => "VM004",
            ModelError::VerifyFailed(_) => "VM005",
            ModelError::LoadFailed(_) => "VM006",
            ModelError::FileNotFound(_) => "VM007",
            ModelError::NetworkError(_) => "VM008",
            ModelError::InsufficientDiskSpace => "VM009",
            ModelError::IoError(_) => "VM010",
        }
    }
}

/// TTS 合成错误
#[derive(Debug, Error)]
pub enum TtsError {
    #[error("TTS 引擎未加载")]
    EngineNotLoaded,

    #[error("不支持的引擎: {0}")]
    UnsupportedEngine(String),

    #[error("不支持的音色: {0}")]
    UnsupportedVoice(String),

    #[error("文本为空")]
    EmptyText,

    #[error("文本分段失败: {0}")]
    SegmentFailed(String),

    #[error("语音合成失败: {0}")]
    SynthesisFailed(String),

    #[error("音频拼接失败: {0}")]
    ConcatFailed(String),

    #[error("音频导出失败: {0}")]
    ExportFailed(String),

    #[error("输入文件不存在: {0}")]
    InputFileNotFound(String),

    #[error("编码检测失败")]
    EncodingDetectionFailed,

    #[error("任务不存在: {0}")]
    TaskNotFound(String),
}

impl TtsError {
    pub fn error_code(&self) -> &str {
        match self {
            TtsError::EngineNotLoaded => "VT001",
            TtsError::UnsupportedEngine(_) => "VT002",
            TtsError::UnsupportedVoice(_) => "VT003",
            TtsError::EmptyText => "VT004",
            TtsError::SegmentFailed(_) => "VT005",
            TtsError::SynthesisFailed(_) => "VT006",
            TtsError::ConcatFailed(_) => "VT007",
            TtsError::ExportFailed(_) => "VT008",
            TtsError::InputFileNotFound(_) => "VT009",
            TtsError::EncodingDetectionFailed => "VT010",
            TtsError::TaskNotFound(_) => "VT011",
        }
    }
}

/// ASR 识别错误
#[derive(Debug, Error)]
pub enum AsrError {
    #[error("ASR 引擎未加载")]
    EngineNotLoaded,

    #[error("模型文件不存在: {0}")]
    ModelNotFound(String),

    #[error("模型加载失败: {0}")]
    LoadFailed(String),

    #[error("不支持的模型: {0}")]
    UnsupportedModel(String),

    #[error("输入文件不存在: {0}")]
    InputFileNotFound(String),

    #[error("不支持的媒体格式: {0}")]
    UnsupportedFormat(String),

    #[error("音频提取失败: {0}")]
    AudioExtractFailed(String),

    #[error("音频切片失败: {0}")]
    SliceFailed(String),

    #[error("音频为空或过短")]
    EmptyAudio,

    #[error("语音识别失败: {0}")]
    RecognizeFailed(String),

    #[error("字幕导出失败: {0}")]
    ExportFailed(String),
}

impl AsrError {
    pub fn error_code(&self) -> &str {
        match self {
            AsrError::EngineNotLoaded => "VA001",
            AsrError::UnsupportedModel(_) => "VA002",
            AsrError::InputFileNotFound(_) => "VA003",
            AsrError::UnsupportedFormat(_) => "VA004",
            AsrError::AudioExtractFailed(_) => "VA005",
            AsrError::SliceFailed(_) => "VA006",
            AsrError::EmptyAudio => "VA007",
            AsrError::RecognizeFailed(_) => "VA008",
            AsrError::ModelNotFound(_) => "VA009",
            AsrError::ExportFailed(_) => "VA011",
            AsrError::LoadFailed(_) => "VA010",
        }
    }
}

/// OCR 识别错误
#[derive(Debug, Error)]
pub enum OcrError {
    #[error("OCR 引擎未加载")]
    EngineNotLoaded,

    #[error("不支持的模型: {0}")]
    UnsupportedModel(String),

    #[error("输入图片不存在: {0}")]
    InputFileNotFound(String),

    #[error("不支持的图片格式: {0}")]
    UnsupportedFormat(String),

    #[error("图片读取失败: {0}")]
    ImageReadFailed(String),

    #[error("文字检测失败: {0}")]
    DetectFailed(String),

    #[error("方向分类失败: {0}")]
    ClassifyFailed(String),

    #[error("文字识别失败: {0}")]
    RecognizeFailed(String),

    #[error("结果导出失败: {0}")]
    ExportFailed(String),

    #[error("引擎加载失败: {0}")]
    LoadFailed(String),
}

impl OcrError {
    pub fn error_code(&self) -> &str {
        match self {
            OcrError::EngineNotLoaded => "VO001",
            OcrError::UnsupportedModel(_) => "VO002",
            OcrError::InputFileNotFound(_) => "VO003",
            OcrError::UnsupportedFormat(_) => "VO004",
            OcrError::ImageReadFailed(_) => "VO005",
            OcrError::DetectFailed(_) => "VO006",
            OcrError::ClassifyFailed(_) => "VO007",
            OcrError::RecognizeFailed(_) => "VO008",
            OcrError::ExportFailed(_) => "VO009",
            OcrError::LoadFailed(_) => "VO010",
        }
    }
}

/// 流水线错误
#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("流水线不存在: {0}")]
    NotFound(String),

    #[error("阶段执行失败: 阶段 {stage}, 原因: {reason}")]
    StageFailed { stage: usize, reason: String },

    #[error("流水线状态不允许此操作")]
    InvalidStatus,

    #[error("不支持的流水线类型")]
    UnsupportedKind,

    #[error("输入参数不完整: {0}")]
    IncompleteInput(String),
}

impl PipelineError {
    pub fn error_code(&self) -> &str {
        match self {
            PipelineError::NotFound(_) => "VP001",
            PipelineError::StageFailed { .. } => "VP002",
            PipelineError::InvalidStatus => "VP003",
            PipelineError::UnsupportedKind => "VP004",
            PipelineError::IncompleteInput(_) => "VP005",
        }
    }
}

/// 配置错误
#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("配置文件读取失败: {0}")]
    ReadFailed(String),

    #[error("配置文件写入失败: {0}")]
    WriteFailed(String),

    #[error("配置项无效: {key} = {value}")]
    InvalidValue { key: String, value: String },

    #[error("配置项不存在: {0}")]
    NotFound(String),
}

impl ConfigError {
    pub fn error_code(&self) -> &str {
        match self {
            ConfigError::ReadFailed(_) => "VCFG001",
            ConfigError::WriteFailed(_) => "VCFG002",
            ConfigError::InvalidValue { .. } => "VCFG003",
            ConfigError::NotFound(_) => "VCFG004",
        }
    }
}

/// 领域层统一错误
#[derive(Debug, Error)]
pub enum DomainError {
    #[error("{0}")]
    Model(#[from] ModelError),

    #[error("{0}")]
    Tts(#[from] TtsError),

    #[error("{0}")]
    Asr(#[from] AsrError),

    #[error("{0}")]
    Ocr(#[from] OcrError),

    #[error("{0}")]
    Pipeline(#[from] PipelineError),

    #[error("{0}")]
    Config(#[from] ConfigError),

    #[error("{0}")]
    Llm(#[from] LlmError),

    #[error("{0}")]
    Material(#[from] MaterialError),

    #[error("{0}")]
    Translation(#[from] TranslationError),

    #[error("{0}")]
    Video(#[from] VideoError),

    #[error("{0}")]
    Crypto(#[from] CryptoError),
}

impl DomainError {
    pub fn error_code(&self) -> &str {
        match self {
            DomainError::Model(e) => e.error_code(),
            DomainError::Tts(e) => e.error_code(),
            DomainError::Asr(e) => e.error_code(),
            DomainError::Ocr(e) => e.error_code(),
            DomainError::Pipeline(e) => e.error_code(),
            DomainError::Config(e) => e.error_code(),
            DomainError::Llm(e) => e.error_code(),
            DomainError::Material(e) => e.error_code(),
            DomainError::Translation(e) => e.error_code(),
            DomainError::Video(e) => e.error_code(),
            DomainError::Crypto(e) => e.error_code(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_error_错误码映射() {
        assert_eq!(ModelError::NotFound("test".into()).error_code(), "VM001");
        assert_eq!(ModelError::DownloadFailed("err".into()).error_code(), "VM004");
        assert_eq!(ModelError::InsufficientDiskSpace.error_code(), "VM009");
    }

    #[test]
    fn tts_error_错误码映射() {
        assert_eq!(TtsError::EngineNotLoaded.error_code(), "VT001");
        assert_eq!(TtsError::EmptyText.error_code(), "VT004");
    }

    #[test]
    fn asr_error_错误码映射() {
        assert_eq!(AsrError::EngineNotLoaded.error_code(), "VA001");
        assert_eq!(AsrError::RecognizeFailed("err".into()).error_code(), "VA008");
        assert_eq!(AsrError::ModelNotFound("m".into()).error_code(), "VA009");
        assert_eq!(AsrError::ExportFailed("e".into()).error_code(), "VA011");
        assert_eq!(AsrError::LoadFailed("l".into()).error_code(), "VA010");
        // VA009 不再冲突：ModelNotFound=VA009, ExportFailed=VA011
        assert_ne!(
            AsrError::ModelNotFound("m".into()).error_code(),
            AsrError::ExportFailed("e".into()).error_code(),
            "ModelNotFound 与 ExportFailed 错误码不能相同"
        );
    }

    #[test]
    fn domain_error_嵌套错误码() {
        let err = DomainError::from(TtsError::EmptyText);
        assert_eq!(err.error_code(), "VT004");
    }
}
