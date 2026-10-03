//! Azure Speech TTS 适配器
//!
//! 通过 Azure Speech Service REST API 实现文本转语音。
//! API 文档: https://learn.microsoft.com/zh-cn/azure/ai-services/speech-service/rest-text-to-speech

use crate::api::base::{ApiConfig, ApiError, BaseApiClient};
use std::time::Duration;
use votex_domain::error::TtsError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;
use votex_domain::tts::dialect::{Dialect, DialectQuality, DialectSupport};
use votex_domain::tts::provider::TtsProvider;
use votex_domain::tts::value_object::{TtsParams, VoiceId};

/// Azure Speech 服务区域
const DEFAULT_REGION: &str = "eastasia";

/// Azure Speech TTS 提供者
pub struct AzureSpeechTtsProvider {
    engine: EngineKind,
    client: BaseApiClient,
    region: String,
    api_key: String,
    sample_rate: u32,
}

impl AzureSpeechTtsProvider {
    /// 创建 Azure Speech TTS 提供者
    ///
    /// 从环境变量 `AZURE_SPEECH_KEY` 和 `AZURE_SPEECH_REGION` 读取配置。
    pub fn new() -> Result<Self, TtsError> {
        let api_key = std::env::var("AZURE_SPEECH_KEY")
            .map_err(|_| TtsError::SynthesisFailed("请设置 AZURE_SPEECH_KEY 环境变量".to_string()))?;
        let region = std::env::var("AZURE_SPEECH_REGION")
            .unwrap_or_else(|_| DEFAULT_REGION.to_string());

        let config = ApiConfig {
            api_key: Some(api_key.clone()),
            endpoint: Some(format!("https://{}.tts.speech.microsoft.com", region)),
            timeout: Duration::from_secs(120),
            retry_max: 3,
            ..Default::default()
        };

        Ok(Self {
            engine: EngineKind::Kokoro,
            client: BaseApiClient::new(config),
            region,
            api_key,
            sample_rate: 24000,
        })
    }

    /// 构建 SSML 请求体
    fn build_ssml(&self, text: &str, voice_id: &str) -> String {
        format!(
            r#"<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="zh-CN">
    <voice name="{}">
        <prosody rate="0%" pitch="0%">
            {}
        </prosody>
    </voice>
</speak>"#,
            voice_id,
            text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
        )
    }
}

impl TtsProvider for AzureSpeechTtsProvider {
    fn engine_kind(&self) -> EngineKind {
        self.engine
    }

    fn load(&mut self, _model: &Model) -> Result<(), TtsError> {
        Ok(())
    }

    fn unload(&mut self) -> Result<(), TtsError> {
        Ok(())
    }

    fn synthesize(
        &self,
        text: &str,
        voice: &VoiceId,
        _params: &TtsParams,
    ) -> Result<AudioData, TtsError> {
        let ssml = self.build_ssml(text, &voice.id);

        let headers = vec![
            ("Ocp-Apim-Subscription-Key".to_string(), self.api_key.clone()),
            ("Content-Type".to_string(), "application/ssml+xml".to_string()),
            ("X-Microsoft-OutputFormat".to_string(), "raw-24khz-16bit-mono-pcm".to_string()),
        ];

        let url = format!(
            "https://{}.tts.speech.microsoft.com/cognitiveservices/v1",
            self.region
        );

        let resp = self
            .client
            .post_bytes(&url, Some(&headers), ssml.as_bytes())
            .map_err(|e| TtsError::SynthesisFailed(match e {
                ApiError::ApiKeyNotConfigured(_) => "Azure Speech Key 未配置".to_string(),
                ApiError::AuthenticationFailed => "Azure Speech 认证失败，请检查 API Key".to_string(),
                ApiError::RateLimited => "请求被限流".to_string(),
                ApiError::NetworkError(s) => format!("网络错误: {}", s),
                ApiError::RequestFailed(s) => format!("请求失败: {}", s),
                ApiError::ParseFailed(s) => format!("解析失败: {}", s),
            }))?;

        let bytes: Vec<u8> = resp.bytes().map_err(|e| {
            TtsError::SynthesisFailed(format!("读取响应失败: {}", e))
        })?.to_vec();

        let samples: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_ne_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0
            })
            .collect();

        Ok(AudioData {
            samples,
            sample_rate: self.sample_rate,
            channels: 1,
        })
    }

    fn list_voices(&self) -> Vec<VoiceId> {
        let engine = self.engine;
        vec![
            VoiceId::new("zh-CN-XiaoxiaoNeural", "晓晓", engine),
            VoiceId::new("zh-CN-XiaoyiNeural", "晓伊", engine),
            VoiceId::new("zh-CN-YunjianNeural", "云健", engine),
            VoiceId::new("zh-CN-YunyangNeural", "云扬", engine),
            VoiceId::new("zh-CN-YunxiNeural", "云溪", engine),
            VoiceId::new("zh-HK-HiuMaanNeural", "晓曼", engine),
            VoiceId::new("zh-TW-HsiaoChenNeural", "晓臻", engine),
        ]
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn is_loaded(&self) -> bool {
        true
    }

    fn supported_dialects(&self) -> Vec<DialectSupport> {
        vec![
            DialectSupport::new(Dialect::Mandarin, DialectQuality::Native)
                .with_voice("zh-CN-XiaoxiaoNeural"),
            DialectSupport::new(Dialect::Cantonese, DialectQuality::Native)
                .with_voice("zh-HK-HiuMaanNeural"),
        ]
    }
}
