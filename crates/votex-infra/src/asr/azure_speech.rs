//! Azure Speech ASR 适配器
//!
//! 通过 Azure Speech Service REST API 实现语音转文字。
//! API 文档: https://learn.microsoft.com/zh-cn/azure/ai-services/speech-service/rest-speech-to-text

use crate::api::base::{ApiConfig, ApiError, BaseApiClient};
use std::time::Duration;
use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, RecognizeOutput, WordTimestamp};
use votex_domain::error::AsrError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;

const DEFAULT_REGION: &str = "eastasia";

/// Azure Speech ASR 提供者
pub struct AzureSpeechAsrProvider {
    engine: EngineKind,
    client: BaseApiClient,
    region: String,
    api_key: String,
}

impl AzureSpeechAsrProvider {
    /// 创建 Azure Speech ASR 提供者
    ///
    /// 从环境变量 `AZURE_SPEECH_KEY` 和 `AZURE_SPEECH_REGION` 读取配置。
    pub fn new() -> Result<Self, AsrError> {
        let api_key = std::env::var("AZURE_SPEECH_KEY")
            .map_err(|_| AsrError::RecognizeFailed("请设置 AZURE_SPEECH_KEY 环境变量".to_string()))?;
        let region = std::env::var("AZURE_SPEECH_REGION")
            .unwrap_or_else(|_| DEFAULT_REGION.to_string());

        let config = ApiConfig {
            api_key: Some(api_key.clone()),
            endpoint: Some(format!("https://{}.stt.speech.microsoft.com", region)),
            timeout: Duration::from_secs(300),
            retry_max: 3,
            ..Default::default()
        };

        Ok(Self {
            engine: EngineKind::AzureAsr,
            client: BaseApiClient::new(config),
            region,
            api_key,
        })
    }
}

impl AsrProvider for AzureSpeechAsrProvider {
    fn engine_kind(&self) -> EngineKind {
        self.engine
    }

    fn load(&self, _model: &Model) -> Result<(), AsrError> {
        // Azure Speech 为在线 API，无需加载模型
        Ok(())
    }

    fn unload(&self) -> Result<(), AsrError> {
        Ok(())
    }

    fn recognize(
        &self,
        audio: &AudioData,
        _params: &AsrParams,
    ) -> Result<RecognizeOutput, AsrError> {
        // 将 f32 PCM 转换为 16-bit PCM WAV 格式
        let wav_bytes = audio_to_wav_bytes(audio)?;

        let headers = vec![
            ("Ocp-Apim-Subscription-Key".to_string(), self.api_key.clone()),
            ("Content-Type".to_string(), "audio/wav; codecs=audio/pcm; samplerate=16000".to_string()),
            ("Accept".to_string(), "application/json".to_string()),
        ];

        let url = format!(
            "https://{}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?language=zh-CN&format=detailed",
            self.region
        );

        let resp = self
            .client
            .post_bytes(&url, Some(&headers), &wav_bytes)
            .map_err(|e| AsrError::RecognizeFailed(match e {
                ApiError::ApiKeyNotConfigured(_) => "Azure Speech Key 未配置".to_string(),
                ApiError::AuthenticationFailed => "Azure Speech 认证失败".to_string(),
                ApiError::RateLimited => "请求被限流".to_string(),
                ApiError::NetworkError(s) => format!("网络错误: {}", s),
                ApiError::RequestFailed(s) => format!("请求失败: {}", s),
                ApiError::ParseFailed(s) => format!("解析失败: {}", s),
            }))?;

        let body: serde_json::Value = resp.json()
            .map_err(|e| AsrError::RecognizeFailed(format!("解析响应失败: {}", e)))?;

        let display_text = body["DisplayText"]
            .as_str()
            .unwrap_or("")
            .to_string();

        // 解析词级时间戳
        let mut word_timestamps = Vec::new();
        if let Some(nbest) = body["NBest"].as_array() {
            if let Some(first) = nbest.first() {
                if let Some(words) = first["Words"].as_array() {
                    for word in words {
                        let w = word["Word"].as_str().unwrap_or("").to_string();
                        let start = word["Offset"].as_f64().unwrap_or(0.0) / 10_000_000.0; // 100ns → ms
                        let duration = word["Duration"].as_f64().unwrap_or(0.0) / 10_000_000.0;
                        word_timestamps.push(WordTimestamp {
                            word: w,
                            start_ms: start,
                            end_ms: start + duration,
                        });
                    }
                }
            }
        }

        Ok(RecognizeOutput {
            text: display_text,
            word_timestamps,
        })
    }

    fn sample_rate(&self) -> u32 {
        16000
    }

    fn is_loaded(&self) -> bool {
        true
    }
}

/// 将 AudioData (f32 PCM) 转换为 WAV 格式字节
fn audio_to_wav_bytes(audio: &AudioData) -> Result<Vec<u8>, AsrError> {
    // 重采样到 16kHz 单声道
    let mono = audio.to_mono_f32_16k();

    // 转换为 16-bit PCM
    let pcm: Vec<i16> = mono
        .iter()
        .map(|&s| {
            (s.max(-1.0).min(1.0) * 32767.0) as i16
        })
        .collect();

    let sample_rate = 16000u32;
    let channels = 1u16;
    let bits_per_sample = 16u16;
    let byte_rate = sample_rate * channels as u32 * (bits_per_sample / 8) as u32;
    let block_align = channels * (bits_per_sample / 8);
    let data_size = pcm.len() as u32 * (bits_per_sample / 8) as u32;
    let file_size = 36 + data_size;

    let mut wav = Vec::with_capacity(file_size as usize);

    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav.extend_from_slice(&1u16.to_le_bytes());  // PCM format
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    for &sample in &pcm {
        wav.extend_from_slice(&sample.to_ne_bytes());
    }

    Ok(wav)
}
