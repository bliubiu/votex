//! 阿里云智能语音 ASR 适配器
//!
//! 通过阿里云 NLS REST API 实现语音转文字。
//! API 文档: https://help.aliyun.com/product/30413.html
//!
//! 需要环境变量：
//! - ALIYUN_ACCESS_KEY_ID
//! - ALIYUN_ACCESS_KEY_SECRET
//! - ALIYUN_APP_KEY（可选）

use std::time::{Duration, SystemTime, UNIX_EPOCH};
use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, RecognizeOutput, WordTimestamp};
use votex_domain::error::AsrError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;

/// 阿里云 NLS 网关
const ALIYUN_ASR_ENDPOINT: &str = "https://nls-gateway.aliyuncs.com/stream/v1/asr";

/// 阿里云 ASR 提供者
pub struct AliyunAsrProvider {
    engine: EngineKind,
    client: reqwest::blocking::Client,
    access_key: String,
    secret_key: String,
    app_key: String,
}

impl AliyunAsrProvider {
    /// 创建阿里云 ASR 提供者
    pub fn new() -> Result<Self, AsrError> {
        let access_key = std::env::var("ALIYUN_ACCESS_KEY_ID")
            .map_err(|_| AsrError::RecognizeFailed("请设置 ALIYUN_ACCESS_KEY_ID 环境变量".to_string()))?;
        let secret_key = std::env::var("ALIYUN_ACCESS_KEY_SECRET")
            .map_err(|_| AsrError::RecognizeFailed("请设置 ALIYUN_ACCESS_KEY_SECRET 环境变量".to_string()))?;
        let app_key = std::env::var("ALIYUN_APP_KEY").unwrap_or_default();

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(300))
            .user_agent("Votex/1.0")
            .build()
            .map_err(|e| AsrError::RecognizeFailed(format!("创建 HTTP 客户端失败: {}", e)))?;

        Ok(Self {
            engine: EngineKind::AliyunAsr,
            client,
            access_key,
            secret_key,
            app_key,
        })
    }

    /// 生成 HMAC-SHA1 签名
    fn sign(method: &str, params: &[(&str, String)], secret_key: &str) -> Result<String, AsrError> {
        use hmac::Mac;
        use sha1::Sha1;

        let mut sorted: Vec<(&str, &String)> = params.iter().map(|(k, v)| (*k, v)).collect();
        sorted.sort_by(|a, b| a.0.cmp(b.0));

        let canonical: String = sorted
            .iter()
            .map(|(k, v)| format!("{}={}", percent_encode(k), percent_encode(v)))
            .collect::<Vec<_>>()
            .join("&");

        let string_to_sign = format!("{}&%2F&{}", method, percent_encode(&canonical));

        let mut mac = hmac::Hmac::<Sha1>::new_from_slice(secret_key.as_bytes())
            .map_err(|e| AsrError::RecognizeFailed(format!("HMAC 初始化失败: {}", e)))?;
        mac.update(string_to_sign.as_bytes());
        let result = mac.finalize();
        let code_bytes = result.into_bytes();
        Ok(base64::Engine::encode(&base64::engine::general_purpose::STANDARD, code_bytes))
    }
}

impl AsrProvider for AliyunAsrProvider {
    fn engine_kind(&self) -> EngineKind {
        self.engine
    }

    fn load(&self, _model: &Model) -> Result<(), AsrError> {
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
        let wav_bytes = audio_to_wav_bytes(audio)?;
        let audio_b64 = base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            &wav_bytes,
        );

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let mut params = vec![
            ("appkey", self.app_key.clone()),
            ("format", "wav".to_string()),
            ("sample_rate", "16000".to_string()),
            ("enable_punctuation_prediction", "true".to_string()),
            ("enable_inverse_text_normalization", "true".to_string()),
            ("audio_data", audio_b64),
        ];
        params.push(("access_key_id", self.access_key.clone()));
        params.push(("timestamp", timestamp.to_string()));
        params.push(("signature_version", "1.0".to_string()));
        params.push(("signature_method", "HMAC-SHA1".to_string()));

        let signature = Self::sign("POST", &params, &self.secret_key)?;

        // 构建 JSON 请求体（签名不包含在签名参数中）
        let mut body_map = serde_json::Map::new();
        for (k, v) in &params {
            body_map.insert(k.to_string(), serde_json::Value::String(v.clone()));
        }
        body_map.insert("signature".to_string(), serde_json::Value::String(signature));
        let body = serde_json::Value::Object(body_map);

        let resp = self
            .client
            .post(ALIYUN_ASR_ENDPOINT)
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .map_err(|e| AsrError::RecognizeFailed(format!("请求失败: {}", e)))?;

        let status = resp.status();
        let result: serde_json::Value = resp.json()
            .map_err(|e| AsrError::RecognizeFailed(format!("解析响应失败: {}", e)))?;

        if status != 200 {
            return Err(AsrError::RecognizeFailed(format!(
                "阿里云识别失败 ({}): {}",
                status,
                result["message"].as_str().unwrap_or("未知错误")
            )));
        }

        let text = result["result"].as_str().unwrap_or("").to_string();

        let word_timestamps = vec![WordTimestamp {
            word: text.clone(),
            start_ms: 0.0,
            end_ms: 0.0,
        }];

        Ok(RecognizeOutput {
            text,
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

/// 将 AudioData 转换为 WAV 字节
fn audio_to_wav_bytes(audio: &AudioData) -> Result<Vec<u8>, AsrError> {
    let mono = audio.to_mono_f32_16k();
    let pcm: Vec<i16> = mono
        .iter()
        .map(|&s| (s.max(-1.0).min(1.0) * 32767.0) as i16)
        .collect();

    let sample_rate = 16000u32;
    let channels = 1u16;
    let bits_per_sample = 16u16;
    let byte_rate = sample_rate * channels as u32 * (bits_per_sample / 8) as u32;
    let block_align = channels * (bits_per_sample / 8);
    let data_size = pcm.len() as u32 * (bits_per_sample / 8) as u32;
    let file_size = 36 + data_size;

    let mut wav = Vec::with_capacity(file_size as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());
    for &sample in &pcm {
        wav.extend_from_slice(&sample.to_ne_bytes());
    }
    Ok(wav)
}

/// 百分号编码
fn percent_encode(s: &str) -> String {
    let mut encoded = String::new();
    for byte in s.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                encoded.push(byte as char);
            }
            _ => {
                encoded.push_str(&format!("%{:02X}", byte));
            }
        }
    }
    encoded
}
