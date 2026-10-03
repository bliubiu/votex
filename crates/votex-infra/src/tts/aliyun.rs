//! 阿里云智能语音 TTS 适配器
//!
//! 通过阿里云 NLS REST API 实现文本转语音。
//! API 文档: https://help.aliyun.com/product/30413.html
//!
//! 使用 HMAC-SHA1 签名认证，需要环境变量：
//! - ALIYUN_ACCESS_KEY_ID
//! - ALIYUN_ACCESS_KEY_SECRET
//! - ALIYUN_APP_KEY（可选）

use std::time::{Duration, SystemTime, UNIX_EPOCH};
use votex_domain::error::TtsError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;
use votex_domain::tts::dialect::{Dialect, DialectQuality, DialectSupport};
use votex_domain::tts::provider::TtsProvider;
use votex_domain::tts::value_object::{TtsParams, VoiceId};

/// 阿里云 NLS 网关
const ALIYUN_TTS_ENDPOINT: &str = "https://nls-gateway.aliyuncs.com/stream/v1/tts";

/// 阿里云语音角色
const ALIYUN_VOICES: &[(&str, &str, &str)] = &[
    ("zhitian", "知甜", "中文女声"),
    ("zhiqi", "知琪", "中文男声"),
    ("zhiyuan", "致远", "中文男声"),
    ("zhixuan", "知璇", "中文女声"),
];

/// 阿里云 TTS 提供者
pub struct AliyunTtsProvider {
    engine: EngineKind,
    client: reqwest::blocking::Client,
    access_key: String,
    secret_key: String,
    app_key: String,
}

impl AliyunTtsProvider {
    /// 创建阿里云 TTS 提供者
    pub fn new() -> Result<Self, TtsError> {
        let access_key = std::env::var("ALIYUN_ACCESS_KEY_ID")
            .map_err(|_| TtsError::SynthesisFailed("请设置 ALIYUN_ACCESS_KEY_ID 环境变量".to_string()))?;
        let secret_key = std::env::var("ALIYUN_ACCESS_KEY_SECRET")
            .map_err(|_| TtsError::SynthesisFailed("请设置 ALIYUN_ACCESS_KEY_SECRET 环境变量".to_string()))?;
        let app_key = std::env::var("ALIYUN_APP_KEY").unwrap_or_default();

        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120))
            .user_agent("Votex/1.0")
            .build()
            .map_err(|e| TtsError::SynthesisFailed(format!("创建 HTTP 客户端失败: {}", e)))?;

        Ok(Self {
            engine: EngineKind::Kokoro,
            client,
            access_key,
            secret_key,
            app_key,
        })
    }

    /// 生成 HMAC-SHA1 签名
    fn sign(method: &str, params: &[(&str, String)], secret_key: &str) -> Result<String, TtsError> {
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
            .map_err(|e| TtsError::SynthesisFailed(format!("HMAC 初始化失败: {}", e)))?;
        mac.update(string_to_sign.as_bytes());
        let result = mac.finalize();
        let code_bytes = result.into_bytes();
        Ok(base64::Engine::encode(&base64::engine::general_purpose::STANDARD, code_bytes))
    }

    /// 构建并发送 TTS 请求
    fn do_synthesize(
        &self,
        text: &str,
        voice: &str,
        sample_rate: u32,
    ) -> Result<Vec<u8>, TtsError> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let params = vec![
            ("appkey", self.app_key.clone()),
            ("format", "wav".to_string()),
            ("sample_rate", sample_rate.to_string()),
            ("voice", voice.to_string()),
            ("text", text.to_string()),
            ("volume", "50".to_string()),
        ];

        let mut all_params = params.clone();
        all_params.push(("access_key_id", self.access_key.clone()));
        all_params.push(("timestamp", timestamp.to_string()));
        all_params.push(("signature_version", "1.0".to_string()));
        all_params.push(("signature_method", "HMAC-SHA1".to_string()));

        let signature = Self::sign("GET", &all_params, &self.secret_key)?;
        all_params.push(("signature", signature));

        let query_string: String = all_params
            .iter()
            .map(|(k, v)| format!("{}={}", percent_encode(k), percent_encode(v)))
            .collect::<Vec<_>>()
            .join("&");

        let url = format!("{}?{}", ALIYUN_TTS_ENDPOINT, query_string);

        let resp = self
            .client
            .get(&url)
            .send()
            .map_err(|e| TtsError::SynthesisFailed(format!("请求失败: {}", e)))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().unwrap_or_default();
            return Err(TtsError::SynthesisFailed(format!("HTTP {}: {}", status, body)));
        }

        let bytes = resp.bytes().map_err(|e| TtsError::SynthesisFailed(e.to_string()))?;
        Ok(bytes.to_vec())
    }
}

impl TtsProvider for AliyunTtsProvider {
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
        let sample_rate = 16000u32;
        let wav_bytes = self.do_synthesize(text, &voice.id, sample_rate)?;

        // 解析 WAV 数据为 AudioData（跳过 WAV 头，取 PCM 数据）
        // WAV header 通常是 44 字节
        let pcm_data = if wav_bytes.len() > 44 {
            wav_bytes[44..].to_vec()
        } else {
            wav_bytes.clone()
        };

        let samples: Vec<f32> = pcm_data
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_ne_bytes([chunk[0], chunk[1]]);
                sample as f32 / 32768.0
            })
            .collect();

        Ok(AudioData {
            samples,
            sample_rate,
            channels: 1,
        })
    }

    fn list_voices(&self) -> Vec<VoiceId> {
        let engine = self.engine;
        ALIYUN_VOICES
            .iter()
            .map(|(id, name, _)| VoiceId::new(id, name, engine))
            .collect()
    }

    fn sample_rate(&self) -> u32 {
        16000
    }

    fn is_loaded(&self) -> bool {
        true
    }

    fn supported_dialects(&self) -> Vec<DialectSupport> {
        vec![
            DialectSupport::new(Dialect::Mandarin, DialectQuality::Native)
                .with_voice("zhitian"),
        ]
    }
}

/// 百分号编码（简化实现，仅处理 URL 不安全字符）
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
