//! AES-256-GCM 加密模块
//!
//! 提供配置敏感字段的加密/解密功能。
//! 支持 ENC(xxx) 标记识别，避免循环加密。

use aes_gcm::{
    aead::{Aead, KeyInit},
    Aes256Gcm, Nonce,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use rand::rngs::OsRng;
use rand::Rng;
use std::path::PathBuf;
use std::sync::Mutex;

use votex_domain::error::CryptoError;

const KEY_SIZE: usize = 32; // 256 bits
const NONCE_SIZE: usize = 12; // 96 bits
const KEY_FILENAME: &str = ".key";

pub struct CryptoEngine {
    key_dir: PathBuf,
    cipher: Mutex<Option<Aes256Gcm>>,
}

impl CryptoEngine {
    pub fn new(key_dir: Option<PathBuf>) -> Self {
        Self {
            key_dir: key_dir.unwrap_or_else(|| std::env::current_dir().unwrap_or_default()),
            cipher: Mutex::new(None),
        }
    }

    /// 初始化加密引擎（加载/生成密钥）
    pub fn initialize(&self) -> Result<(), CryptoError> {
        let key = self.load_or_generate_key()?;
        let cipher = Aes256Gcm::new_from_slice(&key)
            .map_err(|_| CryptoError::InvalidKeyLength)?;
        *self.cipher.lock().unwrap() = Some(cipher);
        Ok(())
    }

    /// 是否已初始化
    pub fn is_initialized(&self) -> bool {
        self.cipher.lock().unwrap().is_some()
    }

    /// 加密文本为 ENC(base64) 格式
    pub fn encrypt(&self, plaintext: &str) -> Result<String, CryptoError> {
        let cipher = self.cipher.lock().unwrap();
        let cipher = cipher.as_ref().ok_or(CryptoError::NotInitialized)?;

        if plaintext.is_empty() {
            return Ok(String::new());
        }

        let nonce_bytes: [u8; NONCE_SIZE] = OsRng.gen();
        let nonce = Nonce::from_slice(&nonce_bytes);

        let ciphertext = cipher
            .encrypt(nonce, plaintext.as_bytes())
            .map_err(|e| CryptoError::EncryptFailed(e.to_string()))?;

        // 存储格式: nonce(base64) + ciphertext(base64)
        let mut payload = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
        payload.extend_from_slice(&nonce_bytes);
        payload.extend_from_slice(&ciphertext);
        let encoded = BASE64.encode(&payload);

        Ok(format!("ENC({})", encoded))
    }

    /// 解密 ENC(base64) 格式的数据
    pub fn decrypt(&self, enc_data: &str) -> Result<String, CryptoError> {
        let cipher = self.cipher.lock().unwrap();
        let cipher = cipher.as_ref().ok_or(CryptoError::NotInitialized)?;

        let payload_b64 = enc_data
            .trim()
            .strip_prefix("ENC(")
            .and_then(|s| s.strip_suffix(')'))
            .ok_or(CryptoError::InvalidFormat)?;

        let decoded = BASE64
            .decode(payload_b64)
            .map_err(|_| CryptoError::InvalidFormat)?;

        if decoded.len() < NONCE_SIZE {
            return Err(CryptoError::InvalidFormat);
        }

        let (nonce_bytes, ciphertext) = decoded.split_at(NONCE_SIZE);
        let nonce = Nonce::from_slice(nonce_bytes);

        let plaintext = cipher
            .decrypt(nonce, ciphertext)
            .map_err(|e| CryptoError::DecryptFailed(e.to_string()))?;

        Ok(String::from_utf8(plaintext)
            .map_err(|_| CryptoError::DecryptFailed("UTF-8 解码失败".to_string()))?)
    }

    /// 检查值是否已加密
    pub fn is_encrypted(value: &str) -> bool {
        value.trim().starts_with("ENC(") && value.trim().ends_with(')')
    }

    /// 从文件加载密钥，文件不存在则生成新密钥
    fn load_or_generate_key(&self) -> Result<Vec<u8>, CryptoError> {
        let key_path = self.key_dir.join(KEY_FILENAME);

        if key_path.exists() {
            let key = std::fs::read(&key_path)
                .map_err(|e| CryptoError::KeyFileNotFound(e.to_string()))?;
            if key.len() == KEY_SIZE {
                return Ok(key);
            }
            return Err(CryptoError::InvalidKeyLength);
        }

        // 生成新密钥
        let key_bytes: [u8; KEY_SIZE] = OsRng.gen();
        std::fs::write(&key_path, &key_bytes)
            .map_err(|e| CryptoError::EncryptFailed(format!("写入密钥文件失败: {}", e)))?;

        tracing::info!("已生成新的加密密钥: {:?}", key_path);
        Ok(key_bytes.to_vec())
    }
}
