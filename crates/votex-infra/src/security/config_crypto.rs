//! 配置加密工具
//!
//! 序列化 application.yml 敏感字段自动加密

use std::path::{Path, PathBuf};

use votex_domain::error::CryptoError;

use super::crypto::CryptoEngine;

/// 需要加密的配置字段路径（YAML 点路径）
const SENSITIVE_FIELDS: &[&str] = &[
    "tts.api_key",
    "asr.api_key",
    "llm.api_key",
    "llm.api_secret",
    "api.azure.api_key",
    "api.aliyun.access_key",
    "api.aliyun.secret_key",
    "api.deepseek.api_key",
];

pub struct ConfigCrypto {
    engine: CryptoEngine,
}

impl ConfigCrypto {
    pub fn new(key_dir: Option<PathBuf>) -> Self {
        Self {
            engine: CryptoEngine::new(key_dir),
        }
    }

    /// 初始化（加载/生成密钥）
    pub fn initialize(&self) -> Result<(), CryptoError> {
        self.engine.initialize()
    }

    /// 加密配置值
    pub fn encrypt_value(&self, value: &str) -> Result<String, CryptoError> {
        if CryptoEngine::is_encrypted(value) {
            return Ok(value.to_string());
        }
        self.engine.encrypt(value)
    }

    /// 解密配置值
    pub fn decrypt_value(&self, value: &str) -> Result<String, CryptoError> {
        if CryptoEngine::is_encrypted(value) {
            return self.engine.decrypt(value);
        }
        Ok(value.to_string())
    }

    /// 加密配置文件中的敏感字段
    pub fn encrypt_config_file(&self, config_path: &Path) -> Result<(), CryptoError> {
        if !config_path.exists() {
            return Ok(());
        }

        let content = std::fs::read_to_string(config_path)
            .map_err(|e| CryptoError::DecryptFailed(format!("读取配置文件失败: {}", e)))?;
        let mut yaml: serde_yml::Value = serde_yml::from_str(&content)
            .map_err(|e| CryptoError::DecryptFailed(format!("解析 YAML 失败: {}", e)))?;

        let mut changed = false;
        for field_path in SENSITIVE_FIELDS {
            if Self::encrypt_yaml_path(&mut yaml, field_path, &self.engine) {
                changed = true;
            }
        }

        if changed {
            let out = serde_yml::to_string(&yaml)
                .map_err(|e| CryptoError::DecryptFailed(format!("序列化 YAML 失败: {}", e)))?;
            std::fs::write(config_path, out)
                .map_err(|e| CryptoError::DecryptFailed(format!("写入配置文件失败: {}", e)))?;
        }

        Ok(())
    }

    /// 递归设置 YAML 路径并加密值
    fn encrypt_yaml_path(value: &mut serde_yml::Value, path: &str, engine: &CryptoEngine) -> bool {
        let parts: Vec<&str> = path.split('.').collect();
        if parts.is_empty() {
            return false;
        }
        Self::encrypt_yaml_recursive(value, &parts, engine)
    }

    fn encrypt_yaml_recursive(value: &mut serde_yml::Value, parts: &[&str], engine: &CryptoEngine) -> bool {
        if parts.is_empty() || !value.is_mapping() {
            return false;
        }
        let key = serde_yml::Value::String(parts[0].to_string());
        if parts.len() == 1 {
            if let Some(val) = value.get_mut(&key) {
                if let Some(s) = val.as_str() {
                    if !s.is_empty() && !CryptoEngine::is_encrypted(s) {
                        if let Ok(enc) = engine.encrypt(s) {
                            *val = serde_yml::Value::String(enc);
                            return true;
                        }
                    }
                }
            }
            false
        } else if let Some(child) = value.get_mut(&key) {
            Self::encrypt_yaml_recursive(child, &parts[1..], engine)
        } else {
            false
        }
    }
}
