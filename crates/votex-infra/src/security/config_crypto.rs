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

    /// 解密 YAML 配置文本中的敏感字段（读取路径，与 [`Self::encrypt_config_file`] 对称）
    ///
    /// 在反序列化 `AppConfig` 之前调用：把 `ENC(...)` 密文还原为明文，
    /// 否则加密落盘的 API key 永远无法被解回使用（加密闭环缺失）。
    ///
    /// - 值不是 `ENC(...)` → 原样保留
    /// - 解密失败（密钥丢失/文件被复制到别的机器）→ 保留密文原值并告警：
    ///   宁可让调用方拿到「用不了」的 key 由服务端明确报错，
    ///   也不能把密文当明文 key 用
    pub fn decrypt_config_content(content: &str, key_dir: Option<PathBuf>) -> String {
        let crypto = ConfigCrypto::new(key_dir);
        if crypto.initialize().is_err() {
            return content.to_string();
        }

        let mut yaml: serde_yml::Value = match serde_yml::from_str(content) {
            Ok(v) => v,
            // YAML 语法错误交给上层按既有契约报错，这里不处理
            Err(_) => return content.to_string(),
        };

        let mut changed = false;
        for field_path in SENSITIVE_FIELDS {
            if Self::decrypt_yaml_path(&mut yaml, field_path, &crypto.engine) {
                changed = true;
            }
        }
        if !changed {
            return content.to_string();
        }

        serde_yml::to_string(&yaml).unwrap_or_else(|_| content.to_string())
    }

    fn decrypt_yaml_path(value: &mut serde_yml::Value, path: &str, engine: &CryptoEngine) -> bool {
        let parts: Vec<&str> = path.split('.').collect();
        if parts.is_empty() {
            return false;
        }
        Self::decrypt_yaml_recursive(value, &parts, engine)
    }

    fn decrypt_yaml_recursive(value: &mut serde_yml::Value, parts: &[&str], engine: &CryptoEngine) -> bool {
        if parts.is_empty() || !value.is_mapping() {
            return false;
        }
        let key = serde_yml::Value::String(parts[0].to_string());
        if parts.len() == 1 {
            if let Some(val) = value.get_mut(&key) {
                if let Some(s) = val.as_str() {
                    if CryptoEngine::is_encrypted(s) {
                        match engine.decrypt(s) {
                            Ok(plain) => {
                                *val = serde_yml::Value::String(plain);
                                return true;
                            }
                            Err(e) => {
                                tracing::warn!("配置敏感字段解密失败，保留密文原值: {}", e);
                            }
                        }
                    }
                }
            }
            false
        } else if let Some(child) = value.get_mut(&key) {
            Self::decrypt_yaml_recursive(child, &parts[1..], engine)
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 临时密钥目录下的加密往返：
    /// 明文 → encrypt_config_file 落盘密文 → decrypt_config_content 还原明文
    #[test]
    fn 配置敏感字段加密解密闭环() {
        let dir = tempfile::tempdir().unwrap();
        let key_dir = dir.path().join("keys");
        let config_path = dir.path().join("application.yml");

        // 1. 初始化密钥（密钥目录需已存在，CryptoEngine 不代建父目录）
        let key_dir = dir.path().join("keys");
        std::fs::create_dir_all(&key_dir).unwrap();
        let crypto = ConfigCrypto::new(Some(key_dir.clone()));
        crypto.initialize().expect("密钥初始化失败");

        // 2. 明文配置
        std::fs::write(
            &config_path,
            "llm:\n  api_key: sk-test-123456\n  model: deepseek-chat\n",
        )
        .unwrap();

        // 3. 加密落盘
        crypto.encrypt_config_file(&config_path).expect("加密失败");
        let encrypted = std::fs::read_to_string(&config_path).unwrap();
        assert!(
            encrypted.contains("ENC("),
            "落盘内容应包含 ENC( 密文前缀: {}",
            encrypted
        );
        assert!(!encrypted.contains("sk-test-123456"), "明文不得出现在落盘内容中");

        // 4. 读取路径解密：还原为明文才能被真正使用
        let content = std::fs::read_to_string(&config_path).unwrap();
        let decrypted = ConfigCrypto::decrypt_config_content(&content, Some(key_dir));
        assert!(
            decrypted.contains("sk-test-123456"),
            "解密后应还原明文 key，实际: {}",
            decrypted
        );

        // 5. YAML Value 级验证：解密产物可直接反序列化且无密文残留
        let mut yaml: serde_yml::Value = serde_yml::from_str(&decrypted).unwrap();
        let key_val = yaml["llm"]["api_key"].as_str().unwrap();
        assert_eq!(key_val, "sk-test-123456", "解密后的 key 应为明文原值");
        assert!(
            !decrypted.contains("ENC("),
            "解密产物不得残留密文前缀: {}",
            decrypted
        );
    }

    /// 无密钥/密文损坏时解密必须保留原值，不得 panic 或产出乱码
    #[test]
    fn 解密失败保留原值() {
        let out = ConfigCrypto::decrypt_config_content(
            "llm:\n  api_key: \"ENC(not-valid-ciphertext)\"\n",
            Some(std::path::PathBuf::from("/nonexistent-keys-dir-for-test")),
        );
        // 密钥不可用 → decrypt_config_content 原样返回，密文保留
        assert!(out.contains("ENC("), "解密失败时必须保留密文原值: {}", out);
    }
}
