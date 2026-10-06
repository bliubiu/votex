//! 安全基础设施：AES-256-GCM 加密与配置密文管理。
//!
//! 主密钥存工作区根 `.key`（Unix 权限 0o600），路径**不依赖进程当前目录**。
//! 详见 `docs/00-安全设计.md`。

pub mod crypto;
pub mod config_crypto;
