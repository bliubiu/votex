//! 基础设施层
//!
//! 提供领域层所需的全部技术实现：ONNX Runtime 推理、SQLite 持久化、
//! 模型下载、音频处理、日志脱敏、加密、编码转换等。
//!
//! # 分层规则
//!
//! - 依赖 `votex-domain`，**不得**依赖 `votex-app` / `votex-cli` / `votex-gui`
//! - 所有 IO 都收敛在本层，领域层与应用层不直接触碰文件 / 网络 / 数据库
//!
//! # 跨模块约定
//!
//! | 约定 | 说明 |
//! | :--- | :--- |
//! | 路径解析 | 一律走 `shared::WorkspacePaths`，**禁止**自行 `current_dir()` |
//! | ONNX Session | 一律走 `shared::OrtSessionFactory`，便于统一执行提供器与超时 |
//! | 模型查找 | 一律走 `shared::ModelFileLocator` |
//! | 采样 | 一律走 `shared::sampling`，引擎内不应自带 softmax / top-k |
//! | 资源限流 | 推理前经 `shared::InferenceGate`，防止 OOM |
//! | 锁 | 锁中毒用 `unwrap_or_else(|e| e.into_inner())` 恢复，**不**用 `unwrap()` |
//!
//! # 子模块一览
//!
//! | 模块 | 职责 |
//! | :--- | :--- |
//! | `shared` | 跨引擎共享设施（Session 工厂、模型定位、采样、闸门、路径解析） |
//! | `config` | YAML 配置与模型清单加载 |
//! | `persistence` | SQLite 仓储实现 |
//! | `download` | 多源下载、断点续传、SHA256 校验、压缩包提取 |
//! | `tts` / `asr` / `ocr` / `translation` / `llm` | 各引擎适配器 |
//! | `audio` | 编解码、重采样、降噪、VAD |
//! | `tokenizer` / `bpe_tokenizer` | 分词器 |
//! | `subtitle` | SRT / LRC 生成 |
//! | `document` | PDF / 图片预处理 |
//! | `video` | 视频素材合成 |
//! | `logging` | 日志轮转与脱敏 |
//! | `security` | AES-256-GCM 加密 |
//! | `api` | 云服务密钥解析与请求构造 |
//! | `encoding` | UTF-8 / GBK 转换 |

// 测试函数使用中文语义命名（如 `解析_非法输入报错`），
// 有助于表达断言意图；仅在测试编译时豁免 snake_case 检查。
#![cfg_attr(test, allow(non_snake_case))]

pub mod config;
pub mod persistence;
pub mod download;
pub mod encoding;
pub mod event;
pub mod logging;
pub mod inference;
pub mod shared;
pub mod tts;
pub mod asr;
pub mod ocr;
pub mod audio;
pub mod document;
pub mod subtitle;
pub mod api;
pub mod security;
pub mod llm;
pub mod video;
pub mod translation;
pub mod bpe_tokenizer;
pub mod provider;
pub mod pipeline;
pub mod tokenizer;

/// 辅助：将字符串转为 DomainError::Config
pub fn config_err(msg: &str) -> votex_domain::error::ConfigError {
    votex_domain::error::ConfigError::ReadFailed(msg.to_string())
}
