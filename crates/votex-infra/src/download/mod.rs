//! 下载基础设施：多源镜像解析、断点续传、完整性校验、压缩包提取。
//!
//! 流程：镜像 URL 解析 → `.part` 临时文件 → 大小 + SHA256 校验 → rename。

pub mod archive;
pub mod downloader;
pub mod mirror_resolver;
pub mod verifier;
