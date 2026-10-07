//! `serve` 子命令：启动本地 HTTP 服务（OpenAI 兼容 + MCP）。
//!
//! 只做参数桥接，全部实现位于 [`crate::server`]。

use anyhow::Result;
use std::path::Path;

use crate::server::{self, ServeConfig};

/// 处理 `serve` 子命令（阻塞运行，直到进程退出）
pub fn handle(
    host: &str,
    port: u16,
    api_key: Option<String>,
    workers: usize,
    models_dir: &Path,
) -> Result<()> {
    server::run(ServeConfig {
        host: host.to_string(),
        port,
        api_key,
        workers,
        models_dir: models_dir.to_path_buf(),
    })
}
