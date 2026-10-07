//! 本地 HTTP 服务（`votex serve`）
//!
//! 复用现有 `votex-app` 用例与领域 trait，**不新增业务逻辑**：
//! 服务层只做「HTTP/MCP 协议 ↔ 用例」的适配。
//!
//! # 端点
//!
//! | 方法 | 路径 | 说明 |
//! | :--- | :--- | :--- |
//! | GET | `/health` | 存活探测 |
//! | GET | `/v1/capabilities` | 能力报告（模型可用性 + 引擎能力 + 音色池） |
//! | GET | `/v1/models` | OpenAI 兼容模型列表 |
//! | POST | `/v1/audio/speech` | OpenAI 兼容语音合成 |
//! | POST | `/v1/audio/transcriptions` | 语音识别（原始音频体 + 查询参数） |
//! | POST | `/mcp` | MCP JSON-RPC（initialize / tools/list / tools/call） |
//!
//! # 依赖选择
//!
//! 使用 `tiny_http`（阻塞 + 线程池），不引入异步运行时：
//! 契合 AGENTS.md「单二进制分发、体积优先、离线运行」约束。
//!
//! # 安全
//!
//! 默认仅监听 `127.0.0.1`；`--api-key` 设置后 `/v1` 与 `/mcp` 均需
//! `Authorization: Bearer <key>`。用户文本/音频仍在本机处理，
//! 服务化不改变「内容零上传」的默认行为。

mod handlers;
mod mcp;

use std::io::Cursor;
use std::net::ToSocketAddrs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde_json::{json, Value};
use tiny_http::{Header, Request, Response};
use tiny_http::Server;
use votex_app::use_case::capability_use_case::{CapabilityReport, CapabilityUseCase};
use votex_domain::model::registry::ModelRegistryEntry;

/// 服务运行参数
#[derive(Debug, Clone)]
pub struct ServeConfig {
    pub host: String,
    pub port: u16,
    pub api_key: Option<String>,
    pub workers: usize,
    pub models_dir: PathBuf,
}

/// 服务共享上下文（线程间只读）
pub struct ServerContext {
    pub models_dir: PathBuf,
    /// 模型清单（启动时加载一次，避免每次请求读盘）
    pub registry: Vec<ModelRegistryEntry>,
    pub api_key: Option<String>,
}

impl ServerContext {
    /// 构建能力报告（供 `/v1/capabilities` 与 MCP 工具复用）
    pub fn capability_report(&self) -> CapabilityReport {
        CapabilityUseCase::build(&self.models_dir, self.registry.clone())
    }
}

/// 启动服务并阻塞运行
pub fn run(cfg: ServeConfig) -> Result<()> {
    let addr = format!("{}:{}", cfg.host, cfg.port);
    // 提前校验地址，给出明确错误而非 panic
    let _ = addr
        .to_socket_addrs()
        .with_context(|| format!("无法解析监听地址: {}", addr))?;

    let server = Arc::new(
        Server::http(&addr).map_err(|e| anyhow::anyhow!("无法监听 {}: {}", addr, e))?,
    );

    let registry = votex_app::platform::registry::load_registry_entries();
    let ctx = Arc::new(ServerContext {
        models_dir: cfg.models_dir.clone(),
        registry,
        api_key: cfg.api_key.clone(),
    });

    println!("声阅（votex）本地服务已启动: http://{addr}");
    println!("  OpenAI 兼容: POST /v1/audio/speech | POST /v1/audio/transcriptions | GET /v1/models");
    println!("  能力发现:   GET /health | GET /v1/capabilities");
    println!("  MCP:        POST /mcp");
    if cfg.api_key.is_some() {
        println!("  鉴权:       已启用 Bearer API Key");
    }
    tracing::info!("本地服务监听 {}", addr);

    let workers = cfg.workers.max(1);
    let mut handles = Vec::with_capacity(workers);
    for _ in 0..workers {
        let server = Arc::clone(&server);
        let ctx = Arc::clone(&ctx);
        handles.push(std::thread::spawn(move || loop {
            match server.recv() {
                Ok(request) => {
                    if let Err(e) = handlers::handle(&ctx, request) {
                        // 单个请求失败不应终止服务
                        tracing::warn!("请求处理失败: {e:#}");
                    }
                }
                Err(e) => {
                    tracing::error!("接收请求失败，工作线程退出: {e}");
                    break;
                }
            }
        }));
    }

    for h in handles {
        let _ = h.join();
    }
    Ok(())
}

/// 生成唯一的临时文件路径（不依赖 tempfile crate）
pub(crate) fn temp_path(ext: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::SeqCst);
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!(
        "votex-serve-{}-{}-{}.{}",
        std::process::id(),
        ts,
        n,
        ext
    ))
}

// ============================================================
// 响应辅助（handlers / mcp 共用）
// ============================================================

/// 统一响应类型（始终为字节流，避免泛型分叉）
pub(crate) type HttpResponse = Response<Cursor<Vec<u8>>>;

/// 构造带 CORS 头的响应
pub(crate) fn build_response(status: u16, body: Vec<u8>, content_type: &str) -> HttpResponse {
    let mut resp = Response::from_data(body).with_status_code(status);
    for (k, v) in [
        ("Content-Type", content_type),
        ("Access-Control-Allow-Origin", "*"),
        ("Access-Control-Allow-Headers", "Content-Type, Authorization"),
        ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
    ] {
        if let Ok(header) = Header::from_bytes(k.as_bytes(), v.as_bytes()) {
            resp = resp.with_header(header);
        }
    }
    resp
}

/// JSON 响应
pub(crate) fn json_response(status: u16, value: &Value) -> HttpResponse {
    let body = serde_json::to_vec(value).unwrap_or_else(|_| b"{}".to_vec());
    build_response(status, body, "application/json; charset=utf-8")
}

/// 错误响应（OpenAI 风格 `{"error":{"message":...}}`）
pub(crate) fn error_response(status: u16, message: &str) -> HttpResponse {
    json_response(status, &json!({ "error": { "message": message } }))
}

/// 提交响应
pub(crate) fn respond(request: Request, resp: HttpResponse) -> Result<()> {
    request
        .respond(resp)
        .map_err(|e| anyhow::anyhow!("写响应失败: {e}"))
}

/// 读取完整请求体
pub(crate) fn read_body(request: &mut Request) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    request
        .as_reader()
        .read_to_end(&mut buf)
        .map_err(|e| anyhow::anyhow!("读取请求体失败: {e}"))?;
    Ok(buf)
}

/// 请求 Content-Type（取不到返回空串）
pub(crate) fn content_type(request: &Request) -> String {
    request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Content-Type"))
        .map(|h| h.value.as_str().to_string())
        .unwrap_or_default()
}
