//! MCP（Model Context Protocol）JSON-RPC 端点
//!
//! 以「Streamable HTTP」的最简形态承载：`POST /mcp` 接收 JSON-RPC 2.0 请求，
//! 返回 JSON-RPC 响应。覆盖 Agent 常用交互：
//!
//! - `initialize` / `ping`
//! - `tools/list`
//! - `tools/call`（`list_engines` / `list_models` / `tts_synthesize` / `asr_transcribe`）
//!
//! 工具全部复用 `votex-app` 用例，内容仍在本机处理。

use std::path::Path;

use anyhow::Result;
use serde_json::{json, Value};
use tiny_http::Request;

use votex_app::use_case::asr_use_case::AsrUseCase;
use votex_app::use_case::tts_use_case::TtsUseCase;
use votex_domain::tts::value_object::{parse_tts_engine, AudioFormat, TTS_ENGINE_HINT};

use super::{build_response, json_response, read_body, respond, temp_path, ServerContext};

/// 处理 MCP 请求
pub(super) fn handle(ctx: &ServerContext, mut request: Request) -> Result<()> {
    let body = read_body(&mut request)?;
    let req: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            let resp = json!({
                "jsonrpc": "2.0",
                "id": Value::Null,
                "error": { "code": -32700, "message": format!("JSON 解析失败: {e}") }
            });
            return respond(request, json_response(200, &resp));
        }
    };

    let id = req.get("id").cloned();
    let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");

    // 通知（无 id）：不返回结果体
    let Some(id) = id else {
        return respond(request, build_response(202, Vec::new(), "application/json"));
    };

    let result: std::result::Result<Value, (i64, String)> = match method {
        "initialize" => Ok(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "votex", "version": env!("CARGO_PKG_VERSION") }
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tool_definitions() })),
        "tools/call" => tools_call(ctx, req.get("params").cloned().unwrap_or(Value::Null)),
        _ => Err((-32601, format!("未知方法: {method}"))),
    };

    let resp = match result {
        Ok(v) => json!({ "jsonrpc": "2.0", "id": id, "result": v }),
        Err((code, message)) => json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": code, "message": message }
        }),
    };
    respond(request, json_response(200, &resp))
}

/// 工具定义（`tools/list`）
fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "list_engines",
            "description": "列出 votex 全部引擎能力（分类 / 本地或在线 / 采样率 / 方言 / 零样本克隆）",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        }),
        json!({
            "name": "list_models",
            "description": "列出模型清单与本地可用性（是否已下载就绪）",
            "inputSchema": { "type": "object", "properties": {}, "additionalProperties": false }
        }),
        json!({
            "name": "tts_synthesize",
            "description": "文本转语音，写入音频文件并返回路径（本地推理，内容不上传网络）",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "text": { "type": "string", "description": "待合成文本" },
                    "engine": { "type": "string", "description": "TTS 引擎：kokoro / indextts25 / qwen3 / cosyvoice3，默认 kokoro" },
                    "voice": { "type": "string", "description": "音色 ID，默认 zf_001" },
                    "speed": { "type": "number", "description": "语速 0.5~2.0，默认 1.0" },
                    "lang": { "type": "string", "description": "语言 zh / en" },
                    "format": { "type": "string", "enum": ["wav", "mp3", "m4a", "flac"], "description": "输出格式，默认 wav" },
                    "output": { "type": "string", "description": "输出文件路径；省略则写入系统临时目录" }
                },
                "required": ["text"]
            }
        }),
        json!({
            "name": "asr_transcribe",
            "description": "音频转文字，返回识别文本与字幕条目",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "input": { "type": "string", "description": "音频文件路径（wav/mp3/m4a/flac）" },
                    "model": { "type": "string", "description": "ASR 模型，默认 whisper-base" },
                    "language": { "type": "string", "description": "语言 zh / en / zhen，默认 zh" },
                    "format": { "type": "string", "enum": ["txt", "srt", "lrc"], "description": "字幕格式，默认 srt" }
                },
                "required": ["input"]
            }
        }),
    ]
}

/// 工具调用（`tools/call`）
fn tools_call(ctx: &ServerContext, params: Value) -> std::result::Result<Value, (i64, String)> {
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "缺少 params.name".to_string()))?;
    let args = params.get("arguments").cloned().unwrap_or_else(|| json!({}));

    match name {
        "list_engines" => {
            let report = ctx.capability_report();
            Ok(text_result(
                &serde_json::to_string_pretty(&report.engines).unwrap_or_default(),
            ))
        }
        "list_models" => {
            let report = ctx.capability_report();
            Ok(text_result(
                &serde_json::to_string_pretty(&report.models).unwrap_or_default(),
            ))
        }
        "tts_synthesize" => Ok(tts_synthesize(&args)),
        "asr_transcribe" => Ok(asr_transcribe(&args)),
        other => Err((-32602, format!("未知工具: {other}"))),
    }
}

fn tts_synthesize(args: &Value) -> Value {
    let text = args.get("text").and_then(|v| v.as_str()).unwrap_or("");
    if text.trim().is_empty() {
        return text_error("缺少 text 参数");
    }
    let engine_str = args
        .get("engine")
        .and_then(|v| v.as_str())
        .unwrap_or("kokoro");
    let Some(engine) = parse_tts_engine(engine_str) else {
        return text_error(&format!(
            "不支持的 TTS 引擎: {engine_str}，可选: {TTS_ENGINE_HINT}"
        ));
    };
    let voice = args
        .get("voice")
        .and_then(|v| v.as_str())
        .unwrap_or("zf_001");
    let speed = args.get("speed").and_then(|v| v.as_f64()).unwrap_or(1.0) as f32;
    let lang = args.get("lang").and_then(|v| v.as_str());
    let format = match args.get("format").and_then(|v| v.as_str()) {
        Some("mp3") => AudioFormat::Mp3,
        Some("m4a") => AudioFormat::M4A,
        Some("flac") => AudioFormat::Flac,
        _ => AudioFormat::Wav,
    };
    let output = args
        .get("output")
        .and_then(|v| v.as_str())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| temp_path(format.extension()));

    match TtsUseCase::new().synthesize(
        text,
        &output,
        engine,
        voice,
        speed,
        format,
        lang,
        None,
        None,
    ) {
        Ok(()) => text_result(&format!(
            "已合成音频: {}（引擎 {}，音色 {voice}，格式 {}）",
            output.display(),
            engine.display_name(),
            format.extension()
        )),
        Err(e) => text_error(&format!("合成失败: {e:#}")),
    }
}

fn asr_transcribe(args: &Value) -> Value {
    let input = args.get("input").and_then(|v| v.as_str()).unwrap_or("");
    if input.is_empty() {
        return text_error("缺少 input 参数");
    }
    if !Path::new(input).exists() {
        return text_error(&format!("音频文件不存在: {input}"));
    }
    let model = args
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("whisper-base");
    let language = args
        .get("language")
        .and_then(|v| v.as_str())
        .unwrap_or("zh");
    let format = args
        .get("format")
        .and_then(|v| v.as_str())
        .unwrap_or("srt");
    let out = temp_path("srt");

    let result = AsrUseCase::new().recognize(Path::new(input), &out, model, language, format, None);
    let _ = std::fs::remove_file(&out);
    match result {
        Ok(r) => text_result(
            &json!({ "text": r.text, "subtitles": r.subtitles }).to_string(),
        ),
        Err(e) => text_error(&format!("识别失败: {e:#}")),
    }
}

fn text_result(text: &str) -> Value {
    json!({
        "content": [ { "type": "text", "text": text } ],
        "isError": false
    })
}

fn text_error(text: &str) -> Value {
    json!({
        "content": [ { "type": "text", "text": text } ],
        "isError": true
    })
}
