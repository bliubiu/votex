//! HTTP 路由与 OpenAI 兼容端点

use anyhow::Result;
use serde::Deserialize;
use serde_json::{json, Value};
use tiny_http::{Method, Request};

use votex_app::use_case::asr_use_case::AsrUseCase;
use votex_app::use_case::tts_use_case::TtsUseCase;
use votex_domain::tts::value_object::{parse_tts_engine, AudioFormat, TTS_ENGINE_HINT};

use super::{
    build_response, content_type, error_response, json_response, read_body, respond, temp_path,
    ServerContext,
};

/// 路由入口
pub fn handle(ctx: &ServerContext, request: Request) -> Result<()> {
    let method = request.method().clone();
    let url = request.url().to_string();
    let (path, query) = split_url(&url);

    // CORS 预检：无需鉴权，直接 204
    if method == Method::Options {
        return respond(request, build_response(204, Vec::new(), "text/plain"));
    }

    // 鉴权：/health 放行以便探活
    if path != "/health" && !is_authorized(ctx, &request) {
        return respond(
            request,
            error_response(401, "未授权：缺少或错误的 API Key（Authorization: Bearer <key>）"),
        );
    }

    match (method, path) {
        (Method::Get, "/health") => respond(
            request,
            json_response(
                200,
                &json!({ "status": "ok", "name": "votex", "version": env!("CARGO_PKG_VERSION") }),
            ),
        ),
        (Method::Get, "/v1/capabilities") => {
            let report = ctx.capability_report();
            match serde_json::to_value(&report) {
                Ok(v) => respond(request, json_response(200, &v)),
                Err(e) => respond(
                    request,
                    error_response(500, &format!("序列化能力报告失败: {e}")),
                ),
            }
        }
        (Method::Get, "/v1/models") => respond(request, json_response(200, &models_payload(ctx))),
        (Method::Post, "/v1/audio/speech") => handle_speech(request),
        (Method::Post, "/v1/audio/transcriptions") => handle_transcription(request, &query),
        (Method::Post, "/mcp") => super::mcp::handle(ctx, request),
        _ => respond(request, error_response(404, &format!("未知端点: {path}"))),
    }
}

// ============================================================
// 端点实现
// ============================================================

/// OpenAI 兼容：`GET /v1/models`
fn models_payload(ctx: &ServerContext) -> Value {
    let report = ctx.capability_report();
    let data: Vec<Value> = report
        .engines
        .iter()
        .filter(|e| e.category == "tts" || e.category == "asr")
        .map(|e| {
            json!({
                "id": e.engine,
                "object": "model",
                "created": 0,
                "owned_by": "votex",
                "votex": {
                    "display_name": e.display_name,
                    "category": e.category,
                    "local": e.local,
                    "sample_rate": e.sample_rate,
                }
            })
        })
        .collect();
    json!({ "object": "list", "data": data })
}

/// 语音合成请求（兼容 OpenAI `/v1/audio/speech`，并接受 votex `engine` 别名）
#[derive(Debug, Deserialize)]
struct SpeechRequest {
    input: String,
    #[serde(default)]
    voice: Option<String>,
    /// OpenAI 字段：模型/引擎标识
    #[serde(default)]
    model: Option<String>,
    /// votex 原生字段别名
    #[serde(default)]
    engine: Option<String>,
    #[serde(default)]
    speed: Option<f32>,
    #[serde(default)]
    lang: Option<String>,
    #[serde(default)]
    response_format: Option<String>,
}

/// OpenAI 兼容：`POST /v1/audio/speech`
fn handle_speech(mut request: Request) -> Result<()> {
    let body = read_body(&mut request)?;
    let req: SpeechRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return respond(
                request,
                error_response(400, &format!("请求体 JSON 解析失败: {e}")),
            )
        }
    };
    if req.input.trim().is_empty() {
        return respond(request, error_response(400, "input 不能为空"));
    }

    let engine_str = req.engine.or(req.model).unwrap_or_else(|| "kokoro".into());
    let engine = match parse_tts_engine(&engine_str) {
        Some(e) => e,
        None => {
            return respond(
                request,
                error_response(
                    400,
                    &format!("不支持的 TTS 引擎: {engine_str}，可选: {TTS_ENGINE_HINT}"),
                ),
            )
        }
    };
    let voice = req.voice.unwrap_or_else(|| "zf_001".into());
    let speed = req.speed.unwrap_or(1.0);
    let format = parse_audio_format(req.response_format.as_deref());
    let out = temp_path(format.extension());

    let result = TtsUseCase::new().synthesize(
        &req.input,
        &out,
        engine,
        &voice,
        speed,
        format,
        req.lang.as_deref(),
        None,
        None,
    );

    match result {
        Ok(()) => match std::fs::read(&out) {
            Ok(bytes) => {
                let _ = std::fs::remove_file(&out);
                respond(request, build_response(200, bytes, content_type_for(format)))
            }
            Err(e) => {
                let _ = std::fs::remove_file(&out);
                respond(request, error_response(500, &format!("读取合成音频失败: {e}")))
            }
        },
        Err(e) => {
            let _ = std::fs::remove_file(&out);
            respond(request, error_response(500, &format!("合成失败: {e:#}")))
        }
    }
}

/// 语音识别：`POST /v1/audio/transcriptions`
///
/// 批一批采用「原始音频体 + 查询参数」，避免引入 multipart 解析：
/// `?model=whisper-base&language=zh&format=srt&ext=wav`。
/// 返回 OpenAI 风格的 `{"text": ...}`，并附带 `subtitles` 数组。
fn handle_transcription(mut request: Request, query: &[(String, String)]) -> Result<()> {
    let ct = content_type(&request);
    let model = query_get(query, "model").unwrap_or("whisper-base").to_string();
    let lang = query_get(query, "language")
        .or_else(|| query_get(query, "lang"))
        .unwrap_or("zh")
        .to_string();
    let format = query_get(query, "format").unwrap_or("srt").to_string();

    let body = read_body(&mut request)?;
    if body.is_empty() {
        return respond(
            request,
            error_response(
                400,
                "音频请求体为空：请以 raw body 上传音频，并用 ?model=&language=&format= 指定参数",
            ),
        );
    }

    let ext = query_get(query, "ext")
        .map(str::to_string)
        .unwrap_or_else(|| guess_audio_ext(&ct));
    let in_path = temp_path(&ext);
    let out_path = temp_path("srt");

    if let Err(e) = std::fs::write(&in_path, &body) {
        return respond(
            request,
            error_response(500, &format!("写入临时音频失败: {e}")),
        );
    }

    let result = AsrUseCase::new().recognize(&in_path, &out_path, &model, &lang, &format, None);
    let _ = std::fs::remove_file(&in_path);
    let _ = std::fs::remove_file(&out_path);

    match result {
        Ok(r) => respond(
            request,
            json_response(200, &json!({ "text": r.text, "subtitles": r.subtitles })),
        ),
        Err(e) => respond(request, error_response(500, &format!("识别失败: {e:#}"))),
    }
}

// ============================================================
// 辅助
// ============================================================

fn is_authorized(ctx: &ServerContext, request: &Request) -> bool {
    let Some(expected) = ctx.api_key.as_deref() else {
        return true;
    };
    request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Authorization"))
        .map(|h| {
            h.value
                .as_str()
                .trim()
                .strip_prefix("Bearer ")
                .map(|t| t.trim() == expected)
                .unwrap_or(false)
        })
        .unwrap_or(false)
}

fn parse_audio_format(fmt: Option<&str>) -> AudioFormat {
    match fmt.map(|s| s.to_ascii_lowercase()).as_deref() {
        Some("mp3") => AudioFormat::Mp3,
        Some("m4a") => AudioFormat::M4A,
        Some("flac") => AudioFormat::Flac,
        Some("m4b") => AudioFormat::M4B,
        _ => AudioFormat::Wav,
    }
}

fn content_type_for(format: AudioFormat) -> &'static str {
    match format {
        AudioFormat::Wav => "audio/wav",
        AudioFormat::Mp3 => "audio/mpeg",
        AudioFormat::M4A | AudioFormat::M4B => "audio/mp4",
        AudioFormat::Flac => "audio/flac",
    }
}

/// 依据 Content-Type 猜测音频扩展名（供 ffmpeg/解码器识别）
fn guess_audio_ext(content_type: &str) -> String {
    let ct = content_type.to_ascii_lowercase();
    if ct.contains("mpeg") || ct.contains("mp3") {
        "mp3".to_string()
    } else if ct.contains("mp4") || ct.contains("m4a") || ct.contains("aac") {
        "m4a".to_string()
    } else if ct.contains("flac") {
        "flac".to_string()
    } else {
        "wav".to_string()
    }
}

fn split_url(url: &str) -> (&str, Vec<(String, String)>) {
    match url.split_once('?') {
        Some((path, q)) => (path, parse_query(q)),
        None => (url, Vec::new()),
    }
}

fn parse_query(q: &str) -> Vec<(String, String)> {
    q.split('&')
        .filter(|s| !s.is_empty())
        .map(|pair| {
            let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
            (url_decode(k), url_decode(v))
        })
        .collect()
}

fn query_get<'a>(query: &'a [(String, String)], key: &str) -> Option<&'a str> {
    query
        .iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
}

fn url_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => {
                match (hex_val(bytes[i + 1]), hex_val(bytes[i + 2])) {
                    (Some(h), Some(l)) => {
                        out.push(h * 16 + l);
                        i += 3;
                    }
                    _ => {
                        out.push(b'%');
                        i += 1;
                    }
                }
            }
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex_val(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 查询串解析与URL解码() {
        let q = parse_query("model=whisper-base&language=zh&name=%E4%B8%AD%E6%96%87+a");
        assert_eq!(query_get(&q, "model"), Some("whisper-base"));
        assert_eq!(query_get(&q, "language"), Some("zh"));
        assert_eq!(query_get(&q, "name"), Some("中文 a"));
        assert_eq!(query_get(&q, "missing"), None);
    }

    #[test]
    fn 音频格式解析() {
        assert_eq!(parse_audio_format(Some("MP3")), AudioFormat::Mp3);
        assert_eq!(parse_audio_format(Some("flac")), AudioFormat::Flac);
        assert_eq!(parse_audio_format(None), AudioFormat::Wav);
        assert_eq!(parse_audio_format(Some("unknown")), AudioFormat::Wav);
    }

    #[test]
    fn 内容类型猜测扩展名() {
        assert_eq!(guess_audio_ext("audio/mpeg"), "mp3");
        assert_eq!(guess_audio_ext("audio/mp4"), "m4a");
        assert_eq!(guess_audio_ext("audio/flac"), "flac");
        assert_eq!(guess_audio_ext("audio/wav"), "wav");
        assert_eq!(guess_audio_ext(""), "wav");
    }
}
