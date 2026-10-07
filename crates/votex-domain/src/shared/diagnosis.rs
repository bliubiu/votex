//! 失败诊断与脱敏
//!
//! 把 use_case 冒泡上来的 anyhow 错误链（可能含模型绝对路径、临时文件路径、
//! API Key 碎片）转换为「中文通用消息 + 原因解释 + 已脱敏详情 + 引擎文档页」。
//!
//! 设计对齐 docs/27 第四节 B（借鉴 VoiceStudio：探测失败信息**故意不原样展示**，
//! 只给通用消息 + "Why?" 面板）与 docs/07 的 GUI 提示文案规范。
//!
//! 本模块零 IO、纯字符串处理：GUI（egui 结果面板）与 CLI 最终错误行均可复用。

/// 一条失败的诊断结果
#[derive(Debug, Clone, PartialEq)]
pub struct Diagnosis {
    /// 一句话中文结论（结果面板着色显示）
    pub summary: String,
    /// 「为什么？」折叠区：原因 + 修复建议；无则为 `None`
    pub explain: Option<String>,
    /// 「详情」折叠区：脱敏后的原始错误链
    pub detail: String,
    /// 对应引擎文档页（`docs/engines/` 下的文件名）
    pub doc_page: Option<&'static str>,
}

/// 诊断入口：模式匹配原始错误链，产出用户可读结论
pub fn diagnose(raw: &str) -> Diagnosis {
    let detail = sanitize(raw);
    let lower = raw.to_lowercase();

    // 引擎文档页优先由引擎标识推断（错误链里常带引擎名/模型名）
    let doc_from_engine = detect_doc_page(&lower);

    // ---- 取消不算失败，直接透传 ----
    if lower.contains("已取消") || lower.contains("cancelled") {
        return Diagnosis {
            summary: "任务已取消".into(),
            explain: None,
            detail,
            doc_page: None,
        };
    }

    // ---- 模型包与运行时不兼容（FunASR vocab_size 教训，见 docs/engines/asr-paraformer.md）----
    if lower.contains("vocab_size") {
        return Diagnosis {
            summary: "模型包与推理运行时不兼容".into(),
            explain: Some(
                "该模型包缺少推理运行时必需的元数据（如 FunASR 自导出模型缺 vocab_size），\
                 直接加载会拖垮整个进程，加载前已被拒绝。\
                 请在「模型」页删除该模型后重新下载（registry 已固定使用官方转换包）。"
                    .into(),
            ),
            detail,
            doc_page: Some("asr-paraformer.md"),
        };
    }

    // ---- 模型未下载 / 文件缺失（VM001/VM005）----
    if 模型缺失(&lower) {
        return Diagnosis {
            summary: "模型未下载或文件缺失".into(),
            explain: Some(
                "请在「模型」页下载对应模型；CLI 可执行 `votex model download <模型名>`。\
                 下载完成后重试当前任务。"
                    .into(),
            ),
            detail,
            doc_page: doc_from_engine,
        };
    }

    // ---- 模型校验失败（VM003）----
    if lower.contains("校验失败") || lower.contains("sha256") || lower.contains("checksum") {
        return Diagnosis {
            summary: "模型文件校验失败".into(),
            explain: Some(
                "模型文件可能已损坏或版本不匹配。请在「模型」页删除后重新下载。".into(),
            ),
            detail,
            doc_page: doc_from_engine,
        };
    }

    // ---- 模型加载失败（VM004）----
    if lower.contains("加载失败") || lower.contains("加载模型") || lower.contains("failed to load")
        || lower.contains("session creation")
    {
        return Diagnosis {
            summary: "模型加载失败".into(),
            explain: Some("请重试一次；若持续失败，查看 logs/ 目录当日日志定位原因。".into()),
            detail,
            doc_page: doc_from_engine,
        };
    }

    // ---- 音色问题（VT004）----
    if (lower.contains("音色") && (lower.contains("不存在") || lower.contains("未找到") || lower.contains("缺失")))
        || lower.contains("缺少转写")
        || lower.contains("转写文本")
        || lower.contains("voice add")
    {
        return Diagnosis {
            summary: "音色不存在或缺少转写文本".into(),
            explain: Some(
                "克隆引擎要求参考音频配套转写：请在「音色库」页入库（填写转写），\
                 或执行 `votex voice add --transcript ...` 后重试。"
                    .into(),
            ),
            detail,
            doc_page: doc_from_engine.or(Some("tts-cosyvoice3.md")),
        };
    }

    // ---- 参考音频契约（IndexTTS-2.5 上限 15s）----
    if lower.contains("参考音频") || lower.contains("clone_ref") || lower.contains("best_window")
        || lower.contains("全静音")
    {
        return Diagnosis {
            summary: "参考音频不符合要求".into(),
            explain: Some(
                "克隆参考音频上限 15 秒：无转写时自动截取「语音最多」的窗口，\
                 有转写时显式拒绝（截取后无法与转写对应，宁可拒绝不静默出错）；\
                 全静音参考会被直接拒绝。请更换或修剪参考音频后重试。"
                    .into(),
            ),
            detail,
            doc_page: Some("tts-indextts25.md"),
        };
    }

    // ---- 在线服务鉴权 ----
    if lower.contains("401") || lower.contains("unauthorized") || lower.contains("invalid_api_key")
        || (lower.contains("api") && (lower.contains("key") || lower.contains("密钥")))
        || lower.contains("鉴权")
    {
        return Diagnosis {
            summary: "在线服务鉴权失败".into(),
            explain: Some(
                "请检查配置文件中的 API Key 与区域设置。注意：在线引擎会将内容传出本机，\
                 仅在明确需要时选用。"
                    .into(),
            ),
            detail,
            doc_page: doc_from_engine,
        };
    }

    // ---- 网络失败（VM007）----
    if lower.contains("网络") || lower.contains("connection") || lower.contains("dns")
        || lower.contains("connect")
    {
        return Diagnosis {
            summary: "网络连接失败".into(),
            explain: Some("在线功能需要网络；本地推理不受影响，可改用本地引擎完成当前任务。".into()),
            detail,
            doc_page: doc_from_engine,
        };
    }

    // ---- 内存 / 显存不足 ----
    if lower.contains("out of memory") || lower.contains("内存不足") || lower.contains("显存")
        || lower.contains("oom")
    {
        return Diagnosis {
            summary: "内存或显存不足".into(),
            explain: Some(
                "建议切换更小模型档位（如 qwen3-tts-0.6b、whisper-base），\
                 或关闭其他占用内存的程序后重试。"
                    .into(),
            ),
            detail,
            doc_page: doc_from_engine,
        };
    }

    // ---- ffmpeg / 音视频处理 ----
    if lower.contains("ffmpeg") {
        return Diagnosis {
            summary: "音视频处理失败".into(),
            explain: Some(
                "确认 ffmpeg 已安装且版本 ≥ 4.4（配音混音 amix normalize=0 需要 4.4+），\
                 或在配音设置中改用默认替换音轨模式。"
                    .into(),
            ),
            detail,
            doc_page: None,
        };
    }

    // ---- 文件系统 ----
    if lower.contains("磁盘空间") || lower.contains("no space") {
        return Diagnosis {
            summary: "磁盘空间不足".into(),
            explain: Some("请清理磁盘后重试。".into()),
            detail,
            doc_page: None,
        };
    }
    if lower.contains("拒绝访问") || lower.contains("denied") || lower.contains("permission") {
        return Diagnosis {
            summary: "文件访问被拒绝".into(),
            explain: Some("请确认文件未被其他程序占用，且当前用户对该目录有读写权限。".into()),
            detail,
            doc_page: None,
        };
    }
    if lower.contains("不可写") || lower.contains("只读") || lower.contains("read-only") {
        return Diagnosis {
            summary: "输出路径不可写".into(),
            explain: Some("请检查输出目录权限，或换一个可写目录。".into()),
            detail,
            doc_page: None,
        };
    }

    // ---- 输入问题 ----
    if lower.contains("不支持的文件格式") || lower.contains("unsupported") && lower.contains("format") {
        return Diagnosis {
            summary: "文件格式不支持".into(),
            explain: Some("支持的格式见对应引擎文档页；音频类任务支持 mp3/wav/flac/m4a。".into()),
            detail,
            doc_page: doc_from_engine,
        };
    }
    if lower.contains("输入文件不存在") || lower.contains("no such file") || lower.contains("找不到文件")
        || lower.contains("输入文件为空")
    {
        return Diagnosis {
            summary: "输入文件不存在或为空".into(),
            explain: Some("请确认文件路径有效；GUI 中可重新选择文件。".into()),
            detail,
            doc_page: None,
        };
    }

    // ---- 超时 ----
    if lower.contains("超时") || lower.contains("timed out") || lower.contains("timeout") {
        return Diagnosis {
            summary: "操作超时".into(),
            explain: Some("首次加载大模型可能较慢（如 qwen3-tts 需 10~60 秒），请重试。".into()),
            detail,
            doc_page: doc_from_engine,
        };
    }

    // ---- 兜底：不猜原因，折叠详情 ----
    Diagnosis {
        summary: "操作失败，请重试".into(),
        explain: Some(
            "未能自动识别具体原因；展开「详情」查看已脱敏的错误信息，\
             问题持续可在 logs/ 目录查看当日完整日志。"
                .into(),
        ),
        detail,
        doc_page: doc_from_engine,
    }
}

/// 模型缺失的宽松判定（错误链措辞多样，取并集避免漏报）
fn 模型缺失(lower: &str) -> bool {
    let 模型词 = lower.contains("模型") || lower.contains("model");
    let 缺失词 = lower.contains("未下载")
        || lower.contains("未安装")
        || lower.contains("未找到")
        || lower.contains("不存在")
        || lower.contains("文件缺失")
        || lower.contains("no such file");
    模型词 && 缺失词
}

/// 从错误链文本推断引擎文档页
fn detect_doc_page(lower: &str) -> Option<&'static str> {
    const RULES: &[(&str, &str)] = &[
        ("kokoro", "tts-kokoro.md"),
        ("indextts", "tts-indextts25.md"),
        ("cosyvoice", "tts-cosyvoice3.md"),
        ("qwen3-tts", "tts-qwen3.md"),
        ("whisper", "asr-whisper.md"),
        ("sensevoice", "asr-sensevoice.md"),
        ("paraformer", "asr-paraformer.md"),
        ("qwen3-asr", "asr-qwen3-asr.md"),
        ("firered", "asr-firered-asr.md"),
        ("wenet", "asr-wenet.md"),
        ("diarization", "asr-speaker-diarization.md"),
        ("说话人", "asr-speaker-diarization.md"),
        ("paddleocr", "ocr-paddleocr.md"),
        ("easyocr", "ocr-easyocr.md"),
        ("hy-mt", "tr-hy-mt-1.5.md"),
        ("nllb", "tr-nllb-200.md"),
        ("m2m", "tr-m2m-100.md"),
        ("opus-mt", "tr-opus-mt.md"),
        ("opus_mt", "tr-opus-mt.md"),
        ("qwen-mt", "tr-qwen-mt.md"),
        ("azure", "online-azure-speech.md"),
        ("aliyun", "online-aliyun.md"),
        ("阿里云", "online-aliyun.md"),
    ];
    RULES
        .iter()
        .find(|(k, _)| lower.contains(k))
        .map(|(_, page)| *page)
}

/// 脱敏：抹去绝对路径（Windows 盘符 / UNC / 常见 Unix 前缀）与密钥碎片。
///
/// 相对路径（`models/asr/...`）与 URL（`https://api.deepseek.com/v1`）保留——
/// 它们不含本机身份信息，且对排障有用。
pub fn sanitize(raw: &str) -> String {
    let no_cr = raw.replace('\r', "");
    let scrubbed_keys = scrub_keys(&no_cr);
    scrub_paths(&scrubbed_keys)
}

/// 抹去 `sk-…`、`Bearer …`、`key=/api_key=/token=…` 的值
fn scrub_keys(s: &str) -> String {
    const KEY_PREFIXES: [&str; 5] = ["api_key=", "apikey=", "api-key=", "key=", "token="];
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0usize;
    while i < b.len() {
        let rest = &s[i..];
        // sk- 长随机令牌 → sk-***
        if rest.starts_with("sk-") {
            let token_len: usize = rest[3..]
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
                .map(char::len_utf8)
                .sum();
            if token_len >= 6 {
                out.push_str("sk-***");
                i += 3 + token_len;
                continue;
            }
        }
        // Bearer 令牌 → Bearer ***
        if b.len() - i >= 7 && b[i..i + 7].eq_ignore_ascii_case(b"bearer ") {
            let skip: usize = rest[7..]
                .chars()
                .take_while(|c| !c.is_whitespace())
                .map(char::len_utf8)
                .sum();
            out.push_str("Bearer ***");
            i += 7 + skip;
            continue;
        }
        // key= / api_key= / token= 系：仅词首匹配，值替换为 ***
        let at_word_start = i == 0 || !(b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_');
        if at_word_start {
            let mut handled = false;
            for prefix in KEY_PREFIXES {
                if b.len() - i >= prefix.len()
                    && b[i..i + prefix.len()].eq_ignore_ascii_case(prefix.as_bytes())
                {
                    let skip: usize = rest[prefix.len()..]
                        .chars()
                        .take_while(|c| *c != '&' && *c != ' ' && *c != '"' && *c != '\n' && *c != ',')
                        .map(char::len_utf8)
                        .sum();
                    out.push_str(prefix);
                    if skip > 0 {
                        out.push_str("***");
                    }
                    i += prefix.len() + skip;
                    handled = true;
                    break;
                }
            }
            if handled {
                continue;
            }
        }
        let ch = rest.chars().next().unwrap();
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

/// 抹去绝对路径 → `<路径>`
fn scrub_paths(s: &str) -> String {
    let delimiters = [' ', '"', '\'', ',', ';', ')', '）', '，', '》', '\n', '\t'];
    let mut out = String::with_capacity(s.len());
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let mut i = 0;
    while i < chars.len() {
        let (idx, c) = chars[i];
        // Windows 盘符路径：X:\ 或 X:/
        let win = c.is_ascii_uppercase()
            && s[idx..].len() > 2
            && s.as_bytes()[idx + 1] == b':'
            && matches!(s.as_bytes()[idx + 2], b'\\' | b'/');
        // UNC：\\
        let unc = c == '\\' && chars.get(i + 1).map_or(false, |(_, c2)| *c2 == '\\');
        // 常见 Unix 绝对前缀（仅令牌起始处）
        let rest = &s[idx..];
        let unix = matches!(
            i,
            0
        ) || chars[i - 1].1.is_whitespace()
            || matches!(chars[i - 1].1, '(' | '[' | '“' | '：' | ':' | '=');
        let unix = unix
            && ["/home/", "/root/", "/tmp/", "/Users/", "/var/", "/opt/", "/mnt/", "/media/"]
                .iter()
                .any(|p| rest.starts_with(p));
        if win || unc || unix {
            out.push_str("<路径>");
            i += 1;
            while i < chars.len() && !delimiters.contains(&chars[i].1) {
                i += 1;
            }
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 脱敏_抹去windows绝对路径() {
        let raw = "合成失败: 模型文件不存在: D:\\19-Training\\Rust\\votex\\models\\asr\\firered-asr-ctc\\model.int8.onnx";
        let out = sanitize(raw);
        assert!(!out.contains("D:\\"), "盘符路径必须抹除: {out}");
        assert!(out.contains("<路径>"));
        assert!(out.contains("合成失败"));
    }

    #[test]
    fn 脱敏_抹去unix绝对路径_保留相对路径与url() {
        let raw = "/home/user/votex/models 加载失败; 相对路径 models/tts 保留; 端点 https://api.deepseek.com/v1 保留";
        let out = sanitize(raw);
        assert!(out.contains("<路径>"));
        assert!(out.contains("models/tts 保留"));
        assert!(out.contains("https://api.deepseek.com/v1"));
    }

    #[test]
    fn 脱敏_抹去密钥碎片() {
        // 独立 sk- 令牌、Bearer 令牌、key= 系键值三种碎片都须脱敏
        let raw = "请求失败: Authorization: Bearer abcdef123456, api_key=sk-abcdefghij123456789, 备用令牌 sk-zzzz9999yyy";
        let out = sanitize(raw);
        assert!(!out.contains("abcdefghij"), "{out}");
        assert!(out.contains("sk-***"), "{out}");
        assert!(out.contains("Bearer ***"), "{out}");
        assert!(out.contains("api_key=***"), "{out}");
    }

    #[test]
    fn 诊断_vocab_size不兼容_指向paraformer文档() {
        let d = diagnose("加载失败: ONNX 元数据缺少 vocab_size (models/asr/paraformer/model_quant.onnx)");
        assert_eq!(d.summary, "模型包与推理运行时不兼容");
        assert_eq!(d.doc_page, Some("asr-paraformer.md"));
        assert!(!d.detail.contains("D:\\"));
    }

    #[test]
    fn 诊断_模型缺失_给出去向() {
        let d = diagnose("合成失败: 模型 firered-asr 未下载");
        assert_eq!(d.summary, "模型未下载或文件缺失");
        assert!(d.explain.as_deref().unwrap_or("").contains("下载"));
        assert_eq!(d.doc_page, Some("asr-firered-asr.md"));
    }

    #[test]
    fn 诊断_音色缺转写_给出voice_add建议() {
        let d = diagnose("合成失败: 音色 \"demo\" 缺少转写文本，请 voice add --transcript");
        assert_eq!(d.summary, "音色不存在或缺少转写文本");
        assert!(d.explain.as_deref().unwrap_or("").contains("voice add"));
    }

    #[test]
    fn 诊断_取消_直接透传() {
        let d = diagnose("任务已取消");
        assert_eq!(d.summary, "任务已取消");
        assert!(d.explain.is_none());
        assert!(d.doc_page.is_none());
    }

    #[test]
    fn 诊断_鉴权失败_不透出密钥() {
        let d = diagnose("TTS 失败: Azure 返回 401 Unauthorized, key=8f2c9d1e");
        assert_eq!(d.summary, "在线服务鉴权失败");
        assert!(!d.detail.contains("8f2c9d1e"));
        assert_eq!(d.doc_page, Some("online-azure-speech.md"));
    }

    #[test]
    fn 诊断_未知错误_兜底并保留脱敏详情() {
        let raw = "合成失败: 未知异常 0xC0000409 (D:\\tmp\\core.bin)";
        let d = diagnose(raw);
        assert_eq!(d.summary, "操作失败，请重试");
        assert!(d.detail.contains("<路径>"));
        assert!(d.explain.is_some());
    }
}
