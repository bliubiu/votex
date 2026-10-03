//! EPUB 文本提取
//!
//! EPUB 本质是 zip 容器：META-INF/container.xml 指向 OPF 包描述文件，
//! OPF 的 manifest（id→href）+ spine（阅读顺序）决定 XHTML 文档序列。
//! 对每个 XHTML 块级元素（p/h1~h6/div/li/br）按换行边界提取文本，
//! 标题行独占一行——可被领域层章节切分（第X章/Chapter N）直接识别。

use anyhow::{anyhow, Context, Result};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::io::Read;
use std::path::Path;

/// 从 EPUB 文件提取纯文本
pub fn extract(path: &Path) -> Result<String> {
    let file = std::fs::File::open(path).context("打开 EPUB 失败")?;
    let mut zip = zip::ZipArchive::new(file).context("读取 EPUB 容器失败")?;

    // 1. container.xml → OPF 路径
    let container = read_entry(&mut zip, "META-INF/container.xml")?;
    let opf_path = find_opf_path(&container)
        .ok_or_else(|| anyhow!("container.xml 中未找到 rootfile full-path"))?;

    // 2. OPF → manifest + spine
    let opf = read_entry(&mut zip, &opf_path)?;
    let (manifest, spine) = parse_opf(&opf)?;

    // 3. 按 spine 顺序提取每个 XHTML 文档
    let opf_dir = std::path::Path::new(&opf_path)
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_default();

    let mut out = String::new();
    for idref in &spine {
        let Some(href) = manifest.get(idref) else { continue };
        // 相对路径解析（OPF 目录 + href），并处理 URL 编码
        let entry_name = resolve_rel(&opf_dir, href);
        let Ok(mut entry) = zip.by_name(&entry_name) else { continue };
        let mut content = String::new();
        if entry.read_to_string(&mut content).is_err() {
            continue;
        }
        let body = extract_xhtml_text(&content);
        if !body.trim().is_empty() {
            out.push_str(&body);
            out.push('\n');
        }
    }
    Ok(out)
}

/// 读取 zip 内的文本条目
fn read_entry(zip: &mut zip::ZipArchive<std::fs::File>, name: &str) -> Result<String> {
    let mut entry = zip
        .by_name(name)
        .with_context(|| format!("EPUB 缺少条目: {}", name))?;
    let mut s = String::new();
    entry.read_to_string(&mut s)?;
    Ok(s)
}

/// 从 container.xml 提取 rootfile full-path
fn find_opf_path(container_xml: &str) -> Option<String> {
    let mut reader = Reader::from_str(container_xml);
    loop {
        match reader.read_event() {
            Ok(Event::Empty(e)) | Ok(Event::Start(e)) => {
                if e.name().as_ref().ends_with(b"rootfile") {
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref().ends_with(b"full-path") {
                            return attr
                                .unescape_value()
                                .map(|v| v.to_string())
                                .ok();
                        }
                    }
                }
            }
            Ok(Event::Eof) => return None,
            Err(_) => return None,
            _ => {}
        }
    }
}

/// 解析 OPF：返回 (manifest id→href, spine idref 有序表)
fn parse_opf(opf_xml: &str) -> Result<(std::collections::HashMap<String, String>, Vec<String>)> {
    let mut manifest = std::collections::HashMap::new();
    let mut spine = Vec::new();
    let mut in_manifest = false;
    let mut in_spine = false;

    let mut reader = Reader::from_str(opf_xml);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) | Ok(Event::Empty(e)) => {
                let name = e.name().as_ref().to_ascii_lowercase();
                match name.as_slice() {
                    b"manifest" => in_manifest = true,
                    b"spine" => in_spine = true,
                    b"item" if in_manifest => {
                        let mut id = None;
                        let mut href = None;
                        for attr in e.attributes().flatten() {
                            let key = attr.key.as_ref().to_ascii_lowercase();
                            match key.as_slice() {
                                b"id" => id = attr.unescape_value().map(|v| v.to_string()).ok(),
                                b"href" => href = attr.unescape_value().map(|v| v.to_string()).ok(),
                                _ => {}
                            }
                        }
                        if let (Some(id), Some(href)) = (id, href) {
                            manifest.insert(id, href);
                        }
                    }
                    b"itemref" if in_spine => {
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref().eq_ignore_ascii_case(b"idref") {
                                let v = attr.unescape_value().map(|v| v.to_string()).unwrap_or_default();
                                if !v.is_empty() {
                                    spine.push(v);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e)) => {
                let name = e.name().as_ref().to_ascii_lowercase();
                match name.as_slice() {
                    b"manifest" => in_manifest = false,
                    b"spine" => in_spine = false,
                    _ => {}
                }
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(anyhow!("OPF 解析失败: {}", e)),
            _ => {}
        }
        buf.clear();
    }
    Ok((manifest, spine))
}

/// 从 XHTML/HTML 提取纯文本：块级元素换行，其余拼接
pub(crate) fn extract_xhtml_text(xml: &str) -> String {
    let mut out = String::new();
    let mut skip_depth = 0usize; // script/style 深度
    let mut in_title = false; // <title> 为文档元数据，不进入正文
    let mut reader = Reader::from_str(xml);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let name = e.name().as_ref().to_ascii_lowercase();
                if matches!(name.as_slice(), b"script" | b"style") {
                    skip_depth += 1;
                    continue;
                }
                if name.as_slice() == b"title" {
                    in_title = true;
                    continue;
                }
                if is_block(&name) {
                    out.push('\n');
                }
            }
            Ok(Event::Empty(e)) => {
                let name = e.name().as_ref().to_ascii_lowercase();
                if is_block(&name) || name.as_slice() == b"br" {
                    out.push('\n');
                }
            }
            Ok(Event::End(e)) => {
                let name = e.name().as_ref().to_ascii_lowercase();
                if matches!(name.as_slice(), b"script" | b"style") {
                    skip_depth = skip_depth.saturating_sub(1);
                    continue;
                }
                if name.as_slice() == b"title" {
                    in_title = false;
                    continue;
                }
                if is_block(&name) {
                    out.push('\n');
                }
            }
            Ok(Event::Text(t)) => {
                if skip_depth == 0 && !in_title {
                    // xml_content() 已完成实体反转义（&amp; 等）；
                    // 实体引用会被拆分为独立 GeneralRef 事件，文本需原样拼接
                    if let Ok(decoded) = t.xml_content() {
                        out.push_str(&decoded);
                    }
                }
            }
            Ok(Event::GeneralRef(r)) => {
                // 实体引用事件（内容为实体名，如 "amp"、"nbsp"、"#x4E2D"）
                if skip_depth == 0 && !in_title {
                    out.push_str(&resolve_entity(&String::from_utf8_lossy(r.as_ref())));
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    // 压缩空行与行内多余空格
    let mut lines: Vec<String> = Vec::new();
    for line in out.lines() {
        let l = collapse_spaces(line);
        if l.is_empty() {
            if lines.last().map(|s: &String| !s.is_empty()).unwrap_or(false) {
                lines.push(String::new());
            }
        } else {
            lines.push(l);
        }
    }
    lines.join("\n")
}

/// 解析 XML/HTML 实体引用名（不含 & 和 ;）
fn resolve_entity(name: &str) -> String {
    match name {
        "amp" => "&".into(),
        "lt" => "<".into(),
        "gt" => ">".into(),
        "quot" => "\"".into(),
        "apos" => "'".into(),
        "nbsp" => "\u{00A0}".into(),
        "mdash" => "—".into(),
        "ndash" => "–".into(),
        "hellip" => "…".into(),
        "ldquo" => "“".into(),
        "rdquo" => "”".into(),
        "lsquo" => "‘".into(),
        "rsquo" => "’".into(),
        _ => {
            // 数字实体：&#NN; / &#xHH;
            if let Some(hex) = name.strip_prefix("#x").or_else(|| name.strip_prefix("#X")) {
                u32::from_str_radix(hex, 16)
                    .ok()
                    .and_then(char::from_u32)
                    .map(|c| c.to_string())
                    .unwrap_or_default()
            } else if let Some(dec) = name.strip_prefix('#') {
                dec.parse::<u32>()
                    .ok()
                    .and_then(char::from_u32)
                    .map(|c| c.to_string())
                    .unwrap_or_default()
            } else {
                String::new() // 未知实体丢弃
            }
        }
    }
}

/// 是否为块级元素（产生换行边界）
fn is_block(name: &[u8]) -> bool {
    matches!(
        name,
        b"p" | b"div" | b"h1" | b"h2" | b"h3" | b"h4" | b"h5" | b"h6" | b"li" | b"tr"
            | b"blockquote" | b"section" | b"article" | b"figure" | b"figcaption"
    )
}

/// 压缩连续空白
fn collapse_spaces(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for c in s.trim().chars() {
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    out
}

/// 解析 OPF 目录相对路径（处理 URL 转码）
fn resolve_rel(base: &std::path::Path, href: &str) -> String {
    let decoded = href
        .replace("%20", " ")
        .replace("%3A", ":")
        .replace("%2F", "/");
    let combined = base.join(&decoded);
    let normalized = combined
        .to_string_lossy()
        .replace('\\', "/");
    normalized.trim_start_matches('/').to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xhtml_提取文本() {
        let xml = r#"<html><head><title>第一章 起点</title></head>
        <body><h1>第一章 起点</h1><p>清晨，他醒了。</p><p>天很蓝 &amp; 云很白。</p>
        <script>var x = 1;</script></body></html>"#;
        let text = extract_xhtml_text(xml);
        assert!(text.contains("第一章 起点"));
        assert!(text.contains("清晨，他醒了。"));
        assert!(text.contains("天很蓝 & 云很白。"));
        assert!(!text.contains("var x"));
    }

    #[test]
    fn opf_解析_spine顺序() {
        let opf = r#"<?xml version="1.0"?>
        <package><manifest>
            <item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
            <item id="c2" href="text/ch2.xhtml" media-type="application/xhtml+xml"/>
        </manifest>
        <spine><itemref idref="c1"/><itemref idref="c2"/></spine></package>"#;
        let (manifest, spine) = parse_opf(opf).unwrap();
        assert_eq!(manifest.get("c1").unwrap(), "ch1.xhtml");
        assert_eq!(spine, vec!["c1", "c2"]);
    }

    #[test]
    fn container_提取opf路径() {
        let container = r#"<?xml version="1.0"?>
        <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
          <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
        </container>"#;
        assert_eq!(find_opf_path(container), Some("OEBPS/content.opf".to_string()));
    }
}
