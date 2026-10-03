//! DOCX 文本提取
//!
//! DOCX 是 zip 容器，正文在 `word/document.xml`：`<w:p>` 是段落边界，
//! `<w:t>` 是文本 run。段落之间插入换行，标题样式段落独占一行
//! （可被章节切分识别）。

use anyhow::{Context, Result};
use quick_xml::events::Event;
use quick_xml::Reader;
use std::io::Read;
use std::path::Path;

/// 从 DOCX 文件提取纯文本
pub fn extract(path: &Path) -> Result<String> {
    let file = std::fs::File::open(path).context("打开 DOCX 失败")?;
    let mut zip = zip::ZipArchive::new(file).context("读取 DOCX 容器失败")?;
    let mut entry = zip
        .by_name("word/document.xml")
        .context("DOCX 缺少 word/document.xml")?;
    let mut xml = String::new();
    entry.read_to_string(&mut xml)?;

    let mut out = String::new();
    let mut in_para = false;
    let mut para = String::new();

    let mut reader = Reader::from_str(&xml);
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let name = e.name().as_ref().to_ascii_lowercase();
                match name.as_slice() {
                    b"w:p" => {
                        in_para = true;
                        para.clear();
                    }
                    b"w:tab" if in_para => para.push('\t'),
                    b"w:br" | b"w:cr" if in_para => para.push('\n'),
                    _ => {}
                }
            }
            Ok(Event::Empty(e)) => {
                let name = e.name().as_ref().to_ascii_lowercase();
                match name.as_slice() {
                    b"w:tab" if in_para => para.push('\t'),
                    b"w:br" | b"w:cr" if in_para => para.push('\n'),
                    _ => {}
                }
            }
            Ok(Event::End(e)) => {
                let name = e.name().as_ref().to_ascii_lowercase();
                if name.as_slice() == b"w:p" {
                    in_para = false;
                    let line = para.trim();
                    if !line.is_empty() {
                        out.push_str(line);
                        out.push('\n');
                    }
                }
            }
            Ok(Event::Text(t)) => {
                if in_para {
                    if let Ok(raw) = t.xml_content() {
                        para.push_str(&raw);
                    }
                }
            }
            Ok(Event::GeneralRef(r)) => {
                if in_para {
                    if let Ok(raw) = r.xml_content() {
                        para.push_str(&raw);
                    }
                }
            }
            Ok(Event::Eof) => break,
            Err(_) => break,
            _ => {}
        }
        buf.clear();
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docx_段落提取_真实容器() {
        // 构造最小 DOCX 容器
        let doc_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
        <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
        <w:body>
        <w:p><w:r><w:t>第一章 起点</w:t></w:r></w:p>
        <w:p><w:r><w:t>清晨，他醒了。</w:t></w:r><w:r><w:t>天很蓝。</w:t></w:r></w:p>
        </w:body></w:document>"#;

        let buf = std::io::Cursor::new(Vec::new());
        let mut zip = zip::ZipWriter::new(buf);
        zip.start_file(
            "word/document.xml",
            zip::write::SimpleFileOptions::default(),
        )
        .unwrap();
        std::io::Write::write_all(&mut zip, doc_xml.as_bytes()).unwrap();
        let bytes = zip.finish().unwrap().into_inner();

        // 写入临时文件再解析
        let dir = tempfile::tempdir().unwrap();
        let docx_path = dir.path().join("test.docx");
        std::fs::write(&docx_path, &bytes).unwrap();
        let text = extract(&docx_path).unwrap();
        assert!(text.contains("第一章 起点"));
        assert!(text.contains("清晨，他醒了。天很蓝。"));
    }
}
