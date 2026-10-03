//! EPUB/DOCX 文档解析端到端测试
//!
//! 构造真实 zip 容器（EPUB / DOCX 最小结构）验证完整提取链路。

use std::io::Write as _;

fn write_zip(entries: Vec<(&str, &str)>) -> Vec<u8> {
    let buf = std::io::Cursor::new(Vec::new());
    let mut zip = zip::ZipWriter::new(buf);
    for (name, content) in entries {
        zip.start_file(name, zip::write::SimpleFileOptions::default()).unwrap();
        zip.write_all(content.as_bytes()).unwrap();
    }
    zip.finish().unwrap().into_inner()
}

const CONTAINER: &str = r#"<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
</container>"#;

const OPF: &str = r#"<?xml version="1.0"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0">
  <manifest>
    <item id="c1" href="ch1.xhtml" media-type="application/xhtml+xml"/>
    <item id="c2" href="ch2.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="c1"/><itemref idref="c2"/></spine>
</package>"#;

const CH1: &str = r#"<html xmlns="http://www.w3.org/1999/xhtml"><head><title>x</title></head>
<body><h1>第一章 起点</h1><p>清晨，他醒了。</p><p>天很蓝 &amp; 云很白。</p></body></html>"#;

const CH2: &str = r#"<html xmlns="http://www.w3.org/1999/xhtml"><head><title>y</title></head>
<body><h1>第二章 转折</h1><p>午后，变天了。</p></body></html>"#;

#[test]
fn epub_端到端提取() {
    let bytes = write_zip(vec![
        ("mimetype", "application/epub+zip"),
        ("META-INF/container.xml", CONTAINER),
        ("OEBPS/content.opf", OPF),
        ("OEBPS/ch1.xhtml", CH1),
        ("OEBPS/ch2.xhtml", CH2),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("book.epub");
    std::fs::write(&path, &bytes).unwrap();

    let text = votex_infra::document::extract_text(&path).unwrap();
    // spine 顺序
    let pos1 = text.find("第一章 起点").expect("ch1 缺失");
    let pos2 = text.find("第二章 转折").expect("ch2 缺失");
    assert!(pos1 < pos2);
    // 正文与实体
    assert!(text.contains("清晨，他醒了。"));
    assert!(text.contains("天很蓝 & 云很白。"));
    assert!(text.contains("午后，变天了。"));
    // 标题独占一行（章节切分可识别）
    assert!(text.lines().any(|l| l.trim() == "第一章 起点"));
}

#[test]
fn docx_端到端提取() {
    let doc_xml = r#"<?xml version="1.0" encoding="UTF-8"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
    <w:body>
    <w:p><w:r><w:t>第一章 起点</w:t></w:r></w:p>
    <w:p><w:r><w:t>清晨，他醒了。</w:t></w:r><w:r><w:t>天很蓝。</w:t></w:r></w:p>
    </w:body></w:document>"#;
    let bytes = write_zip(vec![("word/document.xml", doc_xml)]);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("book.docx");
    std::fs::write(&path, &bytes).unwrap();

    let text = votex_infra::document::extract_text(&path).unwrap();
    assert!(text.contains("第一章 起点"));
    assert!(text.contains("清晨，他醒了。天很蓝。"));
}

#[test]
fn epub_章节切分对接() {
    // 提取结果可直接走领域层章节切分
    let bytes = write_zip(vec![
        ("META-INF/container.xml", CONTAINER),
        ("OEBPS/content.opf", OPF),
        ("OEBPS/ch1.xhtml", CH1),
        ("OEBPS/ch2.xhtml", CH2),
    ]);
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("book.epub");
    std::fs::write(&path, &bytes).unwrap();

    let text = votex_infra::document::extract_text(&path).unwrap();
    let chapters = votex_domain::tts::chapter::split_into_chapters(&text);
    assert_eq!(chapters.len(), 2);
    assert_eq!(chapters[0].title.as_deref(), Some("第一章 起点"));
    assert_eq!(chapters[1].title.as_deref(), Some("第二章 转折"));
    assert!(chapters[0].body.contains("清晨"));
}
