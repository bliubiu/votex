//! 文档文本提取（T3 格式输入扩展）
//!
//! 支持：TXT/MD（含 GBK 自动检测）、EPUB（zip+XHTML）、DOCX（zip+XML）、
//! PDF（文本层，扫描件请走 OCR 流水线）。
//! 提取结果为带换行分隔的纯文本，章节标题独占一行，
//! 直接对接领域层 `chapter::split_into_chapters` 章节切分。

pub mod docx;
pub mod epub;
pub mod pdf;

use anyhow::{Context, Result};
use std::path::Path;
use std::str::FromStr;

/// 支持的文档格式
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocFormat {
    Txt,
    Epub,
    Docx,
    Pdf,
}

impl DocFormat {
    /// 按扩展名识别格式
    pub fn from_path(path: &Path) -> Option<DocFormat> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())?;
        DocFormat::from_str(&ext).ok()
    }
}

impl FromStr for DocFormat {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "txt" | "md" | "markdown" => Ok(DocFormat::Txt),
            "epub" => Ok(DocFormat::Epub),
            "docx" => Ok(DocFormat::Docx),
            "pdf" => Ok(DocFormat::Pdf),
            _ => Err(()),
        }
    }
}

/// 从文档提取纯文本（统一入口）
///
/// TXT/MD 走编码自动检测（UTF-8/GBK），其余格式按内部结构解析。
pub fn extract_text(path: &Path) -> Result<String> {
    if !path.exists() {
        anyhow::bail!("文件不存在: {:?}", path);
    }
    let format = DocFormat::from_path(path).unwrap_or(DocFormat::Txt);
    let text = match format {
        DocFormat::Txt => crate::encoding::detector::EncodingDetector::read_text_file(path)
            .context("读取文本文件失败")?,
        DocFormat::Epub => epub::extract(path).context("解析 EPUB 失败")?,
        DocFormat::Docx => docx::extract(path).context("解析 DOCX 失败")?,
        DocFormat::Pdf => pdf::extract(path).context("解析 PDF 失败（扫描件请使用 OCR 流水线）")?,
    };
    let trimmed = text.trim().to_string();
    if trimmed.is_empty() {
        anyhow::bail!("文档中未提取到文本内容: {:?}", path);
    }
    Ok(trimmed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 格式识别() {
        assert_eq!(DocFormat::from_path(Path::new("a.epub")), Some(DocFormat::Epub));
        assert_eq!(DocFormat::from_path(Path::new("A.DOCX")), Some(DocFormat::Docx));
        assert_eq!(DocFormat::from_path(Path::new("b.pdf")), Some(DocFormat::Pdf));
        assert_eq!(DocFormat::from_path(Path::new("c.txt")), Some(DocFormat::Txt));
        assert_eq!(DocFormat::from_path(Path::new("d.md")), Some(DocFormat::Txt));
        assert_eq!(DocFormat::from_path(Path::new("e.bin")), None);
    }
}
