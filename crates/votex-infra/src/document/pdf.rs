//! PDF 文本提取（文本层）
//!
//! 使用 pdf-extract（lopdf 后端）提取文本层内容。
//! 扫描型 PDF（无文本层）会返回空文本——上层报错并提示走 OCR 流水线
//! （PaddleOCR），不静默产出空结果。

use anyhow::{Context, Result};
use std::path::Path;

/// 从 PDF 提取文本层内容
pub fn extract(path: &Path) -> Result<String> {
    let text = pdf_extract::extract_text(path)
        .with_context(|| format!("PDF 文本层提取失败: {:?}（扫描件请使用 OCR 流水线）", path))?;
    // pdf-extract 输出的换行较碎：把单词级换行合并、段落保留
    Ok(normalize_pdf_text(&text))
}

/// 归一化 PDF 提取文本：合并硬换行，压缩空行
fn normalize_pdf_text(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_line_empty = false;
    for line in raw.lines() {
        let l = line.trim();
        if l.is_empty() {
            if !prev_line_empty && !out.is_empty() {
                out.push('\n');
            }
            prev_line_empty = true;
        } else {
            out.push_str(l);
            out.push('\n');
            prev_line_empty = false;
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pdf_归一化_合并碎行() {
        let raw = "第一段第一行\n第一段第二行\n\n第二段\n\n\n\n第三段";
        let text = normalize_pdf_text(raw);
        assert_eq!(text, "第一段第一行\n第一段第二行\n\n第二段\n\n第三段");
    }
}
