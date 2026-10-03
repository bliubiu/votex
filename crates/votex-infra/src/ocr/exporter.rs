use votex_domain::error::OcrError;
use votex_domain::ocr::value_object::OcrResult;
use votex_domain::ocr::provider::OcrExporter;
use std::path::Path;

/// TXT 导出器
pub struct TxtExporter;

impl OcrExporter for TxtExporter {
    fn format_name(&self) -> &'static str {
        "txt"
    }

    fn extension(&self) -> &'static str {
        "txt"
    }

    fn export(&self, result: &OcrResult, output_path: &Path) -> Result<(), OcrError> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| OcrError::ExportFailed(format!("创建目录失败: {}", e)))?;
        }
        let text = result.full_text();
        let tmp_path = output_path.with_extension("tmp");
        std::fs::write(&tmp_path, &text)
            .map_err(|e| OcrError::ExportFailed(format!("写入文件失败: {}", e)))?;
        std::fs::rename(&tmp_path, output_path)
            .map_err(|e| OcrError::ExportFailed(format!("重命名文件失败: {}", e)))?;
        tracing::info!("TXT 导出完成: {:?} ({} 字符)", output_path, text.chars().count());
        Ok(())
    }
}

/// JSON 导出器（含坐标和置信度）
pub struct JsonExporter;

impl OcrExporter for JsonExporter {
    fn format_name(&self) -> &'static str {
        "json"
    }

    fn extension(&self) -> &'static str {
        "json"
    }

    fn export(&self, result: &OcrResult, output_path: &Path) -> Result<(), OcrError> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| OcrError::ExportFailed(format!("创建目录失败: {}", e)))?;
        }
        let json = serde_json::to_string_pretty(result)
            .map_err(|e| OcrError::ExportFailed(format!("JSON 序列化失败: {}", e)))?;
        let tmp_path = output_path.with_extension("tmp");
        std::fs::write(&tmp_path, &json)
            .map_err(|e| OcrError::ExportFailed(format!("写入文件失败: {}", e)))?;
        std::fs::rename(&tmp_path, output_path)
            .map_err(|e| OcrError::ExportFailed(format!("重命名文件失败: {}", e)))?;
        tracing::info!("JSON 导出完成: {:?} ({} 块)", output_path, result.block_count());
        Ok(())
    }
}

/// Markdown 导出器
pub struct MarkdownExporter;

impl OcrExporter for MarkdownExporter {
    fn format_name(&self) -> &'static str {
        "markdown"
    }

    fn extension(&self) -> &'static str {
        "md"
    }

    fn export(&self, result: &OcrResult, output_path: &Path) -> Result<(), OcrError> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| OcrError::ExportFailed(format!("创建目录失败: {}", e)))?;
        }

        let mut md = String::new();
        md.push_str("# OCR 识别结果\n\n");
        md.push_str(&format!("> 共识别 {} 个文本区域\n\n", result.block_count()));

        for (i, block) in result.blocks.iter().enumerate() {
            md.push_str(&format!("## 区域 {}\n\n", i + 1));
            md.push_str(&format!("- 置信度: {:.1}%\n", block.confidence * 100.0));
            md.push_str(&format!("- 文本: `{}`\n\n", block.text));
        }

        md.push_str("---\n\n### 全文\n\n```\n");
        md.push_str(&result.full_text());
        md.push_str("\n```\n");

        let tmp_path = output_path.with_extension("tmp");
        std::fs::write(&tmp_path, &md)
            .map_err(|e| OcrError::ExportFailed(format!("写入文件失败: {}", e)))?;
        std::fs::rename(&tmp_path, output_path)
            .map_err(|e| OcrError::ExportFailed(format!("重命名文件失败: {}", e)))?;
        tracing::info!("Markdown 导出完成: {:?}", output_path);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use votex_domain::ocr::value_object::{OcrTextBlock, TextBox};
    use tempfile::TempDir;

    fn sample_result() -> OcrResult {
        OcrResult {
            blocks: vec![
                OcrTextBlock {
                    text: "第一行文字".to_string(),
                    confidence: 0.98,
                    box_points: TextBox {
                        x0: 0.0, y0: 0.0, x1: 100.0, y1: 0.0,
                        x2: 100.0, y2: 30.0, x3: 0.0, y3: 30.0,
                    },
                },
                OcrTextBlock {
                    text: "第二行文字".to_string(),
                    confidence: 0.95,
                    box_points: TextBox {
                        x0: 0.0, y0: 30.0, x1: 100.0, y1: 30.0,
                        x2: 100.0, y2: 60.0, x3: 0.0, y3: 60.0,
                    },
                },
            ],
            output_path: std::path::PathBuf::from("out.txt"),
        }
    }

    #[test]
    fn txt_exporter_导出() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("result.txt");
        let exporter = TxtExporter;
        exporter.export(&sample_result(), &path).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("第一行文字"));
        assert!(content.contains("第二行文字"));
    }

    #[test]
    fn json_exporter_导出() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("result.json");
        let exporter = JsonExporter;
        exporter.export(&sample_result(), &path).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("第一行文字"));
        assert!(content.contains("confidence"));
    }

    #[test]
    fn markdown_exporter_导出() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("result.md");
        let exporter = MarkdownExporter;
        exporter.export(&sample_result(), &path).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("# OCR 识别结果"));
        assert!(content.contains("第一行文字"));
    }
}
