use crate::ocr::value_object::OcrResult;

/// OCR 领域服务
pub struct OcrService;

impl OcrService {
    pub fn new() -> Self {
        Self
    }

    /// 获取完整文本
    pub fn full_text(result: &OcrResult) -> String {
        result.full_text()
    }

    /// 统计信息
    pub fn statistics(result: &OcrResult) -> String {
        let total_blocks = result.blocks.len();
        let avg_conf = if total_blocks > 0 {
            result.blocks.iter().map(|b| b.confidence).sum::<f32>() / total_blocks as f32
        } else {
            0.0
        };
        format!(
            "识别块数: {}, 平均置信度: {:.2}%, 总字符数: {}",
            total_blocks,
            avg_conf * 100.0,
            result.full_text().chars().count()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ocr::value_object::*;

    #[test]
    fn ocr_service_统计信息() {
        let result = OcrResult {
            blocks: vec![
                OcrTextBlock {
                    text: "测试".to_string(),
                    confidence: 0.99,
                    box_points: TextBox {
                        x0: 0.0, y0: 0.0, x1: 50.0, y1: 0.0,
                        x2: 50.0, y2: 20.0, x3: 0.0, y3: 20.0,
                    },
                },
            ],
            output_path: std::path::PathBuf::from("out.txt"),
        };
        let stat = OcrService::statistics(&result);
        assert!(stat.contains("识别块数: 1"));
        assert!(stat.contains("99"));
    }
}
