use serde::{Deserialize, Serialize};

use std::path::PathBuf;

/// OCR 识别语种
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OcrLanguage {
    /// 中文 + 英文
    Zh,
    /// 英文
    En,
    /// 中英混合
    ZhEn,
}

impl Default for OcrLanguage {
    fn default() -> Self {
        OcrLanguage::Zh
    }
}

/// OCR 输出格式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OcrOutputFormat {
    /// 纯文本
    Txt,
    /// JSON（含坐标和置信度）
    Json,
    /// Markdown
    Markdown,
    /// DOCX (Word)
    Docx,
}

impl Default for OcrOutputFormat {
    fn default() -> Self {
        OcrOutputFormat::Txt
    }
}

impl OcrOutputFormat {
    /// 返回文件扩展名
    pub fn extension(&self) -> &'static str {
        match self {
            OcrOutputFormat::Txt => "txt",
            OcrOutputFormat::Json => "json",
            OcrOutputFormat::Markdown => "md",
            OcrOutputFormat::Docx => "docx",
        }
    }

    /// 从字符串解析格式
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "json" => OcrOutputFormat::Json,
            "md" | "markdown" => OcrOutputFormat::Markdown,
            "docx" | "word" => OcrOutputFormat::Docx,
            _ => OcrOutputFormat::Txt,
        }
    }
}

/// OCR 识别参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrParams {
    pub language: OcrLanguage,
    pub output_format: OcrOutputFormat,
    /// 是否启用方向分类
    pub use_angle_cls: bool,
    /// 检测置信度阈值
    pub det_db_thresh: f32,
    /// 检测框阈值
    pub det_db_box_thresh: f32,
    /// 输出目录（多页时使用）
    pub output_dir: Option<PathBuf>,
    /// 最大并发数（批量处理时）
    pub max_concurrency: usize,
}

impl Default for OcrParams {
    fn default() -> Self {
        Self {
            language: OcrLanguage::Zh,
            output_format: OcrOutputFormat::Txt,
            use_angle_cls: true,
            det_db_thresh: 0.3,
            det_db_box_thresh: 0.5,
            output_dir: None,
            max_concurrency: 1,
        }
    }
}

/// 文本框坐标（四个角点）
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextBox {
    /// 左上
    pub x0: f32,
    pub y0: f32,
    /// 右上
    pub x1: f32,
    pub y1: f32,
    /// 右下
    pub x2: f32,
    pub y2: f32,
    /// 左下
    pub x3: f32,
    pub y3: f32,
}

/// 单行识别结果
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrTextBlock {
    pub text: String,
    pub confidence: f32,
    pub box_points: TextBox,
}

/// OCR 识别结果（单页）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResult {
    pub blocks: Vec<OcrTextBlock>,
    pub output_path: std::path::PathBuf,
}

impl OcrResult {
    /// 拼接所有文本行
    pub fn full_text(&self) -> String {
        self.blocks
            .iter()
            .map(|b| b.text.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// 识别块数量
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// 平均置信度
    pub fn avg_confidence(&self) -> f32 {
        if self.blocks.is_empty() {
            return 0.0;
        }
        self.blocks.iter().map(|b| b.confidence).sum::<f32>() / self.blocks.len() as f32
    }
}

/// 页面状态
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PageStatus {
    Pending,
    Processing,
    Completed,
    Failed(String),
    Skipped,
}

/// OCR 阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OcrPhase {
    Idle,
    LoadingImage,
    Detecting,
    Classifying,
    Recognizing,
    Exporting,
    Completed,
    Failed,
}

/// OCR 进度
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OcrProgress {
    pub phase: OcrPhase,
    pub current_page: usize,
    pub total_pages: usize,
    pub current_block: usize,
    pub total_blocks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ocr_params_默认值() {
        let params = OcrParams::default();
        assert_eq!(params.language, OcrLanguage::Zh);
        assert!(params.use_angle_cls);
        assert!((params.det_db_thresh - 0.3).abs() < f32::EPSILON);
    }

    #[test]
    fn ocr_result_拼接文本() {
        let result = OcrResult {
            blocks: vec![
                OcrTextBlock {
                    text: "你好".to_string(),
                    confidence: 0.98,
                    box_points: TextBox {
                        x0: 0.0, y0: 0.0, x1: 100.0, y1: 0.0,
                        x2: 100.0, y2: 30.0, x3: 0.0, y3: 30.0,
                    },
                },
                OcrTextBlock {
                    text: "世界".to_string(),
                    confidence: 0.95,
                    box_points: TextBox {
                        x0: 0.0, y0: 30.0, x1: 100.0, y1: 30.0,
                        x2: 100.0, y2: 60.0, x3: 0.0, y3: 60.0,
                    },
                },
            ],
            output_path: std::path::PathBuf::from("output.txt"),
        };
        assert_eq!(result.full_text(), "你好\n世界");
    }
}
