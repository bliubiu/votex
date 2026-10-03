use serde::{Deserialize, Serialize};

/// OCR 请求 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrRequest {
    /// 输入图片路径（单张或多张，用逗号分隔）
    pub input_path: String,
    /// 输出文件或目录路径
    pub output_path: String,
    /// 输出格式 (txt / json / md / docx)
    pub output_format: String,
    /// 是否禁用方向分类
    pub no_cls: bool,
    /// 最大并发数
    pub max_concurrency: usize,
    /// 批量模式：输入路径是目录
    pub batch_mode: bool,
}

/// OCR 响应 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrResponse {
    pub task_id: String,
    pub output_path: String,
    pub blocks_count: usize,
    pub avg_confidence: f32,
    pub duration_ms: u64,
}

/// OCR 任务列表项 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrTaskItem {
    pub task_id: String,
    pub name: String,
    pub status: String,
    pub total_pages: usize,
    pub completed_pages: usize,
    pub failed_pages: usize,
    pub created_at: String,
}

/// OCR 页面状态 DTO
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrPageItem {
    pub index: usize,
    pub image_path: String,
    pub status: String,
    pub confidence: Option<f32>,
    pub block_count: Option<usize>,
    pub retry_count: u8,
    pub error_message: Option<String>,
}
