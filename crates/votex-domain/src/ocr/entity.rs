use crate::ocr::value_object::*;
use crate::shared::value_object::TaskId;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// OCR 页面（单页实体）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrPage {
    /// 页索引（从 0 开始）
    pub index: usize,
    /// 图片路径
    pub image_path: PathBuf,
    /// 页面状态
    pub status: PageStatus,
    /// 识别结果块
    pub blocks: Option<Vec<OcrTextBlock>>,
    /// 平均置信度
    pub confidence: Option<f32>,
    /// 重试次数
    pub retry_count: u8,
    /// 错误信息
    pub error_message: Option<String>,
}

impl OcrPage {
    pub fn new(index: usize, image_path: PathBuf) -> Self {
        Self {
            index,
            image_path,
            status: PageStatus::Pending,
            blocks: None,
            confidence: None,
            retry_count: 0,
            error_message: None,
        }
    }

    /// 是否已完成
    pub fn is_completed(&self) -> bool {
        matches!(self.status, PageStatus::Completed)
    }

    /// 是否失败
    pub fn is_failed(&self) -> bool {
        matches!(self.status, PageStatus::Failed(_))
    }

    /// 是否可重试
    pub fn can_retry(&self) -> bool {
        self.retry_count < 3 && self.is_failed()
    }
}

/// OCR 任务（聚合根）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrTask {
    pub id: TaskId,
    pub name: String,
    pub pages: Vec<OcrPage>,
    pub params: OcrParams,
    pub status: crate::shared::value_object::TaskStatus,
    pub result: Option<OcrResult>,
    pub progress: OcrProgress,
    pub created_at: String,
    pub updated_at: String,
}

impl OcrTask {
    /// 创建单页 OCR 任务
    pub fn new_single(image_path: PathBuf, params: OcrParams) -> Self {
        Self {
            id: TaskId::new(),
            name: image_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("ocr_task")
                .to_string(),
            pages: vec![OcrPage::new(0, image_path)],
            params,
            status: crate::shared::value_object::TaskStatus::Queued,
            result: None,
            progress: OcrProgress {
                phase: OcrPhase::Idle,
                current_page: 0,
                total_pages: 1,
                current_block: 0,
                total_blocks: 0,
            },
            created_at: chrono_now_string(),
            updated_at: chrono_now_string(),
        }
    }

    /// 创建多页 OCR 任务
    pub fn new_batch(image_paths: Vec<PathBuf>, params: OcrParams) -> Self {
        let pages: Vec<OcrPage> = image_paths
            .into_iter()
            .enumerate()
            .map(|(i, path)| OcrPage::new(i, path))
            .collect();
        let total = pages.len();
        Self {
            id: TaskId::new(),
            name: format!("batch_ocr_{}", total),
            pages,
            params,
            status: crate::shared::value_object::TaskStatus::Queued,
            result: None,
            progress: OcrProgress {
                phase: OcrPhase::Idle,
                current_page: 0,
                total_pages: total,
                current_block: 0,
                total_blocks: 0,
            },
            created_at: chrono_now_string(),
            updated_at: chrono_now_string(),
        }
    }

    /// 更新页面状态
    pub fn update_page_status(&mut self, page_index: usize, status: PageStatus) {
        if let Some(page) = self.pages.get_mut(page_index) {
            page.status = status;
        }
        self.updated_at = chrono_now_string();
    }

    /// 更新页面识别结果
    pub fn set_page_result(&mut self, page_index: usize, blocks: Vec<OcrTextBlock>) {
        if let Some(page) = self.pages.get_mut(page_index) {
            page.blocks = Some(blocks.clone());
            page.confidence = Some(
                if blocks.is_empty() {
                    0.0
                } else {
                    blocks.iter().map(|b| b.confidence).sum::<f32>() / blocks.len() as f32
                }
            );
        }
        self.updated_at = chrono_now_string();
    }

    /// 设置任务结果（合并所有页面文本）
    pub fn build_result(&self, output_path: PathBuf) -> OcrResult {
        let blocks: Vec<OcrTextBlock> = self
            .pages
            .iter()
            .filter_map(|p| p.blocks.clone())
            .flatten()
            .collect();
        OcrResult {
            blocks,
            output_path,
        }
    }

    /// 已完成页面数
    pub fn completed_pages(&self) -> usize {
        self.pages.iter().filter(|p| p.is_completed()).count()
    }

    /// 失败页面数
    pub fn failed_pages(&self) -> usize {
        self.pages.iter().filter(|p| p.is_failed()).count()
    }
}

fn chrono_now_string() -> String {
    chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn ocr_page_创建() {
        let page = OcrPage::new(0, PathBuf::from("test.png"));
        assert_eq!(page.index, 0);
        assert_eq!(page.status, PageStatus::Pending);
        assert!(page.blocks.is_none());
    }

    #[test]
    fn ocr_page_可重试检查() {
        let mut page = OcrPage::new(0, PathBuf::from("test.png"));
        page.status = PageStatus::Failed("错误".to_string());
        assert!(page.can_retry());
        page.retry_count = 3;
        assert!(!page.can_retry());
    }

    #[test]
    fn ocr_task_单页创建() {
        let task = OcrTask::new_single(
            PathBuf::from("test.png"),
            OcrParams::default(),
        );
        assert_eq!(task.pages.len(), 1);
        assert_eq!(task.pages[0].image_path, PathBuf::from("test.png"));
        assert_eq!(task.status, crate::shared::value_object::TaskStatus::Queued);
    }

    #[test]
    fn ocr_task_多页创建() {
        let paths = vec![
            PathBuf::from("page1.png"),
            PathBuf::from("page2.png"),
            PathBuf::from("page3.png"),
        ];
        let task = OcrTask::new_batch(paths, OcrParams::default());
        assert_eq!(task.pages.len(), 3);
        assert_eq!(task.progress.total_pages, 3);
    }

    #[test]
    fn ocr_task_更新页面状态() {
        let mut task = OcrTask::new_single(
            PathBuf::from("test.png"),
            OcrParams::default(),
        );
        task.update_page_status(0, PageStatus::Processing);
        assert_eq!(task.pages[0].status, PageStatus::Processing);
    }

    #[test]
    fn ocr_task_设置页面结果() {
        let mut task = OcrTask::new_single(
            PathBuf::from("test.png"),
            OcrParams::default(),
        );
        let blocks = vec![OcrTextBlock {
            text: "测试".to_string(),
            confidence: 0.95,
            box_points: TextBox {
                x0: 0.0, y0: 0.0, x1: 100.0, y1: 0.0,
                x2: 100.0, y2: 30.0, x3: 0.0, y3: 30.0,
            },
        }];
        task.set_page_result(0, blocks);
        assert!(task.pages[0].blocks.is_some());
        assert!(task.pages[0].confidence.unwrap() - 0.95 < f32::EPSILON);
    }

    #[test]
    fn ocr_task_构建合并结果() {
        let mut task = OcrTask::new_single(
            PathBuf::from("test.png"),
            OcrParams::default(),
        );
        task.set_page_result(0, vec![
            OcrTextBlock {
                text: "第一行".to_string(),
                confidence: 0.95,
                box_points: TextBox {
                    x0: 0.0, y0: 0.0, x1: 100.0, y1: 0.0,
                    x2: 100.0, y2: 30.0, x3: 0.0, y3: 30.0,
                },
            },
        ]);
        let result = task.build_result(PathBuf::from("out.txt"));
        assert_eq!(result.full_text(), "第一行");
    }
}
