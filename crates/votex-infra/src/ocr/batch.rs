use std::path::Path;
use std::sync::{Arc, Mutex};
use tokio::sync::Semaphore;
use votex_domain::error::OcrError;
use votex_domain::ocr::provider::OcrProvider;
use votex_domain::ocr::value_object::{OcrParams, OcrResult};
use votex_domain::shared::value_object::{CancellationToken, ProgressCallback, ProgressEvent};

/// 批量 OCR 编排器
/// 使用信号量控制并发推理数量，防止显存/内存溢出
pub struct BatchOcrOrchestrator {
    engine: Arc<Mutex<Box<dyn OcrProvider>>>,
    semaphore: Arc<Semaphore>,
}

impl BatchOcrOrchestrator {
    /// 创建编排器
    /// `max_concurrency`: 最大并发推理数
    pub fn new(engine: Box<dyn OcrProvider>, max_concurrency: usize) -> Self {
        Self {
            engine: Arc::new(Mutex::new(engine)),
            semaphore: Arc::new(Semaphore::new(max_concurrency)),
        }
    }

    /// 并发处理多张图片
    /// 返回 (图片路径, 识别结果) 列表，失败条目包含错误信息
    /// 注意：progress callback 在每页处理完后被调用（而非每个识别块）
    pub fn process_batch<'a>(
        &self,
        image_paths: &[&'a Path],
        params: &OcrParams,
        cancel_token: &CancellationToken,
        on_progress: ProgressCallback,
    ) -> Vec<(&'a Path, Result<OcrResult, OcrError>)> {
        let total = image_paths.len();
        let mut results: Vec<(&'a Path, Result<OcrResult, OcrError>)> = Vec::with_capacity(total);

        // 报告总进度
        if let Some(ref cb) = on_progress {
            cb(ProgressEvent::PhaseChanged {
                phase: "batch_start".to_string(),
            });
            cb(ProgressEvent::PageProgress {
                current: 0,
                total,
                page_index: 0,
            });
        }

        for (i, &path) in image_paths.iter().enumerate() {
            // 检查取消
            if cancel_token.is_cancelled() {
                for j in i..total {
                    results.push((
                        image_paths[j],
                        Err(OcrError::RecognizeFailed("任务已取消".to_string())),
                    ));
                }
                break;
            }

            // 尝试获取信号量许可，失败则串行
            let engine_result = match self.semaphore.try_acquire() {
                Ok(permit) => {
                    let r = {
                        let mut engine = self.engine.lock().unwrap();
                        engine.recognize_with_cancel(path, params, cancel_token, None)
                    };
                    drop(permit);
                    r
                }
                Err(_) => {
                    tracing::debug!("批量 OCR: 达到并发上限，串行处理 (页 {}/{})", i + 1, total);
                    let mut engine = self.engine.lock().unwrap();
                    engine.recognize_with_cancel(path, params, cancel_token, None)
                }
            };
            results.push((path, engine_result));

            // 报告进度
            if let Some(ref cb) = on_progress {
                cb(ProgressEvent::PageProgress {
                    current: i + 1,
                    total,
                    page_index: i,
                });
            }
        }

        if let Some(ref cb) = on_progress {
            cb(ProgressEvent::PhaseChanged {
                phase: "batch_completed".to_string(),
            });
        }

        results
    }

    /// 获取当前引擎
    pub fn engine(&self) -> &Arc<Mutex<Box<dyn OcrProvider>>> {
        &self.engine
    }

    /// 可用信号量许可数
    pub fn available_permits(&self) -> usize {
        self.semaphore.available_permits()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_orchestrator_创建() {
        use crate::ocr::paddleocr::PaddleOcrEngine;
        let engine = PaddleOcrEngine::new();
        let orchestrator = BatchOcrOrchestrator::new(Box::new(engine), 2);
        assert!(!orchestrator.engine().lock().unwrap().is_loaded());
    }

    #[test]
    fn batch_orchestrator_空列表() {
        use crate::ocr::paddleocr::PaddleOcrEngine;
        let engine = PaddleOcrEngine::new();
        let orchestrator = BatchOcrOrchestrator::new(Box::new(engine), 2);
        let params = OcrParams::default();
        let cancel = CancellationToken::new();
        let results = orchestrator.process_batch(&[], &params, &cancel, None);
        assert!(results.is_empty());
    }
}
