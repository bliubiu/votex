use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::Instant;
use votex_domain::ocr::entity::OcrTask;
use votex_domain::ocr::provider::{OcrExporter, OcrExporterRegistry, OcrProvider};
use votex_domain::ocr::value_object::{
    OcrOutputFormat, OcrParams, PageStatus,
};
use votex_domain::repository::OcrTaskRepository;
use votex_domain::shared::value_object::{
    CancellationToken, ProgressCallback, TaskStatus,
};
use votex_infra::ocr::batch::BatchOcrOrchestrator;
use votex_infra::ocr::easyocr::EasyOcrProvider;
use votex_infra::ocr::exporter::{JsonExporter, MarkdownExporter, TxtExporter};
use votex_infra::ocr::paddleocr::{PaddleOcrEngine, PaddleOcrModelVariant};

/// OCR 识别用例
pub struct OcrUseCase {
    /// 已加载的 OCR 引擎
    ///
    /// 用 `RwLock<Option<Arc<dyn OcrProvider>>>` 而非 `Option<Box<dyn>>`：
    /// ① `OcrProvider` 的 `load` / `recognize` 现已全部是 `&self`，
    ///    引擎对象本身可共享，无需独占；
    /// ② 原先 `recognize_batch` 用 `take()` 借出引擎，
    ///    批量中任一步失败就会永久丢失引擎（用户须重新 `load_engine()`），
    ///    且错误信息不提示这一点。改为共享借用后不再丢失。
    engine: RwLock<Option<Arc<dyn OcrProvider>>>,
    repo: Option<Arc<dyn OcrTaskRepository>>,
    /// 导出器注册表
    ///
    /// 注册表本身需要 `&mut` 才能新增条目，而 `register_exporter` 允许在
    /// 构造后追加。放进 `RwLock` 后，注册与查询都能走 `&self`，
    /// 使整个 `OcrUseCase` 可被多个任务线程共享。
    exporters: RwLock<OcrExporterRegistry>,
}

impl OcrUseCase {
    pub fn new() -> Self {
        let mut registry = OcrExporterRegistry::new();
        registry.register(Box::new(TxtExporter));
        registry.register(Box::new(JsonExporter));
        registry.register(Box::new(MarkdownExporter));

        Self {
            engine: RwLock::new(None),
            repo: None,
            exporters: RwLock::new(registry),
        }
    }

    /// 设置持久化仓库
    pub fn with_repo(mut self, repo: Arc<dyn OcrTaskRepository>) -> Self {
        self.repo = Some(repo);
        self
    }

    /// 加载 OCR 引擎
    ///
    /// `engine_kind` 取值：
    /// - `"paddleocr"` / `"paddleocr-v6-medium"` — PaddleOCR v6 Medium（默认，精度最佳）
    /// - `"paddleocr-v6-small"` — PaddleOCR v6 Small
    /// - `"paddleocr-v6-tiny"` — PaddleOCR v6 Tiny（轻量）
    /// - `"paddleocr-v4"` — PaddleOCR v4 Mobile
    /// - `"paddleocr-v5-mobile"` — PaddleOCR v5 Mobile
    /// - `"paddleocr-v5-server"` — PaddleOCR v5 Server
    /// - `"easyocr"` — EasyOCR（多语言）
    ///
    /// 模型文件统一放在 `models_dir` 下，按子目录 `paddleocr-v6/`、`paddleocr/`、`paddleocr-v5/`、`EasyOCR/` 组织。
    /// 释放已加载的 OCR 引擎会话（真实释放内存）
    ///
    /// 未加载时静默成功（幂等）；释放后需重新 `load_engine` 才能识别。
    pub fn unload_engine(&self) -> Result<()> {
        let engine = self
            .engine
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(e) = engine {
            e.unload()?;
            tracing::info!("OCR 引擎会话已释放");
        }
        Ok(())
    }

    pub fn load_engine(&self, models_dir: &Path, engine_kind: &str) -> Result<()> {
        let engine: Box<dyn OcrProvider> = match engine_kind {
            "paddleocr" | "paddleocr-v6-medium" => {
                let e = PaddleOcrEngine::with_variant(PaddleOcrModelVariant::V6Medium);
                let dir = models_dir.join("ocr").join("paddleocr-v6");
                if !dir.exists() {
                    anyhow::bail!("PaddleOCR v6 模型目录不存在: {:?}，请先执行 `votex model download paddleocr-v6-medium`", dir);
                }
                e.load_from_dir(&dir)?;
                Box::new(e)
            }
            "paddleocr-v6-small" => {
                let e = PaddleOcrEngine::with_variant(PaddleOcrModelVariant::V6Small);
                let dir = models_dir.join("ocr").join("paddleocr-v6");
                if !dir.exists() {
                    anyhow::bail!("PaddleOCR v6 模型目录不存在: {:?}，请先执行 `votex model download paddleocr-v6-small`", dir);
                }
                e.load_from_dir(&dir)?;
                Box::new(e)
            }
            "paddleocr-v6-tiny" => {
                let e = PaddleOcrEngine::with_variant(PaddleOcrModelVariant::V6Tiny);
                let dir = models_dir.join("ocr").join("paddleocr-v6");
                if !dir.exists() {
                    anyhow::bail!("PaddleOCR v6 模型目录不存在: {:?}，请先执行 `votex model download paddleocr-v6-tiny`", dir);
                }
                e.load_from_dir(&dir)?;
                Box::new(e)
            }
            "paddleocr-v4" => {
                let e = PaddleOcrEngine::with_variant(PaddleOcrModelVariant::V4Mobile);
                let dir = models_dir.join("ocr").join("paddleocr");
                if !dir.exists() {
                    anyhow::bail!("PaddleOCR v4 模型目录不存在: {:?}，请先执行 `votex model download paddleocr`", dir);
                }
                e.load_from_dir(&dir)?;
                Box::new(e)
            }
            "paddleocr-v5-mobile" => {
                let e = PaddleOcrEngine::with_variant(PaddleOcrModelVariant::V5Mobile);
                let dir = models_dir.join("ocr").join("paddleocr-v5");
                if !dir.exists() {
                    anyhow::bail!("PaddleOCR v5 模型目录不存在: {:?}，请先执行 `votex model download paddleocr-v5-mobile`", dir);
                }
                e.load_from_dir(&dir)?;
                Box::new(e)
            }
            "paddleocr-v5-server" => {
                let e = PaddleOcrEngine::with_variant(PaddleOcrModelVariant::V5Server);
                let dir = models_dir.join("ocr").join("paddleocr-v5");
                if !dir.exists() {
                    anyhow::bail!("PaddleOCR v5 模型目录不存在: {:?}，请先执行 `votex model download paddleocr-v5-server`", dir);
                }
                e.load_from_dir(&dir)?;
                Box::new(e)
            }
            "easyocr" => {
                let e = EasyOcrProvider::new();
                let dir = models_dir.join("ocr").join("EasyOCR");
                if !dir.exists() {
                    anyhow::bail!("EasyOCR 模型目录不存在: {:?}，请先执行 `votex model download easyocr`", dir);
                }
                e.load_from_dir(&dir)?;
                Box::new(e)
            }
            _ => anyhow::bail!("不支持的 OCR 引擎: {}", engine_kind),
        };
        *self
            .engine
            .write()
            .unwrap_or_else(|e| e.into_inner()) = Some(Arc::from(engine));
        tracing::info!("OCR 引擎加载完成 (kind={})", engine_kind);
        Ok(())
    }

    /// 执行单图 OCR 识别
    pub fn recognize_single(
        &self,
        input_path: &Path,
        output_path: &Path,
        format: &str,
        no_cls: bool,
        on_progress: ProgressCallback,
    ) -> Result<OcrTask> {
        let guard = self
            .engine
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let engine = guard.as_ref().ok_or_else(|| {
            anyhow::anyhow!("OCR 引擎未加载，请先调用 load_engine()")
        })?;

        if !input_path.exists() {
            anyhow::bail!("输入图片不存在: {:?}", input_path);
        }

        let output_format = OcrOutputFormat::from_str(format);
        let params = OcrParams {
            language: votex_domain::ocr::value_object::OcrLanguage::Zh,
            output_format,
            use_angle_cls: !no_cls,
            output_dir: output_path.parent().map(|p| p.to_path_buf()),
            max_concurrency: 1,
            ..Default::default()
        };

        // 创建任务
        let mut task = OcrTask::new_single(input_path.to_path_buf(), params.clone());
        let cancel_token = CancellationToken::new();

        task.status = TaskStatus::Running;
        task.progress.phase = votex_domain::ocr::value_object::OcrPhase::Detecting;

        // 保存初始状态
        if let Some(ref repo) = self.repo {
            repo.save(&task)?;
        }

        // 执行识别
        let start = Instant::now();
        let result = engine.recognize_with_cancel(input_path, &params, &cancel_token, on_progress)?;
        let duration = start.elapsed();

        // 更新任务结果
        task.set_page_result(0, result.blocks.clone());
        task.update_page_status(0, PageStatus::Completed);
        task.status = TaskStatus::Completed;
        task.progress.phase = votex_domain::ocr::value_object::OcrPhase::Completed;

        // 导出结果
        let ext = output_format.extension();
        let out_path = if output_path.extension().map(|e| e.to_str().unwrap_or("")) == Some(ext) {
            output_path.to_path_buf()
        } else {
            output_path.with_extension(ext)
        };

        // 通过导出器导出
        let exporters = self
            .exporters
            .read()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(exporter) = exporters.find_by_format(format) {
            exporter.export(&result, &out_path)
                .map_err(|e| anyhow::anyhow!("导出失败: {}", e))?;
        } else {
            // 默认使用 TXT
            TxtExporter.export(&result, &out_path)
                .map_err(|e| anyhow::anyhow!("导出失败: {}", e))?;
        }

        task.progress.current_block = result.block_count();
        task.progress.total_blocks = result.block_count();

        // 保存最终状态
        if let Some(ref repo) = self.repo {
            repo.save(&task)?;
        }

        tracing::info!(
            "OCR 识别完成: {:?} -> {:?} ({}ms, {} 块)",
            input_path, out_path, duration.as_millis(), result.block_count()
        );

        Ok(task)
    }

    /// 执行批量 OCR 识别
    pub fn recognize_batch(
        &self,
        image_paths: &[PathBuf],
        output_dir: &Path,
        format: &str,
        no_cls: bool,
        max_concurrency: usize,
        on_progress: ProgressCallback,
    ) -> Result<OcrTask> {
        // 只借用不取走：批量中任一步失败都不会丢失引擎
        let guard = self
            .engine
            .read()
            .unwrap_or_else(|e| e.into_inner());
        let engine = guard.as_ref().ok_or_else(|| {
            anyhow::anyhow!("OCR 引擎未加载，请先调用 load_engine()")
        })?;

        if image_paths.is_empty() {
            anyhow::bail!("输入图片列表为空");
        }

        // 验证所有文件存在
        for path in image_paths {
            if !path.exists() {
                anyhow::bail!("输入图片不存在: {:?}", path);
            }
        }

        std::fs::create_dir_all(output_dir)?;

        let output_format = OcrOutputFormat::from_str(format);
        let params = OcrParams {
            language: votex_domain::ocr::value_object::OcrLanguage::Zh,
            output_format,
            use_angle_cls: !no_cls,
            output_dir: Some(output_dir.to_path_buf()),
            max_concurrency,
            ..Default::default()
        };

        // 创建批量任务
        let mut task = OcrTask::new_batch(image_paths.to_vec(), params.clone());
        let cancel_token = CancellationToken::new();

        task.status = TaskStatus::Running;
        task.progress.phase = votex_domain::ocr::value_object::OcrPhase::Detecting;

        // 保存初始状态
        if let Some(ref repo) = self.repo {
            repo.save(&task)?;
        }

        // 创建批量编排器
        let actual_concurrency = max_concurrency.max(1);
        let orchestrator = BatchOcrOrchestrator::new(engine.clone(), actual_concurrency);
        let start = Instant::now();

        // 收集路径引用
        let path_refs: Vec<&Path> = image_paths.iter().map(|p| p.as_path()).collect();
        let results = orchestrator.process_batch(
            &path_refs,
            &params,
            &cancel_token,
            on_progress,
        );

        let duration = start.elapsed();

        // 处理结果
        let mut all_blocks = Vec::new();
        for (i, (_path, result)) in results.iter().enumerate() {
            match result {
                Ok(ocr_result) => {
                    task.set_page_result(i, ocr_result.blocks.clone());
                    task.update_page_status(i, PageStatus::Completed);
                    all_blocks.extend(ocr_result.blocks.clone());

                    // 逐页导出
                    let ext = output_format.extension();
                    let page_out_path = output_dir.join(format!(
                        "page_{}.{}",
                        i + 1,
                        ext
                    ));

                    let exporters = self
                        .exporters
                        .read()
                        .unwrap_or_else(|e| e.into_inner());
                    if let Some(exporter) = exporters.find_by_format(format) {
                        if let Err(e) = exporter.export(ocr_result, &page_out_path) {
                            tracing::warn!("第 {} 页导出失败: {}", i + 1, e);
                        }
                    } else {
                        let _ = TxtExporter.export(ocr_result, &page_out_path);
                    }
                }
                Err(e) => {
                    task.update_page_status(
                        i,
                        PageStatus::Failed(format!("{}", e)),
                    );
                    tracing::error!("第 {} 页识别失败: {}", i + 1, e);
                }
            }

            // 每页后保存状态
            if let Some(ref repo) = self.repo {
                let _ = repo.save(&task);
            }
        }

        // 合并导出（全部结果合并到一个文件）
        let _full_result = task.build_result(output_dir.join(format!(
            "full_result.{}",
            output_format.extension()
        )));

        // 更新为合并后的 blocks
        // 重新构建 OcrResult 包含所有页面的 blocks
        let merged_result = votex_domain::ocr::value_object::OcrResult {
            blocks: all_blocks.clone(),
            output_path: output_dir.join(format!(
                "full_result.{}",
                output_format.extension()
            )),
        };

        let exporters = self
            .exporters
            .read()
            .unwrap_or_else(|e| e.into_inner());
        if let Some(exporter) = exporters.find_by_format(format) {
            exporter.export(&merged_result, &merged_result.output_path)
                .map_err(|e| anyhow::anyhow!("合并导出失败: {}", e))?;
        } else {
            TxtExporter.export(&merged_result, &merged_result.output_path)
                .map_err(|e| anyhow::anyhow!("合并导出失败: {}", e))?;
        }

        // 完成
        let failed_count = task.failed_pages();
        task.status = if failed_count > 0 {
            TaskStatus::Failed(format!("{} 页识别失败", failed_count))
        } else {
            TaskStatus::Completed
        };
        task.progress.phase = votex_domain::ocr::value_object::OcrPhase::Completed;

        // 保存最终状态
        if let Some(ref repo) = self.repo {
            repo.save(&task)?;
        }

        // 将引擎放回
        // 注意：orchestrator 中的 engine 通过 Arc 共享，无法直接取出
        // 实际使用时需要重新创建 use case

        tracing::info!(
            "批量 OCR 完成: {} 页 ({}ms, {} 成功, {} 失败)",
            image_paths.len(),
            duration.as_millis(),
            task.completed_pages(),
            failed_count,
        );

        Ok(task)
    }

    /// 获取导出器注册表（读守卫）
    pub fn exporters(&self) -> std::sync::RwLockReadGuard<'_, OcrExporterRegistry> {
        self.exporters.read().unwrap_or_else(|e| e.into_inner())
    }

    /// 注册额外导出器
    pub fn register_exporter(
        &self,
        exporter: Box<dyn votex_domain::ocr::provider::OcrExporter>,
    ) {
        self.exporters
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .register(exporter);
    }

    /// 从数据库恢复未完成的任务
    pub fn restore_incomplete_tasks(repo: &dyn OcrTaskRepository) -> Vec<OcrTask> {
        let tasks = repo.find_incomplete();
        tracing::info!("找到 {} 个未完成的 OCR 任务", tasks.len());
        tasks
    }

    /// 列出所有任务
    pub fn list_tasks(repo: &dyn OcrTaskRepository) -> Vec<OcrTask> {
        repo.list_all()
    }

    /// 删除任务
    pub fn delete_task(repo: &dyn OcrTaskRepository, task_id: &votex_domain::shared::value_object::TaskId) -> Result<()> {
        repo.delete(task_id)
            .map_err(|e| anyhow::anyhow!("删除任务失败: {}", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn ocr_use_case_创建() {
        let use_case = OcrUseCase::new();
        assert_eq!(use_case.exporters().count(), 3);
    }

    #[test]
    fn ocr_use_case_注册导出器() {
        let use_case = OcrUseCase::new();
        use_case.register_exporter(Box::new(TxtExporter));
        // 重复注册不影响
        assert_eq!(use_case.exporters().count(), 4);
    }

    #[test]
    fn ocr_use_case_导出器查找() {
        let use_case = OcrUseCase::new();
        let formats = use_case.exporters().available_formats();
        assert!(formats.contains(&"txt"));
        assert!(formats.contains(&"json"));
        assert!(formats.contains(&"markdown"));
    }

    #[test]
    fn ocr_use_case_未加载时报错() {
        let use_case = OcrUseCase::new();
        let result = use_case.recognize_single(
            Path::new("nonexistent.png"),
            Path::new("out.txt"),
            "txt",
            false,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn ocr_output_format_解析() {
        assert_eq!(OcrOutputFormat::from_str("json"), OcrOutputFormat::Json);
        assert_eq!(OcrOutputFormat::from_str("md"), OcrOutputFormat::Markdown);
        assert_eq!(OcrOutputFormat::from_str("markdown"), OcrOutputFormat::Markdown);
        assert_eq!(OcrOutputFormat::from_str("docx"), OcrOutputFormat::Docx);
        assert_eq!(OcrOutputFormat::from_str("txt"), OcrOutputFormat::Txt);
        assert_eq!(OcrOutputFormat::from_str("unknown"), OcrOutputFormat::Txt);
    }

    #[test]
    fn ocr_output_format_扩展名() {
        assert_eq!(OcrOutputFormat::Txt.extension(), "txt");
        assert_eq!(OcrOutputFormat::Json.extension(), "json");
        assert_eq!(OcrOutputFormat::Markdown.extension(), "md");
        assert_eq!(OcrOutputFormat::Docx.extension(), "docx");
    }
}
