//! OCR 子命令
//!
//! 依赖说明：仅通过 `votex-app` 用例与仓储 trait 工作，
//! 不感知 SQLite，也不感知 `votex-infra`。

use anyhow::Result;
use std::path::Path;
use std::sync::Arc;
use votex_app::use_case::ocr_use_case::OcrUseCase;
use votex_domain::repository::OcrTaskRepository;
use votex_domain::shared::value_object::TaskId;

/// OCR 任务仓储（数据库不可用时为 `None`）
pub type SharedOcrRepo = Option<Arc<dyn OcrTaskRepository>>;

/// 执行单图 OCR 识别
pub fn run_ocr(
    input: &str,
    output: Option<&str>,
    format: &str,
    no_cls: bool,
    engine: &str,
    models_dir: &Path,
    repo: SharedOcrRepo,
) -> Result<()> {
    let input_path = Path::new(input);
    if !input_path.exists() {
        anyhow::bail!("输入图片不存在: {}", input);
    }

    // 初始化 UseCase
    let mut use_case = OcrUseCase::new();

    // 如果提供了仓储，启用持久化
    if let Some(repo) = repo {
        use_case = use_case.with_repo(repo);
    }

    // 加载引擎（统一传入 models/ 目录，引擎会自动定位子目录）
    use_case.load_engine(models_dir, engine)?;

    // 确定输出路径
    let output_path = match output {
        Some(p) => Path::new(p).to_path_buf(),
        None => {
            // 默认：输入文件名 + 格式扩展名
            let ext = match format {
                "json" => "json",
                "md" | "markdown" => "md",
                _ => "txt",
            };
            input_path.with_extension(ext)
        }
    };

    // 执行识别
    let task = use_case.recognize_single(
        input_path,
        &output_path,
        format,
        no_cls,
        None, // 进度回调（CLI 模式暂不启用实时回调）
    )?;

    println!(
        "OCR 识别完成: {}",
        output_path.display()
    );
    println!("  任务 ID: {}", task.id);
    println!("  文本区域: {}", task.progress.current_block);
    println!("  状态: {:?}", task.status);

    Ok(())
}

/// 执行批量 OCR 识别
///
/// `repo` 为 `None` 时不持久化，进度仅打印。
pub fn run_batch_ocr(
    inputs: &[String],
    output_dir: &str,
    format: &str,
    no_cls: bool,
    engine: &str,
    max_concurrency: usize,
    models_dir: &Path,
    repo: SharedOcrRepo,
) -> Result<()> {
    let output_path = Path::new(output_dir);

    // 收集输入路径
    let image_paths: Vec<std::path::PathBuf> = inputs
        .iter()
        .map(|s| Path::new(s).to_path_buf())
        .collect();

    // 初始化 UseCase
    let mut use_case = OcrUseCase::new();

    if let Some(repo) = repo {
        use_case = use_case.with_repo(repo);
    }

    use_case.load_engine(models_dir, engine)?;

    // 执行批量识别
    let task = use_case.recognize_batch(
        &image_paths,
        output_path,
        format,
        no_cls,
        max_concurrency,
        None,
    )?;

    println!("批量 OCR 完成:");
    println!("  任务 ID: {}", task.id);
    println!("  总页数: {}", task.progress.total_pages);
    println!("  完成: {}", task.completed_pages());
    println!("  失败: {}", task.failed_pages());
    println!("  输出目录: {}", output_dir);

    Ok(())
}

/// 列出现有 OCR 任务
pub fn list_ocr_tasks(repo: SharedOcrRepo) -> Result<()> {
    let repo = match repo {
        Some(r) => r,
        None => {
            tracing::warn!("数据库不可用，无法列出 OCR 任务");
            return Ok(());
        }
    };

    let tasks = OcrUseCase::list_tasks(repo.as_ref());

    if tasks.is_empty() {
        println!("暂无 OCR 任务记录");
        return Ok(());
    }

    println!("OCR 任务列表:");
    println!("{:<40} {:<20} {:<10} {:<10}", "任务 ID", "名称", "状态", "页数");
    println!("{}", "-".repeat(80));
    for task in &tasks {
        let status_str = format!("{:?}", task.status);
        println!(
            "{:<40} {:<20} {:<10} {:<10}",
            task.id.as_str(),
            task.name,
            status_str,
            task.pages.len(),
        );
    }

    Ok(())
}

/// 查看 OCR 任务详情
pub fn show_ocr_task(repo: SharedOcrRepo, task_id_str: &str) -> Result<()> {
    let repo = match repo {
        Some(r) => r,
        None => {
            tracing::warn!("数据库不可用，无法查看 OCR 任务");
            return Ok(());
        }
    };

    let task_id = TaskId::from_string(task_id_str.to_string());
    let task = repo
        .find_by_id(&task_id)
        .ok_or_else(|| anyhow::anyhow!("任务不存在: {}", task_id_str))?;

    println!("OCR 任务详情:");
    println!("  ID: {}", task.id);
    println!("  名称: {}", task.name);
    println!("  状态: {:?}", task.status);
    println!("  总页数: {}", task.pages.len());

    for page in &task.pages {
        let status_str = match &page.status {
            votex_domain::ocr::value_object::PageStatus::Pending => "等待中",
            votex_domain::ocr::value_object::PageStatus::Processing => "处理中",
            votex_domain::ocr::value_object::PageStatus::Completed => "已完成",
            votex_domain::ocr::value_object::PageStatus::Failed(msg) => {
                // 截断长错误
                if msg.chars().count() > 30 {
                    &msg.chars().take(30).collect::<String>()
                } else {
                    msg.as_str()
                }
            }
            votex_domain::ocr::value_object::PageStatus::Skipped => "已跳过",
        };
        println!(
            "  第 {} 页: {} (区块: {}, 置信度: {:.1}%)",
            page.index + 1,
            status_str,
            page.blocks.as_ref().map(|b| b.len()).unwrap_or(0),
            page.confidence.unwrap_or(0.0) * 100.0,
        );
    }

    Ok(())
}

/// 删除 OCR 任务
pub fn delete_ocr_task(repo: SharedOcrRepo, task_id_str: &str) -> Result<()> {
    let repo = match repo {
        Some(r) => r,
        None => {
            tracing::warn!("数据库不可用，无法删除 OCR 任务");
            return Ok(());
        }
    };

    let task_id = TaskId::from_string(task_id_str.to_string());
    OcrUseCase::delete_task(repo.as_ref(), &task_id)?;
    println!("OCR 任务已删除: {}", task_id_str);
    Ok(())
}
