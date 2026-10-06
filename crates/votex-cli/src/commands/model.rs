use anyhow::Result;
use std::path::Path;
use std::sync::Arc;
use votex_app::use_case::model_use_case::ModelUseCase;
use votex_domain::model::value_object::ModelId;
use votex_app::platform::{download, registry};
use votex_domain::model::entity::DownloadRecord;
use votex_domain::repository::DownloadRepository;

use crate::commands::root::ModelAction;

/// 处理 model 子命令
pub fn handle(
    action: &ModelAction,
    models_dir: &Path,
    download_repo: Option<Arc<dyn DownloadRepository>>,
) -> Result<()> {
    // 加载模型清单（走统一路径解析，不依赖 cwd）
    let registry_entries = registry::load_registry_entries();
    if registry_entries.is_empty() {
        tracing::warn!("模型清单为空，model 子命令可用功能受限");
    }

    match action {
        ModelAction::List => list_models(models_dir, &registry_entries),
        ModelAction::Download { model_id, mirror } => {
            download_model(model_id, mirror, models_dir, registry_entries, download_repo)
        }
        ModelAction::Import { model_id, path } => {
            import_model(model_id, path, models_dir, registry_entries)
        }
        ModelAction::Verify { model_id } => verify_model(model_id),
        ModelAction::Remove { model_id } => remove_model(model_id),
    }
}

fn list_models(models_dir: &Path, registry: &[votex_domain::model::registry::ModelRegistryEntry]) -> Result<()> {
    let use_case = ModelUseCase::new(models_dir, registry.to_vec());
    let models = use_case.list_models();

    println!("可用模型列表：");
    println!("{:<15} {:<20} {:<10} {:<10}", "ID", "名称", "类型", "状态");
    println!("{}", "-".repeat(60));
    for model in &models {
        // 运行时依赖（ORT 动态库）也列出来 —— 它同样是首次启动必须下载的项
        let kind_str = model.kind.display_name();
        let status_str = format!("{:?}", model.status);
        println!(
            "{:<15} {:<20} {:<10} {:<10}",
            model.id, model.name, kind_str, status_str
        );
    }
    Ok(())
}

fn download_model(
    model_id: &str,
    mirror: &str,
    models_dir: &Path,
    registry: Vec<votex_domain::model::registry::ModelRegistryEntry>,
    download_repo: Option<Arc<dyn DownloadRepository>>,
) -> Result<()> {
    let id = ModelId::new(model_id);

    // 镜像优先级映射与 GUI 共用，避免两处行为不一致
    let priority = download::mirror_priority(mirror);

    let use_case = ModelUseCase::new(models_dir, registry);

    let repo_for_progress = download_repo.clone();
    let model_id_for_progress = model_id.to_string();
    let download_started = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let started = download_started.clone();

    let on_progress = Some(Box::new(move |downloaded: u64, total: u64, source: &str, file_name: &str| {
        if total > 0 {
            let pct = (downloaded as f64 / total as f64) * 100.0;
            print!("\r下载中 [{:>5.1}%] {} / {} 字节 (源: {}, 文件: {})    ",
                pct, downloaded, total, source, file_name);
        } else {
            print!("\r下载中 {} 字节 (源: {}, 文件: {})    ", downloaded, source, file_name);
        }
        use std::io::Write;
        std::io::stdout().flush().ok();

        // 在闭包内部追踪下载状态，不依赖外部变量
        if let Some(ref r) = repo_for_progress {
            if !started.swap(true, std::sync::atomic::Ordering::SeqCst) {
                let _ = r.upsert(&DownloadRecord {
                    model_id: model_id_for_progress.clone(),
                    file_name: file_name.to_string(),
                    url: "unknown".to_string(),
                    bytes_downloaded: downloaded,
                    total_bytes: total,
                    status: "Downloading".to_string(),
                    error_message: None,
                    created_at: "0".to_string(),
                    updated_at: "0".to_string(),
                });
            } else {
                let _ = r.update_progress(&model_id_for_progress, file_name, downloaded, total);
            }
        }
    }) as download::ProgressFn);

    println!("开始下载模型: {} (优先级: {:?})", model_id, priority);
    let model = use_case.download_model(&id, &priority, on_progress)?;

    // 用 clone 后的副本来标记完成
    if let Some(ref r) = download_repo {
        let _ = r.upsert(&DownloadRecord {
            model_id: model_id.to_string(),
            file_name: "completed".to_string(),
            url: "unknown".to_string(),
            bytes_downloaded: 0,
            total_bytes: 0,
            status: "Completed".to_string(),
            error_message: None,
            created_at: "0".to_string(),
            updated_at: "0".to_string(),
        });
    }

    println!("\n模型 {} 下载完成，状态: {:?}", model.id, model.status);
    Ok(())
}

fn import_model(
    model_id: &str,
    path: &str,
    models_dir: &Path,
    registry: Vec<votex_domain::model::registry::ModelRegistryEntry>,
) -> Result<()> {
    let id = ModelId::new(model_id);
    let use_case = ModelUseCase::new(models_dir, registry);
    let model = use_case.import_model(&id, std::path::Path::new(path))?;
    println!("模型 {} 导入成功，状态: {:?}", model.id, model.status);
    Ok(())
}

fn verify_model(model_id: &str) -> Result<()> {
    let id = ModelId::new(model_id);
    println!("校验模型: {}（功能待完善，需持久化模型状态）", id);
    Ok(())
}

fn remove_model(model_id: &str) -> Result<()> {
    let id = ModelId::new(model_id);
    println!("删除模型: {}（功能待完善，需持久化模型状态）", id);
    Ok(())
}
