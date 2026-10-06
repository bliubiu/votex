//! 路径门面：工作区与模型目录解析
//!
//! # 为什么需要
//!
//! GUI 双击启动时 cwd 可能是任意目录（桌面、资源管理器、快捷方式目标目录），
//! 早期代码大量使用 `std::env::current_dir()` 拼模型路径，
//! 导致「在终端能跑、双击打不开模型」这类问题。
//!
//! `WorkspacePaths` 是路径解析的唯一入口，优先级：
//!
//! 1. `VOTEX_WORKSPACE_DIR`
//! 2. `VOTEX_MODELS_DIR`
//! 3. `application.yml` 中的绝对 `models.storage_path`
//! 4. 可执行文件所在目录
//! 5. 当前目录及最多 6 级父目录
//! 6. 兜底 `models/`

use std::path::PathBuf;

/// 工作区根目录
pub fn workspace_root() -> PathBuf {
    votex_infra::shared::WorkspacePaths::workspace_root()
}

/// 模型根目录
pub fn models_dir() -> PathBuf {
    votex_infra::shared::WorkspacePaths::models_dir()
}

/// 模型清单（registry）目录
pub fn registry_dir() -> PathBuf {
    votex_infra::shared::WorkspacePaths::registry_dir()
}

/// ONNX Runtime 动态库目录
pub fn runtime_dir() -> PathBuf {
    votex_infra::shared::WorkspacePaths::runtime_dir()
}

/// 数据目录（默认 `data/`）
pub fn data_dir() -> PathBuf {
    workspace_root().join("data")
}

/// 默认数据库路径
pub fn default_db_path() -> PathBuf {
    data_dir().join("votex.db")
}

/// 定位模型目录
///
/// 传入模型实体，由 [`ModelFileLocator`] 解析实际目录
/// （含 `qwen3-tts` → `qwen3-tts-0.6b` 这类版本变体回退）。
pub fn locate_model_dir(model: &votex_domain::model::entity::Model) -> anyhow::Result<PathBuf> {
    Ok(votex_infra::shared::ModelFileLocator::locate_model_dir(model)?)
}

/// 在模型目录中按候选文件名查找 ONNX 文件
pub fn find_onnx_file(model_dir: &std::path::Path, candidates: &[&str]) -> anyhow::Result<PathBuf> {
    Ok(votex_infra::shared::ModelFileLocator::find_onnx_file(
        model_dir, candidates,
    )?)
}
