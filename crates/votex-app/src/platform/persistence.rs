//! 持久化门面：数据库连接与各 SQLite 仓储构造
//!
//! 表现层通过本模块拿到的始终是 **domain trait object**，
//! 因此 `votex-cli` / `votex-gui` 不需要 `rusqlite` 依赖，
//! 也不需要知道底层表结构。

use std::path::Path;
use std::sync::{Arc, Mutex};

use votex_domain::error::DomainError;
use votex_domain::repository::{
    DownloadRepository, OcrTaskRepository, PipelineRepository, PlaybackRepository,
};

/// 共享数据库连接类型
///
/// 与 infra 层保持同一类型，便于在门面内做 clone。
pub type SharedConnection = Arc<Mutex<rusqlite::Connection>>;

/// 打开数据库连接（自动建表、WAL、`foreign_keys`）
pub fn open_database(db_path: &Path) -> Result<SharedConnection, DomainError> {
    Ok(votex_infra::persistence::db::open_database(db_path)?)
}

/// 打开内存数据库（仅测试用）
pub fn open_in_memory_database() -> Result<SharedConnection, DomainError> {
    Ok(votex_infra::persistence::db::open_in_memory_database()?)
}

/// 构造下载进度仓储
pub fn sqlite_download_repository(conn: SharedConnection) -> impl DownloadRepository {
    votex_infra::persistence::download_repo::SqliteDownloadRepository::new(conn)
}

/// 构造流水线仓储
pub fn sqlite_pipeline_repository(conn: SharedConnection) -> impl PipelineRepository {
    votex_infra::persistence::pipeline_repo::SqlitePipelineRepository::new(conn)
}

/// 构造 OCR 任务仓储
pub fn sqlite_ocr_repository(conn: SharedConnection) -> impl OcrTaskRepository {
    votex_infra::persistence::ocr_repo::SqliteOcrTaskRepository::new(conn)
}

/// 构造播放进度仓储
pub fn sqlite_playback_repository(conn: SharedConnection) -> impl PlaybackRepository {
    votex_infra::persistence::playback_repo::SqlitePlaybackRepository::new(conn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 下载仓储_经trait对象可用() {
        let repo: Arc<dyn DownloadRepository> =
            Arc::new(sqlite_download_repository(open_in_memory_database().unwrap()));

        use votex_domain::model::entity::DownloadRecord;
        repo.upsert(&DownloadRecord {
            model_id: "kokoro-82m".into(),
            file_name: "model.onnx".into(),
            url: "https://example.com/m.onnx".into(),
            bytes_downloaded: 0,
            total_bytes: 100,
            status: "Pending".into(),
            error_message: None,
            created_at: "0".into(),
            updated_at: "0".into(),
        })
        .unwrap();

        assert_eq!(repo.find_by_model("kokoro-82m").len(), 1);
        repo.update_progress("kokoro-82m", "model.onnx", 60, 100).unwrap();
        assert_eq!(
            repo.find_by_model("kokoro-82m")[0].bytes_downloaded,
            60
        );
        repo.delete_by_model("kokoro-82m").unwrap();
        assert!(repo.find_by_model("kokoro-82m").is_empty());
    }

    #[test]
    fn 播放仓储_经trait对象可用() {
        let repo: Arc<dyn PlaybackRepository> =
            Arc::new(sqlite_playback_repository(open_in_memory_database().unwrap()));

        assert!(repo.get("a.wav").unwrap().is_none());
        repo.save("a.wav", 12.5, 300.0).unwrap();
        let p = repo.get("a.wav").unwrap().unwrap();
        assert!((p.position_sec - 12.5).abs() < 1e-6);
        repo.clear("a.wav").unwrap();
        assert!(repo.get("a.wav").unwrap().is_none());
    }
}
