use crate::model::value_object::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 模型实体（聚合根）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: ModelId,
    pub name: String,
    pub kind: ModelKind,
    pub engine: EngineKind,
    pub status: ModelStatus,
    pub file_paths: Vec<FilePath>,
    pub checksums: HashMap<String, Hash>,
    pub size_bytes: u64,
    pub download_source: Option<DownloadSource>,
}

use crate::shared::value_object::{FilePath, Hash};

impl Model {
    pub fn new(id: ModelId, name: &str, kind: ModelKind, engine: EngineKind) -> Self {
        Self {
            id,
            name: name.to_string(),
            kind,
            engine,
            status: ModelStatus::NotDownloaded,
            file_paths: Vec::new(),
            checksums: HashMap::new(),
            size_bytes: 0,
            download_source: None,
        }
    }

    pub fn is_ready(&self) -> bool {
        matches!(self.status, ModelStatus::Ready | ModelStatus::Loaded)
    }

    pub fn is_loaded(&self) -> bool {
        matches!(self.status, ModelStatus::Loaded)
    }
}

/// 下载记录实体
///
/// 追踪单个模型文件的下载进度，支持断点续传。
/// 每个文件一条记录，由 (model_id, file_name) 唯一标识。
///
/// 原先定义在 infra 的 `persistence/download_repo.rs`，属于「领域概念被基础设施绑架」：
/// 表现层为了记录下载进度被迫直接依赖 `votex-infra`。
/// 提升到领域层后，`AppContext` 可以用 `Arc<dyn DownloadRepository>` 持有，
/// CLI / GUI 无需知道底层是 SQLite。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DownloadRecord {
    /// 所属模型 ID（如 "kokoro-82m"）
    pub model_id: String,
    /// 文件名（如 "model.onnx"）
    pub file_name: String,
    /// 下载源 URL
    pub url: String,
    /// 已下载字节数
    pub bytes_downloaded: u64,
    /// 文件总大小（字节），0 表示未知
    pub total_bytes: u64,
    /// 状态：Pending / Downloading / Paused / Completed / Failed
    pub status: String,
    /// 错误信息
    pub error_message: Option<String>,
    /// 创建时间（Unix 时间戳字符串）
    pub created_at: String,
    /// 更新时间（Unix 时间戳字符串）
    pub updated_at: String,
}

impl DownloadRecord {
    /// 下载是否已完成
    pub fn is_completed(&self) -> bool {
        self.status.eq_ignore_ascii_case("completed")
    }

    /// 下载进度百分比；总大小未知时返回 None
    pub fn progress_percent(&self) -> Option<f64> {
        if self.total_bytes == 0 {
            None
        } else {
            Some((self.bytes_downloaded as f64 / self.total_bytes as f64 * 100.0).clamp(0.0, 100.0))
        }
    }
}

/// 播放进度实体
///
/// 按音频文件路径记录上次播放位置，实现「断点续听」。
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PlaybackProgress {
    /// 上次播放位置（秒）
    pub position_sec: f32,
    /// 文件总时长（秒，0 = 未知）
    pub duration_sec: f32,
    /// 更新时间（SQLite `datetime('now','localtime')` 文本）
    pub updated_at: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(status: &str, downloaded: u64, total: u64) -> DownloadRecord {
        DownloadRecord {
            model_id: "kokoro-82m".into(),
            file_name: "model.onnx".into(),
            url: "https://example.com/model.onnx".into(),
            bytes_downloaded: downloaded,
            total_bytes: total,
            status: status.into(),
            error_message: None,
            created_at: "0".into(),
            updated_at: "0".into(),
        }
    }

    #[test]
    fn 下载记录_完成状态判定() {
        assert!(record("Completed", 100, 100).is_completed());
        assert!(record("completed", 100, 100).is_completed());
        assert!(!record("Downloading", 50, 100).is_completed());
    }

    #[test]
    fn 下载记录_进度百分比() {
        assert_eq!(record("Downloading", 50, 100).progress_percent(), Some(50.0));
        // 总大小未知
        assert_eq!(record("Downloading", 50, 0).progress_percent(), None);
        // 已下载超过总大小（服务器给出的 total 偏小）不应超过 100
        assert_eq!(record("Downloading", 150, 100).progress_percent(), Some(100.0));
    }

    #[test]
    fn model_创建与状态检查() {
        let model = Model::new(
            ModelId::new("kokoro-82m"),
            "Kokoro-82M",
            ModelKind::Tts,
            EngineKind::Kokoro,
        );
        assert_eq!(model.name, "Kokoro-82M");
        assert!(!model.is_ready());
        assert!(!model.is_loaded());
    }

    #[test]
    fn model_就绪状态() {
        let mut model = Model::new(
            ModelId::new("kokoro-82m"),
            "Kokoro-82M",
            ModelKind::Tts,
            EngineKind::Kokoro,
        );
        model.status = ModelStatus::Ready;
        assert!(model.is_ready());
        assert!(!model.is_loaded());
    }
}
