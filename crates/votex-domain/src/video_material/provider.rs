use crate::video_material::value_object::VideoMaterial;
use std::path::{Path, PathBuf};

/// 视频素材提供者接口
pub trait VideoMaterialProvider: Send + Sync {
    /// 提供者名称
    fn name(&self) -> &str;

    /// 搜索视频素材
    fn search(
        &self,
        query: &str,
        per_page: u32,
        orientation: &str,
        min_duration: u32,
        max_duration: u32,
    ) -> Result<Vec<VideoMaterial>, MaterialError>;

    /// 下载视频素材到本地
    fn download(&self, material: &VideoMaterial, dest_dir: &Path) -> Result<Option<PathBuf>, MaterialError>;
}

/// 素材搜索错误
#[derive(Debug, thiserror::Error)]
pub enum MaterialError {
    #[error("API 调用失败: {0}")]
    ApiError(String),

    #[error("API Key 未配置")]
    ApiKeyNotConfigured,

    #[error("下载失败: {0}")]
    DownloadFailed(String),

    #[error("不支持的素材源: {0}")]
    UnsupportedSource(String),

    #[error("网络错误: {0}")]
    NetworkError(String),
}
