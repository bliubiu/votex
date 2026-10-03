//! Pexels 视频素材 API 适配器
//!
//! 通过 Pexels API 搜索免费视频素材。
//! API 文档: https://www.pexels.com/api/documentation/

use crate::api::base::{ApiConfig, ApiError, BaseApiClient};
use std::time::Duration;
use std::path::{Path, PathBuf};
use votex_domain::video_material::provider::{MaterialError, VideoMaterialProvider};
use votex_domain::video_material::value_object::VideoMaterial;

/// Pexels API 基础地址
const PEXELS_BASE_URL: &str = "https://api.pexels.com";

/// Pexels 视频素材提供者
pub struct PexelsProvider {
    client: BaseApiClient,
    api_key: String,
}

impl PexelsProvider {
    /// 创建 Pexels 提供者
    ///
    /// 从环境变量 `PEXELS_API_KEY` 读取 API Key。
    pub fn new() -> Result<Self, MaterialError> {
        let api_key =
            std::env::var("PEXELS_API_KEY").map_err(|_| MaterialError::ApiKeyNotConfigured)?;

        let config = ApiConfig {
            endpoint: Some(PEXELS_BASE_URL.to_string()),
            timeout: Duration::from_secs(30),
            retry_max: 2,
            ..Default::default()
        };

        Ok(Self {
            client: BaseApiClient::new(config),
            api_key,
        })
    }
}

impl VideoMaterialProvider for PexelsProvider {
    fn name(&self) -> &str {
        "pexels"
    }

    fn search(
        &self,
        query: &str,
        per_page: u32,
        orientation: &str,
        _min_duration: u32,
        _max_duration: u32,
    ) -> Result<Vec<VideoMaterial>, MaterialError> {
        let headers = vec![
            ("Authorization".to_string(), self.api_key.clone()),
        ];

        let url = format!(
            "{}/videos/search?query={}&per_page={}&orientation={}",
            PEXELS_BASE_URL,
            urlencode(query),
            per_page,
            orientation,
        );

        let resp = self
            .client
            .get(&url, Some(&headers))
            .map_err(|e| map_api_error(e))?;

        let body: serde_json::Value =
            resp.json().map_err(|e| MaterialError::ApiError(e.to_string()))?;

        let mut results = Vec::new();
        if let Some(videos) = body["videos"].as_array() {
            for video in videos {
                let id = video["id"].to_string();
                let url_val = video["url"].as_str().unwrap_or("").to_string();
                let duration = video["duration"].as_f64().unwrap_or(0.0);

                // 获取视频文件下载地址
                let download_url = video["video_files"]
                    .as_array()
                    .and_then(|files| files.first())
                    .and_then(|f| f["link"].as_str())
                    .unwrap_or("")
                    .to_string();

                let mut material = VideoMaterial::new(&id, &url_val);
                material.duration = duration;
                material.download_url = download_url;
                material.provider = "pexels".to_string();
                material.author = video["user"]["name"].as_str().unwrap_or("").to_string();
                material.author_url = video["user"]["url"].as_str().unwrap_or("").to_string();

                // 提取标签
                if let Some(tags) = video["tags"].as_array() {
                    for tag in tags {
                        if let Some(t) = tag.as_str() {
                            material.tags.push(t.to_string());
                        }
                    }
                }

                results.push(material);
            }
        }

        Ok(results)
    }

    fn download(
        &self,
        material: &VideoMaterial,
        dest_dir: &Path,
    ) -> Result<Option<PathBuf>, MaterialError> {
        if material.download_url.is_empty() {
            return Ok(None);
        }

        let ext = "mp4";
        let filename = format!("pexels_{}.{}", material.id, ext);
        let dest_path = dest_dir.join(&filename);

        let resp = self
            .client
            .get(&material.download_url, None)
            .map_err(|e| map_api_error(e))?;

        let bytes = resp
            .bytes()
            .map_err(|e| MaterialError::DownloadFailed(e.to_string()))?;

        std::fs::create_dir_all(dest_dir)
            .map_err(|e| MaterialError::DownloadFailed(e.to_string()))?;
        std::fs::write(&dest_path, &bytes)
            .map_err(|e| MaterialError::DownloadFailed(e.to_string()))?;

        Ok(Some(dest_path))
    }
}

fn map_api_error(e: ApiError) -> MaterialError {
    match e {
        ApiError::ApiKeyNotConfigured(_) => MaterialError::ApiKeyNotConfigured,
        ApiError::AuthenticationFailed => {
            MaterialError::ApiError("Pexels 认证失败".to_string())
        }
        ApiError::RateLimited => MaterialError::ApiError("请求被限流".to_string()),
        ApiError::NetworkError(s) => MaterialError::NetworkError(s),
        ApiError::RequestFailed(s) => MaterialError::ApiError(s),
        ApiError::ParseFailed(s) => MaterialError::ApiError(s),
    }
}

fn urlencode(s: &str) -> String {
    url_encode(s)
}

fn url_encode(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            ' ' => "+".to_string(),
            _ => format!("%{:02X}", c as u8),
        })
        .collect()
}
