//! Pixabay 视频素材 API 适配器
//!
//! 通过 Pixabay API 搜索免费视频素材。
//! API 文档: https://pixabay.com/api/docs/

use crate::api::base::{ApiConfig, ApiError, BaseApiClient};
use std::time::Duration;
use std::path::{Path, PathBuf};
use votex_domain::video_material::provider::{MaterialError, VideoMaterialProvider};
use votex_domain::video_material::value_object::VideoMaterial;

/// Pixabay API 基础地址
const PIXABAY_BASE_URL: &str = "https://pixabay.com";

/// Pixabay 视频素材提供者
pub struct PixabayProvider {
    client: BaseApiClient,
    api_key: String,
}

impl PixabayProvider {
    /// 创建 Pixabay 提供者
    ///
    /// 从环境变量 `PIXABAY_API_KEY` 读取 API Key。
    pub fn new() -> Result<Self, MaterialError> {
        let api_key =
            std::env::var("PIXABAY_API_KEY").map_err(|_| MaterialError::ApiKeyNotConfigured)?;

        let config = ApiConfig {
            endpoint: Some(PIXABAY_BASE_URL.to_string()),
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

impl VideoMaterialProvider for PixabayProvider {
    fn name(&self) -> &str {
        "pixabay"
    }

    fn search(
        &self,
        query: &str,
        per_page: u32,
        orientation: &str,
        _min_duration: u32,
        _max_duration: u32,
    ) -> Result<Vec<VideoMaterial>, MaterialError> {
        // Pixabay API 使用 query parameter 传 API Key
        let orient_param = match orientation {
            "portrait" => "vertical",
            "landscape" => "horizontal",
            _ => "all",
        };

        let url = format!(
            "{}/api/videos/?key={}&q={}&per_page={}&orientation={}&safesearch=true",
            PIXABAY_BASE_URL,
            self.api_key,
            urlencode(query),
            per_page,
            orient_param,
        );

        let resp = self
            .client
            .get(&url, None)
            .map_err(|e| map_api_error(e))?;

        let body: serde_json::Value =
            resp.json().map_err(|e| MaterialError::ApiError(e.to_string()))?;

        let mut results = Vec::new();
        if let Some(hits) = body["hits"].as_array() {
            for video in hits {
                let id = video["id"].to_string();
                let page_url = video["pageURL"].as_str().unwrap_or("").to_string();
                let duration = video["duration"].as_f64().unwrap_or(0.0);
                let tags_str = video["tags"].as_str().unwrap_or("");

                // 获取各种分辨率的视频地址
                let download_url = video["videos"]["medium"]["url"]
                    .as_str()
                    .or_else(|| video["videos"]["small"]["url"].as_str())
                    .or_else(|| video["videos"]["large"]["url"].as_str())
                    .unwrap_or("")
                    .to_string();

                let mut material = VideoMaterial::new(&id, &page_url);
                material.duration = duration;
                material.download_url = download_url;
                material.provider = "pixabay".to_string();
                material.author = video["user"].as_str().unwrap_or("").to_string();
                material.author_url = match video["user_id"].as_u64() {
                    Some(uid) => format!("https://pixabay.com/users/{}", uid),
                    None => String::new(),
                };

                // 解析标签
                for tag in tags_str.split(',') {
                    let tag = tag.trim();
                    if !tag.is_empty() {
                        material.tags.push(tag.to_string());
                    }
                }

                // 获取分辨率信息
                if let Some(video_info) = video["videos"]["medium"].as_object() {
                    material.width = video_info.get("width").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                    material.height = video_info.get("height").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
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
        let filename = format!("pixabay_{}.{}", material.id, ext);
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
            MaterialError::ApiError("Pixabay 认证失败".to_string())
        }
        ApiError::RateLimited => MaterialError::ApiError("请求被限流".to_string()),
        ApiError::NetworkError(s) => MaterialError::NetworkError(s),
        ApiError::RequestFailed(s) => MaterialError::ApiError(s),
        ApiError::ParseFailed(s) => MaterialError::ApiError(s),
    }
}

fn urlencode(s: &str) -> String {
    s.chars()
        .map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            ' ' => "+".to_string(),
            _ => format!("%{:02X}", c as u8),
        })
        .collect()
}
