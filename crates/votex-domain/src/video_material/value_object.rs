use serde::{Deserialize, Serialize};

/// 视频素材
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoMaterial {
    pub id: String,
    pub url: String,
    pub thumbnail_url: String,
    pub width: u32,
    pub height: u32,
    pub duration: f64,
    pub provider: String,
    pub author: String,
    pub author_url: String,
    pub download_url: String,
    pub tags: Vec<String>,
}

impl VideoMaterial {
    pub fn new(id: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            url: url.into(),
            thumbnail_url: String::new(),
            width: 0,
            height: 0,
            duration: 0.0,
            provider: String::new(),
            author: String::new(),
            author_url: String::new(),
            download_url: String::new(),
            tags: Vec::new(),
        }
    }

    /// 视频方向
    pub fn orientation(&self) -> &str {
        if self.width > self.height {
            "landscape"
        } else if self.height > self.width {
            "portrait"
        } else {
            "square"
        }
    }

    /// 分辨率标签
    pub fn resolution_label(&self) -> String {
        format!("{}x{}", self.width, self.height)
    }
}
