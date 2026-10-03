//! 视频素材提供者模块
//!
//! 支持 Pexels 和 Pixabay 等在线素材平台的 API 查询。

pub mod pexels;
pub mod pixabay;

use votex_domain::video_material::provider::VideoMaterialProvider;

/// 根据素材源名称创建提供者
pub fn create_material_provider(
    source: &str,
) -> Result<Box<dyn VideoMaterialProvider>, String> {
    match source.to_lowercase().as_str() {
        "pexels" => Ok(Box::new(
            pexels::PexelsProvider::new().map_err(|e| e.to_string())?,
        )),
        "pixabay" => Ok(Box::new(
            pixabay::PixabayProvider::new().map_err(|e| e.to_string())?,
        )),
        _ => Err(format!("不支持的素材源: {}", source)),
    }
}
