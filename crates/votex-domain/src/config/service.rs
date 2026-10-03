use crate::config::value_object::AppConfig;

/// 配置领域服务
pub struct ConfigService;

impl ConfigService {
    pub fn new() -> Self {
        Self
    }

    /// 生成默认配置
    pub fn default_config() -> AppConfig {
        AppConfig::default()
    }
}
