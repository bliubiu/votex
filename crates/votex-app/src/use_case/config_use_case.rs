use votex_domain::config::value_object::AppConfig;
use anyhow::Result;

/// 配置管理用例
pub struct ConfigUseCase;

impl ConfigUseCase {
    /// 加载配置
    pub fn load_config(_config_path: &std::path::Path) -> Result<AppConfig> {
        // 实际通过 ConfigRepo 加载，后续阶段实现
        Ok(AppConfig::default())
    }

    /// 生成默认配置
    pub fn default_config() -> AppConfig {
        AppConfig::default()
    }
}
