use votex_domain::config::value_object::AppConfig;
use anyhow::Result;
use std::path::Path;

/// 配置仓储实现
pub struct ConfigRepo {
    config_path: std::path::PathBuf,
}

impl ConfigRepo {
    pub fn new(config_path: &Path) -> Self {
        Self {
            config_path: config_path.to_path_buf(),
        }
    }

    /// 加载配置，文件不存在则生成默认配置
    pub fn load(&self) -> Result<AppConfig> {
        if !self.config_path.exists() {
            let config = AppConfig::default();
            self.save(&config)?;
            tracing::info!("配置文件不存在，已生成默认配置: {:?}", self.config_path);
            return Ok(config);
        }

        let content = std::fs::read_to_string(&self.config_path)?;
        let config: AppConfig = serde_yml::from_str(&content)?;
        tracing::info!("配置文件加载成功: {:?}", self.config_path);
        Ok(config)
    }

    /// 保存配置
    pub fn save(&self, config: &AppConfig) -> Result<()> {
        if let Some(parent) = self.config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = serde_yml::to_string(config)?;
        std::fs::write(&self.config_path, content)?;
        tracing::info!("配置文件保存成功: {:?}", self.config_path);
        Ok(())
    }

    /// 配置文件路径
    pub fn config_path(&self) -> &Path {
        &self.config_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn config_repo_文件不存在时生成默认配置() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("application.yml");
        let repo = ConfigRepo::new(&path);

        let config = repo.load().unwrap();
        assert_eq!(config.tts.default_engine, "kokoro");
        assert!(path.exists());
    }

    #[test]
    fn config_repo_保存和加载() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("application.yml");
        let repo = ConfigRepo::new(&path);

        let mut config = AppConfig::default();
        config.tts.default_speed = 1.5;
        repo.save(&config).unwrap();

        let loaded = repo.load().unwrap();
        assert!((loaded.tts.default_speed - 1.5).abs() < 0.001);
    }
}
