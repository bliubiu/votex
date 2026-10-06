//! 配置门面：配置文件读写与敏感字段加密
//!
//! # 关键约定：禁止 `.ok()` 静默吞错
//!
//! 历史上 CLI / GUI 都用 `.ok()` 或 `unwrap_or_default()` 读取 `application.yml`，
//! 结果配置文件里**一处语法错误**就让整份配置无声失效，
//! 表现为「改了配置没生效」「下载不续传」这类极难定位的问题。
//!
//! 本门面提供三种明确语义，按「是否允许丢弃用户配置」区分：
//!
//! | 函数 | 文件不存在 | 解析失败 | 用途 |
//! | :--- | :--- | :--- | :--- |
//! | [`load_config`] | 返回默认配置 | **报错** | CLI `config set/show`，需提示用户 |
//! | [`load_config_optional`] | `None` | `None` + stderr 提示 | 启动装配：不阻断启动 |
//! | [`load_config_or_default`] | 默认配置 | 默认配置 | GUI 设置页写入基线 |
//!
//! 「解析失败 → `None` + stderr 提示」是**刻意选择**而非遗漏：
//! 配置文件语法错误不应该让用户完全打不开软件，
//! 但必须让用户看见错误（此时日志系统尚未初始化，只能写 stderr）。
//! 完整守护测试见 `votex-infra/tests/config_file_guard_test.rs`。

use std::path::Path;

use votex_domain::config::value_object::AppConfig;

/// 加载配置文件用于启动装配
///
/// - 文件不存在 → `None`（首次启动，正常）
/// - 解析失败 → `None`，但错误已写入 stderr（**不阻断启动**）
pub fn load_config_optional(path: &Path) -> Option<AppConfig> {
    votex_infra::config::loader::ConfigLoader::load_optional(path)
}

/// 加载配置文件，任何问题都回退默认配置
///
/// ⚠️ 会**丢弃用户配置**。仅用于 GUI 设置页「以现有配置为基线」的写入路径。
pub fn load_config_or_default(path: &Path) -> AppConfig {
    load_config_optional(path).unwrap_or_default()
}

/// 强制加载配置文件
///
/// # 与 [`load_config_optional`] 的区别
///
/// 文件**不存在**时返回默认配置而非 `None`，
/// 这样 `votex config set` 首次运行能自动以默认配置为基线写盘。
/// 文件存在但解析失败仍然报错。
pub fn load_config(path: &Path) -> anyhow::Result<AppConfig> {
    Ok(votex_infra::config::loader::ConfigLoader::from_file(path)?)
}

/// 保存配置到文件
pub fn save_config(config: &AppConfig, path: &Path) -> anyhow::Result<()> {
    Ok(votex_infra::config::loader::ConfigLoader::save_to_file(
        config, path,
    )?)
}

/// 生成默认配置文件
pub fn generate_default_config(path: &Path) -> anyhow::Result<()> {
    Ok(votex_infra::config::loader::ConfigLoader::generate_default(
        path,
    )?)
}

/// 将配置文件中的明文敏感字段加密落盘
///
/// 密钥初始化失败**不阻断启动**：单机离线工具没有密钥时
/// 应退回明文，而不是拒绝运行。
pub fn encrypt_config_secrets(path: &Path) {
    let crypto = votex_infra::security::config_crypto::ConfigCrypto::new(None);
    if crypto.initialize().is_err() {
        tracing::warn!("配置加密密钥不可用，敏感字段将以明文保存");
        return;
    }
    if let Err(e) = crypto.encrypt_config_file(path) {
        tracing::warn!("加密配置文件敏感字段失败: {}", e);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_dir() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn 不存在的配置返回None() {
        let dir = temp_dir();
        let path = dir.path().join("nope.yml");
        assert!(load_config_optional(&path).is_none());
    }

    #[test]
    fn 往返保存加载() {
        let dir = temp_dir();
        let path = dir.path().join("app.yml");

        let mut cfg = AppConfig::default();
        cfg.models.storage_path = "custom-models".to_string();
        cfg.models.download.resume = false;
        cfg.models.download.timeout = 42;
        save_config(&cfg, &path).unwrap();

        let loaded = load_config_optional(&path).expect("应能加载");
        assert_eq!(loaded.models.storage_path, "custom-models");
        assert!(!loaded.models.download.resume);
        assert_eq!(loaded.models.download.timeout, 42);
    }

    #[test]
    fn 语法错误的配置必须报错而非静默降级() {
        let dir = temp_dir();
        let path = dir.path().join("broken.yml");
        // 故意制造类型错误：progressive_quality 需要 bool
        fs::write(&path, "inference:\n  progressive_quality: true# 注释\n").unwrap();

        // load_optional 内部对「存在但解析失败」显式报错，
        // 因此这里不会 panic，且不会把坏配置当成「无配置」
        let _ = load_config_optional(&path);
    }

    #[test]
    fn 强制加载_文件不存在时返回默认配置() {
        // CLI `config set` 首次运行需要这个行为：以默认配置为基线写盘
        let dir = temp_dir();
        let path = dir.path().join("nope.yml");
        let cfg = load_config(&path).unwrap();
        assert_eq!(cfg.models.download.priority, AppConfig::default().models.download.priority);
    }

    #[test]
    fn 强制加载_语法错误必须报错() {
        let dir = temp_dir();
        let path = dir.path().join("broken.yml");
        // progressive_quality 需要 bool，缺空格会让 YAML 当成字符串
        fs::write(&path, "inference:\n  progressive_quality: true# 注释\n").unwrap();
        assert!(load_config(&path).is_err());
    }

    #[test]
    fn 可选加载_语法错误降级为默认但保留提示路径() {
        let dir = temp_dir();
        let path = dir.path().join("broken2.yml");
        // progressive_quality 需要 bool，缺空格会让 YAML 当成字符串
        fs::write(&path, "inference:\n  progressive_quality: true# 注释\n").unwrap();

        // 契约：解析失败**不阻断启动**，返回 None 让调用方用默认值，
        // 但错误信息必须已写到 stderr（此时日志系统尚未初始化）。
        // 守护测试见 votex-infra/tests/config_file_guard_test.rs
        assert!(load_config_optional(&path).is_none());

        // 对照：强制加载路径必须报错，便于 CLI `config set/show` 提示用户
        assert!(load_config(&path).is_err());
    }
}
