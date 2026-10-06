//! 下载门面：进度回调类型与下载器构造
//!
//! CLI 的 `model download` 需要把进度打到 stdout 并落库，
//! GUI 需要把进度画到进度条。这两处共用同一个回调签名。

/// 下载进度回调：`(|已下载, 总大小, 源, 文件名|)`
pub type ProgressFn = votex_infra::download::downloader::ProgressFn;

/// 按配置构造下载器
///
/// `resume` 与 `timeout` 来自 `application.yml` 的 `models.download`，
/// 由 [`crate::use_case::model_use_case::set_global_download_config`] 下发。
pub fn build_downloader(
    config: &votex_domain::config::value_object::DownloadConfig,
) -> anyhow::Result<votex_infra::download::downloader::Downloader> {
    Ok(votex_infra::download::downloader::Downloader::with_config(
        config,
    )?)
}

/// 当前生效的下载配置
pub fn current_download_config() -> votex_domain::config::value_object::DownloadConfig {
    crate::use_case::model_use_case::global_download_config()
}

/// 镜像名 → 优先级列表
///
/// CLI `--mirror cn` 与 GUI 下拉框共用同一套映射，
/// 避免两处各写一份 if/else 导致行为不一致。
pub fn mirror_priority(mirror: &str) -> Vec<String> {
    match mirror {
        "cn" | "china" => vec!["modelscope".into(), "hf-mirror".into(), "gitee".into()],
        _ => vec![
            "modelscope".into(),
            "hf-mirror".into(),
            "github".into(),
            "huggingface".into(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 国内镜像映射不含境外源() {
        let p = mirror_priority("cn");
        assert!(p.contains(&"modelscope".to_string()));
        assert!(p.contains(&"hf-mirror".to_string()));
        assert!(!p.contains(&"huggingface".to_string()));
    }

    #[test]
    fn 默认镜像含全部源() {
        let p = mirror_priority("default");
        assert_eq!(p.len(), 4);
        assert!(p.contains(&"huggingface".to_string()));
    }

    #[test]
    fn 未知镜像回退到默认() {
        assert_eq!(mirror_priority("不存在的镜像"), mirror_priority("default"));
    }
}
