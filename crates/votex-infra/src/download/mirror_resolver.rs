use std::path::PathBuf;
use votex_domain::model::registry::{ArchiveSpec, ModelRegistryEntry};

/// 下载文件配置
#[derive(Debug, Clone)]
pub struct DownloadFile {
    pub url: String,
    pub dest: PathBuf,
    pub expected_sha256: Option<String>,
    /// 预期文件大小（字节），用于校验下载完整性
    pub expected_size: Option<u64>,
    /// 下载内容为压缩包时，指定从中提取的成员文件
    ///
    /// `expected_sha256` / `expected_size` 在该场景下对应**解压后的目标文件**，
    /// 而非压缩包本身——校验的是真正要加载执行的文件。
    pub archive: Option<ArchiveSpec>,
}

/// 镜像源解析器
///
/// 根据模型清单和镜像优先级，将每个文件的多个镜像源按优先级排列，
/// 生成平铺的 `Vec<DownloadFile>`，供 `Downloader` 依次尝试。
pub struct MirrorResolver;

impl MirrorResolver {
    /// 解析模型文件列表，按镜像优先级排序
    ///
    /// # 参数
    /// * `entry` - 模型清单条目
    /// * `priority` - 镜像优先级列表（如 `["modelscope", "hf-mirror", "github", "huggingface"]`）
    ///
    /// # 返回
    /// 平铺的 `Vec<DownloadFile>`，同一文件的多个源按优先级排列，
    /// 不同文件之间保持清单中的顺序。
    ///
    /// 文件条目若声明了 `platforms`（如 ONNX Runtime 的三个平台产物），
    /// 仅在匹配当前操作系统时才会进入结果，避免下载用不上的库。
    pub fn resolve(entry: &ModelRegistryEntry, priority: &[String]) -> Vec<DownloadFile> {
        let storage_dir = entry.storage_dir();
        let mut files = Vec::new();

        for file_entry in &entry.files {
            if !file_entry.matches_current_platform() {
                tracing::debug!(
                    "跳过非当前平台文件: {}（仅适用于 {:?}）",
                    file_entry.name,
                    file_entry.platforms.as_deref().unwrap_or(&[])
                );
                continue;
            }

            let dest = PathBuf::from(&storage_dir).join(&file_entry.name);
            let sha256 = file_entry.sha256.clone();
            let archive = file_entry.archive.clone();

            // 按优先级顺序收集该文件的各镜像 URL
            let mut has_source = std::collections::HashSet::new();
            let mut added = false;

            let expected_size = file_entry.size;

            // 第一遍：按 priority 顺序添加
            for mirror_name in priority {
                if let Some(url) = file_entry.sources.get(mirror_name.as_str()) {
                    files.push(DownloadFile {
                        url: url.clone(),
                        dest: dest.clone(),
                        expected_sha256: sha256.clone(),
                        expected_size,
                        archive: archive.clone(),
                    });
                    has_source.insert(mirror_name.as_str());
                    added = true;
                }
            }

            // 第二遍：添加 priority 列表中未出现的其他镜像源
            for (mirror_name, url) in &file_entry.sources {
                if !has_source.contains(mirror_name.as_str()) {
                    files.push(DownloadFile {
                        url: url.clone(),
                        dest: dest.clone(),
                        expected_sha256: sha256.clone(),
                        expected_size,
                        archive: archive.clone(),
                    });
                    added = true;
                }
            }

            // 至少有一个源，否则 log 警告
            if !added {
                tracing::warn!(
                    "模型 {} 文件 {} 没有配置任何镜像源",
                    entry.id,
                    file_entry.name
                );
            }
        }

        files
    }

    /// 从 URL 提取简短的源名称（保留供 Downloader 使用）
    pub fn extract_source_name(url: &str) -> String {
        if url.contains("hf-mirror.com") {
            "hf-mirror".to_string()
        } else if url.contains("modelscope.cn") {
            "modelscope".to_string()
        } else if url.contains("huggingface.co") {
            "huggingface".to_string()
        } else if url.contains("github.com") {
            "github".to_string()
        } else if url.contains("gitee.com") {
            "gitee".to_string()
        } else {
            "unknown".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use votex_domain::model::registry::ModelFileEntry;

    fn make_test_entry() -> ModelRegistryEntry {
        let mut sources1 = HashMap::new();
        sources1.insert("huggingface".into(), "https://huggingface.co/test/model.onnx".into());
        sources1.insert("hf-mirror".into(), "https://hf-mirror.com/test/model.onnx".into());
        sources1.insert("github".into(), "https://github.com/test/model.onnx".into());

        let mut sources2 = HashMap::new();
        sources2.insert("huggingface".into(), "https://huggingface.co/test/config.json".into());
        sources2.insert("hf-mirror".into(), "https://hf-mirror.com/test/config.json".into());

        ModelRegistryEntry {
            id: "test-model".into(),
            name: "Test Model".into(),
            kind: "Tts".into(),
            engine: "Test".into(),
            sub_dir: None,
            tokenizer: None,
            files: vec![
                ModelFileEntry {
                    name: "model.onnx".into(),
                    sha256: None,
                    size: None,
                    required: false,
                    sources: sources1,
                    ..Default::default()
                },
                ModelFileEntry {
                    name: "config.json".into(),
                    sha256: None,
                    size: None,
                    required: false,
                    sources: sources2,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    }

    #[test]
    fn mirror_resolver_按优先级排序() {
        let entry = make_test_entry();
        let priority: Vec<String> = vec![
            "github".into(),
            "hf-mirror".into(),
            "huggingface".into(),
        ];

        let files = MirrorResolver::resolve(&entry, &priority);

        // model.onnx: github → hf-mirror → huggingface
        assert_eq!(files.len(), 5); // 3 + 2
        assert!(files[0].url.contains("github.com"));
        assert!(files[0].dest.ends_with("model.onnx"));
        assert!(files[1].url.contains("hf-mirror.com"));
        assert!(files[2].url.contains("huggingface.co"));
        // config.json: hf-mirror → huggingface (无 github)
        assert!(files[3].url.contains("hf-mirror.com"));
        assert!(files[3].dest.ends_with("config.json"));
        assert!(files[4].url.contains("huggingface.co"));
    }

    #[test]
    fn mirror_resolver_默认优先级回退() {
        let entry = make_test_entry();
        let priority: Vec<String> = vec!["modelscope".into()]; // 无匹配

        let files = MirrorResolver::resolve(&entry, &priority);

        // modelscope 不匹配，其他源应被追加
        assert!(!files.is_empty());
        // 第一个应该是 modelscope ？没有，所以第一个是任意剩余源
        assert_eq!(files.len(), 5);
    }

    #[test]
    fn mirror_resolver_sub_dir包含在dest中() {
        let mut sources = HashMap::new();
        sources.insert("huggingface".into(), "https://hf.co/test/model.onnx".into());

        let entry = ModelRegistryEntry {
            id: "test".into(),
            name: "Test".into(),
            kind: "Ocr".into(),
            engine: "PaddleOCR".into(),
            sub_dir: Some("custom-dir".into()),
            tokenizer: None,
            files: vec![ModelFileEntry {
                name: "sub/file.onnx".into(),
                sha256: None,
                size: None,
                required: false,
                sources,
                ..Default::default()
            }],
            ..Default::default()
        };

        let priority: Vec<String> = vec!["huggingface".into()];
        let files = MirrorResolver::resolve(&entry, &priority);

        assert_eq!(files.len(), 1);
        let dest_str = files[0].dest.to_string_lossy();
        // 跨平台：Windows 使用 \，Unix 使用 /
        // 注意：file_entry.name 中的 / 不会被转换，所以实际路径可能是 ocr/custom-dir\sub/file.onnx
        assert!(dest_str.contains("custom-dir"), "dest should include sub_dir: {dest_str}");
        assert!(dest_str.contains("sub"), "dest should include sub_dir: {dest_str}");
        assert!(dest_str.contains("file.onnx"), "dest should include sub_dir: {dest_str}");
    }

    #[test]
    fn mirror_resolver_sha256传递() {
        let mut sources = HashMap::new();
        sources.insert("huggingface".into(), "https://hf.co/test/model.onnx".into());

        let entry = ModelRegistryEntry {
            id: "test".into(),
            name: "Test".into(),
            kind: "Tts".into(),
            engine: "Test".into(),
            sub_dir: None,
            tokenizer: None,
            files: vec![ModelFileEntry {
                name: "model.onnx".into(),
                sha256: Some("abc123".into()),
                size: None,
                required: false,
                sources,
                ..Default::default()
            }],
            ..Default::default()
        };

        let priority: Vec<String> = vec!["huggingface".into()];
        let files = MirrorResolver::resolve(&entry, &priority);

        assert_eq!(files[0].expected_sha256.as_deref(), Some("abc123"));
    }

    #[test]
    fn mirror_resolver_extract_source_name() {
        assert_eq!(
            MirrorResolver::extract_source_name("https://hf-mirror.com/test/model.zip"),
            "hf-mirror"
        );
        assert_eq!(
            MirrorResolver::extract_source_name("https://huggingface.co/test/model.zip"),
            "huggingface"
        );
        assert_eq!(
            MirrorResolver::extract_source_name("https://github.com/test/model.zip"),
            "github"
        );
        assert_eq!(
            MirrorResolver::extract_source_name("https://modelscope.cn/test/model.zip"),
            "modelscope"
        );
        assert_eq!(
            MirrorResolver::extract_source_name("https://gitee.com/test/model.zip"),
            "gitee"
        );
    }
}
