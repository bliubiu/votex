use anyhow::Result;
use std::path::Path;
use votex_domain::model::registry::ModelRegistryEntry;

/// 模型清单加载器
///
/// 从 `models/registry/*.yaml` 加载所有模型定义，
/// 替代代码中硬编码的模型元数据和 URL 列表。
pub struct ModelRegistryLoader;

impl ModelRegistryLoader {
    /// 从指定目录加载所有模型清单文件
    ///
    /// 扫描目录下所有 `*.yaml`（或 `*.yml`）文件，
    /// 反序列化为 `ModelRegistryEntry`。
    /// 格式错误的文件会被跳过，仅记录警告。
    pub fn load(registry_dir: &Path) -> Result<Vec<ModelRegistryEntry>> {
        if !registry_dir.exists() {
            tracing::warn!("模型清单目录不存在: {:?}", registry_dir);
            return Ok(Vec::new());
        }

        let mut entries = Vec::new();

        let mut dir_reader = match std::fs::read_dir(registry_dir) {
            Ok(reader) => reader,
            Err(e) => {
                tracing::error!("读取模型清单目录失败 {:?}: {}", registry_dir, e);
                return Ok(Vec::new());
            }
        };

        while let Some(entry) = dir_reader.next() {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    tracing::warn!("读取目录条目失败: {}", e);
                    continue;
                }
            };

            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            // 只处理 .yaml / .yml 文件
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if ext != "yaml" && ext != "yml" {
                continue;
            }

            match Self::load_single(&path) {
                Ok(Some(entry)) => entries.push(entry),
                Ok(None) => { /* 跳过空文件 */ }
                Err(e) => {
                    tracing::warn!("跳过无效的模型清单文件 {:?}: {}", path, e);
                }
            }
        }

        // 按 id 排序，保证顺序稳定
        entries.sort_by(|a, b| a.id.cmp(&b.id));

        tracing::info!("已加载 {} 个模型清单", entries.len());
        Ok(entries)
    }

    /// 加载单个清单文件
    fn load_single(path: &Path) -> Result<Option<ModelRegistryEntry>> {
        let content = std::fs::read_to_string(path)?;

        if content.trim().is_empty() {
            return Ok(None);
        }

        let entry: ModelRegistryEntry = serde_yml::from_str(&content)?;

        if entry.id.is_empty() {
            anyhow::bail!("模型清单缺少 id 字段");
        }

        tracing::debug!("已加载模型清单: {} ({})", entry.id, entry.name);
        Ok(Some(entry))
    }

    /// 从所有清单中查找指定 ID 的模型
    pub fn find_by_id<'a>(
        entries: &'a [ModelRegistryEntry],
        id: &str,
    ) -> Option<&'a ModelRegistryEntry> {
        entries.iter().find(|e| e.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn registry_loader_加载目录() {
        let dir = tempfile::tempdir().unwrap();

        // 创建两个清单文件
        let yaml1 = r#"
id: kokoro-82m
name: Kokoro-82M
kind: Tts
engine: Kokoro
files:
  - name: model.onnx
    sources:
      huggingface: https://example.com/model.onnx
"#;
        let yaml2 = r#"
id: whisper-base
name: Whisper Base
kind: Asr
engine: Whisper
files:
  - name: encoder.onnx
    sources:
      huggingface: https://example.com/encoder.onnx
"#;

        fs::write(dir.path().join("kokoro-82m.yaml"), yaml1).unwrap();
        fs::write(dir.path().join("whisper-base.yaml"), yaml2).unwrap();

        let entries = ModelRegistryLoader::load(dir.path()).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].id, "kokoro-82m");
        assert_eq!(entries[1].id, "whisper-base");
    }

    #[test]
    fn registry_loader_忽略非yaml文件() {
        let dir = tempfile::tempdir().unwrap();

        fs::write(dir.path().join("readme.txt"), "hello").unwrap();
        fs::write(
            dir.path().join("test.yaml"),
            r#"
id: test
name: Test
kind: Tts
engine: Test
files: []
"#,
        )
        .unwrap();

        let entries = ModelRegistryLoader::load(dir.path()).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "test");
    }

    #[test]
    fn registry_loader_目录不存在返回空() {
        let entries =
            ModelRegistryLoader::load(Path::new("C:\\non_existent_registry_dir_xyz")).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn registry_loader_无效yaml跳过() {
        let dir = tempfile::tempdir().unwrap();

        fs::write(dir.path().join("bad.yaml"), "这不是有效 YAML: {{{").unwrap();
        fs::write(
            dir.path().join("good.yaml"),
            r#"
id: good
name: Good
kind: Tts
engine: Test
files: []
"#,
        )
        .unwrap();

        let entries = ModelRegistryLoader::load(dir.path()).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, "good");
    }

    #[test]
    fn registry_loader_find_by_id() {
        let dir = tempfile::tempdir().unwrap();

        fs::write(
            dir.path().join("a.yaml"),
            r#"
id: model-a
name: Model A
kind: Tts
engine: Test
files: []
"#,
        )
        .unwrap();
        fs::write(
            dir.path().join("b.yaml"),
            r#"
id: model-b
name: Model B
kind: Asr
engine: Test
files: []
"#,
        )
        .unwrap();

        let entries = ModelRegistryLoader::load(dir.path()).unwrap();

        let found = ModelRegistryLoader::find_by_id(&entries, "model-b").unwrap();
        assert_eq!(found.name, "Model B");

        let not_found = ModelRegistryLoader::find_by_id(&entries, "model-c");
        assert!(not_found.is_none());
    }
}
