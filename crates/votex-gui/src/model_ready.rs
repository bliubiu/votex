//! 模型就绪判定（纯逻辑）
//!
//! # 为什么要抽出来
//!
//! `model_detector::is_model_ready` 依赖 `load_registry()` 这个
//! `OnceLock<Vec<ModelRegistryEntry>>` 全局单例——单例在进程内只能初始化一次，
//! 测试里没法用不同的清单驱动同一个断言。
//!
//! 这里把「给定清单 + 目录 → 是否就绪」的判定拆成纯函数，
//! 全局单例只在 [`crate::model_detector`] 那一层读一次，
//! 判定规则本身可被任意清单驱动测试。
//!
//! # 判定规则
//!
//! - 清单里找不到该模型 → 未就绪
//! - 模型目录不存在 → 未就绪
//! - 清单声明了 `required` 文件 → **全部**存在才算就绪
//! - 未声明任何 `required` → **任意一个**文件存在即算就绪（宽松策略）

use std::path::Path;

use votex_domain::model::registry::{ModelFileEntry, ModelRegistryEntry};

/// 按文件列表判定模型是否就绪
///
/// `storage_dir` 是相对 `models_dir` 的子目录（如 `tts/kokoro-82m`）。
pub fn is_ready_with(
    entries: &[ModelRegistryEntry],
    models_dir: &Path,
    model_id: &str,
) -> bool {
    let Some(entry) = entries.iter().find(|e| e.id == model_id) else {
        return false;
    };

    let storage_path = models_dir.join(entry.storage_dir());
    if !storage_path.exists() {
        return false;
    }

    files_satisfied(&entry.files, &storage_path)
}

/// 按文件列表判定是否满足就绪条件
///
/// 与 [`is_ready_with`] 分离，便于在测试里直接构造文件列表，
/// 不必伪造完整的 `ModelRegistryEntry`。
pub fn files_satisfied(files: &[ModelFileEntry], storage_path: &Path) -> bool {
    let required: Vec<&ModelFileEntry> = files.iter().filter(|f| f.required).collect();

    if !required.is_empty() {
        // 声明了必需文件 → 必须全部存在
        required.iter().all(|f| storage_path.join(&f.name).exists())
    } else {
        // 无必需声明 → 任意一个存在即可（宽松策略）
        files.iter().any(|f| storage_path.join(&f.name).exists())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(name: &str, required: bool) -> ModelFileEntry {
        ModelFileEntry {
            name: name.to_string(),
            sha256: None,
            size: None,
            required,
            ..Default::default()
        }
    }

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("votex_gui_ready_{}", tag));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    // ===== files_satisfied：required 语义 =====

    #[test]
    fn required_全部存在才算就绪() {
        let d = tmpdir("req_all");
        std::fs::write(d.join("a.onnx"), b"x").unwrap();
        std::fs::write(d.join("b.bin"), b"x").unwrap();

        let files = vec![file("a.onnx", true), file("b.bin", true)];
        assert!(files_satisfied(&files, &d));

        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn required_缺任一即不就绪() {
        let d = tmpdir("req_missing");
        std::fs::write(d.join("a.onnx"), b"x").unwrap();
        // b.bin 缺失

        let files = vec![file("a.onnx", true), file("b.bin", true)];
        assert!(!files_satisfied(&files, &d));

        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn 混合required与可选_只看必需项() {
        let d = tmpdir("mixed");
        std::fs::write(d.join("model.onnx"), b"x").unwrap();
        // 可选文件 notes.txt 不存在，不影响就绪

        let files = vec![file("model.onnx", true), file("notes.txt", false)];
        assert!(files_satisfied(&files, &d));

        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn 无required声明_任意存在即可() {
        let d = tmpdir("loose");
        std::fs::write(d.join("whatever.onnx"), b"x").unwrap();

        let files = vec![file("whatever.onnx", false), file("missing.bin", false)];
        assert!(files_satisfied(&files, &d));

        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn 无required且全部缺失_不就绪() {
        let d = tmpdir("loose_none");
        let files = vec![file("a.onnx", false), file("b.onnx", false)];
        assert!(!files_satisfied(&files, &d));

        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn 空文件列表_不就绪() {
        let d = tmpdir("empty_files");
        // 目录存在但没有任何文件可查——不能算就绪，
        // 否则会显示「已就绪」但引擎启动时找不到权重。
        assert!(!files_satisfied(&[], &d));

        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn 子目录路径正确拼接() {
        // 文件名可能含 '/'（如 voices/af_heart.bin）
        let d = tmpdir("subdir");
        std::fs::create_dir_all(d.join("voices")).unwrap();
        std::fs::write(d.join("voices").join("af_heart.bin"), b"x").unwrap();

        let files = vec![file("voices/af_heart.bin", true)];
        assert!(files_satisfied(&files, &d));

        let _ = std::fs::remove_dir_all(&d);
    }

    // ===== is_ready_with：清单层判定 =====

    #[test]
    fn is_ready_未知模型返回否() {
        let d = tmpdir("unknown");
        assert!(!is_ready_with(&[], &d, "never-existed"));

        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn is_ready_目录不存在返回否() {
        let d = tmpdir("nodir");
        let entries = vec![ModelRegistryEntry {
            id: "demo".to_string(),
            name: "Demo".to_string(),
            kind: "Tts".to_string(),
            engine: "Kokoro".to_string(),
            sub_dir: Some("does-not-exist".to_string()),
            files: vec![file("model.onnx", true)],
            ..Default::default()
        }];

        assert!(!is_ready_with(&entries, &d, "demo"));

        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn is_ready_目录与文件齐备返回真() {
        let d = tmpdir("complete");
        let sub = d.join("tts").join("demo");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("model.onnx"), b"x").unwrap();

        let entries = vec![ModelRegistryEntry {
            id: "demo".to_string(),
            name: "Demo".to_string(),
            kind: "Tts".to_string(),
            engine: "Kokoro".to_string(),
            sub_dir: Some("demo".to_string()),
            files: vec![file("model.onnx", true)],
            ..Default::default()
        }];

        assert!(is_ready_with(&entries, &d, "demo"));

        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn is_ready_目录存在但文件缺失返回否() {
        let d = tmpdir("emptydir");
        std::fs::create_dir_all(d.join("tts").join("demo")).unwrap();

        let entries = vec![ModelRegistryEntry {
            id: "demo".to_string(),
            name: "Demo".to_string(),
            kind: "Tts".to_string(),
            engine: "Kokoro".to_string(),
            sub_dir: Some("demo".to_string()),
            files: vec![file("model.onnx", true)],
            ..Default::default()
        }];

        assert!(!is_ready_with(&entries, &d, "demo"));

        let _ = std::fs::remove_dir_all(&d);
    }
}
