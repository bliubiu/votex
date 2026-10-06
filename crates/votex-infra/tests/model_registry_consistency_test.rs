//! 模型清单（registry）一致性测试
//!
//! 对应 `docs/20-功能E2E验证测试计划.md` 的 **E2E-MDL-01 / G0-12**，
//! 是零推理开销、可进 CI 的静态门禁。
//!
//! 覆盖两条契约：
//! 1. **每份 `models/registry/*.yaml` 都能反序列化**——`ModelRegistryLoader::load`
//!    对格式错误的文件只 `tracing::warn!` 后跳过，命令退出码仍为 0，
//!    导致 `model list` 少列几行而用户完全无从察觉（缺陷 F45/F54）。
//! 2. **每份清单声明的 `files[].name` 都真实存在于磁盘**——registry 与磁盘失真时，
//!    `model download` 会下出运行时永不加载的副本（缺陷 F45）。
//!
//! 白名单：`qwen-mt` 是**在线模型**（`DASHSCOPE_API_KEY`），无本地目录属预期。

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use votex_domain::model::registry::ModelRegistryEntry;

/// 仓库根目录（测试进程的 CWD 是 crate 目录，故向上两级）
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .canonicalize()
        .expect("无法定位仓库根目录")
}

fn registry_dir() -> PathBuf {
    repo_root().join("models").join("registry")
}

/// 在线模型（依赖 API Key，无本地文件）——允许"无目录/无文件"
const ONLINE_MODEL_WHITELIST: &[&str] = &["qwen-mt"];

/// 按 `kind` 映射分类目录名（与 `ModelRegistryEntry::kind_dir` 保持一致）
fn kind_dir(kind: &str) -> &'static str {
    match kind {
        "Tts" => "tts",
        "Asr" => "asr",
        "Ocr" => "ocr",
        "Translation" => "translation",
        _ => "other",
    }
}

/// 收集所有 registry YAML 的解析结果，并保留失败原因用于断言信息
fn parse_all() -> (BTreeMap<String, ModelRegistryEntry>, Vec<String>) {
    let mut ok = BTreeMap::new();
    let mut errs = Vec::new();

    let dir = registry_dir();
    let files = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("读取模型清单目录失败 {}: {:#}", dir.display(), e));

    for entry in files.filter_map(|e| e.ok()).map(|e| e.path()) {
        let ext = entry.extension().and_then(|s| s.to_str()).unwrap_or("");
        if ext != "yaml" && ext != "yml" {
            continue;
        }
        let content = std::fs::read_to_string(&entry)
            .unwrap_or_else(|e| panic!("读取 {} 失败: {:#}", entry.display(), e));
        match serde_yml::from_str::<ModelRegistryEntry>(&content) {
            Ok(e) => {
                ok.insert(
                    entry.file_name().unwrap_or_default().to_string_lossy().into_owned(),
                    e,
                );
            }
            Err(e) => errs.push(format!("{}: {:#}", entry.display(), e)),
        }
    }
    (ok, errs)
}

/// 契约 1：所有 registry YAML 均可反序列化，且 id 不为空
#[test]
fn registry_每份清单都能解析且id非空() {
    let (ok, errs) = parse_all();
    assert!(
        errs.is_empty(),
        "以下模型清单无法反序列化（ModelRegistryLoader 会静默跳过，导致 `model list` 缺项）:\n{}",
        errs.join("\n")
    );
    assert!(
        !ok.is_empty(),
        "models/registry 下未发现任何 .yaml 清单（仓库根解析错误?）"
    );
    for (file, entry) in &ok {
        assert!(
            !entry.id.is_empty(),
            "{} 缺少 id 字段",
            file
        );
    }
}

/// 契约 2：清单声明的每个文件都存在于磁盘（在线模型走白名单）
#[test]
fn registry_声明文件与磁盘一致() {
    let (ok, errs) = parse_all();
    assert!(errs.is_empty(), "清单解析失败:\n{}", errs.join("\n"));

    let root = repo_root();
    let models_dir = root.join("models");

    let mut missing: Vec<String> = Vec::new();
    let mut checked = 0usize;

    for (file, entry) in &ok {
        if ONLINE_MODEL_WHITELIST.contains(&entry.id.as_str()) {
            continue;
        }
        // 与 ModelRegistryEntry::dir 推导一致：target_dir 优先于 kind + sub_dir
        let dir = match entry.target_dir.as_deref() {
            Some(d) => models_dir.join(d),
            None => models_dir
                .join(kind_dir(&entry.kind))
                .join(entry.sub_dir.as_deref().unwrap_or(&entry.id)),
        };
        if !dir.is_dir() {
            missing.push(format!(
                "[{}] {} 模型目录不存在: {}",
                entry.id,
                file,
                dir.display()
            ));
            continue;
        }
        for f in &entry.files {
            if dir.join(&f.name).is_file() {
                checked += 1;
            } else {
                missing.push(format!("[{}] {} 缺少文件: {}", entry.id, file, f.name));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "以下 registry 声明与磁盘不一致（下载后会得到运行时永不加载的副本）:\n{}",
        missing.join("\n")
    );
    assert!(
        checked > 0,
        "未校验到任何文件，路径解析可能失效（仓库根 = {}）",
        root.display()
    );
    println!("registry 与磁盘一致，已校验文件数: {}", checked);
}

/// 契约 3：反序列化的 `engine` 字段能被 `model_use_case::parse_engine_kind` 覆盖。
///
/// 该映射表在 `votex-app` 内且为私有函数，无法直接调用；
/// 这里维护一份**镜像白名单**，新增 `EngineKind` 变体时必须同步两侧，
/// 否则 `model list` 会静默丢弃该条目（缺陷 F54）。
#[test]
fn registry_engine字段都在映射白名单内() {
    const PARSE_ENGINE_KIND_ALLOWED: &[&str] = &[
        // ==== 与 votex-app/src/use_case/model_use_case.rs::parse_engine_kind 保持一致 ====
        "Kokoro",
        "IndexTTS2",
        "IndexTTS25",
        "Whisper",
        "SenseVoice",
        "Paraformer",
        "Qwen3Asr",
        "PaddleOCR",
        "EasyOcr",
        "WeNet",
        "CosyVoice3",
        "FireRedAsr",
        "Qwen3Tts",
        "OpusMt",
        "QwenMt",
        "Nllb",
        "M2m100",
        "HyMt1_5",
        "CTranslate2",
        "OpenAi",
        "Ollama",
        "Pexels",
        "Pixabay",
        "Coverr",
        "OnnxRuntime",
    ];

    let (ok, errs) = parse_all();
    assert!(errs.is_empty(), "清单解析失败:\n{}", errs.join("\n"));

    let unmapped: Vec<String> = ok
        .values()
        .filter(|e| !PARSE_ENGINE_KIND_ALLOWED.contains(&e.engine.as_str()))
        .map(|e| format!("[{}] engine={}", e.id, e.engine))
        .collect();

    assert!(
        unmapped.is_empty(),
        "以下 engine 值无法被 parse_engine_kind 映射，会被 `model list` 静默丢弃（并在 \
         `model download` 时报\"未知模型\"）:\n{}",
        unmapped.join("\n")
    );
}