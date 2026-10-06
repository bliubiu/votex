//! 翻译提供者模块
//!
//! 支持多种翻译引擎：
//! - LLM 在线翻译（DeepSeek）
//! - 离线词典翻译
//! - Opus-MT ONNX 离线翻译（中英双向）
//! - Qwen-MT DashScope API 在线翻译（多语言）
//! - NLLB-200 ONNX 离线翻译（200+ 语言）
//! - M2M-100 ONNX 离线翻译（100 语言）
//! - HY-MT1.5 Decoder-only ONNX 离线翻译
//!
//! # 模型会话池
//!
//! 离线模型动辄数百 MB~数 GB，每次翻译都重建 ONNX Session 会带来分钟级的加载开销。
//! [`TranslationModelPool`] 按引擎名缓存已加载的 Provider，实现「一次加载、反复使用」。

pub mod llm_translate;
pub mod dict_translate;
pub mod opus_mt;
pub mod qwen_mt;
pub mod nllb;
pub mod m2m100;
pub mod hy_mt;
pub mod ctranslate2;
pub mod model_pool;
pub mod cache;

use std::path::{Path, PathBuf};
use votex_domain::translation::provider::TranslationProvider;

pub use cache::TranslationCache;
pub use model_pool::TranslationModelPool;

/// 翻译引擎注册表：引擎名 → 可能的模型子目录名（按优先级排列）
///
/// 历史原因导致注册表 id、磁盘目录名、代码查找名可能不一致，
/// 这里统一列出候选名，逐个探测，避免「模型已下载却加载不到」。
struct EngineSpec {
    /// 引擎在 CLI / GUI 中使用的名字
    aliases: &'static [&'static str],
    /// 模型子目录候选名（相对于 models/translation/）
    dir_candidates: &'static [&'static str],
    /// 判断模型是否就绪所需的标志文件
    ready_marker: &'static str,
}

const ENGINE_SPECS: &[EngineSpec] = &[
    EngineSpec {
        aliases: &["hy-mt-1.5", "hymt1.5", "hy-mt", "hy-mt-1.5-1.8b"],
        dir_candidates: &["hy-mt-1.5", "hy-mt-1.5-1.8b"],
        ready_marker: "tokenizer.json",
    },
    EngineSpec {
        aliases: &["nllb-200", "nllb", "nllb-200-distilled-600m"],
        dir_candidates: &["nllb-200-distilled-600m"],
        ready_marker: "sentencepiece.bpe.model",
    },
    EngineSpec {
        aliases: &["m2m-100", "m2m100", "m2m-100-418m"],
        dir_candidates: &["m2m-100", "m2m-100-418m"],
        ready_marker: "tokenizer.json",
    },
];

/// 归一化引擎名（大小写、别名统一）
pub fn normalize_engine_name(name: &str) -> String {
    let lower = name.to_lowercase();
    let normalized = match lower.as_str() {
        "hymt1.5" | "hy-mt" | "hy-mt-1.5-1.8b" => "hy-mt-1.5",
        "nllb" | "nllb-200" | "nllb-200-distilled-600m" => "nllb-200",
        "m2m100" | "m2m-100-418m" => "m2m-100",
        "deepseek" => "llm",
        "dictionary" => "dict",
        "ct2" => "ctranslate2",
        other => other,
    };
    normalized.to_string()
}

/// 在候选目录中找到第一个「已就绪」的模型目录
///
/// 就绪判定：目录存在且包含引擎要求的标志文件。
fn resolve_model_dir(models_dir: &Path, candidates: &[&str], marker: &str) -> Option<PathBuf> {
    for candidate in candidates {
        let dir = models_dir.join("translation").join(candidate);
        if dir.join(marker).exists() {
            return Some(dir);
        }
    }
    None
}

/// 创建翻译提供者（使用默认模型目录 `models/`）
pub fn create_translation_provider(name: &str) -> Result<Box<dyn TranslationProvider>, String> {
    create_translation_provider_with_dir(name, Path::new("models"))
}

/// 创建翻译提供者，并自动加载本地模型（若已下载）
///
/// # 参数
/// - `name`: 引擎名或别名（如 "hy-mt-1.5" / "opus-mt" / "nllb"）
/// - `models_dir`: 模型根目录（其下按 `translation/<引擎>` 组织）
pub fn create_translation_provider_with_dir(
    name: &str,
    models_dir: &Path,
) -> Result<Box<dyn TranslationProvider>, String> {
    let engine = normalize_engine_name(name);

    match engine.as_str() {
        "llm" => Ok(Box::new(
            llm_translate::LlmTranslationProvider::new().map_err(|e| e.to_string())?,
        )),

        "dict" => Ok(Box::new(dict_translate::DictTranslationProvider::new())),

        "qwen-mt" => Ok(Box::new(
            qwen_mt::QwenMtProvider::new().map_err(|e| e.to_string())?,
        )),

        "ctranslate2" => Ok(Box::new(ctranslate2::CTranslate2Provider::new())),

        "opus-mt" => {
            let provider = opus_mt::OpusMtProvider::new();
            if let Some(dir) =
                resolve_model_dir(models_dir, &["opus-mt-zh-en"], "encoder_model.onnx")
            {
                provider
                    .load_zh_en_from_dir(&dir)
                    .map_err(|e| format!("加载 opus-mt zh→en 失败: {}", e))?;
            }
            if let Some(dir) =
                resolve_model_dir(models_dir, &["opus-mt-en-zh"], "encoder_model.onnx")
            {
                provider
                    .load_en_zh_from_dir(&dir)
                    .map_err(|e| format!("加载 opus-mt en→zh 失败: {}", e))?;
            }
            Ok(Box::new(provider))
        }

        _ => {
            let spec = ENGINE_SPECS
                .iter()
                .find(|s| s.aliases.iter().any(|a| *a == engine))
                .ok_or_else(|| format!("不支持的翻译引擎: {}", name))?;

            let dir = resolve_model_dir(models_dir, spec.dir_candidates, spec.ready_marker);
            create_offline_provider(&engine, dir.as_deref())
        }
    }
}

/// 创建离线 ONNX 引擎
///
/// 模型目录不存在时不报错，而是返回一个「未加载」的 Provider：
/// 这样 GUI 可以列出引擎并提示用户去下载模型，而不是直接崩溃。
fn create_offline_provider(
    engine: &str,
    model_dir: Option<&Path>,
) -> Result<Box<dyn TranslationProvider>, String> {
    macro_rules! build {
        ($provider:expr) => {{
            let provider = $provider;
            if let Some(dir) = model_dir {
                provider
                    .load_from_dir(dir)
                    .map_err(|e| format!("加载 {} 模型失败: {}", engine, e))?;
            }
            Ok(Box::new(provider))
        }};
    }

    match engine {
        "hy-mt-1.5" => build!(hy_mt::HyMtProvider::new()),
        "nllb-200" => build!(nllb::NllbProvider::new()),
        "m2m-100" => build!(m2m100::M2m100Provider::new()),
        _ => Err(format!("不支持的翻译引擎: {}", engine)),
    }
}

/// 列出所有可用引擎名（供 CLI / GUI 使用）
pub fn list_engine_names() -> Vec<&'static str> {
    vec!["dict", "opus-mt", "nllb-200", "m2m-100", "hy-mt-1.5", "qwen-mt", "llm"]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 引擎名归一化() {
        assert_eq!(normalize_engine_name("HY-MT-1.5"), "hy-mt-1.5");
        assert_eq!(normalize_engine_name("hymt1.5"), "hy-mt-1.5");
        assert_eq!(normalize_engine_name("hy-mt-1.5-1.8b"), "hy-mt-1.5");
        assert_eq!(normalize_engine_name("nllb"), "nllb-200");
        assert_eq!(normalize_engine_name("M2M100"), "m2m-100");
        assert_eq!(normalize_engine_name("deepseek"), "llm");
        assert_eq!(normalize_engine_name("dictionary"), "dict");
        assert_eq!(normalize_engine_name("ct2"), "ctranslate2");
    }

    #[test]
    fn 不支持的引擎返回错误() {
        // 注意：Result 的成功类型是 Box<dyn Trait>，不可 unwrap_err（要求 T: Debug）
        let result = create_translation_provider_with_dir("not-exist", Path::new("models"));
        match result {
            Ok(_) => panic!("不支持的引擎不应创建成功"),
            Err(e) => assert!(e.contains("不支持的翻译引擎"), "错误信息不符: {}", e),
        }
    }

    #[test]
    fn 词典引擎无需模型即可创建() {
        let provider = create_translation_provider_with_dir("dict", Path::new("models")).unwrap();
        assert_eq!(provider.name(), "dict");
        assert!(provider.is_loaded());
    }

    #[test]
    fn 模型目录缺失时离线引擎仍可创建但未加载() {
        let dir = std::env::temp_dir().join("votex-test-no-models");
        let provider = create_translation_provider_with_dir("nllb-200", &dir).unwrap();
        assert!(!provider.is_loaded(), "模型不存在时应处于未加载状态");
    }

    #[test]
    fn 候选目录探测_优先第一个就绪目录() {
        let base = tempfile::tempdir().unwrap();
        let second = base.path().join("translation").join("m2m-100-418m");
        std::fs::create_dir_all(&second).unwrap();
        std::fs::write(second.join("tokenizer.json"), "{}").unwrap();

        let found = resolve_model_dir(base.path(), &["m2m-100", "m2m-100-418m"], "tokenizer.json");
        assert_eq!(found.unwrap(), second, "应回退到第二个就绪的候选目录");
    }

    #[test]
    fn 候选目录探测_无就绪目录返回None() {
        let base = tempfile::tempdir().unwrap();
        assert!(resolve_model_dir(base.path(), &["m2m-100"], "tokenizer.json").is_none());
    }

    #[test]
    fn 引擎列表包含主要引擎() {
        let names = list_engine_names();
        for n in ["dict", "opus-mt", "nllb-200", "m2m-100", "hy-mt-1.5"] {
            assert!(names.contains(&n), "缺少引擎: {}", n);
        }
        assert!(
            !names.contains(&"ctranslate2"),
            "CTranslate2 未实现，不应出现在可选列表中"
        );
    }
}
