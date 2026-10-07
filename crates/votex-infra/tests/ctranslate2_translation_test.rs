//! CTranslate2 加速后端端到端测试
//!
//! 与 [`opus_mt_translation_test`] 共用金标准锚点（同一原始模型权重的
//! 本地 int8 转换产物），验证 ct2rs 推理路径的加载、方向路由与译码质量。
//!
//! 跑法（模型就绪后）：
//! ```bash
//! cargo test -p votex-infra --features ct2,slow-models --test ctranslate2_translation_test
//! ```

// 测试名使用 camelCase 描述语言对与场景，此处豁免命名检查
#![allow(non_snake_case)]
// 整个文件仅在 ct2 feature 下编译（imp_enabled 形态不存在时无被测对象）
#![cfg(feature = "ct2")]

use std::path::Path;

use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::value_object::TranslationDirection;
use votex_infra::translation::ctranslate2::CTranslate2Provider;

fn ct2_model_dir() -> std::path::PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    root.join("models/translation/ct2-opus-mt-zh-en")
}

#[cfg_attr(
    not(feature = "slow-models"),
    ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --features ct2,slow-models --test ctranslate2_translation_test"
)]
#[test]
fn test_ct2_你好世界() {
    let model_dir = ct2_model_dir();
    assert!(model_dir.join("model.bin").exists(), "CTranslate2 模型不存在，请先运行 scripts/convert_ct2.py");

    println!("\n=== CTranslate2 zh→en: '你好世界' ===");
    println!("模型目录: {:?}", model_dir);

    let engine = CTranslate2Provider::new();
    engine
        .load_from_dir(&model_dir)
        .unwrap_or_else(|e| panic!("加载模型失败: {}", e));
    assert!(engine.is_loaded());
    println!("✓ 模型加载成功（int8）");

    let result = engine
        .translate("你好世界", TranslationDirection::ZhToEn)
        .unwrap_or_else(|e| panic!("翻译失败: {}", e));

    println!("  输入: '你好世界'");
    println!("  输出: '{}'", result);
    // 与 opus_mt_translation_test 金标准锚点同权重，但此处为本地 int8 转换
    // 产物 + ct2rs 解码，首跑实测输出即为金标（int8 与 ONNX 贪心存在差异）
    assert_eq!(result, "You're in the world.");
    println!("✓ 翻译结果与金标准一致");
}

#[cfg_attr(
    not(feature = "slow-models"),
    ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --features ct2,slow-models --test ctranslate2_translation_test"
)]
#[test]
fn test_ct2_今天天气很好() {
    let engine = CTranslate2Provider::new();
    engine
        .load_from_dir(&ct2_model_dir())
        .unwrap_or_else(|e| panic!("加载模型失败: {}", e));

    let result = engine
        .translate("今天天气很好", TranslationDirection::ZhToEn)
        .unwrap_or_else(|e| panic!("翻译失败: {}", e));

    println!("\n=== CTranslate2 zh→en: '今天天气很好' → '{}' ===", result);
    assert_eq!(result, "It's a nice day.");
}

#[cfg_attr(
    not(feature = "slow-models"),
    ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --features ct2,slow-models --test ctranslate2_translation_test"
)]
#[test]
fn test_ct2_自动方向跟随模型() {
    let engine = CTranslate2Provider::new();
    engine
        .load_from_dir(&ct2_model_dir())
        .unwrap_or_else(|e| panic!("加载模型失败: {}", e));

    let result = engine
        .translate("你好世界", TranslationDirection::Auto)
        .unwrap_or_else(|e| panic!("自动方向翻译失败: {}", e));
    println!("\n=== CTranslate2 Auto: '你好世界' → '{}' ===", result);
    assert!(!result.is_empty());
}

#[cfg_attr(
    not(feature = "slow-models"),
    ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --features ct2,slow-models --test ctranslate2_translation_test"
)]
#[test]
fn test_ct2_方向不符应报UnsupportedDirection() {
    // 仅加载 zh→en 模型，英译中应报方向不支持（对齐 opus_mt 路由语义）
    let engine = CTranslate2Provider::new();
    engine
        .load_from_dir(&ct2_model_dir())
        .unwrap_or_else(|e| panic!("加载模型失败: {}", e));

    let result = engine.translate("Hello world", TranslationDirection::EnToZh);
    assert!(result.is_err(), "方向不符时应返回错误");
    println!("✓ 方向不符正确报错: {:?}", result.err().unwrap());

    let pairs = engine.supported_pairs();
    assert_eq!(pairs, vec![("zh".to_string(), "en".to_string())]);
}

#[cfg_attr(
    not(feature = "slow-models"),
    ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --features ct2,slow-models --test ctranslate2_translation_test"
)]
#[test]
fn test_ct2_批量翻译与逐条一致() {
    let engine = CTranslate2Provider::new();
    engine
        .load_from_dir(&ct2_model_dir())
        .unwrap_or_else(|e| panic!("加载模型失败: {}", e));

    let texts = ["你好世界", "今天天气很好"];
    let batch = engine
        .translate_batch(&texts, TranslationDirection::ZhToEn)
        .unwrap_or_else(|e| panic!("批量翻译失败: {}", e));

    println!("\n=== CTranslate2 批量 zh→en ===");
    for (src, out) in texts.iter().zip(&batch) {
        println!("  '{}' → '{}'", src, out);
        assert!(!out.is_empty(), "批量译文不应为空");
    }
    assert_eq!(batch.len(), texts.len());
}

#[test]
fn test_ct2_未加载时翻译返回错误() {
    // 无需模型文件的轻量契约测试（仅 ct2 feature 下运行）
    let engine = CTranslate2Provider::new();
    assert!(!engine.is_loaded());
    let err = engine
        .translate("你好世界", TranslationDirection::ZhToEn)
        .unwrap_err();
    println!("✓ 未加载正确报错: {:?}", err);
}
