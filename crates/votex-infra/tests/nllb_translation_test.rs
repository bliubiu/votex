//! NLLB-200 翻译端到端测试
//!
//! 验证 NLLB-200-distilled-600M 离线翻译引擎的编码器-解码器流水线。
//! 注意：该模型文件较大（~2GB），首次运行需要确保已下载完成。

use std::path::Path;

use votex_domain::model::registry::ModelRegistryEntry;
use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::value_object::TranslationDirection;
use votex_infra::translation::nllb::NllbProvider;

fn nllb_model_dir() -> std::path::PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    root.join("models/translation/nllb-200-distilled-600m")
}

fn nllb_registry_yaml() -> std::path::PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    root.join("models/registry/nllb-200-distilled-600m.yaml")
}

/// 从注册表 YAML 加载分词器配置
fn load_nllb_tokenizer_config() -> votex_domain::model::registry::TokenizerConfig {
    let yaml_path = nllb_registry_yaml();
    assert!(yaml_path.exists(), "NLLB 注册表文件不存在: {:?}", yaml_path);
    let content = std::fs::read_to_string(&yaml_path)
        .unwrap_or_else(|e| panic!("读取注册表文件失败: {}", e));
    let entry: ModelRegistryEntry = serde_yml::from_str(&content)
        .unwrap_or_else(|e| panic!("解析注册表 YAML 失败: {}", e));
    entry.tokenizer.unwrap_or_default()
}

fn load_nllb() -> NllbProvider {
    let model_dir = nllb_model_dir();
    let tokenizer_config = load_nllb_tokenizer_config();
    assert!(
        model_dir.join("encoder_model_int8.onnx").exists()
            || model_dir.join("encoder_model.onnx").exists(),
        "编码器模型不存在（已尝试 INT8 和 FP32）: {:?}",
        model_dir
    );
    assert!(
        model_dir.join("decoder_model_int8.onnx").exists()
            || model_dir.join("decoder_model.onnx").exists(),
        "解码器模型不存在（已尝试 INT8 和 FP32）: {:?}",
        model_dir
    );
    assert!(
        model_dir.join("sentencepiece.bpe.model").exists(),
        "分词器模型不存在"
    );

    let engine = NllbProvider::new()
        .with_tokenizer_config(tokenizer_config);
    engine
        .load_from_dir(&model_dir)
        .unwrap_or_else(|e| panic!("加载 NLLB 模型失败: {}", e));
    engine
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test nllb_translation_test --features slow-models")]
#[test]
fn test_nllb_model_files_exist() {
    let model_dir = nllb_model_dir();
    assert!(
        model_dir.join("encoder_model_int8.onnx").exists()
            || model_dir.join("encoder_model.onnx").exists(),
        "编码器模型不存在"
    );
    assert!(
        model_dir.join("decoder_model_int8.onnx").exists()
            || model_dir.join("decoder_model.onnx").exists(),
        "解码器模型不存在"
    );
    assert!(model_dir.join("sentencepiece.bpe.model").exists());
    println!("✓ NLLB 模型文件齐全 (INT8 优先)");
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test nllb_translation_test --features slow-models")]
#[test]
fn test_nllb_load_model() {
    let _engine = load_nllb();
    println!("✓ NLLB 模型加载成功");
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test nllb_translation_test --features slow-models")]
#[test]
fn test_nllb_translate_zh_en() {
    let engine = load_nllb();

    println!("\n=== NLLB-200 zh→en: '你好世界' ===");

    let result = engine
        .translate("你好世界", TranslationDirection::ZhToEn)
        .unwrap_or_else(|e| panic!("翻译失败: {}", e));

    println!("  输入: '你好世界'");
    println!("  输出: '{}'", result);

    assert!(!result.is_empty(), "翻译结果不应为空");
    println!("✓ zh→en 翻译完成");
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test nllb_translation_test --features slow-models")]
#[test]
fn test_nllb_translate_en_zh() {
    let engine = load_nllb();

    println!("\n=== NLLB-200 en→zh: 'Hello world' ===");

    let result = engine
        .translate("Hello world", TranslationDirection::EnToZh)
        .unwrap_or_else(|e| panic!("翻译失败: {}", e));

    println!("  输入: 'Hello world'");
    println!("  输出: '{}'", result);

    assert!(!result.is_empty(), "翻译结果不应为空");
    println!("✓ en→zh 翻译完成");
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test nllb_translation_test --features slow-models")]
#[test]
fn test_nllb_empty_text_should_error() {
    let engine = load_nllb();

    let result = engine.translate("", TranslationDirection::Auto);
    assert!(result.is_err(), "空文本应返回错误");
    println!("✓ 空文本正确返回错误");
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test nllb_translation_test --features slow-models")]
#[test]
fn test_nllb_translate_sentence() {
    let engine = load_nllb();

    println!("\n=== NLLB-200 zh→en: '今天天气怎么样' ===");

    let result = engine
        .translate("今天天气怎么样", TranslationDirection::ZhToEn)
        .unwrap_or_else(|e| panic!("翻译失败: {}", e));

    println!("  输入: '今天天气怎么样'");
    println!("  输出: '{}'", result);

    assert!(!result.is_empty(), "翻译结果不应为空");
    println!("✓ 长句翻译完成");
}
