//! Opus-MT 双向翻译端到端测试
//!
//! 验证 Opus-MT (zh↔en) 离线翻译引擎的编码器-解码器流水线。
//! zh→en 使用 golden 数据中已验证过的测试用例。
//! en→zh 使用独立验证数据。

use std::path::Path;

use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::value_object::TranslationDirection;
use votex_infra::translation::opus_mt::OpusMtProvider;

fn zh_en_model_dir() -> std::path::PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    root.join("models/translation/opus-mt-zh-en")
}

fn en_zh_model_dir() -> std::path::PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    root.join("models/translation/opus-mt-en-zh")
}

fn load_bidirectional() -> OpusMtProvider {
    let mut engine = OpusMtProvider::new();
    engine
        .load_zh_en_from_dir(&zh_en_model_dir())
        .unwrap_or_else(|e| panic!("加载 zh→en 模型失败: {}", e));
    engine
        .load_en_zh_from_dir(&en_zh_model_dir())
        .unwrap_or_else(|e| panic!("加载 en→zh 模型失败: {}", e));
    engine
}

#[test]
fn test_opus_mt_你好世界() {
    let model_dir = zh_en_model_dir();
    assert!(model_dir.join("encoder_model.onnx").exists(), "编码器模型不存在");
    assert!(model_dir.join("decoder_model.onnx").exists(), "解码器模型不存在");

    println!("\n=== Opus-MT zh→en: '你好世界' ===");
    println!("模型目录: {:?}", model_dir);

    let mut engine = OpusMtProvider::new();
    engine
        .load_from_dir(&model_dir)
        .unwrap_or_else(|e| panic!("加载模型失败: {}", e));
    println!("✓ 模型加载成功");

    let result = engine
        .translate("你好世界", TranslationDirection::ZhToEn)
        .unwrap_or_else(|e| panic!("翻译失败: {}", e));

    println!("  输入: '你好世界'");
    println!("  输出: '{}'", result);

    // 根据 golden 数据，预期输出 "- Good world."
    assert_eq!(result, "- Good world.");
    println!("✓ 翻译结果与 golden 数据一致");
}

#[test]
fn test_opus_mt_今天天气很好() {
    let model_dir = zh_en_model_dir();
    assert!(model_dir.join("encoder_model.onnx").exists(), "编码器模型不存在");
    assert!(model_dir.join("decoder_model.onnx").exists(), "解码器模型不存在");

    println!("\n=== Opus-MT zh→en: '今天天气很好' ===");
    println!("模型目录: {:?}", model_dir);

    let mut engine = OpusMtProvider::new();
    engine
        .load_from_dir(&model_dir)
        .unwrap_or_else(|e| panic!("加载模型失败: {}", e));
    println!("✓ 模型加载成功");

    let result = engine
        .translate("今天天气很好", TranslationDirection::ZhToEn)
        .unwrap_or_else(|e| panic!("翻译失败: {}", e));

    println!("  输入: '今天天气很好'");
    println!("  输出: '{}'", result);

    // 根据 golden 数据，预期输出 "It's a nice day."
    assert_eq!(result, "It's a nice day.");
    println!("✓ 翻译结果与 golden 数据一致");
}

#[test]
fn test_opus_mt_自动检测方向() {
    let model_dir = zh_en_model_dir();

    let mut engine = OpusMtProvider::new();
    engine
        .load_from_dir(&model_dir)
        .unwrap_or_else(|e| panic!("加载模型失败: {}", e));

    let result = engine
        .translate("你好世界", TranslationDirection::Auto)
        .unwrap_or_else(|e| panic!("自动方向翻译失败: {}", e));

    println!("  自动方向输入: '你好世界'");
    println!("  输出: '{}'", result);
    assert_eq!(result, "- Good world.");
}

#[test]
fn test_opus_mt_英译中未加载模型应报错() {
    let model_dir = zh_en_model_dir();

    let mut engine = OpusMtProvider::new();
    engine
        .load_from_dir(&model_dir)
        .unwrap_or_else(|e| panic!("加载模型失败: {}", e));

    let result = engine.translate("Hello world", TranslationDirection::EnToZh);
    assert!(result.is_err(), "未加载 en→zh 模型时应返回错误");
    println!("✓ 英译中正确返回错误: {:?}", result.err().unwrap());
}

#[test]
fn test_opus_mt_空文本应报错() {
    let model_dir = zh_en_model_dir();

    let mut engine = OpusMtProvider::new();
    engine
        .load_from_dir(&model_dir)
        .unwrap_or_else(|e| panic!("加载模型失败: {}", e));

    let result = engine.translate("", TranslationDirection::ZhToEn);
    assert!(result.is_err(), "空文本应该返回错误");
    println!("✓ 空文本正确返回错误");
}

#[test]
fn test_opus_mt_en_zh_HelloWorld() {
    let engine = load_bidirectional();

    println!("\n=== Opus-MT en→zh: 'Hello world' ===");

    let result = engine
        .translate("Hello world", TranslationDirection::EnToZh)
        .unwrap_or_else(|e| panic!("翻译失败: {}", e));

    println!("  输入: 'Hello world'");
    println!("  输出: '{}'", result);

    // en→zh 模型输出应为中文
    assert!(!result.is_empty(), "en→zh 翻译结果不应为空");
    assert!(!result.contains("Hello"), "结果应为中文而非英文");
    println!("✓ en→zh 翻译结果包含中文");
}

#[test]
fn test_opus_mt_en_zh_good_morning() {
    let engine = load_bidirectional();

    println!("\n=== Opus-MT en→zh: 'Good morning' ===");

    let result = engine
        .translate("Good morning", TranslationDirection::EnToZh)
        .unwrap_or_else(|e| panic!("翻译失败: {}", e));

    println!("  输入: 'Good morning'");
    println!("  输出: '{}'", result);

    assert!(!result.is_empty(), "en→zh 翻译结果不应为空");
    println!("✓ en→zh 翻译完成");
}

#[test]
fn test_opus_mt_bidirectional_roundtrip() {
    let engine = load_bidirectional();

    println!("\n=== Opus-MT 双向回环测试 ===");

    // zh → en
    let en = engine
        .translate("今天天气很好", TranslationDirection::ZhToEn)
        .unwrap_or_else(|e| panic!("zh→en 翻译失败: {}", e));
    println!("  zh→en: '今天天气很好' → '{}'", en);
    assert_eq!(en, "It's a nice day.");

    // en → zh
    let zh = engine
        .translate("Hello world", TranslationDirection::EnToZh)
        .unwrap_or_else(|e| panic!("en→zh 翻译失败: {}", e));
    println!("  en→zh: 'Hello world' → '{}'", zh);

    assert!(!en.is_empty());
    assert!(!zh.is_empty());
    println!("✓ 双向翻译测试完成");
}
