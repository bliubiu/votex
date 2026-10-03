//! M2M-100 翻译端到端测试
//!
//! 验证 M2M-100-418M Encoder-Decoder 离线翻译引擎的自回归流水线。
//! 模型文件需从 Xenova/m2m100_418M ONNX 导出下载。

use std::path::Path;

use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::value_object::TranslationDirection;
use votex_infra::translation::m2m100::M2m100Provider;

fn model_dir() -> std::path::PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    root.join("models/translation/m2m-100")
}

fn model_files_ready() -> bool {
    let dir = model_dir();
    let encoder = dir.join("encoder_model.int8.onnx").exists()
        || dir.join("encoder_model.onnx").exists();
    let decoder = dir.join("decoder_model.int8.onnx").exists()
        || dir.join("decoder_model.onnx").exists();
    let tokenizer = dir.join("sentencepiece.bpe.model").exists();
    encoder && decoder && tokenizer
}

fn init_tracing() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_test_writer()
        .try_init();
}

fn load_engine() -> M2m100Provider {
    init_tracing();
    let dir = model_dir();
    assert!(
        dir.join("encoder_model.int8.onnx").exists() || dir.join("encoder_model.onnx").exists(),
        "编码器文件不存在: {:?}",
        dir
    );
    assert!(
        dir.join("decoder_model.int8.onnx").exists() || dir.join("decoder_model.onnx").exists(),
        "解码器文件不存在: {:?}",
        dir
    );
    assert!(
        dir.join("sentencepiece.bpe.model").exists(),
        "分词器文件不存在"
    );

    let mut engine = M2m100Provider::new();
    engine
        .load_from_dir(&dir)
        .unwrap_or_else(|e| panic!("加载 M2M-100 模型失败: {}", e));
    engine
}

#[test]
fn test_m2m100_model_files_exist() {
    let dir = model_dir();
    let encoder_ok = dir.join("encoder_model.int8.onnx").exists()
        || dir.join("encoder_model.onnx").exists();
    let decoder_ok = dir.join("decoder_model.int8.onnx").exists()
        || dir.join("decoder_model.onnx").exists();
    let tokenizer_ok = dir.join("sentencepiece.bpe.model").exists();

    println!("M2M-100 模型目录: {:?}", dir);
    println!("  编码器: {}", if encoder_ok { "✓" } else { "✗ 缺失" });
    println!("  解码器: {}", if decoder_ok { "✓" } else { "✗ 缺失" });
    println!("  分词器: {}", if tokenizer_ok { "✓" } else { "✗ 缺失" });

    assert!(encoder_ok, "编码器 ONNX 文件不存在");
    assert!(decoder_ok, "解码器 ONNX 文件不存在");
    assert!(tokenizer_ok, "分词器文件不存在");
}

#[test]
fn test_m2m100_load_model() {
    if !model_files_ready() {
        eprintln!("⚠ 模型文件不完整，跳过加载测试");
        return;
    }
    let _engine = load_engine();
    println!("✓ M2M-100 模型加载成功");
}

#[test]
fn test_m2m100_translate_zh_en() {
    if !model_files_ready() {
        eprintln!("⚠ 模型文件不完整，跳过翻译测试");
        return;
    }
    let engine = load_engine();

    println!("\n=== M2M-100 zh→en: '你好世界' ===");
    let result = engine
        .translate("你好世界", TranslationDirection::ZhToEn)
        .unwrap_or_else(|e| panic!("翻译失败: {}", e));
    println!("  输入: '你好世界'");
    println!("  输出: '{}'", result);
    assert!(!result.is_empty(), "翻译结果不应为空");
    println!("✓ zh→en 翻译完成");
}

#[test]
fn test_m2m100_translate_en_zh() {
    if !model_files_ready() {
        eprintln!("⚠ 模型文件不完整，跳过翻译测试");
        return;
    }
    let engine = load_engine();

    println!("\n=== M2M-100 en→zh: 'Hello world' ===");
    let result = engine
        .translate("Hello world", TranslationDirection::EnToZh)
        .unwrap_or_else(|e| panic!("翻译失败: {}", e));
    println!("  输入: 'Hello world'");
    println!("  输出: '{}'", result);
    assert!(!result.is_empty(), "翻译结果不应为空");
    println!("✓ en→zh 翻译完成");
}

#[test]
fn test_m2m100_empty_text() {
    if !model_files_ready() {
        eprintln!("⚠ 模型文件不完整，跳过空文本测试");
        return;
    }
    let engine = load_engine();
    let result = engine.translate("", TranslationDirection::ZhToEn);
    assert!(result.is_err(), "空文本应返回错误");
    println!("✓ 空文本正确返回错误");
}

#[test]
fn test_m2m100_translate_sentence() {
    if !model_files_ready() {
        eprintln!("⚠ 模型文件不完整，跳过长句测试");
        return;
    }
    let engine = load_engine();

    println!("\n=== M2M-100 zh→en: '今天天气怎么样' ===");
    let result = engine
        .translate("今天天气怎么样", TranslationDirection::ZhToEn)
        .unwrap_or_else(|e| panic!("翻译失败: {}", e));
    println!("  输入: '今天天气怎么样'");
    println!("  输出: '{}'", result);
    assert!(!result.is_empty(), "翻译结果不应为空");
    println!("✓ 长句翻译完成");
}

#[test]
fn test_m2m100_translate_paragraph() {
    if !model_files_ready() {
        eprintln!("⚠ 模型文件不完整，跳过段落测试");
        return;
    }
    let engine = load_engine();

    println!("\n=== M2M-100 zh→en: 段落翻译 ===");
    let text = "人工智能正在改变世界。深度学习模型可以处理自然语言、图像识别等复杂任务。";
    let result = engine
        .translate(text, TranslationDirection::ZhToEn)
        .unwrap_or_else(|e| panic!("翻译失败: {}", e));
    println!("  输入: '{}'", text);
    println!("  输出: '{}'", result);
    assert!(!result.is_empty(), "翻译结果不应为空");
    assert!(result.len() > 10, "段落翻译应包含至少 10 个字符");
    println!("✓ 段落翻译完成");
}
