//! HY-MT1.5 翻译端到端测试
//!
//! 验证 HY-MT1.5-1.8B Decoder-only 离线翻译引擎的自回归流水线。
//! 注意：模型权重文件较大（model_q4.onnx_data ~1.4GB），首次运行需要确保已下载完成。

use std::path::Path;

use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::value_object::TranslationDirection;
use votex_infra::translation::hy_mt::HyMtProvider;

fn hy_mt_model_dir() -> std::path::PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    root.join("models/translation/hy-mt-1.5")
}

/// 检查模型文件是否齐全（不 panic，返回 bool）
fn model_files_ready() -> bool {
    let dir = hy_mt_model_dir();
    let model_ok = dir.join("model_q4.onnx").exists() && dir.join("model_q4.onnx_data").exists();
    let tokenizer_ok = dir.join("tokenizer.json").exists();
    model_ok && tokenizer_ok
}

fn load_hy_mt() -> HyMtProvider {
    let model_dir = hy_mt_model_dir();
    assert!(
        model_dir.join("model_q4.onnx").exists() || model_dir.join("model.onnx").exists(),
        "ONNX 模型文件不存在: {:?}",
        model_dir
    );
    assert!(
        model_dir.join("model_q4.onnx_data").exists(),
        "ONNX 模型权重文件不存在 (model_q4.onnx_data ~1.4GB, 需从 HF 下载)"
    );
    assert!(
        model_dir.join("tokenizer.json").exists(),
        "分词器文件不存在"
    );

    let mut engine = HyMtProvider::new();
    engine
        .load_from_dir(&model_dir)
        .unwrap_or_else(|e| panic!("加载 HY-MT1.5 模型失败: {}", e));
    engine
}

// ===================== 文件存在性测试 =====================

#[test]
fn test_hy_mt_model_files_exist() {
    let model_dir = hy_mt_model_dir();
    let model_ok = model_dir.join("model_q4.onnx").exists()
        || model_dir.join("model.onnx").exists();
    let weights_ok = model_dir.join("model_q4.onnx_data").exists();
    let tokenizer_ok = model_dir.join("tokenizer.json").exists();

    println!("HY-MT1.5 模型目录: {:?}", model_dir);
    println!("  ONNX 模型: {}", if model_ok { "✓" } else { "✗ 缺失" });
    println!("  ONNX 权重: {}", if weights_ok { "✓" } else { "✗ 缺失 (~1.4GB 需下载)" });
    println!("  Tokenizer: {}", if tokenizer_ok { "✓" } else { "✗ 缺失" });

    assert!(model_ok, "ONNX 模型文件不存在");
    assert!(tokenizer_ok, "分词器文件不存在");

    if !weights_ok {
        println!("⚠ 权重文件未下载，跳过推理测试。");
        println!("  下载命令:");
        println!("    Invoke-WebRequest -Uri .../onnx/model_q4.onnx_data -OutFile <model_dir>/model_q4.onnx_data");
    }
}

#[test]
fn test_hy_mt_load_model() {
    if !model_files_ready() {
        eprintln!("⚠ 模型文件不完整，跳过加载测试");
        return;
    }
    let _engine = load_hy_mt();
    println!("✓ HY-MT1.5 模型加载成功");
}

// ===================== 翻译推理测试 =====================

#[test]
fn test_hy_mt_translate_zh_en() {
    if !model_files_ready() {
        eprintln!("⚠ 模型文件不完整，跳过翻译测试");
        return;
    }
    let engine = load_hy_mt();

    println!("\n=== HY-MT1.5 zh→en: '你好世界' ===");

    let result = engine
        .translate("你好世界", TranslationDirection::ZhToEn)
        .unwrap_or_else(|e| panic!("翻译失败: {}", e));

    println!("  输入: '你好世界'");
    println!("  输出: '{}'", result);

    assert!(!result.is_empty(), "翻译结果不应为空");
    println!("✓ zh→en 翻译完成");
}

#[test]
fn test_hy_mt_translate_en_zh() {
    if !model_files_ready() {
        eprintln!("⚠ 模型文件不完整，跳过翻译测试");
        return;
    }
    let engine = load_hy_mt();

    println!("\n=== HY-MT1.5 en→zh: 'Hello world' ===");

    let result = engine
        .translate("Hello world", TranslationDirection::EnToZh)
        .unwrap_or_else(|e| panic!("翻译失败: {}", e));

    println!("  输入: 'Hello world'");
    println!("  输出: '{}'", result);

    assert!(!result.is_empty(), "翻译结果不应为空");
    println!("✓ en→zh 翻译完成");
}

#[test]
fn test_hy_mt_empty_text_should_error() {
    if !model_files_ready() {
        eprintln!("⚠ 模型文件不完整，跳过空文本测试");
        return;
    }
    let engine = load_hy_mt();

    let result = engine.translate("", TranslationDirection::ZhToEn);
    assert!(result.is_err(), "空文本应返回错误");
    println!("✓ 空文本正确返回错误");
}

#[test]
fn test_hy_mt_translate_sentence() {
    if !model_files_ready() {
        eprintln!("⚠ 模型文件不完整，跳过长句测试");
        return;
    }
    let engine = load_hy_mt();

    println!("\n=== HY-MT1.5 zh→en: '今天天气怎么样' ===");

    let result = engine
        .translate("今天天气怎么样", TranslationDirection::ZhToEn)
        .unwrap_or_else(|e| panic!("翻译失败: {}", e));

    println!("  输入: '今天天气怎么样'");
    println!("  输出: '{}'", result);

    assert!(!result.is_empty(), "翻译结果不应为空");
    println!("✓ 长句翻译完成");
}

#[test]
fn test_hy_mt_translate_paragraph() {
    if !model_files_ready() {
        eprintln!("⚠ 模型文件不完整，跳过段落测试");
        return;
    }
    let engine = load_hy_mt();

    println!("\n=== HY-MT1.5 zh→en: 段落翻译 ===");

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
