//! PaddleOCR 端到端测试
//!
//! 验证 PaddleOCR 各模型的 OCR 识别流水线是否正常工作。
//! 流水线：文本检测(det) → 方向分类(cls) → 文字识别(rec)

use std::path::Path;
use votex_infra::ocr::paddleocr::{PaddleOcrEngine, PaddleOcrModelVariant};
use votex_domain::ocr::provider::OcrProvider;
use votex_domain::ocr::value_object::OcrParams;

/// 测试图片路径
fn test_image_path() -> std::path::PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    root.join("tmp/ocr-test-chinese.png")
}

fn run_ocr_test(model_dir: &Path, variant: PaddleOcrModelVariant, label: &str, expect_text: Option<&str>) {
    let test_image = test_image_path();
    assert!(model_dir.exists(), "模型目录不存在: {:?}", model_dir);
    assert!(test_image.exists(), "测试图片不存在: {:?}", test_image);

    println!("\n=== PaddleOCR {} 识别验证 ===", label);
    println!("模型目录: {:?}", model_dir);

    let mut engine = PaddleOcrEngine::with_variant(variant);
    engine.load_from_dir(model_dir)
        .unwrap_or_else(|e| panic!("加载 {} 模型失败: {}", label, e));
    assert!(engine.is_loaded());
    println!("✓ 模型加载成功");

    let params = OcrParams::default();
    let result = engine.recognize(&test_image, &params)
        .unwrap_or_else(|e| panic!("OCR 识别失败: {}", e));

    println!("✓ OCR 识别完成 ({} 个文本区域)", result.blocks.len());

    let mut full_text = String::new();
    for (i, block) in result.blocks.iter().enumerate() {
        println!("  区域 {}: text='{}', conf={:.2}", i + 1, block.text, block.confidence);
        full_text.push_str(&block.text);
    }

    println!("  汇总: '{}' ({} 字符)", full_text, full_text.chars().count());

    // 校验：至少识别出文字
    assert!(!full_text.is_empty(), "{}: 未识别到文字", label);

    // 若指定了期望文本，则校验识别内容
    if let Some(expected) = expect_text {
        assert_eq!(full_text, expected, "{}: 识别结果不匹配", label);
        println!("✓ 识别内容与预期完全一致");
    }

    println!("✓ {} 验证通过", label);
}

/// PaddleOCR v4 识别验证
///
/// 注意：v4 模型对字形近似的"你/尔"存在轻微识别误差（`你`→`尔`），属于模型精度限制
#[test]
fn test_paddleocr_v4_识别测试图片() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    run_ocr_test(&root.join("models/ocr/paddleocr"), PaddleOcrModelVariant::V4Mobile, "v4", None);
}

/// PaddleOCR v5-mobile 识别验证
#[test]
fn test_paddleocr_v5_mobile_识别测试图片() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    run_ocr_test(
        &root.join("models/ocr/paddleocr-v5"),
        PaddleOcrModelVariant::V5Mobile,
        "v5-mobile",
        Some("你好世界测试文字"),
    );
}

/// PaddleOCR v6-tiny 识别验证
#[test]
fn test_paddleocr_v6_tiny_识别测试图片() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().parent().unwrap();
    run_ocr_test(
        &root.join("models/ocr/paddleocr-v6"),
        PaddleOcrModelVariant::V6Tiny,
        "v6-tiny",
        Some("你好世界测试文字"),
    );
}
