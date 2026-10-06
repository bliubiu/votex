//! Qwen-MT 在线翻译 API 集成测试
//!
//! 前置条件：
//! - 设置环境变量 DASHSCOPE_API_KEY（阿里云百炼 API Key）
//! - 可访问互联网
//!
//! 测试覆盖：中→英、英→中、自动检测方向

use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::value_object::TranslationDirection;

/// 检查环境变量，跳过未配置 Key 的测试
fn check_env() -> Option<votex_infra::translation::qwen_mt::QwenMtProvider> {
    if std::env::var("DASHSCOPE_API_KEY").is_err() {
        eprintln!("跳过 Qwen-MT 测试：未设置 DASHSCOPE_API_KEY 环境变量");
        return None;
    }
    match votex_infra::translation::qwen_mt::QwenMtProvider::new() {
        Ok(provider) => Some(provider),
        Err(e) => {
            eprintln!("跳过 Qwen-MT 测试：创建提供者失败: {}", e);
            None
        }
    }
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test qwen_mt_translation_test --features slow-models")]
#[test]
fn test_qwen_mt_zh_to_en() {
    let provider = match check_env() {
        Some(p) => p,
        None => return,
    };

    let result = provider.translate("你好，世界！", TranslationDirection::ZhToEn);
    assert!(result.is_ok(), "中→英翻译失败: {:?}", result.err());
    let translated = result.unwrap();
    assert!(!translated.is_empty(), "翻译结果不应为空");
    println!("中→英: '你好，世界！' -> '{}'", translated);
    // 期望包含 "Hello" 或 "World"（大小写不敏感）
    let lower = translated.to_lowercase();
    assert!(
        lower.contains("hello") || lower.contains("world"),
        "翻译 '{}' 应包含 'Hello' 或 'World'",
        translated
    );
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test qwen_mt_translation_test --features slow-models")]
#[test]
fn test_qwen_mt_en_to_zh() {
    let provider = match check_env() {
        Some(p) => p,
        None => return,
    };

    let result = provider.translate("Hello, World!", TranslationDirection::EnToZh);
    assert!(result.is_ok(), "英→中翻译失败: {:?}", result.err());
    let translated = result.unwrap();
    assert!(!translated.is_empty(), "翻译结果不应为空");
    println!("英→中: 'Hello, World!' -> '{}'", translated);
    // 期望包含中文
    assert!(
        translated.contains("世界") || translated.contains("你好"),
        "翻译 '{}' 应包含中文",
        translated
    );
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test qwen_mt_translation_test --features slow-models")]
#[test]
fn test_qwen_mt_auto_direction() {
    let provider = match check_env() {
        Some(p) => p,
        None => return,
    };

    // 自动检测：输入中文，期望输出英文
    let result = provider.translate("今天天气真好。", TranslationDirection::Auto);
    assert!(result.is_ok(), "自动方向翻译失败: {:?}", result.err());
    let translated = result.unwrap();
    assert!(!translated.is_empty(), "翻译结果不应为空");
    println!("自动方向(中): '今天天气真好。' -> '{}'", translated);
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test qwen_mt_translation_test --features slow-models")]
#[test]
fn test_qwen_mt_empty_text() {
    let provider = match check_env() {
        Some(p) => p,
        None => return,
    };

    let result = provider.translate("", TranslationDirection::ZhToEn);
    assert!(result.is_err(), "空文本应返回错误");
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test qwen_mt_translation_test --features slow-models")]
#[test]
fn test_qwen_mt_long_text() {
    let provider = match check_env() {
        Some(p) => p,
        None => return,
    };

    let text = "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，\
                它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。\
                该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。";

    let result = provider.translate(text, TranslationDirection::ZhToEn);
    assert!(result.is_ok(), "长文本翻译失败: {:?}", result.err());
    let translated = result.unwrap();
    assert!(!translated.is_empty(), "翻译结果不应为空");
    println!("长文本翻译前 {} 字, 翻译后 {} 字", text.len(), translated.len());
}
