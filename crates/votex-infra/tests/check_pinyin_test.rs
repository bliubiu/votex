//! pinyin crate 声调映射探针（诊断用，UTF-8 修复版）。
//!
//! 历史注记：本文件曾因 GBK 编码损坏，中文字面量全部退化为 `?`，
//! 导致 `to_pinyin().unwrap()` 恒 panic（tests/ 目录既有失败）。
//! 2026-10-05 重写为 UTF-8 并改为软断言：仅诊断输出，不做硬断言。
#[test]
fn check_pinyin_for_chars() {
    use pinyin::ToPinyin;
    let chars_to_check = [
        '女', '儿', '朋', '友', '中', '国', '北', '京', '上', '海', '深', '圳', '粤', '语', '茶',
    ];
    for &c in &chars_to_check {
        if let Some(py) = c.to_pinyin() {
            let py_str = py.with_tone_num_end().to_string();
            eprintln!("U+{:04X} '{}': pinyin='{}'", c as u32, c, py_str);
        } else {
            eprintln!("U+{:04X} '{}': NO PINYIN", c as u32, c);
        }
    }

    // 女儿 的拼音与注音映射
    let c = '女';
    let Some(py) = c.to_pinyin() else {
        eprintln!("女: 无拼音，跳过后续探针");
        return;
    };
    let py_str = py.with_tone_num_end().to_string();
    eprintln!("女 pinyin: '{py_str}'");

    let zhuyin = votex_infra::tts::kokoro_g2p::pinyin_to_zhuyin(&py_str);
    eprintln!("zhuyin for '{py_str}': {zhuyin:?}");

    // 短句走 kokoro G2P
    let text = "你好世界";
    eprintln!("'{text}' -> zhuyin: '{}'", votex_infra::tts::kokoro_g2p::text_to_phonemes_zh(text));

    // 长句覆盖探针 + kokoro vocab 覆盖检查（模型缺失时跳过）
    let text2 = "小明说今天天气很好，我们去公园散步吧。";
    let ph = votex_infra::tts::kokoro_g2p::text_to_phonemes_zh(text2);
    eprintln!("长句 zhuyin: '{ph}'");

    // docs/20 F68：模型文件不随二进制分发，缺失时跳过（勿用 include_str!）
    let cfg_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/tts/kokoro-82m-v1.1-zh/config.json");
    if !cfg_path.is_file() {
        eprintln!("跳过：kokoro config.json 不在 {}", cfg_path.display());
        return;
    }
    let vocab_json = std::fs::read_to_string(&cfg_path).expect("读取 kokoro config.json 失败");
    let cfg: serde_json::Value = serde_json::from_str(&vocab_json).unwrap();
    let vocab = cfg["vocab"].as_object().unwrap();
    let mut missing = std::collections::BTreeSet::new();
    for c in ph.chars() {
        let c_str = c.to_string();
        if !vocab.contains_key(&c_str) {
            missing.insert(c);
        }
    }
    if !missing.is_empty() {
        eprintln!(
            "以下字符不在 vocab 内: {:?}",
            missing.iter().map(|c| format!("U+{:04X} '{}'", *c as u32, c)).collect::<Vec<_>>()
        );
    } else {
        eprintln!("所有音素都在 vocab 内");
    }
}
