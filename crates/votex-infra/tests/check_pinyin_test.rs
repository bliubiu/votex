/// ?? pinyin crate ????????
#[test]
fn check_pinyin_for_chars() {
    use pinyin::ToPinyin;
    let chars_to_check = ['?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?', '?'];
    for &c in &chars_to_check {
        if let Some(py) = c.to_pinyin() {
            let py_str = py.with_tone_num_end().to_string();
            eprintln!("U+{:04X} '{}': pinyin='{}'", c as u32, c, py_str);
        } else {
            eprintln!("U+{:04X} '{}': NO PINYIN", c as u32, c);
        }
    }
    
    // ???? ?
    let c = '?';
    let py = c.to_pinyin().unwrap();
    let py_str = py.with_tone_num_end().to_string();
    eprintln!("\n? pinyin: '{}'", py_str);
    
    // ?? zhuyin mapping for nv3
    let zhuyin = votex_infra::tts::kokoro_g2p::pinyin_to_zhuyin(&py_str);
    eprintln!("zhuyin for '{}': {:?}", py_str, zhuyin);
    
    // ???????? G2P
    let text = "????";
    eprintln!("\n'{}' -> zhuyin: '{}'", text, votex_infra::tts::kokoro_g2p::text_to_phonemes_zh(text));
    
    // ????????G2P
    let text2 = "????????????????????????????????????";
    let ph = votex_infra::tts::kokoro_g2p::text_to_phonemes_zh(text2);
    eprintln!("\n??? zhuyin: '{}'", ph);
    
    // ????????? vocab ?
    //
    // docs/20 F68：① 原路径 `models/kokoro-82m-v1.1-zh/config.json` 写错（漏 `tts/`），
    // `include_str!` 在**编译期**报 `os error 3`，直接阻断 `cargo check --workspace
    // --all-targets`；② 模型文件不随二进制分发，`include_str!` 会让「没下载模型的机器」
    // 连编译都过不了 —— 改为运行时读取并在模型缺失时跳过。
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
        eprintln!("\n?? ?? vocab ???: {:?}", missing.iter().map(|c| format!("U+{:04X} '{}'", *c as u32, c)).collect::<Vec<_>>());
    } else {
        eprintln!("\n? ?????? vocab ?");
    }
}
