//! docs/20 F67 取证：与官方 Python 参考逐项比对的 Rust 侧数据
//!
//! 官方参考 `models/tts/cosyvoice/scripts/onnx_inference_pure.py` 的 tokenize_text 是
//! `self.tokenizer.encode(text, add_special_tokens=False)`，其中 tokenizer 走 HF
//! `GPT2Tokenizer.from_pretrained(model_dir)`（读 `vocab.json` + `merges.txt`）。
//!
//! Rust 侧 `cosyvoice.rs:1507` 用 `ByteLevelBpeTokenizer::from_vocab_merges` 构建，
//! pre_tokenizer/decoder/post_processor 均为 `ByteLevel::new(false, false, true)`
//! —— 与 HF GPT2Tokenizer 的默认配置等价（add_prefix_space=false、use_regex=true）。
//!
//! 本测试打印与 Python 完全相同的字段（token id 列表 / 长度），用于逐元素 diff。
//! 若两侧 token id 完全一致，则「分词器不是 F67 根因」得到**跨实现**确认。
//!
//! 模型缺失则跳过（模型不随二进制分发）。

use votex_domain::tts::tokenizer::TextTokenizer;
use votex_infra::tokenizer::ByteLevelBpeTokenizer;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn read_utf8(p: &std::path::Path) -> String {
    std::fs::read_to_string(p)
        .unwrap_or_else(|e| panic!("读取 {:?} 失败: {}", p, e))
        .trim()
        .to_string()
}

/// 只打印码点，避免控制台编码往返
fn codepoints(s: &str) -> String {
    s.chars()
        .map(|c| format!("{:04X}", c as u32))
        .collect::<Vec<_>>()
        .join(" ")
}

#[test]
fn docs20_f67_分词结果与官方参考逐项可比对() {
    let root = repo_root();
    let vocab = root.join("models/tts/cosyvoice/vocab.json");
    let merges = root.join("models/tts/cosyvoice/merges.txt");
    if !vocab.is_file() || !merges.is_file() {
        eprintln!("跳过：词表不在 {}", vocab.display());
        return;
    }

    let target_path = root.join("tmp/e2e_pl_tiny.txt");
    let prompt_path = root.join("models/tts/cosyvoice/prompts/zh_prompt.txt");
    if !target_path.is_file() || !prompt_path.is_file() {
        eprintln!(
            "跳过：语料缺失（target={} prompt={}）",
            target_path.is_file(),
            prompt_path.is_file()
        );
        return;
    }

    // 与 cosyvoice.rs:1507 完全相同的构建路径
    let tok = ByteLevelBpeTokenizer::from_vocab_merges(&vocab, &merges)
        .expect("构建 ByteLevel BPE 分词器失败");

    let target_text = read_utf8(&target_path);
    let prompt_text = read_utf8(&prompt_path);

    println!("[INPUT] target chars={}", target_text.chars().count());
    println!("[INPUT] prompt chars={}", prompt_text.chars().count());
    println!("[INPUT] target codepoints={}", codepoints(&target_text));
    println!("[INPUT] prompt codepoints={}", codepoints(&prompt_text));

    let prompt_ids = tok.encode(&prompt_text).expect("编码 prompt 文本失败");
    let target_ids = tok.encode(&target_text).expect("编码目标文本失败");

    println!("[TOK] prompt_text_tokens={} ids={:?}", prompt_ids.len(), prompt_ids);
    println!("[TOK] tts_text_tokens={} ids={:?}", target_ids.len(), target_ids);

    let combined_len = prompt_ids.len() + target_ids.len();
    println!("[TOK] combined={}", combined_len);

    // 与 release 日志 `分段量纲 text_emb: shape=[57, 896]` 对照：
    // 57 = prompt(42) + target(15)
    assert_eq!(
        combined_len,
        57,
        "合并 token 数与 release 日志记录的 text_emb 长度 57 不符，分词行为已变化"
    );
    assert_eq!(
        target_ids.len(),
        target_text.chars().count(),
        "目标文本 token 数应等于字符数（中文 1 token/字），实际 {} vs {}",
        target_ids.len(),
        target_text.chars().count()
    );
}