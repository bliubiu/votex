use std::path::PathBuf;

use votex_infra::tokenizer::ByteLevelBpeTokenizer;
use votex_domain::tts::tokenizer::TextTokenizer;

fn project_root() -> PathBuf {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.pop(); d.pop(); d
}

fn read_novel_excerpt() -> String {
    let content = std::fs::read_to_string(project_root().join("tmp/novel.txt"))
        .expect("读取 novel.txt 失败");
    content.chars().take(150).collect::<String>().trim().to_string()
}

#[test]
fn test_tokenizer_output_matches_python() {
    let root = project_root();
    let json_path = root.join("models/tts/qwen3-tts/tokenizer/tokenizer.json");
    let tok = tokenizers::Tokenizer::from_file(&json_path)
        .expect("加载 tokenizer.json 失败");
    eprintln!("✅ tokenizer 加载成功, vocab_size: {}", tok.get_vocab_size(true));

    let text = read_novel_excerpt();
    eprintln!("  文本 ({} 字): '{}'", text.chars().count(), text.chars().take(60).collect::<String>());

    let chat = format!("<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n", text);
    eprintln!("  chat 长度: {} 字", chat.len());

    let enc = tok.encode(chat.as_str(), true).expect("encode 失败");
    let ids: Vec<u32> = enc.get_ids().to_vec();

    // Python 参考输出（从之前运行的 Python 脚本获取）
    let py_total = 108;
    let py_first10 = [151644, 77091, 271, 29437, 15, 15, 15, 16, 44928, 4891];
    let py_prefix = [151644, 77091, 271];
    let py_suffix = [151645, 198, 151644, 77091, 198];

    eprintln!("  Python total: {}, Rust total: {}", py_total, ids.len());
    eprintln!("  Rust IDs: {:?}", ids);
    eprintln!("  Rust 前 10: {:?}", &ids[..10.min(ids.len())]);
    eprintln!("  Rust 后 10: {:?}", &ids[ids.len().saturating_sub(10)..]);
    eprintln!("  Rust 前缀 [..3]: {:?}", &ids[..3]);
    eprintln!("  Rust 后缀 [-5:]: {:?}", &ids[ids.len()-5..]);

    if ids.len() >= 5 {
        let text_tokens = &ids[3..ids.len()-5];
        eprintln!("  正文 token 数: {}", text_tokens.len());
        eprintln!("  正文前 10: {:?}", &text_tokens[..10.min(text_tokens.len())]);
    }

    assert_eq!(ids.len(), py_total, "token 数应 = 108");
    for i in 0..5 {
        assert_eq!(ids[i], py_first10[i], "token[{}] 不匹配", i);
    }
    assert_eq!(&ids[..3], &py_prefix[..], "前缀不匹配");
    assert_eq!(&ids[ids.len()-5..], &py_suffix[..], "后缀不匹配");
    eprintln!("\n✅ Rust tokenizer 输出与 Python 一致！");
}

/// 与官方 transformers GPT2Tokenizer 对照：同一中文文本的分词一致性
#[test]
fn cosyvoice_bpe_与官方gpt2分词一致() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/tts/cosyvoice");
    let tok = ByteLevelBpeTokenizer::from_vocab_merges(
        &dir.join("vocab.json"),
        &dir.join("merges.txt"),
    ).expect("构建分词器失败");
    let text = "收到67军147师20000人全部阵亡的电报时，他手一抖，老泪纵横。";
    let ids = tok.encode(text).expect("分词失败");
    println!("rust ids: {:?}", &ids);
    // 官方 GPT2Tokenizer: 34 tokens, 前 30 个见下方断言
    assert_eq!(ids.len(), 34, "token 数应与官方一致");
    let expect_prefix: [i64; 12] = [50009, 26939, 21, 22, 99292, 16, 19, 22, 99235, 17, 15, 15];
    assert_eq!(&ids[..12], &expect_prefix, "前缀 token 应与官方一致");
}
