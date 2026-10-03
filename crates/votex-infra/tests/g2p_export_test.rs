/// ???? G2P ?????? Python ?????
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

fn load_novel_sample(path: &PathBuf, start: usize, end: usize) -> String {
    let content = std::fs::read_to_string(path).unwrap();
    let lines: Vec<&str> = content.lines().collect();
    let sample: Vec<&str> = lines.iter()
        .skip(start.saturating_sub(1))
        .take(end - start + 1)
        .copied()
        .collect();
    let cleaned: Vec<&str> = sample.iter().map(|s| s.trim()).filter(|s| !s.is_empty()).collect();
    cleaned.join("")
}

#[test]
fn export_g2p_phonemes() {
    let novel_path = workspace_root().join("tmp/novel.txt");
    
    let samples = vec![
        ("????", load_novel_sample(&novel_path, 3, 4)),
        ("??", load_novel_sample(&novel_path, 37, 38)),
        ("????", load_novel_sample(&novel_path, 50, 54)),
        ("????", load_novel_sample(&novel_path, 66, 71)),
    ];
    
    eprintln!("\n=== G2P ?????? Python ???===\n");
    
    for (name, text) in &samples {
        let phonemes = votex_infra::tts::kokoro_g2p::text_to_phonemes_zh(text);
        println!("[{}]??({}?): {}", name, text.chars().count(), text);
        println!("[{}]??({}): {}", name, phonemes.chars().count(), phonemes);
        println!();
    }
}
