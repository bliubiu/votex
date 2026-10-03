/// Phoneme diagnostic: G2P output vs Kokoro VOCAB
use std::collections::HashMap;
use std::sync::LazyLock;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

fn load_config_json() -> Option<serde_json::Value> {
    let path = workspace_root().join("models/kokoro-82m/config.json");
    let content = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Kokoro VOCAB (loaded from config.json at runtime)
static VOCAB: LazyLock<HashMap<char, i64>> = LazyLock::new(|| {
    let cfg = load_config_json().expect("kokoro-82m/config.json not found (run model download first)");
    let vocab = cfg["vocab"].as_object().expect("vocab field missing in config.json");
    let mut map = HashMap::with_capacity(vocab.len());
    for (ch, id) in vocab {
        let chars: Vec<char> = ch.chars().collect();
        if chars.len() == 1 {
            map.insert(chars[0], id.as_i64().unwrap_or(0));
        }
    }
    map
});

#[test]
fn phoneme_vocab_diagnostics() {
    let texts = [
        "????????????",
        "??",
        "??",
        "??",
        "??",
        "??",
        "?",
    ];

    for &text in &texts {
        let phonemes = votex_infra::tts::kokoro_g2p::text_to_phonemes(text);
        println!("??: '{}'", text);
        println!("phonemes: '{}'", phonemes);
        println!("phonemes ??: {}", phonemes.len());
        
        // ????????? VOCAB ?
        let mut missing = Vec::new();
        let mut present = Vec::new();
        for c in phonemes.chars() {
            if VOCAB.contains_key(&c) {
                present.push(c);
            } else {
                missing.push(c);
            }
        }
        
        if missing.is_empty() {
            println!("  ? ?? {} ???? VOCAB ?", phonemes.len());
        } else {
            println!("  ? ?? {} ???? VOCAB ?:", missing.len());
            for c in &missing {
                let hex: Vec<String> = c.escape_unicode().map(|e| format!("{}", e)).collect();
                println!("    U+{:04X} ({})", *c as u32, hex.join(""));
            }
        }
        println!("  VOCAB ???: {:?}", present.iter().collect::<String>());
        println!();
    }

    // ??????
    println!("=== VOCAB ?????? ===");
    let special = ['\u{032F}', '\u{02E5}', '\u{02E7}', '\u{02E9}', '\u{2192}', '\u{2197}', '\u{2193}', '\u{2198}', ';'];
    for &c in &special {
        match VOCAB.get(&c) {
            Some(id) => println!("  U+{:04X} '{}' ? token ID {}", c as u32, c, id),
            None => println!("  U+{:04X} '{}' ? ? ?? VOCAB ?", c as u32, c),
        }
    }

    // ?? VOCAB ????? Unicode ???? ASCII + ? CJK?
    println!("\n=== VOCAB ???? ASCII ? CJK ?? ===");
    let mut special_chars: Vec<_> = VOCAB.keys()
        .filter(|c| **c > '\u{007F}' && (**c < '\u{4E00}' || **c > '\u{9FFF}'))
        .collect();
    special_chars.sort_by(|a, b| (**a as u32).cmp(&(**b as u32)));
    for c in special_chars {
        let hex = format!("U+{:04X}", *c as u32);
        println!("  {} '{}' ? ID {}", hex, c, VOCAB.get(c).unwrap());
    }
}
