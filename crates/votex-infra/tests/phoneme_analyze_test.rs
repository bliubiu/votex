/// test pinyin to IPA mapping analysis
use std::collections::HashMap;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

fn load_pinyin_ipa_map() -> Option<HashMap<String, String>> {
    let path = workspace_root().join("models/kokoro-82m/pinyin_to_ipa.json");
    let content = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&content).ok()
}

#[test]
fn analyze_mapping_combining_chars() {
    let map = match load_pinyin_ipa_map() {
        Some(m) => m,
        None => {
            eprintln!("pinyin_to_ipa.json not found, skipping test");
            return;
        }
    };

    // ?????????Unicode Combining Diacritics: U+0300~U+036F, U+1DC0~U+1DFF, U+FE20~U+FE2F?
    let mut char_count: HashMap<u32, usize> = HashMap::new();
    let mut combined_entries: Vec<(String, u32, u32)> = Vec::new();  // (entry_key, first_cp, second_cp)

    for (key, val) in &map {
        let chars: Vec<u32> = val.chars().map(|c| c as u32).collect();
        for (i, &cp) in chars.iter().enumerate() {
            if (0x0300..=0x036F).contains(&cp) || (0x1DC0..=0x1DFF).contains(&cp) || (0xFE20..=0xFE2F).contains(&cp) {
                *char_count.entry(cp).or_insert(0) += 1;
                let prev = if i > 0 { chars[i-1] } else { 0 };
                combined_entries.push((key.clone(), prev, cp));
            }
        }
    }

    println!("=== ???????? ===");
    for (cp, count) in &char_count {
        let c = char::from_u32(*cp).unwrap();
        let hex_str = format!("U+{:04X}", cp);
        // ??????
        let precedents: std::collections::BTreeSet<u32> = combined_entries.iter()
            .filter(|e| e.2 == *cp)
            .map(|e| e.1)
            .collect();
        let prev_chars: Vec<String> = precedents.iter().map(|&p| {
            if let Some(ch) = char::from_u32(p) {
                format!("U+{:04X} '{}'", p, ch)
            } else {
                format!("U+{:04X}", p)
            }
        }).collect();
        println!("  {} '{}' ({}?) ????: {}", hex_str, c, count, prev_chars.join(", "));
    }

    // ????????
    println!("\n=== ????????????? ===");
    for (key, val) in &map {
        if val.contains('\u{FF0C}') || val.contains('\u{3002}') || val.contains('\u{FF01}') || val.contains('\u{FF1F}') {
            println!("  {} ? {}", key, val);
        }
    }

    // ?? VOCAB ? 10 ???
    println!("\n=== ???? 20 ? ===");
    for (i, (key, val)) in map.iter().enumerate().take(20) {
        println!("  {} ? {}", key, val);
    }

    println!("\n=== ????? ===");
    println!("  ???: {}", map.len());

    // ?? COMBINING VERTICAL LINE BELOW (U+0329)
    let syl_count = combined_entries.iter().filter(|e| e.2 == 0x0329).count();
    println!("  U+0329 (?????) ??: {} ?", syl_count);

    // ?? COMBINING INVERTED BREVE BELOW (U+032F)
    let breve_count = combined_entries.iter().filter(|e| e.2 == 0x032F).count();
    println!("  U+032F (?????) ??: {} ?", breve_count);
}
