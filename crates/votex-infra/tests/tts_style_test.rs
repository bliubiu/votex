/// Kokoro style vector test
use std::path::PathBuf;
use std::collections::HashMap;
use std::sync::LazyLock;

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

fn phonemes_to_token_ids(phonemes: &str) -> Vec<i64> {
    let mut ids = vec![0i64];
    for c in phonemes.chars() {
        if let Some(&token_id) = VOCAB.get(&c) {
            ids.push(token_id);
        }
    }
    ids.push(0);
    ids
}

#[test]
fn test_different_style_shapes() {
    use ndarray::Array2;
    use ort::value::Value;

    let model_path = {
        let mut p = workspace_root();
        p.push("models/kokoro-82m/kokoro-v1.0.int8.onnx");
        p
    };
    if !model_path.exists() {
        eprintln!("? ???????");
        return;
    }

    let voice_path = {
        let mut p = workspace_root();
        p.push("models/kokoro-82m/voices/af_bella.bin");
        p
    };
    let data = std::fs::read(&voice_path).unwrap();
    let voice_floats: Vec<f32> = data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();

    // ????????? "?" ? token_ids = [0, 56, 51, 169, 0]
    let test_text = "?";
    let phonemes = votex_infra::tts::kokoro_g2p::text_to_phonemes(test_text);
    let token_ids = phonemes_to_token_ids(&phonemes);
    let seq_len = token_ids.len();
    eprintln!("??: '{}'", test_text);
    eprintln!("phonemes: '{}'", phonemes);
    eprintln!("token_ids: {:?} (len={})", token_ids, seq_len);

    votex_infra::shared::ensure_ort_dylib_path();
    let mut session = ort::session::Session::builder()
        .unwrap()
        .commit_from_file(&model_path)
        .expect("?? ONNX ????");

    let input_ids = Array2::from_shape_vec((1, seq_len), token_ids).unwrap();
    let input_ids_value = Value::from_array(input_ids).unwrap();

    // ???? style ??
    let style_variants = [
        ("style[1,256] pack[0]", 0usize, false),       // ?????
        ("style[1,256] pack[12]", 12usize, false),     // phonemes_len-1
        ("style[seq_len,256]", 0, true),                // ? token ??
    ];

    for (label, pack_idx, repeat) in &style_variants {
        eprintln!("\n--- ??: {} ---", label);

        let style_slice = if *repeat {
            // ?? [seq_len, 256]?? pack[0] ?? seq_len ?
            let base = &voice_floats[0..256];
            let mut all_styles = Vec::with_capacity(seq_len * 256);
            for _ in 0..seq_len {
                all_styles.extend_from_slice(base);
            }
            let style_array = Array2::from_shape_vec((seq_len, 256), all_styles).unwrap();
            match Value::from_array(style_array) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("  ? style[seq_len,256] ????: {}", e);
                    None
                }
            }
        } else {
            let offset = *pack_idx * 256;
            let style: Vec<f32> = voice_floats[offset..offset+256].to_vec();
            let style_array = Array2::from_shape_vec((1, 256), style).unwrap();
            match Value::from_array(style_array) {
                Ok(v) => Some(v),
                Err(e) => {
                    eprintln!("  ? style[1,256] ????: {}", e);
                    None
                }
            }
        };

        if let Some(style_value) = style_slice {
            let speed_array = ndarray::arr1(&[1.0f32]);
            let speed_value = Value::from_array(speed_array).unwrap();

            match session.run(ort::inputs![input_ids_value.clone(), style_value, speed_value]) {
                Ok(outputs) => {
                    if let Ok(audio) = outputs[0].try_extract_array::<f32>() {
                        let shape = audio.shape();
                        let num_samples = shape[1];
                        let mut samples = Vec::with_capacity(num_samples);
                        for i in 0..num_samples {
                            samples.push(audio[[0, i]]);
                        }
                        let max_amp = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                        let rms = (samples.iter().map(|s| s * s).sum::<f32>() / num_samples as f32).sqrt();
                        eprintln!("  ? ??: {} ?? ({}ms), max_amp={:.6}, rms={:.6}",
                            num_samples, num_samples * 1000 / 24000, max_amp, rms);
                    }
                }
                Err(e) => {
                    eprintln!("  ? ????: {}", e);
                }
            }
        }
    }
}
