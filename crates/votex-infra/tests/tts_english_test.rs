/// English TTS test with Kokoro ONNX model
use std::path::PathBuf;
use std::collections::HashMap;
use std::sync::LazyLock;
use ndarray::Array2;
use ort::value::Value;

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
fn english_tts_test() {
    let model_path = {
        let mut p = workspace_root();
        p.push("models/kokoro-82m/kokoro-v1.0.int8.onnx");
        p
    };
    let voice_path = {
        let mut p = workspace_root();
        p.push("models/kokoro-82m/voices/af_bella.bin");
        p
    };

    if !model_path.exists() || !voice_path.exists() {
        eprintln!("? ??????????");
        return;
    }

    // ?? voice data
    let data = std::fs::read(&voice_path).unwrap();
    let voice_floats: Vec<f32> = data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
    eprintln!("voice ??: {} float32 ({} ? 256)", voice_floats.len(), voice_floats.len() / 256);

    // ?????????? phonemes
    // "Hello world" ? Kokoro ?? phonemes??????? IPA?
    let test_cases = [
        ("hello world", "h?l?o? w???ld"),
        ("hello world simple", "h?lo? w?rld"),
        ("test", "t??st"),
        ("hello", "h?l?o?"),
    ];

    votex_infra::shared::ensure_ort_dylib_path();
    let mut session = ort::session::Session::builder()
        .unwrap()
        .commit_from_file(&model_path)
        .expect("?? ONNX ????");

    for (text, phonemes) in &test_cases {
        let token_ids = phonemes_to_token_ids(phonemes);
        if token_ids.len() <= 2 {
            eprintln!("'{}' ? phonemes='{}' ? token_ids={:?} (??: ?)", text, phonemes, token_ids);
            continue;
        }

        let seq_len = token_ids.len();

        // ???? phoneme ????? VOCAB ?
        let missing: Vec<char> = phonemes.chars().filter(|c| !VOCAB.contains_key(c)).collect();
        if !missing.is_empty() {
            eprintln!("'{}' ? phonemes='{}' ? ? VOCAB ????: {:?}", text, phonemes, missing);
            continue;
        }

        // ? style vector?? pack[0] ? pack[len-1]?
        let style_idx = (seq_len.saturating_sub(2)).min(509);  // seq_len-2 ? phonemes_len
        let offset = style_idx * 256;
        let style: Vec<f32> = voice_floats[offset..offset+256].to_vec();
        let style_array = Array2::from_shape_vec((1, 256), style).unwrap();
        let style_value = Value::from_array(style_array).unwrap();

        let input_ids = Array2::from_shape_vec((1, seq_len), token_ids).unwrap();
        let input_ids_value = Value::from_array(input_ids).unwrap();
        let speed_array = ndarray::arr1(&[1.0f32]);
        let speed_value = Value::from_array(speed_array).unwrap();

        match session.run(ort::inputs![input_ids_value, style_value, speed_value]) {
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
                    let duration_ms = num_samples * 1000 / 24000;
                    eprintln!("'{}' (phonemes='{}'): {} samples ({}ms), max_amp={:.6}, rms={:.6}",
                        text, phonemes, num_samples, duration_ms, max_amp, rms);

                    // ?? WAV
                    let mut out = workspace_root();
                    out.push("target/test_output");
                    std::fs::create_dir_all(&out).ok();
                    let wav_path = out.join(format!("en_{}.wav", text.replace(' ', "_")));

                    let spec = hound::WavSpec {
                        channels: 1,
                        sample_rate: 24000,
                        bits_per_sample: 16,
                        sample_format: hound::SampleFormat::Int,
                    };
                    let mut writer = hound::WavWriter::create(&wav_path, spec).unwrap();
                    for &sample in &samples {
                        let amp = (sample * (i16::MAX as f32)).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
                        writer.write_sample(amp).ok();
                    }
                    writer.finalize().ok();
                    eprintln!("  ???: {:?}", wav_path);
                }
            }
            Err(e) => {
                eprintln!("'{}': ? ????: {}", text, e);
            }
        }
    }
}
