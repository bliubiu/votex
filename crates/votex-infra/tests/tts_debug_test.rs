/// TTS ?????????????????????ONNX ??????????????????token ID ?????????style ????????????????????????
use std::path::PathBuf;
use votex_domain::tts::provider::TtsProvider;
use votex_domain::asr::provider::AsrProvider;
use votex_domain::tts::value_object::{VoiceId, Speed, Pitch, Volume, SegmentSize, DenoiseLevel};
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{ModelId, ModelKind, EngineKind};
use std::collections::HashMap;
use std::sync::LazyLock;

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

/// Kokoro VOCAB (loaded from config.json at runtime)
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

/// ?????? phonemes_to_token_ids?????? kokoro.rs ?????????
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
fn tts_deep_diagnostics() {
    let model_path = {
        let mut p = workspace_root();
        p.push("models/kokoro-82m/kokoro-v1.0.int8.onnx");
        p
    };
    if !model_path.exists() {
        eprintln!("model file not found: {:?}", model_path);
        return;
    }

    // ========== 1. ???????????? phoneme???token ID ?????? ==========
    eprintln!("\n=== Token ID ???????????? ===");
    let test_text = "test debug: hello world";
    let phonemes = votex_infra::tts::kokoro_g2p::text_to_phonemes(test_text);
    let ids = phonemes_to_token_ids(&phonemes);
    eprintln!("??????: '{}'", test_text);
    eprintln!("phonemes: '{}'", phonemes);
    eprintln!("token ?????? (??????={}): {:?}", ids.len(), ids);
    eprintln!("BOS: {}, EOS: {}", ids[0], ids[ids.len()-1]);

    // ========== 3. ???????????????==========
    eprintln!("\n=== voice vector analysis ===");
    let voice_path = {
        let mut p = workspace_root();
        p.push("models/kokoro-82m/voices/af_bella.bin");
        p
    };
    if !voice_path.exists() {
        eprintln!("voice file not found: {:?}", voice_path);
        return;
    }

    let voice_data = std::fs::read(&voice_path).unwrap();
    let floats: Vec<f32> = voice_data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    eprintln!("voice ??????: {} float32, {} ?? 256", floats.len(), floats.len() / 256);

    // check style vector structure
    for idx in [0, 50, 89, 100, 200, 300, 400, 509] {
        if idx < floats.len() / 256 {
            let offset = idx * 256;
            let vec = &floats[offset..offset+256];
            let mean = vec.iter().sum::<f32>() / 256.0;
            let min = vec.iter().copied().fold(f32::MAX, f32::min);
            let max = vec.iter().copied().fold(f32::MIN, f32::max);
            eprintln!("  pack[{}]: mean={:.6}, min={:.6}, max={:.6}", idx, mean, min, max);
        }
    }

    // ?????????single character test
    let short_texts = ["hello", "world", "test", "rust", "tts", "kokoro"];
    for &text in &short_texts {
        let ph = votex_infra::tts::kokoro_g2p::text_to_phonemes(text);
        let tok_ids = phonemes_to_token_ids(&ph);
        eprintln!("'{}' ???phonemes='{}' ???token_ids={:?} (len={})",
            text, ph, tok_ids, tok_ids.len());

        // Check if all chars in phonemes are in VOCAB
        let mut missing = false;
        for c in ph.chars() {
            if !VOCAB.contains_key(&c) {
                eprintln!("  ???'{}' (U+{:04X}) not in VOCAB!", c, c as u32);
                missing = true;
            }
        }
        if !missing {
            eprintln!("  ???all in VOCAB");
        }
    }

    // ========== 4. ???????????????????????? ASR ?????? ==========
    eprintln!("\n=== TTS ?????????????????????===");
    let mut tts = votex_infra::tts::kokoro::KokoroProvider::new();
    let model = Model::new(
        ModelId::new("kokoro-82m"),
        &model_path.to_string_lossy(),
        ModelKind::Tts,
        EngineKind::Kokoro,
    );
    tts.load(&model).expect("TTS ??????????????????");

    let voice = VoiceId::new("af_bella", "Bella", EngineKind::Kokoro);
    let params = votex_domain::tts::value_object::TtsParams {
        engine: EngineKind::Kokoro,
        voice: voice.clone(),
        speed: Speed::default(),
        pitch: Pitch::default(),
        volume: Volume::default(),
        segment_size: SegmentSize::S500,
        segment_silence_ms: 500,
        crossfade_ms: 10,
        num_to_chinese: false,
        denoise: false,
        denoise_level: DenoiseLevel::Low,
        emotion: None,
        dialect: None,
    };

    for text in ["hello", "world test"] {
        match tts.synthesize(text, &voice, &params) {
            Ok(audio) => {
                let max_amp = audio.samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
                let rms = (audio.samples.iter().map(|s| s * s).sum::<f32>() / audio.samples.len() as f32).sqrt();
                eprintln!("'{}': {} samples, max_amp={:.6}, rms={:.6}, duration={}ms",
                    text, audio.samples.len(), max_amp, rms, audio.duration_ms());

                // ?????? WAV
                let mut out = workspace_root();
                out.push("target/test_output");
                std::fs::create_dir_all(&out).ok();
                let wav = out.join(format!("tts_{}.wav", text));
                votex_infra::audio::wav::WavWriter::write(&audio, &wav).ok();
                eprintln!("  saved to {:?}", wav);
            }
            Err(e) => {
                eprintln!("'{}': synthesis failed: {}", text, e);
            }
        }
    }

    tts.unload().ok();

    // ========== 5. ASR recognition of TTS output ==========
    eprintln!("\n=== ASR recognition test ===");
    let whisper_path = {
        let mut p = workspace_root();
        p.push("models/whisper-base/ggml-base.bin");
        p
    };
    if !whisper_path.exists() {
        eprintln!("Whisper model not found, skip ASR test");
        return;
    }

    let mut asr = votex_infra::asr::whisper::WhisperProvider::new();
    let whisper_model = Model::new(
        ModelId::new("whisper-base"),
        &whisper_path.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::Whisper,
    );
    asr.load(&whisper_model).expect("Whisper load failed");

    let test_audio_dir = {
        let mut p = workspace_root();
        p.push("target/test_output");
        p
    };

    for text in ["hello", "world test"] {
        let wav_path = test_audio_dir.join(format!("tts_{}.wav", text));
        if !wav_path.exists() {
                eprintln!("WAV file not found: {:?}", wav_path);
            continue;
        }
        let audio_data = votex_infra::audio::wav::read_wav(&wav_path)
            .expect("WAV ????????????");

        let asr_params = votex_domain::asr::value_object::AsrParams {
            model: ModelId::new("whisper-base"),
            language: votex_domain::asr::value_object::Language::Zh,
            auto_punctuation: true,
            auto_slice: false,
            slice_length: votex_domain::asr::value_object::SliceLength::S30,
            denoise: false,
            denoise_level: votex_domain::asr::value_object::DenoiseLevel::Low,
            output_format: votex_domain::asr::value_object::SubtitleFormat::Txt,
        };

        match asr.recognize(&audio_data, &asr_params) {
            Ok(result) => {
                let trimmed = result.text.trim();
                let match_str = if trimmed == text { " MATCH" } else { "MISMATCH" };
                eprintln!("  TTS: '{}' -> ASR: '{}' {}", text, trimmed, match_str);
            }
            Err(e) => {
                eprintln!("  '{}': ASR recognition failed: {}", text, e);
            }
        }
    }

    asr.unload().ok();
}
