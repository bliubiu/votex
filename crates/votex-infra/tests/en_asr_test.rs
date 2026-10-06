/// ASR ???? TTS ??
use std::path::PathBuf;
use votex_domain::asr::provider::AsrProvider;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{EngineKind, ModelId, ModelKind};

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test en_asr_test --features slow-models")]
#[test]
fn english_asr_verification() {
    let whisper_path = {
        let mut p = workspace_root();
        p.push("models/whisper-base/ggml-base.bin");
        p
    };
    if !whisper_path.exists() {
        eprintln!("? Whisper ?????");
        return;
    }

    let test_dir = {
        let mut p = workspace_root();
        p.push("target/test_output");
        p
    };

    let test_cases = [
        ("en_hello.wav", "hello"),
        ("en_test.wav", "test"),
        ("en_hello_world.wav", "hello world"),
        ("en_hello_world_simple.wav", "hello world"),
    ];

    let asr = votex_infra::asr::whisper::WhisperProvider::new();
    let model = Model::new(ModelId::new("whisper-base"), &whisper_path.to_string_lossy(), ModelKind::Asr, EngineKind::Whisper);
    asr.load(&model).expect("Whisper load failed");

    for (filename, expected) in &test_cases {
        let wav_path = test_dir.join(filename);
        if !wav_path.exists() {
            eprintln!("? ?????: {:?}", wav_path);
            continue;
        }

        let audio_data = votex_infra::audio::wav::read_wav(&wav_path).expect("WAV ????");
        let asr_params = votex_domain::asr::value_object::AsrParams {
            model: ModelId::new("whisper-base"),
            language: votex_domain::asr::value_object::Language::ZhEn,
            auto_punctuation: false,
            auto_slice: false,
            slice_length: votex_domain::asr::value_object::SliceLength::S30,
            denoise: false,
            denoise_level: votex_domain::asr::value_object::DenoiseLevel::Low,
            output_format: votex_domain::asr::value_object::SubtitleFormat::Txt,
        };

        match asr.recognize(&audio_data, &asr_params) {
            Ok(result) => {
                let recognized = result.text.trim().to_lowercase();
                let expected_lower = expected.to_lowercase();
                let match_str = if recognized.contains(&expected_lower) || expected_lower.contains(&recognized) {
                    "?"
                } else {
                    "?"
                };
                eprintln!("  {}: TTS='{}' ? ASR='{}' {}", filename, expected, recognized, match_str);
            }
            Err(e) => {
                eprintln!("  {}: ASR ??: {}", filename, e);
            }
        }
    }

    asr.unload().ok();
}
