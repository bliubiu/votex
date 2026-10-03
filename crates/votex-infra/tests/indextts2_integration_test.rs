/// IndexTTS2 ??????????????????
use std::time::Instant;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{EngineKind, ModelId, ModelKind};
use votex_domain::tts::provider::TtsProvider;
use votex_domain::tts::value_object::{
    DenoiseLevel, Pitch, SegmentSize, Speed, TtsParams, VoiceId, Volume,
};

fn workspace_root() -> std::path::PathBuf {
    let mut dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

fn save_wav(samples: &[f32], sample_rate: u32, filename: &str) {
    let mut output_dir = workspace_root();
    output_dir.push("target/test_output");
    std::fs::create_dir_all(&output_dir).ok();
    let wav_path = output_dir.join(filename);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&wav_path, spec).expect("?????? WAV ????????????");
    for &sample in samples {
        let amp = (sample * (i16::MAX as f32)).clamp(i16::MIN as f32, i16::MAX as f32) as i16;
        writer.write_sample(amp).ok();
    }
    writer.finalize().ok();
    eprintln!("?????????: {:?} ({:.1}KB)", wav_path, samples.len() as f64 * 2.0 / 1024.0);
}

#[test]
fn test_indextts2_load_and_synthesize() {
    let base_dir = workspace_root().join("models/indextts2");
    let model_path = base_dir.join("gpt.onnx");
    if !model_path.exists() {
        eprintln!("??????: gpt.onnx ?????????({:?})", model_path);
        return;
    }

    eprintln!("\n========== IndexTTS2 ?????? TTS ?????? ==========");

    let mut provider = votex_infra::tts::indextts2::IndexTTS2Provider::new();
    let model = Model::new(
        ModelId::new("indextts2"),
        "IndexTTS-2",
        ModelKind::Tts,
        EngineKind::IndexTTS2,
    );

    let load_start = Instant::now();
    match provider.load(&model) {
        Ok(()) => eprintln!("??????????????? ({:.2}s)", load_start.elapsed().as_secs_f64()),
        Err(e) => {
            eprintln!("???????????????: {}", e);
            return;
        }
    }

    let params = TtsParams {
        engine: EngineKind::IndexTTS2,
        voice: VoiceId::new("default", "????????????", EngineKind::IndexTTS2),
        speed: Speed::default(),
        pitch: Pitch::default(),
        volume: Volume::default(),
        segment_size: SegmentSize::S500,
        segment_silence_ms: 300,
        crossfade_ms: 50,
        num_to_chinese: true,
        denoise: false,
        denoise_level: DenoiseLevel::Low,
        emotion: None,
        dialect: None,
    };

    // test text
    let test_text = "test indextts2 synthesis: hello world";
    eprintln!("\n????????????: '{}'", test_text);

    let syn_start = Instant::now();
    match provider.synthesize(test_text, &params.voice, &params) {
        Ok(audio) => {
            let dur = syn_start.elapsed();
            let audio_dur = audio.samples.len() as f64 / audio.sample_rate as f64;
            eprintln!(
                "???????????????: {} samples, {:.1}s ??????, ?????? {:.1}s, ?????????{:.2}x",
                audio.samples.len(),
                audio_dur,
                dur.as_secs_f64(),
                audio_dur / dur.as_secs_f64()
            );
            // ???????????????????????????????????????????????????
            let rms = (audio.samples.iter().map(|s| s * s).sum::<f32>() / audio.samples.len() as f32).sqrt();
            eprintln!("  ?????? RMS: {:.4} (??????={})", rms, rms > 0.001);
            save_wav(&audio.samples, audio.sample_rate, "indextts2_test.wav");
        }
        Err(e) => {
            eprintln!("???????????????: {}", e);
        }
    }

    provider.unload().ok();
    eprintln!("???IndexTTS2 ????????????\n");
}
