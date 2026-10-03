/// ??????ONNX Runtime ????????????
use std::time::Instant;
use ort::session::Session;

#[test]
fn test_minimal_ort_load_v11() {
    let model_dir = {
        let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        d.pop(); d.pop();
        d.join("models/kokoro-82m-v1.1-zh")
    };
    let model_path = model_dir.join("kokoro-v1.1-zh.os18.onnx");
    let fallback_path = model_dir.join("kokoro-v1.1-zh.onnx");
    let model_path = if model_path.exists() { model_path } else { fallback_path };

    eprintln!("\n========== ORT ??????????????????==========");
    eprintln!("??????: {:?}", model_path);
    eprintln!("??????: {} MB", std::fs::metadata(&model_path).unwrap().len() / 1024 / 1024);
    
    let t0 = Instant::now();
    
    eprintln!("?????? Session builder...");
    votex_infra::shared::ensure_ort_dylib_path();
    let mut builder = match Session::builder() {
        Ok(b) => b,
        Err(e) => { eprintln!("???builder ??????: {}", e); return; }
    };
    
    eprintln!("???????????? (????????? 120 ???...");
    let t1 = Instant::now();
    
    match builder.commit_from_file(&model_path) {
        Ok(session) => {
            let elapsed = t1.elapsed();
            eprintln!("?????????????????????! ?????? {:.1}s", elapsed.as_secs_f64());
            let _ = session; // ????????????
        }
        Err(e) => {
            let elapsed = t1.elapsed();
            eprintln!("???????????????: {}", e);
            eprintln!("  ?????? {:.1}s", elapsed.as_secs_f64());
        }
    }
    
    eprintln!("????????? {:.1}s", t0.elapsed().as_secs_f64());
}

#[test]
fn test_kokoro_provider_load() {
    use votex_infra::tts::kokoro::KokoroProvider;
    use votex_domain::model::entity::Model;
    use votex_domain::model::value_object::{ModelId, ModelKind, EngineKind};
    use votex_domain::tts::provider::TtsProvider;

    let v11_dir = {
        let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        d.pop(); d.pop();
        d.join("models/kokoro-82m-v1.1-zh")
    };
    let model_path = v11_dir.join("kokoro-v1.1-zh.os18.onnx");

    eprintln!("\n========== KokoroProvider ???????????? ==========");
    eprintln!("??????: {:?}", model_path);
    eprintln!("??????: {}", model_path.exists());

    let model = Model::new(
        ModelId::new("kokoro-82m-v1.1-zh"),
        "Kokoro-82M-v1.1-zh",
        ModelKind::Tts,
        EngineKind::Kokoro,
    );

    let mut provider = KokoroProvider::new();
    let t0 = Instant::now();

    match provider.load(&model) {
        Ok(()) => {
            eprintln!("???KokoroProvider ????????????! ?????? {:.1}s", t0.elapsed().as_secs_f64());
        }
        Err(e) => {
            eprintln!("???KokoroProvider ????????????: {}", e);
            eprintln!("  ?????? {:.1}s", t0.elapsed().as_secs_f64());
        }
    }
}
