/// Compare raw Session loading vs library provider loading
use std::time::Instant;
use ort::session::Session;

fn model_path() -> std::path::PathBuf {
    let mut d = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    d.pop(); d.pop();
    d.join("models/kokoro-82m-v1.1-zh/kokoro-v1.1-zh.os18.onnx")
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test raw_session_vs_lib_test --features slow-models")]
#[test]
fn test_raw_session_first() {
    // ??????????????? ???????????? raw Session::builder().commit_from_file()
    // ?????????KokoroProvider???????????? votex-infra ?????? tts ??????
    let mp = model_path();
    eprintln!("\n========== Raw Session ???????????? ==========");
    eprintln!("??????: {:?}", mp);
    eprintln!("??????: {} MB", std::fs::metadata(&mp).unwrap().len() / 1024 / 1024);

    let t0 = Instant::now();
    votex_infra::shared::ensure_ort_dylib_path();
    let mut builder = match Session::builder() {
        Ok(b) => b,
        Err(e) => { eprintln!("???builder: {}", e); return; }
    };
    match builder.commit_from_file(&mp) {
        Ok(_) => eprintln!("????????? Session ???????????? ({:.1}s)", t0.elapsed().as_secs_f64()),
        Err(e) => eprintln!("????????? Session ????????????: {} ({:.1}s)", e, t0.elapsed().as_secs_f64()),
    }
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test raw_session_vs_lib_test --features slow-models")]
#[test]
fn test_raw_session_after_lib_load() {
    // ??????????????? ???????????????????????????raw Session
    // load provider first, then try raw Session
    use votex_infra::tts::kokoro::KokoroProvider;
    use votex_domain::model::entity::Model;
    use votex_domain::model::value_object::{ModelId, ModelKind, EngineKind};
    use votex_domain::tts::provider::TtsProvider;

    let mp = model_path();
    eprintln!("\n========== ?????????KokoroProvider?????????Raw Session ==========");

    // 1. load provider
    let provider = KokoroProvider::new();
    let model = Model::new(ModelId::new("kokoro-82m-v1.1-zh"), "Kokoro-82M-v1.1-zh", ModelKind::Tts, EngineKind::Kokoro);
    
    let t1 = Instant::now();
    match provider.load(&model) {
        Ok(()) => eprintln!("???KokoroProvider ???????????? ({:.1}s)", t1.elapsed().as_secs_f64()),
        Err(e) => eprintln!("???KokoroProvider ??????: {} ({:.1}s)", e, t1.elapsed().as_secs_f64()),
    }

    // 2. Raw Session ??????????????????
    votex_infra::shared::ensure_ort_dylib_path();
    let mut builder = match Session::builder() {
        Ok(b) => b,
        Err(e) => { eprintln!("????????? raw builder: {}", e); return; }
    };
    match builder.commit_from_file(&mp) {
        Ok(_) => eprintln!("????????? raw Session ????????????"),
        Err(e) => eprintln!("????????? raw Session ??????: {}", e),
    }
}
