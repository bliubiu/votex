/// 多引擎 TTS → ASR 端到端质量对比（素材：《奶爸大文豪》第一章）
///
/// 对 Qwen3-TTS（0.6B direct）与 CosyVoice 3.0 依次执行与 Kokoro e2e
/// 相同的闭环流程：合成 → 保存 WAV → SenseVoice 转写 → 覆盖率/序列相似度，
/// 输出统一口径的质量指标，用于与 Kokoro v1.1-zh（394.5s / 5.8 字每秒）对照。
///
/// 运行方式（两引擎串行执行，避免同时占用数 GB 内存）:
/// ```bash
/// cargo test -p votex-infra --test tts_e2e_multi_engine_test -- --test-threads=1 --nocapture
/// ```
use std::path::PathBuf;
use std::time::Instant;

use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, Language};
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{EngineKind, ModelId, ModelKind};
use votex_domain::tts::provider::TtsProvider;
use votex_domain::tts::value_object::{
    DenoiseLevel, Pitch, SegmentSize, Speed, TtsParams, VoiceId, Volume,
};

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

/// 《奶爸大文豪》第一章文本（UTF-8，2293 字）
fn novel_ch1_path() -> PathBuf {
    workspace_root().join("tmp/e2e_ch1.txt")
}

/// 取前 N 字作为合成输入（CPU 上自回归模型极慢，全章需小时级，这里取代表性片段）
fn read_excerpt(n: usize) -> String {
    let content = std::fs::read_to_string(novel_ch1_path())
        .expect("读取 tmp/e2e_ch1.txt 失败");
    content.chars().take(n).collect::<String>().trim().to_string()
}

fn make_tts_params(engine: EngineKind) -> TtsParams {
    TtsParams {
        engine,
        voice: VoiceId::new("default", "默认音色", engine),
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
    }
}

fn make_asr_params() -> AsrParams {
    AsrParams {
        model: ModelId::new("sensevoice"),
        language: Language::Zh,
        auto_punctuation: true,
        auto_slice: false,
        slice_length: votex_domain::asr::value_object::SliceLength::S30,
        denoise: false,
        denoise_level: votex_domain::asr::value_object::DenoiseLevel::Low,
        output_format: votex_domain::asr::value_object::SubtitleFormat::Txt,
    }
}

fn is_punct(c: char) -> bool {
    c.is_ascii_punctuation()
        || matches!(c,
            '，' | '。' | '、' | '？' | '！' | '：' | '；' | '“' | '”' | '‘' | '’' |
            '（' | '）' | '【' | '】' | '《' | '》' | '—' | '…' | '·' | '\n' | '\r' | '　' | ' ')
}

/// 与 Kokoro e2e 同口径的 ASR 闭环比对
fn asr_loop_verify(engine_name: &str, text: &str, audio: &votex_domain::shared::value_object::AudioData) {
    let sensevoice_dir = workspace_root().join("models/asr/sensevoice");
    if !sensevoice_dir.exists() {
        eprintln!("⚠ SenseVoice 模型目录不存在，跳过 ASR 闭环");
        return;
    }
    let mut asr = votex_infra::asr::sensevoice::SenseVoiceProvider::new();
    let asr_model = Model::new(
        ModelId::new("sensevoice"),
        &sensevoice_dir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::SenseVoice,
    );
    asr.load(&asr_model).expect("SenseVoice 加载失败");
    let result = asr
        .recognize(audio, &make_asr_params())
        .expect("SenseVoice 识别失败");
    asr.unload().ok();
    let recognized = result.text.trim();

    let orig_chars: Vec<char> = text.chars().filter(|c| !is_punct(*c)).collect();
    let recog_chars: Vec<char> = recognized.chars().filter(|c| !is_punct(*c)).collect();

    let recog_set: std::collections::HashSet<char> = recog_chars.iter().copied().collect();
    let covered = orig_chars.iter().filter(|c| recog_set.contains(c)).count();
    let coverage = covered as f64 / orig_chars.len() as f64 * 100.0;

    // LCS 序列相似度（限幅控制计算量）
    let (m, n) = (orig_chars.len(), recog_chars.len());
    let limit_m = m.min(n + 50);
    let limit_n = n.min(m + 50);
    let mut dp = vec![vec![0usize; limit_n + 1]; limit_m + 1];
    for i in 1..=limit_m {
        for j in 1..=limit_n {
            dp[i][j] = if orig_chars[i - 1] == recog_chars[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }
    let lcs = dp[limit_m][limit_n];
    let seq_similarity = lcs as f64 / m as f64 * 100.0;

    eprintln!("\n=== {} 质量指标 ===", engine_name);
    eprintln!("原始文本 ({} 字): '{}'", text.chars().count(), &text.chars().take(60).collect::<String>());
    eprintln!("ASR 结果 ({} 字): '{}'", recognized.chars().count(), &recognized.chars().take(80).collect::<String>());
    eprintln!("  内容覆盖率: {}/{} ({:.1}%)", covered, orig_chars.len(), coverage);
    eprintln!("  序列相似度 (LCS): {}/{} ({:.1}%)", lcs, orig_chars.len(), seq_similarity);

    assert!(!recognized.is_empty(), "{}: ASR 识别结果不应为空", engine_name);
    assert!(
        coverage > 30.0,
        "{}: TTS→ASR 内容覆盖率过低 ({:.1}%)", engine_name, coverage
    );
}

// ============================================================
// Qwen3-TTS 0.6B（direct 布局，fp16 权重）
// ============================================================

#[test]
fn test_qwen3tts_06b_e2e_naidadawenhao() {
    let model_dir = workspace_root().join("models/tts/qwen3-tts-0.6b");
    if !model_dir.exists() || !novel_ch1_path().exists() {
        eprintln!("⚠ 模型或素材缺失，跳过");
        return;
    }

    let text = read_excerpt(150);
    eprintln!("\n########## Qwen3-TTS 0.6B e2e（《奶爸大文豪》{} 字）##########", text.chars().count());

    let mut tts = votex_infra::tts::qwen3_tts::Qwen3TtsProvider::new();
    let model = Model::new(
        ModelId::new("qwen3-tts-0.6b"),
        "Qwen3-TTS-0.6B",
        ModelKind::Tts,
        EngineKind::Qwen3Tts,
    );
    let t0 = Instant::now();
    tts.load(&model).expect("Qwen3-TTS 加载失败");
    eprintln!("✓ 模型加载完成 ({:.1}s)", t0.elapsed().as_secs_f32());

    let voice = VoiceId::new("serena", "Serena（自然女声）", EngineKind::Qwen3Tts);
    let params = make_tts_params(EngineKind::Qwen3Tts);

    let t1 = Instant::now();
    let audio = tts.synthesize(&text, &voice, &params).expect("Qwen3-TTS 合成失败");
    let synth_secs = t1.elapsed().as_secs_f64();
    let audio_secs = audio.samples.len() as f64 / audio.sample_rate as f64;
    tts.unload().ok();

    eprintln!("✓ 合成完成: {} 采样点, {} Hz, 音频 {:.1}s, 耗时 {:.1}s ({:.2}x 实时), 语速 {:.1} 字/s",
        audio.samples.len(), audio.sample_rate, audio_secs, synth_secs,
        synth_secs / audio_secs, text.chars().count() as f64 / audio_secs);

    let out_dir = workspace_root().join("target/test_output");
    std::fs::create_dir_all(&out_dir).unwrap();
    let wav_path = out_dir.join("qwen3tts_06b_naidada.wav");
    votex_infra::audio::wav::WavWriter::write(&audio, &wav_path).expect("WAV 写入失败");
    eprintln!("  WAV: {:?}", wav_path);

    asr_loop_verify("Qwen3-TTS 0.6B", &text, &audio);
}

// ============================================================
// CosyVoice 3.0（零样本克隆，英文女声 prompt）
// ============================================================

#[test]
fn test_cosyvoice3_e2e_naidadawenhao() {
    let model_dir = workspace_root().join("models/tts/cosyvoice");
    if !model_dir.exists() || !novel_ch1_path().exists() {
        eprintln!("⚠ 模型或素材缺失，跳过");
        return;
    }

    let text = read_excerpt(150);
    eprintln!("\n########## CosyVoice 3.0 e2e（《奶爸大文豪》{} 字）##########", text.chars().count());

    let mut tts = votex_infra::tts::cosyvoice::CosyVoiceProvider::new();
    let model = Model::new(
        ModelId::new("cosyvoice"),
        "CosyVoice-3.0",
        ModelKind::Tts,
        EngineKind::CosyVoice3,
    );
    let t0 = Instant::now();
    tts.load(&model).expect("CosyVoice 加载失败");
    eprintln!("✓ 模型加载完成 ({:.1}s)", t0.elapsed().as_secs_f32());

    let voice = VoiceId::new("default", "默认（英文女声 prompt 克隆）", EngineKind::CosyVoice3);
    let params = make_tts_params(EngineKind::CosyVoice3);

    let t1 = Instant::now();
    let audio = tts.synthesize(&text, &voice, &params).expect("CosyVoice 合成失败");
    let synth_secs = t1.elapsed().as_secs_f64();
    let audio_secs = audio.samples.len() as f64 / audio.sample_rate as f64;
    tts.unload().ok();

    eprintln!("✓ 合成完成: {} 采样点, {} Hz, 音频 {:.1}s, 耗时 {:.1}s ({:.2}x 实时), 语速 {:.1} 字/s",
        audio.samples.len(), audio.sample_rate, audio_secs, synth_secs,
        synth_secs / audio_secs, text.chars().count() as f64 / audio_secs);

    let out_dir = workspace_root().join("target/test_output");
    std::fs::create_dir_all(&out_dir).unwrap();
    let wav_path = out_dir.join("cosyvoice3_naidada.wav");
    votex_infra::audio::wav::WavWriter::write(&audio, &wav_path).expect("WAV 写入失败");
    eprintln!("  WAV: {:?}", wav_path);

    asr_loop_verify("CosyVoice 3.0", &text, &audio);
}

// ---------------------------------------------------------------------------
// CosyVoice LLM decode 单步性能基准（定位 Rust 1.8s/token vs 官方 0.2s/token）
// 运行: cargo test -p votex-infra --test tts_e2e_multi_engine_test bench_decode单步 -- --ignored --nocapture
// 对比 release: cargo test -p votex-infra --release --test tts_e2e_multi_engine_test bench_decode单步 -- --ignored --nocapture
// ---------------------------------------------------------------------------
fn rand_val() -> f64 {
    use std::cell::Cell;
    thread_local! {
        static S: Cell<u64> = const { Cell::new(0x2545F4914F6CDD1D) };
    }
    S.with(|s| {
        let x = s.get();
        s.set(x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407));
        (x >> 33) as f64 / (1u64 << 31) as f64
    })
}

fn bench_decode(threads: usize, past_len: usize, steps: usize) {
    use ort::session::builder::GraphOptimizationLevel;
    use ort::session::Session;
    use ort::value::Value;

    votex_infra::shared::ensure_ort_dylib_path();
    let path = workspace_root().join("models/tts/cosyvoice/llm_backbone_decode_fp16_fixed.onnx");
    if !path.exists() {
        eprintln!("模型不存在，跳过: {:?}", path);
        return;
    }

    let mut session = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::All)
        .unwrap()
        .with_intra_threads(threads)
        .unwrap()
        .commit_from_file(&path)
        .unwrap();

    let kv: ndarray::Array<f32, ndarray::IxDyn> = ndarray::Array::from_shape_fn(
        ndarray::IxDyn(&[48, 1, 2, past_len, 64]),
        // 与 Python 基准一致的正态量级（±1），避免 denormal 浮点干扰计时
        |_| ((rand_val() - 0.5) * 2.0) as f32,
    );
    let emb_f16 = ndarray::Array3::<half::f16>::from_shape_fn((1, 1, 896), |_| {
        half::f16::from_f32((rand_val() - 0.5) as f32)
    });
    let mask_f16 =
        ndarray::Array2::<half::f16>::from_shape_fn((1, past_len + 1), |_| half::f16::ONE);

    for _ in 0..3 {
        let kv_t = Value::from_array(kv.clone()).unwrap();
        let emb_t = Value::from_array(emb_f16.clone()).unwrap();
        let mask_t = Value::from_array(mask_f16.clone()).unwrap();
        let _ = session.run(ort::inputs![emb_t, mask_t, kv_t]).unwrap();
    }

    let mut t_create = std::time::Duration::ZERO;
    let mut t_run = std::time::Duration::ZERO;
    let mut t_extract = std::time::Duration::ZERO;

    for _ in 0..steps {
        let t0 = Instant::now();
        let kv_t = Value::from_array(kv.clone()).unwrap();
        let emb_t = Value::from_array(emb_f16.clone()).unwrap();
        let mask_t = Value::from_array(mask_f16.clone()).unwrap();
        t_create += t0.elapsed();

        let t1 = Instant::now();
        let outputs = session.run(ort::inputs![emb_t, mask_t, kv_t]).unwrap();
        t_run += t1.elapsed();

        let t2 = Instant::now();
        let _hidden: ndarray::Array<f32, ndarray::IxDyn> =
            outputs[0].try_extract_array::<f32>().unwrap().to_owned();
        let _new_kv: ndarray::Array<f32, ndarray::IxDyn> =
            outputs[1].try_extract_array::<f32>().unwrap().to_owned();
        t_extract += t2.elapsed();
    }

    let total = t_create + t_run + t_extract;
    println!(
        "[bench] threads={} past_len={}: 总 {:.0} ms/步 | 张量创建 {:.1} | run {:.1} | 提取 {:.1}",
        threads,
        past_len,
        total.as_secs_f64() * 1000.0 / steps as f64,
        t_create.as_secs_f64() * 1000.0 / steps as f64,
        t_run.as_secs_f64() * 1000.0 / steps as f64,
        t_extract.as_secs_f64() * 1000.0 / steps as f64,
    );
}

#[test]
#[ignore = "性能基准，需要本地 cosyvoice 模型"]
fn bench_decode单步_多线程对比() {
    for threads in [8usize, 4, 2, 1] {
        bench_decode(threads, 800, 10);
    }
}
