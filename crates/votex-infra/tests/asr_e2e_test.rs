//! ASR 引擎端到端验证：SenseVoice + Qwen3-ASR
//!
//! 素材使用 IndexTTS-2.5 已落盘的合成音频（**文本已知**，构成 TTS→ASR 闭环）：
//! - `tmp/indextts25_e2e_yue.wav`：粤语短句「今日天气几好，我哋一齐去饮茶啦。」（3.55s）
//!
//! 判据（由硬到软）：
//! 1. 模型加载成功（models/asr/sensevoice、models/asr/qwen3-asr）
//! 2. SenseVoice：识别结果包含关键实词（天气/饮茶/今日 任一）——
//!    粤语短句对 SenseVoice 是可识别语种（模型原生支持 yue）
//! 3. Qwen3-ASR：识别结果非空且含 CJK 字符（粤语口音对该模型属域外场景，
//!    只要求中文内容被识别，不做逐字断言）
//!
//! 运行：
//! ```bash
//! cargo test -p votex-infra --test asr_e2e_test -- --nocapture
//! ```
//! 模型缺失时跳过（与 tts_asr_e2e_test 约定一致）。

use std::path::PathBuf;

use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, Language};
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{EngineKind, ModelId, ModelKind};

use votex_infra::audio::wav;

/// 工作区根目录（测试进程 CWD 是 crate 目录，必须显式解析）
fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

/// e2e 素材：IndexTTS-2.5 粤语短句（ground truth：今日天气几好，我哋一齐去饮茶啦。）
const GROUND_TRUTH: &str = "今日天气几好，我哋一齐去饮茶啦。";
const GROUND_TRUTH_KEYWORDS: [&str; 3] = ["天气", "饮茶", "今日"];

fn sample_audio() -> (Vec<f32>, u32) {
    let path = workspace_root().join("tmp/indextts25_e2e_yue.wav");
    assert!(path.is_file(), "缺少 e2e 音频素材 tmp/indextts25_e2e_yue.wav（先跑 indextts25 e2e）");
    let audio = wav::read_wav(&path).expect("读取 wav 失败");
    (audio.samples, audio.sample_rate)
}

fn contains_cjk(s: &str) -> bool {
    s.chars().any(|c| ('\u{4E00}'..='\u{9FFF}').contains(&c))
}

fn sensevoice_params() -> AsrParams {
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

fn qwen3_asr_params() -> AsrParams {
    AsrParams {
        model: ModelId::new("qwen3-asr"),
        ..sensevoice_params()
    }
}

/// SenseVoice e2e：已知文本的合成音频必须被识别出关键实词
#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test asr_e2e_test --features slow-models")]
#[test]
fn sensevoice_识别indextts25合成粤语短句() {
    let model_dir = workspace_root().join("models/asr/sensevoice");
    if !model_dir.is_dir() {
        eprintln!("⚠ SenseVoice 模型目录不存在: {:?}，跳过", model_dir);
        return;
    }
    let (samples, sample_rate) = sample_audio();
    // sensevoice::find_model_dir 用 CWD 相对路径，必须切到工作区根
    std::env::set_current_dir(workspace_root()).expect("切换工作目录失败");
    eprintln!("\n########## SenseVoice ASR e2e ##########");
    eprintln!("  音频: {} 样本 @ {} Hz（{:.2}s）", samples.len(), sample_rate,
        samples.len() as f64 / sample_rate as f64);
    eprintln!("  ground truth: {GROUND_TRUTH}");

    let asr = votex_infra::asr::sensevoice::SenseVoiceProvider::new();
    let model = Model::new(
        ModelId::new("sensevoice"),
        "SenseVoice",
        ModelKind::Asr,
        EngineKind::SenseVoice,
    );
    asr.load(&model).expect("SenseVoice 加载失败");
    eprintln!("✓ SenseVoice 加载成功");

    let audio = votex_domain::shared::value_object::AudioData {
        samples,
        sample_rate,
        channels: 1,
    };
    let out = asr
        .recognize(&audio, &sensevoice_params())
        .expect("SenseVoice 识别失败");
    eprintln!("✓ 识别结果: {}", out.text);

    assert!(contains_cjk(&out.text), "识别结果无中文: {}", out.text);
    let hit = GROUND_TRUTH_KEYWORDS.iter().any(|k| out.text.contains(k));
    assert!(hit, "识别结果未命中关键词 {GROUND_TRUTH_KEYWORDS:?}: {}", out.text);
    eprintln!("✓ SenseVoice e2e 通过：命中已知文本关键词");
}

/// Qwen3-ASR e2e：同一素材，要求非空 + 中文内容（粤语口音不做逐字断言）
#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test asr_e2e_test --features slow-models")]
#[test]
fn qwen3_asr_识别indextts25合成粤语短句() {
    let model_dir = workspace_root().join("models/asr/qwen3-asr");
    if !model_dir.is_dir() {
        eprintln!("⚠ Qwen3-ASR 模型目录不存在: {:?}，跳过", model_dir);
        return;
    }
    // qwen3_asr.rs 是 ort ONNX 实现；目录内只有 safetensors 时无法加载，
    // 属产品级资产缺口（docs/05 宣称的 Candle 纯 Rust 路线尚未落地）——显式跳过并说明
    let has_onnx = std::fs::read_dir(&model_dir)
        .map(|rd| rd.filter_map(|e| e.ok()).any(|e| e.file_name().to_string_lossy().ends_with(".onnx")))
        .unwrap_or(false);
    if !has_onnx {
        eprintln!(
            "⚠ Qwen3-ASR 目录内无 ONNX 权重（仅 safetensors），ort 实现无法加载，e2e 被阻塞。\n             需先完成权重导出（safetensors → ONNX）或落地 docs/05 的 Candle 纯 Rust 路线，跳过测试"
        );
        return;
    }
    let (samples, sample_rate) = sample_audio();
    // qwen3_asr::find_model_dir 用 CWD 相对路径，必须切到工作区根
    std::env::set_current_dir(workspace_root()).expect("切换工作目录失败");
    eprintln!("\n########## Qwen3-ASR e2e ##########");
    eprintln!("  音频: {} 样本 @ {} Hz", samples.len(), sample_rate);
    eprintln!("  ground truth: {GROUND_TRUTH}");

    let asr = votex_infra::asr::qwen3_asr::Qwen3AsrProvider::new();
    let model = Model::new(
        ModelId::new("qwen3-asr"),
        "Qwen3-ASR",
        ModelKind::Asr,
        EngineKind::Qwen3Asr,
    );
    asr.load(&model).expect("Qwen3-ASR 加载失败");
    eprintln!("✓ Qwen3-ASR 加载成功");

    let audio = votex_domain::shared::value_object::AudioData {
        samples,
        sample_rate,
        channels: 1,
    };
    let out = asr
        .recognize(&audio, &qwen3_asr_params())
        .expect("Qwen3-ASR 识别失败");
    eprintln!("✓ 识别结果: {}", out.text);

    assert!(!out.text.trim().is_empty(), "识别结果为空");
    assert!(contains_cjk(&out.text), "识别结果无中文: {}", out.text);
    eprintln!("✓ Qwen3-ASR e2e 通过：非空中文识别");
}
