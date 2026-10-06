//! Paraformer ASR 集成测试
//!
//! 回归背景：此前"自行实现"的特征提取只把帧能量乘线性系数填满 80 维，
//! 不是 mel 特征，识别输出是无效数据。现改走 sherpa-onnx 绑定
//! （真实 fbank + LFR + CMVN，与 firered/qwen3 同范式），registry 源已
//! 换为 sherpa-onnx 官方转换包（sherpa-onnx-paraformer-zh-2024-03-09，
//! zh+en+yue）。
//!
//! 历史教训：FunASR 原始导出（model_quant.onnx）缺 sherpa-onnx 要求的
//! vocab_size 元数据，sherpa C++ 端会 abort 整个进程——provider 在加载前
//! 明确拒绝（见 paraformer.rs 护栏），拒绝测试保留以防旧资产回流入库。
//!
//! 运行：`cargo test -p votex-infra --test paraformer_integration_test --features slow-models`

use std::path::PathBuf;

use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, Language, SubtitleFormat};
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{EngineKind, ModelId, ModelKind};
use votex_domain::shared::value_object::AudioData;
use votex_infra::asr::paraformer::ParaformerProvider;

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

fn model_dir() -> PathBuf {
    workspace_root().join("models/asr/paraformer")
}

/// 测试音频与 firered-asr 共用同一套（sherpa-onnx 各转换包的标准测试集）
fn test_wavs_dir() -> PathBuf {
    workspace_root().join("models/asr/firered-asr-ctc/test_wavs")
}

fn make_params() -> AsrParams {
    AsrParams {
        model: ModelId::new("paraformer"),
        language: Language::Zh,
        auto_punctuation: true,
        auto_slice: true,
        slice_length: Default::default(),
        denoise: false,
        denoise_level: Default::default(),
        output_format: SubtitleFormat::Srt,
    }
}

fn read_test_wav(path: &PathBuf) -> AudioData {
    votex_infra::audio::wav::read_wav(path).expect("读取测试 WAV 失败")
}

#[test]
#[cfg_attr(
    not(feature = "slow-models"),
    ignore = "需本地 Paraformer 模型目录；跑法: cargo test -p votex-infra --test paraformer_integration_test --features slow-models"
)]
fn paraformer_识别0wav内容() {
    let model_dir = model_dir();
    let wav_path = test_wavs_dir().join("0.wav");
    if !model_dir.join("model.int8.onnx").exists() || !wav_path.exists() {
        eprintln!("⚠ sherpa 转换模型（model.int8.onnx）或测试音频不存在，跳过");
        return;
    }

    let provider = ParaformerProvider::new();
    let model = Model::new(
        ModelId::new("paraformer"),
        &model_dir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::Paraformer,
    );
    provider.load(&model).expect("sherpa 转换模型应能加载");
    assert!(provider.is_loaded());

    let audio_data = read_test_wav(&wav_path);
    let result = provider
        .recognize(&audio_data, &make_params())
        .expect("识别失败");

    let text = result.text.trim();
    eprintln!("Paraformer 0.wav 识别结果: '{}'", text);
    assert!(!text.is_empty(), "识别结果不应为空");
    // 音频内容为"昨天是星期一，今天是星期二，后天是星期三"。
    // 与 firered 测试同一断言口径：按"星期/礼拜"语义断言，
    // 不锁定具体字符（模型可能有中英混说表达）
    assert!(
        text.contains("星期") || text.contains("礼拜"),
        "应识别出星期相关内容，得到: '{}'",
        text
    );

    provider.unload().ok();
    assert!(!provider.is_loaded(), "unload 后应处于未加载状态");
}

#[test]
#[cfg_attr(
    not(feature = "slow-models"),
    ignore = "需本地 Paraformer 模型目录；跑法: cargo test -p votex-infra --test paraformer_integration_test --features slow-models"
)]
fn paraformer_FunASR导出必须被明确拒绝() {
    let model_dir = model_dir();
    // 护栏只在「目录里只有 FunASR 导出、没有 sherpa 转换模型」时才有意义：
    // 换源后目录中已有 model.int8.onnx，加载会直接成功
    if model_dir.join("model.int8.onnx").exists()
        || model_dir.join("model.onnx").exists()
        || !model_dir.join("model_quant.onnx").exists()
    {
        eprintln!("⚠ 目录中已有 sherpa 转换模型或无 FunASR 导出，拒绝护栏不适用，跳过");
        return;
    }

    let provider = ParaformerProvider::new();
    let model = Model::new(
        ModelId::new("paraformer"),
        &model_dir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::Paraformer,
    );
    // 回归护栏：FunASR 原始导出缺 sherpa-onnx 所需的 vocab_size 元数据，
    // sherpa C++ 端会直接 abort 整个进程——必须在调用前拒绝并给出指引
    let err = provider
        .load(&model)
        .expect_err("FunASR 导出必须被拒绝加载");
    let msg = format!("{}", err);
    assert!(
        msg.contains("vocab_size") || msg.contains("sherpa"),
        "拒绝信息应说明原因与替代方案，实际: {}",
        msg
    );
    assert!(!provider.is_loaded());
    assert!(provider.unload().is_ok());
}

#[test]
#[cfg_attr(
    not(feature = "slow-models"),
    ignore = "加载 237MB Paraformer 模型并推理；跑法: cargo test -p votex-infra --test paraformer_integration_test --features slow-models"
)]
fn paraformer_未加载时识别报EngineNotLoaded() {
    let provider = ParaformerProvider::new();
    let audio_data = AudioData::silence(16000, 500);
    let result = provider.recognize(&audio_data, &make_params());
    assert!(
        matches!(result, Err(votex_domain::error::AsrError::EngineNotLoaded)),
        "未加载时必须报 EngineNotLoaded，实际: {:?}",
        result
    );
}
