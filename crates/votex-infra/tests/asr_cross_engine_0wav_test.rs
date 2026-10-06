//! ASR 引擎 0.wav 交叉验证
//!
//! 音频真值（实测各引擎共识）：**中英混说**——
//! 「昨天是 Monday, today is 礼拜二, the day after tomorrow 是星期三」。
//! 注意：这段是 sherpa-onnx 标准测试集的混说样本，并非纯中文
//! "昨天是星期一……"（5 个引擎独立输出英文星期词可证）。
//!
//! 背景：FireRedASR 集成测试曾因断言 `contains("一")||contains("第")||contains("的")`
//! 失败——不是模型失效，而是断言假定了纯中文音频；混说样本里"星期一"
//! 实际读作 "Monday"。本测试把全部本地引擎放到同一音频上对比，
//! 统一按「星期/礼拜/英文星期词」语义断言，回答"是否只有某个引擎识别不出内容"：
//!
//! | 引擎 | 语义 | 备注 |
//! |---|---|---|
//! | Qwen3-ASR | ✅ | 混说结构最完整 |
//! | SenseVoice | ✅ | 混说表达 |
//! | Paraformer（sherpa 官方包） | ✅ | 混说表达 |
//! | FireRed CTC/AED | ✅ | 见 firered_asr_integration_test |
//! | WeNet | ⚠️ | 首尾正确，中段同音字噪声（"礼巴二/猫蓉"） |
//! | Whisper-base | ❌ | 经典重复幻觉（"天天天…"），诊断性用例 |
//!
//! 运行：`cargo test -p votex-infra --test asr_cross_engine_0wav_test --features slow-models -- --nocapture`

use std::path::PathBuf;

use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, Language, SubtitleFormat};
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{EngineKind, ModelId, ModelKind};
use votex_domain::shared::value_object::AudioData;
use votex_infra::asr::paraformer::ParaformerProvider;
use votex_infra::asr::qwen3_asr::Qwen3AsrProvider;
use votex_infra::asr::sensevoice::SenseVoiceProvider;
use votex_infra::asr::wenet::WeNetProvider;
use votex_infra::asr::whisper::WhisperProvider;

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

fn test_wavs_dir() -> PathBuf {
    workspace_root().join("models/asr/firered-asr-ctc/test_wavs")
}

fn make_params() -> AsrParams {
    AsrParams {
        model: ModelId::new("cross-engine"),
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

/// 断言识别结果包含星期语义（各引擎按同一口径对比）
fn assert_weekday_semantic(engine_label: &str, text: &str) {
    assert!(
        !text.trim().is_empty(),
        "[{}] 识别结果不应为空",
        engine_label
    );
    assert!(
        text.contains("星期") || text.contains("礼拜") || text.contains("MONDAY")
            || text.contains("monday") || text.contains("TODAY") || text.contains("today"),
        "[{}] 应识别出星期相关语义，得到: '{}'",
        engine_label,
        text
    );
}

macro_rules! cross_engine_case {
    ($fn_name:ident, $label:expr, $provider_ty:ty, $dir:expr, $engine:expr) => {
        #[test]
        #[cfg_attr(
            not(feature = "slow-models"),
            ignore = concat!(
                "需本地模型与推理；跑法: cargo test -p votex-infra --test asr_cross_engine_0wav_test --features slow-models"
            )
        )]
        fn $fn_name() {
            let wav_path = test_wavs_dir().join("0.wav");
            let model_dir = workspace_root().join($dir);
            if !model_dir.exists() || !wav_path.exists() {
                eprintln!("⚠ 模型目录或测试音频不存在，跳过: {:?}", model_dir);
                return;
            }

            let provider = <$provider_ty>::new();
            let model = Model::new(
                ModelId::new($dir),
                &model_dir.to_string_lossy(),
                ModelKind::Asr,
                $engine,
            );
            provider.load(&model).expect("模型加载失败");

            let audio_data = read_test_wav(&wav_path);
            let result = provider
                .recognize(&audio_data, &make_params())
                .expect("识别失败");

            eprintln!("[{}] 0.wav 识别结果: '{}'", $label, result.text.trim());
            assert_weekday_semantic($label, &result.text);

            provider.unload().ok();
        }
    };
}

cross_engine_case!(
    cross_engine_sensevoice_0wav,
    "SenseVoice",
    SenseVoiceProvider,
    "models/asr/sensevoice",
    EngineKind::SenseVoice
);

cross_engine_case!(
    cross_engine_qwen3_asr_0wav,
    "Qwen3-ASR",
    Qwen3AsrProvider,
    "models/asr/qwen3-asr",
    EngineKind::Qwen3Asr
);

cross_engine_case!(
    cross_engine_wenet_0wav,
    "WeNet",
    WeNetProvider,
    "models/asr/wenet",
    EngineKind::WeNet
);

cross_engine_case!(
    cross_engine_paraformer_0wav,
    "Paraformer",
    ParaformerProvider,
    "models/asr/paraformer",
    EngineKind::Paraformer
);

/// Whisper-base：诊断性用例（不参与语义断言）
///
/// 实测（2026-10-06）：0.wav（中英混说）输出
/// 「昨天是 Monday 天天是拜二 天天×52」——出现 Whisper 经典的
/// 重复幻觉（repetition hallucination），base 是同系列最小模型，
/// 该行为属已知模型质量局限而非推理管线缺陷。此处只断言非空并打印结果，
/// 供跨引擎对比；语义断言见其他四个引擎用例。
#[test]
#[cfg_attr(
    not(feature = "slow-models"),
    ignore = "需本地模型与推理；跑法: cargo test -p votex-infra --test asr_cross_engine_0wav_test --features slow-models"
)]
fn cross_engine_whisper_base_0wav_诊断() {
    let wav_path = test_wavs_dir().join("0.wav");
    let model_dir = workspace_root().join("models/asr/whisper-base");
    if !model_dir.exists() || !wav_path.exists() {
        eprintln!("⚠ 模型目录或测试音频不存在，跳过");
        return;
    }

    let provider = WhisperProvider::new();
    let model = Model::new(
        ModelId::new("whisper-base"),
        &model_dir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::Whisper,
    );
    provider.load(&model).expect("模型加载失败");

    let audio_data = read_test_wav(&wav_path);
    let result = provider
        .recognize(&audio_data, &make_params())
        .expect("识别失败");

    eprintln!("[Whisper-base] 0.wav 识别结果: '{}'", result.text.trim());
    assert!(!result.text.trim().is_empty(), "识别结果不应为空");

    provider.unload().ok();
}
