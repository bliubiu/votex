//! IndexTTS-2.5 粤语（`yue`）通道端到端验证
//!
//! 背景：IndexTTS-2 与 IndexTTS-2.5 官方均未把粤语列为一等语种（官方五语
//! zh/en/ja/es/ar）。2.5 的粤语来自 tokenizer 词表里的 `yue` 语言标记 +
//! 训练数据覆盖，属「词表级支持」。因此必须验证该标记在 votex 侧真正生效，
//! 而不是被 `dialect_to_lang` 之外的路径静默回落成 `zh`。
//!
//! 判据（由硬到软，逐级加强）：
//!  1. 真实 tiktoken 词表中 `<|yue|>` 已注册为 special token，且 id 与 `<|zh|>` 不同
//!  2. `lang_id("yue")` 落在已注册语言区间内（< `N_REGISTERED_LANGUAGES`）
//!  3. `Frontend::prepare` 对 `yue` / `zh` 注入**不同的首个 token**
//!  4. 端到端：同一文本 `dialect=None` 与 `dialect=Cantonese` 合成的音频必须显著不同
//!
//! 第 4 条是决定性判据：若两次输出雷同，说明 `yue` 未真正进入 GPT 条件。
//!
//! 运行：
//! ```bash
//! cargo test -p votex-infra --test indextts25_yue_e2e_test -- --nocapture --test-threads=1
//! ```
//! 素材：`tmp/e2e_ch1.txt`（《奶爸大文豪》第一章，普通话）、
//! `models/tts/indextts25/prompts/yue_*.wav`（粤语参考音频）。
//!
//! 注意：`tmp/e2e_ch1.txt` 是**普通话**文本。以 `dialect=Cantonese` 合成它，
//! 验证的是「yue 音系被激活」而非「粤语音质」——粤语音质须用粤语文本另行评估。

use std::path::{Path, PathBuf};
use std::time::Instant;

use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{EngineKind, ModelId, ModelKind};
use votex_domain::shared::value_object::AudioData;
use votex_domain::tts::dialect::Dialect;
use votex_domain::tts::provider::TtsProvider;
use votex_domain::tts::value_object::{
    DenoiseLevel, Pitch, SegmentSize, Speed, TtsParams, VoiceId, Volume,
};

use votex_infra::tts::indextts25::bpe::N_REGISTERED_LANGUAGES;
use votex_infra::tts::indextts25::{lang_id, Frontend, IndexTts25Provider};

// ============================================================
// 路径与参数
// ============================================================

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

fn model_dir() -> PathBuf {
    workspace_root().join("models/tts/indextts25")
}

fn vocab_path() -> PathBuf {
    model_dir().join(votex_infra::tts::indextts25::frontend::VOCAB_FILE_NAME)
}

fn novel_ch1_path() -> PathBuf {
    workspace_root().join("tmp/e2e_ch1.txt")
}

fn engine_model() -> Model {
    Model::new(
        ModelId::new("indextts-2.5-onnx"),
        "IndexTTS-2.5-onnx",
        ModelKind::Tts,
        EngineKind::IndexTTS25,
    )
}

fn tts_params(engine: EngineKind, voice: &str, dialect: Option<Dialect>) -> TtsParams {
    TtsParams {
        engine,
        voice: VoiceId::new(voice, voice, engine),
        speed: Speed::default(),
        pitch: Pitch::default(),
        volume: Volume::default(),
        segment_size: SegmentSize::S500,
        segment_silence_ms: 300,
        crossfade_ms: 0,
        num_to_chinese: false,
        denoise: false,
        denoise_level: DenoiseLevel::Low,
        emotion: None,
        dialect,
    }
}

/// 取 `tmp/e2e_ch1.txt` 前 n 字。IndexTTS-2.5 是 fp32 CPU 自回归模型，
/// 全章（8000+ 字）需小时级，这里只取代表性片段。
fn novel_excerpt(n: usize) -> String {
    let content = std::fs::read_to_string(novel_ch1_path()).expect("读取 tmp/e2e_ch1.txt 失败");
    content.chars().take(n).collect::<String>().trim().to_string()
}

// ============================================================
// 1-3：免模型层（只需 900KB 词表，秒级）
// ============================================================

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test indextts25_yue_e2e_test --features slow-models")]
#[test]
fn yue_special_token_已在真实词表注册且区别于zh() {
    let vocab = vocab_path();
    if !vocab.is_file() {
        eprintln!("⚠ 未找到词表 {:?}，跳过", vocab);
        return;
    }
    let enc = votex_infra::tts::indextts25::Tiktoken::load(&vocab).expect("加载 tiktoken 词表失败");

    // 词表规模契约（mergeable 58836 + special 1674 = GPT 文本嵌入表行数）
    assert_eq!(enc.n_vocab(), votex_infra::tts::indextts25::EXPECTED_N_VOCAB);

    let yue = enc.special_id("<|yue|>");
    let zh = enc.special_id("<|zh|>");
    eprintln!("  <|yue|> = {yue:?}    <|zh|> = {zh:?}");

    assert!(yue.is_some(), "<|yue|> 未注册为 special token——粤语通道不存在");
    assert!(zh.is_some(), "<|zh|> 未注册为 special token");
    assert_ne!(yue, zh, "yue 与 zh 共用同一 special id，方言条件不可能生效");

    // 越界语言标记（闽南语/吴语）必须未注册，兑现 provider.rs 的已知限制
    for tag in ["<|minnan|>", "<|wuyu|>"] {
        assert!(
            enc.special_id(tag).is_none(),
            "{tag} 竟被注册，超出已注册语言区间（与 provider.rs 注释矛盾，需复核）"
        );
    }
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test indextts25_yue_e2e_test --features slow-models")]
#[test]
fn lang_id_yue_落在已注册区间且与zh不同() {
    let yue = lang_id("yue");
    let zh = lang_id("zh");
    eprintln!("  lang_id(yue) = {yue}    lang_id(zh) = {zh}    已注册上限 = {N_REGISTERED_LANGUAGES}");

    assert_eq!(yue, 99, "yue 在 LANGUAGE_DICT 中的序号应为 99");
    assert_eq!(zh, 1);
    assert_ne!(yue, zh);
    assert!(
        yue < N_REGISTERED_LANGUAGES as u32,
        "yue 序号 {yue} 越界（>= {N_REGISTERED_LANGUAGES}），special token 会超出 GPT 文本嵌入表"
    );

    // 已记录事实：以下方言虽在 LANGUAGE_DICT 中，但序号越界无法注册
    for (tag, expect_oor) in [("minnan", true), ("wuyu", true)] {
        let id = lang_id(tag);
        let oor = id >= N_REGISTERED_LANGUAGES as u32;
        assert_eq!(oor, expect_oor, "{tag} 越界状态与登记不符（id={id}）");
        eprintln!("  lang_id({tag}) = {id} -> 越界={oor}（不在已注册区间）");
    }
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test indextts25_yue_e2e_test --features slow-models")]
#[test]
fn frontend_对yue与zh注入不同首token() {
    let vocab = vocab_path();
    if !vocab.is_file() {
        eprintln!("⚠ 未找到词表 {:?}，跳过", vocab);
        return;
    }
    // use_normalization=false：与 provider::load 的实际取值一致（归一化未移植）
    let fe = Frontend::new(&vocab, false).expect("构建 Frontend 失败");

    let text = "你好，今天天气几好。";
    let segs_yue = fe.prepare(text, "yue", 450).expect("prepare(yue) 失败");
    let segs_zh = fe.prepare(text, "zh", 450).expect("prepare(zh) 失败");

    assert_eq!(segs_yue.len(), 1, "短文本应只有一段");
    assert_eq!(segs_zh.len(), 1);

    let yue_tok = &segs_yue[0];
    let zh_tok = &segs_zh[0];
    eprintln!("  yue 首 token = {}", yue_tok[0]);
    eprintln!("  zh  首 token = {}", zh_tok[0]);
    eprintln!("  yue token 数 = {}    zh token 数 = {}", yue_tok.len(), zh_tok.len());

    assert_ne!(
        yue_tok[0], zh_tok[0],
        "yue 与 zh 注入的首 token 相同——语言条件未进入文本序列"
    );
    // 首 token 即语言标记本身（<|yue|> / <|zh|> 是 special，不会被 BPE 合并）
    let enc = votex_infra::tts::indextts25::Tiktoken::load(&vocab).expect("加载词表失败");
    assert_eq!(yue_tok[0], enc.special_id("<|yue|>").unwrap());
    assert_eq!(zh_tok[0], enc.special_id("<|zh|>").unwrap());

    // yue 不走中文归一化、也不做小写化，与 zh 分支存在实质差异
    assert_ne!(yue_tok, zh_tok, "yue 与 zh 的完整 token 序列相同");
}

// ============================================================
// 4：端到端（需 5.1GB 模型，fp32 CPU 自回归，耗时以分钟计）
// ============================================================

/// 归一化差异度：diff_rms / signal_rms。0 = 完全相同，1 = 能量相当的不同信号。
fn divergence(a: &AudioData, b: &AudioData) -> f64 {
    let n = a.samples.len().min(b.samples.len());
    assert!(n > 0, "音频为空");
    let sig: f64 = a.samples[..n].iter().map(|v| (*v as f64).powi(2)).sum();
    let dif: f64 = a.samples[..n]
        .iter()
        .zip(&b.samples[..n])
        .map(|(x, y)| (*x as f64 - *y as f64).powi(2))
        .sum();
    if sig <= 0.0 {
        return f64::INFINITY;
    }
    (dif / sig).sqrt()
}

fn write_wav(audio: &AudioData, name: &str) -> PathBuf {
    let dir = workspace_root().join("tmp");
    std::fs::create_dir_all(&dir).expect("创建 tmp 目录失败");
    let path = dir.join(name);
    votex_infra::audio::wav::WavWriter::write(audio, &path).expect("WAV 写入失败");
    path
}

/// 决定性判据：同一文本、同一音色，`dialect=Cantonese` 与 `dialect=None` 必须产出不同音频。
#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test indextts25_yue_e2e_test --features slow-models")]
#[test]
fn e2e_粤语通道生效_zh与yue输出必须不同() {
    if !model_dir().is_dir() || !novel_ch1_path().exists() {
        eprintln!("⚠ 模型或素材缺失，跳过（模型 {:?}）", model_dir());
        return;
    }
    // 粤语参考音频必须存在，否则退化成 default.wav，测不到方言差异
    let prompts = model_dir().join("prompts");
    if !prompts.join("yue_long_10350.wav").is_file() {
        eprintln!("⚠ 缺少粤语参考音频 prompts/yue_long_10350.wav，跳过");
        return;
    }

    let text = novel_excerpt(40);
    let voice = "yue_long_10350";
    eprintln!("\n########## IndexTTS-2.5 粤语通道 e2e（{} 字）##########", text.chars().count());
    eprintln!("  文本: {text}");
    eprintln!("  音色: {voice}（12.90s 粤语参考音频）");

    let tts = IndexTts25Provider::new();
    let t0 = Instant::now();
    tts.load(&engine_model()).expect("IndexTTS-2.5 加载失败");
    eprintln!("✓ 模型加载完成 ({:.1}s)", t0.elapsed().as_secs_f32());

    // zh 通道
    let t1 = Instant::now();
    let zh_audio = tts
        .synthesize(&text, &VoiceId::new(voice, voice, EngineKind::IndexTTS25), &tts_params(EngineKind::IndexTTS25, voice, None))
        .expect("zh 合成失败");
    let zh_secs = t1.elapsed().as_secs_f64();
    let zh_out = write_wav(&zh_audio, "indextts25_yue_e2e_zh.wav");
    eprintln!(
        "✓ zh  合成: 音频 {:.1}s, 耗时 {:.1}s ({:.2}x 实时) -> {:?}",
        zh_audio.samples.len() as f64 / zh_audio.sample_rate as f64,
        zh_secs,
        zh_secs / (zh_audio.samples.len() as f64 / zh_audio.sample_rate as f64),
        zh_out
    );

    // yue 通道
    let t2 = Instant::now();
    let yue_audio = tts
        .synthesize(
            &text,
            &VoiceId::new(voice, voice, EngineKind::IndexTTS25),
            &tts_params(EngineKind::IndexTTS25, voice, Some(Dialect::Cantonese)),
        )
        .expect("yue 合成失败");
    let yue_secs = t2.elapsed().as_secs_f64();
    let yue_out = write_wav(&yue_audio, "indextts25_yue_e2e_yue.wav");
    eprintln!(
        "✓ yue 合成: 音频 {:.1}s, 耗时 {:.1}s ({:.2}x 实时) -> {:?}",
        yue_audio.samples.len() as f64 / yue_audio.sample_rate as f64,
        yue_secs,
        yue_secs / (yue_audio.samples.len() as f64 / yue_audio.sample_rate as f64),
        yue_out
    );

    tts.unload().ok();

    // ---- 决定性判据 ----
    let d = divergence(&zh_audio, &yue_audio);
    eprintln!("\n  zh/yue 归一化差异度 = {d:.4}（0 = 完全相同）");
    assert!(
        d > 0.05,
        "dialect=Cantonese 与 dialect=None 输出几乎相同（差异度 {d:.4}）——yue 通道未生效"
    );
    eprintln!("✓ yue 通道生效：粤语条件已进入 GPT，输出显著区别于 zh");
}

/// 粤语音质试听素材：用真正的粤语文本合成，而非普通话文本套 yue 音系。
#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test indextts25_yue_e2e_test --features slow-models")]
#[test]
fn e2e_粤语文本合成_供试听() {
    if !model_dir().is_dir() {
        eprintln!("⚠ 模型缺失，跳过");
        return;
    }
    if !model_dir().join("prompts/yue_long_10350.wav").is_file() {
        eprintln!("⚠ 缺少粤语参考音频，跳过");
        return;
    }

    // 取自数据集 index.csv 的真实粤语句子
    let cases: [(&str, &str); 3] = [
        ("yue_long_10350", "系吖，都系一啲广州嘅老屋，好有特色。"),
        ("yue_long_10139", "好嘅。窗帘都好污糟吖，要拆落嚟洗吓喇。"),
        ("yue_short_01662", "呢个收纳箱可以叠起身，悭位啲？"),
    ];

    let tts = IndexTts25Provider::new();
    tts.load(&engine_model()).expect("IndexTTS-2.5 加载失败");

    for (voice, text) in cases {
        let t0 = Instant::now();
        let audio = tts
            .synthesize(
                text,
                &VoiceId::new(voice, voice, EngineKind::IndexTTS25),
                &tts_params(EngineKind::IndexTTS25, voice, Some(Dialect::Cantonese)),
            )
            .expect("粤语合成失败");
        let secs = t0.elapsed().as_secs_f64();
        let audio_secs = audio.samples.len() as f64 / audio.sample_rate as f64;
        let out = write_wav(&audio, &format!("indextts25_yue_{voice}.wav"));
        eprintln!(
            "  [{voice}] {} 字 -> 音频 {:.1}s / 耗时 {:.1}s ({:.2}x 实时)  {:?}",
            text.chars().count(),
            audio_secs,
            secs,
            secs / audio_secs,
            out
        );
        eprintln!("      {text}");
    }

    tts.unload().ok();
}

/// 生成 ASR 闭环 e2e 素材：粤语合成 asr_e2e_test 的 ground truth 短句。
///
/// `asr_e2e_test.rs` 依赖 `tmp/indextts25_e2e_yue.wav`（文本已知：
/// 「今日天气几好，我哋一齐去饮茶啦。」）。tmp 被清理后可用本测试一键重建。
#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test indextts25_yue_e2e_test --features slow-models")]
#[test]
fn e2e_生成asr闭环素材_今日天气几好() {
    if !model_dir().is_dir() {
        eprintln!("⚠ 模型缺失，跳过");
        return;
    }
    if !model_dir().join("prompts/yue_long_10350.wav").is_file() {
        eprintln!("⚠ 缺少粤语参考音频，跳过");
        return;
    }

    const GROUND_TRUTH: &str = "今日天气几好，我哋一齐去饮茶啦。";
    let voice = "yue_long_10350";

    let tts = IndexTts25Provider::new();
    tts.load(&engine_model()).expect("IndexTTS-2.5 加载失败");

    let t0 = Instant::now();
    let audio = tts
        .synthesize(
            GROUND_TRUTH,
            &VoiceId::new(voice, voice, EngineKind::IndexTTS25),
            &tts_params(EngineKind::IndexTTS25, voice, Some(Dialect::Cantonese)),
        )
        .expect("粤语合成失败");
    let secs = t0.elapsed().as_secs_f64();
    let audio_secs = audio.samples.len() as f64 / audio.sample_rate as f64;
    let out = write_wav(&audio, "indextts25_e2e_yue.wav");

    tts.unload().ok();

    eprintln!(
        "✓ ASR 闭环素材已生成: '{GROUND_TRUTH}' -> 音频 {:.1}s / 耗时 {:.1}s ({:.2}x 实时)  {out:?}",
        audio_secs,
        secs,
        secs / audio_secs
    );
    assert!(audio_secs > 1.0, "合成音频过短（{audio_secs:.1}s），疑似空输出");
}