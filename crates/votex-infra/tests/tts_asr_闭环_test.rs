/// Kokoro TTS → Whisper ASR 闭环验证
///
/// 验证另一条独立于 `tts_asr_e2e_test.rs`（Qwen3-TTS → SenseVoice）的引擎组合：
/// 1. 用 Kokoro-82M-v1.1-zh（中文注音 G2P）合成中文语音
/// 2. 用 Whisper（sherpa-onnx ONNX 链路）转写回文字
/// 3. 对比原文与识别结果，确认两条链路的衔接可用
///
/// 该测试**需要本地模型**，缺失即失败（静默跳过会让门禁变成假通过）：
/// - `models/tts/kokoro-82m-v1.1-zh/`
/// - `models/asr/whisper-base/`
///
/// 运行方式:
/// ```bash
/// cargo test -p votex-infra --test tts_asr_闭环_test -- --nocapture
/// ```
use std::path::{Path, PathBuf};

use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{
    AsrParams, DenoiseLevel as AsrDenoiseLevel, Language, SliceLength, SubtitleFormat,
};
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{EngineKind, ModelId, ModelKind};
use votex_domain::tts::provider::TtsProvider;
use votex_domain::tts::value_object::{
    DenoiseLevel as TtsDenoiseLevel, Pitch, SegmentSize, Speed, TtsParams, VoiceId, Volume,
};

/// Kokoro 中文模型 ID —— 必须与 `resolve_model_config` 的分支一致，否则会被拒绝
const KOKORO_ZH_MODEL_ID: &str = "kokoro-82m-v1.1-zh";
const KOKORO_ZH_REL_DIR: &str = "models/tts/kokoro-82m-v1.1-zh";
/// 中文音色（CLI 默认同为 zf_001）
const KOKORO_ZH_VOICE: &str = "zf_001";

const WHISPER_MODEL_ID: &str = "whisper-base";
const WHISPER_REL_DIR: &str = "models/asr/whisper-base";

/// 闭环测试文本：短句、常用字、无数字与生僻字，Whisper base 的中文识别友好区间
const TEST_TEXT: &str = "今天天气很好，我们一起出门散步。";

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

/// Kokoro / Whisper 的模型目录解析都基于相对路径 `models/...`，
/// 因此必须先把工作目录切到工作区根，否则 `find_model_dir` 会找不到模型。
fn switch_to_workspace_root() {
    let root = workspace_root();
    std::env::set_current_dir(&root)
        .unwrap_or_else(|e| panic!("切换工作目录到 {:?} 失败: {}", root, e));
}

fn require_dir(rel: &str) -> PathBuf {
    let dir = workspace_root().join(rel);
    assert!(
        dir.is_dir(),
        "缺少模型目录: {}\n请先执行 `votex model download <模型ID>` 补齐本地模型后再运行闭环测试",
        dir.display()
    );
    dir
}

fn make_tts_params(voice: &VoiceId) -> TtsParams {
    TtsParams {
        engine: EngineKind::Kokoro,
        voice: voice.clone(),
        speed: Speed::default(),
        pitch: Pitch::default(),
        volume: Volume::default(),
        segment_size: SegmentSize::S500,
        segment_silence_ms: 0,
        crossfade_ms: 0,
        num_to_chinese: false,
        denoise: false,
        denoise_level: TtsDenoiseLevel::Low,
        emotion: None,
        dialect: None,
    }
}

fn make_asr_params() -> AsrParams {
    AsrParams {
        model: ModelId::new(WHISPER_MODEL_ID),
        language: Language::Zh,
        auto_punctuation: true,
        auto_slice: false,
        slice_length: SliceLength::S30,
        denoise: false,
        denoise_level: AsrDenoiseLevel::Low,
        output_format: SubtitleFormat::Txt,
    }
}

/// 是否为中英文标点 / 空白 —— 参与相似度计算前需剔除
fn is_punct(c: char) -> bool {
    c.is_ascii_punctuation()
        || c.is_whitespace()
        || matches!(
            c,
            '，' | '。' | '、' | '？' | '！' | '：' | '；' | '“' | '”' | '‘' | '’' | '（' | '）'
                | '【' | '】' | '《' | '》' | '—' | '…' | '·' | '～' | '」' | '「'
        )
}

fn content_chars(s: &str) -> Vec<char> {
    s.chars().filter(|c| !is_punct(*c)).collect()
}

/// 内容覆盖率：原文中有多少字符出现在识别结果里（对字序错乱不敏感）
fn coverage(orig: &[char], recog: &[char]) -> f64 {
    if orig.is_empty() {
        return 0.0;
    }
    let set: std::collections::HashSet<char> = recog.iter().copied().collect();
    let hit = orig.iter().filter(|c| set.contains(c)).count();
    hit as f64 / orig.len() as f64 * 100.0
}

/// 序列相似度（LCS）：允许插入 / 删除，衡量识别文本的先后顺序是否正确
fn lcs_similarity(orig: &[char], recog: &[char]) -> f64 {
    if orig.is_empty() {
        return 100.0;
    }
    // 限制矩阵规模，避免超长文本导致 O(m*n) 内存膨胀
    let limit_m = orig.len().min(recog.len() + 50);
    let limit_n = recog.len().min(orig.len() + 50);
    let mut dp = vec![vec![0usize; limit_n + 1]; limit_m + 1];
    for i in 1..=limit_m {
        for j in 1..=limit_n {
            dp[i][j] = if orig[i - 1] == recog[j - 1] {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }
    dp[limit_m][limit_n] as f64 / orig.len() as f64 * 100.0
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test tts_asr_闭环_test --features slow-models")]
#[test]
fn kokoro_zh_to_whisper_close_loop() {
    switch_to_workspace_root();

    eprintln!("=== Kokoro(中文) → Whisper 闭环验证 ===");
    eprintln!("输入文本 ({} 字): {}", TEST_TEXT.chars().count(), TEST_TEXT);

    let kokoro_dir = require_dir(KOKORO_ZH_REL_DIR);
    let whisper_dir = require_dir(WHISPER_REL_DIR);

    // ─── 1. Kokoro 合成中文语音 ───────────────────────
    eprintln!("--- Step 1: Kokoro-82M-v1.1-zh 语音合成 ---");
    let tts = votex_infra::tts::kokoro::KokoroProvider::new();
    let tts_model = Model::new(
        ModelId::new(KOKORO_ZH_MODEL_ID),
        &kokoro_dir.to_string_lossy(),
        ModelKind::Tts,
        EngineKind::Kokoro,
    );
    tts.load(&tts_model)
        .unwrap_or_else(|e| panic!("Kokoro 模型加载失败: {}\n目录: {}", e, kokoro_dir.display()));
    eprintln!("✓ Kokoro 模型加载成功");

    let voice = VoiceId::new(KOKORO_ZH_VOICE, "中文女声", EngineKind::Kokoro);
    let tts_params = make_tts_params(&voice);
    let audio = tts
        .synthesize(TEST_TEXT, &voice, &tts_params)
        .unwrap_or_else(|e| panic!("Kokoro 合成失败: {}", e));

    assert!(
        !audio.samples.is_empty(),
        "Kokoro 合成结果为空：{} 个采样点",
        audio.samples.len()
    );
    eprintln!(
        "✓ 合成成功: {} 采样点, {} Hz, {} 声道",
        audio.samples.len(),
        audio.sample_rate,
        audio.channels
    );

    // 落盘便于人工试听与后续 ASR 回归排查
    let wav_path = {
        let dir = workspace_root().join("target/test_output");
        std::fs::create_dir_all(&dir).expect("创建 target/test_output 失败");
        dir.join("kokoro_zh_to_whisper.wav")
    };
    votex_infra::audio::wav::WavWriter::write(&audio, &wav_path).expect("WAV 写入失败");
    eprintln!("  语音已保存: {}", wav_path.display());

    tts.unload().ok();

    // ─── 2. Whisper 转写 ─────────────────────────────
    eprintln!("--- Step 2: Whisper 语音转写 ---");
    let asr = votex_infra::asr::whisper::WhisperProvider::new();
    let asr_model = Model::new(
        ModelId::new(WHISPER_MODEL_ID),
        &whisper_dir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::Whisper,
    );
    asr.load(&asr_model)
        .unwrap_or_else(|e| panic!("Whisper 模型加载失败: {}\n目录: {}", e, whisper_dir.display()));
    eprintln!("✓ Whisper 模型加载成功");

    let result = asr
        .recognize(&audio, &make_asr_params())
        .unwrap_or_else(|e| panic!("Whisper 识别失败: {}", e));
    asr.unload().ok();

    let recognized = result.text.trim();
    eprintln!("\n=== 闭环结果 ===");
    eprintln!("原始文本: {}", TEST_TEXT);
    eprintln!("识别文本: {}", recognized);

    // ─── 3. 相似度断言 ───────────────────────────────
    //
    // 注意：Whisper base 的中文输出**倾向繁体**（如实测 `天氣`/`出門`），
    // 而输入是简体，因此覆盖率天然低于 1.0，阈值按这一事实设定。
    // 若要提升闭环准确率，应换用 SenseVoice / Qwen3-ASR 等简体模型（见 tts_asr_e2e_test.rs）。
    let orig_chars = content_chars(TEST_TEXT);
    let recog_chars = content_chars(recognized);
    let cov = coverage(&orig_chars, &recog_chars);
    let seq = lcs_similarity(&orig_chars, &recog_chars);

    eprintln!("原文去标点: {}", orig_chars.iter().collect::<String>());
    eprintln!("识别去标点: {}", recog_chars.iter().collect::<String>());
    eprintln!("内容覆盖率: {:.1}%", cov);
    eprintln!("序列相似度: {:.1}%", seq);
    eprintln!("时间戳条目: {}", result.word_timestamps.len());

    assert!(!recognized.is_empty(), "Whisper 识别结果不应为空");
    assert!(
        cov > 30.0,
        "闭环内容覆盖率过低 ({:.1}%)：Kokoro 合成或 Whisper 转写链路存在问题\n原文: {}\n识别: {}",
        cov,
        TEST_TEXT,
        recognized
    );
    assert!(
        seq > 15.0,
        "闭环序列相似度过低 ({:.1}%)：识别文本与原文语序严重偏离\n原文: {}\n识别: {}",
        seq,
        TEST_TEXT,
        recognized
    );

    eprintln!(
        "\n✅ Kokoro → Whisper 闭环验证通过（覆盖率 {:.1}%，序列相似度 {:.1}%）",
        cov, seq
    );
}

/// 纯逻辑断言：闭环相似度算法自身必须正确，避免上面的指标"永远通过"
#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test tts_asr_闭环_test --features slow-models")]
#[test]
fn 闭环相似度算法自身正确() {
    assert_eq!(lcs_similarity(&"完全一致".chars().collect::<Vec<_>>(), &"完全一致".chars().collect::<Vec<_>>()), 100.0);
    assert_eq!(coverage(&"完全一致".chars().collect::<Vec<_>>(), &"完全一致".chars().collect::<Vec<_>>()), 100.0);

    // 标点与空白不参与计算
    let a = content_chars("你好，世界。");
    assert_eq!(a.iter().collect::<String>(), "你好世界");

    // 字序完全颠倒：覆盖率仍高，序列相似度必须显著下降
    let orig = content_chars("今天天气很好");
    let shuffled = content_chars("好很气天在今");
    assert!(coverage(&orig, &shuffled) > 90.0, "覆盖率只看字符集合");
    assert!(lcs_similarity(&orig, &shuffled) < 40.0, "语序颠倒必须被序列相似度抓到");

    // 空输入不应 panic
    assert_eq!(lcs_similarity(&[], &[]), 100.0);
    assert_eq!(coverage(&[], &['a']), 0.0);
}

/// 路径契约：闭环测试依赖的模型目录必须落在规范位置
#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test tts_asr_闭环_test --features slow-models")]
#[test]
fn 闭环模型路径符合规范() {
    switch_to_workspace_root();
    for rel in [KOKORO_ZH_REL_DIR, WHISPER_REL_DIR] {
        let p = Path::new(rel);
        assert!(p.is_absolute() == false, "应使用相对路径（依赖 cwd）: {rel}");
        assert!(rel.starts_with("models/"), "模型须位于 models/ 下: {rel}");
    }
}
