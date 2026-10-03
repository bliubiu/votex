/// TTS → ASR 端到端流水线验证
///
/// 1. 使用 Qwen3-TTS 将 tmp/novel.txt 小说片段合成为语音
/// 2. 使用 SenseVoice ASR 将生成的语音转写为文字
/// 3. 对比原始文本与 ASR 识别结果，验证流水线运行正常
///
/// 运行方式:
/// ```bash
/// cargo test -p votex-infra --test tts_asr_e2e_test -- --nocapture
/// ```
use std::path::PathBuf;
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

fn qwen3_tts_dir() -> PathBuf {
    let mut dir = workspace_root();
    dir.push("models/tts/qwen3-tts");
    dir
}

fn sensevoice_dir() -> PathBuf {
    let mut dir = workspace_root();
    dir.push("models/asr/sensevoice");
    dir
}

fn novel_path() -> PathBuf {
    let mut dir = workspace_root();
    dir.push("tmp/novel.txt");
    dir
}

/// 读取小说前几段作为 TTS 输入（控制在 200 字以内避免合成时间过长）
fn read_novel_excerpt() -> String {
    let content = std::fs::read_to_string(novel_path())
        .expect("读取 novel.txt 失败");
    // 取前 150 字左右
    let excerpt: String = content.chars().take(150).collect();
    // 清理可能的乱码，保留中文和标点
    excerpt.trim().to_string()
}

fn make_tts_params() -> TtsParams {
    TtsParams {
        engine: EngineKind::Qwen3Tts,
        voice: VoiceId::new("default", "默认音色", EngineKind::Qwen3Tts),
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

#[test]
fn test_qwen3tts_to_sensevoice_pipeline() {
    // ─── 检查模型是否都存在 ─────────────────────────
    if !qwen3_tts_dir().exists() {
        eprintln!("⚠ Qwen3-TTS 模型目录不存在: {:?}，跳过测试", qwen3_tts_dir());
        return;
    }
    if !sensevoice_dir().exists() {
        eprintln!("⚠ SenseVoice 模型目录不存在: {:?}，跳过测试", sensevoice_dir());
        return;
    }
    if !novel_path().exists() {
        eprintln!("⚠ novel.txt 不存在: {:?}，跳过测试", novel_path());
        return;
    }

    let text = read_novel_excerpt();
    eprintln!("=== TTS → ASR 端到端流水线验证 ===\n");
    eprintln!("输入文本 ({} 字):", text.chars().count());
    eprintln!("  '{}'\n", text);

    // ─── 1. Qwen3-TTS 合成语音 ───────────────────────
    eprintln!("--- Step 1: Qwen3-TTS 语音合成 ---");
    let mut tts = votex_infra::tts::qwen3_tts::Qwen3TtsProvider::new();
    let tts_model = Model::new(
        ModelId::new("qwen3-tts"),
        &qwen3_tts_dir().to_string_lossy(),
        ModelKind::Tts,
        EngineKind::Qwen3Tts,
    );

    let tts_result = tts.load(&tts_model);
    if let Err(ref e) = tts_result {
        eprintln!("❌ Qwen3-TTS 加载失败: {}", e);
        eprintln!("  可能原因：模型文件不完整或 INT4 ONNX 文件缺失");
        eprintln!("  跳过 TTS 步骤，使用 SenseVoice 直接识别原始音频");
        tts_result.expect("Qwen3-TTS 加载失败");
    }
    eprintln!("✓ Qwen3-TTS 模型加载成功");

    let voice = VoiceId::new("default", "默认音色", EngineKind::Qwen3Tts);
    let params = make_tts_params();
    let audio = tts.synthesize(&text, &voice, &params)
        .expect("Qwen3-TTS 合成失败");
    eprintln!("✓ TTS 合成成功: {} 采样点, {} Hz\n", audio.samples.len(), audio.sample_rate);

    // 保存到 WAV 文件便于人工检查
    let output_dir = {
        let mut p = workspace_root();
        p.push("target/test_output");
        std::fs::create_dir_all(&p).expect("创建输出目录失败");
        p
    };
    let wav_path = output_dir.join("qwen3tts_output.wav");
    votex_infra::audio::wav::WavWriter::write(&audio, &wav_path)
        .expect("WAV 写入失败");
    eprintln!("  语音已保存至: {:?}\n", wav_path);

    tts.unload().ok();

    // ─── 2. SenseVoice ASR 识别 ──────────────────────
    eprintln!("--- Step 2: SenseVoice ASR 语音识别 ---");
    let mut asr = votex_infra::asr::sensevoice::SenseVoiceProvider::new();
    let asr_model = Model::new(
        ModelId::new("sensevoice"),
        &sensevoice_dir().to_string_lossy(),
        ModelKind::Asr,
        EngineKind::SenseVoice,
    );
    asr.load(&asr_model).expect("SenseVoice 加载失败");
    eprintln!("✓ SenseVoice 模型加载成功");

    let asr_params = make_asr_params();
    let result = asr.recognize(&audio, &asr_params)
        .expect("SenseVoice 识别失败");
    let recognized = result.text.trim();

    asr.unload().ok();

    // ─── 3. 对比分析（基于内容覆盖率，非严格位置匹配） ──
    eprintln!("\n=== 流水线验证结果 ===");
    eprintln!("原始文本:    '{}'", text);
    eprintln!("ASR 识别结果: '{}'", recognized);
    eprintln!("识别字数:     {}", recognized.chars().count());

    // 过滤中英文标点和空白
    fn is_punct(c: char) -> bool {
        c.is_ascii_punctuation() || matches!(c,
            '，' | '。' | '、' | '？' | '！' | '：' | '；' | '“' | '”' | '‘' | '’' |
            '（' | '）' | '【' | '】' | '《' | '》' | '—' | '…' | '·' | '\n' | '\r' | '　' | ' ')
    }
    let orig_chars: Vec<char> = text.chars().filter(|c| !is_punct(*c)).collect();
    let recog_chars: Vec<char> = recognized.chars().filter(|c| !is_punct(*c)).collect();

    // 方法1：内容覆盖率（原始文本中有多少字符出现在 ASR 结果中）
    let recog_set: std::collections::HashSet<char> = recog_chars.iter().copied().collect();
    let covered = orig_chars.iter().filter(|c| recog_set.contains(c)).count();
    let coverage = if orig_chars.is_empty() {
        0.0
    } else {
        covered as f64 / orig_chars.len() as f64 * 100.0
    };

    // 方法2：序列编辑相似度（允许插入/删除，计算最长公共子序列）
    let lcs_len = {
        let m = orig_chars.len();
        let n = recog_chars.len();
        // 只计算前 min(m, n) + 50 的范围内的 LCS，控制计算量
        let limit_m = m.min(n + 50);
        let limit_n = n.min(m + 50);
        let mut dp = vec![vec![0usize; limit_n + 1]; limit_m + 1];
        for i in 1..=limit_m {
            for j in 1..=limit_n {
                if orig_chars[i - 1] == recog_chars[j - 1] {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
                }
            }
        }
        dp[limit_m][limit_n]
    };
    let seq_similarity = if orig_chars.is_empty() {
        100.0
    } else {
        lcs_len as f64 / orig_chars.len() as f64 * 100.0
    };

    eprintln!("\n  内容覆盖率: {}/{} ({:.1}%)", covered, orig_chars.len(), coverage);
    eprintln!("  序列相似度 (LCS): {}/{} ({:.1}%)", lcs_len, orig_chars.len(), seq_similarity);
    eprintln!("  原始去标点: '{}'", orig_chars.iter().collect::<String>());
    eprintln!("  识别去标点: '{}'", recog_chars.iter().collect::<String>());

    // 结果必须非空
    assert!(!recognized.is_empty(), "ASR 识别结果不应为空");
    // 使用内容覆盖率作为主要判断指标（>30% 即说明流水线基本正常）
    assert!(coverage > 30.0,
        "TTS→ASR 流水线内容覆盖率过低 ({:.1}%)，说明流水线存在问题", coverage);
    // 序列相似度也至少 >15%
    assert!(seq_similarity > 15.0,
        "TTS→ASR 流水线序列相似度过低 ({:.1}%)", seq_similarity);

    eprintln!("\n✅ TTS → ASR 端到端流水线验证通过 (覆盖率: {:.1}%, 序列相似度: {:.1}%)", coverage, seq_similarity);
}
