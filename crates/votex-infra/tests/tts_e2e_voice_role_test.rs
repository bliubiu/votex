/// 音色区分度 / 多角色音色 / 自然度 端到端验证（Kokoro、Qwen3-TTS、CosyVoice3、IndexTTS-2.5 四引擎）
///
/// 既有 e2e（`tts_asr_e2e_test` / `tts_e2e_multi_engine_test`）只验证 ASR 闭环内容
/// 正确性，未覆盖音色与多角色维度。本测试补齐三条音频级判据：
///
/// 1. **音色可控**：同一句台词换音色合成，campplus 说话人嵌入（192 维）余弦相似度满足
///    「同音色异文本 > 跨音色同文本 + 0.05」——音频级证明音色参数真实生效；
/// 2. **多角色**：`extract_role_candidates` 提取角色 → `build_role_map` 分配音色 →
///    逐角色合成真实台词 → 跨角色相似度显著低于同音色基线（角色间可区分），
///    并断言自动分配不丢路由（旁白 / 台词兜底 / 每个角色都有音色）；
/// 3. **自然度**：客观信号指标（时长 / RMS / 削波率 / 直流偏置 / 语音占比 / 首尾静音 /
///    样本间最大跳变）+ SenseVoice ASR 闭环内容覆盖率（可懂度代理）。
///
/// 素材：`tmp/e2e_ch1.txt`（《奶爸大文豪》第一章，含真实弯引号台词）。
/// 试听产物：`target/test_output/voice_role/*.wav`（人工抽检无爆音 / 截断 / 吞字）。
///
/// 运行方式（四引擎串行，避免同时占用数 GB 内存）:
/// ```bash
/// cargo test -p votex-infra --test tts_e2e_voice_role_test --features slow-models -- --test-threads=1 --nocapture
/// ```
use std::collections::HashSet;
use std::path::PathBuf;

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session as OrtSession;

use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, Language};
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{EngineKind, ModelId, ModelKind};
use votex_domain::shared::value_object::AudioData;
use votex_domain::tts::provider::TtsProvider;
use votex_domain::tts::role::{
    build_role_map, extract_role_candidates, RoleAssignments, RoleCandidate,
};
use votex_domain::tts::value_object::{
    DenoiseLevel, Pitch, SegmentSize, Speed, TtsParams, VoiceGender, VoiceId, VoiceLocale,
    VoiceMeta, Volume,
};
use votex_infra::tts::indextts25::{dsp, engines};

/// 多角色台词素材不足时的回退（第一章真实对白，保证 ≥8 字）
const FALLBACK_ROLES: [(&str, &str); 2] = [
    ("张行军", "重儿，你真的醒了，爸爸感觉像是做梦一样。"),
    ("芃芃", "爸爸刚醒，当然要好好照顾啦。"),
];

// ===================== 基础设施 =====================

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

/// Kokoro / IndexTTS25 等走相对路径 `models/...` 查找模型，先把工作目录切到工作区根
fn switch_to_workspace_root() {
    let root = workspace_root();
    std::env::set_current_dir(&root)
        .unwrap_or_else(|e| panic!("切换工作目录到 {:?} 失败: {}", root, e));
}

/// 读取第一章全文（角色提取与台词素材来源）
fn read_chapter() -> String {
    let path = workspace_root().join("tmp/e2e_ch1.txt");
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("读取素材 {:?} 失败: {}", path, e))
}

fn make_tts_params(engine: EngineKind, voice: &VoiceId) -> TtsParams {
    TtsParams {
        engine,
        voice: voice.clone(),
        speed: Speed::default(),
        pitch: Pitch::default(),
        volume: Volume::default(),
        segment_size: SegmentSize::S500,
        segment_silence_ms: 0,
        crossfade_ms: 0,
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

/// 是否为中英文标点 / 空白（覆盖率计算前剔除）
fn is_punct(c: char) -> bool {
    c.is_ascii_punctuation()
        || c.is_whitespace()
        || matches!(
            c,
            '，' | '。' | '、' | '？' | '！' | '：' | '；' | '“' | '”' | '‘' | '’' | '（' | '）'
                | '【' | '】' | '《' | '》' | '—' | '…' | '·' | '～' | '」' | '「'
        )
}

/// 去掉括注（如「（peng两声）」）与包裹引号，只留可朗读正文
fn sanitize_line(line: &str) -> String {
    let mut out = String::new();
    let mut depth = 0usize;
    for ch in line.chars() {
        match ch {
            '（' | '(' => depth += 1,
            '）' | ')' => depth = depth.saturating_sub(1),
            c if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out.trim()
        .trim_matches(|c| matches!(c, '"' | '“' | '”' | '‘' | '’'))
        .to_string()
}

/// 按展示名推断性别（Qwen3 音色无 zf_/zm_ 前缀约定，展示名含「女声/男声/大叔音」）
fn gender_by_display(display: &str) -> VoiceGender {
    if display.contains('女') {
        VoiceGender::Female
    } else if display.contains("男") || display.contains("大叔") {
        VoiceGender::Male
    } else {
        VoiceGender::Neutral
    }
}

// ===================== campplus 说话人相似度 =====================

/// campplus 说话人嵌入验证器（复用 IndexTTS-2.5 的公开 dsp/engines 接口）
struct SpeakerSim {
    session: OrtSession,
}

impl SpeakerSim {
    fn new() -> Self {
        votex_infra::shared::ensure_ort_dylib_path();
        let root = workspace_root();
        let candidates = [
            root.join("models/tts/indextts25/campplus.onnx"),
            root.join("models/tts/cosyvoice/campplus.onnx"),
        ];
        let path = candidates
            .iter()
            .find(|p| p.is_file())
            .cloned()
            .unwrap_or_else(|| {
                panic!(
                    "未找到 campplus 说话人嵌入模型，音色区分度验证无法进行，候选路径: {:?}",
                    candidates
                )
            });
        let session = OrtSession::builder()
            .expect("ort Session builder 创建失败")
            .with_optimization_level(GraphOptimizationLevel::All)
            .expect("设置图优化级别失败")
            .with_intra_threads(1)
            .expect("设置 ORT 线程数失败")
            .commit_from_file(&path)
            .unwrap_or_else(|e| panic!("campplus.onnx 加载失败 {:?}: {}", path, e));
        Self { session }
    }

    /// 提取 192 维说话人嵌入（非 16kHz 输入先 sinc-Hann 重采样）
    fn embed(&mut self, audio: &AudioData) -> Vec<f32> {
        assert!(!audio.samples.is_empty(), "待验证音频为空");
        let audio16 = if audio.sample_rate == 16000 {
            audio.samples.clone()
        } else {
            dsp::resample_sinc_hann(&audio.samples, audio.sample_rate as usize, 16000, 6, 0.99)
                .unwrap_or_else(|e| panic!("重采样到 16kHz 失败: {}", e))
        };
        let fbank =
            dsp::campplus_fbank(&audio16).unwrap_or_else(|e| panic!("campplus fbank 提取失败: {}", e));
        let frames = fbank.len() / 80;
        assert!(frames > 0, "音频过短，fbank 帧数为 0（样本 {} 个）", audio16.len());
        let fbank_t = ndarray::Array3::from_shape_vec((1, frames, 80), fbank)
            .unwrap_or_else(|e| panic!("fbank 形状错误: {}", e))
            .into_dyn();
        let emb = engines::campplus_run(&mut self.session, fbank_t)
            .unwrap_or_else(|e| panic!("campplus 推理失败: {}", e));
        emb.as_slice().expect("说话人嵌入非连续").to_vec()
    }

    /// 两段音频的说话人嵌入余弦相似度（同人典型 >0.5，异人典型 <0.4）
    fn similarity(&mut self, a: &AudioData, b: &AudioData) -> f32 {
        let (ea, eb) = (self.embed(a), self.embed(b));
        assert_eq!(ea.len(), eb.len(), "嵌入维度不一致");
        let dot: f32 = ea.iter().zip(&eb).map(|(x, y)| x * y).sum();
        let na: f32 = ea.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = eb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(na > 0.0 && nb > 0.0, "零范数嵌入");
        dot / (na * nb)
    }
}

// ===================== 自然度（客观信号指标） =====================

/// 客观信号指标断言：音频非空非静音、无削波、无直流偏置、无样本级爆音、首尾无超长静音
fn verify_naturalness(tag: &str, name: &str, audio: &AudioData) {
    let n = audio.samples.len();
    assert!(n > 0, "[{}] {} 合成结果为空", tag, name);
    let dur = n as f64 / audio.sample_rate as f64;
    let rms = (audio.samples.iter().map(|s| (*s as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
    let peak = audio
        .samples
        .iter()
        .fold(0.0f32, |m, s| m.max(s.abs())) as f64;
    let dc = audio.samples.iter().map(|s| *s as f64).sum::<f64>() / n as f64;
    let clip_ratio =
        audio.samples.iter().filter(|s| s.abs() >= 0.995).count() as f64 / n as f64;
    let voiced_ratio =
        audio.samples.iter().filter(|s| s.abs() > 0.003).count() as f64 / n as f64;
    let head_silence = audio
        .samples
        .iter()
        .position(|s| s.abs() > 0.003)
        .map(|i| i as f64 / n as f64)
        .unwrap_or(1.0);
    let tail_silence = audio
        .samples
        .iter()
        .rev()
        .position(|s| s.abs() > 0.003)
        .map(|i| i as f64 / n as f64)
        .unwrap_or(1.0);
    let max_step = audio
        .samples
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0f32, f32::max) as f64;

    eprintln!(
        "[{}] {} 自然度: 时长 {:.2}s / {} Hz | RMS {:.4} | 峰值 {:.3} | 直流 {:.5} | 削波率 {:.3}% | 语音占比 {:.1}% | 首静音 {:.1}% | 尾静音 {:.1}% | 最大样本跳变 {:.3}",
        tag, name, dur, audio.sample_rate, rms, peak, dc,
        clip_ratio * 100.0, voiced_ratio * 100.0,
        head_silence * 100.0, tail_silence * 100.0, max_step
    );

    assert!(dur >= 0.4, "[{}] {} 音频过短: {:.2}s", tag, name, dur);
    assert!(
        (0.005..=0.7).contains(&rms),
        "[{}] {} RMS 超界: {:.4}（过低=近静音，过高=近削波）",
        tag, name, rms
    );
    assert!(peak > 0.01, "[{}] {} 峰值过低（近静音）: {:.4}", tag, name, peak);
    assert!(
        clip_ratio < 0.01,
        "[{}] {} 削波率过高: {:.3}%（>1% 样本触顶，存在破音）",
        tag, name, clip_ratio * 100.0
    );
    assert!(dc.abs() < 0.03, "[{}] {} 直流偏置过大: {:.5}", tag, name, dc);
    assert!(
        voiced_ratio >= 0.05,
        "[{}] {} 语音占比过低: {:.1}%（主体为静音）",
        tag, name, voiced_ratio * 100.0
    );
    assert!(
        head_silence <= 0.5 && tail_silence <= 0.5,
        "[{}] {} 首/尾静音过长: {:.1}% / {:.1}%（存在截断或空洞风险）",
        tag, name, head_silence * 100.0, tail_silence * 100.0
    );
    assert!(
        max_step < 1.2,
        "[{}] {} 样本间跳变过大: {:.3}（爆音 / 波形不连续）",
        tag, name, max_step
    );
}

// ===================== ASR 闭环（可懂度代理） =====================

/// SenseVoice ASR 闭环：识别结果须与合成台词有基本重合；模型缺失时跳过
fn asr_loop_verify(tag: &str, text: &str, audio: &AudioData) {
    let dir = workspace_root().join("models/asr/sensevoice");
    if !dir.is_dir() {
        eprintln!("[{}] ⚠ SenseVoice 模型不存在，跳过 ASR 闭环: {:?}", tag, dir);
        return;
    }
    let asr = votex_infra::asr::sensevoice::SenseVoiceProvider::new();
    let model = Model::new(
        ModelId::new("sensevoice"),
        &dir.to_string_lossy(),
        ModelKind::Asr,
        EngineKind::SenseVoice,
    );
    asr.load(&model).expect("SenseVoice 加载失败");
    let result = asr.recognize(audio, &make_asr_params()).expect("SenseVoice 识别失败");
    asr.unload().ok();
    let recognized = result.text.trim();

    let orig: Vec<char> = text.chars().filter(|c| !is_punct(*c)).collect();
    let recog: Vec<char> = recognized.chars().filter(|c| !is_punct(*c)).collect();
    let set: HashSet<char> = recog.iter().copied().collect();
    let covered = orig.iter().filter(|c| set.contains(c)).count();
    let coverage = if orig.is_empty() {
        0.0
    } else {
        covered as f64 / orig.len() as f64 * 100.0
    };

    eprintln!("[{}] ASR 闭环: 原文 {} 字 → 识别 {} 字，覆盖率 {:.1}%", tag, orig.len(), recog.len(), coverage);
    eprintln!("  原文: {}", text);
    eprintln!("  识别: {}", recognized);

    assert!(!recognized.is_empty(), "[{}] ASR 识别结果为空（音频不可懂）", tag);
    assert!(
        coverage > 30.0,
        "[{}] ASR 闭环内容覆盖率过低: {:.1}%（自然度/可懂度不达标）",
        tag, coverage
    );
}

// ===================== 多角色素材与音色挑选 =====================

/// 取两个有真实台词（去括注后 ≥8 字，太短则 campplus 帧数不足）的角色
fn pick_role_lines(cands: &[RoleCandidate]) -> [(String, String); 2] {
    let mut picked: Vec<(String, String)> = Vec::new();
    for c in cands {
        if let Some(line) = &c.sample_line {
            let clean = sanitize_line(line);
            if clean.chars().count() >= 8 {
                picked.push((c.name.clone(), clean));
            }
        }
        if picked.len() == 2 {
            break;
        }
    }
    if picked.len() == 2 {
        let r2 = picked.pop().unwrap();
        let r1 = picked.pop().unwrap();
        [r1, r2]
    } else {
        eprintln!(
            "⚠ 素材中可试听台词不足 2 段（提取到 {} 段），回退到第一章固定对白",
            picked.len()
        );
        let [(n1, l1), (n2, l2)] = FALLBACK_ROLES;
        [(n1.to_string(), l1.to_string()), (n2.to_string(), l2.to_string())]
    }
}

/// 为角色挑音色：依次按 `prefer` 性别顺序，跳过 `used`（保证两角色音色不同）
fn pick_voice_for_role(
    pool: &[VoiceMeta],
    prefer: &[VoiceGender],
    used: Option<&str>,
) -> Option<String> {
    for g in prefer {
        if let Some(v) = VoiceMeta::pick_for_chinese(pool, *g) {
            if used != Some(v.voice_id.as_str()) {
                return Some(v.voice_id.clone());
            }
        }
    }
    pool.iter()
        .map(|v| v.voice_id.as_str())
        .find(|id| used != Some(*id))
        .map(str::to_string)
}

// ===================== 共用合成与断言 =====================

fn synth(
    tts: &dyn TtsProvider,
    engine: EngineKind,
    text: &str,
    voice: &VoiceId,
    tag: &str,
    what: &str,
) -> AudioData {
    let params = make_tts_params(engine, voice);
    let audio = tts
        .synthesize(text, voice, &params)
        .unwrap_or_else(|e| panic!("[{}] {} 合成失败（voice={}）: {}", tag, what, voice.id, e));
    assert!(!audio.samples.is_empty(), "[{}] {} 合成结果为空", tag, what);
    eprintln!(
        "[{}] {} 合成完成: {} 采样点, {} Hz, {:.1}s",
        tag,
        what,
        audio.samples.len(),
        audio.sample_rate,
        audio.samples.len() as f64 / audio.sample_rate as f64
    );
    audio
}

fn save_wav(tag: &str, file_stem: &str, audio: &AudioData) {
    let dir = workspace_root().join("target/test_output/voice_role");
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("创建输出目录失败: {}", e));
    let path = dir.join(format!("{}_{}.wav", tag, file_stem));
    votex_infra::audio::wav::WavWriter::write(audio, &path)
        .unwrap_or_else(|e| panic!("WAV 写入失败 {:?}: {}", path, e));
    eprintln!("[{}] 试听产物: {}", tag, path.display());
}

/// 音色可控 + 多角色可区分（campplus 说话人相似度）
///
/// - `m1` 角色1台词 × 音色1、`b1` 同台词 × 音色2（跨音色同文本）、
///   `m2` 角色2台词 × 音色2（同音色异文本基线）
fn verify_voice_and_roles(tag: &str, sim: &mut SpeakerSim, m1: &AudioData, b1: &AudioData, m2: &AudioData) {
    let same = sim.similarity(b1, m2); // 同音色（v2）异文本
    let cross = sim.similarity(m1, b1); // 同文本跨音色
    let cross_role = sim.similarity(m1, m2); // 跨角色（异音色异文本）
    eprintln!(
        "[{}] campplus 相似度: 同音色异文本={:.3} | 跨音色同文本={:.3} | 跨角色={:.3}",
        tag, same, cross, cross_role
    );
    assert!(
        same > cross + 0.05,
        "[{}] 音色不可控：同音色相似度 {:.3} 未比跨音色 {:.3} 高出 0.05（换音色没有真实生效）",
        tag, same, cross
    );
    assert!(
        cross_role < same - 0.05,
        "[{}] 多角色音色未区分：跨角色相似度 {:.3} 与同音色基线 {:.3} 差距不足 0.05",
        tag, cross_role, same
    );
}

/// 多音色引擎共用流程：角色提取 → 自动分配完整性 → 显式双角色 → 三段合成 →
/// 音色/多角色/自然度断言 → ASR 闭环
fn run_role_audio_case(
    tag: &str,
    engine: EngineKind,
    tts: &dyn TtsProvider,
    pool: &[VoiceMeta],
    text: &str,
) {
    assert!(
        pool.len() >= 2,
        "[{}] 音色池应至少 2 个音色才能验证多角色，实际 {}",
        tag,
        pool.len()
    );

    // ── 1. 角色提取 + 自动音色分配（domain 全链路，音频级验证的前置） ──
    let cands = extract_role_candidates(text, 10);
    assert!(!cands.is_empty(), "[{}] 素材应能提取出角色候选（多角色链路第一步）", tag);
    eprintln!(
        "[{}] 角色候选 {} 个: {:?}",
        tag,
        cands.len(),
        cands.iter().map(|c| (&c.name, c.gender)).collect::<Vec<_>>()
    );

    let auto = build_role_map(&cands, pool, &RoleAssignments::default());
    assert!(auto.narrator.is_some(), "[{}] 自动分配应产生旁白音色", tag);
    assert!(auto.dialogue_default.is_some(), "[{}] 自动分配应产生台词兜底音色", tag);
    for c in &cands {
        assert!(
            auto.roles.contains_key(&c.name),
            "[{}] 角色 {} 未分配到音色（路由丢失）",
            tag,
            c.name
        );
    }
    // 历史回归：中文音色池的旁白不得落到英文音色（af_maple 字典序排在 zf_ 前）
    if pool.iter().any(|v| v.locale == VoiceLocale::Chinese) {
        let narrator = auto.narrator.as_deref().unwrap_or_default();
        assert!(
            narrator.starts_with("zf_") || narrator.starts_with("zm_") || narrator.starts_with("xf_"),
            "[{}] 中文音色池的旁白须为中文音色，实际 {}",
            tag,
            narrator
        );
    }
    eprintln!(
        "[{}] 自动分配: narrator={:?} dialogue_default={:?} roles={} 条",
        tag,
        auto.narrator,
        auto.dialogue_default,
        auto.roles.len()
    );

    // ── 2. 角色台词素材 ──
    let [(r1_name, r1_line), (r2_name, r2_line)] = pick_role_lines(&cands);
    eprintln!("[{}] 角色1 {} → \"{}\"", tag, r1_name, r1_line);
    eprintln!("[{}] 角色2 {} → \"{}\"", tag, r2_name, r2_line);

    // ── 3. 显式指定两角色音色（模拟用户在 GUI 确认角色表，最高优先级路径） ──
    let g1 = cands
        .iter()
        .find(|c| c.name == r1_name)
        .map(|c| c.gender)
        .unwrap_or(VoiceGender::Female);
    let g2 = cands
        .iter()
        .find(|c| c.name == r2_name)
        .map(|c| c.gender)
        .unwrap_or(VoiceGender::Male);
    let v1 = pick_voice_for_role(pool, &[g1, VoiceGender::Female, VoiceGender::Male], None)
        .expect("音色池为空");
    let v2 = pick_voice_for_role(
        pool,
        &[g2, VoiceGender::Male, VoiceGender::Female],
        Some(&v1),
    )
    .expect("音色池为空");
    assert_ne!(v1, v2, "[{}] 两角色音色必须不同", tag);

    let mut asg = RoleAssignments::default();
    asg.roles.insert(r1_name.clone(), v1.clone());
    asg.roles.insert(r2_name.clone(), v2.clone());
    let map = build_role_map(&cands, pool, &asg);
    assert_eq!(
        map.roles.get(&r1_name).map(String::as_str),
        Some(v1.as_str()),
        "[{}] 显式指定的角色1音色未生效",
        tag
    );
    assert_eq!(
        map.roles.get(&r2_name).map(String::as_str),
        Some(v2.as_str()),
        "[{}] 显式指定的角色2音色未生效",
        tag
    );

    // ── 4. 三段合成：m1=角色1台词×v1 / b1=同台词×v2（跨音色） / m2=角色2台词×v2（同音色基线） ──
    let id1 = VoiceId::new(&v1, &v1, engine);
    let id2 = VoiceId::new(&v2, &v2, engine);
    let m1 = synth(tts, engine, &r1_line, &id1, tag, "角色1台词×音色1");
    let b1 = synth(tts, engine, &r1_line, &id2, tag, "同台词×音色2");
    let m2 = synth(tts, engine, &r2_line, &id2, tag, "角色2台词×音色2");

    for (stem, audio) in [("role1_v1", &m1), ("role1_v2", &b1), ("role2_v2", &m2)] {
        verify_naturalness(tag, stem, audio);
        save_wav(tag, stem, audio);
    }

    // ── 5. 音色可控 + 多角色可区分 ──
    let mut sim = SpeakerSim::new();
    verify_voice_and_roles(tag, &mut sim, &m1, &b1, &m2);

    // ── 6. ASR 闭环（可懂度代理，以角色1台词为真值） ──
    asr_loop_verify(tag, &r1_line, &m1);
}

/// CosyVoice 专用流程：单音色（prompt 克隆）引擎 —— 断言多角色退化行为确定 +
/// 克隆音色忠实度 + 自然度 + ASR 闭环
fn run_cosyvoice_case(tag: &str, tts: &dyn TtsProvider, text: &str) {
    let voices = tts.list_voices();
    assert_eq!(
        voices.len(),
        1,
        "[{}] CosyVoice 当前为 prompt 克隆单音色引擎，音色列表应恰为 1，实际 {:?}",
        tag,
        voices.iter().map(|v| &v.id).collect::<Vec<_>>()
    );
    let default_id = voices[0].id.clone();
    let default_name = voices[0].display_name.clone();

    // ── 1. 单音色池下的多角色退化：所有角色必须统一落到唯一音色，且不丢路由 ──
    let cands = extract_role_candidates(text, 10);
    assert!(!cands.is_empty(), "[{}] 素材应能提取出角色候选", tag);
    let pool: Vec<VoiceMeta> = voices
        .into_iter()
        .map(|v| VoiceMeta::from_voice_id(v, VoiceGender::Neutral))
        .collect();
    let map = build_role_map(&cands, &pool, &RoleAssignments::default());
    assert_eq!(
        map.narrator.as_deref(),
        Some(default_id.as_str()),
        "[{}] 单音色池旁白应为唯一音色",
        tag
    );
    assert_eq!(
        map.dialogue_default.as_deref(),
        Some(default_id.as_str()),
        "[{}] 单音色池台词兜底应为唯一音色",
        tag
    );
    for c in &cands {
        assert_eq!(
            map.roles.get(&c.name).map(String::as_str),
            Some(default_id.as_str()),
            "[{}] 单音色池下角色 {} 应统一落到唯一音色（退化行为须确定，不丢路由）",
            tag,
            c.name
        );
    }
    eprintln!(
        "[{}] 单音色池多角色退化: {} 个角色全部映射到 {}",
        tag,
        cands.len(),
        default_id.as_str()
    );

    // ── 2. 两角色台词合成（同一克隆音色） ──
    let [(r1_name, r1_line), (r2_name, r2_line)] = pick_role_lines(&cands);
    eprintln!("[{}] 角色1 {} → \"{}\"", tag, r1_name, r1_line);
    eprintln!("[{}] 角色2 {} → \"{}\"", tag, r2_name, r2_line);
    let id = VoiceId::new(
        default_id.as_str(),
        default_name.as_str(),
        EngineKind::CosyVoice3,
    );
    let m1 = synth(tts, EngineKind::CosyVoice3, &r1_line, &id, tag, "角色1台词");
    let m2 = synth(tts, EngineKind::CosyVoice3, &r2_line, &id, tag, "角色2台词");
    verify_naturalness(tag, "role1", &m1);
    verify_naturalness(tag, "role2", &m2);
    save_wav(tag, "role1", &m1);
    save_wav(tag, "role2", &m2);

    // ── 3. 克隆音色忠实度：输出 vs 实际使用的 prompt 参考 vs 另一说话人参考 ──
    let base = workspace_root().join("models/tts/cosyvoice/prompts");
    let zh_wav = base.join("zh_prompt.wav");
    let zh_txt = base.join("zh_prompt.txt");
    let nova = base.join("en_female_nova_greeting.wav");
    // 与 synthesize 内的选择逻辑一致：zh_prompt 双件齐全时用中文 prompt，否则英文
    let (used, other) = if zh_wav.is_file() && zh_txt.is_file() {
        (Some(zh_wav), nova.is_file().then_some(nova))
    } else if nova.is_file() {
        (Some(nova), zh_wav.is_file().then_some(zh_wav))
    } else {
        (None, None)
    };

    let mut sim = SpeakerSim::new();
    match (used, other) {
        (Some(u), Some(o)) => {
            let used_audio = votex_infra::audio::wav::read_wav(&u)
                .unwrap_or_else(|e| panic!("读取 prompt 参考 {:?} 失败: {}", u, e));
            let other_audio = votex_infra::audio::wav::read_wav(&o)
                .unwrap_or_else(|e| panic!("读取对照参考 {:?} 失败: {}", o, e));
            let same = sim.similarity(&m1, &m2); // 同一克隆音色两次合成
            let clone_f = sim.similarity(&m1, &used_audio); // 输出 vs 克隆源
            let cross = sim.similarity(&m1, &other_audio); // 输出 vs 另一说话人
            eprintln!(
                "[{}] 克隆音色相似度: 同音色两次合成={:.3} | vs 克隆源={:.3} | vs 另一说话人={:.3}",
                tag, same, clone_f, cross
            );
            assert!(
                clone_f > cross + 0.05,
                "[{}] 克隆音色不忠实：与克隆源相似度 {:.3} 未高于与另一说话人 {:.3} 达 0.05",
                tag, clone_f, cross
            );
            assert!(
                same > cross + 0.05,
                "[{}] 同一克隆音色两次合成相似度 {:.3} 未高于跨说话人 {:.3} 达 0.05",
                tag, same, cross
            );
        }
        _ => {
            eprintln!("[{}] ⚠ 缺少 prompt 参考音频，跳过克隆保真断言（仍执行自然度与 ASR 闭环）", tag);
        }
    }

    // ── 4. ASR 闭环 ──
    asr_loop_verify(tag, &r1_line, &m1);
}

// ===================== 四引擎 e2e =====================

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test tts_e2e_voice_role_test --features slow-models")]
#[test]
fn kokoro_音色多角色与自然度_e2e() {
    switch_to_workspace_root();
    let dir = workspace_root().join("models/tts/kokoro-82m-v1.1-zh");
    if !dir.is_dir() {
        eprintln!("⚠ Kokoro 中文模型目录不存在: {:?}，跳过", dir);
        return;
    }
    let text = read_chapter();

    let tts = votex_infra::tts::kokoro::KokoroProvider::new();
    let model = Model::new(
        ModelId::new("kokoro-82m-v1.1-zh"),
        &dir.to_string_lossy(),
        ModelKind::Tts,
        EngineKind::Kokoro,
    );
    tts.load(&model)
        .unwrap_or_else(|e| panic!("Kokoro 模型加载失败: {}（目录 {:?}）", e, dir));

    let voices = tts.list_voices();
    assert!(voices.len() >= 20, "[kokoro] 中文音色池应 ≥20，实际 {}", voices.len());
    let pool: Vec<VoiceMeta> = voices
        .into_iter()
        .map(|v| {
            let gender = VoiceGender::from_voice_id(&v.id);
            VoiceMeta::from_voice_id(v, gender)
        })
        .collect();
    eprintln!(
        "[kokoro] 音色池 {} 个（女 {} / 男 {} / 童 {} / 中性 {}）",
        pool.len(),
        VoiceMeta::filter_by_gender(&pool, VoiceGender::Female).len(),
        VoiceMeta::filter_by_gender(&pool, VoiceGender::Male).len(),
        VoiceMeta::filter_by_gender(&pool, VoiceGender::Child).len(),
        VoiceMeta::filter_by_gender(&pool, VoiceGender::Neutral).len()
    );

    run_role_audio_case("kokoro", EngineKind::Kokoro, &tts, &pool, &text);
    tts.unload().ok();
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test tts_e2e_voice_role_test --features slow-models")]
#[test]
fn qwen3_音色多角色与自然度_e2e() {
    switch_to_workspace_root();
    // 与既有 e2e 同口径：优先 0.6B（CPU 推理更快），其次 1.7B
    let dir = ["qwen3-tts-0.6b", "qwen3-tts-1.7b", "qwen3-tts"]
        .iter()
        .map(|d| workspace_root().join("models/tts").join(d))
        .find(|d| d.is_dir());
    let Some(dir) = dir else {
        eprintln!("⚠ Qwen3-TTS 模型目录不存在，跳过");
        return;
    };
    let text = read_chapter();
    let id = dir.file_name().unwrap().to_string_lossy().to_string();

    let tts = votex_infra::tts::qwen3_tts::Qwen3TtsProvider::new();
    let model = Model::new(
        ModelId::new(&id),
        &dir.to_string_lossy(),
        ModelKind::Tts,
        EngineKind::Qwen3Tts,
    );
    tts.load(&model)
        .unwrap_or_else(|e| panic!("Qwen3-TTS 模型加载失败: {}（目录 {:?}）", e, dir));

    let voices = tts.list_voices();
    assert!(!voices.is_empty(), "[qwen3] 加载后应列出音色（变体状态未生效？）");
    let pool: Vec<VoiceMeta> = voices
        .into_iter()
        .map(|v| {
            let gender = gender_by_display(&v.display_name);
            VoiceMeta::from_voice_id(v, gender)
        })
        .collect();
    eprintln!(
        "[qwen3] 音色池（{}）: {:?}",
        id,
        pool.iter().map(|v| (&v.voice_id, v.gender)).collect::<Vec<_>>()
    );

    run_role_audio_case("qwen3", EngineKind::Qwen3Tts, &tts, &pool, &text);
    tts.unload().ok();
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test tts_e2e_voice_role_test --features slow-models")]
#[test]
fn indextts25_音色多角色与自然度_e2e() {
    switch_to_workspace_root();
    let dir = workspace_root().join("models/tts/indextts25");
    if !dir.is_dir() {
        eprintln!("⚠ IndexTTS-2.5 模型目录不存在: {:?}，跳过", dir);
        return;
    }
    let text = read_chapter();

    let tts = votex_infra::tts::indextts25::IndexTts25Provider::new();
    let model = Model::new(
        ModelId::new("indextts25"),
        &dir.to_string_lossy(),
        ModelKind::Tts,
        EngineKind::IndexTTS25,
    );
    tts.load(&model)
        .unwrap_or_else(|e| panic!("IndexTTS-2.5 模型加载失败: {}（目录 {:?}）", e, dir));

    let voices = tts.list_voices();
    let ids: Vec<String> = voices.iter().map(|v| v.id.clone()).collect();
    assert!(
        voices.len() >= 2,
        "[indextts25] 至少需 2 个参考音色（prompts/*.wav）才能验证多角色，实际 {:?}",
        ids
    );
    let pool: Vec<VoiceMeta> = voices
        .into_iter()
        .map(|v| VoiceMeta::from_voice_id(v, VoiceGender::Neutral))
        .collect();
    eprintln!("[indextts25] 参考音色池: {:?}", ids);

    run_role_audio_case("indextts25", EngineKind::IndexTTS25, &tts, &pool, &text);
    tts.unload().ok();
}

#[cfg_attr(not(feature = "slow-models"), ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --test tts_e2e_voice_role_test --features slow-models")]
#[test]
fn cosyvoice3_音色多角色与自然度_e2e() {
    switch_to_workspace_root();
    let dir = workspace_root().join("models/tts/cosyvoice");
    if !dir.is_dir() {
        eprintln!("⚠ CosyVoice 模型目录不存在: {:?}，跳过", dir);
        return;
    }
    let text = read_chapter();

    let tts = votex_infra::tts::cosyvoice::CosyVoiceProvider::new();
    let model = Model::new(
        ModelId::new("cosyvoice"),
        &dir.to_string_lossy(),
        ModelKind::Tts,
        EngineKind::CosyVoice3,
    );
    tts.load(&model)
        .unwrap_or_else(|e| panic!("CosyVoice 模型加载失败: {}（目录 {:?}）", e, dir));

    run_cosyvoice_case("cosyvoice3", &tts, &text);
    tts.unload().ok();
}
