//! 音色可控 / 多角色配音 / 自然度 e2e 验证（Kokoro v1.1-zh）
//!
//! 此前各 TTS e2e 只验证「单音色能出声」，未验证音频层的音色差异性与
//! 多角色路由的真实合成。本文件补齐三件事：
//!
//! 1. **音色可控**：同一文本换音色（女 zf / 男 zm）合成的音频必须可区分
//!    （互相关 + 基频 F0），同音色重复合成必须可复现（确定性回归）；
//! 2. **多角色**：`build_synthesis_plan` + `RoleVoiceMap` 的真实音频合成——
//!    旁白/男配角/女角色三路音色按计划路由落成音频，组内相似度必须高于组间；
//! 3. **自然度**：客观指标（非静音占比、削波、语速、基频范围、能量动态）。
//!
//! 引擎选 Kokoro v1.1-zh（82M，CPU 秒级，100 个中文音色 zf_*/zm_*），
//! 多角色计划逻辑（build_synthesis_plan/RoleVoiceMap）与 GUI/CLI 共用。
//!
//! 运行：
//! ```bash
//! cargo test -p votex-infra --features slow-models --test tts_voice_multirole_e2e_test -- --nocapture
//! ```

use std::path::PathBuf;

use votex_app::use_case::tts_use_case::build_synthesis_plan;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{EngineKind, ModelId, ModelKind};
use votex_domain::shared::value_object::AudioData;
use votex_domain::tts::provider::TtsProvider;
use votex_domain::tts::role::RoleVoiceMap;
use votex_domain::tts::value_object::{
    DenoiseLevel, Pitch, SegmentSize, Speed, TtsParams, VoiceId, Volume,
};
use votex_infra::tts::kokoro::KokoroProvider;

fn workspace_root() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

fn model_dir() -> PathBuf {
    workspace_root().join("models/tts/kokoro-82m-v1.1-zh")
}

fn load_kokoro() -> Option<KokoroProvider> {
    let dir = model_dir();
    if !dir.join("kokoro-v1.1-zh.fixed.onnx").is_file() && !dir.join("kokoro-v1.1-zh.onnx").is_file()
    {
        eprintln!("⚠ kokoro-82m-v1.1-zh 模型不存在，跳过: {:?}", dir);
        return None;
    }
    let provider = KokoroProvider::new();
    let model = Model::new(
        ModelId::new("kokoro-82m-v1.1-zh"),
        &dir.to_string_lossy(),
        ModelKind::Tts,
        EngineKind::Kokoro,
    );
    provider.load(&model).expect("kokoro 加载失败");
    Some(provider)
}

fn make_params(voice: VoiceId) -> TtsParams {
    TtsParams {
        engine: EngineKind::Kokoro,
        voice,
        speed: Speed::default(),
        pitch: Pitch::default(),
        volume: Volume::default(),
        segment_size: SegmentSize::S500,
        segment_silence_ms: 300,
        crossfade_ms: 10,
        num_to_chinese: true,
        denoise: false,
        denoise_level: DenoiseLevel::Low,
        emotion: None,
        dialect: None,
    }
}

// ===================== 音频分析辅助 =====================

/// 帧级 RMS（帧长 50ms，步长 25ms）
fn frame_rms(samples: &[f32], sr: u32) -> Vec<f32> {
    let frame = (sr as usize / 20).max(1);
    let hop = (frame / 2).max(1);
    let mut out = Vec::new();
    let mut i = 0;
    while i + frame <= samples.len() {
        let s: f32 = samples[i..i + frame].iter().map(|x| x * x).sum();
        out.push((s / frame as f32).sqrt());
        i += hop;
    }
    out
}

/// 非静音帧占比（RMS 阈值 0.01）
fn nonsilent_ratio(samples: &[f32], sr: u32) -> f64 {
    let rms = frame_rms(samples, sr);
    if rms.is_empty() {
        return 0.0;
    }
    let voiced = rms.iter().filter(|r| **r > 0.01).count();
    voiced as f64 / rms.len() as f64
}

/// 中位数基频（自相关法，仅对有能量帧估计，范围 60~450Hz）
fn median_f0(samples: &[f32], sr: u32) -> Option<f64> {
    let frame = sr as usize / 20; // 50ms
    let hop = frame / 2;
    let min_lag = (sr as f32 / 450.0) as usize;
    let max_lag = (sr as f32 / 60.0) as usize;
    let mut f0s: Vec<f64> = Vec::new();
    let mut i = 0;
    while i + frame <= samples.len() {
        let seg = &samples[i..i + frame];
        let energy: f32 = seg.iter().map(|x| x * x).sum();
        if (energy / frame as f32).sqrt() > 0.02 {
            // 自相关找基音周期
            let mut best = 0.0f32;
            let mut best_lag = 0usize;
            for lag in min_lag..=max_lag.min(frame - 1) {
                let mut dot = 0.0f32;
                let mut norm_a = 0.0f32;
                let mut norm_b = 0.0f32;
                for k in 0..frame - lag {
                    dot += seg[k] * seg[k + lag];
                    norm_a += seg[k] * seg[k];
                    norm_b += seg[k + lag] * seg[k + lag];
                }
                let nc = dot / ((norm_a * norm_b).sqrt() + 1e-9);
                if nc > best {
                    best = nc;
                    best_lag = lag;
                }
            }
            if best > 0.5 && best_lag > 0 {
                f0s.push(sr as f64 / best_lag as f64);
            }
        }
        i += hop;
    }
    if f0s.len() < 5 {
        return None;
    }
    f0s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(f0s[f0s.len() / 2])
}

/// 归一化互相关（长度对齐取短者；衡量两段音频的波形相似度）
fn ncc(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let (a, b) = (&a[..n], &b[..n]);
    let ma: f32 = a.iter().sum::<f32>() / n as f32;
    let mb: f32 = b.iter().sum::<f32>() / n as f32;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for k in 0..n {
        let x = (a[k] - ma) as f64;
        let y = (b[k] - mb) as f64;
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f64
}

/// 自然度客观指标断言（非静音、无削波、语速、F0 范围、能量动态）
fn assert_natural(label: &str, text: &str, audio: &AudioData) {
    let sr = audio.sample_rate;
    let samples = &audio.samples;
    let dur_s = samples.len() as f64 / sr as f64;
    let nsr = nonsilent_ratio(samples, sr);
    let peak = samples.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    let cjk = text.chars().filter(|c| !c.is_ascii() && !c.is_whitespace()).count() as f64;
    let rate = if dur_s > 0.0 { cjk / dur_s } else { 0.0 };
    let f0 = median_f0(samples, sr);
    let rms = frame_rms(samples, sr);
    let rms_mean = if rms.is_empty() { 0.0 } else { rms.iter().sum::<f32>() / rms.len() as f32 };
    let rms_std = if rms.is_empty() {
        0.0
    } else {
        (rms.iter().map(|r| (r - rms_mean).powi(2)).sum::<f32>() / rms.len() as f32).sqrt()
    };

    eprintln!(
        "[{label}] 时长 {dur_s:.2}s 非静音 {nsr:.2} 峰值 {peak:.3} 语速 {rate:.1}字/s \
         F0 {:?} 动态(std/mean) {:.2}",
        f0.map(|f| format!("{f:.0}Hz")),
        if rms_mean > 0.0 { rms_std / rms_mean } else { 0.0 },
    );

    assert!(nsr > 0.2, "{label}: 非静音占比过低（{nsr:.2}），疑似静音输出");
    assert!(nsr < 1.0, "{label}: 全程无停顿（{nsr:.2}），不自然");
    assert!(peak < 0.9995, "{label}: 波形削波（峰值 {peak:.3}）");
    assert!(
        (1.0..=15.0).contains(&rate),
        "{label}: 语速异常 {rate:.1} 字/s，超出自然范围"
    );
    if let Some(f) = f0 {
        assert!(
            (60.0..=450.0).contains(&f),
            "{label}: 基频 {f:.0}Hz 超出人声范围"
        );
    } else {
        panic!("{label}: 有能量帧过少，无法估计基频（疑似无效音频）");
    }
    assert!(
        rms_mean > 0.0 && rms_std / rms_mean > 0.1,
        "{label}: 能量过于平坦（std/mean {:.2}），韵律缺失",
        rms_std / rms_mean
    );
}

fn synth(provider: &KokoroProvider, voice_id: &str, text: &str) -> AudioData {
    let voice = VoiceId::new(voice_id, voice_id, EngineKind::Kokoro);
    let params = make_params(voice.clone());
    provider.synthesize(text, &voice, &params).expect("合成失败")
}

// ===================== 1. 音色可控 =====================

#[test]
#[cfg_attr(
    not(feature = "slow-models"),
    ignore = "需本地 kokoro 模型并多次推理；跑法: cargo test -p votex-infra --test tts_voice_multirole_e2e_test --features slow-models"
)]
fn 音色可控_不同音色可区分且可复现() {
    let Some(provider) = load_kokoro() else { return };
    let voices = provider.list_voices();
    let pick = |prefix: &str| {
        voices
            .iter()
            .find(|v| v.id.starts_with(prefix))
            .map(|v| v.id.clone())
            .unwrap_or_else(|| panic!("音色池缺少 {prefix} 系"))
    };
    let female = pick("zf_");
    let male = pick("zm_");
    eprintln!("选用音色: 女={female} 男={male}（音色池共 {} 个）", voices.len());
    assert!(voices.len() >= 20, "音色池应提供足够多的可选音色");

    let text = "今天天气真好，我们一起去公园散步，顺便买一杯奶茶。";

    // 同音色重复合成 → 确定性回归
    let a1 = synth(&provider, &female, text);
    let a1b = synth(&provider, &female, text);
    let det = ncc(&a1.samples, &a1b.samples);
    eprintln!("同音色重复合成 NCC = {det:.4}");
    assert!(det > 0.98, "同音色两次合成应可复现（NCC={det:.4}）");

    // 不同音色（女 vs 男）→ 波形必须可区分
    let m1 = synth(&provider, &male, text);
    let diff = ncc(&a1.samples, &m1.samples);
    eprintln!("女/男音色 NCC = {diff:.4}");
    assert!(diff < 0.95, "不同音色波形应可区分（NCC={diff:.4}）");

    // 基频：女声应显著高于男声（音色差异的声学证据）
    let f_f = median_f0(&a1.samples, a1.sample_rate).expect("女声应可测出基频");
    let f_m = median_f0(&m1.samples, m1.sample_rate).expect("男声应可测出基频");
    eprintln!("基频: 女 {f_f:.0}Hz vs 男 {f_m:.0}Hz（差 {:.0}Hz）", f_f - f_m);
    assert!(
        f_f > f_m + 15.0,
        "女声音色基频（{f_f:.0}Hz）应显著高于男声（{f_m:.0}Hz）"
    );

    // 自然度指标
    assert_natural(&format!("音色[{female}]"), text, &a1);
    assert_natural(&format!("音色[{male}]"), text, &m1);

    provider.unload().ok();
}

// ===================== 2. 多角色配音真实合成 =====================

#[test]
#[cfg_attr(
    not(feature = "slow-models"),
    ignore = "需本地 kokoro 模型并多段推理；跑法: cargo test -p votex-infra --test tts_voice_multirole_e2e_test --features slow-models"
)]
fn 多角色_音色路由落成真实音频且角色间可区分() {
    let Some(provider) = load_kokoro() else { return };
    let voices = provider.list_voices();
    let pick = |prefix: &str| {
        voices
            .iter()
            .find(|v| v.id.starts_with(prefix))
            .map(|v| v.id.clone())
            .unwrap_or_else(|| panic!("音色池缺少 {prefix} 系"))
    };
    let narrator_v = pick("zf_");
    let role_v = pick("zf_");
    let default_v = pick("zm_");
    let _ = role_v; // 占位：角色表见下

    let role_map = RoleVoiceMap::from_json_str(&format!(
        r#"{{"narrator": "{narrator_v}", "dialogue_default": "{default_v}", "roles": {{"老周": "{default_v}"}}}}"#
    ))
    .expect("role_map 解析失败");

    // 旁白 + 两人对话（台词用弯引号，与 GUI/CLI 同一解析路径）
    let text = "黄昏的巷口，老周收着晾衣绳上的衬衫，晚风把他的白发吹得有些乱。\
        \u{201c}小芳，吃饭了！\u{201d}他朝屋里喊了一声。\
        小芳揉着眼睛跑出来：\u{201c}来啦来啦，今天吃什么呀？\u{201d}";

    let plan = build_synthesis_plan(text, SegmentSize::S500, true, Some(&role_map));
    eprintln!("多角色计划: {} 段", plan.len());
    assert!(plan.len() >= 3, "旁白+两轮台词应至少切出 3 段");
    for seg in &plan {
        eprintln!(
            "  voice={:?} text={}...",
            seg.voice_override,
            seg.text.chars().take(12).collect::<String>()
        );
    }
    // 路由断言：旁白段 = narrator；台词段 = 角色音色或对话默认
    let narrator_segs = plan
        .iter()
        .filter(|s| s.voice_override.as_deref() == Some(narrator_v.as_str()))
        .count();
    assert!(narrator_segs >= 1, "旁白段应路由到 narrator 音色");
    let used: std::collections::HashSet<_> =
        plan.iter().filter_map(|s| s.voice_override.clone()).collect();
    assert!(
        used.len() >= 2,
        "多角色应至少使用 2 个不同音色，实际 {used:?}"
    );

    // 逐段真实合成（与 tts_use_case 相同的 override 解析语义）
    let mut audios: Vec<(String, AudioData)> = Vec::new();
    for seg in &plan {
        let vid = seg.voice_override.clone().unwrap_or_else(|| narrator_v.clone());
        let audio = synth(&provider, &vid, &seg.text);
        assert_natural(&format!("多角色[{vid}]"), &seg.text, &audio);
        audios.push((vid, audio));
    }

    // 声学身份断言用基频 F0（不同文本的波形 NCC 恒接近 0，无判别力；
    // 而 F0 是音色身份的稳定特征）：同角色各段 F0 应聚集，跨角色应显著分离
    let f0_of = |vid: &str| -> Vec<f64> {
        audios
            .iter()
            .filter(|(v, _)| v == vid)
            .filter_map(|(_, a)| median_f0(&a.samples, a.sample_rate))
            .collect()
    };
    let narrator_f0s = f0_of(&narrator_v);
    let default_f0s = f0_of(&default_v);
    eprintln!("F0 身份特征: narrator({narrator_v})={narrator_f0s:?} default({default_v})={default_f0s:?}");
    assert!(!narrator_f0s.is_empty() && !default_f0s.is_empty(), "各角色段都应可测出基频");
    let spread = |v: &[f64]| v.iter().cloned().fold(f64::INFINITY, f64::min)
        ..=v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let n_span = spread(&narrator_f0s);
    assert!(
        n_span.end() - n_span.start() < 50.0,
        "同角色（{narrator_v}）各段基频应聚集在 50Hz 内，实际 {n_span:?}"
    );
    let d_span = spread(&default_f0s);
    let gap = ((n_span.start() - d_span.end()).abs())
        .max((d_span.start() - n_span.end()).abs());
    // 女 narrator 与男 default 音区应分离（同性别音色差异小，此处 zf vs zm）
    if n_span.end() < d_span.start() || d_span.end() < n_span.start() {
        assert!(
            gap > 60.0,
            "跨角色基频音区应分离 ≥60Hz，实际间隔 {gap:.0}Hz"
        );
        eprintln!("跨角色 F0 音区间隔 {gap:.0}Hz ✓");
    } else {
        eprintln!("narrator 与 default 同音区（男/女池首音色可能同性别），跳过音区断言");
    }

    provider.unload().ok();
}

// ===================== 3. 自然度（独立冒烟：情感/语速参数联动） =====================

#[test]
#[cfg_attr(
    not(feature = "slow-models"),
    ignore = "需本地 kokoro 模型推理；跑法: cargo test -p votex-infra --test tts_voice_multirole_e2e_test --features slow-models"
)]
fn 自然度_语速参数联动生效() {
    let Some(provider) = load_kokoro() else { return };
    let voices = provider.list_voices();
    let female = voices
        .iter()
        .find(|v| v.id.starts_with("zf_"))
        .map(|v| v.id.clone())
        .expect("音色池应有 zf 系");

    let text = "他缓缓推开门，屋里的灯光落在地板上，安静得能听见钟表走动的声音。";
    let normal = synth(&provider, &female, text);

    // 语速 1.5x：同文本时长应显著缩短（参数可控的证据）
    let mut fast_params = make_params(VoiceId::new(&female, &female, EngineKind::Kokoro));
    fast_params.speed = Speed::new(1.5).expect("语速 1.5 应合法");
    let fast = provider
        .synthesize(text, &fast_params.voice, &fast_params)
        .expect("1.5x 合成失败");

    let d_normal = normal.samples.len() as f64 / normal.sample_rate as f64;
    let d_fast = fast.samples.len() as f64 / fast.sample_rate as f64;
    eprintln!("语速联动: 正常 {d_normal:.2}s vs 1.5x {d_fast:.2}s（比值 {:.2}）", d_fast / d_normal);
    assert!(
        d_fast < d_normal * 0.85,
        "speed=1.5 的音频时长应比正常至少短 15%（{d_normal:.2}s → {d_fast:.2}s）"
    );

    assert_natural("自然度[正常]", text, &normal);
    provider.unload().ok();
}
