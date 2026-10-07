//! 翻译引擎 E2E 效果对比测试
//!
//! 语料：`tmp/PhilosopherStone.txt`（英文小说《哈利·波特与魔法石》开篇）。
//!
//! 流程：
//! 1. **Pass 1 英译中**：从英文小说切分 4 个句子，交给 opus-mt / nllb-200 /
//!    m2m-100 / hy-mt-1.5 各自英译中，记录译文与耗时；
//! 2. **Pass 2 中译英**：以 Pass 1 中 opus-mt 的中文译文为共享输入，
//!    交给 opus-mt / ctranslate2（`--features ct2`）/ nllb-200 / m2m-100 /
//!    hy-mt-1.5 各自中译英，记录译文、耗时与回译相似度；
//! 3. **报告落盘**：`tmp/translation_e2e_report.md`，含双向译文明细、
//!    加载/推理耗时、吞吐与回译 Dice 相似度，便于横向对比引擎效果。
//!
//! 跑法（模型就绪后）：
//! ```bash
//! cargo test -p votex-infra --features ct2,slow-models --test translation_e2e_comparison_test -- --nocapture --test-threads=1
//! ```

// 测试名使用 camelCase 描述语言对与场景，此处豁免命名检查
#![allow(non_snake_case)]

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::value_object::TranslationDirection;

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn corpus_path() -> PathBuf {
    workspace_root().join("tmp/PhilosopherStone.txt")
}

fn report_path() -> PathBuf {
    workspace_root().join("tmp/translation_e2e_report.md")
}

// ============================================================
// 语料切分
// ============================================================

/// 从英文语料切分测试句子（40~120 字符，取前 N 段）
///
/// 切分策略：按 ". " 分句（把换行折叠为空格），保留长度合适、
/// 无引号干扰的句子，句尾补回句号。碎片（如 "Mr"）经长度过滤自然剔除。
fn pick_sentences(text: &str, count: usize) -> Vec<String> {
    let flattened = text.replace(['\n', '\r'], " ");
    flattened
        .split(". ")
        .map(|s| s.trim())
        .filter(|s| (40..=120).contains(&s.len()))
        .filter(|s| !s.contains('"') && !s.contains('\''))
        .take(count)
        .map(|s| format!("{}.", s))
        .collect()
}

// ============================================================
// 量化指标：字符 bigram Dice 相似度（回译质量参考）
// ============================================================

fn normalize_for_similarity(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect()
}

fn bigrams(s: &str) -> HashSet<Vec<char>> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 2 {
        return HashSet::from([chars]);
    }
    chars.windows(2).map(|w| w.to_vec()).collect()
}

/// 两个字符串的 bigram Dice 系数（0.0 ~ 1.0，越高越相似）
fn dice_similarity(a: &str, b: &str) -> f64 {
    let (a, b) = (normalize_for_similarity(a), normalize_for_similarity(b));
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let (ga, gb) = (bigrams(&a), bigrams(&b));
    let inter = ga.intersection(&gb).count();
    2.0 * inter as f64 / (ga.len() + gb.len()) as f64
}

// ============================================================
// 记录结构与报告生成
// ============================================================

/// 单引擎单方向的翻译记录
struct PassRecord {
    engine: String,
    load_ms: u128,
    translate_ms: u128,
    input_chars: usize,
    outputs: Vec<String>,
}

impl PassRecord {
    fn throughput(&self) -> f64 {
        if self.translate_ms == 0 {
            return 0.0;
        }
        self.input_chars as f64 / (self.translate_ms as f64 / 1000.0)
    }
}

/// 单引擎双向记录（含回译相似度）
struct EngineResult {
    en_zh: Option<PassRecord>,
    zh_en: Option<PassRecord>,
    /// Pass 2 回译英文与原文的 Dice 相似度均值（0.0~1.0）
    roundtrip_score: Option<f64>,
    /// 引擎不可用原因（模型缺失 / feature 未启用）
    unavailable: Option<String>,
}

impl EngineResult {
    fn new() -> Self {
        Self { en_zh: None, zh_en: None, roundtrip_score: None, unavailable: None }
    }
}

fn fmt_dur(ms: u128) -> String {
    if ms >= 1000 {
        format!("{:.2}s", ms as f64 / 1000.0)
    } else {
        format!("{}ms", ms)
    }
}

fn write_report(
    sentences: &[String],
    zh_reference: &[String],
    results: &[(String, EngineResult)],
    ct2_enabled: bool,
) -> std::io::Result<String> {
    let mut md = String::new();
    md.push_str("# 翻译引擎 E2E 对比报告\n\n");
    md.push_str(&format!(
        "- 语料：`tmp/PhilosopherStone.txt`（英文小说开篇）\n- 测试句子：{} 段（40~120 字符）\n- 中译英输入：Pass 1 中 opus-mt 的中文译文（所有引擎共享）\n- 回译相似度：Pass 2 英文译文与英文原文的字符 bigram Dice 系数（均值）\n\n",
        sentences.len()
    ));

    // 原文与中文参照
    md.push_str("## 测试语料\n\n");
    for (i, s) in sentences.iter().enumerate() {
        md.push_str(&format!("{}. EN: {}\n   ZH(opus-mt): {}\n", i + 1, s, zh_reference[i]));
    }
    md.push('\n');

    // Pass 1 英译中
    md.push_str("## Pass 1 英译中（EN → ZH）\n\n");
    md.push_str("| 引擎 | 加载耗时 | 翻译耗时 | 吞吐 (chars/s) | 状态 |\n|---|---|---|---|---|\n");
    for (name, r) in results {
        match &r.en_zh {
            Some(p) => {
                md.push_str(&format!(
                    "| {} | {} | {} | {:.0} | ✓ |\n",
                    name,
                    fmt_dur(p.load_ms),
                    fmt_dur(p.translate_ms),
                    p.throughput()
                ));
            }
            None => {
                let why = r.unavailable.clone().unwrap_or_else(|| "未参与".into());
                md.push_str(&format!("| {} | - | - | - | {} |\n", name, why));
            }
        }
    }
    md.push_str("\n### 英译中译文明细\n\n");
    for (i, s) in sentences.iter().enumerate() {
        md.push_str(&format!("**句 {}** `{}`\n\n", i + 1, s));
        for (name, r) in results {
            if let Some(p) = &r.en_zh {
                md.push_str(&format!("- **{}**: {}\n", name, p.outputs[i]));
            }
        }
        md.push('\n');
    }

    // Pass 2 中译英
    md.push_str("## Pass 2 中译英（ZH → EN，输入为 opus-mt 英译中产物）\n\n");
    md.push_str("| 引擎 | 加载耗时 | 翻译耗时 | 吞吐 (chars/s) | 回译相似度 | 状态 |\n|---|---|---|---|---|---|\n");
    for (name, r) in results {
        match &r.zh_en {
            Some(p) => {
                let score = r
                    .roundtrip_score
                    .map(|s| format!("{:.3}", s))
                    .unwrap_or_else(|| "-".into());
                md.push_str(&format!(
                    "| {} | {} | {} | {:.0} | {} | ✓ |\n",
                    name,
                    fmt_dur(p.load_ms),
                    fmt_dur(p.translate_ms),
                    p.throughput(),
                    score
                ));
            }
            None => {
                let why = r.unavailable.clone().unwrap_or_else(|| "未参与".into());
                md.push_str(&format!("| {} | - | - | - | - | {} |\n", name, why));
            }
        }
    }
    if !ct2_enabled {
        md.push_str("\n> 注：CTranslate2 需 `--features ct2` 编译启用。\n");
    }
    md.push_str("\n### 中译英译文明细（附回译原文对照）\n\n");
    for (i, s) in sentences.iter().enumerate() {
        md.push_str(&format!("**句 {}** 原文 `{}`\n", i + 1, s));
        md.push_str(&format!("中文输入: `{}`\n\n", zh_reference[i]));
        for (name, r) in results {
            if let Some(p) = &r.zh_en {
                md.push_str(&format!("- **{}**: {}\n", name, p.outputs[i]));
            }
        }
        md.push('\n');
    }

    std::fs::write(report_path(), &md)?;
    Ok(report_path().to_string_lossy().to_string())
}

// ============================================================
// 引擎装配辅助
// ============================================================

/// 引擎统计打印
fn print_pass(title: &str, records: &[&PassRecord]) {
    println!("\n=== {} ===", title);
    println!(
        "{:<14} {:>10} {:>10} {:>12}",
        "引擎", "加载", "翻译", "吞吐(chars/s)"
    );
    for r in records {
        println!(
            "{:<14} {:>10} {:>10} {:>12.0}",
            r.engine,
            fmt_dur(r.load_ms),
            fmt_dur(r.translate_ms),
            r.throughput()
        );
    }
}

// ============================================================
// 轻量测试（无模型依赖）
// ============================================================

#[test]
fn test_语料切分与相似度工具() {
    let text = std::fs::read_to_string(corpus_path())
        .unwrap_or_else(|e| panic!("读取语料失败（tmp/PhilosopherStone.txt）: {}", e));
    let sentences = pick_sentences(&text, 4);
    assert_eq!(sentences.len(), 4, "应从小说中切出 4 个测试句");
    for s in &sentences {
        assert!(
            (40..=120).contains(&s.chars().count()),
            "句子长度越界: {} ({})",
            s.len(),
            s
        );
    }
    println!("切分结果:");
    for (i, s) in sentences.iter().enumerate() {
        println!("  {}. {} ({} chars)", i + 1, s, s.len());
    }

    // 相似度工具自检：全同 = 1，无关 ≈ 0
    assert!((dice_similarity("Hello world", "hello world!") - 1.0).abs() < 1e-9);
    assert!(dice_similarity("abc", "xyz") < 0.2);
    println!("✓ Dice 相似度工具自检通过");
}

// ============================================================
// 主测试：双向 E2E 对比（需模型，slow-models 门控）
// ============================================================

#[cfg_attr(
    not(feature = "slow-models"),
    ignore = "需本地模型与推理，跑法: cargo test -p votex-infra --features ct2,slow-models --test translation_e2e_comparison_test -- --nocapture --test-threads=1"
)]
#[test]
fn test_翻译引擎双向e2e对比() {
    let t0 = Instant::now();

    // ---------- 语料准备 ----------
    let text = std::fs::read_to_string(corpus_path())
        .unwrap_or_else(|e| panic!("读取语料失败: {}", e));
    let sentences = pick_sentences(&text, 4);
    assert_eq!(sentences.len(), 4, "语料切分不足 4 段");
    let sent_refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
    println!("语料就绪: {} 段测试句（{}）", sentences.len(), fmt_dur(t0.elapsed().as_millis()));

    let mut results: Vec<(String, EngineResult)> = Vec::new();

    // ---------- Pass 1 英译中：opus-mt（同时产出 Pass 2 的共享中文输入） ----------
    let zh_reference = {
        let model_en_zh = workspace_root().join("models/translation/opus-mt-en-zh");
        assert!(model_en_zh.join("encoder_model.onnx").exists(), "opus-mt-en-zh 模型缺失");

        let engine = votex_infra::translation::opus_mt::OpusMtProvider::new();
        let load_start = Instant::now();
        engine
            .load_en_zh_from_dir(&model_en_zh)
            .unwrap_or_else(|e| panic!("加载 opus-mt-en-zh 失败: {}", e));
        let load_ms = load_start.elapsed().as_millis();

        let translate_start = Instant::now();
        let mut outs = Vec::new();
        for s in &sent_refs {
            outs.push(
                engine
                    .translate(s, TranslationDirection::EnToZh)
                    .unwrap_or_else(|e| panic!("opus-mt 英译中失败: {}", e)),
            );
        }
        let translate_ms = translate_start.elapsed().as_millis();

        let input_chars: usize = sent_refs.iter().map(|s| s.chars().count()).sum();
        let rec = PassRecord {
            engine: "opus-mt".into(),
            load_ms,
            translate_ms,
            input_chars,
            outputs: outs.clone(),
        };
        let throughput = rec.throughput();
        println!("\n[Pass 1] opus-mt en→zh: 加载 {} 翻译 {} 吞吐 {:.0} chars/s",
            fmt_dur(load_ms), fmt_dur(translate_ms), throughput);

        let mut r = EngineResult::new();
        r.en_zh = Some(rec);
        results.push(("opus-mt".into(), r));
        outs
    };

    // ---------- Pass 2 中译英：opus-mt zh→en ----------
    {
        let model_zh_en = workspace_root().join("models/translation/opus-mt-zh-en");
        assert!(model_zh_en.join("encoder_model.onnx").exists(), "opus-mt-zh-en 模型缺失");

        let engine = votex_infra::translation::opus_mt::OpusMtProvider::new();
        let load_start = Instant::now();
        engine
            .load_zh_en_from_dir(&model_zh_en)
            .unwrap_or_else(|e| panic!("加载 opus-mt-zh-en 失败: {}", e));
        let load_ms = load_start.elapsed().as_millis();

        let zh_refs: Vec<&str> = zh_reference.iter().map(|s| s.as_str()).collect();
        let translate_start = Instant::now();
        let mut outs = Vec::new();
        for s in &zh_refs {
            outs.push(
                engine
                    .translate(s, TranslationDirection::ZhToEn)
                    .unwrap_or_else(|e| panic!("opus-mt 中译英失败: {}", e)),
            );
        }
        let translate_ms = translate_start.elapsed().as_millis();

        let input_chars: usize = zh_reference.iter().map(|s| s.chars().count()).sum();
        // 回译相似度：译回英文 vs 英文原文
        let score = outs
            .iter()
            .zip(&sentences)
            .map(|(back, orig)| dice_similarity(back, orig))
            .sum::<f64>()
            / outs.len() as f64;
        let rec = PassRecord {
            engine: "opus-mt".into(),
            load_ms,
            translate_ms,
            input_chars,
            outputs: outs,
        };
        println!("[Pass 2] opus-mt zh→en: 加载 {} 翻译 {} 回译相似度 {:.3}",
            fmt_dur(load_ms), fmt_dur(translate_ms), score);

        let entry = results
            .iter_mut()
            .find(|(n, _)| n == "opus-mt")
            .map(|(_, r)| r)
            .unwrap();
        entry.zh_en = Some(rec);
        entry.roundtrip_score = Some(score);
    }

    // ---------- nllb-200 双向 ----------
    {
        let model_dir = workspace_root().join("models/translation/nllb-200-distilled-600m");
        let ready = (model_dir.join("encoder_model_int8.onnx").exists()
            || model_dir.join("encoder_model.onnx").exists())
            && (model_dir.join("decoder_model_int8.onnx").exists()
                || model_dir.join("decoder_model.onnx").exists())
            && model_dir.join("sentencepiece.bpe.model").exists();
        let mut r = EngineResult::new();
        if !ready {
            r.unavailable = Some("模型缺失".into());
            results.push(("nllb-200".into(), r));
        } else {
            let yaml = workspace_root().join("models/registry/nllb-200-distilled-600m.yaml");
            let config: votex_domain::model::registry::TokenizerConfig = {
                let content = std::fs::read_to_string(&yaml).unwrap();
                serde_yml::from_str::<votex_domain::model::registry::ModelRegistryEntry>(&content)
                    .unwrap()
                    .tokenizer
                    .unwrap_or_default()
            };
            let engine = votex_infra::translation::nllb::NllbProvider::new()
                .with_tokenizer_config(config);

            // en→zh
            let load_start = Instant::now();
            engine
                .load_from_dir(&model_dir)
                .unwrap_or_else(|e| panic!("加载 nllb 失败: {}", e));
            let load_ms_en = load_start.elapsed().as_millis();
            let translate_start = Instant::now();
            let mut outs = Vec::new();
            for s in &sent_refs {
                outs.push(
                    engine
                        .translate(s, TranslationDirection::EnToZh)
                        .unwrap_or_else(|e| panic!("nllb 英译中失败: {}", e)),
                );
            }
            let tr_ms_en = translate_start.elapsed().as_millis();
            r.en_zh = Some(PassRecord {
                engine: "nllb-200".into(),
                load_ms: load_ms_en,
                translate_ms: tr_ms_en,
                input_chars: sent_refs.iter().map(|s| s.chars().count()).sum(),
                outputs: outs,
            });

            // zh→en（模型已加载，加载耗时计 0——nllb 单模型双向）
            let zh_refs: Vec<&str> = zh_reference.iter().map(|s| s.as_str()).collect();
            let translate_start = Instant::now();
            let mut outs = Vec::new();
            for s in &zh_refs {
                outs.push(
                    engine
                        .translate(s, TranslationDirection::ZhToEn)
                        .unwrap_or_else(|e| panic!("nllb 中译英失败: {}", e)),
                );
            }
            let tr_ms_zh = translate_start.elapsed().as_millis();
            r.zh_en = Some(PassRecord {
                engine: "nllb-200".into(),
                load_ms: 0,
                translate_ms: tr_ms_zh,
                input_chars: zh_reference.iter().map(|s| s.chars().count()).sum(),
                outputs: outs,
            });
            let score = r.zh_en.as_ref().unwrap().outputs
                .iter()
                .zip(&sentences)
                .map(|(back, orig)| dice_similarity(back, orig))
                .sum::<f64>()
                / sentences.len() as f64;
            r.roundtrip_score = Some(score);
            println!("[nllb-200] en→zh {} / zh→en {} 回译 {:.3}",
                fmt_dur(tr_ms_en), fmt_dur(tr_ms_zh), score);
            results.push(("nllb-200".into(), r));
        }
    }

    // ---------- m2m-100 双向 ----------
    {
        let model_dir = workspace_root().join("models/translation/m2m-100");
        let ready = (model_dir.join("encoder_model.int8.onnx").exists()
            || model_dir.join("encoder_model.onnx").exists())
            && (model_dir.join("decoder_model.int8.onnx").exists()
                || model_dir.join("decoder_model.onnx").exists())
            && model_dir.join("sentencepiece.bpe.model").exists();
        let mut r = EngineResult::new();
        if !ready {
            r.unavailable = Some("模型缺失".into());
            results.push(("m2m-100".into(), r));
        } else {
            let engine = votex_infra::translation::m2m100::M2m100Provider::new();
            let load_start = Instant::now();
            engine
                .load_from_dir(&model_dir)
                .unwrap_or_else(|e| panic!("加载 m2m-100 失败: {}", e));
            let load_ms_en = load_start.elapsed().as_millis();

            let translate_start = Instant::now();
            let mut outs = Vec::new();
            for s in &sent_refs {
                outs.push(
                    engine
                        .translate(s, TranslationDirection::EnToZh)
                        .unwrap_or_else(|e| panic!("m2m-100 英译中失败: {}", e)),
                );
            }
            let tr_ms_en = translate_start.elapsed().as_millis();
            r.en_zh = Some(PassRecord {
                engine: "m2m-100".into(),
                load_ms: load_ms_en,
                translate_ms: tr_ms_en,
                input_chars: sent_refs.iter().map(|s| s.chars().count()).sum(),
                outputs: outs,
            });

            let zh_refs: Vec<&str> = zh_reference.iter().map(|s| s.as_str()).collect();
            let translate_start = Instant::now();
            let mut outs = Vec::new();
            for s in &zh_refs {
                outs.push(
                    engine
                        .translate(s, TranslationDirection::ZhToEn)
                        .unwrap_or_else(|e| panic!("m2m-100 中译英失败: {}", e)),
                );
            }
            let tr_ms_zh = translate_start.elapsed().as_millis();
            r.zh_en = Some(PassRecord {
                engine: "m2m-100".into(),
                load_ms: 0,
                translate_ms: tr_ms_zh,
                input_chars: zh_reference.iter().map(|s| s.chars().count()).sum(),
                outputs: outs,
            });
            let score = r.zh_en.as_ref().unwrap().outputs
                .iter()
                .zip(&sentences)
                .map(|(back, orig)| dice_similarity(back, orig))
                .sum::<f64>()
                / sentences.len() as f64;
            r.roundtrip_score = Some(score);
            println!("[m2m-100] en→zh {} / zh→en {} 回译 {:.3}",
                fmt_dur(tr_ms_en), fmt_dur(tr_ms_zh), score);
            results.push(("m2m-100".into(), r));
        }
    }

    // ---------- hy-mt-1.5 双向 ----------
    {
        let model_dir = workspace_root().join("models/translation/hy-mt-1.5");
        let ready = (model_dir.join("model_q4.onnx").exists() || model_dir.join("model.onnx").exists())
            && model_dir.join("tokenizer.json").exists();
        let mut r = EngineResult::new();
        if !ready {
            r.unavailable = Some("模型缺失".into());
            results.push(("hy-mt-1.5".into(), r));
        } else {
            let engine = votex_infra::translation::hy_mt::HyMtProvider::new();
            let load_start = Instant::now();
            engine
                .load_from_dir(&model_dir)
                .unwrap_or_else(|e| panic!("加载 hy-mt-1.5 失败: {}", e));
            let load_ms_en = load_start.elapsed().as_millis();

            let translate_start = Instant::now();
            let mut outs = Vec::new();
            for s in &sent_refs {
                outs.push(
                    engine
                        .translate(s, TranslationDirection::EnToZh)
                        .unwrap_or_else(|e| panic!("hy-mt-1.5 英译中失败: {}", e)),
                );
            }
            let tr_ms_en = translate_start.elapsed().as_millis();
            r.en_zh = Some(PassRecord {
                engine: "hy-mt-1.5".into(),
                load_ms: load_ms_en,
                translate_ms: tr_ms_en,
                input_chars: sent_refs.iter().map(|s| s.chars().count()).sum(),
                outputs: outs,
            });

            let zh_refs: Vec<&str> = zh_reference.iter().map(|s| s.as_str()).collect();
            let translate_start = Instant::now();
            let mut outs = Vec::new();
            for s in &zh_refs {
                outs.push(
                    engine
                        .translate(s, TranslationDirection::ZhToEn)
                        .unwrap_or_else(|e| panic!("hy-mt-1.5 中译英失败: {}", e)),
                );
            }
            let tr_ms_zh = translate_start.elapsed().as_millis();
            r.zh_en = Some(PassRecord {
                engine: "hy-mt-1.5".into(),
                load_ms: 0,
                translate_ms: tr_ms_zh,
                input_chars: zh_reference.iter().map(|s| s.chars().count()).sum(),
                outputs: outs,
            });
            let score = r.zh_en.as_ref().unwrap().outputs
                .iter()
                .zip(&sentences)
                .map(|(back, orig)| dice_similarity(back, orig))
                .sum::<f64>()
                / sentences.len() as f64;
            r.roundtrip_score = Some(score);
            println!("[hy-mt-1.5] en→zh {} / zh→en {} 回译 {:.3}",
                fmt_dur(tr_ms_en), fmt_dur(tr_ms_zh), score);
            results.push(("hy-mt-1.5".into(), r));
        }
    }

    // ---------- CTranslate2 中译英（ct2 feature 门控） ----------
    #[cfg(feature = "ct2")]
    {
        let model_dir = workspace_root().join("models/translation/ct2-opus-mt-zh-en");
        let mut r = EngineResult::new();
        if model_dir.join("model.bin").exists() {
            let engine = votex_infra::translation::ctranslate2::CTranslate2Provider::new();
            let load_start = Instant::now();
            engine
                .load_from_dir(&model_dir)
                .unwrap_or_else(|e| panic!("加载 ct2 模型失败: {}", e));
            let load_ms = load_start.elapsed().as_millis();

            let zh_refs: Vec<&str> = zh_reference.iter().map(|s| s.as_str()).collect();
            let translate_start = Instant::now();
            let mut outs = Vec::new();
            for s in &zh_refs {
                outs.push(
                    engine
                        .translate(s, TranslationDirection::ZhToEn)
                        .unwrap_or_else(|e| panic!("ct2 中译英失败: {}", e)),
                );
            }
            let tr_ms = translate_start.elapsed().as_millis();
            r.zh_en = Some(PassRecord {
                engine: "ctranslate2".into(),
                load_ms,
                translate_ms: tr_ms,
                input_chars: zh_reference.iter().map(|s| s.chars().count()).sum(),
                outputs: outs,
            });
            let score = r.zh_en.as_ref().unwrap().outputs
                .iter()
                .zip(&sentences)
                .map(|(back, orig)| dice_similarity(back, orig))
                .sum::<f64>()
                / sentences.len() as f64;
            r.roundtrip_score = Some(score);
            println!("[ctranslate2] zh→en: 加载 {} 翻译 {} 回译 {:.3}",
                fmt_dur(load_ms), fmt_dur(tr_ms), score);
        } else {
            r.unavailable = Some("模型缺失".into());
        }
        results.push(("ctranslate2".into(), r));
    }
    #[cfg(not(feature = "ct2"))]
    {
        let mut r = EngineResult::new();
        r.unavailable = Some("需 --features ct2".into());
        results.push(("ctranslate2".into(), r));
    }

    // ---------- 汇总打印 ----------
    println!("\n================= 汇总对比 =================");
    let en_zh_recs: Vec<&PassRecord> =
        results.iter().filter_map(|(_, r)| r.en_zh.as_ref()).collect();
    let zh_en_recs: Vec<&PassRecord> =
        results.iter().filter_map(|(_, r)| r.zh_en.as_ref()).collect();
    if !en_zh_recs.is_empty() {
        print_pass("Pass 1 英译中 (EN→ZH)", &en_zh_recs);
    }
    if !zh_en_recs.is_empty() {
        print_pass("Pass 2 中译英 (ZH→EN)", &zh_en_recs);
    }
    println!("\n回译相似度（Pass 2 英文译文 vs 原文，越高越接近原文）:");
    for (name, r) in &results {
        if let Some(s) = r.roundtrip_score {
            println!("  {:<14} {:.3}", name, s);
        }
    }
    println!("===========================================");
    println!("总耗时: {}", fmt_dur(t0.elapsed().as_millis()));

    // ---------- 报告落盘 ----------
    let report_file = write_report(&sentences, &zh_reference, &results, cfg!(feature = "ct2"))
        .unwrap_or_else(|e| panic!("写报告失败: {}", e));
    println!("报告已落盘: {}", report_file);

    // 基本契约：参与引擎的译文非空
    for (name, r) in &results {
        if let Some(p) = &r.en_zh {
            for (i, o) in p.outputs.iter().enumerate() {
                assert!(!o.trim().is_empty(), "{} 英译中第 {} 句译文为空", name, i + 1);
            }
        }
        if let Some(p) = &r.zh_en {
            for (i, o) in p.outputs.iter().enumerate() {
                assert!(!o.trim().is_empty(), "{} 中译英第 {} 句译文为空", name, i + 1);
            }
        }
    }
}
