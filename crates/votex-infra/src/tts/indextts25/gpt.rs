//! GPT 自回归 mel code 生成主循环（gpt_sampler.generate_codes 忠实移植）
//!
//! # 循环契约（与参考实现逐行对齐）
//!
//! 1. prefill：`(logits, state) = backend.prefill(text_ids, conds, lang_id)`
//! 2. 惩罚上下文以伪前缀 `{STOP_TEXT_TOKEN, START_MEL_TOKEN}` = `{1, 8192}` 起始
//! 3. 每步：重复惩罚 →（greedy ? argmax : 采样链）→ 命中 stop token 则停止
//! 4. step 的 mel 位置索引为 `i + 2`（start_mel 占据 mel 位置 0，位置 1 被**跳过**——
//!    已由金标验证的 quirk，见蓝本 §3.3）
//! 5. 返回的 mel codes **不含** stop token
//!
//! # 确定性
//!
//! greedy 模式与参考实现逐 token 一致（同一模型权重 + 相同输入）。
//! 采样模式依赖 RNG 流：参考实现为 numpy PCG64，本模块为 StdRng，
//! 同 seed 不同流（蓝本 §7.4 统计对拍约定）。

use anyhow::Result;
use ndarray::ArrayD;
use std::collections::BTreeSet;

use super::engines::{gpt_prefill_run, gpt_step_run, IndexTts25Engines};
use super::sampler::{
    apply_repetition_penalty, argmax_first, sample, GptParams, GptRng, START_MEL_TOKEN,
    STOP_TEXT_TOKEN,
};

/// 生成 mel codes（stop token 不含在返回值内）
///
/// # 参数
/// - `engines`: 已加载的引擎（需要 `gpt_prefill` / `gpt_step` 两个 session）
/// - `text_ids`: 单条分段文本 token（长度 T，内部补 batch 维）
/// - `conds`: GPT 说话人条件 `[1,3,1280]`（`build_speaker` 产出）
/// - `lang_id`: 语言 id（`lang_id("yue")` 等，bpe 模块语言表序号）
/// - `params`: 采样参数（默认 top_k=30 / top_p=0.8 / temp=0.8 / rep=10.0）
/// - `seed`: 采样种子；greedy 模式下不消费
pub fn generate_codes(
    engines: &IndexTts25Engines,
    text_ids: &[i64],
    conds: &ArrayD<f32>,
    lang_id: i64,
    params: &GptParams,
    seed: u64,
) -> Result<Vec<i64>> {
    if text_ids.is_empty() {
        anyhow::bail!("text_ids 为空");
    }
    let text_ids_t = ndarray::Array2::<i64>::from_shape_vec((1, text_ids.len()), text_ids.to_vec())
        .map_err(|e| anyhow::anyhow!("text_ids 形状转换失败: {e}"))?
        .into_dyn();

    // 1. prefill（EngineNotLoaded → anyhow，双层 ? 依次解包）
    let (logits0, mut state) = engines
        .gpt_prefill
        .with_mut(|s| gpt_prefill_run(s, text_ids_t, conds.clone(), lang_id))
        .map_err(|_| anyhow::anyhow!("gpt_prefill 引擎未加载"))??;

    let mut logits = logits0;
    let mut codes: Vec<i64> = Vec::with_capacity(params.max_mel_tokens.min(4096));
    let mut seen: BTreeSet<i64> = BTreeSet::new();
    seen.insert(STOP_TEXT_TOKEN);
    seen.insert(START_MEL_TOKEN);

    let mut rng = GptRng::new(seed);

    // 2. 自回归循环（在 gpt_step 的锁内完成整个循环，避免每步重入锁）
    engines.gpt_step.with_mut(|s| -> Result<()> {
        for i in 0..params.max_mel_tokens {
            // 重复惩罚（f32 原地）
            apply_repetition_penalty(&mut logits, &seen, params.repetition_penalty);

            let token = if params.greedy {
                argmax_first(&logits) as i64
            } else {
                let u = rng.next_f64();
                sample(&logits, u, params.temperature, params.top_k, params.top_p) as i64
            };

            if token == params.stop_token {
                break;
            }
            codes.push(token);
            seen.insert(token);

            // step：mel 位置索引 i + 2（quirk，见模块文档）
            logits = gpt_step_run(s, &mut state, token, i as i64 + 2)?;
        }
        Ok(())
    })
    .map_err(|_| anyhow::anyhow!("gpt_step 引擎未加载"))??;

    Ok(codes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 伪 text_ids / conds（结构校验用；数值对拍走 slow-models 集成测试）
    fn dummy_cond() -> ArrayD<f32> {
        ndarray::Array3::<f32>::zeros((1, 3, 1280)).into_dyn()
    }

    #[test]
    fn 参数默认值与参考实现一致() {
        let p = GptParams::default();
        assert!(!p.greedy);
        assert_eq!(p.top_k, 30);
        assert!((p.top_p - 0.8).abs() < f64::EPSILON);
        assert!((p.temperature - 0.8).abs() < f64::EPSILON);
        assert!((p.repetition_penalty - 10.0).abs() < f64::EPSILON);
        assert_eq!(p.max_mel_tokens, 1500);
        assert_eq!(p.stop_token, 8193);
    }

    #[test]
    fn 空text_ids直接报错() {
        let engines = IndexTts25Engines::new();
        let cond = dummy_cond();
        let err = generate_codes(&engines, &[], &cond, 1, &GptParams::default(), 0);
        assert!(err.is_err());
    }

    #[test]
    fn 引擎未加载时报错而非panic() {
        let engines = IndexTts25Engines::new();
        let cond = dummy_cond();
        let err = generate_codes(&engines, &[100, 200], &cond, 1, &GptParams::default(), 42);
        assert!(err.is_err());
    }

    /// 真实 GPT 模型 greedy 循环金标对拍（slow-models 门控，加载 ~8.5GB session）。
    ///
    /// 金标 `tmp/indextts25_refs/gpt_loop_golden.json` 由参考实现生成：
    /// 零 conds + 短文本 zh + greedy 24 步。硬契约：codes 逐 token 一致；
    /// prefill logits 仅作诊断打印（跨 ORT 版本允许 ulp 级差异）。
    #[test]
    #[cfg_attr(
        not(feature = "slow-models"),
        ignore = "加载 GB 级 GPT 模型，--features slow-models 启用"
    )]
    fn gpt循环_真实模型greedy金标逐token一致() {
        let Some(model_dir) = super::super::engines::find_model_dir() else {
            panic!("未找到 IndexTTS-2.5 模型目录（models/tts/indextts25 或 HF 缓存）");
        };
        let golden_path = crate::shared::workspace_paths::WorkspacePaths::workspace_root()
            .join("tmp/indextts25_refs/gpt_loop_golden.json");
        let golden: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(&golden_path).expect("读取 gpt_loop_golden.json"),
        )
        .expect("解析金标 JSON");

        let text_ids: Vec<i64> = golden["text_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();
        let lang_id = golden["lang_id"].as_i64().unwrap();
        let max_steps = golden["max_mel_tokens"].as_u64().unwrap() as usize;
        let want_codes: Vec<i64> = golden["codes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect();

        let engines = IndexTts25Engines::new();
        engines
            .load(&model_dir, 8)
            .expect("GPT session 加载失败");

        let conds = ndarray::Array3::<f32>::zeros((1, 3, 1280)).into_dyn();
        let params = GptParams {
            greedy: true,
            max_mel_tokens: max_steps,
            ..GptParams::default()
        };
        let codes =
            generate_codes(&engines, &text_ids, &conds, lang_id, &params, 0)
                .expect("generate_codes 失败");

        assert_eq!(
            codes,
            want_codes,
            "greedy codes 与参考实现不一致（len want={} got={}）",
            want_codes.len(),
            codes.len()
        );
        engines.unload_all();
    }
}
