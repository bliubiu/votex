//! GPT 采样循环的纯函数部分（gpt_sampler.py 忠实移植）
//!
//! 复刻 HF logits-processor 顺序：RepetitionPenalty(10.0) → Temperature →
//! TopK → TopP → 多项分布（greedy 时直接 argmax）。
//!
//! # 与参考实现的数值契约
//!
//! - 重复惩罚：参考实现直接在 **f32** logits 上原地乘/除（numpy 标量弱类型提升），
//!   本模块同样在 f32 上执行。
//! - 采样链：进入采样后先 `f32 → f64`，再除温度——与 `logits.astype(np.float64)`
//!   一致，全程 f64 运算。
//! - TopK 阈值：`np.partition(logits, -k)[-k]` 等价于「第 k 大值」（含重复值），
//!   用 `select_nth_unstable_by` 复刻；**等于阈值的元素保留**（`<` 严格比较）。
//! - TopP：按降序累计概率，`cum > top_p` 的位置右移一位后置 `-inf`
//!   （首位永不移除，保证至少一个 token 可采）。
//! - 多项分布：`searchsorted(cumsum(probs), u)`，`u` 由调用方注入——
//!   生产路径用 [`GptRng`]，测试路径注入固定 `u` 与 numpy 金标逐位对拍。
//!   RNG 流本身与 `np.random.default_rng`（PCG64）不同（本模块用 StdRng），
//!   属于蓝本 §7.4 约定的「统计对拍」范畴。

use std::collections::BTreeSet;

/// 伪前缀 token：text-stop（对应伪 input_ids 的 text-stop）与 start-mel
///
/// 重复惩罚上下文以 `{STOP_TEXT_TOKEN, START_MEL_TOKEN}` 起始，
/// 对齐 inference_speech 的 `generate()` 调用。
pub const START_TEXT_TOKEN: i64 = 0;
pub const STOP_TEXT_TOKEN: i64 = 1;
pub const START_MEL_TOKEN: i64 = 8192;
pub const STOP_MEL_TOKEN: i64 = 8193;

/// GPT 生成参数（与参考实现 `generate_codes` 签名一致）
#[derive(Debug, Clone)]
pub struct GptParams {
    /// 贪心解码（跳过整条采样链，仅重复惩罚后 argmax）
    pub greedy: bool,
    pub top_k: usize,
    pub top_p: f64,
    pub temperature: f64,
    pub repetition_penalty: f64,
    pub max_mel_tokens: usize,
    pub stop_token: i64,
}

impl Default for GptParams {
    fn default() -> Self {
        Self {
            greedy: false,
            top_k: 30,
            top_p: 0.8,
            temperature: 0.8,
            repetition_penalty: 10.0,
            max_mel_tokens: 1500,
            stop_token: STOP_MEL_TOKEN,
        }
    }
}

/// 重复惩罚：已出现的 token，正 logit 除以 penalty，负 logit 乘以 penalty
///
/// `seen` 中超出 logits 长度的 id 会被忽略（防御越界）。
pub fn apply_repetition_penalty(logits: &mut [f32], seen: &BTreeSet<i64>, penalty: f64) {
    if penalty == 1.0 || seen.is_empty() {
        return;
    }
    let penalty = penalty as f32;
    for &tid in seen {
        if tid < 0 {
            continue;
        }
        if let Some(l) = logits.get_mut(tid as usize) {
            if *l < 0.0 {
                *l *= penalty;
            } else {
                *l /= penalty;
            }
        }
    }
}

/// argmax（返回**首个**最大值下标，与 `np.argmax` 的平局语义一致）
///
/// `Iterator::max_by` 平局时返回最后一个元素，与 numpy 相反，必须手写折叠。
pub fn argmax_first(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best
}

/// 采样：温度缩放 → TopK → TopP → softmax 累积分布上按 `u` 查表
///
/// # 参数
/// - `logits`: 惩罚后的 f32 logits（长度 = n_vocab 头部，8194）
/// - `u`: [0,1) 均匀随机数；测试时注入固定值即可复现 numpy 金标
///
/// # 返回
/// 选中的 token id（logits 下标）
pub fn sample(logits: &[f32], u: f64, temperature: f64, top_k: usize, top_p: f64) -> usize {
    let n = logits.len();
    assert!(n > 0, "logits 为空");
    let mut l: Vec<f64> = logits.iter().map(|&v| v as f64).collect();

    // 温度（max(temperature, 1e-6)）
    let t = temperature.max(1e-6);
    for v in &mut l {
        *v /= t;
    }

    // TopK：第 k 大值以下置 -inf（等于阈值保留）
    if top_k > 0 {
        let k = top_k.min(n);
        let idx = n - k;
        let mut copy = l.clone();
        copy.select_nth_unstable_by(idx, f64::total_cmp);
        let thresh = copy[idx];
        for v in &mut l {
            if *v < thresh {
                *v = f64::NEG_INFINITY;
            }
        }
    }

    // TopP：降序累计概率，越过 top_p 后（右移一位保留首个越界项）置 -inf
    if top_p > 0.0 && top_p < 1.0 {
        let mut order: Vec<usize> = (0..n).collect();
        // 降序；total_cmp 保证 NaN 确定性排序（正常 logits 无 NaN）
        order.sort_by(|&a, &b| l[b].total_cmp(&l[a]));
        let max0 = l[order[0]];
        let probs: Vec<f64> = order.iter().map(|&i| (l[i] - max0).exp()).collect();
        let sum: f64 = probs.iter().sum();
        if sum.is_finite() && sum > 0.0 {
            let mut cum = 0.0f64;
            let mut cumv = Vec::with_capacity(n);
            for p in &probs {
                cum += p / sum;
                cumv.push(cum);
            }
            // remove = cum > top_p；remove[1:] = remove[:-1]；remove[0] = False
            let mut remove = vec![false; n];
            for i in (1..n).rev() {
                remove[i] = cumv[i - 1] > top_p;
            }
            for (i, &r) in remove.iter().enumerate() {
                if r {
                    l[order[i]] = f64::NEG_INFINITY;
                }
            }
        }
    }

    // softmax（减最大值）+ 累积分布查表
    let mx = l.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = l.iter().map(|&v| (v - mx).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        // 全 -inf / NaN 的退化输入：回退到 0（Python 侧会产出 NaN，属未定义行为）
        return 0;
    }
    let mut cum = 0.0f64;
    let mut cumv = Vec::with_capacity(n);
    for e in &exps {
        cum += e / sum;
        cumv.push(cum);
    }
    // np.searchsorted(cumsum, u)：第一个 cum >= u 的下标；防御 u 超界
    let idx = cumv.partition_point(|&c| c < u);
    idx.min(n - 1)
}

/// 生产用 RNG 封装（StdRng / ChaCha12）
///
/// 与 `np.random.default_rng`（PCG64）流不同——见模块文档「统计对拍」约定。
pub struct GptRng {
    inner: rand::rngs::StdRng,
}

impl GptRng {
    pub fn new(seed: u64) -> Self {
        use rand::SeedableRng;
        Self {
            inner: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// 抽取一个 [0,1) 均匀随机数
    pub fn next_f64(&mut self) -> f64 {
        rand::Rng::gen(&mut self.inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 与 numpy 金标对拍的辅助：从 JSON 读入用例
    ///
    /// 金标由 `tmp/make_gpt_refs.py` 生成：每组用例包含 logits（f32）、
    /// seen 集合、惩罚后 logits、以及若干固定 u 的采样结果。
    fn load_cases() -> serde_json::Value {
        let path = crate::shared::workspace_paths::WorkspacePaths::workspace_root()
            .join("tmp")
            .join("indextts25_refs")
            .join("gpt_sampler_cases.json");
        let data = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("金标文件不存在 {:?}: {}", path, e));
        serde_json::from_str(&data).expect("金标 JSON 解析失败")
    }

    /// 金标目录缺失时相关对拍测试直接跳过（金标由脚本生成，不入库）
    fn 金标缺失时跳过() -> bool {
        let dir = crate::shared::workspace_paths::WorkspacePaths::workspace_root()
            .join("tmp/indextts25_refs");
        !dir.exists()
    }

    #[test]
    fn 重复惩罚_正负logit方向与numpy一致() {
        let mut logits = vec![2.0f32, -3.0, 1.0e-4, 0.0, -1.0e-4];
        let seen: BTreeSet<i64> = [0i64, 1, 2, 3, 4].into_iter().collect();
        apply_repetition_penalty(&mut logits, &seen, 10.0);
        // 正数除 10、负数乘 10、0 不变（f32 精确运算值，注意 1e-4×10 ≠ 字面量 1e-3）
        assert_eq!(logits[0], 2.0f32 / 10.0);
        assert_eq!(logits[1], -3.0f32 * 10.0);
        assert_eq!(logits[2], 1.0e-4f32 / 10.0);
        assert_eq!(logits[3], 0.0);
        assert_eq!(logits[4], -1.0e-4f32 * 10.0);
    }

    #[test]
    fn 重复惩罚_penalty为1时为no_op() {
        let mut logits = vec![2.0f32, -3.0];
        apply_repetition_penalty(&mut logits, &[0i64, 1].into_iter().collect(), 1.0);
        assert_eq!(logits, vec![2.0, -3.0]);
    }

    #[test]
    fn argmax_平局取首个与numpy一致() {
        assert_eq!(argmax_first(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax_first(&[-1.0, -5.0, -5.0]), 0);
    }

    #[test]
    fn 采样链与numpy金标对拍() {
        if 金标缺失时跳过() {
            eprintln!("⚠ 金标目录 tmp/indextts25_refs 不存在，跳过（由金标生成脚本产出）");
            return;
        }
        let cases = load_cases();
        let sample_cases = cases["sample"]
            .as_array()
            .expect("sample 用例缺失");
        assert!(!sample_cases.is_empty(), "sample 金标为空");
        for case in sample_cases {
            let logits: Vec<f32> = case["logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let temperature = case["temperature"].as_f64().unwrap();
            let top_k = case["top_k"].as_u64().unwrap() as usize;
            let top_p = case["top_p"].as_f64().unwrap();
            let expected: Vec<usize> = case["tokens"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let us: Vec<f64> = case["us"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            for ((&u, &want), i) in us.iter().zip(&expected).zip(0..) {
                let got = sample(&logits, u, temperature, top_k, top_p);
                assert_eq!(got, want, "sample 用例 {} u={} 不一致", i, u);
            }
        }
    }

    #[test]
    fn greedy_argmax链与numpy金标对拍() {
        if 金标缺失时跳过() {
            eprintln!("⚠ 金标目录 tmp/indextts25_refs 不存在，跳过（由金标生成脚本产出）");
            return;
        }
        let cases = load_cases();
        for case in cases["greedy"].as_array().expect("greedy 用例缺失") {
            let mut logits: Vec<f32> = case["logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap() as f32)
                .collect();
            let seen: BTreeSet<i64> = case["seen"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap())
                .collect();
            let penalty = case["repetition_penalty"].as_f64().unwrap();
            apply_repetition_penalty(&mut logits, &seen, penalty);
            let want = case["token"].as_u64().unwrap() as usize;
            assert_eq!(argmax_first(&logits), want);
        }
    }

    #[test]
    fn 采样链_topk等于全量时不截断() {
        // top_k >= len 时与 top_k = len 等价：全部保留，仅温度生效
        let logits = vec![1.0f32, 2.0, 3.0];
        let a = sample(&logits, 0.5, 1.0, 3, 1.0);
        let b = sample(&logits, 0.5, 1.0, 1000, 1.0);
        assert_eq!(a, b);
    }
}
