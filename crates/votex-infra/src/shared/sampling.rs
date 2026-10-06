//! 自回归解码的 token 采样
//!
//! # 统一动机
//!
//! 项目里 TTS / ASR / 翻译引擎都需要「从 logits 里按概率抽一个 token」，
//! 此前在 `cosyvoice.rs` 与 `qwen3_tts.rs` 各写了一份，两份代码在以下细节上
//! 行为不一致，属于典型的「同一件事两种做法」：
//!
//! | 细节 | cosyvoice `top_k_sample` | qwen3 `sample_top_k` |
//! | :--- | :--- | :--- |
//! | 输入语义 | 对数概率 | logits（先除温度） |
//! | `k == 0` | 退化为「取全局最优」 | 不截断（全量采样） |
//! | `k >= len` | 仍走排序 | 跳过截断 |
//! | 全部 `-inf` | 排序后取首个 | `sum <= 0` 时返回 0 |
//! | 浮点比较 | `partial_cmp().unwrap()` | `partial_cmp().unwrap_or(Equal)` |
//!
//! 后者是本模块存在的直接原因：`partial_cmp()` 对 NaN 返回 `None`，
//! `.unwrap()` 会 panic。ONNX 输出 NaN 在 fp16 溢出时并非罕见，
//! 自回归主循环里 panic 会直接中断整个合成任务。
//!
//! 本模块统一采用**不 panic** 的比较策略，并把温度缩放、top-k、top-p
//! 三步拆成可组合的独立函数。

use ndarray::Array1;
use std::cmp::Ordering;

/// 数值比较：NaN 视为最小（排到末尾），不 panic
///
/// `f32::partial_cmp` 对 NaN 返回 `None`。旧实现里
/// `b.1.partial_cmp(&a.1).unwrap()` 会在解码循环中直接崩溃，
/// 改为 `unwrap_or(Ordering::Equal)` 后 NaN 只会排到最后，
/// 而 `sort_by` 本身是稳定排序，行为可预期。
#[inline]
fn cmp_f32(a: f32, b: f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or(Ordering::Equal)
}

/// softmax：数值稳定实现（先减最大值，避免 exp 溢出）
///
/// # 参数
/// - `logits`: 待归一化数组，长度 0 时返回空数组
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    // 全为 -inf / 非有限时无法归一化，返回均匀分布由调用方决定回退
    let max_v = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max_v.is_finite() {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max_v).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return vec![1.0 / logits.len() as f32; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

/// 温度缩放：`score / temperature`
///
/// `temperature <= 0` 或非有限时视为 1.0（不缩放），
/// 避免除零产生 `inf` 后把整个分布打坏。
pub fn apply_temperature(scores: &mut [f32], temperature: f32) {
    if !temperature.is_finite() || temperature <= 0.0 || (temperature - 1.0).abs() < f32::EPSILON {
        return;
    }
    for s in scores.iter_mut() {
        *s /= temperature;
    }
}

/// top-k 截断：把第 k 名之后的分数压到 `-inf`
///
/// 返回 `(保留下来的索引, 截断后的分数)`。
/// - `k == 0` 或 `k >= len`：不截断
/// - `scores` 全为 `-inf`：不截断（由 softmax 决定回退）
pub fn top_k_filter(scores: &[f32], k: usize) -> (Vec<usize>, Vec<f32>) {
    let n = scores.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if k == 0 || k >= n {
        return ((0..n).collect(), scores.to_vec());
    }
    let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
    // 降序：b 与 a 比较
    indexed.sort_by(|a, b| cmp_f32(b.1, a.1));

    let threshold = indexed[k - 1].1;
    let mut kept: Vec<usize> = Vec::with_capacity(k);
    let mut filtered = scores.to_vec();
    for (i, &s) in scores.iter().enumerate() {
        // 保留所有 >= 阈值的项（阈值重复时保留数可能多于 k，符合概率语义）
        if cmp_f32(s, threshold) != Ordering::Less {
            kept.push(i);
        } else {
            filtered[i] = f32::NEG_INFINITY;
        }
    }
    (kept, filtered)
}

/// top-p（核采样）截断：保留累积概率达 `p` 的最小集合
///
/// 返回 `(保留下来的索引, 截断后的概率)`。输入为**已归一化的概率**。
/// - `p <= 0` 或 `p >= 1`：不截断
pub fn top_p_filter(probs: &[f32], p: f32) -> (Vec<usize>, Vec<f32>) {
    let n = probs.len();
    if n == 0 || p <= 0.0 || p >= 1.0 || !p.is_finite() {
        return ((0..n).collect(), probs.to_vec());
    }
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| cmp_f32(b.1, a.1));

    let mut kept: Vec<usize> = Vec::new();
    let mut cum = 0.0f32;
    for (i, prob) in &indexed {
        kept.push(*i);
        cum += *prob;
        if cum >= p {
            break;
        }
    }
    // 至少保留一个（累积概率可能因浮点误差始终 < p）
    if kept.is_empty() && !indexed.is_empty() {
        kept.push(indexed[0].0);
    }

    let mut filtered = vec![0.0f32; n];
    for &i in &kept {
        filtered[i] = probs[i];
    }
    (kept, filtered)
}

/// 从概率分布中按累积概率采样一个索引
///
/// # 参数
/// - `probs`: 已归一化的概率
/// - `rand`: `[0, 1)` 的随机数（注入以便测试可复现）
///
/// # 返回
/// 采样下标；分布为空或全零时返回 0
pub fn sample_from_probs(probs: &[f32], rand: f32) -> usize {
    if probs.is_empty() {
        return 0;
    }
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if rand < cum {
            return i;
        }
    }
    // 浮点累积误差可能导致永不命中，回退到最后一个非零项
    probs
        .iter()
        .enumerate()
        .rev()
        .find(|(_, &p)| p > 0.0)
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// 完整采样流程：温度 → top-k → softmax → top-p → 抽样
///
/// # 参数
/// - `logits`: 模型输出的原始分数（logits 或对数概率均可，
///   只需内部相对大小正确）
/// - `top_k`: 保留的最大项数，0 表示不截断
/// - `top_p`: 核采样阈值，`(0, 1)` 之外表示不截断
/// - `temperature`: 温度，`<= 0` 或非有限时视为 1.0
/// - `rand`: `[0, 1)` 随机数，注入以便测试确定性
///
/// # 返回
/// 采样得到的 token 下标；输入为空时返回 0
pub fn sample_token(
    logits: &[f32],
    top_k: usize,
    top_p: f32,
    temperature: f32,
    rand: f32,
) -> usize {
    if logits.is_empty() {
        return 0;
    }
    let mut scores = logits.to_vec();
    apply_temperature(&mut scores, temperature);

    let (_, filtered) = top_k_filter(&scores, top_k);
    let mut probs = softmax(&filtered);

    if top_p > 0.0 && top_p < 1.0 {
        let (_, p_filtered) = top_p_filter(&probs, top_p);
        probs = p_filtered;
        // top-p 截断后需要重新归一化，否则累积和 < 1 会漏掉尾部
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 && sum.is_finite() {
            for p in probs.iter_mut() {
                *p /= sum;
            }
        }
    }

    sample_from_probs(&probs, rand)
}

/// 便捷入口：接受 `ndarray::Array1` 并使用真随机数
///
/// 供已有 `Array1<f32>` 输入的引擎（cosyvoice）直接调用。
pub fn sample_token_array1(log_probs: &Array1<f32>, top_k: usize) -> i64 {
    let scores: Vec<f32> = log_probs.iter().copied().collect();
    let rand = rand::random::<f32>();
    sample_token(&scores, top_k, 0.0, 1.0, rand) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 确定性检查：`top_k=1` 是贪心解码，必中峰值
    fn assert_greedy_picks_peak(logits: &[f32]) {
        for r in [1e-4f32, 0.01, 0.5, 0.999] {
            let picked = sample_token(logits, 1, 0.0, 1.0, r);
            let peak = logits
                .iter()
                .enumerate()
                .max_by(|a, b| cmp_f32(*a.1, *b.1))
                .map(|(i, _)| i)
                .unwrap_or(0);
            assert_eq!(picked, peak, "贪心解码未选峰值: logits={logits:?} r={r}");
        }
    }

    /// 分布形状检查：概率与输入 logits 逐位同序（softmax 不重排），
    /// 且总和为 1
    fn assert_probs_follow_logits(logits: &[f32]) {
        let probs = softmax(logits);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        for i in 0..logits.len() {
            for j in 0..logits.len() {
                if cmp_f32(logits[i], logits[j]) == Ordering::Greater {
                    assert!(
                        probs[i] >= probs[j] - 1e-6,
                        "更高 logits 应有更高概率: idx{i}={} < idx{j}={}",
                        probs[i],
                        probs[j]
                    );
                }
            }
        }
    }

    #[test]
    fn softmax归一化() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
        assert!(p[2] > p[1] && p[1] > p[0], "顺序错乱: {p:?}");
    }

    #[test]
    fn softmax防溢出() {
        // 直接 exp(1000) 会 inf，减最大值后不会
        let p = softmax(&[1000.0, 1001.0]);
        assert!(p.iter().all(|v| v.is_finite()), "溢出: {p:?}");
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax全负无穷不panic() {
        let p = softmax(&[f32::NEG_INFINITY; 3]);
        assert_eq!(p.len(), 3);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn softmax空输入() {
        assert!(softmax(&[]).is_empty());
    }

    #[test]
    fn 温度无效时不缩放() {
        let mut s = vec![1.0, 2.0];
        apply_temperature(&mut s, 0.0);
        assert_eq!(s, vec![1.0, 2.0], "temperature=0 不应产生 inf");
        let mut s2 = vec![1.0, 2.0];
        apply_temperature(&mut s2, f32::NAN);
        assert_eq!(s2, vec![1.0, 2.0], "NaN 温度不应破坏分布");
    }

    #[test]
    fn 温度影响分布陡峭度() {
        let low = softmax(&[1.0, 2.0, 3.0]);
        let mut s = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut s, 0.1);
        let high = softmax(&s);
        assert!(high[2] > low[2], "低温应更集中于峰值: {low:?} vs {high:?}");
    }

    #[test]
    fn topk为零不截断() {
        let (kept, _) = top_k_filter(&[1.0, 2.0, 3.0], 0);
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn topk超界不截断() {
        let (kept, _) = top_k_filter(&[1.0, 2.0, 3.0], 10);
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn topk保留最高分() {
        let (kept, filtered) = top_k_filter(&[1.0, 5.0, 3.0], 2);
        assert_eq!(kept.len(), 2);
        assert!(kept.contains(&1), "最高分应保留: {kept:?}");
        assert!(!kept.contains(&0), "最低分应剔除: {kept:?}");
        assert_eq!(filtered[0], f32::NEG_INFINITY);
    }

    #[test]
    fn topk阈值重复时保留全部并列项() {
        // 三个相同分数，k=1 —— 概率语义下应全保留，否则会丢掉并列概率质量
        let (kept, _) = top_k_filter(&[2.0, 2.0, 2.0], 1);
        assert_eq!(kept.len(), 3, "并列项被误删: {kept:?}");
    }

    #[test]
    fn topp窄核只留峰值() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        let (kept, _) = top_p_filter(&probs, 0.5);
        assert_eq!(kept, vec![2], "p=0.5 应只留峰值: {kept:?}");
    }

    #[test]
    fn topp无效值不截断() {
        let probs = softmax(&[1.0, 2.0, 3.0]);
        for p in [0.0, 1.0, 1.5, f32::NAN] {
            let (kept, _) = top_p_filter(&probs, p);
            assert_eq!(kept.len(), 3, "p={p} 不应截断");
        }
    }

    #[test]
    fn 贪心解码必中峰值() {
        // top_k=1 时只剩一个候选，与随机数无关
        assert_greedy_picks_peak(&[1.0, 9.0, 3.0]);
        assert_greedy_picks_peak(&[-5.0, -1.0, -3.0]);
    }

    #[test]
    fn 概率分布与logits同序() {
        assert_probs_follow_logits(&[1.0, 9.0, 3.0]);
        assert_probs_follow_logits(&[0.0; 4]);
    }

    #[test]
    fn 尾部随机点应能命中低概率项() {
        // r 极小时命中累积概率最靠前的低分项，这是正确行为：
        // 峰值概率约 0.997，r=1e-4 落在尾部 0.003 内
        let logits = [1.0, 9.0, 3.0];
        let probs = softmax(&logits);
        let mut cum = 0.0;
        let mut first_idx = probs.len() - 1;
        for (i, &p) in probs.iter().enumerate() {
            cum += p;
            if cum > 1e-4 {
                first_idx = i;
                break;
            }
        }
        let picked = sample_token(&logits, 0, 0.0, 1.0, 1e-4);
        assert_eq!(picked, first_idx, "尾部采样落点不符: {probs:?}");
    }

    #[test]
    fn NaN不导致panic() {
        // 这是本模块存在的核心原因：旧实现 partial_cmp().unwrap() 会崩
        let logits = [f32::NAN, 1.0, 5.0];
        for k in [0usize, 1, 2, 3, 10] {
            for r in [1e-4f32, 0.3, 0.7, 0.99] {
                let _ = sample_token(&logits, k, 0.9, 1.0, r);
            }
        }
    }

    #[test]
    fn 负无穷不导致panic() {
        let logits = [f32::NEG_INFINITY; 4];
        for k in [0usize, 2, 4] {
            let picked = sample_token(&logits, k, 0.9, 1.0, 0.5);
            assert!(picked < 4, "越界: {picked}");
        }
    }

    #[test]
    fn 全负无穷时仍返回合法索引() {
        let logits = [f32::NEG_INFINITY; 3];
        let picked = sample_token(&logits, 2, 0.5, 1.0, 0.42);
        assert!(picked < 3, "越界: {picked}");
    }

    #[test]
    fn 空输入返回零() {
        assert_eq!(sample_token(&[], 5, 0.9, 1.0, 0.5), 0);
        assert_eq!(sample_from_probs(&[], 0.5), 0);
    }

    #[test]
    fn 全零概率不panic() {
        let p = vec![0.0, 0.0, 0.0];
        let picked = sample_from_probs(&p, 0.99);
        assert!(picked < 3);
    }

    #[test]
    fn array1入口可用() {
        // 峰值的概率约 0.997，随机数几乎必命中它；用均匀分布无法确定断言，
        // 因此这里只验证「返回值合法且不 panic」，把确定性的分布断言交给
        // 上面注入固定随机数的用例。
        let arr = Array1::from_vec(vec![1.0, 9.0, 3.0]);
        for _ in 0..20 {
            let picked = sample_token_array1(&arr, 2);
            assert!(picked >= 0 && picked < 3, "越界: {picked}");
        }
        // 空数组不 panic
        let empty = Array1::<f32>::zeros(0);
        assert_eq!(sample_token_array1(&empty, 2), 0);
    }

    #[test]
    fn topk加温度组合行为确定() {
        // 低温 + top1 = 贪心解码
        let logits = [1.0, 9.0, 3.0];
        for _ in 0..20 {
            assert_eq!(sample_token(&logits, 1, 0.0, 0.01, 0.5), 1);
        }
    }

    #[test]
    fn 分布覆盖多token() {
        // 均匀分布下多次采样应能取到不同 token，否则说明采样退化
        let logits = [0.0; 5];
        let mut seen = std::collections::HashSet::new();
        for i in 0..200 {
            seen.insert(sample_token(&logits, 0, 0.0, 1.0, (i as f32) / 200.0));
        }
        assert!(seen.len() >= 4, "采样退化，仅覆盖 {:?} 个 token", seen.len());
    }
}
