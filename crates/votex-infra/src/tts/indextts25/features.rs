//! IndexTTS-2.5 w2v-bert 输入特征提取（SeamlessM4T 特征的 Rust 移植）。
//!
//! 蓝本：transformers 5.18.0 `SeamlessM4TFeatureExtractor`（numpy 路径，
//! `audio_utils.spectrogram` / `mel_filter_bank` / `window_function`）+
//! `index_tts_2_5_onnx/core/features.py::W2VFeatures`。
//!
//! 处理链（对 16kHz mono f32 波形）：
//! 1. Kaldi 16-bit 合规缩放：波形 ×2^15（f32 → f64 计算）
//! 2. 分帧 400/160（严格截断，等价 kaldi snip_edges），每帧：去直流 → 预加重 0.97
//!    （首样本 ×(1-0.97)）→ povey 对称窗（np.hanning(400)^0.85）→ rfft 512 → |X|²
//!    （FFT 结果先裁剪到 f32 精度再取模，对齐 numpy complex64 存储语义）
//! 3. kaldi mel 滤波器组 257→80（mel 空间三角化、20..8000Hz、无归一化系数）、
//!    下限 1.192092955078125e-07、自然对数 → [T,80]
//! 4. 逐 mel 维零均值单位方差归一（方差 ddof=1，+1e-7）
//! 5. 帧数 pad 到偶数（填充值 1.0，来自 preprocessor_config.json padding_value），
//!    attention_mask 全 1
//! 6. stride=2 堆叠：相邻两帧 80 维 → 160 维，帧数减半；mask 取奇数索引
//!
//! 金标对拍：`tmp/indextts25_refs/`（make_w2v_refs.py 生成）。

use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use realfft::{RealFftPlanner, RealToComplex};

// —— SeamlessM4T（w2v-bert-2.0）预处理常量（preprocessor_config.json + transformers 源码）——
const FRAME_LENGTH: usize = 400; // 25ms @16k
const HOP_LENGTH: usize = 160; // 10ms @16k
const FFT_LENGTH: usize = 512;
const N_FREQ_BINS: usize = 257; // FFT_LENGTH / 2 + 1
const N_MELS: usize = 80;
const MIN_FREQ: f64 = 20.0;
const MAX_FREQ: f64 = 8000.0; // 16000 / 2
const PREEMPHASIS: f64 = 0.97;
/// Kaldi float32 最小正规数（transformers `_extract_fbank_features` 硬编码）。
const MEL_FLOOR: f64 = 1.192_092_955_078_125e-07;
/// 逐维归一化的方差稳定项。
const NORM_EPS: f64 = 1e-7;
/// 帧数 pad 的填充值（注意：preprocessor_config.json 中 padding_value=1，非默认 0.0）。
const PAD_VALUE: f64 = 1.0;
/// 相邻两帧 80 维堆叠为 160 维，帧数减半。
const STRIDE: usize = 2;

/// 特征维度（stride 堆叠后）。
pub const N_FEATURES: usize = N_MELS * STRIDE;

/// w2v-bert 输入特征提取器（无外部文件依赖：窗函数与 mel 滤波器组均闭式构造）。
pub struct W2vBertFeatures {
    mel_filters: Vec<[f64; N_MELS]>, // [257][80]
    window: Vec<f64>,                // povey 对称窗，长度 FRAME_LENGTH
    fft: Arc<dyn RealToComplex<f64>>,
}

/// `extract` 的输出（行优先扁平数组）。
pub struct W2vFeaturesOutput {
    /// 堆叠后的输入特征，行优先 [te, 160]。
    pub input_features: Vec<f32>,
    /// 注意力掩码 [te]（单条音频：全 1）。
    pub attention_mask: Vec<i64>,
    /// 堆叠后的帧数。
    pub te: usize,
}

impl Default for W2vBertFeatures {
    fn default() -> Self {
        Self::new()
    }
}

/// kaldi mel 尺度：`1127 * ln(1 + f/700)`。
fn hz_to_mel_kaldi(freq: f64) -> f64 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// kaldi mel 滤波器组 [257 频点][80 滤波器]。
///
/// 对齐 `mel_filter_bank(num_frequency_bins=257, num_mel_filters=80, min=20, max=8000,
/// norm=None, mel_scale="kaldi", triangularize_in_mel_space=True)`：
/// 三角形在 mel 空间等距切分（82 个端点），FFT 频点也转到 mel 空间后求斜率。
fn build_mel_filters() -> Vec<[f64; N_MELS]> {
    let mel_min = hz_to_mel_kaldi(MIN_FREQ);
    let mel_max = hz_to_mel_kaldi(MAX_FREQ);
    let n_points = N_MELS + 2;
    let step = (mel_max - mel_min) / (n_points - 1) as f64;
    let mut mel_freqs: Vec<f64> = (0..n_points).map(|i| mel_min + i as f64 * step).collect();
    mel_freqs[n_points - 1] = mel_max; // np.linspace endpoint 精确语义

    // FFT 频点（mel 空间）：bin_width = 16000 / (256*2) = 31.25 Hz
    let fft_bin_width = 16000.0 / (((N_FREQ_BINS - 1) * 2) as f64);
    let fft_freqs_mel: Vec<f64> = (0..N_FREQ_BINS)
        .map(|k| hz_to_mel_kaldi(fft_bin_width * k as f64))
        .collect();

    let mut filters = vec![[0.0f64; N_MELS]; N_FREQ_BINS];
    for (i, item) in filters.iter_mut().enumerate() {
        for (m, weight) in item.iter_mut().enumerate() {
            let down = (fft_freqs_mel[i] - mel_freqs[m]) / (mel_freqs[m + 1] - mel_freqs[m]);
            let up = (mel_freqs[m + 2] - fft_freqs_mel[i]) / (mel_freqs[m + 2] - mel_freqs[m + 1]);
            *weight = down.min(up).max(0.0);
        }
    }
    filters
}

/// povey 对称窗：`np.hanning(400)^0.85 = (0.5 - 0.5*cos(2πn/399))^0.85`。
fn build_povey_window() -> Vec<f64> {
    (0..FRAME_LENGTH)
        .map(|n| {
            let h = 0.5
                - 0.5 * (std::f64::consts::TAU * n as f64 / (FRAME_LENGTH - 1) as f64).cos();
            h.powf(0.85)
        })
        .collect()
}

impl W2vBertFeatures {
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f64>::new();
        Self {
            mel_filters: build_mel_filters(),
            window: build_povey_window(),
            fft: planner.plan_fft_forward(FFT_LENGTH),
        }
    }

    /// 16kHz mono 波形 → (input_features [te,160] f32, attention_mask [te] i64)。
    pub fn extract(&self, audio_16k: &[f32]) -> Result<W2vFeaturesOutput> {
        let n_samples = audio_16k.len();
        if n_samples < FRAME_LENGTH {
            bail!("音频过短：{n_samples} 样本 < 帧长 {FRAME_LENGTH}");
        }
        // Kaldi 16-bit 合规缩放（f32 ×2^15 → f64）
        let scaled: Vec<f64> = audio_16k
            .iter()
            .map(|s| (*s as f64) * 32768.0)
            .collect();
        let n_frames = 1 + (n_samples - FRAME_LENGTH) / HOP_LENGTH;
        if n_frames < 2 {
            // ddof=1 方差至少需要 2 帧
            bail!("音频过短：仅 {n_frames} 帧，逐维归一化（ddof=1）需要至少 2 帧");
        }

        // 分帧频谱（f64；FFT 结果先裁剪到 f32 对齐 numpy complex64 存储）
        let mut power = vec![0.0f64; n_frames * N_FREQ_BINS];
        let mut input = self.fft.make_input_vec();
        let mut spectrum = self.fft.make_output_vec();
        for (f, item) in power.chunks_mut(N_FREQ_BINS).enumerate() {
            let base = f * HOP_LENGTH;
            // numpy：buffer = zeros(512) 后仅前 400 点为帧内容，其余保持零
            input.fill(0.0);
            input[..FRAME_LENGTH].copy_from_slice(&scaled[base..base + FRAME_LENGTH]);
            // 去直流（先于预加重）
            let mean = input[..FRAME_LENGTH].iter().sum::<f64>() / FRAME_LENGTH as f64;
            for v in &mut input[..FRAME_LENGTH] {
                *v -= mean;
            }
            // 预加重：new[i] = old[i] - 0.97*old[i-1]（按旧值语义，逆序原地），
            // 首样本 ×(1-0.97)
            for i in (1..FRAME_LENGTH).rev() {
                input[i] -= PREEMPHASIS * input[i - 1];
            }
            input[0] *= 1.0 - PREEMPHASIS;
            // 加窗
            for (v, w) in input.iter_mut().zip(&self.window) {
                *v *= w;
            }
            self.fft
                .process(&mut input, &mut spectrum)
                .map_err(|e| anyhow!("FFT 执行失败: {e}"))?;
            for (k, spec) in spectrum.iter().enumerate() {
                let re = spec.re as f32 as f64;
                let im = spec.im as f32 as f64;
                item[k] = re * re + im * im;
            }
        }

        // mel 滤波 + 下限 + 对数 → [n_frames][80]
        let mut mel = vec![0.0f64; n_frames * N_MELS];
        for (f, row) in power.chunks(N_FREQ_BINS).enumerate() {
            let out = &mut mel[f * N_MELS..(f + 1) * N_MELS];
            for (k, &p) in row.iter().enumerate() {
                let filter = &self.mel_filters[k];
                for (m, &w) in filter.iter().enumerate() {
                    out[m] += w * p;
                }
            }
            for v in out.iter_mut() {
                *v = (*v).max(MEL_FLOOR).ln();
            }
        }

        // 逐 mel 维零均值单位方差归一（ddof=1，+1e-7）
        for m in 0..N_MELS {
            let col = (0..n_frames).map(|f| mel[f * N_MELS + m]);
            let mean = col.clone().sum::<f64>() / n_frames as f64;
            let var = col.map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n_frames - 1) as f64;
            let denom = (var + NORM_EPS).sqrt();
            for f in 0..n_frames {
                mel[f * N_MELS + m] = (mel[f * N_MELS + m] - mean) / denom;
            }
        }

        // 帧数 pad 到偶数：特征填充值 1.0（padding_value），mask 的 pad 位置为 0
        let padded_frames = n_frames.div_ceil(STRIDE) * STRIDE;
        let mut padded = vec![PAD_VALUE; padded_frames * N_MELS];
        padded[..mel.len()].copy_from_slice(&mel);
        let mut frame_mask = vec![1i64; n_frames];
        frame_mask.resize(padded_frames, 0);

        // stride=2 堆叠：[padded, 80] → [padded/2, 160]；mask 取奇数索引
        let te = padded_frames / STRIDE;
        let mut input_features = vec![0.0f32; te * N_FEATURES];
        for f in 0..padded_frames {
            for m in 0..N_MELS {
                input_features[(f / STRIDE) * N_FEATURES + (f % STRIDE) * N_MELS + m] =
                    padded[f * N_MELS + m] as f32;
            }
        }
        let attention_mask: Vec<i64> = (0..te).map(|t2| frame_mask[2 * t2 + 1]).collect();

        Ok(W2vFeaturesOutput {
            input_features,
            attention_mask,
            te,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// 金标目录（tmp/indextts25_refs/，make_w2v_refs.py 生成）。
    fn refs_dir() -> PathBuf {
        crate::shared::workspace_paths::WorkspacePaths::workspace_root()
            .join("tmp")
            .join("indextts25_refs")
    }

    #[test]
    fn w2v特征_与python金标对齐() {
        let dir = refs_dir();
        if !dir.is_dir() {
            eprintln!("跳过：金标目录不存在 {}", dir.display());
            return;
        }
        let feats = W2vBertFeatures::new();
        for name in ["w2v_15s", "w2v_short", "w2v_odd"] {
            let wave_path = dir.join(format!("{name}.wave.f32"));
            if !wave_path.is_file() {
                eprintln!("跳过：缺少金标 {name}");
                continue;
            }
            let wave_bytes = std::fs::read(&wave_path).unwrap();
            let wave: Vec<f32> = wave_bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let out = feats.extract(&wave).unwrap();

            let expect_feat = std::fs::read(dir.join(format!("{name}.input_features.f32")))
                .unwrap();
            let expect_feat: Vec<f32> = expect_feat
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let expect_mask = std::fs::read(dir.join(format!("{name}.attention_mask.i64")))
                .unwrap();
            let expect_mask: Vec<i64> = expect_mask
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect();

            assert_eq!(out.te * N_FEATURES, expect_feat.len(), "{name} 维度不一致");
            let max_diff = out
                .input_features
                .iter()
                .zip(&expect_feat)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_diff < 1e-4,
                "{name} 特征最大偏差 {max_diff} 超过容差 1e-4"
            );
            assert_eq!(out.attention_mask, expect_mask, "{name} mask 不一致");
            eprintln!("{name}: te={} max_abs_diff={max_diff:.3e}", out.te);
        }
    }
}
