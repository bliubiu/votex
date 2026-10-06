//! IndexTTS-2.5 DSP（dsp.py 忠实移植，全部纯函数、无文件依赖）
//!
//! # 数值契约
//!
//! - `resample_sinc_hann`：torchaudio `sinc_interp_hann` 的逐行移植
//!   （f64 核 → f32 卷积）。这是 P1-14 遗留的重采样抗混叠问题的**正确参照实现**。
//! - `mel_spectrogram`：s2mel `audio.py` 移植（center=False、reflect pad
//!   `(n_fft-hop)/2`、periodic hann、`mag=sqrt(re²+im²+1e-9)`、librosa slaney
//!   mel 基、`log(clip(x, 1e-5))`）。numpy rfft 在 f64 域计算，本模块同样全程
//!   f64（窗后帧、频谱、mel 矩阵乘），仅最终 log 结果裁剪到 f32。
//! - `kaldi_fbank`：kaldi-native-fbank 语义（povey 窗 400/160@16k、预加重 0.97、
//!   去直流、round_to_power_of_two→512 点 FFT、**功率谱不除 N**、kaldi mel
//!   80 bins（low_freq=20，Mel=1127·ln(1+f/700)）、log 下限 f32 epsilon）。
//!   与 #44 的 `features.rs`（HF SeamlessM4T 路径）是**两套不同算法**，勿混用。
//! - `campplus_fbank`：kaldi fbank + 整句 mean 归一化。

use anyhow::{bail, Result};
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

// ===================== 1. sinc-Hann 重采样 =====================

/// torchaudio.functional.resample(sinc_interp_hann) 的精确移植
///
/// # 参数
/// - `y`: 输入信号（f32）
/// - `orig_freq` / `new_freq`: 采样率（内部先约去 gcd）
/// - `lowpass_filter_width`: 核半宽参数（默认 6）
/// - `rolloff`: 低通滚降（默认 0.99）
pub fn resample_sinc_hann(
    y: &[f32],
    orig_freq: usize,
    new_freq: usize,
    lowpass_filter_width: usize,
    rolloff: f64,
) -> Result<Vec<f32>> {
    if orig_freq == 0 || new_freq == 0 {
        bail!("采样率不能为 0");
    }
    if orig_freq == new_freq {
        return Ok(y.to_vec());
    }
    let g = gcd(orig_freq, new_freq);
    let orig = orig_freq / g;
    let new = new_freq / g;
    let base_freq = orig.min(new) as f64 * rolloff;
    let width = ((lowpass_filter_width as f64 * orig as f64 / base_freq).ceil()) as usize;

    // idx = arange(-width, width + orig) / orig  （f64）
    let k = 2 * width + orig;
    let idx: Vec<f64> = (-(width as i64)..(width as i64 + orig as i64))
        .map(|v| v as f64 / orig as f64)
        .collect();

    // t = (arange(0, -new, -1) / new)[:, None] + idx[None, :]  → [new, k]
    // arange(0,-new,-1)[j] = -j，即 t[j][i] = idx[i] - j/new
    let mut kernels = vec![0.0f32; new * k];
    for j in 0..new {
        let j0 = j as f64 / new as f64;
        let row = &mut kernels[j * k..(j + 1) * k];
        for (i, &iv) in idx.iter().enumerate() {
            // t = (idx - j/new) * base_freq，clip 到 ±lowpass_filter_width
            let mut t = (iv - j0) * base_freq;
            t = t.clamp(-(lowpass_filter_width as f64), lowpass_filter_width as f64);
            // window = cos(t * pi / lfw / 2)^2
            let window = (t * std::f64::consts::PI / lowpass_filter_width as f64 / 2.0)
                .cos()
                .powi(2);
            // t *= pi；sinc = sin(t)/t（t==0 → 1）
            let t2 = t * std::f64::consts::PI;
            let sinc = if t2 == 0.0 { 1.0 } else { t2.sin() / t2 };
            row[i] = (sinc * window * (base_freq / orig as f64)) as f32;
        }
    }

    // 零填充 + 步长 orig 取帧 + 矩阵乘（f32 @ f32^T）
    let mut yp = vec![0.0f32; y.len() + 2 * width + orig];
    yp[width..width + y.len()].copy_from_slice(y);
    let n_pos = if yp.len() >= k { (yp.len() - k) / orig + 1 } else { 0 };
    let mut out = vec![0.0f32; n_pos * new];
    for p in 0..n_pos {
        let frame = &yp[p * orig..p * orig + k];
        let out_row = &mut out[p * new..(p + 1) * new];
        for (j, o) in out_row.iter_mut().enumerate() {
            let ker = &kernels[j * k..(j + 1) * k];
            let mut acc = 0.0f32;
            for (&s, &w) in frame.iter().zip(ker) {
                acc += s * w;
            }
            *o = acc;
        }
    }
    // target_length = ceil(new * len / orig)
    let target_length = (new * y.len()).div_ceil(orig);
    Ok(out.into_iter().take(target_length).collect())
}

fn gcd(a: usize, b: usize) -> usize {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

// ===================== 2. mel 频谱（s2mel audio.py） =====================

/// MEL_PARAMS（dsp.py 固定参数）
pub const MEL_SAMPLING_RATE: usize = 22050;
const MEL_N_FFT: usize = 1024;
const MEL_HOP: usize = 256;
const MEL_WIN: usize = 1024;
const MEL_NUM_BINS: usize = 80;

/// periodic hann：`torch.hann_window(win, periodic=True)`（f64 计算后裁 f32）
fn periodic_hann(win_size: usize) -> Vec<f32> {
    (0..win_size)
        .map(|n| {
            (0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / win_size as f64).cos()) as f32
        })
        .collect()
}

/// librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=None)
/// 的闭式构造（htk=False → slaney mel 刻度，norm='slaney'）
pub fn librosa_mel_basis(sr: usize, n_fft: usize, n_mels: usize, fmin: f64, fmax: Option<f64>) -> Vec<f64> {
    let n_freqs = n_fft / 2 + 1;
    let fftfreqs: Vec<f64> = (0..n_freqs).map(|i| i as f64 * sr as f64 / n_fft as f64).collect();
    let top = fmax.unwrap_or(sr as f64 / 2.0);
    let mel_f = mel_frequencies(n_mels + 2, fmin, top);

    let fdiff: Vec<f64> = mel_f.windows(2).map(|w| w[1] - w[0]).collect();
    // ramps[i][j] = mel_f[i] - fftfreqs[j]
    let mut weights = vec![0.0f64; n_mels * n_freqs];
    for i in 0..n_mels {
        for (j, &f) in fftfreqs.iter().enumerate() {
            let lower = -(mel_f[i] - f) / fdiff[i];
            let upper = (mel_f[i + 2] - f) / fdiff[i + 1];
            weights[i * n_freqs + j] = 0.0f64.max(lower.min(upper));
        }
    }
    // slaney 归一化：2 / (mel_f[i+2] - mel_f[i])
    for i in 0..n_mels {
        let enorm = 2.0 / (mel_f[i + 2] - mel_f[i]);
        for w in &mut weights[i * n_freqs..(i + 1) * n_freqs] {
            *w *= enorm;
        }
    }
    weights
}

/// librosa `mel_frequencies(htk=False)`：在 **mel 空间**等距取点后反变换回 Hz
/// （slaney 刻度：`mel = f/(200/3)`（<1kHz）、`15 + ln(f/1000)/logstep`（≥1kHz））
fn mel_frequencies(n_mels: usize, fmin: f64, fmax: f64) -> Vec<f64> {
    let f_sp = 200.0 / 3.0;
    let min_log_hz = 1000.0f64;
    let min_log_mel = min_log_hz / f_sp; // 15.0
    let logstep = (6.4f64).ln() / 27.0;
    let hz_to_mel = |f: f64| if f < min_log_hz { f / f_sp } else { min_log_mel + (f / min_log_hz).ln() / logstep };
    let mel_to_hz = |m: f64| if m < min_log_mel { f_sp * m } else { min_log_hz * (logstep * (m - min_log_mel)).exp() };
    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);
    (0..n_mels)
        .map(|i| mel_to_hz(mel_min + (mel_max - mel_min) * i as f64 / (n_mels - 1) as f64))
        .collect()
}

/// numpy `np.pad(mode="reflect")` 的单边索引折叠
fn reflect_index(t: i64, len: usize) -> usize {
    let len = len as i64;
    let mut t = t;
    // 折叠周期 2*(len-1)，对短信号同样成立
    let period = 2 * (len - 1);
    t = t.rem_euclid(period);
    if t >= len {
        t = period - t;
    }
    t as usize
}

/// s2mel `mel_spectrogram`：f32 [T] → log-mel [num_mels, frames]（行主序）
///
/// 参数固定 MEL_PARAMS：22050 Hz / n_fft=1024 / hop=256 / win=1024 / 80 mel。
pub fn mel_spectrogram(y: &[f32]) -> Result<Vec<f32>> {
    if y.len() < 2 {
        bail!("音频过短（< 2 样本），无法 reflect padding");
    }
    let pad = (MEL_N_FFT - MEL_HOP) / 2;
    // reflect pad（f32）
    let padded_len = y.len() + 2 * pad;
    let mut padded = vec![0.0f32; padded_len];
    for (i, p) in padded.iter_mut().enumerate().take(padded_len) {
        let src = reflect_index(i as i64 - pad as i64, y.len());
        *p = y[src];
    }
    let n_frames = 1 + (padded_len - MEL_N_FFT) / MEL_HOP;

    // 帧 × periodic_hann（f32）→ f64 rfft → mag → mel（f64）→ log → f32
    let window = periodic_hann(MEL_WIN);
    let basis = librosa_mel_basis(MEL_SAMPLING_RATE, MEL_N_FFT, MEL_NUM_BINS, 0.0, None);
    let n_freqs = MEL_N_FFT / 2 + 1;

    let mut planner = RealFftPlanner::<f64>::new();
    let fft: Arc<dyn RealToComplex<f64>> = planner.plan_fft_forward(MEL_N_FFT);
    let mut input = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();

    let mut out = vec![0.0f32; MEL_NUM_BINS * n_frames];
    for f in 0..n_frames {
        let base = f * MEL_HOP;
        for (i, slot) in input.iter_mut().enumerate() {
            *slot = padded[base + i] as f64 * window[i] as f64;
        }
        fft.process(&mut input, &mut spectrum)
            .map_err(|e| anyhow::anyhow!("rfft 失败: {e}"))?;
        // mag = sqrt(re² + im² + 1e-9)
        let mut mag = [0.0f64; MEL_N_FFT / 2 + 1];
        for (i, c) in spectrum.iter().enumerate() {
            mag[i] = (c.re * c.re + c.im * c.im + 1e-9).sqrt();
        }
        // mel = basis @ mag（f64）
        for m in 0..MEL_NUM_BINS {
            let row = &basis[m * n_freqs..(m + 1) * n_freqs];
            let mut acc = 0.0f64;
            for (&w, &v) in row.iter().zip(&mag) {
                acc += w * v;
            }
            let v = acc.max(1e-5).ln();
            out[m * n_frames + f] = v as f32;
        }
    }
    Ok(out)
}

// ===================== 3. kaldi fbank（knf 语义） =====================

const KALDI_FRAME_LENGTH: usize = 400; // 25ms @16k
const KALDI_FRAME_SHIFT: usize = 160; // 10ms @16k
const KALDI_PREEMPH: f32 = 0.97;
const KALDI_FFT_SIZE: usize = 512; // round_to_power_of_two(400)
const KALDI_NUM_BINS: usize = 80;
const KALDI_LOW_FREQ: f64 = 20.0;

/// kaldi povey 窗：`(0.5 - 0.5 cos(2πn/(N-1)))^0.85`
fn povey_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let a = 2.0 * std::f64::consts::PI * i as f64 / (n as f64 - 1.0);
            (0.5 - 0.5 * a.cos()).powf(0.85) as f32
        })
        .collect()
}

/// kaldi mel 滤波器组（knf `InitKaldiMelBanks` 忠实移植，全 f32 语义）
///
/// 三角形端点在 mel 域等距（`delta=(mel_high-mel_low)/(bins+1)`），
/// 权重插值也在 **mel 域**进行（kaldi 特性，与 librosa 的 Hz 域不同）：
/// `mel = MelScale(freq)`，开区间 `(left, right)`，`mel<=center` 走上升沿。
/// `MelScale(f) = 1127·ln(1+f/700)`（f32）。
fn kaldi_mel_banks(samp_freq: f32, nfft: usize, num_bins: usize, low_freq: f32) -> Vec<f32> {
    let n_freqs = nfft / 2 + 1;
    let mel = |f: f32| 1127.0f32 * (1.0f32 + f / 700.0f32).ln();
    let high_freq = samp_freq / 2.0;
    let mel_low = mel(low_freq);
    let mel_high = mel(high_freq);
    let delta = (mel_high - mel_low) / (num_bins as f32 + 1.0);
    // num_bins + 2 个等距 mel 点（f32，与 knf 一致）
    let points: Vec<f32> = (0..num_bins + 2)
        .map(|i| mel_low + delta * i as f32)
        .collect();

    let fft_bin_width = samp_freq / nfft as f32;
    let mut banks = vec![0.0f32; num_bins * n_freqs];
    for b in 0..num_bins {
        let (left, center, right) = (points[b], points[b + 1], points[b + 2]);
        for j in 0..n_freqs {
            let m = mel(fft_bin_width * j as f32);
            if m > left && m < right {
                let w = if m <= center {
                    (m - left) / (center - left)
                } else {
                    (right - m) / (right - center)
                };
                banks[b * n_freqs + j] = w;
            }
        }
    }
    banks
}

/// kaldi fbank（knf 语义）：16k f32 [-1,1] → log-mel [frames, 80]
///
/// 帧提取 `snip_edges=True`（仅整帧）；去直流 → 预加重（首样本 ×(1-0.97)）→
/// povey 加窗 → 512 点 rfft → 功率谱（**不除 N**）→ mel → `ln(max(x, f32::EPSILON))`。
pub fn kaldi_fbank(y: &[f32], samp_freq: usize, num_mel_bins: usize) -> Result<Vec<f32>> {
    // 本实现的常量按 16k 推导；采样率变化时显式拒绝（当前管线只用 16k）
    if samp_freq != 16000 {
        bail!("kaldi_fbank 仅支持 16 kHz（当前 {samp_freq}）");
    }
    if num_mel_bins != KALDI_NUM_BINS {
        bail!("kaldi_fbank 仅支持 80 mel bins（当前 {num_mel_bins}）");
    }
    if y.len() < KALDI_FRAME_LENGTH {
        bail!("音频短于单帧（{} 样本 < {}）", y.len(), KALDI_FRAME_LENGTH);
    }
    let n_frames = (y.len() - KALDI_FRAME_LENGTH) / KALDI_FRAME_SHIFT + 1;
    let window = povey_window(KALDI_FRAME_LENGTH);
    let banks = kaldi_mel_banks(samp_freq as f32, KALDI_FFT_SIZE, num_mel_bins, KALDI_LOW_FREQ as f32);
    let n_freqs = KALDI_FFT_SIZE / 2 + 1;

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(KALDI_FFT_SIZE);
    let mut input = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();

    let mut out = vec![0.0f32; n_frames * num_mel_bins];
    let mut frame = vec![0.0f32; KALDI_FRAME_LENGTH];
    for f in 0..n_frames {
        let base = f * KALDI_FRAME_SHIFT;
        frame.copy_from_slice(&y[base..base + KALDI_FRAME_LENGTH]);
        // 去直流（knf frame_opts.remove_dc_offset 默认 true）
        let mean = frame.iter().sum::<f32>() / KALDI_FRAME_LENGTH as f32;
        for v in &mut frame {
            *v -= mean;
        }
        // 预加重：逆序原地 f[i] -= 0.97*f[i-1]；首样本 ×(1-0.97)
        for i in (1..KALDI_FRAME_LENGTH).rev() {
            frame[i] -= KALDI_PREEMPH * frame[i - 1];
        }
        frame[0] *= 1.0 - KALDI_PREEMPH;
        // 加窗
        for (v, &w) in frame.iter_mut().zip(&window) {
            *v *= w;
        }
        // 512 点 rfft（400 帧尾部补零）
        input[..KALDI_FRAME_LENGTH].copy_from_slice(&frame);
        input[KALDI_FRAME_LENGTH..].fill(0.0);
        fft.process(&mut input, &mut spectrum)
            .map_err(|e| anyhow::anyhow!("rfft 失败: {e}"))?;
        // kaldi 功率谱：re²+im²（无 1/N 归一化）
        let mut power = [0.0f32; KALDI_FFT_SIZE / 2 + 1];
        for (i, c) in spectrum.iter().enumerate() {
            power[i] = c.re * c.re + c.im * c.im;
        }
        // mel 能量 + log（下限 f32 epsilon）
        for b in 0..num_mel_bins {
            let row = &banks[b * n_freqs..(b + 1) * n_freqs];
            let mut acc = 0.0f32;
            for (&w, &p) in row.iter().zip(&power) {
                acc += w * p;
            }
            out[f * num_mel_bins + b] = acc.max(f32::EPSILON).ln();
        }
    }
    Ok(out)
}

/// campplus fbank：kaldi fbank + 整句 mean 归一化 → [1, T, 80]
pub fn campplus_fbank(audio_16k: &[f32]) -> Result<Vec<f32>> {
    let feat = kaldi_fbank(audio_16k, 16000, 80)?;
    let n_frames = feat.len() / 80;
    // 逐 mel 维（列）减均值
    let mut col_sum = vec![0.0f32; 80];
    for frame in feat.chunks_exact(80) {
        for (b, &v) in frame.iter().enumerate() {
            col_sum[b] += v;
        }
    }
    let mut out = feat;
    for frame in out.chunks_exact_mut(80) {
        for (b, v) in frame.iter_mut().enumerate() {
            *v -= col_sum[b] / n_frames as f32;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 金标目录缺失时相关对拍测试直接跳过（金标由脚本生成，不入库）
    fn 金标缺失时跳过() -> bool {
        let dir = crate::shared::workspace_paths::WorkspacePaths::workspace_root()
            .join("tmp/indextts25_refs");
        !dir.exists()
    }

    /// 读取金标 npy（f32，float32 little-endian）
    pub(crate) fn read_ref_npy(rel: &str) -> (Vec<usize>, Vec<f32>) {
        let path = crate::shared::workspace_paths::WorkspacePaths::workspace_root()
            .join("tmp/indextts25_refs")
            .join(rel);
        let data = std::fs::read(&path).unwrap_or_else(|e| panic!("金标不存在 {:?}: {}", path, e));
        read_npy_bytes(&data).unwrap_or_else(|e| panic!("{rel}: {e}"))
    }

    /// 最小 npy 解析（f32）
    pub(crate) fn read_npy_bytes(data: &[u8]) -> Result<(Vec<usize>, Vec<f32>)> {
        if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
            bail!("无效 NPY 魔数");
        }
        let ver = data[6];
        let (hdr_len, hdr_off) = match ver {
            1 => (u16::from_le_bytes([data[8], data[9]]) as usize, 10usize),
            2 | 3 => (
                u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize,
                12usize,
            ),
            v => bail!("不支持 NPY v{v}"),
        };
        let hdr = std::str::from_utf8(&data[hdr_off..hdr_off + hdr_len])?;
        let shape_part = hdr
            .split("'shape': (")
            .nth(1)
            .and_then(|s| s.split(')').next())
            .ok_or_else(|| anyhow::anyhow!("无 shape"))?;
        let shape: Vec<usize> = shape_part
            .split(',')
            .map(|s| s.trim().trim_end_matches(','))
            .filter(|s| !s.is_empty())
            .filter_map(|s| s.parse().ok())
            .collect();
        let data_start = hdr_off + hdr_len;
        let pad = (4 - data_start % 4) % 4; // numpy 8 对齐：npy v1 头 64 对齐——用实际数据起点
        let data_start = data_start + pad;
        let floats: Vec<f32> = data[data_start..]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok((shape, floats))
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "长度不一致: {} vs {}", a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn 重采样_与torchaudio金标对齐() {
        if 金标缺失时跳过() {
            eprintln!("⚠ 金标目录 tmp/indextts25_refs 不存在，跳过（由金标生成脚本产出）");
            return;
        }
        let (shape, input) = read_ref_npy("resample_input.npy");
        let (_, want) = read_ref_npy("resample_22050_to_16000.npy");
        assert_eq!(shape.len(), 1);
        let got = resample_sinc_hann(&input, 22050, 16000, 6, 0.99).unwrap();
        let d = max_diff(&got, &want);
        println!("resample max_diff = {d}");
        assert!(d < 1e-4, "resample 偏差过大: {d}");
    }

    #[test]
    fn 重采样_等采样率恒等() {
        let y = vec![0.1f32, -0.2, 0.3];
        assert_eq!(resample_sinc_hann(&y, 16000, 16000, 6, 0.99).unwrap(), y);
    }

    #[test]
    fn mel频谱_与s2mel金标对齐() {
        if 金标缺失时跳过() {
            eprintln!("⚠ 金标目录 tmp/indextts25_refs 不存在，跳过（由金标生成脚本产出）");
            return;
        }
        let (shape, input) = read_ref_npy("mel_input.npy");
        let (wshape, want) = read_ref_npy("mel_output.npy");
        assert_eq!(shape.len(), 1);
        assert_eq!(wshape, vec![80, want.len() / 80]);
        let got = mel_spectrogram(&input).unwrap();
        let d = max_diff(&got, &want);
        println!("mel max_diff = {d}");
        assert!(d < 1e-4, "mel 偏差过大: {d}");
    }

    #[test]
    fn kaldi_fbank_与knf金标对齐() {
        if 金标缺失时跳过() {
            eprintln!("⚠ 金标目录 tmp/indextts25_refs 不存在，跳过（由金标生成脚本产出）");
            return;
        }
        let (shape, input) = read_ref_npy("kaldi_input.npy");
        let (wshape, want) = read_ref_npy("kaldi_fbank_output.npy");
        assert_eq!(shape.len(), 1);
        assert_eq!(wshape[1], 80);
        let got = kaldi_fbank(&input, 16000, 80).unwrap();
        assert_eq!(got.len(), want.len(), "帧数不一致");
        let d = max_diff(&got, &want);
        println!("kaldi fbank max_diff = {d}");
        // 容差 5e-4：knf 用 kissfft（f32），Rust 用 realfft（f32），同为正确的
        // f32 FFT 但舍入路径不同；低能量 mel bin 的相对噪声在 log 域 ~3e-4。
        // 语义已逐行对齐 knf 源码（InitKaldiMelBanks / feature-window）。
        assert!(d < 5e-4, "kaldi fbank 偏差过大: {d}");
    }

    #[test]
    fn campplus_fbank_均值归一化后与金标对齐() {
        if 金标缺失时跳过() {
            eprintln!("⚠ 金标目录 tmp/indextts25_refs 不存在，跳过（由金标生成脚本产出）");
            return;
        }
        let (_, input) = read_ref_npy("kaldi_input.npy");
        let (_, want) = read_ref_npy("campplus_fbank_output.npy");
        let got = campplus_fbank(&input).unwrap();
        let d = max_diff(&got, &want);
        println!("campplus fbank max_diff = {d}");
        // 同上：f32 FFT 舍入噪声（见 kaldi_fbank 测试注释）
        assert!(d < 5e-4, "campplus fbank 偏差过大: {d}");
    }

    #[test]
    fn reflect_index_折叠语义与numpy一致() {
        // len=5: 索引 -1 → 1, -2 → 2, 5 → 3, 6 → 2
        assert_eq!(reflect_index(-1, 5), 1);
        assert_eq!(reflect_index(-2, 5), 2);
        assert_eq!(reflect_index(5, 5), 3);
        assert_eq!(reflect_index(6, 5), 2);
        assert_eq!(reflect_index(0, 5), 0);
    }

    #[test]
    fn 常量_预加重系数与kaldi一致() {
        assert!((KALDI_PREEMPH - 0.97).abs() < f32::EPSILON);
    }
}
