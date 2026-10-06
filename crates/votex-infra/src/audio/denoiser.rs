//! 音频降噪模块
//!
//! 纯 Rust 实现，无外部原生依赖。
//! - 低级: 噪声门 (Noise Gate) — 基于 RMS 的静音段压制
//! - 中级: 频谱减法 (Spectral Subtraction) — FFT 域降噪
//! - 高级: 噪声门 + 频谱减法叠加
//!
//! 频谱减法使用 realfft 库进行实数 FFT 变换，
//! 适合语音降噪场景（噪声谱变化缓慢，语音谱稀疏）。

use anyhow::Result;
use realfft::RealFftPlanner;
use votex_domain::shared::value_object::AudioData;

/// 降噪配置
#[derive(Debug, Clone)]
pub struct DenoiseConfig {
    /// 噪声门 RMS 阈值 (0.0 ~ 1.0)，低于此值的窗口被静音
    pub gate_threshold: f32,
    /// FFT 窗口大小（必须为 2 的幂，建议 512/1024/2048）
    pub fft_size: usize,
    /// 噪声估计遗忘因子 (0.0 ~ 1.0)，越大则噪声估计越平滑
    pub alpha: f32,
    /// 频谱减法降噪强度 (0.0 ~ 1.0)，越大降噪越强
    pub subtraction_factor: f32,
    /// 噪声底噪保护 (dB)，防止过度降噪导致音乐性噪声
    pub noise_floor: f32,
    /// 窗口重叠比例 (0.0 ~ 1.0)
    pub overlap: f32,
}

impl Default for DenoiseConfig {
    fn default() -> Self {
        Self {
            gate_threshold: 0.015,
            fft_size: 1024,
            alpha: 0.90,
            subtraction_factor: 0.85,
            noise_floor: 0.01,
            overlap: 0.5,
        }
    }
}

/// 噪声门降噪
///
/// 将音频分成短窗口，计算每个窗口的 RMS 值，
/// 低于阈值的窗口直接静音。
fn noise_gate(audio: &[f32], sample_rate: u32, threshold: f32) -> Vec<f32> {
    let window_size = (sample_rate as usize / 100).max(64); // 10ms 窗口
    let mut output = audio.to_vec();

    for chunk in output.chunks_mut(window_size) {
        let rms = (chunk.iter().map(|&s| s * s).sum::<f32>() / chunk.len() as f32).sqrt();
        if rms < threshold {
            // 淡出
            let fade_len = (chunk.len() / 4).max(1);
            for (i, sample) in chunk.iter_mut().enumerate() {
                let gain = if i < fade_len {
                    // 淡出
                    let progress = i as f32 / fade_len as f32;
                    1.0 - progress + progress * 0.001
                } else {
                    0.001 // 接近静音而非完全静音，避免 click
                };
                *sample *= gain;
            }
        }
    }

    output
}

/// 频谱减法降噪
///
/// 使用 realfft 进行短时傅里叶变换 (STFT)，
/// 在频域减去估计的噪声谱，然后逆变换回时域。
fn spectral_subtraction(audio: &[f32], config: &DenoiseConfig) -> Vec<f32> {
    let fft_size = config.fft_size;
    let hop_size = (fft_size as f32 * (1.0 - config.overlap)) as usize;
    let hop_size = hop_size.max(1);

    // 初始化 FFT 规划器
    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(fft_size);
    let c2r = planner.plan_fft_inverse(fft_size);

    // 准备 Hanning 窗
    let window: Vec<f32> = (0..fft_size)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / fft_size as f32).cos()))
        .collect();
    let _window_power: f32 = window.iter().map(|&w| w * w).sum::<f32>() / fft_size as f32;

    // 总帧数
    let num_frames = if audio.len() >= fft_size {
        1 + (audio.len() - fft_size + hop_size - 1) / hop_size
    } else {
        0
    };

    if num_frames == 0 {
        return audio.to_vec();
    }

    let freq_size = fft_size / 2 + 1;
    let mut noise_estimate = vec![0.0f32; freq_size];
    let mut output = vec![0.0f32; audio.len() + fft_size];
    // 累积的窗权重（用于重叠相加归一化）
    //
    // 旧实现按「帧计数」归一化（overlap_count）。Hann 窗 + 50% 重叠满足
    // COLA 性质（任一样本的窗权重之和恒为 1.0），本不应再额外除法；
    // 除以帧数会把稳态振幅压到约 50%（-6 dB），且在窗口边缘因帧数变化
    // 产生位置相关失真。改为除以窗权重之和后，任意 overlap 配置都成立
    let mut weight_sum = vec![0.0f32; audio.len() + fft_size];

    // 用于噪声估计的初始帧数
    const NOISE_ESTIMATION_FRAMES: usize = 10;

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_size;
        let mut fft_input: Vec<f32> = (0..fft_size)
            .map(|i| {
                if start + i < audio.len() {
                    audio[start + i] * window[i]
                } else {
                    0.0
                }
            })
            .collect();

        // 前向 FFT
        let mut spectrum = r2c.make_output_vec();
        r2c.process(&mut fft_input, &mut spectrum).unwrap();

        // 计算幅度谱
        let mut magnitude: Vec<f32> = spectrum.iter().map(|&c| c.norm_sqr().sqrt()).collect();

        // 噪声估计阶段：先用「更新前」的估计做谱减（首帧估计为 0，等于直通），
        // 再更新估计值。
        //
        // 旧实现在这里直接 `continue`，前 10 帧完全不参与重叠相加 ——
        // fft_size=1024、overlap=0.5（hop=512）下覆盖到第 5120 个采样点，
        // 16 kHz 时相当于**开头约 320 毫秒被静音**，对 ASR 而言等于丢掉开头内容。
        let estimating = frame_idx < NOISE_ESTIMATION_FRAMES;

        // 频谱减法
        for i in 0..freq_size {
            let noise = noise_estimate[i];
            let signal = magnitude[i];

            if signal > noise * config.subtraction_factor {
                let subtracted = signal - noise * config.subtraction_factor;
                magnitude[i] = subtracted.max(config.noise_floor);
            } else {
                magnitude[i] = config.noise_floor;
            }
        }

        // 更新噪声估计：估计阶段无条件更新；其后只在非语音段更新
        let is_speech = magnitude.iter().sum::<f32>() > noise_estimate.iter().sum::<f32>() * 0.5;
        if estimating || !is_speech {
            for i in 0..freq_size {
                noise_estimate[i] = noise_estimate[i] * config.alpha + magnitude[i] * (1.0 - config.alpha);
            }
        }

        // 用降噪后的幅度恢复复数（保持原始相位）
        for i in 0..freq_size {
            let original = spectrum[i];
            let original_mag = original.norm_sqr().sqrt().max(1e-10);
            let gain = magnitude[i] / original_mag;
            let re = original.re * gain;
            let im = original.im * gain;
            spectrum[i] = rustfft::num_complex::Complex::new(re, im);
        }

        // 逆 FFT
        let mut time_output = c2r.make_output_vec();
        c2r.process(&mut spectrum, &mut time_output).unwrap();

        // 重叠相加（同时累积窗权重，供后续归一化使用）
        for i in 0..fft_size {
            let idx = start + i;
            if idx < output.len() {
                output[idx] += time_output[i] / fft_size as f32;
                weight_sum[idx] += window[i];
            }
        }
    }

    // 归一化：除以窗权重之和（而非帧数），保证任意 overlap 配置下幅值正确
    for i in 0..audio.len() {
        if weight_sum[i] > 1e-6 {
            output[i] /= weight_sum[i];
        }
    }

    output.truncate(audio.len());
    output
}

/// 对 AudioData 执行降噪
///
/// 根据 DenoiseLevel 选择降噪策略：
/// - Low: 仅噪声门
/// - Medium: 仅频谱减法
/// - High: 噪声门 + 频谱减法
pub fn denoise(audio: &AudioData, level: votex_domain::tts::value_object::DenoiseLevel) -> Result<AudioData> {
    let config = match level {
        votex_domain::tts::value_object::DenoiseLevel::Low => DenoiseConfig {
            gate_threshold: 0.02,
            subtraction_factor: 0.0, // 不执行频谱减法
            ..Default::default()
        },
        votex_domain::tts::value_object::DenoiseLevel::Medium => DenoiseConfig {
            gate_threshold: 0.0, // 不执行噪声门
            subtraction_factor: 0.75,
            ..Default::default()
        },
        votex_domain::tts::value_object::DenoiseLevel::High => DenoiseConfig {
            gate_threshold: 0.015,
            subtraction_factor: 0.90,
            ..Default::default()
        },
    };

    // 多声道必须按声道拆分后逐声道处理。
    //
    // `samples` 是 L,R,L,R 交错的一维流；旧实现整段交给单声道 FFT/OLA，
    // 卷积会在左右声道样本之间来回跳跃，破坏声像与内容。
    // （原单元测试只断言了 `channels == 2`，元数据没变，所以从未暴露该问题）
    let channels = (audio.channels as usize).max(1);

    let samples = if channels == 1 {
        process_channel(&audio.samples, &config, audio.sample_rate)
    } else {
        let frames = audio.samples.len() / channels;
        let mut interleaved = audio.samples.clone();
        for ch in 0..channels {
            // 拆分出单个声道
            let channel_samples: Vec<f32> = (0..frames)
                .map(|f| audio.samples[f * channels + ch])
                .collect();
            let processed = process_channel(&channel_samples, &config, audio.sample_rate);
            // 重新交错写回
            for (f, v) in processed.iter().enumerate() {
                if f < frames {
                    interleaved[f * channels + ch] = *v;
                }
            }
        }
        interleaved
    };

    Ok(AudioData {
        samples,
        sample_rate: audio.sample_rate,
        channels: audio.channels,
    })
}

/// 对单个声道的采样执行完整降噪链路（噪声门 + 频谱减法）
fn process_channel(samples: &[f32], config: &DenoiseConfig, sample_rate: u32) -> Vec<f32> {
    let gated = if config.gate_threshold > 0.0 {
        noise_gate(samples, sample_rate, config.gate_threshold)
    } else {
        samples.to_vec()
    };

    if config.subtraction_factor > 0.0 && sample_rate >= 8000 {
        spectral_subtraction(&gated, config)
    } else {
        gated
    }
}

/// 便捷函数：低强度降噪（噪声门）
pub fn denoise_low(audio: &AudioData) -> Result<AudioData> {
    denoise(audio, votex_domain::tts::value_object::DenoiseLevel::Low)
}

/// 便捷函数：中强度降噪（频谱减法）
pub fn denoise_medium(audio: &AudioData) -> Result<AudioData> {
    denoise(audio, votex_domain::tts::value_object::DenoiseLevel::Medium)
}

/// 便捷函数：高强度降噪（两者叠加）
pub fn denoise_high(audio: &AudioData) -> Result<AudioData> {
    denoise(audio, votex_domain::tts::value_object::DenoiseLevel::High)
}

#[cfg(test)]
mod tests {
    use super::*;
    use votex_domain::tts::value_object::DenoiseLevel;

    #[test]
    fn test_noise_gate_纯静音保持不变() {
        let audio = AudioData::silence(16000, 1000);
        let denoised = denoise(&audio, DenoiseLevel::Low).unwrap();
        let max_val = denoised
            .samples
            .iter()
            .map(|&s| s.abs())
            .fold(0.0f32, f32::max);
        assert!(max_val < 0.01, "静音经噪声门后应保持接近零");
    }

    #[test]
    fn test_噪声门_低电平信号被压制() {
        let audio = AudioData {
            samples: vec![0.005f32; 16000],
            sample_rate: 16000,
            channels: 1,
        };
        let denoised = denoise(&audio, DenoiseLevel::Low).unwrap();
        let max_val = denoised
            .samples
            .iter()
            .map(|&s| s.abs())
            .fold(0.0f32, f32::max);
        assert!(max_val < 0.01, "低于阈值的信号应被压制");
    }

    #[test]
    fn test_噪声门_高电平信号保留() {
        let mut samples = vec![0.0f32; 16000];
        // 中间插入一段高电平
        for i in 4000..12000 {
            samples[i] = 0.5;
        }
        let audio = AudioData {
            samples,
            sample_rate: 16000,
            channels: 1,
        };
        let denoised = denoise(&audio, DenoiseLevel::Low).unwrap();
        // 中间段应该被保留
        let mid_max: f32 = denoised.samples[4000..12000]
            .iter()
            .map(|&s| s.abs())
            .fold(0.0f32, f32::max);
        assert!(mid_max > 0.1, "高电平信号应被保留");
    }

    #[test]
    fn test_频谱减法_静音输入保持不变() {
        let audio = AudioData::silence(16000, 500);
        let denoised = denoise(&audio, DenoiseLevel::Medium).unwrap();
        assert_eq!(denoised.samples.len(), audio.samples.len());
    }

    #[test]
    fn test_等级配置不同() {
        let audio = AudioData::silence(16000, 200);
        // 三种等级都不应崩溃
        assert!(denoise(&audio, DenoiseLevel::Low).is_ok());
        assert!(denoise(&audio, DenoiseLevel::Medium).is_ok());
        assert!(denoise(&audio, DenoiseLevel::High).is_ok());
    }

    #[test]
    fn test_低采样率音频() {
        let audio = AudioData::silence(8000, 200);
        assert!(denoise(&audio, DenoiseLevel::Medium).is_ok());
    }

    #[test]
    fn test_立体声降噪() {
        let audio = AudioData {
            samples: vec![0.1f32; 32000],
            sample_rate: 16000,
            channels: 2,
        };
        let denoised = denoise(&audio, DenoiseLevel::Low).unwrap();
        assert_eq!(denoised.channels, 2);
        assert_eq!(denoised.sample_rate, 16000);
    }
}
