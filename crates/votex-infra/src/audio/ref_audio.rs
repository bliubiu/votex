//! 克隆参考音频工程工具（借鉴 VoiceStudio 的参考音频契约）
//!
//! 零样本克隆对参考音频质量敏感：过长参考会被引擎静默截断（IndexTTS-2.5
//! 截到 15s），过短或全是静音的参考会产出明显的机器感/杂音。这里提供：
//!
//! - [`best_window`]：按短时能量选取「语音最多」的窗口（best_window 策略），
//!   替代引擎侧的静默截头（head 策略）
//! - [`silence_ratio`]：静音帧占比，用于入库时拦截"录了半天没说话"的参考

use votex_domain::shared::value_object::AudioData;

/// 帧长（毫秒）：短时能量分析窗口
const FRAME_MS: u32 = 20;
/// 静音帧判定阈值（RMS），低于此视为静音帧
const SILENCE_RMS_THRESHOLD: f32 = 0.01;

/// 计算静音帧占比（0.0 ~ 1.0）
///
/// 按 20ms 帧计算 RMS，RMS 低于 [`SILENCE_RMS_THRESHOLD`] 的帧视为静音帧。
pub fn silence_ratio(audio: &AudioData) -> f32 {
    let frame_len = (audio.sample_rate as usize * FRAME_MS as usize / 1000)
        .max(1)
        * audio.channels as usize;
    if audio.samples.is_empty() {
        return 1.0;
    }
    let frames = audio.samples.chunks(frame_len);
    let mut silent = 0u64;
    let mut total = 0u64;
    for frame in frames {
        total += 1;
        let energy = frame.iter().map(|s| s * s).sum::<f32>() / frame.len() as f32;
        if energy.sqrt() < SILENCE_RMS_THRESHOLD {
            silent += 1;
        }
    }
    if total == 0 {
        return 1.0;
    }
    silent as f32 / total as f32
}

/// 按短时能量选取「语音最多」的窗口（best_window 策略）
///
/// 以 20ms 帧为粒度滑动求和能量，返回能量和最大的连续 `window_ms` 窗口。
/// 时长不超过窗口时原样返回（克隆引用）。
pub fn best_window(audio: &AudioData, window_ms: u64) -> AudioData {
    let channels = audio.channels.max(1) as usize;
    let samples_per_ms = audio.sample_rate as usize / 1000;
    let window_len = (window_ms as usize * samples_per_ms).max(1) * channels;
    if audio.samples.len() <= window_len {
        return audio.clone();
    }

    let frame_len = (samples_per_ms * FRAME_MS as usize).max(1) * channels;
    let n_frames = audio.samples.len() / frame_len;
    if n_frames == 0 {
        // 音频比一个分析帧还短：直接头截
        return AudioData {
            samples: audio.samples[..window_len].to_vec(),
            sample_rate: audio.sample_rate,
            channels: audio.channels,
        };
    }

    // 帧能量前缀和 → 任意窗口能量 O(1) 求值
    let mut prefix = vec![0f64; n_frames + 1];
    for (i, frame) in audio.samples[..n_frames * frame_len].chunks(frame_len).enumerate() {
        let e: f64 = frame.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        prefix[i + 1] = prefix[i] + e;
    }

    // 窗口对齐到帧边界，保证截出的音频整帧
    let window_frames = window_len.div_ceil(frame_len).max(1);
    let mut best_start_frame = 0usize;
    let mut best_energy = f64::MIN;
    for start in 0..=(n_frames.saturating_sub(window_frames)) {
        let e = prefix[start + window_frames] - prefix[start];
        if e > best_energy {
            best_energy = e;
            best_start_frame = start;
        }
    }

    let start = best_start_frame * frame_len;
    let end = ((best_start_frame + window_frames) * frame_len).min(audio.samples.len());
    AudioData {
        samples: audio.samples[start..end].to_vec(),
        sample_rate: audio.sample_rate,
        channels: audio.channels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 构造「前半静音 + 后半正弦」的音频，验证 best_window 选中有人声的一半
    #[test]
    fn best_window_选取能量最高窗口() {
        let rate = 16000u32;
        let total_ms = 4000u32;
        let mut samples = vec![0f32; rate as usize * total_ms as usize / 1000];
        // 后 2 秒放 0.5 幅度正弦（前 2 秒纯静音）
        for (i, s) in samples.iter_mut().enumerate().skip(rate as usize * 2) {
            *s = ((i as f32 * 0.05).sin()) * 0.5;
        }
        let audio = AudioData { samples, sample_rate: rate, channels: 1 };

        let picked = best_window(&audio, 2000);
        assert_eq!(picked.duration_ms(), 2000, "窗口时长应为 2000ms");
        // 选出的窗口应落在有声段：RMS 明显大于 0
        let rms: f32 = (picked.samples.iter().map(|s| s * s).sum::<f32>() / picked.samples.len() as f32).sqrt();
        assert!(rms > 0.2, "应选中正弦段（RMS={rms}）而非静音段");
    }

    #[test]
    fn best_window_短于窗口原样返回() {
        let audio = AudioData::silence(16000, 1000);
        let picked = best_window(&audio, 5000);
        assert_eq!(picked.duration_ms(), 1000);
    }

    #[test]
    fn best_window_全静音时返回头部窗口() {
        let audio = AudioData::silence(16000, 4000);
        let picked = best_window(&audio, 2000);
        assert_eq!(picked.duration_ms(), 2000);
    }

    #[test]
    fn 静音占比_纯静音与有声() {
        let silence = AudioData::silence(16000, 1000);
        assert!((silence_ratio(&silence) - 1.0).abs() < 1e-6, "纯静音占比应为 1.0");

        let loud = AudioData {
            samples: vec![0.5f32; 16000],
            sample_rate: 16000,
            channels: 1,
        };
        assert!(silence_ratio(&loud) < 0.01, "全幅音频静音占比应接近 0");
    }

    #[test]
    fn 静音占比_空音频() {
        let empty = AudioData { samples: vec![], sample_rate: 16000, channels: 1 };
        assert!((silence_ratio(&empty) - 1.0).abs() < 1e-6);
    }
}
