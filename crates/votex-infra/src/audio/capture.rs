//! 麦克风采集（实时听写音频输入）
//!
//! cpal 默认输入设备 → 单声道 f32 → 线性插值重采样到目标采样率（16kHz），
//! 每个回调产出的重采样片段经 channel 发给消费方（流式识别工作线程）。
//!
//! # 为什么自己写重采样
//!
//! 麦克风原生采样率通常为 44.1k/48kHz，而识别引擎期望 16kHz。
//! 请求设备直接输出 16kHz 在 WASAPI 共享模式下不保证可行，
//! 因此采集原生采样率后做有状态线性插值——对语音识别精度足够，
//! 实现简单且可单测。
//!
//! # 回调线程模型
//!
//! cpal 回调是实时音频线程，只做三件事：格式转换、下混重采样、
//! channel 发送；禁止分配之外的阻塞操作。重采样器状态经
//! `Arc<Mutex<..>>` 跨回调保持（同一流的所有回调串行执行，
//! 锁不会形成竞争）。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, Context, Result};
use cpal::traits::{HostTrait, DeviceTrait};

/// 目标采样率：全部本地 ASR 引擎统一 16kHz
pub const TARGET_SAMPLE_RATE: u32 = 16000;

/// 有状态线性插值重采样器（交错多声道输入 → 单声道输出）
///
/// 内部保留未消费的尾部样本；`process` 任意分段调用，
/// 输出拼接后与整体重采样一致（误差 ≤ 1 个输出样本）。
#[derive(Debug)]
pub struct LinearResampler {
    /// 每个输出样本对应的输入帧数 = 输入采样率 / 输出采样率
    /// （44.1k→16k 为 2.756，直通为 1.0）
    step: f64,
    channels: usize,
    /// 未消费的交错样本缓冲（已消费帧之后的尾部 + 新输入）
    buf: Vec<f32>,
    /// 下一输出样本在 buf 中的帧位置（小数为帧内插值系数）
    pos: f64,
}

impl LinearResampler {
    pub fn new(input_rate: u32, output_rate: u32, channels: usize) -> Self {
        Self {
            step: input_rate.max(1) as f64 / output_rate.max(1) as f64,
            channels: channels.max(1),
            buf: Vec::new(),
            pos: 0.0,
        }
    }

    /// 送入一段交错样本，返回重采样后的单声道样本
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        self.buf.extend_from_slice(input);
        let ch = self.channels;
        let total_frames = self.buf.len() / ch;
        let mut out = Vec::with_capacity((input.len() / ch) * (1.0 / self.step) as usize + 4);

        // 需要 pos 和 pos+1 两帧才能插值；最后一帧留给下一轮
        while self.pos + 1.0 < total_frames as f64 {
            let i = self.pos as usize;
            let frac = self.pos - i as f64;
            let mut mono = 0.0f32;
            for c in 0..ch {
                let a = self.buf[i * ch + c];
                let b = self.buf[(i + 1) * ch + c];
                mono += a + (b - a) * frac as f32;
            }
            out.push(mono / ch as f32);
            self.pos += self.step;
        }

        // 丢弃已消费帧，保留 pos 所在帧起的数据；pos 折算到新缓冲
        let consumed = (self.pos as usize).min(total_frames.saturating_sub(1));
        if consumed > 0 {
            self.buf.drain(..consumed * ch);
            self.pos -= consumed as f64;
        }
        out
    }
}

/// 麦克风采集句柄
///
/// 持有 cpal 流；`stop()` 置位停止标志并终结流（流 Drop 即停止采集）。
/// 非 Send：cpal::Stream 要求在创建线程释放。
pub struct MicCaptureHandle {
    stream: Option<cpal::Stream>,
    stop: Arc<AtomicBool>,
    stopped: bool,
}

impl MicCaptureHandle {
    /// 置位停止标志（回调线程随后自然退出）
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::SeqCst);
    }

    /// 停止采集并释放音频流（必须在与 [`start_default_mic`] 相同的线程调用）
    pub fn stop(&mut self) {
        self.request_stop();
        if !self.stopped {
            self.stopped = true;
            self.stream = None; // Drop 流即停止采集
        }
    }

    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.stop)
    }
}

/// 回调共享状态：跨回调保持的重采样器
struct StreamState {
    resampler: LinearResampler,
}

/// 枚举可用的输入设备名（GUI 下拉/诊断用）
pub fn list_input_devices() -> Vec<String> {
    let host = cpal::default_host();
    host.input_devices()
        .map(|devices| devices.filter_map(|d| d.name().ok()).collect())
        .unwrap_or_default()
}

/// 启动默认麦克风采集
///
/// - 采集默认输入设备的原生采样率/声道
/// - 回调内转单声道 + 重采样到 `target_rate`
/// - 每次回调通过 `tx` 发送一个重采样片段
///
/// 返回句柄：`stop()` 停止采集；句柄 drop 也会停止。
pub fn start_default_mic(target_rate: u32, tx: Sender<Vec<f32>>) -> Result<MicCaptureHandle> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("未找到麦克风输入设备，请检查系统录音设置"))?;
    let device_name = device.name().unwrap_or_default();

    let input_config = device
        .default_input_config()
        .context("读取麦克风默认配置失败")?;
    let config: cpal::StreamConfig = input_config.clone().into();
    let in_rate = config.sample_rate.0;
    let channels = config.channels as usize;

    tracing::info!(
        "启动麦克风采集: 设备={}, 原生采样率={}Hz, 声道={}, 目标={}Hz",
        device_name,
        in_rate,
        channels,
        target_rate
    );

    if channels == 0 {
        return Err(anyhow!("麦克风声道数为 0，无法采集"));
    }

    let stop = Arc::new(AtomicBool::new(false));

    // 回调共享状态（同一流回调串行执行，锁无竞争）
    let state = Arc::new(Mutex::new(StreamState {
        resampler: LinearResampler::new(in_rate, target_rate, channels),
    }));


    let stream = match input_config.sample_format() {
        cpal::SampleFormat::F32 => {
            build_for_format::<f32>(&device, &config, tx.clone(), stop.clone(), state.clone())
        }
        cpal::SampleFormat::I16 => {
            build_for_format::<i16>(&device, &config, tx.clone(), stop.clone(), state.clone())
        }
        cpal::SampleFormat::U16 => {
            build_for_format::<u16>(&device, &config, tx.clone(), stop.clone(), state.clone())
        }
        other => return Err(anyhow!("不支持的麦克风采样格式: {:?}", other)),
    }?;

    Ok(MicCaptureHandle {
        stream: Some(stream),
        stop,
        stopped: false,
    })
}

/// 按具体采样格式构建输入流（f32/i16/u16 统一转 f32 后重采样）
fn build_for_format<T: cpal::SizedSample + cpal::FromSample<f32> + num_traits::ToPrimitive>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    tx: Sender<Vec<f32>>,
    stop: Arc<AtomicBool>,
    state: Arc<Mutex<StreamState>>,
) -> Result<cpal::Stream> {
    device
        .build_input_stream(
            config,
            move |data: &[T], _: &cpal::InputCallbackInfo| {
                if stop.load(Ordering::SeqCst) {
                    return;
                }
                let f32_samples: Vec<f32> = data.iter().filter_map(|s| s.to_f32()).collect();
                // 下混 + 重采样到目标采样率（状态跨回调保持）
                let chunk = state
                    .lock()
                    .map(|mut s| s.resampler.process(&f32_samples))
                    .unwrap_or_default();
                if !chunk.is_empty() {
                    // 消费方退出时发送失败，静默丢弃（停止流程）
                    let _ = tx.send(chunk);
                }
            },
            |err| tracing::warn!("麦克风采集回调错误: {}", err),
            None,
        )
        .context("创建麦克风输入流失败")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 44800 个输入样本（44.1kHz×1s）重采样到 16kHz ≈ 16000 个输出样本
    #[test]
    fn 重采样_44k到16k_长度正确() {
        let mut r = LinearResampler::new(44100, 16000, 1);
        let input: Vec<f32> = (0..44100).map(|i| ((i as f32) * 0.01).sin()).collect();
        let out = r.process(&input);
        assert!(
            (out.len() as i32 - 15980).abs() < 30,
            "重采样输出长度应接近 16000，实际 {}",
            out.len()
        );
    }

    /// 分段调用与一次性调用结果一致（状态跨段保持）
    #[test]
    fn 重采样_分段与整体一致() {
        let input: Vec<f32> = (0..48000).map(|i| ((i as f32) * 0.02).sin()).collect();

        let mut whole = LinearResampler::new(48000, 16000, 1);
        let all_at_once = whole.process(&input);

        let mut chunked = LinearResampler::new(48000, 16000, 1);
        let mut acc = Vec::new();
        for chunk in input.chunks(480) {
            acc.extend(chunked.process(chunk));
        }
        acc.extend(chunked.process(&[])); // 冲刷尾部

        assert_eq!(all_at_once.len(), acc.len());
        for (a, b) in all_at_once.iter().zip(acc.iter()) {
            assert!((a - b).abs() < 1e-6, "分段重采样与整体不一致: {} vs {}", a, b);
        }
    }

    /// 立体声输入下混后取平均
    #[test]
    fn 重采样_立体声下混() {
        let mut r = LinearResampler::new(16000, 16000, 2);
        // 恒定信号：L=1.0, R=0.0 → 单声道 0.5
        let input = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let out = r.process(&input);
        assert!(out.len() >= 3);
        for v in &out {
            assert!((*v - 0.5).abs() < 1e-6, "下混值应为 0.5，实际 {}", v);
        }
    }

    /// 直通（输入输出同采样率）不改变样本数
    #[test]
    fn 重采样_同采样率直通() {
        let mut r = LinearResampler::new(16000, 16000, 1);
        let input = vec![0.25f32; 1600];
        let out = r.process(&input);
        assert_eq!(out.len(), 1599); // 最后 1 帧留待下一轮
        assert!((out[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn 重采样_空输入安全() {
        let mut r = LinearResampler::new(44100, 16000, 2);
        let out = r.process(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn 设备枚举_不panic() {
        // 只验证在无设备环境下不会 panic，结果长度取决于运行环境
        let devices = list_input_devices();
        let _ = devices.len();
    }
}
