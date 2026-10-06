//! 进程级共享用例实例
//!
//! # 为什么需要这一层
//!
//! TTS / ASR / OCR 引擎的权重是 GB 级的，加载一次需要数秒到数十秒。
//! GUI 此前在**每个任务**里 `TtsUseCase::new()`，导致：
//!
//! - 连续合成 10 段语音 → 重复加载 10 次模型
//! - 每次加载都会在 `OrtSessionFactory` 里重建 ONNX session，内存峰值翻倍
//!
//! P2-17 把 provider 的 `load` / `unload` / `recognize` 全部改为 `&self`
//! 之后，用例对象本身不再持有可变状态，可以安全地被多个任务线程共享。
//! 这是「接口改造 → 调用方简化」的典型收益。
//!
//! # 线程安全前提
//!
//! 单例只在 `&self` 上暴露方法，内部可变状态全部由 `EngineState`
//! （`Mutex<Option<T>>`）保护。ONNX Runtime 的 `Session::run` 本身
//! 是 `&mut self`，因此 `EngineState::with_mut` 在同一时刻只允许
//! 一个线程推理——多个任务线程会在这里排队，而不是并发进入 ORT。
//! 这与 `InferenceGate` 的推理闸门是两层独立的限流。
//!
//! # 不适用场景
//!
//! CLI 是一次性进程，仍然直接 `TtsUseCase::new()` 即可，
//! 不需要走这一层。

use std::sync::{Arc, OnceLock};

use crate::use_case::asr_use_case::AsrUseCase;
use crate::use_case::ocr_use_case::OcrUseCase;
use crate::use_case::tts_use_case::TtsUseCase;

/// 进程级共享 TTS 用例
///
/// 首次调用时才创建实例，之后复用。模型只加载一次。
pub fn shared_tts() -> Arc<TtsUseCase> {
    static INSTANCE: OnceLock<Arc<TtsUseCase>> = OnceLock::new();
    INSTANCE
        .get_or_init(|| Arc::new(TtsUseCase::new()))
        .clone()
}

/// 进程级共享 ASR 用例
pub fn shared_asr() -> Arc<AsrUseCase> {
    static INSTANCE: OnceLock<Arc<AsrUseCase>> = OnceLock::new();
    INSTANCE
        .get_or_init(|| Arc::new(AsrUseCase::new()))
        .clone()
}

/// 进程级共享 OCR 用例
pub fn shared_ocr() -> Arc<OcrUseCase> {
    static INSTANCE: OnceLock<Arc<OcrUseCase>> = OnceLock::new();
    INSTANCE
        .get_or_init(|| Arc::new(OcrUseCase::new()))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 共享单例_多次获取同一实例() {
        let a = shared_tts();
        let b = shared_tts();
        assert!(Arc::ptr_eq(&a, &b), "TTS 用例单例应返回同一实例");
    }

    #[test]
    fn 不同能力的单例互相独立() {
        // 三种能力是不同类型，不可能 ptr_eq；
        // 这里确认各自都能独立初始化而不 panic。
        let _ = shared_asr();
        let _ = shared_ocr();
    }
}
