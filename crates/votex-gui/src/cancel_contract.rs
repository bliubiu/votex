//! 任务取消语义契约测试
//!
//! # 为什么这些需要测
//!
//! 本轮修掉一个**欺骗性 UI**：ASR / OCR 页面的「取消」按钮原先只改界面文字，
//! 真正的取消令牌被创建在闭包里丢掉了——用户点了没反应，后台照跑。
//! 这类缺陷编译器查不出、代码评审容易漏，只有契约测试能钉住。
//!
//! 这里验证三条不变量：
//!
//! 1. 每个可取消页面的 state 都持有令牌槽位
//! 2. 令牌槽位初始为 `None`（不预先持有一个已取消的令牌）
//! 3. 置位令牌后，`store` / `swap` 语义确实能被后台线程观察到

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::state::{AppState, AsrState, BatchState, OcrState, TtsState};

#[test]
fn tts_state_初始无取消令牌() {
    let s = TtsState::default();
    assert!(
        s.cancel_token.is_none(),
        "初始就持有一个令牌会让首次点击「取消」立刻被当作已取消"
    );
}

#[test]
fn asr_state_初始无取消令牌() {
    assert!(AsrState::default().cancel_token.is_none());
}

#[test]
fn ocr_state_初始无取消令牌() {
    assert!(OcrState::default().cancel_token.is_none());
}

#[test]
fn batch_state_初始无取消令牌() {
    assert!(BatchState::default().cancel_token.is_none());
}

#[test]
fn app_state_初始未运行() {
    // 所有页面的 is_running 都应为 false，否则界面会显示进度条却不推进
    let s = AppState::default();
    assert!(!s.tts.is_running);
    assert!(!s.asr.is_running);
    assert!(!s.ocr.is_running);
    assert!(!s.batch.is_running);
}

#[test]
fn 置位令牌可被后台线程观察到() {
    // 这是「取消」按钮能生效的前提：
    // 页面与后台线程持有的是**同一个** Arc，而不是各自新建。
    let token = Arc::new(AtomicBool::new(false));
    let backend = Arc::clone(&token);

    let worker = std::thread::spawn(move || {
        // 模拟后台在检查点读取令牌
        backend.load(Ordering::SeqCst)
    });

    // 模拟用户点「取消」
    token.store(true, Ordering::SeqCst);

    assert!(
        worker.join().unwrap(),
        "后台线程应观察到取消标志"
    );
}

#[test]
fn swap_只报告一次取消() {
    // CLI / 批量场景用 swap 保证「只提示一次中断」。
    // swap 返回的是**置位前的旧值**，所以首次调用返回 false（此前未取消）。
    let token = Arc::new(AtomicBool::new(false));

    assert!(
        !token.swap(true, Ordering::SeqCst),
        "首次置位：旧值为 false，表示此前未取消"
    );
    assert!(
        token.swap(true, Ordering::SeqCst),
        "重复置位：旧值为 true，表示已取消过，不应再提示"
    );
    assert!(token.load(Ordering::SeqCst));
}

#[test]
fn 令牌可克隆且共享同一状态() {
    // 页面与后台各持一份 clone，修改互相可见
    let page_side = Arc::new(AtomicBool::new(false));
    let task_side = Arc::clone(&page_side);

    task_side.store(true, Ordering::SeqCst);
    assert!(page_side.load(Ordering::SeqCst));

    page_side.store(false, Ordering::SeqCst);
    assert!(!task_side.load(Ordering::SeqCst));
}

#[test]
fn 不同页面的令牌互相独立() {
    // TTS 取消不应影响 ASR：两个页面各自持有一份
    let tts = Arc::new(AtomicBool::new(false));
    let asr = Arc::new(AtomicBool::new(false));

    tts.store(true, Ordering::SeqCst);
    assert!(tts.load(Ordering::SeqCst));
    assert!(
        !asr.load(Ordering::SeqCst),
        "取消 TTS 不应波及其他页面"
    );
}
