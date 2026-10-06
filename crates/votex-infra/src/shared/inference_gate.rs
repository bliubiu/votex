//! 全局推理闸门
//!
//! 三道防线防止推理任务占满系统资源：
//! 1. **并发许可**：同时运行的推理任务数受 `max_permits` 限制
//! 2. **内存压力联动**：Yellow 收缩并发到 1，Red 暂停派发等待回落
//! 3. **提交前预检**：预估需求内存 + 系统预留不足时拒绝接单（等待）
//!
//! 许可为 RAII guard：持有即占用，Drop 自动释放。
//! 与 `ResourceMonitor` 联动，可从 `application.yml` 的 `resource:` 段配置。
//!
//! # 取消机制
//!
//! 旧实现的 `acquire()` 是**不可中断**的无限等待：Red 压力下闸门关闭后，
//! 排队中的任务只能等内存回落到 Green。这在长文本合成场景下有实际风险 ——
//! 用户点「停止」后任务仍挂在闸门上，界面表现为「点了没反应」。
//!
//! 现在每次 `acquire()` 传入 [`GateCancel`]：取消后 `acquire` 立即返回
//! `Err(GateError::Cancelled)`，等待中的任务立刻退出，GUI 才能真正响应停止。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use super::resource_monitor::{MemoryPressure, ResourceMonitor};

/// 预留给操作系统的内存（MB），可用内存低于该值时拒绝新任务
const RESERVE_MEMORY_MB: u64 = 1024;

/// 闸门等待的重评估周期（毫秒）
///
/// `Condvar` 没有超时唤醒源，只能靠定时重醒来重新评估内存压力。
const POLL_INTERVAL_MS: u64 = 500;

#[derive(Debug)]
struct GateState {
    /// 当前持有的许可数
    held: usize,
    /// 最大并发许可数（Green 状态下）
    max_permits: usize,
}

/// 闸门拒绝派发的错误
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateError {
    /// 任务在排队期间被取消
    Cancelled,
}

impl std::fmt::Display for GateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GateError::Cancelled => write!(f, "推理任务已取消"),
        }
    }
}

impl std::error::Error for GateError {}

/// 取消令牌
///
/// 克隆代价低（内部 `Arc`），可随任务一路传到闸门。
/// 典型用法：
/// ```ignore
/// let cancel = GateCancel::new();
/// // 另一个线程点「停止」时调用 cancel.cancel()
/// let _permit = gate.acquire(&cancel, need_mb)?;
/// ```
#[derive(Clone, Debug, Default)]
pub struct GateCancel {
    flag: Arc<AtomicBool>,
}

impl GateCancel {
    /// 创建未取消的令牌
    pub fn new() -> Self {
        Self::default()
    }

    /// 复用既有的取消标志构造令牌
    ///
    /// 项目里取消状态有两种表示：本模块的 [`GateCancel`] 与应用层的
    /// `Option<Arc<AtomicBool>>`（TTS 用例的 `cancel` 参数）。若两者各存一份，
    /// GUI 置位其中之一将无法作用于另一处，因此这里直接共享底层标志。
    ///
    /// - `flag` 为 `None` 时（调用方不要求可取消），返回未置位的令牌
    pub fn from_atomic(flag: Option<Arc<AtomicBool>>) -> Self {
        match flag {
            Some(f) => Self { flag: f },
            // 无法共享时退化为自有标志：永不置位，行为等价于「不取消」
            None => Self::new(),
        }
    }

    /// 请求取消
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    /// 是否已请求取消
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }
}

/// 推理闸门（进程内单例）
#[derive(Debug)]
pub struct InferenceGate {
    state: Mutex<GateState>,
    cv: Condvar,
}

static GATE: OnceLock<std::sync::Arc<InferenceGate>> = OnceLock::new();

impl InferenceGate {
    pub fn global() -> &'static std::sync::Arc<InferenceGate> {
        GATE.get_or_init(|| std::sync::Arc::new(InferenceGate {
            state: Mutex::new(GateState {
                held: 0,
                max_permits: 2,
            }),
            cv: Condvar::new(),
        }))
    }

    /// 配置最大并发许可数（Green 状态下；Yellow 自动收缩为 1）
    pub fn configure_max_permits(&self, max: usize) {
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        g.max_permits = max.clamp(1, 16);
        self.cv.notify_all();
    }

    /// 当前配置的最大许可数
    pub fn max_permits(&self) -> usize {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).max_permits
    }

    /// 当前已持有的许可数
    pub fn held_permits(&self) -> usize {
        self.state.lock().unwrap_or_else(|e| e.into_inner()).held
    }

    /// 唤醒所有等待者（取消令牌被置位后调用，让其立即重新检查）
    pub fn notify_waiters(&self) {
        self.cv.notify_all();
    }

    /// 获取一个推理许可（阻塞等待，可取消）
    ///
    /// - `need_memory_mb`：本任务预估需要的内存（含模型 + 段音频）
    /// - `cancel`：取消令牌；置位后本调用立即返回 `Err(Cancelled)`
    ///
    /// 等待条件（全部满足才放行）：
    /// 1. 未收到取消请求
    /// 2. `held < max_permits`（Yellow 状态下有效上限为 1）
    /// 3. 系统压力非 Red
    /// 4. 可用内存 - 预留 > need_memory_mb 中超出已计入的部分
    pub fn acquire(
        self: &std::sync::Arc<Self>,
        cancel: &GateCancel,
        need_memory_mb: u64,
    ) -> Result<GatePermit, GateError> {
        let monitor = ResourceMonitor::global();
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            // 取消检查放在最前：已取消的任务不应再占许可
            if cancel.is_cancelled() {
                return Err(GateError::Cancelled);
            }

            let (snap, pressure) = monitor.snapshot();
            let effective_max = match pressure {
                MemoryPressure::Red => 0,
                MemoryPressure::Yellow => 1,
                MemoryPressure::Green => g.max_permits,
            };

            let memory_ok = snap.available_mem_mb > RESERVE_MEMORY_MB + need_memory_mb;
            let permit_ok = g.held < effective_max;

            if permit_ok && memory_ok {
                g.held += 1;
                return Ok(GatePermit::new(self));
            }

            // 等待：内存紧张时用较短轮询周期，仅等许可时也用同一周期
            // （Condvar 无超时唤醒源，靠定时重醒来重新评估压力）
            let (guard, _timeout) = self
                .cv
                .wait_timeout(g, std::time::Duration::from_millis(POLL_INTERVAL_MS))
                .unwrap_or_else(|e| e.into_inner());
            g = guard;
        }
    }

    /// 不阻塞地尝试获取许可（用于「忙」时给用户即时反馈而非干等）
    ///
    /// 返回 `Ok(None)` 表示当前不满足条件但未被取消 —— 调用方应转为「排队等待」
    /// 并使用 [`Self::acquire`]。
    pub fn try_acquire(
        self: &std::sync::Arc<Self>,
        cancel: &GateCancel,
        need_memory_mb: u64,
    ) -> Result<Option<GatePermit>, GateError> {
        if cancel.is_cancelled() {
            return Err(GateError::Cancelled);
        }
        let monitor = ResourceMonitor::global();
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        let (snap, pressure) = monitor.snapshot();
        let effective_max = match pressure {
            MemoryPressure::Red => 0,
            MemoryPressure::Yellow => 1,
            MemoryPressure::Green => g.max_permits,
        };
        let memory_ok = snap.available_mem_mb > RESERVE_MEMORY_MB + need_memory_mb;
        if g.held < effective_max && memory_ok {
            g.held += 1;
            return Ok(Some(GatePermit::new(self)));
        }
        Ok(None)
    }

    fn release(&self) {
        let mut g = self.state.lock().unwrap_or_else(|e| e.into_inner());
        g.held = g.held.saturating_sub(1);
        self.cv.notify_all();
    }
}

/// RAII 推理许可
///
/// Drop 时自动释放许可。
///
/// # 为什么持有闸门引用而不是用全局单例
///
/// 旧实现的 `drop` 固定调用 `InferenceGate::global().release()`，
/// 意味着「从任意闸门实例获取的许可都归还给全局单例」——
/// 这让 `InferenceGate` 无法被独立实例化（单测只能串行使用全局单例，
/// 用例间通过 `configure_max_permits` 隐式耦合），
/// 也让 `notify_waiters` 这类实例方法在测试里无法验证。
#[derive(Debug)]
pub struct GatePermit {
    gate: std::sync::Arc<InferenceGate>,
}

impl GatePermit {
    fn new(gate: &std::sync::Arc<InferenceGate>) -> Self {
        Self {
            gate: std::sync::Arc::clone(gate),
        }
    }
}

impl Drop for GatePermit {
    fn drop(&mut self) {
        self.gate.release();
    }
}

/// 根据物理内存推导合理的默认并发数
///
/// 公式：max(1, min(4, 可用于推理的内存 GB / 单模型预估 GB))
pub fn derive_max_permits(available_for_inference_gb: u64, model_gb_estimate: u64) -> usize {
    if model_gb_estimate == 0 {
        return 1;
    }
    ((available_for_inference_gb / model_gb_estimate).clamp(1, 4)) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// 独立闸门实例（避免污染全局单例，保证用例间互不干扰）
    fn new_gate(max: usize) -> std::sync::Arc<InferenceGate> {
        std::sync::Arc::new(InferenceGate {
            state: Mutex::new(GateState {
                held: 0,
                max_permits: max,
            }),
            cv: Condvar::new(),
        })
    }

    #[test]
    fn 闸门_许可获取与释放() {
        let gate = new_gate(2);
        let cancel = GateCancel::new();
        assert_eq!(gate.max_permits(), 2);

        let p1 = gate.acquire(&cancel, 0).expect("应获取到许可");
        let p2 = gate.acquire(&cancel, 0).expect("应获取到许可");
        assert_eq!(gate.held_permits(), 2);
        drop(p1);
        drop(p2);
        // RAII 释放：许可数应归零，若泄漏此值不会归零
        assert_eq!(gate.held_permits(), 0);

        let p3 = gate.acquire(&cancel, 0).expect("释放后应能再次获取");
        drop(p3);
    }

    #[test]
    fn 闸门_已取消应立即返回错误() {
        let gate = new_gate(1);
        let cancel = GateCancel::new();
        cancel.cancel();

        // 关键：即使条件完全满足，已取消也必须拒绝
        let result = gate.acquire(&cancel, 0);
        assert_eq!(result.unwrap_err(), GateError::Cancelled);
        assert_eq!(gate.held_permits(), 0, "被取消的任务不应占用许可");
    }

    #[test]
    fn 闸门_排队中被取消可中断() {
        use std::sync::Arc;
        let gate = new_gate(1);
        let holder = GateCancel::new();
        // 占满唯一许可
        let _p = gate.acquire(&holder, 0).expect("应获取到许可");

        let waiter_cancel = GateCancel::new();
        let gate_for_thread = Arc::clone(&gate);
        let cancel_for_thread = waiter_cancel.clone();
        let handle = std::thread::spawn(move || {
            gate_for_thread.acquire(&cancel_for_thread, 0)
        });

        // 等一小会儿确保等待方已进入阻塞
        std::thread::sleep(Duration::from_millis(50));
        waiter_cancel.cancel();
        gate.notify_waiters();

        let result = handle.join().expect("等待线程 panic");
        assert_eq!(result.unwrap_err(), GateError::Cancelled, "排队任务应被取消");
        assert_eq!(gate.held_permits(), 1, "取消不应影响已持有的许可");
    }

    #[test]
    fn 闸门_取消令牌可跨线程共享() {
        use std::sync::Arc;
        let gate = new_gate(1);
        let holder = GateCancel::new();
        let _p = gate.acquire(&holder, 0).expect("应获取到许可");

        let waiter_cancel = GateCancel::new();
        let g = Arc::clone(&gate);
        let c = waiter_cancel.clone();
        let handle = std::thread::spawn(move || g.acquire(&c, 0));

        std::thread::sleep(Duration::from_millis(30));
        // 从另一个线程取消
        let trigger = waiter_cancel.clone();
        std::thread::spawn(move || {
            trigger.cancel();
        });
        gate.notify_waiters();

        assert_eq!(handle.join().expect("等待线程 panic").unwrap_err(), GateError::Cancelled);
    }

    #[test]
    fn 闸门_取消令牌默认值与幂等性() {
        let c = GateCancel::new();
        assert!(!c.is_cancelled());
        c.cancel();
        assert!(c.is_cancelled());
        // 重复取消不应 panic
        c.cancel();
        assert!(c.is_cancelled());
        // 克隆体共享状态
        let c2 = c.clone();
        assert!(c2.is_cancelled());
    }

    #[test]
    fn 闸门_from_atomic应共享底层标志() {
        use std::sync::Arc;
        let flag = Arc::new(AtomicBool::new(false));
        let c = GateCancel::from_atomic(Some(Arc::clone(&flag)));
        assert!(!c.is_cancelled());

        // 外部置位 → 令牌应立即感知（这是与 TTS 用例 cancel 参数联通的关键）
        flag.store(true, Ordering::SeqCst);
        assert!(c.is_cancelled(), "令牌未共享底层 AtomicBool");

        // 反向：令牌置位 → 外部也应看到
        let flag2 = Arc::new(AtomicBool::new(false));
        let c2 = GateCancel::from_atomic(Some(Arc::clone(&flag2)));
        c2.cancel();
        assert!(flag2.load(Ordering::SeqCst), "外部标志未感知取消");
    }

    #[test]
    fn 闸门_from_atomic接受空值() {
        // 调用方不要求可取消时不应 panic，且永不置位
        let c = GateCancel::from_atomic(None);
        assert!(!c.is_cancelled());
        c.cancel();
        assert!(c.is_cancelled());
    }

    #[test]
    fn 闸门_共享令牌的排队任务可被外部中断() {
        use std::sync::Arc;
        let gate = new_gate(1);
        let holder = GateCancel::new();
        let _p = gate.acquire(&holder, 0).expect("应获取到许可");

        // 模拟 TTS 用例：外部 Arc<AtomicBool> 被 GUI 置位
        let flag = Arc::new(AtomicBool::new(false));
        let gate_cancel = GateCancel::from_atomic(Some(Arc::clone(&flag)));
        let g = Arc::clone(&gate);
        let handle = std::thread::spawn(move || g.acquire(&gate_cancel, 0));

        std::thread::sleep(Duration::from_millis(30));
        flag.store(true, Ordering::SeqCst);
        gate.notify_waiters();

        assert_eq!(
            handle.join().expect("等待线程 panic").unwrap_err(),
            GateError::Cancelled
        );
    }

    #[test]
    fn 闸门_try_acquire不阻塞() {
        let gate = new_gate(1);
        let cancel = GateCancel::new();

        // 有空位时应立即拿到
        let got = gate.try_acquire(&cancel, 0).expect("未取消");
        assert!(got.is_some());
        let _held = got;

        // 占满后应返回 None 而非阻塞
        assert!(gate.try_acquire(&cancel, 0).expect("未取消").is_none());

        // 取消后应返回 Err
        cancel.cancel();
        assert_eq!(
            gate.try_acquire(&cancel, 0).unwrap_err(),
            GateError::Cancelled
        );
    }

    #[test]
    fn 闸门_并发上限推导() {
        assert_eq!(derive_max_permits(16, 4), 4);
        assert_eq!(derive_max_permits(8, 4), 2);
        assert_eq!(derive_max_permits(2, 4), 1);
        assert_eq!(derive_max_permits(100, 0), 1);
    }

    #[test]
    fn 闸门_全局单例可配置() {
        let gate = InferenceGate::global();
        gate.configure_max_permits(0);
        assert_eq!(gate.max_permits(), 1, "0 应被夹到最小 1");
        gate.configure_max_permits(999);
        assert_eq!(gate.max_permits(), 16, "超上限应被夹到 16");
        gate.configure_max_permits(2);
    }
}
