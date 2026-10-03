//! 全局推理闸门
//!
//! 三道防线防止推理任务占满系统资源：
//! 1. **并发许可**：同时运行的推理任务数受 `max_permits` 限制
//! 2. **内存压力联动**：Yellow 收缩并发到 1，Red 暂停派发等待回落
//! 3. **提交前预检**：预估需求内存 + 系统预留不足时拒绝接单（等待）
//!
//! 许可为 RAII guard：持有即占用，Drop 自动释放。
//! 与 `ResourceMonitor` 联动，可从 `application.yml` 的 `resource:` 段配置。

use std::sync::{Condvar, Mutex, OnceLock};

use super::resource_monitor::{MemoryPressure, ResourceMonitor};

/// 预留给操作系统的内存（MB），可用内存低于该值时拒绝新任务
const RESERVE_MEMORY_MB: u64 = 1024;

struct GateState {
    /// 当前持有的许可数
    held: usize,
    /// 最大并发许可数（Green 状态下）
    max_permits: usize,
}

/// 推理闸门（进程内单例）
pub struct InferenceGate {
    state: Mutex<GateState>,
    cv: Condvar,
}

static GATE: OnceLock<InferenceGate> = OnceLock::new();

impl InferenceGate {
    pub fn global() -> &'static InferenceGate {
        GATE.get_or_init(|| InferenceGate {
            state: Mutex::new(GateState {
                held: 0,
                max_permits: 2,
            }),
            cv: Condvar::new(),
        })
    }

    /// 配置最大并发许可数（Green 状态下；Yellow 自动收缩为 1）
    pub fn configure_max_permits(&self, max: usize) {
        let mut g = self.state.lock().unwrap();
        g.max_permits = max.clamp(1, 16);
        self.cv.notify_all();
    }

    /// 当前配置的最大许可数
    pub fn max_permits(&self) -> usize {
        self.state.lock().unwrap().max_permits
    }

    /// 获取一个推理许可（阻塞等待）
    ///
    /// - `need_memory_mb`：本任务预估需要的内存（含模型 + 段音频）
    ///
    /// 等待条件（全部满足才放行）：
    /// 1. `held < max_permits`（Yellow 状态下有效上限为 1）
    /// 2. 系统压力非 Red
    /// 3. 可用内存 - 预留 > need_memory_mb 中超出已计入的部分
    pub fn acquire(&self, need_memory_mb: u64) -> GatePermit {
        let monitor = ResourceMonitor::global();
        let mut g = self.state.lock().unwrap();
        loop {
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
                return GatePermit::new();
            }

            // 等待：内存紧张时用较短轮询周期（500ms），仅等许可时也用 500ms
            // （ Condvar 无超时唤醒源，靠定时重醒来重新评估压力）
            g = self
                .cv
                .wait_timeout(g, std::time::Duration::from_millis(500))
                .unwrap()
                .0;
        }
    }

    fn release(&self) {
        let mut g = self.state.lock().unwrap();
        g.held = g.held.saturating_sub(1);
        self.cv.notify_all();
    }
}

/// RAII 推理许可
///
/// Drop 时自动释放许可（通过全局单例归还）。
pub struct GatePermit {
    _priv: (),
}

impl GatePermit {
    fn new() -> Self {
        Self { _priv: () }
    }
}

impl Drop for GatePermit {
    fn drop(&mut self) {
        Self::global_release();
    }
}

impl GatePermit {
    fn global_release() {
        InferenceGate::global().release();
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

    #[test]
    fn 闸门_许可获取与释放() {
        let gate = InferenceGate::global();
        gate.configure_max_permits(2);
        assert_eq!(gate.max_permits(), 2);

        // 请求 0 内存（测试环境内存必然满足）
        let p1 = gate.acquire(0);
        let p2 = gate.acquire(0);
        drop(p1);
        drop(p2);
        // 如果泄漏，后续 acquire 会卡住——能走到这里即无泄漏
        let p3 = gate.acquire(0);
        drop(p3);
    }

    #[test]
    fn 闸门_并发上限推导() {
        assert_eq!(derive_max_permits(16, 4), 4);
        assert_eq!(derive_max_permits(8, 4), 2);
        assert_eq!(derive_max_permits(2, 4), 1);
        assert_eq!(derive_max_permits(100, 0), 1);
    }
}
