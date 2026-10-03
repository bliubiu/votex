//! 系统资源监控
//!
//! 基于 sysinfo 采样内存/CPU，输出压力分级供推理闸门决策：
//! - Green：资源充裕，允许满配置并发
//! - Yellow：内存吃紧，收缩并发到 1
//! - Red：可用内存濒临耗尽，暂停派发新任务（等待回落）
//!
//! 防止长文本批量合成占满系统资源触发 OOM/宕机/hang。

use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// 当前进程 PID（sysinfo 0.33 返回 Result，统一在此处理）
fn current_pid() -> sysinfo::Pid {
    sysinfo::get_current_pid().unwrap_or_else(|_| sysinfo::Pid::from_u32(0))
}

/// 内存压力分级
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryPressure {
    /// 资源充裕
    Green,
    /// 内存吃紧（已用 > yellow 阈值）
    Yellow,
    /// 濒临耗尽（已用 > red 阈值），暂停派发
    Red,
}

/// 资源快照
#[derive(Debug, Clone)]
pub struct ResourceSnapshot {
    /// 物理内存总量 MB
    pub total_mem_mb: u64,
    /// 可用内存 MB
    pub available_mem_mb: u64,
    /// 内存已用百分比（0~100）
    pub mem_used_pct: f32,
    /// 本进程 RSS MB
    pub process_rss_mb: u64,
    /// 系统 CPU 使用率百分比（0~100）
    pub cpu_used_pct: f32,
}

impl ResourceSnapshot {
    /// 根据阈值判定压力级别
    pub fn pressure(&self, yellow_pct: f32, red_pct: f32) -> MemoryPressure {
        if self.mem_used_pct >= red_pct {
            MemoryPressure::Red
        } else if self.mem_used_pct >= yellow_pct {
            MemoryPressure::Yellow
        } else {
            MemoryPressure::Green
        }
    }
}

struct Inner {
    sys: sysinfo::System,
    thresholds: (f32, f32), // (yellow, red) 已用百分比
    last_sample: Option<(Instant, ResourceSnapshot, MemoryPressure)>,
    min_interval: Duration,
}

/// 全局资源监控器（进程内单例）
pub struct ResourceMonitor {
    inner: Mutex<Inner>,
}

static MONITOR: OnceLock<ResourceMonitor> = OnceLock::new();

impl ResourceMonitor {
    /// 获取全局监控器（默认阈值 yellow=70 / red=85）
    pub fn global() -> &'static ResourceMonitor {
        MONITOR.get_or_init(|| ResourceMonitor {
            inner: Mutex::new(Inner {
                sys: {
                    let mut s = sysinfo::System::new();
                    s.refresh_memory();
                    s.refresh_cpu_usage();
                    s.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[current_pid()]), true);
                    s
                },
                thresholds: (70.0, 85.0),
                last_sample: None,
                min_interval: Duration::from_millis(500),
            }),
        })
    }

    /// 设置压力阈值（已用百分比）
    pub fn set_thresholds(&self, yellow_pct: f32, red_pct: f32) {
        let mut g = self.inner.lock().unwrap();
        g.thresholds = (yellow_pct.clamp(10.0, 99.0), red_pct.clamp(20.0, 100.0));
        g.last_sample = None; // 阈值变化后强制重采样
    }

    /// 采样当前资源状态（内部节流，500ms 内复用上次结果）
    pub fn snapshot(&self) -> (ResourceSnapshot, MemoryPressure) {
        let mut g = self.inner.lock().unwrap();
        if let Some((at, snap, pressure)) = &g.last_sample {
            if at.elapsed() < g.min_interval {
                return (snap.clone(), *pressure);
            }
        }

        g.sys.refresh_memory();
        g.sys.refresh_cpu_usage();
        g.sys
            .refresh_processes(sysinfo::ProcessesToUpdate::Some(&[current_pid()]), false);

        let total = g.sys.total_memory(); // bytes
        let available = g.sys.available_memory();
        let rss = g
            .sys
            .process(current_pid())
            .map(|p| p.memory())
            .unwrap_or(0);

        let total_mb = total / 1024 / 1024;
        let available_mb = available / 1024 / 1024;
        let used_pct = if total > 0 {
            ((total - available) as f32 / total as f32 * 100.0).clamp(0.0, 100.0)
        } else {
            0.0
        };
        let cpu = g.sys.global_cpu_usage().clamp(0.0, 100.0);

        let snap = ResourceSnapshot {
            total_mem_mb: total_mb,
            available_mem_mb: available_mb,
            mem_used_pct: used_pct,
            process_rss_mb: rss / 1024 / 1024,
            cpu_used_pct: cpu,
        };
        let pressure = snap.pressure(g.thresholds.0, g.thresholds.1);
        g.last_sample = Some((Instant::now(), snap.clone(), pressure));
        (snap, pressure)
    }

    /// 快捷查询当前压力级别
    pub fn pressure(&self) -> MemoryPressure {
        self.snapshot().1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 资源监控_快照可获取() {
        let m = ResourceMonitor::global();
        let (snap, pressure) = m.snapshot();
        assert!(snap.total_mem_mb > 0, "应能读到总内存");
        assert!(snap.mem_used_pct >= 0.0 && snap.mem_used_pct <= 100.0);
        // 只验证级别合法
        let _ = pressure;
    }

    #[test]
    fn 压力分级_阈值判定() {
        let snap = ResourceSnapshot {
            total_mem_mb: 16000,
            available_mem_mb: 1600,
            mem_used_pct: 90.0,
            process_rss_mb: 100,
            cpu_used_pct: 10.0,
        };
        assert_eq!(snap.pressure(70.0, 85.0), MemoryPressure::Red);

        let snap = ResourceSnapshot {
            mem_used_pct: 75.0,
            ..snap
        };
        assert_eq!(snap.pressure(70.0, 85.0), MemoryPressure::Yellow);

        let snap = ResourceSnapshot {
            mem_used_pct: 50.0,
            ..snap
        };
        assert_eq!(snap.pressure(70.0, 85.0), MemoryPressure::Green);
    }
}
