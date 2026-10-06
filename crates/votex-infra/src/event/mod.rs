//! 事件基础设施：同步事件总线。
//!
//! `publish` 必须「锁内快照处理器列表、锁外回调」—— 锁内回调会在处理器
//! 反向注册/注销时自死锁，并把锁毒化扩散到全局。

use std::sync::{Arc, Mutex};
use votex_domain::event::{DomainEvent, EventBus};

/// 事件处理器类型
///
/// 使用 `Arc` 而非 `Box`，以便在发布时克隆出快照、在锁外回调
type Handler = Arc<dyn Fn(DomainEvent) + Send + Sync>;

/// 同步事件总线
///
/// 存储一组事件处理器，发布事件时同步调用所有处理器。
/// 适用于 CLI 模式，不依赖异步运行时。
pub struct SyncEventBus {
    handlers: Mutex<Vec<Handler>>,
}

impl SyncEventBus {
    /// 创建空的事件总线
    pub fn new() -> Self {
        Self {
            handlers: Mutex::new(Vec::new()),
        }
    }

    /// 注册事件处理器
    ///
    /// # 示例
    /// ```ignore
    /// use votex_infra::event::SyncEventBus;
    /// use votex_domain::event::EventBus;
    ///
    /// let bus = SyncEventBus::new();
    /// bus.subscribe(|event| {
    ///     println!("收到事件: {:?}", event);
    /// });
    /// ```
    pub fn subscribe<F>(&self, handler: F)
    where
        F: Fn(DomainEvent) + Send + Sync + 'static,
    {
        // 锁中毒时恢复使用：一次 handler panic 不应让事件总线永久不可用
        let mut handlers = self.handlers.lock().unwrap_or_else(|e| e.into_inner());
        handlers.push(Arc::new(handler));
    }

    /// 清空所有处理器
        pub fn clear(&self) {
        let mut handlers = self.handlers.lock().unwrap_or_else(|e| e.into_inner());
        handlers.clear();
    }

    /// 获取处理器数量
        pub fn handler_count(&self) -> usize {
        let handlers = self.handlers.lock().unwrap_or_else(|e| e.into_inner());
        handlers.len()
    }
}

impl Default for SyncEventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBus for SyncEventBus {
    /// 发布事件，同步调用所有已注册的处理器
    ///
    /// **关键约束：锁内只取快照，锁外才回调。**
    ///
    /// 若在持锁状态下调用 handler 会有两个严重后果：
    /// 1. 任何 handler 内部再调用 `subscribe()`（GUI 中很常见：响应 A 事件后
    ///    注册 B 的回调）会立即**自死锁**
    /// 2. 任一 handler panic 会毒化该互斥锁，之后 `publish` / `subscribe` /
    ///    `handler_count` 全部 panic，整个应用不可恢复
    fn publish(&self, event: DomainEvent) {
        let snapshot: Vec<Handler> = {
            let handlers = self.handlers.lock().unwrap_or_else(|e| e.into_inner());
            handlers.iter().map(Arc::clone).collect()
        };

        for handler in snapshot {
            handler(event.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn 事件总线发布事件() {
        let bus = SyncEventBus::new();
        let counter = std::sync::Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let task_id = votex_domain::shared::value_object::TaskId::new();
        let event = DomainEvent::TtsTaskCompleted {
            task_id: task_id.clone(),
        };

        bus.subscribe(move |e| {
            if matches!(e, DomainEvent::TtsTaskCompleted { .. }) {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }
        });

        bus.publish(event);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn 事件总线多个处理器() {
        let bus = SyncEventBus::new();
        let counter = std::sync::Arc::new(AtomicUsize::new(0));
        let c1 = counter.clone();
        let c2 = counter.clone();

        let task_id = votex_domain::shared::value_object::TaskId::new();
        let event = DomainEvent::AsrTaskCompleted {
            task_id,
        };

        bus.subscribe(move |_| { c1.fetch_add(1, Ordering::SeqCst); });
        bus.subscribe(move |_| { c2.fetch_add(1, Ordering::SeqCst); });

        bus.publish(event);
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn 事件总线处理器数量和清空() {
        let bus = SyncEventBus::new();
        assert_eq!(bus.handler_count(), 0);

        bus.subscribe(|_| {});
        assert_eq!(bus.handler_count(), 1);

        bus.clear();
        assert_eq!(bus.handler_count(), 0);
    }
}
