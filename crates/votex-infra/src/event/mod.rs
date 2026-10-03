use std::sync::Mutex;
use votex_domain::event::{DomainEvent, EventBus};

/// 同步事件总线
///
/// 存储一组事件处理器，发布事件时同步调用所有处理器。
/// 适用于 CLI 模式，不依赖异步运行时。
pub struct SyncEventBus {
    handlers: Mutex<Vec<Box<dyn Fn(DomainEvent) + Send + Sync>>>,
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
        let mut handlers = self.handlers.lock().expect("获取事件处理器锁失败");
        handlers.push(Box::new(handler));
    }

    /// 清空所有处理器
    #[allow(dead_code)]
    pub fn clear(&self) {
        let mut handlers = self.handlers.lock().expect("获取事件处理器锁失败");
        handlers.clear();
    }

    /// 获取处理器数量
    #[allow(dead_code)]
    pub fn handler_count(&self) -> usize {
        let handlers = self.handlers.lock().expect("获取事件处理器锁失败");
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
    fn publish(&self, event: DomainEvent) {
        let handlers = self.handlers.lock().expect("获取事件处理器锁失败");
        for handler in handlers.iter() {
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
