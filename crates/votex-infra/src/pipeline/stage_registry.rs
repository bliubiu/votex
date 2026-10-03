use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use votex_domain::pipeline::handler::{StageHandler, StageRegistry};
use votex_domain::pipeline::value_object::StageKind;

/// 默认阶段注册表
///
/// 基于 Mutex + HashMap + Arc 的线程安全阶段处理器注册表。
/// 支持动态注册和查询阶段处理器。
pub struct DefaultStageRegistry {
    handlers: Mutex<HashMap<StageKind, Arc<dyn StageHandler>>>,
}

impl DefaultStageRegistry {
    pub fn new() -> Self {
        Self {
            handlers: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for DefaultStageRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl StageRegistry for DefaultStageRegistry {
    fn register(&mut self, handler: Arc<dyn StageHandler>) {
        let kind = handler.kind();
        if let Ok(mut handlers) = self.handlers.lock() {
            handlers.insert(kind, handler);
        }
    }

    fn get(&self, kind: StageKind) -> Option<Arc<dyn StageHandler>> {
        self.handlers.lock().ok().and_then(|h| h.get(&kind).cloned())
    }

    fn registered_kinds(&self) -> Vec<StageKind> {
        self.handlers
            .lock()
            .map(|h| h.keys().copied().collect())
            .unwrap_or_default()
    }
}

/// 创建线程安全的共享注册表
pub type SharedStageRegistry = Arc<Mutex<DefaultStageRegistry>>;

pub fn new_shared_registry() -> SharedStageRegistry {
    Arc::new(Mutex::new(DefaultStageRegistry::new()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use votex_domain::pipeline::entity::Stage;
    use votex_domain::pipeline::handler::{StageHandler, StageParams};
    use votex_domain::pipeline::value_object::StageKind;
    use votex_domain::shared::value_object::ProgressEvent;

    struct MockHandler {
        kind: StageKind,
    }

    impl StageHandler for MockHandler {
        fn kind(&self) -> StageKind {
            self.kind
        }

        fn execute(
            &self,
            _stage: &Stage,
            params: StageParams,
            _progress: &dyn Fn(ProgressEvent),
        ) -> Result<StageParams, Box<dyn std::error::Error + Send>> {
            Ok(params)
        }
    }

    #[test]
    fn default_registry_注册和获取() {
        let mut registry = DefaultStageRegistry::new();
        registry.register(Arc::new(MockHandler {
            kind: StageKind::TextPreprocess,
        }));

        let handler = registry.get(StageKind::TextPreprocess);
        assert!(handler.is_some());
        assert_eq!(handler.unwrap().kind(), StageKind::TextPreprocess);
    }

    #[test]
    fn default_registry_不存在的阶段返回None() {
        let registry = DefaultStageRegistry::new();
        assert!(registry.get(StageKind::TtsSynthesize).is_none());
    }

    #[test]
    fn default_registry_列出已注册阶段() {
        let mut registry = DefaultStageRegistry::new();
        registry.register(Arc::new(MockHandler {
            kind: StageKind::TextPreprocess,
        }));
        registry.register(Arc::new(MockHandler {
            kind: StageKind::TtsSynthesize,
        }));

        let kinds = registry.registered_kinds();
        assert_eq!(kinds.len(), 2);
        assert!(kinds.contains(&StageKind::TextPreprocess));
        assert!(kinds.contains(&StageKind::TtsSynthesize));
    }
}
