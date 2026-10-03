//! 翻译运行时（进程级共享）
//!
//! # 为什么需要
//!
//! GUI 每次触发翻译都会新建一个用例，若每个用例各自持有一个
//! [`TranslationModelPool`]，会话池就形同虚设——点一次翻译仍然重新加载
//! 数 GB 的 ONNX 模型。
//!
//! 本模块把「模型会话池」与「翻译缓存」提升为**进程级单例**，
//! 所有用例共享，做到真正的「一次加载、反复使用」。

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use votex_infra::translation::cache::TranslationCache;
use votex_infra::translation::TranslationModelPool;

static POOL: OnceLock<Arc<TranslationModelPool>> = OnceLock::new();
static CACHE: OnceLock<Arc<TranslationCache>> = OnceLock::new();

/// 使用指定模型目录初始化运行时
///
/// 重复调用只有第一次生效（返回已初始化的实例）。
/// 应在应用启动时、读取到配置后立即调用。
pub fn init(models_dir: impl Into<PathBuf>) -> Arc<TranslationModelPool> {
    let pool = Arc::new(TranslationModelPool::new(models_dir.into()));
    // 已初始化则保留原实例
    let pool = POOL.get_or_init(|| pool);
    let _ = CACHE.get_or_init(|| Arc::new(TranslationCache::with_default_capacity()));
    Arc::clone(pool)
}

/// 获取共享会话池（未显式初始化时使用默认目录 `models/`）
pub fn pool() -> Arc<TranslationModelPool> {
    POOL.get_or_init(|| Arc::new(TranslationModelPool::new(Path::new("models"))))
        .clone()
}

/// 获取共享翻译缓存
pub fn cache() -> Arc<TranslationCache> {
    CACHE
        .get_or_init(|| Arc::new(TranslationCache::with_default_capacity()))
        .clone()
}

/// 用指定容量初始化缓存（未初始化时生效）
pub fn init_cache(capacity: usize) -> Arc<TranslationCache> {
    CACHE
        .get_or_init(|| Arc::new(TranslationCache::new(capacity)))
        .clone()
}

/// 是否已显式初始化
pub fn initialized() -> bool {
    POOL.get().is_some()
}

/// 释放全部已加载的模型（切换模型目录或更新模型文件后调用）
pub fn release_all() {
    if let Some(p) = POOL.get() {
        p.clear();
    }
}

/// 释放指定引擎的模型
pub fn release(engine: &str) {
    if let Some(p) = POOL.get() {
        p.invalidate(engine);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 运行时_缓存为单例() {
        let a = cache();
        let b = cache();
        assert!(Arc::ptr_eq(&a, &b), "缓存应为进程级单例");
    }

    #[test]
    fn 运行时_会话池为单例() {
        let a = pool();
        let b = pool();
        assert!(Arc::ptr_eq(&a, &b), "会话池应为进程级单例");
    }

    #[test]
    fn 运行时_不同用例共享会话池缓存项() {
        // 用一个不会真实加载模型的引擎验证共享性
        let p = pool();
        p.get("dict").unwrap();
        assert!(pool().cached_count() >= 1, "二次取用应命中同一会话池");
    }

    #[test]
    fn 运行时_释放全部模型() {
        pool().get("dict").unwrap();
        release_all();
        assert_eq!(pool().cached_count(), 0);
    }

    #[test]
    fn 运行时_释放指定引擎() {
        pool().get("dict").unwrap();
        release("dict");
        assert_eq!(pool().cached_count(), 0);
    }
}
