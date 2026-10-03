//! 翻译模型会话池
//!
//! # 解决的问题
//!
//! 离线翻译模型体积巨大（HY-MT1.5 Q4 约 1.4GB、NLLB INT8 约 1.9GB），
//! 每次调用 `create_translation_provider` 都会重新走一遍：
//! 读盘 → 图优化（Level3）→ 建 Session，耗时可达数十秒到数分钟。
//!
//! 本模块按引擎名缓存已构建的 Provider，实现「一次加载、反复使用」。
//!
//! # 线程安全
//!
//! 内部使用 `Mutex<HashMap<..>>`；取用后立即释放锁，推理过程不持锁，
//! 因此多线程并发翻译不会互相阻塞。

use super::create_translation_provider_with_dir;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use votex_domain::translation::provider::TranslationProvider;

/// 翻译模型会话池
pub struct TranslationModelPool {
    /// 引擎名 → 已加载的 Provider
    providers: Mutex<HashMap<String, Arc<dyn TranslationProvider>>>,
    /// 模型根目录
    models_dir: PathBuf,
}

impl TranslationModelPool {
    /// 创建会话池
    pub fn new(models_dir: impl Into<PathBuf>) -> Self {
        Self {
            providers: Mutex::new(HashMap::new()),
            models_dir: models_dir.into(),
        }
    }

    /// 使用默认模型目录 `models/`
    pub fn with_default_dir() -> Self {
        Self::new(PathBuf::from("models"))
    }

    /// 获取（必要时加载）指定引擎的 Provider
    ///
    /// 同名引擎只会加载一次，后续调用直接返回缓存实例。
    pub fn get(&self, name: &str) -> Result<Arc<dyn TranslationProvider>, String> {
        let key = super::normalize_engine_name(name);

        // 先查缓存（快路径）
        if let Some(provider) = self.providers.lock().ok().and_then(|m| m.get(&key).cloned()) {
            return Ok(provider);
        }

        tracing::info!("翻译会话池：首次加载引擎 {}", key);
        let provider: Arc<dyn TranslationProvider> =
            create_translation_provider_with_dir(&key, Path::new(&self.models_dir))?.into();

        if let Ok(mut map) = self.providers.lock() {
            map.insert(key, Arc::clone(&provider));
        }
        Ok(provider)
    }

    /// 丢弃指定引擎的缓存（模型文件更新后调用）
    pub fn invalidate(&self, name: &str) {
        let key = super::normalize_engine_name(name);
        if let Ok(mut map) = self.providers.lock() {
            if let Some(provider) = map.remove(&key) {
                tracing::info!("翻译会话池：已释放引擎 {}", key);
                drop(provider);
            }
        }
    }

    /// 清空全部缓存
    pub fn clear(&self) {
        if let Ok(mut map) = self.providers.lock() {
            let count = map.len();
            map.clear();
            if count > 0 {
                tracing::info!("翻译会话池：已清空 {} 个引擎", count);
            }
        }
    }

    /// 已缓存的引擎数量
    pub fn cached_count(&self) -> usize {
        self.providers.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// 模型根目录
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// 已缓存的引擎名列表
    pub fn cached_engines(&self) -> Vec<String> {
        self.providers
            .lock()
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 会话池_同名引擎只创建一次() {
        let pool = TranslationModelPool::new(std::env::temp_dir());
        let a = pool.get("dict").unwrap();
        let b = pool.get("dict").unwrap();
        assert_eq!(pool.cached_count(), 1, "同名引擎应命中缓存");
        assert_eq!(a.name(), b.name());
        assert!(Arc::ptr_eq(&a, &b), "两次取用应返回同一个 Arc 实例");
    }

    #[test]
    fn 会话池_别名归一化到同一缓存项() {
        let pool = TranslationModelPool::new(std::env::temp_dir());
        pool.get("dict").unwrap();
        pool.get("dictionary").unwrap();
        assert_eq!(pool.cached_count(), 1, "别名应归一化到同一缓存项");
    }

    #[test]
    fn 会话池_失效后重新创建() {
        let pool = TranslationModelPool::new(std::env::temp_dir());
        pool.get("dict").unwrap();
        assert_eq!(pool.cached_count(), 1);
        pool.invalidate("dict");
        assert_eq!(pool.cached_count(), 0);
        pool.get("dict").unwrap();
        assert_eq!(pool.cached_count(), 1);
    }

    #[test]
    fn 会话池_清空全部缓存() {
        let pool = TranslationModelPool::new(std::env::temp_dir());
        pool.get("dict").unwrap();
        pool.clear();
        assert_eq!(pool.cached_count(), 0);
        assert!(pool.cached_engines().is_empty());
    }

    #[test]
    fn 会话池_未知引擎返回错误且不计入缓存() {
        let pool = TranslationModelPool::new(std::env::temp_dir());
        assert!(pool.get("no-such-engine").is_err());
        assert_eq!(pool.cached_count(), 0);
    }
}
