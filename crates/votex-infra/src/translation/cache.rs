//! 翻译结果缓存
//!
//! # 为什么需要
//!
//! 同一段文本（如批量任务中的重复段落、GUI 反复试译）重复走一遍模型推理
//! 纯属浪费：离线模型单句数百毫秒到数秒，在线 API 还要花钱。
//!
//! # 缓存键
//!
//! `引擎名 + 方向 + 术语表指纹 + 原文` 的哈希。
//! 术语表指纹保证「换了一套术语表，缓存自动失效」。
//!
//! # 淘汰策略
//!
//! LRU（最近最少使用），容量可配。非加密哈希（DefaultHasher），仅用于缓存键。

use std::collections::{HashMap, VecDeque};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Mutex;

/// 翻译结果缓存
pub struct TranslationCache {
    entries: Mutex<HashMap<u64, String>>,
    /// 访问顺序队列（队首为最久未使用）
    order: Mutex<VecDeque<u64>>,
    capacity: usize,
    hits: Mutex<usize>,
    misses: Mutex<usize>,
}

impl TranslationCache {
    /// 创建缓存
    ///
    /// `capacity` 为 0 时表示不缓存。
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            order: Mutex::new(VecDeque::new()),
            capacity,
            hits: Mutex::new(0),
            misses: Mutex::new(0),
        }
    }

    /// 使用默认容量（2000 条）
    pub fn with_default_capacity() -> Self {
        Self::new(2000)
    }

    /// 容量
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// 是否启用
    pub fn enabled(&self) -> bool {
        self.capacity > 0
    }

    /// 计算缓存键
    ///
    /// 参数与 `get` 保持一致，便于外部判断命中。
    pub fn make_key(engine: &str, direction: &str, glossary_fingerprint: u64, text: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        engine.hash(&mut hasher);
        direction.hash(&mut hasher);
        glossary_fingerprint.hash(&mut hasher);
        text.hash(&mut hasher);
        hasher.finish()
    }

    /// 术语表指纹（空术语表为 0）
    pub fn glossary_fingerprint(glossary_text: &str) -> u64 {
        if glossary_text.is_empty() {
            return 0;
        }
        let mut hasher = DefaultHasher::new();
        glossary_text.hash(&mut hasher);
        hasher.finish()
    }

    /// 查询缓存
    pub fn get(&self, key: u64) -> Option<String> {
        if !self.enabled() {
            return None;
        }
        let value = self.entries.lock().ok().and_then(|m| m.get(&key).cloned());
        match value {
            Some(v) => {
                self.touch(key);
                if let Ok(mut h) = self.hits.lock() {
                    *h += 1;
                }
                Some(v)
            }
            None => {
                if let Ok(mut m) = self.misses.lock() {
                    *m += 1;
                }
                None
            }
        }
    }

    /// 写入缓存
    pub fn put(&self, key: u64, value: String) {
        if !self.enabled() {
            return;
        }
        if let Ok(mut m) = self.entries.lock() {
            m.insert(key, value);
        }
        self.touch(key);
        self.evict_if_needed();
    }

    /// 记录访问，把 key 移到队列末尾（最近使用）
    fn touch(&self, key: u64) {
        if let Ok(mut order) = self.order.lock() {
            order.retain(|k| *k != key);
            order.push_back(key);
        }
    }

    /// 超出容量时淘汰最久未使用的条目
    fn evict_if_needed(&self) {
        let mut order = match self.order.lock() {
            Ok(o) => o,
            Err(_) => return,
        };
        while order.len() > self.capacity {
            let Some(victim) = order.pop_front() else {
                break;
            };
            if let Ok(mut m) = self.entries.lock() {
                m.remove(&victim);
            }
        }
    }

    /// 当前条目数
    pub fn len(&self) -> usize {
        self.entries.lock().map(|m| m.len()).unwrap_or(0)
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// 清空
    pub fn clear(&self) {
        if let Ok(mut m) = self.entries.lock() {
            m.clear();
        }
        if let Ok(mut o) = self.order.lock() {
            o.clear();
        }
    }

    /// 命中次数
    pub fn hits(&self) -> usize {
        self.hits.lock().map(|h| *h).unwrap_or(0)
    }

    /// 未命中次数
    pub fn misses(&self) -> usize {
        self.misses.lock().map(|m| *m).unwrap_or(0)
    }

    /// 命中率（0.0 ~ 1.0），无请求时为 0.0
    pub fn hit_rate(&self) -> f64 {
        let hits = self.hits();
        let misses = self.misses();
        let total = hits + misses;
        if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 缓存_基本存取() {
        let cache = TranslationCache::new(10);
        let key = TranslationCache::make_key("dict", "zh-en", 0, "你好");
        assert!(cache.get(key).is_none());
        cache.put(key, "hello".to_string());
        assert_eq!(cache.get(key).as_deref(), Some("hello"));
    }

    #[test]
    fn 缓存_不同引擎不互相命中() {
        let a = TranslationCache::make_key("dict", "zh-en", 0, "你好");
        let b = TranslationCache::make_key("opus-mt", "zh-en", 0, "你好");
        assert_ne!(a, b);
    }

    #[test]
    fn 缓存_术语表变化导致键变化() {
        let fp1 = TranslationCache::glossary_fingerprint("甲->A");
        let fp2 = TranslationCache::glossary_fingerprint("甲->B");
        let a = TranslationCache::make_key("dict", "zh-en", fp1, "你好");
        let b = TranslationCache::make_key("dict", "zh-en", fp2, "你好");
        assert_ne!(a, b, "术语表变了缓存键必须变");
    }

    #[test]
    fn 缓存_空术语表指纹为零() {
        assert_eq!(TranslationCache::glossary_fingerprint(""), 0);
    }

    #[test]
    fn 缓存_超出容量淘汰最久未使用() {
        let cache = TranslationCache::new(2);
        let k1 = TranslationCache::make_key("e", "d", 0, "1");
        let k2 = TranslationCache::make_key("e", "d", 0, "2");
        let k3 = TranslationCache::make_key("e", "d", 0, "3");
        cache.put(k1, "a".into());
        cache.put(k2, "b".into());
        // 访问 k1，使其成为最近使用
        assert_eq!(cache.get(k1).as_deref(), Some("a"));
        cache.put(k3, "c".into());

        assert_eq!(cache.len(), 2);
        assert!(cache.get(k2).is_none(), "k2 是最久未使用的，应被淘汰");
        assert!(cache.get(k1).is_some());
        assert!(cache.get(k3).is_some());
    }

    #[test]
    fn 缓存_容量为零时不缓存() {
        let cache = TranslationCache::new(0);
        assert!(!cache.enabled());
        let k = TranslationCache::make_key("e", "d", 0, "x");
        cache.put(k, "v".into());
        assert!(cache.get(k).is_none());
        assert!(cache.is_empty());
    }

    #[test]
    fn 缓存_统计命中率() {
        let cache = TranslationCache::new(10);
        let k = TranslationCache::make_key("e", "d", 0, "x");
        cache.get(k); // miss
        cache.put(k, "v".into());
        cache.get(k); // hit
        assert_eq!(cache.hits(), 1);
        assert_eq!(cache.misses(), 1);
        assert!((cache.hit_rate() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn 缓存_清空() {
        let cache = TranslationCache::new(10);
        let k = TranslationCache::make_key("e", "d", 0, "x");
        cache.put(k, "v".into());
        assert_eq!(cache.len(), 1);
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn 缓存_无请求时命中率为零() {
        let cache = TranslationCache::new(10);
        assert_eq!(cache.hit_rate(), 0.0);
    }
}
