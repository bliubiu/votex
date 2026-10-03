//! 翻译用例
//!
//! 对外提供统一的翻译入口，内部复用 [`TranslationPipeline`] 完成
//! 术语表处理、长文本分段、缓存与简繁转换。

use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use votex_domain::error::TranslationError;
use votex_domain::translation::glossary::Glossary;
use votex_domain::translation::options::TranslationOptions;
use votex_domain::translation::script::ChineseScript;
use votex_domain::translation::value_object::TranslationDirection;
use votex_infra::translation::cache::TranslationCache;
use votex_infra::translation::TranslationModelPool;

use crate::services::translation_pipeline::{TranslationOutcome, TranslationPipeline};

/// 翻译用例
pub struct TranslationUseCase {
    pipeline: TranslationPipeline,
    /// 模型会话池（进程内共享，避免重复加载）
    pool: Arc<TranslationModelPool>,
    /// 翻译缓存（进程内共享）
    cache: Arc<TranslationCache>,
}

impl TranslationUseCase {
    /// 创建翻译用例
    ///
    /// 复用进程级共享的会话池与缓存（见 [`crate::services::translation_runtime`]），
    /// 因此多次创建用例不会重复加载模型。
    pub fn new(engine: &str) -> Result<Self> {
        let pool = crate::services::translation_runtime::pool();
        let cache = crate::services::translation_runtime::cache();
        Self::with_pool(pool, cache, engine)
    }

    /// 创建翻译用例，指定模型根目录
    ///
    /// 会先用该目录初始化共享运行时，再复用它。
    pub fn with_models_dir(engine: &str, models_dir: &Path) -> Result<Self> {
        let pool =
            crate::services::translation_runtime::init(models_dir.to_path_buf());
        let cache = crate::services::translation_runtime::cache();
        Self::with_pool(pool, cache, engine)
    }

    /// 复用已有的会话池创建用例（推荐：多个用例共享同一池，避免重复加载模型）
    pub fn with_pool(
        pool: Arc<TranslationModelPool>,
        cache: Arc<TranslationCache>,
        engine: &str,
    ) -> Result<Self> {
        let provider = pool
            .get(engine)
            .map_err(|e| anyhow::anyhow!("创建翻译引擎失败: {}", e))?;
        Ok(Self {
            pipeline: TranslationPipeline::new(provider, Some(Arc::clone(&cache))),
            pool,
            cache,
        })
    }

    /// 切换到另一个引擎（复用同一个会话池与缓存）
    pub fn switch_engine(&mut self, engine: &str) -> Result<()> {
        let provider = self
            .pool
            .get(engine)
            .map_err(|e| anyhow::anyhow!("创建翻译引擎失败: {}", e))?;
        self.pipeline = TranslationPipeline::new(provider, Some(Arc::clone(&self.cache)));
        Ok(())
    }

    /// 翻译文本（简单入口）
    ///
    /// `direction` 支持：`zh-en` / `en-zh` / `auto` / `zh->en` / `ja->zh` 等。
    pub fn translate(&self, text: &str, direction: &str) -> Result<String> {
        let options = TranslationOptions::new(Self::parse_direction(direction)?);
        Ok(self.pipeline.translate(text, &options)?.text)
    }

    /// 翻译文本（完整选项）
    pub fn translate_with_options(
        &self,
        text: &str,
        options: &TranslationOptions,
    ) -> Result<TranslationOutcome> {
        Ok(self.pipeline.translate(text, options)?)
    }

    /// 带术语表翻译
    pub fn translate_with_glossary(
        &self,
        text: &str,
        direction: &str,
        glossary: Glossary,
    ) -> Result<String> {
        let options =
            TranslationOptions::new(Self::parse_direction(direction)?).with_glossary(glossary);
        Ok(self.pipeline.translate(text, &options)?.text)
    }

    /// 批量翻译
    pub fn translate_batch(&self, texts: &[String], direction: &str) -> Result<Vec<String>> {
        let options = TranslationOptions::new(Self::parse_direction(direction)?);
        let outcomes = self.pipeline.translate_batch(texts, &options)?;
        Ok(outcomes.into_iter().map(|o| o.text).collect())
    }

    /// 解析翻译方向
    ///
    /// 同时兼容 `-` 与 `->` 两种分隔符，与领域层
    /// [`TranslationDirection::from_str`] 行为一致。
    pub fn parse_direction(direction: &str) -> Result<TranslationDirection> {
        // 先把 "->" 归一成 "-"，再交给领域层统一解析，避免两套解析逻辑分叉
        let normalized = direction.replace("->", "-").replace("→", "-");
        TranslationDirection::from_str(&normalized)
            .ok_or_else(|| anyhow::anyhow!("不支持的翻译方向: {}", direction))
    }

    /// 获取引擎名称
    pub fn engine_name(&self) -> &str {
        self.pipeline.engine_name()
    }

    /// 底层流水线（需要分段/流式/进度等高级能力时使用）
    pub fn pipeline(&self) -> &TranslationPipeline {
        &self.pipeline
    }

    /// 引擎是否已加载模型（在线引擎恒为 true）
    pub fn is_loaded(&self) -> bool {
        self.pipeline.provider().is_loaded()
    }

    /// 缓存命中率（0.0 ~ 1.0）
    pub fn cache_hit_rate(&self) -> f64 {
        self.cache.hit_rate()
    }

    /// 清空翻译缓存
    pub fn clear_cache(&self) {
        self.cache.clear();
    }

    /// 释放指定引擎的模型（模型文件更新后调用）
    pub fn reload_engine(&self, engine: &str) {
        self.pool.invalidate(engine);
    }

    /// 模型根目录
    pub fn models_dir(&self) -> PathBuf {
        self.pool.models_dir().to_path_buf()
    }
}

/// 从 JSON 文件加载术语表
pub fn load_glossary_from_file(path: &Path) -> Result<Glossary> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("读取术语表文件失败 {:?}: {}", path, e))?;
    Glossary::from_json(&content).map_err(|e| anyhow::anyhow!("解析术语表失败: {}", e))
}

/// 根据目标语言代码推断书写系统
pub fn script_for_target(target: &str) -> ChineseScript {
    ChineseScript::from_lang_code(target)
}

/// 把领域错误转换为可读文案
pub fn describe_error(err: &TranslationError) -> String {
    format!("[{}] {}", err.error_code(), err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 方向解析_横线格式() {
        assert_eq!(
            TranslationUseCase::parse_direction("zh-en").unwrap(),
            TranslationDirection::ZhToEn
        );
        assert_eq!(
            TranslationUseCase::parse_direction("en-zh").unwrap(),
            TranslationDirection::EnToZh
        );
        assert_eq!(
            TranslationUseCase::parse_direction("auto").unwrap(),
            TranslationDirection::Auto
        );
    }

    #[test]
    fn 方向解析_箭头格式与横线格式结果一致() {
        assert_eq!(
            TranslationUseCase::parse_direction("zh->en").unwrap(),
            TranslationUseCase::parse_direction("zh-en").unwrap(),
            "两种分隔符必须解析出相同结果"
        );
        assert_eq!(
            TranslationUseCase::parse_direction("ja->zh").unwrap(),
            TranslationUseCase::parse_direction("ja-zh").unwrap()
        );
    }

    #[test]
    fn 方向解析_unicode箭头() {
        assert_eq!(
            TranslationUseCase::parse_direction("ja→zh").unwrap(),
            TranslationDirection::ByLanguagePair {
                source: "ja".into(),
                target: "zh".into()
            }
        );
    }

    #[test]
    fn 方向解析_非法输入报错() {
        assert!(TranslationUseCase::parse_direction("这是一个非常长的非法方向字符串").is_err());
    }

    #[test]
    fn 书写系统推断() {
        assert_eq!(script_for_target("zh-Hant"), ChineseScript::Traditional);
        assert_eq!(script_for_target("zh-Hans"), ChineseScript::Simplified);
        assert_eq!(script_for_target("en"), ChineseScript::None);
    }

    #[test]
    fn 错误文案含错误码() {
        let e = TranslationError::EmptyText;
        assert!(describe_error(&e).contains("VTR004"));
    }

    #[test]
    fn 词典引擎可用且无需模型() {
        let uc = TranslationUseCase::new("dict").unwrap();
        assert_eq!(uc.engine_name(), "dict");
        assert!(uc.is_loaded());
    }

    #[test]
    fn 用例_批量翻译() {
        let uc = TranslationUseCase::new("dict").unwrap();
        let texts = vec!["你好".to_string(), "世界".to_string()];
        let out = uc.translate_batch(&texts, "zh-en").unwrap();
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn 用例_术语表生效() {
        let uc = TranslationUseCase::new("dict").unwrap();
        let glossary = Glossary::from_entries(vec![
            votex_domain::translation::glossary::GlossaryEntry::new("你好", "Hi"),
        ]);
        let out = uc.translate_with_glossary("你好", "zh-en", glossary).unwrap();
        assert_eq!(out, "Hi");
    }

    #[test]
    fn 用例_切换引擎复用会话池() {
        let pool = Arc::new(TranslationModelPool::new(std::env::temp_dir()));
        let cache = Arc::new(TranslationCache::new(10));
        let mut uc = TranslationUseCase::with_pool(Arc::clone(&pool), Arc::clone(&cache), "dict")
            .unwrap();
        assert_eq!(uc.engine_name(), "dict");
        uc.switch_engine("dictionary").unwrap();
        assert_eq!(uc.engine_name(), "dict");
        assert_eq!(pool.cached_count(), 1, "别名引擎应复用同一缓存项");
    }

    #[test]
    fn 术语表文件加载_文件不存在时报错() {
        assert!(load_glossary_from_file(Path::new("/nonexistent/glossary.json")).is_err());
    }

    #[test]
    fn 术语表文件加载_往返() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g.json");
        let g = Glossary::from_entries(vec![
            votex_domain::translation::glossary::GlossaryEntry::new("甲", "A"),
        ]);
        std::fs::write(&path, g.to_json().unwrap()).unwrap();
        let loaded = load_glossary_from_file(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.entries()[0].dst, "A");
    }
}
