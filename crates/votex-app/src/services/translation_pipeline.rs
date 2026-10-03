//! 翻译流水线
//!
//! 把「翻译质量工程」的各个环节编排成一条固定管线：
//!
//! ```text
//! 原文
//!  → 术语表前处理（专有名词 → 占位符）
//!  → 长文本分段
//!  → 逐段：查缓存 → 引擎翻译（LLM 另附术语表与上下文提示）
//!  → 拼接 → 术语表后处理（占位符 → 指定译文）→ 简繁转换
//!  → 译文
//! ```
//!
//! 关键在于：**无论底层引擎是否支持术语表，专有名词都能保持一致**。
//! 支持 prompt 的引擎额外得到术语提示；不支持的引擎只看到占位符，
//! 无从「自由发挥」，回填阶段再把占位符换成用户指定的译法。

use std::sync::Arc;
use votex_domain::error::TranslationError;
use votex_domain::translation::glossary::GlossaryContext;
use votex_domain::translation::options::{TranslationHints, TranslationOptions};
use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::script::ChineseConverter;
use votex_domain::translation::segment::{SegmentConfig, TextSegmenter};
use votex_domain::translation::value_object::TranslationDirection;
use votex_infra::translation::cache::TranslationCache;

/// 翻译结果
#[derive(Debug, Clone)]
pub struct TranslationOutcome {
    /// 最终译文
    pub text: String,
    /// 切分出的片段数（1 表示未分段）
    pub segments: usize,
    /// 命中缓存的片段数
    pub cached_segments: usize,
    /// 是否启用了术语表
    pub glossary_used: bool,
}

/// 翻译流水线
pub struct TranslationPipeline {
    provider: Arc<dyn TranslationProvider>,
    cache: Option<Arc<TranslationCache>>,
    converter: ChineseConverter,
}

impl TranslationPipeline {
    /// 创建流水线
    pub fn new(
        provider: Arc<dyn TranslationProvider>,
        cache: Option<Arc<TranslationCache>>,
    ) -> Self {
        Self {
            provider,
            cache,
            converter: ChineseConverter::new(),
        }
    }

    /// 不带缓存的流水线
    pub fn without_cache(provider: Arc<dyn TranslationProvider>) -> Self {
        Self::new(provider, None)
    }

    /// 底层引擎
    pub fn provider(&self) -> &Arc<dyn TranslationProvider> {
        &self.provider
    }

    /// 引擎名
    pub fn engine_name(&self) -> &str {
        self.provider.name()
    }

    /// 翻译文本
    pub fn translate(
        &self,
        text: &str,
        options: &TranslationOptions,
    ) -> Result<TranslationOutcome, TranslationError> {
        let mut sink = NoopSink;
        self.run(text, options, &mut sink)
    }

    /// 翻译文本（带进度回调，进度以片段为单位）
    pub fn translate_with_progress(
        &self,
        text: &str,
        options: &TranslationOptions,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<TranslationOutcome, TranslationError> {
        let mut sink = ProgressSink { on_progress };
        self.run(text, options, &mut sink)
    }

    /// 流式翻译
    ///
    /// 每完成一个片段就把「已累积的译文」回调一次，便于 GUI 实时刷新。
    /// 单片段（未分段）时只回调一次，即最终译文。
    pub fn translate_stream(
        &self,
        text: &str,
        options: &TranslationOptions,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<TranslationOutcome, TranslationError> {
        let mut sink = StreamSink { on_chunk };
        self.run(text, options, &mut sink)
    }

    /// 批量翻译
    ///
    /// 每条文本独立走完整管线（各自查缓存、各自回填术语）。
    pub fn translate_batch(
        &self,
        texts: &[String],
        options: &TranslationOptions,
    ) -> Result<Vec<TranslationOutcome>, TranslationError> {
        let mut noop = |_: usize, _: usize| {};
        self.translate_batch_with_progress(texts, options, &mut noop)
    }

    /// 批量翻译（进度以「条目」为单位）
    pub fn translate_batch_with_progress(
        &self,
        texts: &[String],
        options: &TranslationOptions,
        on_progress: &mut dyn FnMut(usize, usize),
    ) -> Result<Vec<TranslationOutcome>, TranslationError> {
        let total = texts.len();
        let mut results = Vec::with_capacity(total);
        for (index, text) in texts.iter().enumerate() {
            results.push(self.translate(text, options)?);
            on_progress(index + 1, total);
        }
        Ok(results)
    }

    /// 管线主流程
    fn run(
        &self,
        text: &str,
        options: &TranslationOptions,
        sink: &mut dyn Sink,
    ) -> Result<TranslationOutcome, TranslationError> {
        let source = text.trim();
        if source.is_empty() {
            return Err(TranslationError::EmptyText);
        }

        if !self.provider.supports(&options.direction) {
            return Err(TranslationError::UnsupportedDirection);
        }

        // ---- 1. 术语表前处理：专有名词 → 占位符 ----
        let (processed, glossary_ctx) = options.glossary.apply_before(source);
        let glossary_used = !glossary_ctx.is_empty();

        // ---- 2. 构造提示（术语表 + 历史上下文）----
        let hints = self.build_hints(options, &glossary_ctx);

        // ---- 3. 长文本分段 ----
        let max_chars = self.resolve_max_chars(options);
        let segments = if source.chars().count() > max_chars {
            TextSegmenter::segment(&processed, &SegmentConfig::with_max_chars(max_chars))
        } else {
            vec![processed.clone()]
        };
        if segments.is_empty() {
            return Err(TranslationError::SegmentFailed("分段结果为空".to_string()));
        }

        // ---- 4. 逐段翻译 ----
        let direction = &options.direction;
        let fingerprint = TranslationCache::glossary_fingerprint(&hints.glossary_text);
        let total = segments.len();
        let mut parts: Vec<String> = Vec::with_capacity(total);
        let mut cached_segments = 0usize;

        for (index, segment) in segments.iter().enumerate() {
            let translated = self.translate_one(
                segment,
                direction,
                &hints,
                options.use_cache,
                fingerprint,
                &mut cached_segments,
            )?;
            parts.push(translated);

            // 片段完成：汇报进度，并输出「已累积的译文」（含术语回填与简繁转换）
            let partial = self.finish(&parts, options, &glossary_ctx);
            sink.on_segment(&partial, index + 1, total);
        }

        let final_text = self.finish(&parts, options, &glossary_ctx);

        Ok(TranslationOutcome {
            text: final_text,
            segments: total,
            cached_segments,
            glossary_used,
        })
    }

    /// 拼接 → 术语回填 → 简繁转换
    fn finish(
        &self,
        parts: &[String],
        options: &TranslationOptions,
        glossary_ctx: &GlossaryContext,
    ) -> String {
        let joined = TextSegmenter::join(parts);
        let restored = options.glossary.apply_after(&joined, glossary_ctx);
        self.converter.convert(&restored, options.target_script)
    }

    /// 翻译单个片段（含缓存查询与写入）
    fn translate_one(
        &self,
        segment: &str,
        direction: &TranslationDirection,
        hints: &TranslationHints,
        use_cache: bool,
        fingerprint: u64,
        cached_segments: &mut usize,
    ) -> Result<String, TranslationError> {
        let cache = match (&self.cache, use_cache) {
            (Some(c), true) => Some(c),
            _ => None,
        };

        let key = cache.map(|_| {
            TranslationCache::make_key(
                self.provider.name(),
                &direction.as_str(),
                fingerprint,
                segment,
            )
        });

        if let (Some(c), Some(k)) = (cache, key) {
            if let Some(hit) = c.get(k) {
                *cached_segments += 1;
                return Ok(hit);
            }
        }

        let translated = if hints.is_empty() {
            self.provider.translate(segment, direction.clone())?
        } else {
            self.provider
                .translate_with_hints(segment, direction.clone(), hints)?
        };

        if let (Some(c), Some(k)) = (cache, key) {
            c.put(k, translated.clone());
        }

        Ok(translated)
    }

    /// 构造提示信息
    ///
    /// 只有支持 prompt 的引擎才把术语表写进去；不支持的引擎靠占位符保证一致性。
    fn build_hints(
        &self,
        options: &TranslationOptions,
        glossary_ctx: &GlossaryContext,
    ) -> TranslationHints {
        let mut hints = TranslationHints::default();

        if self.provider.supports_glossary_prompt() {
            // 只输出本次命中的术语，避免 prompt 被无关条目撑大
            hints.glossary_text = options.glossary.to_prompt_text_of(glossary_ctx.matched());
        }
        hints.context = options.context.clone();
        hints
    }

    /// 决定单段最大字符数
    fn resolve_max_chars(&self, options: &TranslationOptions) -> usize {
        if options.max_segment_chars > 0 {
            return options.max_segment_chars;
        }
        let engine_limit = self.provider.max_input_chars();
        // usize::MAX 表示引擎不限制，此时用一个合理的默认值
        if engine_limit == usize::MAX {
            512
        } else {
            engine_limit
        }
    }
}

/// 片段级输出目标
///
/// 用 trait 统一「无输出 / 进度回调 / 流式回调」三种消费方式，
/// 避免把 run() 写成三份几乎一样的代码。
trait Sink {
    fn on_segment(&mut self, partial: &str, done: usize, total: usize);
}

struct NoopSink;

impl Sink for NoopSink {
    fn on_segment(&mut self, _partial: &str, _done: usize, _total: usize) {}
}

struct ProgressSink<'a> {
    on_progress: &'a mut dyn FnMut(usize, usize),
}

impl Sink for ProgressSink<'_> {
    fn on_segment(&mut self, _partial: &str, done: usize, total: usize) {
        (self.on_progress)(done, total);
    }
}

struct StreamSink<'a> {
    on_chunk: &'a mut dyn FnMut(&str),
}

impl Sink for StreamSink<'_> {
    fn on_segment(&mut self, partial: &str, _done: usize, _total: usize) {
        (self.on_chunk)(partial);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use votex_domain::translation::glossary::{Glossary, GlossaryEntry};
    use votex_domain::translation::script::ChineseScript;
    use votex_domain::translation::value_object::TranslationDirection;

    /// 测试用引擎：把原文原样返回，方便观察管线做了什么变换
    struct EchoProvider {
        translate_calls: std::sync::atomic::AtomicUsize,
    }

    impl EchoProvider {
        fn new() -> Self {
            Self {
                translate_calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    impl TranslationProvider for EchoProvider {
        fn name(&self) -> &str {
            "echo"
        }

        fn translate(
            &self,
            text: &str,
            _direction: TranslationDirection,
        ) -> Result<String, TranslationError> {
            self.translate_calls
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(text.to_string())
        }
    }

    /// 测试用引擎：声明支持 prompt 注入，并记录收到的术语表
    struct RecordingProvider {
        received: std::sync::Mutex<Vec<String>>,
    }

    impl RecordingProvider {
        fn new() -> Self {
            Self {
                received: std::sync::Mutex::new(Vec::new()),
            }
        }
        fn received(&self) -> Vec<String> {
            self.received.lock().unwrap().clone()
        }
    }

    impl TranslationProvider for RecordingProvider {
        fn name(&self) -> &str {
            "recorder"
        }

        fn translate(
            &self,
            text: &str,
            _direction: TranslationDirection,
        ) -> Result<String, TranslationError> {
            Ok(text.to_string())
        }

        fn supports_glossary_prompt(&self) -> bool {
            true
        }

        fn translate_with_hints(
            &self,
            text: &str,
            _direction: TranslationDirection,
            hints: &TranslationHints,
        ) -> Result<String, TranslationError> {
            self.received.lock().unwrap().push(hints.glossary_text.clone());
            Ok(text.to_string())
        }
    }

    #[test]
    fn 管线_空文本报错() {
        let p = TranslationPipeline::without_cache(Arc::new(EchoProvider::new()));
        let r = p.translate("", &TranslationOptions::default());
        assert!(matches!(r, Err(TranslationError::EmptyText)));
    }

    #[test]
    fn 管线_基础翻译不分段() {
        let p = TranslationPipeline::without_cache(Arc::new(EchoProvider::new()));
        let out = p
            .translate("你好世界", &TranslationOptions::new(TranslationDirection::ZhToEn))
            .unwrap();
        assert_eq!(out.text, "你好世界");
        assert_eq!(out.segments, 1);
        assert_eq!(out.cached_segments, 0);
        assert!(!out.glossary_used);
    }

    #[test]
    fn 管线_术语表占位与回填() {
        let p = TranslationPipeline::without_cache(Arc::new(EchoProvider::new()));
        let glossary = Glossary::from_entries(vec![GlossaryEntry::new("爱丽丝", "Alice")]);
        let options = TranslationOptions::new(TranslationDirection::ZhToEn).with_glossary(glossary);

        let out = p.translate("爱丽丝走进了花园", &options).unwrap();
        assert_eq!(out.text, "Alice走进了花园");
        assert!(out.glossary_used);
    }

    #[test]
    fn 管线_长文本自动分段() {
        let p = TranslationPipeline::without_cache(Arc::new(EchoProvider::new()));
        let long = "今天天气很好。".repeat(100);
        let options =
            TranslationOptions::new(TranslationDirection::ZhToEn).with_max_segment_chars(50);
        let out = p.translate(&long, &options).unwrap();
        assert!(out.segments > 1, "长文本应被分段，实际 {} 段", out.segments);
    }

    #[test]
    fn 管线_缓存命中后不再调用引擎() {
        let provider = Arc::new(EchoProvider::new());
        let cache = Arc::new(TranslationCache::new(100));
        let p = TranslationPipeline::new(provider.clone(), Some(cache));
        let options = TranslationOptions::new(TranslationDirection::ZhToEn);

        p.translate("你好", &options).unwrap();
        let calls_after_first = provider.translate_calls.load(std::sync::atomic::Ordering::SeqCst);
        let out = p.translate("你好", &options).unwrap();

        assert_eq!(out.cached_segments, 1, "第二次应命中缓存");
        assert_eq!(
            provider.translate_calls.load(std::sync::atomic::Ordering::SeqCst),
            calls_after_first,
            "命中缓存后不应再调用引擎"
        );
    }

    #[test]
    fn 管线_关闭缓存则每次都调用引擎() {
        let provider = Arc::new(EchoProvider::new());
        let cache = Arc::new(TranslationCache::new(100));
        let p = TranslationPipeline::new(provider.clone(), Some(cache));
        let options = TranslationOptions::new(TranslationDirection::ZhToEn).without_cache();

        p.translate("你好", &options).unwrap();
        p.translate("你好", &options).unwrap();
        assert_eq!(
            provider.translate_calls.load(std::sync::atomic::Ordering::SeqCst),
            2
        );
    }

    #[test]
    fn 管线_术语表变化时缓存自动失效() {
        let provider = Arc::new(EchoProvider::new());
        let cache = Arc::new(TranslationCache::new(100));
        let p = TranslationPipeline::new(provider.clone(), Some(cache));

        let g1 = Glossary::from_entries(vec![GlossaryEntry::new("甲", "A")]);
        let g2 = Glossary::from_entries(vec![GlossaryEntry::new("甲", "B")]);

        let out1 = p
            .translate("甲", &TranslationOptions::new(TranslationDirection::ZhToEn).with_glossary(g1))
            .unwrap();
        let out2 = p
            .translate("甲", &TranslationOptions::new(TranslationDirection::ZhToEn).with_glossary(g2))
            .unwrap();

        assert_eq!(out1.text, "A");
        assert_eq!(out2.text, "B", "术语表变了译文必须跟着变");
    }

    #[test]
    fn 管线_支持prompt的引擎收到术语表() {
        let provider = Arc::new(RecordingProvider::new());
        let p = TranslationPipeline::without_cache(provider.clone());
        let glossary = Glossary::from_entries(vec![GlossaryEntry::new("爱丽丝", "Alice")]);
        let options = TranslationOptions::new(TranslationDirection::ZhToEn).with_glossary(glossary);

        p.translate("爱丽丝来了", &options).unwrap();
        let received = provider.received();
        assert_eq!(received.len(), 1);
        assert!(received[0].contains("爱丽丝->Alice"));
    }

    #[test]
    fn 管线_不支持prompt的引擎不会收到术语表() {
        let provider = Arc::new(EchoProvider::new());
        let p = TranslationPipeline::without_cache(provider.clone());
        let glossary = Glossary::from_entries(vec![GlossaryEntry::new("爱丽丝", "Alice")]);
        let options = TranslationOptions::new(TranslationDirection::ZhToEn).with_glossary(glossary);

        let out = p.translate("爱丽丝来了", &options).unwrap();
        assert_eq!(out.text, "Alice来了", "占位符方案仍应保证术语一致");
    }

    #[test]
    fn 管线_简繁转换() {
        let p = TranslationPipeline::without_cache(Arc::new(EchoProvider::new()));
        let options = TranslationOptions::new(TranslationDirection::EnToZh)
            .with_target_script(ChineseScript::Traditional);
        let out = p.translate("我们爱中国", &options).unwrap();
        assert_eq!(out.text, "我們愛中國");
    }

    #[test]
    fn 管线_进度回调按片段汇报() {
        let p = TranslationPipeline::without_cache(Arc::new(EchoProvider::new()));
        let long = "今天天气很好。".repeat(50);
        let options =
            TranslationOptions::new(TranslationDirection::ZhToEn).with_max_segment_chars(30);

        let mut seen = Vec::new();
        let out = p
            .translate_with_progress(&long, &options, &mut |done, total| {
                seen.push((done, total))
            })
            .unwrap();

        assert_eq!(seen.len(), out.segments);
        assert_eq!(seen.last().unwrap().0, out.segments, "最后一次应为全部完成");
    }

    #[test]
    fn 管线_流式回调逐步输出累积译文() {
        let p = TranslationPipeline::without_cache(Arc::new(EchoProvider::new()));
        let long = "今天天气很好。".repeat(50);
        let options =
            TranslationOptions::new(TranslationDirection::ZhToEn).with_max_segment_chars(30);

        let mut chunks = Vec::new();
        let out = p
            .translate_stream(&long, &options, &mut |c| chunks.push(c.to_string()))
            .unwrap();

        assert_eq!(chunks.len(), out.segments);
        assert_eq!(chunks.last().unwrap(), &out.text, "最后一块应为完整译文");
        assert!(chunks[0].len() < out.text.len(), "首块应为部分译文");
    }

    #[test]
    fn 管线_批量翻译逐条处理() {
        let p = TranslationPipeline::without_cache(Arc::new(EchoProvider::new()));
        let texts = vec!["你好".to_string(), "世界".to_string()];
        let out = p
            .translate_batch(&texts, &TranslationOptions::new(TranslationDirection::ZhToEn))
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].text, "你好");
        assert_eq!(out[1].text, "世界");
    }

    #[test]
    fn 管线_批量进度按条目汇报() {
        let p = TranslationPipeline::without_cache(Arc::new(EchoProvider::new()));
        let texts = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let mut seen = Vec::new();
        p.translate_batch_with_progress(
            &texts,
            &TranslationOptions::new(TranslationDirection::ZhToEn),
            &mut |done, total| seen.push((done, total)),
        )
        .unwrap();
        assert_eq!(seen, vec![(1, 3), (2, 3), (3, 3)]);
    }

    #[test]
    fn 管线_不支持的方向被拒绝() {
        // 词典引擎只支持 zh→en
        let provider = Arc::new(votex_infra::translation::dict_translate::DictTranslationProvider::new());
        let p = TranslationPipeline::without_cache(provider);
        let r = p.translate("hello", &TranslationOptions::new(TranslationDirection::EnToZh));
        assert!(matches!(r, Err(TranslationError::UnsupportedDirection)));
    }
}
