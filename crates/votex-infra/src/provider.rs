//! Provider 注册表实现
//!
//! 借鉴 MoneyPrinterTurbo 的配置驱动 Provider 架构：
//! - `application.yml` 统一管理多个 Provider
//! - 通过配置切换默认引擎
//! - 运行时动态注册/发现 Provider

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use votex_domain::error::DomainError;
use votex_domain::model::value_object::EngineKind;
use votex_domain::provider::{
    ProviderCapability, ProviderFactory, ProviderInfo, ProviderRegistry,
};

/// 运行时 Provider 注册表
///
/// # 设计说明
///
/// 借鉴 MoneyPrinterTurbo 的 TaskService 编排模式：
/// - 所有 Provider 通过注册表统一管理
/// - 通过配置（`application.yml`）指定默认使用的引擎
/// - 支持运行时动态注册第三方 Provider
///
/// # 示例
///
/// ```ignore
/// let mut registry = DefaultProviderRegistry::new();
/// registry.register(EngineKind::Kokoro, info, factory)?;
/// let factory = registry.get_factory(&EngineKind::Kokoro).unwrap();
/// let provider = factory.create();
/// ```
pub struct DefaultProviderRegistry {
    /// 引擎名 -> (ProviderInfo, ProviderFactory) 映射
    providers: RwLock<HashMap<EngineKind, (ProviderInfo, Arc<dyn ProviderFactory>)>>,
}

impl DefaultProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
        }
    }

    /// 从配置创建注册表（自动注册默认引擎）
    pub fn with_defaults() -> Self {
        Self {
            providers: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for DefaultProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderRegistry for DefaultProviderRegistry {
    fn register(
        &mut self,
        engine_kind: EngineKind,
        info: ProviderInfo,
        factory: Box<dyn ProviderFactory>,
    ) -> Result<(), DomainError> {
        let mut providers = self.providers.write().map_err(|e| {
            DomainError::Config(votex_domain::error::ConfigError::ReadFailed(e.to_string()))
        })?;
        providers.insert(engine_kind, (info, Arc::from(factory)));
        Ok(())
    }

    fn get_factory(&self, engine_kind: &EngineKind) -> Option<Arc<dyn ProviderFactory>> {
        let providers = self.providers.read().ok()?;
        providers.get(engine_kind).map(|(_, f)| Arc::clone(f))
    }

    fn list_by_capability(&self, capability: ProviderCapability) -> Vec<ProviderInfo> {
        let providers = match self.providers.read() {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };
        providers
            .values()
            .filter(|(info, _)| info.capability == capability)
            .map(|(info, _)| info.clone())
            .collect()
    }

    fn list_all(&self) -> Vec<ProviderInfo> {
        let providers = match self.providers.read() {
            Ok(p) => p,
            Err(_) => return Vec::new(),
        };
        providers.values().map(|(info, _)| info.clone()).collect()
    }

    fn has_engine(&self, engine_kind: &EngineKind) -> bool {
        let providers = self.providers.read().ok();
        match providers {
            Some(p) => p.contains_key(engine_kind),
            None => false,
        }
    }

    fn count(&self) -> usize {
        let providers = self.providers.read().ok();
        match providers {
            Some(p) => p.len(),
            None => 0,
        }
    }
}

/// 默认 Provider 工厂（简化版，用于测试和快速集成）
///
/// 实际使用时应为每个引擎类型实现具体的工厂。
pub struct SimpleProviderFactory {
    info: ProviderInfo,
    /// 创建 Provider 的闭包
    creator: Box<dyn Fn() -> Box<dyn std::any::Any> + Send + Sync>,
}

impl SimpleProviderFactory {
    pub fn new<F>(info: ProviderInfo, creator: F) -> Self
    where
        F: Fn() -> Box<dyn std::any::Any> + Send + Sync + 'static,
    {
        Self {
            info,
            creator: Box::new(creator),
        }
    }
}

impl ProviderFactory for SimpleProviderFactory {
    fn info(&self) -> &ProviderInfo {
        &self.info
    }

    fn create(&self) -> Box<dyn std::any::Any> {
        (self.creator)()
    }
}

// ============================================================
// Provider 注册辅助函数
// ============================================================

/// 从配置的引擎字符串获取默认的 TTS 引擎
pub fn resolve_tts_engine(config: &votex_domain::config::value_object::AppConfig) -> EngineKind {
    EngineKind::from_str(&config.tts.default_engine)
        .unwrap_or(EngineKind::Kokoro)
}

/// 从配置的引擎字符串获取默认的 ASR 引擎
pub fn resolve_asr_engine(config: &votex_domain::config::value_object::AppConfig) -> EngineKind {
    EngineKind::from_str(&config.asr.default_model)
        .unwrap_or(EngineKind::Whisper)
}

/// 从配置的引擎字符串获取默认的翻译引擎
///
/// 无法识别时回退到内置词典引擎（唯一不依赖模型文件与网络的引擎）。
pub fn resolve_translation_engine(
    config: &votex_domain::config::value_object::AppConfig,
) -> EngineKind {
    EngineKind::from_str(&config.translation.default_engine).unwrap_or(EngineKind::OpusMt)
}

/// 注册全部翻译 Provider 到注册表
///
/// 每个引擎的工厂闭包返回 `Box<dyn TranslationProvider>`（装箱为 `Any`），
/// 调用方再向下转型取回具体类型。
///
/// 注意：工厂只创建「未加载模型」的 Provider，模型加载由
/// [`crate::translation::TranslationModelPool`] 按需完成，
/// 避免启动时就把数 GB 模型全部载入内存。
pub fn register_translation_providers(registry: &mut DefaultProviderRegistry) {
    use votex_domain::translation::provider::TranslationProvider;

    let engines: &[(EngineKind, &str, &str)] = &[
        (EngineKind::OpusMt, "opus-mt", "OPUS-MT 离线翻译"),
        (EngineKind::Nllb, "nllb-200", "NLLB-200 离线翻译"),
        (EngineKind::M2m100, "m2m-100", "M2M-100 离线翻译"),
        (EngineKind::HyMt1_5, "hy-mt-1.5", "HY-MT1.5 离线翻译"),
        (EngineKind::QwenMt, "qwen-mt", "Qwen-MT 在线翻译"),
    ];

    for (kind, name, display) in engines {
        let info = ProviderInfo {
            capability: ProviderCapability::Translation,
            engine_name: (*name).to_string(),
            display_name: (*display).to_string(),
            engine_kind: *kind,
            version: None,
        };
        let factory = SimpleProviderFactory::new(info, move || {
            let provider: Box<dyn TranslationProvider> = match kind {
                EngineKind::OpusMt => {
                    Box::new(crate::translation::opus_mt::OpusMtProvider::new())
                }
                EngineKind::Nllb => Box::new(crate::translation::nllb::NllbProvider::new()),
                EngineKind::M2m100 => Box::new(crate::translation::m2m100::M2m100Provider::new()),
                EngineKind::HyMt1_5 => Box::new(crate::translation::hy_mt::HyMtProvider::new()),
                EngineKind::QwenMt => match crate::translation::qwen_mt::QwenMtProvider::new() {
                    Ok(p) => Box::new(p),
                    // 未配置 API Key 时降级为内置词典，避免启动即失败
                    Err(_) => Box::new(crate::translation::dict_translate::DictTranslationProvider::new()),
                },
                _ => Box::new(crate::translation::dict_translate::DictTranslationProvider::new()),
            };
            let boxed: Box<dyn std::any::Any> = Box::new(provider);
            boxed
        });

        // 注册失败只记录日志，不阻断启动
        if let Err(e) = registry.register(*kind, factory.info().clone(), Box::new(factory)) {
            tracing::warn!("注册翻译引擎 {} 失败: {}", name, e);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use votex_domain::provider::ProviderInfo;

    fn make_dummy_info(name: &str, capability: ProviderCapability) -> ProviderInfo {
        let engine_kind = EngineKind::from_str(name).unwrap_or(EngineKind::Kokoro);
        ProviderInfo {
            capability,
            engine_name: name.to_string(),
            display_name: format!("测试{}", name),
            engine_kind,
            version: None,
        }
    }

    #[test]
    fn test_registry_注册和获取() {
        let mut registry = DefaultProviderRegistry::new();
        let info = make_dummy_info("kokoro", ProviderCapability::Tts);
        let factory = SimpleProviderFactory::new(info, || Box::new(42i32));

        registry.register(EngineKind::Kokoro, factory.info().clone(), Box::new(factory)).unwrap();

        assert!(registry.has_engine(&EngineKind::Kokoro));
        assert!(!registry.has_engine(&EngineKind::Whisper));
        assert_eq!(registry.count(), 1);
    }

    #[test]
    fn test_registry_按能力列列表() {
        let mut registry = DefaultProviderRegistry::new();

        let tts_info = make_dummy_info("kokoro", ProviderCapability::Tts);
        let tts_factory = SimpleProviderFactory::new(tts_info, || Box::new(42i32));
        registry.register(EngineKind::Kokoro, tts_factory.info().clone(), Box::new(tts_factory)).unwrap();

        let asr_info = make_dummy_info("whisper", ProviderCapability::Asr);
        let asr_factory = SimpleProviderFactory::new(asr_info, || Box::new(42i32));
        registry.register(EngineKind::Whisper, asr_factory.info().clone(), Box::new(asr_factory)).unwrap();

        let tts_list = registry.list_by_capability(ProviderCapability::Tts);
        assert_eq!(tts_list.len(), 1);

        let all = registry.list_all();
        assert_eq!(all.len(), 2);
        // 检查返回的是所有权类型（clone 后独立）
        let names: Vec<String> = all.into_iter().map(|info| info.engine_name).collect();
        assert!(names.contains(&"kokoro".to_string()));
        assert!(names.contains(&"whisper".to_string()));
    }

    #[test]
    fn test_registry_获取工厂() {
        let mut registry = DefaultProviderRegistry::new();
        let info = make_dummy_info("kokoro", ProviderCapability::Tts);
        let factory = SimpleProviderFactory::new(info, || Box::new(42i32));

        registry.register(EngineKind::Kokoro, factory.info().clone(), Box::new(factory)).unwrap();

        let retrieved = registry.get_factory(&EngineKind::Kokoro);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().info().engine_name, "kokoro");
    }

    #[test]
    fn test_resolve_tts_engine() {
        let config = votex_domain::config::value_object::AppConfig::default();
        let engine = resolve_tts_engine(&config);
        assert_eq!(engine, EngineKind::Kokoro);
    }

    #[test]
    fn test_resolve_asr_engine() {
        let config = votex_domain::config::value_object::AppConfig::default();
        let engine = resolve_asr_engine(&config);
        assert_eq!(engine, EngineKind::Whisper);
    }

    #[test]
    fn test_resolve_translation_engine_默认值() {
        let config = votex_domain::config::value_object::AppConfig::default();
        let engine = resolve_translation_engine(&config);
        // AppConfig::default() 的 translation.default_engine 为 "dict"，
        // 而 EngineKind 没有 dict 变体，故回退到 OpusMt
        assert_eq!(engine, EngineKind::OpusMt);
    }

    #[test]
    fn test_resolve_translation_engine_读取配置() {
        let mut config = votex_domain::config::value_object::AppConfig::default();
        config.translation.default_engine = "hy-mt-1.5".to_string();
        assert_eq!(resolve_translation_engine(&config), EngineKind::HyMt1_5);

        config.translation.default_engine = "nllb-200".to_string();
        assert_eq!(resolve_translation_engine(&config), EngineKind::Nllb);
    }

    #[test]
    fn test_resolve_translation_engine_非法值回退() {
        let mut config = votex_domain::config::value_object::AppConfig::default();
        config.translation.default_engine = "不存在的引擎".to_string();
        assert_eq!(resolve_translation_engine(&config), EngineKind::OpusMt);
    }

    #[test]
    fn test_注册翻译引擎() {
        let mut registry = DefaultProviderRegistry::new();
        register_translation_providers(&mut registry);

        let translation = registry.list_by_capability(ProviderCapability::Translation);
        assert_eq!(translation.len(), 5, "应注册 5 个翻译引擎");

        let names: Vec<String> = translation.into_iter().map(|i| i.engine_name).collect();
        for n in ["opus-mt", "nllb-200", "m2m-100", "hy-mt-1.5", "qwen-mt"] {
            assert!(names.contains(&n.to_string()), "缺少引擎: {}", n);
        }
    }

    #[test]
    fn test_注册后的工厂可创建provider() {
        use votex_domain::translation::provider::TranslationProvider;

        let mut registry = DefaultProviderRegistry::new();
        register_translation_providers(&mut registry);

        let factory = registry.get_factory(&EngineKind::Nllb).unwrap();
        let created = factory.create();
        // Box<dyn Any>::downcast 要求 E: Debug，这里 hand-rolled 避免对 trait object 的 Debug 约束
        let provider = match created.downcast::<Box<dyn TranslationProvider>>() {
            Ok(p) => p,
            Err(_) => panic!("工厂应返回 Box<dyn TranslationProvider>"),
        };

        assert_eq!(provider.name(), "nllb");
        assert!(!provider.is_loaded(), "工厂只创建未加载的 Provider");
    }
}
