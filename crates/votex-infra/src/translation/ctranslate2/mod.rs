//! CTranslate2 翻译加速后端（`ct2` feature 门控双形态）
//!
//! CTranslate2 只有 C++ / Python 接口，无官方 C API——无法用 `libloading`
//! 运行时直调，因此通过 [`ct2rs`](https://crates.io/crates/ct2rs) 绑定接入
//! （vendor 模式从源码编译 C++ 库，需 CMake + MSVC + ninja）。
//!
//! 为保证默认构建零影响，本模块按 `ct2` feature 拆成两个形态：
//!
//! - [`imp_enabled`]（`--features ct2`）：真实 ct2rs 推理实现（int8 量化）
//! - [`imp_fallback`]（默认）：降级骨架——保留模型探测与状态管理，
//!   `translate` 返回「未启用 ct2 feature」的构建引导提示
//!
//! 两个形态对外暴露同一个 `CTranslate2Provider` 类型与一致的行为契约
//! （见下方通用契约测试），调用方无需感知 feature 差异。
//!
//! # 模型来源
//! 使用 `scripts/convert_ct2.py`（uv + 官方转换器，int8）将 HF 模型本地转换，
//! 起步支持 opus-mt-zh-en（Marian）。

#[cfg(feature = "ct2")]
mod imp_enabled;
#[cfg(feature = "ct2")]
pub use imp_enabled::CTranslate2Provider;

#[cfg(not(feature = "ct2"))]
mod imp_fallback;
#[cfg(not(feature = "ct2"))]
pub use imp_fallback::CTranslate2Provider;

#[cfg(test)]
mod tests {
    use super::*;
    use votex_domain::error::TranslationError;
    use votex_domain::translation::provider::TranslationProvider;
    use votex_domain::translation::value_object::TranslationDirection;

    /// 两形态通用契约：新建 Provider 必须处于未加载状态
    #[test]
    fn 新建的Provider未加载() {
        let provider = CTranslate2Provider::new();
        assert!(!provider.is_loaded(), "new 后应处于未加载状态");
    }

    /// 两形态通用契约：未加载时翻译返回 ModelNotLoaded
    #[test]
    fn 未加载时翻译返回ModelNotLoaded() {
        let provider = CTranslate2Provider::new();
        let err = provider
            .translate("你好世界", TranslationDirection::ZhToEn)
            .unwrap_err();
        assert!(
            matches!(err, TranslationError::ModelNotLoaded),
            "应返回 ModelNotLoaded，实际: {:?}",
            err
        );
    }

    /// 两形态通用契约：空文本返回 EmptyText（先于加载状态检查）
    #[test]
    fn 空文本返回EmptyText() {
        let provider = CTranslate2Provider::new();
        let err = provider
            .translate("   ", TranslationDirection::Auto)
            .unwrap_err();
        assert!(
            matches!(err, TranslationError::EmptyText),
            "应返回 EmptyText，实际: {:?}",
            err
        );
    }
}
