//! CTranslate2 降级形态（默认构建，未启用 `ct2` feature）
//!
//! 职责：
//! - 保留模型目录探测（`model.bin` 就绪标记）与引擎状态管理，
//!   供工厂接线（`create_translation_provider_with_dir`）复用
//! - `translate` 返回「未启用 ct2 feature」的构建引导提示，
//!   配合失败诊断脱敏面板给出明确的中文指引
//!
//! 相对旧骨架的修正：移除 `libloading` 动态加载假设——
//! CTranslate2 无官方 C API，运行时直调不可行（已核实官方仅提供 C++/Python 接口）。

use std::path::Path;

use votex_domain::error::TranslationError;
use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::value_object::TranslationDirection;

use crate::shared::EngineState;

/// CTranslate2 翻译引擎（降级形态）
pub struct CTranslate2Provider {
    /// 已探测到 `model.bin` 的模型目录（诊断信息用）
    ///
    /// 走 `EngineState`：`load()` 是 `&self`，裸 `Option` 无法写入。
    model_dir: EngineState<String>,
}

impl CTranslate2Provider {
    pub fn new() -> Self {
        Self {
            model_dir: EngineState::new(),
        }
    }

    /// 从 CTranslate2 模型目录加载（降级形态仅做探测，不真正加载）
    ///
    /// - `model.bin` 缺失 → 返回 `ModelNotLoaded`
    /// - `model.bin` 存在 → 记录目录并返回成功（`is_loaded = true`），
    ///   后续 `translate` 返回构建引导提示
    pub fn load_from_dir(&self, model_dir: &Path) -> Result<(), TranslationError> {
        if !model_dir.join("model.bin").exists() {
            return Err(TranslationError::ModelNotLoaded);
        }

        self.model_dir
            .load(model_dir.to_string_lossy().to_string());
        tracing::info!(
            "CTranslate2 模型目录已探测到（ct2 feature 未启用，暂不加载推理）: {:?}",
            model_dir
        );
        Ok(())
    }

    /// 构建引导提示：当前二进制未编译 CTranslate2 后端
    fn build_hint_error() -> TranslationError {
        TranslationError::ApiError(
            "CTranslate2 翻译后端未启用：当前构建未开启 ct2 feature，\
             请以 `cargo build --features ct2` 重新构建；\
             也可改用 opus-mt 等原生 ONNX 引擎完成翻译。"
                .to_string(),
        )
    }
}

impl TranslationProvider for CTranslate2Provider {
    fn name(&self) -> &str {
        "ctranslate2"
    }

    fn translate(
        &self,
        text: &str,
        _direction: TranslationDirection,
    ) -> Result<String, TranslationError> {
        if text.trim().is_empty() {
            return Err(TranslationError::EmptyText);
        }

        if !self.model_dir.is_loaded() {
            return Err(TranslationError::ModelNotLoaded);
        }
        Err(Self::build_hint_error())
    }

    fn load(&self, model_dir: &Path) -> Result<(), TranslationError> {
        self.load_from_dir(model_dir)
    }

    fn unload(&self) {
        self.model_dir.unload();
    }

    fn is_loaded(&self) -> bool {
        self.model_dir.is_loaded()
    }
}
