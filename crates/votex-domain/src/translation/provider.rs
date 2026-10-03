use crate::error::TranslationError;
use crate::translation::glossary::Glossary;
use crate::translation::options::TranslationHints;
use crate::translation::value_object::TranslationDirection;
use std::path::Path;

/// 翻译提供者接口
///
/// # 设计说明
///
/// 只有 `name` 与 `translate` 是必须实现的；其余方法均提供默认实现，
/// 引擎按需覆写，这样新增引擎的成本仍然很低。
///
/// 默认实现的语义：
/// - `is_loaded` 默认 `true`：在线 API / 词典类引擎无需加载模型。
/// - `translate_batch` 默认逐条调用 `translate`，引擎可覆写为真批处理提效。
/// - `translate_stream` 默认一次性返回，引擎可覆写为逐块回调。
/// - `supports_glossary_prompt` 默认 `false`：纯 Encoder-Decoder 离线模型无法注入术语表，
///   此时由上层流水线改用「占位符替换」方案保证术语一致。
pub trait TranslationProvider: Send + Sync {
    /// 提供者名称（唯一标识，如 "opus-mt"）
    fn name(&self) -> &str;

    /// 翻译文本
    fn translate(
        &self,
        text: &str,
        direction: TranslationDirection,
    ) -> Result<String, TranslationError>;

    // ========== 生命周期 ==========

    /// 从模型目录加载（离线引擎覆写）
    ///
    /// 在线引擎与词典引擎保持默认实现即可。
    fn load(&mut self, _model_dir: &Path) -> Result<(), TranslationError> {
        Ok(())
    }

    /// 释放模型资源
    fn unload(&mut self) {}

    /// 模型是否已加载
    fn is_loaded(&self) -> bool {
        true
    }

    // ========== 能力声明 ==========

    /// 支持的语言对列表，空表示「不限制」
    fn supported_pairs(&self) -> Vec<(String, String)> {
        Vec::new()
    }

    /// 是否支持指定方向
    fn supports(&self, direction: &TranslationDirection) -> bool {
        let pairs = self.supported_pairs();
        if pairs.is_empty() {
            return true;
        }
        let (src, tgt) = match direction {
            TranslationDirection::ZhToEn => ("zh".to_string(), "en".to_string()),
            TranslationDirection::EnToZh => ("en".to_string(), "zh".to_string()),
            TranslationDirection::Auto => return true,
            TranslationDirection::ByLanguagePair { source, target } => {
                (source.clone(), target.clone())
            }
        };
        pairs.iter().any(|(s, t)| *s == src && *t == tgt)
    }

    /// 是否为在线引擎（在线引擎不享受离线缓存语义）
    fn is_online(&self) -> bool {
        false
    }

    /// 单段可承受的最大输入字符数
    ///
    /// 上层流水线据此决定是否分段。`usize::MAX` 表示不限制。
    fn max_input_chars(&self) -> usize {
        usize::MAX
    }

    // ========== 批量 ==========

    /// 批量翻译
    ///
    /// 默认实现逐条调用 `translate`；引擎可覆写为真正的批处理以提升吞吐。
    fn translate_batch(
        &self,
        texts: &[&str],
        direction: TranslationDirection,
    ) -> Result<Vec<String>, TranslationError> {
        texts
            .iter()
            .map(|t| self.translate(t, direction.clone()))
            .collect()
    }

    // ========== 术语表 ==========

    /// 是否支持把术语表写进 prompt（LLM / Chat 类引擎返回 true）
    fn supports_glossary_prompt(&self) -> bool {
        false
    }

    /// 带提示信息的翻译（术语表 + 历史上下文）
    ///
    /// 默认实现忽略提示直接翻译——离线 NMT 引擎无法理解 prompt 指令，
    /// 术语一致性由上层流水线的「占位符替换」保证。
    ///
    /// 提示信息按**调用**传递而不是写进 `self`：会话池中的 Provider 是共享
    /// 实例，若存到 `self` 上，并发翻译时会互相覆盖。
    fn translate_with_hints(
        &self,
        text: &str,
        direction: TranslationDirection,
        _hints: &TranslationHints,
    ) -> Result<String, TranslationError> {
        self.translate(text, direction)
    }

    // ========== 流式 ==========

    /// 流式翻译，逐块回调 `on_chunk`，返回完整译文
    ///
    /// 默认实现：一次性翻译后整体回调一次。
    fn translate_stream(
        &self,
        text: &str,
        direction: TranslationDirection,
        on_chunk: &mut dyn FnMut(&str),
    ) -> Result<String, TranslationError> {
        let result = self.translate(text, direction)?;
        on_chunk(&result);
        Ok(result)
    }
}

/// 描述术语表在当前引擎上的生效方式（用于日志与 GUI 提示）
///
/// 术语表对所有引擎都有效：LLM 引擎走 prompt 注入，离线引擎走占位符替换。
pub fn describe_glossary_mode(provider: &dyn TranslationProvider, glossary: &Glossary) -> String {
    if glossary.is_empty() {
        return "未启用术语表".to_string();
    }
    if provider.supports_glossary_prompt() {
        format!("{}: prompt 注入 + 占位符回填", provider.name())
    } else {
        format!("{}: 占位符替换", provider.name())
    }
}
