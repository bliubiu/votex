//! 翻译请求选项
//!
//! 把「翻译质量工程」相关的可选项集中在一起，避免用例方法签名无限膨胀。

use crate::translation::glossary::Glossary;
use crate::translation::script::ChineseScript;
use crate::translation::value_object::TranslationDirection;

/// 翻译选项
#[derive(Debug, Clone)]
pub struct TranslationOptions {
    /// 翻译方向
    pub direction: TranslationDirection,
    /// 术语表（可为空）
    pub glossary: Glossary,
    /// 单段最大字符数；`0` 表示由引擎能力自动决定
    pub max_segment_chars: usize,
    /// 是否使用翻译缓存
    pub use_cache: bool,
    /// 目标中文书写系统（简/繁/不转换）
    pub target_script: ChineseScript,
    /// 历史上下文（原文, 译文），供 LLM 引擎做 few-shot 以保持术语与人称一致
    ///
    /// 只取最后若干轮，避免 prompt 过长。
    pub context: Vec<(String, String)>,
    /// 上下文最多保留的轮数
    pub max_context_turns: usize,
}

impl Default for TranslationOptions {
    fn default() -> Self {
        Self {
            direction: TranslationDirection::ZhToEn,
            glossary: Glossary::new(),
            max_segment_chars: 0,
            use_cache: true,
            target_script: ChineseScript::None,
            context: Vec::new(),
            max_context_turns: 3,
        }
    }
}

impl TranslationOptions {
    /// 以指定方向创建选项
    pub fn new(direction: TranslationDirection) -> Self {
        Self {
            direction,
            ..Default::default()
        }
    }

    /// 设置术语表
    pub fn with_glossary(mut self, glossary: Glossary) -> Self {
        self.glossary = glossary;
        self
    }

    /// 设置单段最大字符数
    pub fn with_max_segment_chars(mut self, max_segment_chars: usize) -> Self {
        self.max_segment_chars = max_segment_chars;
        self
    }

    /// 关闭翻译缓存
    pub fn without_cache(mut self) -> Self {
        self.use_cache = false;
        self
    }

    /// 设置目标书写系统
    pub fn with_target_script(mut self, script: ChineseScript) -> Self {
        self.target_script = script;
        self
    }

    /// 设置历史上下文（会自动裁剪到 max_context_turns 轮）
    pub fn with_context(mut self, context: Vec<(String, String)>) -> Self {
        self.context = Self::trim_context(context, self.max_context_turns);
        self
    }

    /// 设置上下文轮数上限
    pub fn with_max_context_turns(mut self, turns: usize) -> Self {
        self.max_context_turns = turns;
        self.context = Self::trim_context(std::mem::take(&mut self.context), turns);
        self
    }

    /// 追加一轮翻译结果到上下文
    pub fn push_context(&mut self, source: String, target: String) {
        self.context.push((source, target));
        self.context = Self::trim_context(std::mem::take(&mut self.context), self.max_context_turns);
    }

    /// 裁剪上下文，只保留最后 `turns` 轮
    fn trim_context(mut context: Vec<(String, String)>, turns: usize) -> Vec<(String, String)> {
        if turns == 0 {
            return Vec::new();
        }
        if context.len() > turns {
            context.drain(..context.len() - turns);
        }
        context
    }
}

/// 随单次翻译调用下发的提示信息
///
/// 这些信息按**调用**传递而不是写进 Provider 内部状态：
/// 会话池中的 Provider 是共享实例，若把提示信息存在 `self` 上，
/// 并发翻译时会互相覆盖。
#[derive(Debug, Clone, Default)]
pub struct TranslationHints {
    /// 术语表文本，格式 `原文->译文 #备注`（每行一条），可为空
    pub glossary_text: String,
    /// 历史上下文（原文, 译文），供 LLM 引擎做 few-shot
    pub context: Vec<(String, String)>,
}

impl TranslationHints {
    /// 空提示
    pub fn empty() -> Self {
        Self::default()
    }

    /// 是否没有任何提示
    pub fn is_empty(&self) -> bool {
        self.glossary_text.is_empty() && self.context.is_empty()
    }

    /// 只带术语表
    pub fn with_glossary_text(glossary_text: impl Into<String>) -> Self {
        Self {
            glossary_text: glossary_text.into(),
            context: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 选项_默认值合理() {
        let o = TranslationOptions::default();
        assert!(o.use_cache);
        assert_eq!(o.max_segment_chars, 0);
        assert_eq!(o.target_script, ChineseScript::None);
        assert_eq!(o.max_context_turns, 3);
    }

    #[test]
    fn 选项_上下文自动裁剪到指定轮数() {
        let ctx = vec![
            ("a1".into(), "b1".into()),
            ("a2".into(), "b2".into()),
            ("a3".into(), "b3".into()),
            ("a4".into(), "b4".into()),
        ];
        let o = TranslationOptions::default()
            .with_max_context_turns(2)
            .with_context(ctx);
        assert_eq!(o.context.len(), 2);
        assert_eq!(o.context[0].0, "a3");
        assert_eq!(o.context[1].0, "a4");
    }

    #[test]
    fn 选项_追加上下文后仍受轮数限制() {
        let mut o = TranslationOptions::default().with_max_context_turns(2);
        o.push_context("1".into(), "一".into());
        o.push_context("2".into(), "二".into());
        o.push_context("3".into(), "三".into());
        assert_eq!(o.context.len(), 2);
        assert_eq!(o.context[1].0, "3");
    }

    #[test]
    fn 选项_轮数为零时清空上下文() {
        let o = TranslationOptions::default()
            .with_context(vec![("a".into(), "b".into())])
            .with_max_context_turns(0);
        assert!(o.context.is_empty());
    }
}
