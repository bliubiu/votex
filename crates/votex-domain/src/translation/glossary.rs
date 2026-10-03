//! 术语表（Glossary）
//!
//! 用于保证专有名词（人名、地名、作品术语等）在翻译前后保持一致。
//!
//! # 工作原理
//!
//! 借鉴「占位符替换」思路，分三步：
//!
//! 1. **翻译前**：把原文中命中术语表的词替换为不可见的占位符
//!    （使用 Unicode 私用区 `U+E000`~`U+E001`，不会被分词器切开、不会与原文冲突）。
//!    同时保留原始术语表，供 LLM 引擎注入 prompt。
//! 2. **翻译中**：引擎只看到占位符，无法「自由发挥」把专有名词译错。
//! 3. **翻译后**：把译文中的占位符替换回用户指定的译文。
//!
//! # 为什么不让引擎直接翻译术语
//!
//! 即使把术语表写进 prompt，LLM 也常常忽略或部分遵守；对纯 Encoder-Decoder
//! 的离线模型（NLLB / M2M-100 / Opus-MT）则根本无法注入。占位符方案对
//! **所有引擎一视同仁**，是唯一通用且可靠的做法。

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 占位符起始标记（Unicode 私用区，正常文本中不会出现）
const PLACEHOLDER_START: char = '\u{E000}';
/// 占位符结束标记
const PLACEHOLDER_END: char = '\u{E001}';

/// 单条术语
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GlossaryEntry {
    /// 原文（源语言术语）
    pub src: String,
    /// 指定译文（目标语言）
    pub dst: String,
    /// 备注/注释，会一并注入 LLM prompt 帮助模型理解
    #[serde(default)]
    pub info: Option<String>,
    /// 是否要求整词匹配（默认 false）
    ///
    /// 开启后，"Ann" 不会匹配 "Anna"。对 CJK 文本该选项影响很小。
    #[serde(default)]
    pub whole_word: bool,
    /// 是否区分大小写（默认 false，即忽略大小写）
    #[serde(default)]
    pub case_sensitive: bool,
}

impl GlossaryEntry {
    /// 创建一条术语
    pub fn new(src: impl Into<String>, dst: impl Into<String>) -> Self {
        Self {
            src: src.into(),
            dst: dst.into(),
            info: None,
            whole_word: false,
            case_sensitive: false,
        }
    }

    /// 附加备注
    pub fn with_info(mut self, info: impl Into<String>) -> Self {
        self.info = Some(info.into());
        self
    }

    /// 要求整词匹配
    pub fn whole_word(mut self, whole_word: bool) -> Self {
        self.whole_word = whole_word;
        self
    }

    /// 是否区分大小写
    pub fn case_sensitive(mut self, case_sensitive: bool) -> Self {
        self.case_sensitive = case_sensitive;
        self
    }
}

/// 一次术语替换产生的上下文
///
/// 翻译前由 [`Glossary::apply_before`] 生成，翻译后交给 [`Glossary::apply_after`] 回填。
#[derive(Debug, Clone, Default)]
pub struct GlossaryContext {
    /// 占位符 → 用户指定译文 的映射
    placeholders: HashMap<String, String>,
    /// 本次实际命中的术语（用于注入 LLM prompt）
    matched: Vec<GlossaryEntry>,
}

impl GlossaryContext {
    /// 是否为空（没有任何术语命中）
    pub fn is_empty(&self) -> bool {
        self.placeholders.is_empty()
    }

    /// 本次命中的术语列表
    pub fn matched(&self) -> &[GlossaryEntry] {
        &self.matched
    }

    /// 占位符映射表
    pub fn placeholders(&self) -> &HashMap<String, String> {
        &self.placeholders
    }
}

/// 术语表
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Glossary {
    entries: Vec<GlossaryEntry>,
}

impl Glossary {
    /// 创建空术语表
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// 从术语列表创建，并按「长词优先」排序
    ///
    /// 长词优先可以避免 "Ann" 先抢占 "Anna" 中的子串。
    pub fn from_entries(mut entries: Vec<GlossaryEntry>) -> Self {
        entries.retain(|e| !e.src.is_empty() && !e.dst.is_empty());
        entries.sort_by(|a, b| b.src.chars().count().cmp(&a.src.chars().count()));
        Self { entries }
    }

    /// 添加一条术语
    pub fn add(&mut self, entry: GlossaryEntry) {
        if entry.src.is_empty() || entry.dst.is_empty() {
            return;
        }
        self.entries.push(entry);
        self.entries
            .sort_by(|a, b| b.src.chars().count().cmp(&a.src.chars().count()));
    }

    /// 术语条数
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 全部术语
    pub fn entries(&self) -> &[GlossaryEntry] {
        &self.entries
    }

    /// 翻译前处理：把命中的术语替换为占位符
    ///
    /// # 返回
    /// - 处理后的文本（术语已被占位符替换）
    /// - [`GlossaryContext`]：含占位符映射与命中术语，翻译后需要用它回填
    pub fn apply_before(&self, text: &str) -> (String, GlossaryContext) {
        let mut ctx = GlossaryContext::default();
        if self.entries.is_empty() || text.is_empty() {
            return (text.to_string(), ctx);
        }

        let mut result = text.to_string();
        let mut index = 0usize;

        for entry in &self.entries {
            let Some(hit) = find_occurrence(&result, &entry.src, entry.case_sensitive, entry.whole_word)
            else {
                continue;
            };

            let placeholder = format!("{}{}{}", PLACEHOLDER_START, index, PLACEHOLDER_END);
            index += 1;

            // 替换该术语的所有出现位置
            result = replace_all(&result, &entry.src, &placeholder, entry.case_sensitive, entry.whole_word);
            ctx.placeholders.insert(placeholder, entry.dst.clone());
            ctx.matched.push(entry.clone());

            // hit 仅用于确认命中，避免为未出现的术语生成占位符
            let _ = hit;
        }

        (result, ctx)
    }

    /// 翻译后处理：把占位符回填为用户指定译文
    ///
    /// 引擎有可能改变占位符周边的大小写，这里按大小写不敏感方式回填。
    pub fn apply_after(&self, text: &str, ctx: &GlossaryContext) -> String {
        if ctx.placeholders.is_empty() || text.is_empty() {
            return text.to_string();
        }

        let mut result = text.to_string();
        for (placeholder, dst) in &ctx.placeholders {
            result = replace_ignore_case(&result, placeholder, dst);
        }
        result
    }

    /// 生成注入 LLM prompt 的术语表文本
    ///
    /// 格式（与主流开源翻译工具一致）：每行 `原文->译文 #备注`
    pub fn to_prompt_text(&self) -> String {
        self.to_prompt_text_of(&self.entries)
    }

    /// 只输出本次命中的术语，避免 prompt 被无关条目撑大
    pub fn to_prompt_text_of(&self, entries: &[GlossaryEntry]) -> String {
        let mut lines = Vec::with_capacity(entries.len());
        for e in entries {
            match &e.info {
                Some(info) if !info.trim().is_empty() => {
                    lines.push(format!("{}->{} #{}", e.src, e.dst, info.trim()))
                }
                _ => lines.push(format!("{}->{}", e.src, e.dst)),
            }
        }
        lines.join("\n")
    }

    /// 从 JSON 文本加载
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        let entries: Vec<GlossaryEntry> = serde_json::from_str(json)?;
        Ok(Self::from_entries(entries))
    }

    /// 序列化为 JSON 文本
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.entries)
    }
}

/// 判断字符是否为「词内字符」（用于整词匹配边界判断）
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// 在文本中查找 `needle` 的一次出现，返回字节偏移
fn find_occurrence(
    text: &str,
    needle: &str,
    case_sensitive: bool,
    whole_word: bool,
) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }

    let (haystack, needle_cmp) = if case_sensitive {
        (text.to_string(), needle.to_string())
    } else {
        (text.to_lowercase(), needle.to_lowercase())
    };

    let mut search_start = 0usize;
    while let Some(rel) = haystack[search_start..].find(&needle_cmp) {
        let abs = search_start + rel;
        if !whole_word || is_whole_word_at(text, abs, abs + needle_cmp.len()) {
            return Some(abs);
        }
        search_start = abs + needle_cmp.len().max(1);
        if search_start >= haystack.len() {
            break;
        }
    }
    None
}

/// 判断 `[start, end)` 在 text 中是否构成整词（两侧都不是词内字符）
fn is_whole_word_at(text: &str, start: usize, end: usize) -> bool {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let before_ok = match chars.iter().position(|(i, _)| *i == start) {
        Some(0) | None => true,
        Some(pos) => !is_word_char(chars[pos - 1].1),
    };
    let after_ok = match chars.iter().position(|(i, _)| *i == end) {
        Some(pos) => !is_word_char(chars[pos].1),
        None => true,
    };
    before_ok && after_ok
}

/// 替换全部出现位置
fn replace_all(
    text: &str,
    needle: &str,
    replacement: &str,
    case_sensitive: bool,
    whole_word: bool,
) -> String {
    if needle.is_empty() {
        return text.to_string();
    }

    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        match find_occurrence(rest, needle, case_sensitive, whole_word) {
            Some(pos) => {
                out.push_str(&rest[..pos]);
                out.push_str(replacement);
                rest = &rest[pos + needle.len()..];
                if rest.is_empty() {
                    break;
                }
            }
            None => {
                out.push_str(rest);
                break;
            }
        }
    }
    out
}

/// 大小写不敏感替换（占位符本身可能被引擎改写大小写）
fn replace_ignore_case(text: &str, needle: &str, replacement: &str) -> String {
    if needle.is_empty() {
        return text.to_string();
    }
    let lower_text = text.to_lowercase();
    let lower_needle = needle.to_lowercase();

    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    let mut rest_lower = lower_text.as_str();
    loop {
        match rest_lower.find(&lower_needle) {
            Some(pos) => {
                out.push_str(&rest[..pos]);
                out.push_str(replacement);
                rest = &rest[pos + needle.len()..];
                rest_lower = &rest_lower[pos + lower_needle.len()..];
                if rest.is_empty() {
                    break;
                }
            }
            None => {
                out.push_str(rest);
                break;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 术语表_基础占位与回填() {
        let g = Glossary::from_entries(vec![GlossaryEntry::new("爱丽丝", "Alice")]);
        let (processed, ctx) = g.apply_before("爱丽丝走进了花园。");
        assert!(processed.contains('\u{E000}'), "原文术语应被替换为占位符");
        assert!(!processed.contains("爱丽丝"));

        // 模拟引擎输出：占位符原样保留
        let translated = processed.replace("走进了花园。", " walked into the garden.");
        let restored = g.apply_after(&translated, &ctx);
        assert_eq!(restored, "Alice walked into the garden.");
    }

    #[test]
    fn 术语表_未命中时不改变原文() {
        let g = Glossary::from_entries(vec![GlossaryEntry::new(" Bob", "鲍勃")]);
        let (processed, ctx) = g.apply_before("今天天气很好。");
        assert_eq!(processed, "今天天气很好。");
        assert!(ctx.is_empty());
    }

    #[test]
    fn 术语表_长词优先避免子串抢占() {
        let g = Glossary::from_entries(vec![
            GlossaryEntry::new("Ann", "安"),
            GlossaryEntry::new("Anna", "安娜"),
        ]);
        let (processed, ctx) = g.apply_before("Anna and Ann");
        assert_eq!(ctx.matched()[0].src, "Anna", "长词应排在前");
        let restored = g.apply_after(&processed, &ctx);
        assert_eq!(restored, "安娜 and 安");
    }

    #[test]
    fn 术语表_忽略大小写() {
        let g = Glossary::from_entries(vec![GlossaryEntry::new("python", "派森")]);
        let (processed, ctx) = g.apply_before("I love PYTHON.");
        let restored = g.apply_after(&processed, &ctx);
        assert_eq!(restored, "I love 派森.");
    }

    #[test]
    fn 术语表_区分大小写时不误匹配() {
        let g = Glossary::from_entries(vec![GlossaryEntry::new("Python", "派森").case_sensitive(true)]);
        let (processed, _ctx) = g.apply_before("I love python.");
        assert_eq!(processed, "I love python.");
    }

    #[test]
    fn 术语表_整词匹配() {
        let g = Glossary::from_entries(vec![GlossaryEntry::new("Ann", "安").whole_word(true)]);
        let (processed, _ctx) = g.apply_before("Anna");
        assert_eq!(processed, "Anna", "整词模式下 Anna 不应被 Ann 命中");
    }

    #[test]
    fn 术语表_同一术语多处出现全部替换() {
        let g = Glossary::from_entries(vec![GlossaryEntry::new("北京", "Beijing")]);
        let (processed, ctx) = g.apply_before("北京到北京");
        let restored = g.apply_after(&processed, &ctx);
        assert_eq!(restored, "Beijing到Beijing");
    }

    #[test]
    fn 术语表_空译文条目被丢弃() {
        let g = Glossary::from_entries(vec![GlossaryEntry::new("foo", "")]);
        assert!(g.is_empty());
    }

    #[test]
    fn 术语表_prompt文本含备注() {
        let g = Glossary::from_entries(vec![
            GlossaryEntry::new("爱丽丝", "Alice").with_info("女主角"),
            GlossaryEntry::new("鲍勃", "Bob"),
        ]);
        let text = g.to_prompt_text();
        assert_eq!(text, "爱丽丝->Alice #女主角\n鲍勃->Bob");
    }

    #[test]
    fn 术语表_占位符位于私用区不会污染正常文本() {
        let g = Glossary::from_entries(vec![GlossaryEntry::new("甲", "A")]);
        let (processed, ctx) = g.apply_before("甲乙丙");
        assert!(processed.contains('\u{E000}'));
        let restored = g.apply_after(&processed, &ctx);
        assert_eq!(restored, "A乙丙");
    }

    #[test]
    fn 术语表_json往返() {
        let g = Glossary::from_entries(vec![GlossaryEntry::new("甲", "A").with_info("备注")]);
        let json = g.to_json().unwrap();
        let g2 = Glossary::from_json(&json).unwrap();
        assert_eq!(g2.len(), 1);
        assert_eq!(g2.entries()[0].dst, "A");
        assert_eq!(g2.entries()[0].info.as_deref(), Some("备注"));
    }
}
