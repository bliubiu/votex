//! IndexTTS-2.5 文本前端（Rust 移植）。
//!
//! 对照蓝本：`index_tts_2_5_onnx/core/frontend.py::Frontend`，处理顺序严格一致：
//! 1. 清洗（CHAR_REP_MAP 全角标点/引号/省略号等映射）
//! 2. 文本归一化（zh/zhen/en，Task #49 起真实移植，见 `normalizer.rs`）
//! 3. 大小写归一（ja/zh/zhen/en 小写；es 大写）
//! 4. 发音标注 `<词|读音>`（假名读音空格包裹；其余用 `<|SPECIAL_TOKEN_1/2|>` 包裹大写读音）
//! 5. 残留 `<|x|>` 内容大写
//! 6. 按预算分段（`budget = min(max_tokens, 602-2) - len("<|lang|> ")`，
//!    `<|SPECIAL_TOKEN_n|>...<|SPECIAL_TOKEN_n|>` 原子保留；标点后切分；超长逐字切）
//! 7. `encode("<|lang|> " + seg)` 后过滤 id<=1（0/1 是 start/stop-text）
//!
//! 归一化说明：`use_normalization=true` 且 lang ∈ (zh, zhen, en) 时走
//! `TextNormalizer`（wetext 0.1.8 FST 引擎 + 参考包装层，详见 `normalizer.rs`
//! 模块文档；yue/ja 等语言参考实现本就不归一化）。FST 不可用时（物化失败等
//! 罕见情况）记录警告并按 no-op 处理，与参考实现 wetext 缺失时的降级一致。
//! 金标对拍（docs/25 §7）以参考实现 `--no-normalization` 生成，对拍时前端
//! 需以 `use_normalization=false` 构建。

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Result};

use super::bpe::Tiktoken;

/// GPT 文本位置容量（text_pos_capacity，参考实现默认值）。
const TEXT_POS_CAPACITY: usize = 602;

/// 每段最大文本 token 数（参考实现默认值）。
pub const DEFAULT_MAX_TEXT_TOKENS_PER_SEGMENT: usize = 120;

/// tiktoken 词表文件名（registry 16 文件之一）
pub const VOCAB_FILE_NAME: &str = "multilingual_zh_ja_yue_char_del.tiktoken";

/// 中文/全角标点与符号清洗表。**顺序即正则交替顺序**：Python 正则左侧优先，
/// 例如 "，" 先于 "，，、" 意味着 "，，，" 实际被逐字替换为 ",,,"（"，，，"→"…" 为死规则），
/// 移植必须保持同序才能与参考实现逐字节一致。
/// normalizer.rs 复用本表（ZH_CHAR_REP_MAP = {"$": ".", **本表}）。
pub(crate) const CHAR_REP_MAP: &[(&str, &str)] = &[
    ("：", ","),
    ("；", ","),
    (";", ","),
    ("，", ","),
    ("。", "."),
    ("！", "!"),
    ("？", "?"),
    ("\n", " "),
    ("·", "-"),
    ("、", ","),
    ("...", "…"),
    (",,,", "…"),
    ("，，，", "…"),
    ("……", "…"),
    ("\u{201C}", "'"), // “
    ("\u{201D}", "'"), // ”
    ("\"", "'"),
    ("\u{2018}", "'"), // ‘
    ("\u{2019}", "'"), // ’
    ("（", "'"),
    ("）", "'"),
    ("(", "'"),
    (")", "'"),
    ("《", "'"),
    ("》", "'"),
    ("【", "'"),
    ("】", "'"),
    ("[", "'"),
    ("]", "'"),
    ("—", "-"),
    ("～", "-"),
    ("~", "-"),
    ("「", "'"),
    ("」", "'"),
    (":", ","),
];

/// 分段切分标点集合（等价 `re.split(r'(?<=[，。！？、；：,\.!\?;:\n])'）。
fn is_split_punct(ch: char) -> bool {
    matches!(
        ch,
        '，' | '。' | '！' | '？' | '、' | '；' | '：' | ',' | '.' | '!' | '?' | ';' | ':' | '\n'
    )
}

/// 判断是否全为假名（平假名或片假名；空串为 false，与参考实现 is_kana 一致）。
fn is_kana(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    if s.chars().all(|c| ('\u{3040}'..='\u{309F}').contains(&c)) {
        return true;
    }
    s.chars().all(|c| ('\u{30A0}'..='\u{30FF}').contains(&c))
}

/// 是否含汉字（\u4e00-\u9fff，决定发音标注用 SPECIAL_TOKEN_1 还是 2）。
fn has_chinese(s: &str) -> bool {
    s.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
}

/// IndexTTS-2.5 文本前端：清洗 -> 分段 -> token ids。
pub struct Frontend {
    encoding: Tiktoken,
    text_pos_capacity: usize,
    use_normalization: bool,
    normalizer: Option<super::normalizer::TextNormalizer>,
    clean_pattern: fancy_regex::Regex,
    clean_map: HashMap<&'static str, &'static str>,
    pron_pattern: fancy_regex::Regex,
    protected_pattern: fancy_regex::Regex,
    angle_pattern: fancy_regex::Regex,
}

impl Frontend {
    /// 从词表文件构建（`use_normalization` 见模块文档的归一化差异说明）。
    pub fn new(vocab_path: &Path, use_normalization: bool) -> Result<Self> {
        let encoding = Tiktoken::load(vocab_path)?;
        Self::with_encoding(encoding, use_normalization)
    }

    /// 从已构建的编码器构建（供测试注入合成词表）。
    pub fn with_encoding(encoding: Tiktoken, use_normalization: bool) -> Result<Self> {
        // 归一化器：FST 加载失败时降级为 no-op（与参考实现 wetext 缺失时的
        // ImportError -> 警告 + IdentityNormalizer 一致），不阻断前端构建。
        let normalizer = if use_normalization {
            match super::normalizer::TextNormalizer::new() {
                Ok(n) => Some(n),
                Err(e) => {
                    tracing::warn!(
                        "IndexTTS-2.5 文本归一化初始化失败，按 no-op 处理: {e:#}"
                    );
                    None
                }
            }
        } else {
            None
        };
        let clean_pattern = fancy_regex::Regex::new(
            &CHAR_REP_MAP
                .iter()
                .map(|(k, _)| regex::escape(k))
                .collect::<Vec<_>>()
                .join("|"),
        )
        .map_err(|e| anyhow!("清洗正则编译失败: {e}"))?;
        let clean_map: HashMap<&'static str, &'static str> =
            CHAR_REP_MAP.iter().copied().collect();
        let pron_pattern = fancy_regex::Regex::new(r"<([^|>\n]+)\|([^>\n]+)>")
            .map_err(|e| anyhow!("发音标注正则编译失败: {e}"))?;
        let protected_pattern =
            fancy_regex::Regex::new(r"<\|SPECIAL_TOKEN_\d+\|>.*?<\|SPECIAL_TOKEN_\d+\|>")
                .map_err(|e| anyhow!("受保护片段正则编译失败: {e}"))?;
        let angle_pattern = fancy_regex::Regex::new(r"<\|([^|]+)\|>")
            .map_err(|e| anyhow!("尖括号标记正则编译失败: {e}"))?;
        Ok(Self {
            encoding,
            text_pos_capacity: TEXT_POS_CAPACITY,
            use_normalization,
            normalizer,
            clean_pattern,
            clean_map,
            pron_pattern,
            protected_pattern,
            angle_pattern,
        })
    }

    /// 编码文本的 token 数（等价 `encode(text, allowed_special='all').len()`）。
    pub fn token_len(&self, text: &str) -> Result<usize> {
        Ok(self.encoding.encode(text)?.len())
    }

    /// 全流程：返回每段的裸 token id 列表（无 0/1 填充，prefill 侧自行添加）。
    pub fn prepare(
        &self,
        text: &str,
        lang: &str,
        max_text_tokens_per_segment: usize,
    ) -> Result<Vec<Vec<u32>>> {
        let lang = lang.to_lowercase();
        let lang_prefix = format!("<|{lang}|> ");

        // 1. 清洗
        let mut text = self.clean_text(text)?;
        // 2. 归一化（zh/zhen/en 走 TextNormalizer；yue/ja 等参考实现本就不归一）
        if self.use_normalization && matches!(lang.as_str(), "zh" | "zhen" | "en") {
            if let Some(normalizer) = &self.normalizer {
                text = normalizer.normalize(&text)?;
            }
            // normalizer 为 None（初始化失败降级）时按 no-op 处理，构造时已警告
        }
        // 3. 大小写归一
        match lang.as_str() {
            "ja" | "zh" | "zhen" | "en" => text = text.to_lowercase(),
            "es" => text = text.to_uppercase(),
            _ => {}
        }
        // 4. 发音标注
        let text = self.apply_pronunciation_annotations(&text)?;
        // 5. 残留 <|x|> 内容大写
        let text = self.uppercase_angle_tokens(&text)?;

        // 6. 分段 + 7. 编码过滤
        let segments = self.split_text_by_tokens(&text, max_text_tokens_per_segment, &lang_prefix)?;
        segments
            .iter()
            .map(|seg| {
                Ok(self
                    .encoding
                    .encode(&format!("{lang_prefix}{seg}"))?
                    .into_iter()
                    .filter(|t| *t > 1)
                    .collect())
            })
            .collect()
    }

    /// 按预算把文本切成若干段（等价 split_text_by_tokens）。
    pub fn split_text_by_tokens(
        &self,
        text: &str,
        max_tokens: usize,
        lang_prefix: &str,
    ) -> Result<Vec<String>> {
        let budget = (max_tokens.min(self.text_pos_capacity - 2))
            .saturating_sub(self.token_len(lang_prefix)?)
            .max(1);
        if self.token_len(text)? <= budget {
            return Ok(vec![text.to_string()]);
        }

        let mut chunks: Vec<String> = Vec::new();
        for (piece, atomic) in self.split_atomic_pieces(text)? {
            if atomic {
                chunks.push(piece);
                continue;
            }
            for part in split_after_punctuation(&piece) {
                if part.is_empty() {
                    continue;
                }
                if self.token_len(&part)? <= budget {
                    chunks.push(part);
                    continue;
                }
                // 超长片段逐字切
                let mut current = String::new();
                for ch in part.chars() {
                    let mut candidate = current.clone();
                    candidate.push(ch);
                    if !current.is_empty() && self.token_len(&candidate)? > budget {
                        chunks.push(std::mem::take(&mut current));
                    }
                    current.push(ch);
                }
                if !current.is_empty() {
                    chunks.push(current);
                }
            }
        }

        let mut segments: Vec<String> = Vec::new();
        let mut current = String::new();
        for chunk in chunks {
            if !current.is_empty() {
                let mut candidate = current.clone();
                candidate.push_str(&chunk);
                if self.token_len(&candidate)? > budget {
                    segments.push(std::mem::take(&mut current));
                }
            }
            current.push_str(&chunk);
        }
        if !current.is_empty() {
            segments.push(current);
        }
        if segments.is_empty() {
            return Ok(vec![text.to_string()]);
        }
        Ok(segments)
    }

    /// 按 `<|SPECIAL_TOKEN_n|>...<|SPECIAL_TOKEN_n|>` 切出原子片段。
    fn split_atomic_pieces(&self, text: &str) -> Result<Vec<(String, bool)>> {
        let mut pieces = Vec::new();
        let mut pos = 0;
        for mat in self.protected_pattern.find_iter(text) {
            let m = mat.map_err(|e| anyhow!("受保护片段正则执行失败: {e}"))?;
            if m.start() > pos {
                pieces.push((text[pos..m.start()].to_string(), false));
            }
            pieces.push((m.as_str().to_string(), true));
            pos = m.end();
        }
        if pos < text.len() {
            pieces.push((text[pos..].to_string(), false));
        }
        Ok(pieces)
    }

    /// 清洗：CHAR_REP_MAP 替换（match 结果即键，查表取替换值）。
    fn clean_text(&self, text: &str) -> Result<String> {
        let mut out = String::with_capacity(text.len());
        let mut last = 0;
        for mat in self.clean_pattern.find_iter(text) {
            let m = mat.map_err(|e| anyhow!("清洗正则执行失败: {e}"))?;
            let rep = self
                .clean_map
                .get(m.as_str())
                .copied()
                .ok_or_else(|| anyhow!("清洗表缺少匹配项: {:?}", m.as_str()))?;
            out.push_str(&text[last..m.start()]);
            out.push_str(rep);
            last = m.end();
        }
        out.push_str(&text[last..]);
        Ok(out)
    }

    /// 发音标注 `<词|读音>`：假名读音两侧空格包裹；否则按词中是否含汉字
    /// 用 `<|SPECIAL_TOKEN_2|>` / `<|SPECIAL_TOKEN_1|>` 包裹大写读音。
    fn apply_pronunciation_annotations(&self, text: &str) -> Result<String> {
        let mut out = String::with_capacity(text.len());
        let mut last = 0;
        for caps in self.pron_pattern.captures_iter(text) {
            let caps = caps.map_err(|e| anyhow!("发音标注正则执行失败: {e}"))?;
            let Some(whole) = caps.get(0) else { continue };
            let word = caps.get(1).map(|g| g.as_str()).unwrap_or_default();
            let pron = caps.get(2).map(|g| g.as_str()).unwrap_or_default().to_uppercase();
            let replacement = if is_kana(&pron) {
                format!(" {pron} ")
            } else {
                let token = if has_chinese(word) {
                    "SPECIAL_TOKEN_2"
                } else {
                    "SPECIAL_TOKEN_1"
                };
                format!("<|{token}|>{pron}<|{token}|>")
            };
            out.push_str(&text[last..whole.start()]);
            out.push_str(&replacement);
            last = whole.end();
        }
        out.push_str(&text[last..]);
        Ok(out)
    }

    /// 残留 `<|x|>` 标记内容大写（等价 `re.sub(r'<\|([^|]+)\|>', ...)`）。
    fn uppercase_angle_tokens(&self, text: &str) -> Result<String> {
        let mut out = String::with_capacity(text.len());
        let mut last = 0;
        for caps in self.angle_pattern.captures_iter(text) {
            let caps = caps.map_err(|e| anyhow!("尖括号标记正则执行失败: {e}"))?;
            let Some(whole) = caps.get(0) else { continue };
            let inner = caps.get(1).map(|g| g.as_str()).unwrap_or_default().to_uppercase();
            out.push_str(&text[last..whole.start()]);
            out.push_str(&format!("<|{inner}|>"));
            last = whole.end();
        }
        out.push_str(&text[last..]);
        Ok(out)
    }
}

/// 在标点集合内字符之后切分（等价 Python 零宽 lookbehind split；
/// 产生的空片段由调用方过滤）。
fn split_after_punctuation(piece: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut start = 0;
    for (idx, ch) in piece.char_indices() {
        if is_split_punct(ch) {
            let end = idx + ch.len_utf8();
            parts.push(piece[start..end].to_string());
            start = end;
        }
    }
    if start < piece.len() {
        parts.push(piece[start..].to_string());
    }
    parts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tts::indextts25::bpe::lang_id;

    fn synthetic_frontend() -> Frontend {
        // 合成词表：256 单字节 + abc 链，special 仅 <|x|>
        let mut ranks = HashMap::new();
        for b in 0..=255u8 {
            ranks.insert(vec![b], b as u32);
        }
        ranks.insert(b"ab".to_vec(), 256);
        ranks.insert(b"abc".to_vec(), 257);
        let enc = Tiktoken::from_parts(ranks, vec!["<|x|>".to_string()]).unwrap();
        Frontend::with_encoding(enc, false).unwrap()
    }

    #[test]
    fn 清洗_全角标点与省略号() {
        let f = synthetic_frontend();
        assert_eq!(f.clean_text("你好。再见！").unwrap(), "你好.再见!");
        assert_eq!(f.clean_text("，，，").unwrap(), ",,,", "左优先语义：逐字替换");
        assert_eq!(f.clean_text("……").unwrap(), "…");
        assert_eq!(f.clean_text("“引号”").unwrap(), "'引号'");
        assert_eq!(f.clean_text("a\nb").unwrap(), "a b");
        assert_eq!(f.clean_text("A：B").unwrap(), "A,B");
    }

    #[test]
    fn 清洗_左右优先顺序与参考实现一致() {
        let f = synthetic_frontend();
        // "..."（三个 ASCII 点）命中后不再作为三个 "." 处理（"." 本身不在表中，行为一致）
        assert_eq!(f.clean_text("wait...").unwrap(), "wait…");
        // "、" 与 "，" 均映射为 ","
        assert_eq!(f.clean_text("甲、乙").unwrap(), "甲,乙");
    }

    #[test]
    fn 发音标注_假名与字母读音() {
        let f = synthetic_frontend();
        // 假名读音：两侧空格包裹
        assert_eq!(f.apply_pronunciation_annotations("<东京|とうきょう>").unwrap(), " とうきょう ");
        // 含汉字词 + 字母读音：SPECIAL_TOKEN_2 包裹大写读音
        assert_eq!(
            f.apply_pronunciation_annotations("<深圳|shenzhen>").unwrap(),
            "<|SPECIAL_TOKEN_2|>SHENZHEN<|SPECIAL_TOKEN_2|>"
        );
        // 纯字母词：SPECIAL_TOKEN_1
        assert_eq!(
            f.apply_pronunciation_annotations("<GPU|gpu>").unwrap(),
            "<|SPECIAL_TOKEN_1|>GPU<|SPECIAL_TOKEN_1|>"
        );
        // 标注外的文本保持原样
        assert_eq!(
            f.apply_pronunciation_annotations("我说<深圳|shenzhen>话").unwrap(),
            "我说<|SPECIAL_TOKEN_2|>SHENZHEN<|SPECIAL_TOKEN_2|>话"
        );
    }

    #[test]
    fn 尖括号残留标记大写() {
        let f = synthetic_frontend();
        assert_eq!(f.uppercase_angle_tokens("<|yue|> text").unwrap(), "<|YUE|> text");
        assert_eq!(f.uppercase_angle_tokens("plain").unwrap(), "plain");
    }

    #[test]
    fn 切分_标点后断句并过滤空段() {
        assert_eq!(split_after_punctuation("你好。再见！"), vec!["你好。", "再见！"]);
        assert_eq!(split_after_punctuation("！！"), vec!["！", "！"]);
        assert_eq!(split_after_punctuation("abc"), vec!["abc"]);
        assert!(split_after_punctuation("").is_empty());
    }

    #[test]
    fn 假名判定() {
        assert!(is_kana("かな"));
        assert!(is_kana("カナ"));
        assert!(!is_kana("かなカナ"), "混排不算假名串");
        assert!(!is_kana("汉字"));
        assert!(!is_kana(""));
    }

    #[test]
    fn 分段_预算内不切分() {
        let f = synthetic_frontend();
        let segs = f.split_text_by_tokens("abc", 120, "<|en|> ").unwrap();
        assert_eq!(segs, vec!["abc"]);
    }

    #[test]
    fn 分段_超长按标点与预算切分() {
        let f = synthetic_frontend();
        // 预算 = min(120, 600) - 3("<|en|> " = "<|", "en", "|>", " ") —— 用 token_len 实测
        let prefix_len = f.token_len("<|en|> ").unwrap();
        let long_text = format!("{}。", "abcde. ".repeat(200));
        let segs = f.split_text_by_tokens(&long_text, 120, "<|en|> ").unwrap();
        assert!(segs.len() > 1, "超长文本必须分段");
        let budget = 120usize.min(600) - prefix_len;
        for seg in &segs {
            assert!(
                f.token_len(seg).unwrap() <= budget,
                "每段 token 数不得超过预算 {budget}"
            );
        }
    }

    #[test]
    fn prepare_合成词表_过滤01与语言前缀() {
        let f = synthetic_frontend();
        let segs = f.prepare("你好。世界", "en", 120).unwrap();
        assert_eq!(segs.len(), 1);
        for ids in &segs {
            assert!(ids.iter().all(|t| *t > 1), "0/1 必须被过滤");
        }
    }

    /// 真实词表端到端：语言前缀 `<|yue|>` 必须解析为 special id 58937。
    #[test]
    fn prepare_真实词表_粤语前缀与过滤() {
        let Some(vocab) = crate::tts::indextts25::bpe::find_real_vocab() else {
            eprintln!("跳过：未找到真实 tiktoken 词表");
            return;
        };
        let f = Frontend::new(&vocab, false).unwrap();
        let segs = f.prepare("哈喽，今日天气几好！GO！", "yue", 120).unwrap();
        assert_eq!(segs.len(), 1);
        let ids = &segs[0];
        assert_eq!(ids[0], 58937, "首 token 必须是 <|yue|> special id");
        assert!(ids.iter().all(|t| *t > 1), "0/1 必须被过滤");
        // 语言 id 锚点（gpt_prefill 的 lang_id 输入）
        assert_eq!(lang_id("yue"), 99);
    }
}
