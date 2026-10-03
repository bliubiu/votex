//! 长文本分段器
//!
//! 离线翻译模型都有输入长度上限（NLLB 200 token、M2M-100 512、Opus-MT 64 步解码），
//! 长文本整段送入会被**静默截断**，导致译文残缺。
//!
//! 本模块把长文本切成模型可承受的片段，逐段翻译后再拼接。
//!
//! # 切分优先级
//! 1. 空行（段落边界）—— 语义最完整
//! 2. 句末标点（`。！？!?.；;` 等）—— 句子边界
//! 3. 逗号、顿号 —— 长句的次优断点
//! 4. 硬切（按字符数）—— 兜底，避免单句超长

/// 分段配置
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentConfig {
    /// 单段最大字符数
    pub max_chars: usize,
    /// 是否在段落之间插入空行拼接
    pub keep_paragraph_break: bool,
}

impl Default for SegmentConfig {
    fn default() -> Self {
        Self {
            max_chars: 200,
            keep_paragraph_break: true,
        }
    }
}

impl SegmentConfig {
    /// 使用指定最大长度创建配置
    pub fn with_max_chars(max_chars: usize) -> Self {
        Self {
            max_chars: max_chars.max(1),
            keep_paragraph_break: true,
        }
    }
}

/// 长文本分段器
pub struct TextSegmenter;

impl TextSegmenter {
    /// 按段落切分（保留空行信息）
    ///
    /// 连续空行视为段落分隔；单换行视为段内软换行，予以保留。
    pub fn split_paragraphs(text: &str) -> Vec<String> {
        let mut paragraphs = Vec::new();
        let mut current = String::new();

        for line in text.lines() {
            if line.trim().is_empty() {
                if !current.trim().is_empty() {
                    paragraphs.push(current.trim().to_string());
                    current.clear();
                }
                continue;
            }
            if !current.is_empty() {
                current.push('\n');
            }
            current.push_str(line);
        }
        if !current.trim().is_empty() {
            paragraphs.push(current.trim().to_string());
        }

        if paragraphs.is_empty() && !text.trim().is_empty() {
            paragraphs.push(text.trim().to_string());
        }
        paragraphs
    }

    /// 在一段文本内按句子边界切分
    pub fn split_sentences(paragraph: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        for ch in paragraph.chars() {
            current.push(ch);
            if is_sentence_end(ch) {
                let s = current.trim();
                if !s.is_empty() {
                    sentences.push(s.to_string());
                }
                current.clear();
            }
        }
        let tail = current.trim();
        if !tail.is_empty() {
            sentences.push(tail.to_string());
        }
        sentences
    }

    /// 把超长句子按次级标点或直接按字符数硬切
    fn split_long_sentence(sentence: &str, max_chars: usize) -> Vec<String> {
        if sentence.chars().count() <= max_chars {
            return vec![sentence.to_string()];
        }

        // 先尝试按次级标点切
        let mut chunks = Vec::new();
        let mut current = String::new();
        for ch in sentence.chars() {
            current.push(ch);
            if is_minor_break(ch) && current.chars().count() >= max_chars / 2 {
                chunks.push(current.trim().to_string());
                current.clear();
            }
        }
        let tail = current.trim();
        if !tail.is_empty() {
            chunks.push(tail.to_string());
        }

        // 仍有过长片段则按字符数硬切
        let mut result = Vec::new();
        for chunk in chunks {
            if chunk.chars().count() <= max_chars {
                result.push(chunk);
            } else {
                let chars: Vec<char> = chunk.chars().collect();
                let mut start = 0;
                while start < chars.len() {
                    let end = (start + max_chars).min(chars.len());
                    result.push(chars[start..end].iter().collect::<String>());
                    start = end;
                }
            }
        }
        result
    }

    /// 将长文本切分为多个可翻译片段
    ///
    /// # 参数
    /// - `text`: 待翻译的原文
    /// - `config`: 分段配置
    pub fn segment(text: &str, config: &SegmentConfig) -> Vec<String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }

        let mut segments = Vec::new();
        for paragraph in Self::split_paragraphs(trimmed) {
            for sentence in Self::split_sentences(&paragraph) {
                segments.extend(Self::split_long_sentence(&sentence, config.max_chars));
            }
        }

        // 相邻短片段合并，减少请求次数、也有利于模型理解上下文
        Self::merge_small(&mut segments, config.max_chars);
        segments
    }

    /// 合并相邻短片段，直到接近 max_chars
    fn merge_small(segments: &mut Vec<String>, max_chars: usize) {
        if segments.len() <= 1 {
            return;
        }

        let mut merged: Vec<String> = Vec::with_capacity(segments.len());
        let mut current = String::new();

        for seg in segments.drain(..) {
            if current.is_empty() {
                current = seg;
                continue;
            }
            let combined_len = current.chars().count() + seg.chars().count();
            if combined_len <= max_chars {
                current = format!("{}{}", current, seg);
            } else {
                merged.push(std::mem::take(&mut current));
                current = seg;
            }
        }
        if !current.is_empty() {
            merged.push(current);
        }
        *segments = merged;
    }

    /// 把译文片段拼回完整文本
    ///
    /// 中文译文直接相连；含拉丁字母的片段之间补一个空格。
    pub fn join(parts: &[String]) -> String {
        let mut out = String::new();
        for (i, part) in parts.iter().enumerate() {
            let part = part.trim();
            if part.is_empty() {
                continue;
            }
            if i > 0 && !out.is_empty() {
                let need_space =
                    ends_with_latin(out.chars().last()) && starts_with_latin(part.chars().next());
                if need_space {
                    out.push(' ');
                }
            }
            out.push_str(part);
        }
        out
    }
}

/// 句末标点
fn is_sentence_end(ch: char) -> bool {
    matches!(ch, '。' | '！' | '？' | '!' | '?' | '…') || ch == '.' || ch == '；' || ch == ';'
}

/// 次级断点（长句内部可切分处）
fn is_minor_break(ch: char) -> bool {
    matches!(ch, '，' | '、' | ',' | '：' | ':' | '）' | ')' | '」' | '』' | '"')
}

fn ends_with_latin(ch: Option<char>) -> bool {
    ch.map(|c| c.is_ascii_alphanumeric()).unwrap_or(false)
}

fn starts_with_latin(ch: Option<char>) -> bool {
    ch.map(|c| c.is_ascii_alphanumeric()).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 分段_短文本不切分() {
        let segs = TextSegmenter::segment("你好世界", &SegmentConfig::default());
        assert_eq!(segs, vec!["你好世界".to_string()]);
    }

    #[test]
    fn 分段_空文本返回空() {
        assert!(TextSegmenter::segment("   ", &SegmentConfig::default()).is_empty());
        assert!(TextSegmenter::segment("", &SegmentConfig::default()).is_empty());
    }

    #[test]
    fn 分段_按句子边界切分() {
        let text = "今天天气很好。我们出去散步吧！你觉得呢？";
        let segs = TextSegmenter::segment(text, &SegmentConfig::with_max_chars(10));
        assert!(segs.len() >= 3, "应至少切成 3 段，实际: {:?}", segs);
        assert!(segs.iter().all(|s| s.chars().count() <= 10));
    }

    #[test]
    fn 分段_空行视为段落边界() {
        let text = "第一段内容。\n\n第二段内容。";
        let segs = TextSegmenter::split_paragraphs(text);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0], "第一段内容。");
        assert_eq!(segs[1], "第二段内容。");
    }

    #[test]
    fn 分段_超长句硬切保证不超上限() {
        let long = "这是一个很长的句子".repeat(40);
        let segs = TextSegmenter::segment(&long, &SegmentConfig::with_max_chars(50));
        assert!(segs.len() > 1);
        assert!(
            segs.iter().all(|s| s.chars().count() <= 50),
            "存在超长片段: {:?}",
            segs.iter().map(|s| s.chars().count()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn 分段_短片段会自动合并() {
        let segs = TextSegmenter::segment("你好。世界。", &SegmentConfig::with_max_chars(200));
        assert_eq!(segs.len(), 1, "短句应合并为一段: {:?}", segs);
    }

    #[test]
    fn 拼接_英文片段自动补空格() {
        let parts = vec!["Hello".to_string(), "world".to_string()];
        assert_eq!(TextSegmenter::join(&parts), "Hello world");
    }

    #[test]
    fn 拼接_中文片段不插空格() {
        let parts = vec!["你好".to_string(), "世界".to_string()];
        assert_eq!(TextSegmenter::join(&parts), "你好世界");
    }

    #[test]
    fn 拼接_跳过空片段() {
        let parts = vec!["你好".to_string(), "  ".to_string(), "世界".to_string()];
        assert_eq!(TextSegmenter::join(&parts), "你好世界");
    }

    #[test]
    fn 分段_内容无损() {
        let text = "今天天气很好。我们出去散步吧！你觉得呢？";
        let segs = TextSegmenter::segment(text, &SegmentConfig::with_max_chars(10));
        let joined: String = segs.concat();
        let cleaned: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        let joined_cleaned: String = joined.chars().filter(|c| !c.is_whitespace()).collect();
        assert_eq!(joined_cleaned, cleaned);
    }
}
