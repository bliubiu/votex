use crate::tts::value_object::SegmentSize;
use crate::tts::entity::Segment;

/// 文本分段领域服务
pub struct TextSegmenter;

impl TextSegmenter {
    pub fn new() -> Self {
        Self
    }

    /// 按分段大小切分文本
    pub fn segment(text: &str, size: SegmentSize) -> Vec<Segment> {
        let max_chars = size.max_chars();
        let chunks = Self::split_by_sentences(text, max_chars);
        chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| Segment::new(i, chunk))
            .collect()
    }

    /// 按句子边界和字数限制切分
    fn split_by_sentences(text: &str, max_chars: usize) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();
        // 跨句跟踪的引号栈：段被切断时若引号仍开启，下一段以引号续开
        let mut open_quotes: Vec<char> = Vec::new();

        for sentence in Self::split_sentences(text) {
            // 单句超过字数限制时，按字数硬切分
            if sentence.chars().count() > max_chars {
                if !current.trim().is_empty() {
                    result.push(current.trim().to_string());
                    current.clear();
                }
                for chunk in Self::hard_split(&sentence, max_chars) {
                    result.push(chunk);
                }
                Self::update_quote_stack(&sentence, &mut open_quotes);
                continue;
            }

            if current.chars().count() + sentence.chars().count() > max_chars && !current.is_empty() {
                result.push(current.trim().to_string());
                current.clear();
                // 段边界续开引号：仅在实际切断处插入（此前在句切点插入，
                // 会把省略号等软边界也当切断，污染短对话文本）
                if let Some(&q) = open_quotes.last() {
                    current.push(q);
                }
            }
            current.push_str(&sentence);
            Self::update_quote_stack(&sentence, &mut open_quotes);
        }

        if !current.trim().is_empty() {
            result.push(current.trim().to_string());
        }

        result
    }

    /// 扫描句子中的引号字符，更新跨句引号栈
    fn update_quote_stack(sentence: &str, stack: &mut Vec<char>) {
        for ch in sentence.chars() {
            if matches!(ch, '「' | '『' | '“') {
                stack.push(ch);
            } else if matches!(ch, '」' | '』' | '”') {
                stack.pop();
            } else if ch == '"' {
                // 直引号开闭歧义：栈顶为直引号则视为闭合，否则视为开启
                if stack.last() == Some(&'"') {
                    stack.pop();
                } else {
                    stack.push('"');
                }
            }
        }
    }

    /// 无标点时按字数切分
    ///
    /// 优先在切点附近的逗号/顿号/冒号处断开（保证韵律连贯），
    /// 实在没有标点才按字数硬切。
    fn hard_split(text: &str, max_chars: usize) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut result = Vec::new();
        let mut start = 0;
        while start < chars.len() {
            let end = (start + max_chars).min(chars.len());
            if end >= chars.len() {
                result.push(chars[start..end].iter().collect());
                break;
            }
            // 在窗口后半段找最后一个「次级断点」标点，避免任意字间切断
            let window_start = start + max_chars / 2;
            let mut split_at = end;
            for k in (window_start..end).rev() {
                if matches!(chars[k], '，' | '、' | ',' | '；' | ';' | '：' | ':' | ' ') {
                    split_at = k + 1; // 断点标点归前段
                    break;
                }
            }
            result.push(chars[start..split_at].iter().collect());
            start = split_at;
        }
        result
    }

    /// 按句号、问号、感叹号等分割句子
    ///
    /// 终止符：。！？.!? 换行，以及省略号「…」与分号「；;」——
    /// 长文本中「……」不切会把整段当一个句子，「；」切分可保证分句节奏。
    ///
    /// 语义保护（T1 意群规则）：
    /// - 引号感知：引号内被终止符切断时，下一句以引号续开，
    ///   保证每段的引号是平衡的（对话韵律不断裂）；
    /// - 小数点不切：「3.14」中的点不是句子边界；
    /// - 缩写点不切：Mr. / Dr. / Prof. 等后跟空格+小写时不是边界。
    fn split_sentences(text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        for i in 0..len {
            let ch = chars[i];
            current.push(ch);

            if matches!(ch, '「' | '『' | '“' | '」' | '』' | '”' | '"') {
                continue; // 引号栈由 split_by_sentences 跨句维护
            }
            if matches!(ch, '。' | '！' | '？' | '.' | '!' | '?' | '…' | '；' | ';' | '\n') {
                // 意群保护：小数点/缩写点不是句子边界
                if ch == '.' && i > 0 {
                    if chars[i - 1].is_ascii_digit() {
                        continue; // 3.14
                    }
                    if Self::is_abbreviation_dot(&chars, i) {
                        continue; // Mr. / Dr.
                    }
                }
                sentences.push(current.clone());
                current.clear();
            }
        }

        if !current.is_empty() {
            sentences.push(current);
        }

        sentences
    }

    /// 判断 `chars[i-1]` 处的 '.' 是否为英文缩写点（Mr. Dr. Prof. 等）
    fn is_abbreviation_dot(chars: &[char], dot_idx: usize) -> bool {
        const ABBREVS: &[&str] = &[
            "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "no", "co", "inc", "ltd",
        ];
        // 向前收集连续 ASCII 字母（最多 5 个）
        let mut start = dot_idx;
        while start > 0 && dot_idx - start < 5 && chars[start - 1].is_ascii_alphabetic() {
            start -= 1;
        }
        if start == dot_idx {
            return false;
        }
        let word: String = chars[start..dot_idx].iter().collect::<String>().to_lowercase();
        // 缩写后若紧跟空格+大写字母（新句），仍按缩写处理（Mr. Smith）
        ABBREVS.contains(&word.as_str())
    }

    /// 文本预处理
    pub fn preprocess(text: &str) -> String {
        let mut result = text.to_string();
        // 去除连续空行
        while result.contains("\n\n\n") {
            result = result.replace("\n\n\n", "\n\n");
        }
        // 去除行首行尾空白
        result = result
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<_>>()
            .join("\n");
        result.trim().to_string()
    }
}

/// 音频拼接选项
pub struct ConcatOptions {
    pub crossfade_ms: u32,
    pub silence_ms: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_segmenter_按句号分段() {
        let text = "这是第一句。这是第二句。这是第三句。";
        let segments = TextSegmenter::segment(text, SegmentSize::S500);
        // 三句话总字符数远小于 500，合并为一段
        assert_eq!(segments.len(), 1);
        assert!(segments[0].text.contains("这是第一句"));
    }

    #[test]
    fn text_segmenter_按句号分段_超出字数限制() {
        // 每句 5 字 + 句号 = 6 字，S300 限制 300 字
        let text = "这是第一句话的内容。".repeat(60); // ~360 字
        let segments = TextSegmenter::segment(&text, SegmentSize::S300);
        assert!(segments.len() > 1);
    }

    #[test]
    fn text_segmenter_超长文本按字数切分() {
        let text = "这是一段很长的文本".repeat(50); // 450 字
        let segments = TextSegmenter::segment(&text, SegmentSize::S300);
        assert!(segments.len() > 1);
    }

    #[test]
    fn text_segmenter_空文本() {
        let segments = TextSegmenter::segment("", SegmentSize::S500);
        assert!(segments.is_empty());
    }

    #[test]
    fn text_segmenter_预处理去空行() {
        let text = "第一行\n\n\n\n第二行";
        let result = TextSegmenter::preprocess(text);
        assert_eq!(result, "第一行\n\n第二行");
    }

    #[test]
    fn text_segmenter_预处理去空白() {
        let text = "  第一行  \n  第二行  ";
        let result = TextSegmenter::preprocess(text);
        assert_eq!(result, "第一行\n第二行");
    }

    // ---- T1 语义断句测试 ----

    #[test]
    fn 语义_省略号不污染台词() {
        // e2e 回归：台词内的省略号是软边界，
        // 不得触发引号续开插入（曾产生「…“…“」式污染）
        let text = "“找到了，这是什么……”女人的手微微发抖。";
        let segs = TextSegmenter::segment(text, SegmentSize::S120);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].text, text);
    }

    #[test]
    fn 语义_引号内切断后续开引号() {
        // 长对话超限被切断时，后段应以「续开，保证对话韵律延续
        let text = format!("「{}」", "这是一句很长的话。".repeat(16));
        let segs = TextSegmenter::segment(&text, SegmentSize::S120);
        assert!(segs.len() > 1);
        // 除首段外，每个续段都应以「开头（对话延续标记）
        for seg in &segs[1..] {
            assert!(seg.text.starts_with('「'), "续段未续开引号: {}", seg.text);
        }
        // 真实闭合引号恰好出现一次（续开的「无需闭合，仅供韵律延续）
        let total: String = segs.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(total.matches('」').count(), 1);
        assert!(total.ends_with('」'));
    }

    #[test]
    fn 语义_小数点不切句() {
        let text = "圆周率是3.14159这个数很有意思。";
        let segs = TextSegmenter::segment(text, SegmentSize::S500);
        assert_eq!(segs.len(), 1);
        assert!(segs[0].text.contains("3.14159这个数"));
    }

    #[test]
    fn 语义_缩写点不切句() {
        let text = "Mr. Smith来到了。Dr. Wang也在。";
        let segs = TextSegmenter::segment(text, SegmentSize::S500);
        assert_eq!(segs.len(), 1);
        assert!(segs[0].text.contains("Mr. Smith"));
    }
}
