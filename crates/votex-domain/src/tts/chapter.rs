/// 章节感知切分
///
/// 识别中文小说/文档的章节标记行（「第X章」「序章」「Chapter 1」等），
/// 把长文本切分为章节单元。章节边界强制开新段——此前章节标题会与正文
/// 粘在同一段里，导致标题被「读进」上一段末尾、韵律断裂。
///
/// 典型标记（行首匹配，行长 ≤ 60 字符，行内不含句末标点）：
/// - `第一章` / `第 12 回` / `第三百卷` / `第 5 节` / `第X篇/部/集/话/幕`
/// - `Chapter 3` / `chapter iii`
/// - `序章` `楔子` `引子` `终章` `尾声` `番外` `前言` `后记`

/// 章节单元
#[derive(Debug, Clone, PartialEq)]
pub struct Chapter {
    /// 章节标题（标记行原文）；无标记的正文块为 None
    pub title: Option<String>,
    /// 章节正文（不含标记行）
    pub body: String,
}

/// 章节单位字（pub(crate)：role.rs 的噪声过滤复用同一套语义，避免两处定义漂移）
pub(crate) const CHAPTER_UNITS: &[char] = &['章', '回', '卷', '节', '篇', '部', '集', '话', '幕', '场'];

/// 合法的编号字符（中文数字/阿拉伯数字/间隔符）
fn is_numberish(c: char) -> bool {
    matches!(c, '零' | '一' | '二' | '三' | '四' | '五' | '六' | '七' | '八' | '九' | '十'
        | '百' | '千' | '万' | '亿' | '两' | '〇' | 'Ｏ' | '○' | 'O' | 'o'
        | '0'..='9' | '０'..='９' | ' ' | '　' | '.' | '、')
}

/// 「第 + 编号 + 单位字」强信号判定（编号部分只允许数字字符，
/// 防「第三天，他早早起床。」这类叙述句误判）
///
/// 单位字后的余文需 ≤ 30 字：真实章节标题较短，
/// 而「第二部分连续更新了十二章，……出问题了？」这类叙述句余文很长
fn is_numbered_marker(trimmed: &str) -> bool {
    let Some(rest) = trimmed.strip_prefix('第') else {
        return false;
    };
    let rest = rest.trim_start();
    let mut num_end = 0;
    for c in rest.chars() {
        if is_numberish(c) {
            num_end += 1;
        } else {
            break;
        }
    }
    if num_end == 0 || num_end > 10 {
        return false;
    }
    let after: String = rest.chars().skip(num_end).take(2).collect();
    let Some(unit_char) = after.chars().next() else {
        return false;
    };
    if !CHAPTER_UNITS.contains(&unit_char) {
        return false;
    }
    // 余文长度（单位字之后，不含单位字本身）
    let remainder_chars = rest.chars().count().saturating_sub(num_end + 1);
    remainder_chars <= 30
}

/// 固定章节标记（前缀匹配即可判定）
const FIXED_MARKERS: &[&str] = &[
    "序章", "楔子", "引子", "终章", "尾声", "番外", "前言", "后记", "序言", "自序", "尾声",
];

/// 判断一行文本是否为章节标记行
pub fn is_chapter_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.chars().count() > 60 {
        return false;
    }

    // 模式一：第X章/回/卷/节/篇/部/集/话/幕/场 —— 强信号，先行判定。
    // 必须先于句末标点排除：真实章节标题可含问号/叹号（如「第0001章 女儿？」），
    // 但「第 + 编号 + 单位字 + 短余文」已足够排除普通叙述句；
    // 唯句号/省略号结尾仍判叙述（「第一章的内容让我印象深刻。」——
    // 书名极少以句号或省略号收尾）
    if is_numbered_marker(trimmed) {
        return !trimmed.ends_with('。')
            && !trimmed.ends_with('.')
            && !trimmed.ends_with('…');
    }

    // 行尾是句末标点 → 更可能是叙述句，排除（仅对弱信号模式生效）
    if trimmed.ends_with('。') || trimmed.ends_with('！') || trimmed.ends_with('？')
        || trimmed.ends_with('.') || trimmed.ends_with('!') || trimmed.ends_with('?')
    {
        return false;
    }

    // 模式二：Chapter N（不区分大小写）
    let lower = trimmed.to_lowercase();
    if let Some(rest) = lower.strip_prefix("chapter ") {
        let rest_trim = rest.trim_start();
        return rest_trim.chars().next().is_some_and(|c| c.is_ascii_digit() || c.is_ascii_alphabetic());
    }

    // 模式三：固定标记（整行以标记开头且总长较短）
    FIXED_MARKERS.iter().any(|m| {
        trimmed.starts_with(m) && trimmed.chars().count() <= m.chars().count() + 30
    })
}

/// 把长文本按章节标记切分
///
/// 无任何标记时返回单个 body = 原文的伪章节（title: None）。
/// 标记行之前的正文（如版权页）作为 title: None 的首块。
pub fn split_into_chapters(text: &str) -> Vec<Chapter> {
    let mut chapters: Vec<Chapter> = Vec::new();
    let mut current_body = String::new();
    let mut found_any = false;

    for line in text.lines() {
        if is_chapter_line(line) {
            found_any = true;
            // 收尾上一个块
            let body = current_body.trim().to_string();
            if !body.is_empty() || !chapters.is_empty() {
                // 已有标题的块才保留空 body；纯前置块为空则丢弃
                chapters.push(Chapter { title: None, body });
            }
            chapters.push(Chapter {
                title: Some(line.trim().to_string()),
                body: String::new(),
            });
            current_body.clear();
        } else {
            current_body.push_str(line);
            current_body.push('\n');
        }
    }

    let body = current_body.trim().to_string();
    if !body.is_empty() {
        chapters.push(Chapter { title: None, body });
    }

    if !found_any {
        // 无章节标记：整体作为单块
        let body = text.trim().to_string();
        if body.is_empty() {
            return Vec::new();
        }
        return vec![Chapter { title: None, body }];
    }

    // 合并「标记后紧跟的 None 块」到前一个标记章节的 body
    let mut merged: Vec<Chapter> = Vec::new();
    for ch in chapters {
        if ch.title.is_none() && !ch.body.is_empty() {
            if let Some(last) = merged.last_mut() {
                if last.title.is_some() && last.body.is_empty() {
                    last.body = ch.body;
                    continue;
                }
            }
        }
        if ch.title.is_none() && ch.body.is_empty() {
            continue; // 空块丢弃
        }
        merged.push(ch);
    }

    if merged.is_empty() {
        let body = text.trim().to_string();
        if body.is_empty() {
            return Vec::new();
        }
        return vec![Chapter { title: None, body }];
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 章节行识别_中文数字() {
        assert!(is_chapter_line("第一章 大梦初醒"));
        assert!(is_chapter_line("第十二章"));
        assert!(is_chapter_line("第 3 回 宝玉初试云雨"));
        assert!(is_chapter_line("第三百卷 终"));
        // 强信号优先于句末标点排除：真实书名可含问号/叹号
        assert!(is_chapter_line("第0001章 女儿？"));
        assert!(is_chapter_line("第0027章 你是谁！"));
        // 强信号+编号白名单仍需防叙述句
        assert!(!is_chapter_line("第三天，他早早起床。"));
    }

    #[test]
    fn 章节行识别_英文() {
        assert!(is_chapter_line("Chapter 1 The Beginning"));
        assert!(is_chapter_line("chapter 12"));
    }

    #[test]
    fn 章节行识别_固定标记() {
        assert!(is_chapter_line("序章"));
        assert!(is_chapter_line("楔子"));
        assert!(is_chapter_line("尾声 一些补充"));
        assert!(is_chapter_line("番外 春日野餐记"));
    }

    #[test]
    fn 非章节行不误判() {
        assert!(!is_chapter_line("第三天，他早早起床。"));
        assert!(!is_chapter_line("他说第一章写得很烂。"));
        assert!(!is_chapter_line("第一章的内容让我印象深刻。"));
        assert!(!is_chapter_line("这是正文，提到第二章将在下周发布。"));
        assert!(!is_chapter_line(""));
        // 《奶爸大文豪》e2e 实测误报：余文过长 / 省略号结尾的叙述句
        assert!(!is_chapter_line(
            "第二部分连续更新了十二章，罗伟看得很爽，但是也一脸懵逼，这是怎搞的，难道五年级植物人电脑出问题了？"
        ));
        assert!(!is_chapter_line("第一节写诗人伫立在林间交叉路口……"));
        assert!(!is_chapter_line("第二章，审讯秘书。"));
    }

    #[test]
    fn 切分_标准小说结构() {
        let text = "版权信息，请勿转载。\n第一章 起点\n清晨，他醒了。\n天很蓝。\n第二章 转折\n午后，变天了。";
        let chapters = split_into_chapters(text);
        assert_eq!(chapters.len(), 3);
        assert_eq!(chapters[0].title, None);
        assert!(chapters[0].body.contains("版权信息"));
        assert_eq!(chapters[1].title.as_deref(), Some("第一章 起点"));
        assert!(chapters[1].body.contains("清晨"));
        assert_eq!(chapters[2].title.as_deref(), Some("第二章 转折"));
        assert!(chapters[2].body.contains("午后"));
    }

    #[test]
    fn 切分_无标记时单块() {
        let text = "这是一段没有章节标记的普通文本。";
        let chapters = split_into_chapters(text);
        assert_eq!(chapters.len(), 1);
        assert_eq!(chapters[0].title, None);
        assert_eq!(chapters[0].body, text);
    }

    #[test]
    fn 切分_空文本() {
        assert!(split_into_chapters("   \n  ").is_empty());
    }
}
