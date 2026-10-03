//! 多角色配音：对话/叙述切分 + 说话人识别 + 角色音色映射
//!
//! 竞品对标（Readify「多角色配音」）：把小说中的引号台词与旁白区分开，
//! 按说话人分配不同音色。纯规则实现（引号配对 + 动词启发式），
//! 覆盖网文 90% 的「XX说：『……』」场景，不依赖 LLM。

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// 角色 → 音色映射表（从 JSON 加载）
///
/// JSON 格式：
/// ```json
/// {
///   "narrator": "zf_xiaoxiao",
///   "dialogue_default": "zm_yunjian",
///   "roles": { "张三": "zm_yunjian", "李四": "zf_xiaobei" }
/// }
/// ```
/// - `narrator`：旁白音色
/// - `dialogue_default`：未识别说话人的台词音色
/// - `roles`：说话人名 → 音色 ID
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleVoiceMap {
    #[serde(default)]
    pub narrator: Option<String>,
    #[serde(default)]
    pub dialogue_default: Option<String>,
    #[serde(default)]
    pub roles: BTreeMap<String, String>,
}

impl RoleVoiceMap {
    pub fn from_json_str(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// 角色表是否为空（全空 = 多角色未启用）
    pub fn is_empty(&self) -> bool {
        self.narrator.is_none() && self.dialogue_default.is_none() && self.roles.is_empty()
    }
}

/// 对话/叙述片段
#[derive(Debug, Clone, PartialEq)]
pub struct DialoguePiece {
    /// 片段文本（台词含引号本身，保留引号韵律停顿）
    pub text: String,
    /// 是否为引号台词
    pub is_dialogue: bool,
    /// 识别出的说话人（仅台词片段，可能为 None）
    pub speaker: Option<String>,
}

/// 引号开启字符 → 关闭字符映射
fn closing_of(open: char) -> Option<char> {
    match open {
        '「' => Some('」'),
        '『' => Some('』'),
        '“' => Some('”'),
        _ => None,
    }
}

/// 说话人动词表（长词优先匹配）
const SPEECH_VERBS: &[&str] = &[
    "低声道", "沉声道", "冷声道", "淡淡道", "缓缓道", "大声道",
    "说道", "笑道", "怒道", "问道", "答道", "喊道", "叫道", "骂道",
    "嘀咕道", "吩咐道", "催促道", "反驳道", "解释道", "补充道",
    "回答道", "反问道", "低声说", "沉声说",
    "说", "道", "问", "答", "喊", "叫", "骂", "叹", "吼", "嚷", "念",
];

/// 把文本按引号切分为台词/叙述片段序列
///
/// 引号配对支持：「」『』“” 与直引号 `"`（直引号按开闭交替处理）。
/// 未闭合的引号到文本末尾都算台词（容错）。
pub fn split_dialogue(text: &str) -> Vec<DialoguePiece> {
    let mut pieces: Vec<DialoguePiece> = Vec::new();
    let mut narration = String::new();
    let mut dialogue = String::new();
    let mut open: Option<char> = None;
    let mut in_straight = false;

    let flush_narration = |pieces: &mut Vec<DialoguePiece>, narration: &mut String| {
        let t = narration.trim().to_string();
        if !t.is_empty() {
            pieces.push(DialoguePiece { text: t, is_dialogue: false, speaker: None });
        }
        narration.clear();
    };
    let flush_dialogue = |pieces: &mut Vec<DialoguePiece>, dialogue: &mut String| {
        let t = dialogue.trim().to_string();
        if !t.is_empty() {
            let speaker = detect_speaker_from_context(pieces.last().map(|p| p.text.as_str()));
            pieces.push(DialoguePiece { text: t, is_dialogue: true, speaker });
        }
        dialogue.clear();
    };

    for ch in text.chars() {
        if open.is_some() || in_straight {
            // 台词内部（弯引号配对 或 直引号交替）
            let open_ch = open.unwrap_or('"');
            dialogue.push(ch);
            if in_straight && ch == '"' {
                in_straight = false;
                flush_dialogue(&mut pieces, &mut dialogue);
            } else if open.is_some() && Some(ch) == closing_of(open_ch) {
                open = None;
                flush_dialogue(&mut pieces, &mut dialogue);
            }
        } else if ch == '"' {
            flush_narration(&mut pieces, &mut narration);
            dialogue.push(ch);
            in_straight = true;
        } else if closing_of(ch).is_some() {
            // 开启引号（「『“）
            flush_narration(&mut pieces, &mut narration);
            dialogue.push(ch);
            open = Some(ch);
        } else if matches!(ch, '」' | '』' | '”') {
            // 无配对的关闭引号：按普通文本处理
            narration.push(ch);
        } else {
            narration.push(ch);
        }
    }
    if dialogue_not_empty(&dialogue) {
        flush_dialogue(&mut pieces, &mut dialogue);
    }
    flush_narration(&mut pieces, &mut narration);

    pieces
}

fn dialogue_not_empty(d: &str) -> bool {
    !d.trim().is_empty()
}

/// 从台词前的叙述尾部识别说话人
///
/// 启发式：取引号前一行/前句的尾部文本，匹配说话人动词
/// （「XX说：」「XX喊道」），动词前的连续中英文数字串即人名。
pub fn detect_speaker_from_context(narration_tail: Option<&str>) -> Option<String> {
    let tail = narration_tail?;
    // 只取最后一个自然行，避免跨段误关联
    let line = tail.lines().last()?.trim_end();
    // 剥离尾部标点（「XX说：」「XX喊道，」等形态）
    let line = line.trim_end_matches(|c| {
        matches!(c, '，' | '。' | '：' | ':' | '、' | '！' | '？' | ' ' | '　' | '"' | '“' | '”' | '…')
    });
    if line.is_empty() {
        return None;
    }

    // 动词按长度降序尝试（最长匹配）
    let mut verbs: Vec<&str> = SPEECH_VERBS.to_vec();
    verbs.sort_by_key(|v| std::cmp::Reverse(v.chars().count()));
    for verb in verbs {
        if line.ends_with(verb) {
            let prefix = &line[..line.len() - verb.len()];
            let name = trailing_name(prefix);
            if let Some(n) = name {
                return Some(n);
            }
        }
    }
    None
}

/// 取字符串尾部的连续人名字符（CJK/字母/数字/·），最长 4 字（常见中文姓名上限）
fn trailing_name(s: &str) -> Option<String> {
    let chars: Vec<char> = s.chars().collect();
    let mut name: Vec<char> = Vec::new();
    for c in chars.iter().rev() {
        if c.is_alphanumeric() || *c == '·' || *c == '・' {
            name.push(*c);
            if name.len() >= 4 {
                break;
            }
        } else if !name.is_empty() {
            break;
        } else if matches!(c, '，' | '。' | '：' | ':' | '、' | ' ' | '　' | '"' | '“') {
            continue; // 跳过尾部标点
        } else {
            break;
        }
    }
    if name.is_empty() {
        return None;
    }
    name.reverse();
    let name: String = name.into_iter().collect();
    // 纯数字/单字符置信度过低，返回 None
    if name.chars().all(|c| c.is_ascii_digit()) || name.chars().count() < 2 {
        return None;
    }
    Some(name)
}

/// 解析片段应使用的音色 ID
///
/// 台词 + 说话人命中角色表 → 角色音色；
/// 台词未命中 → dialogue_default；
/// 叙述 → narrator。
pub fn resolve_voice(piece: &DialoguePiece, map: &RoleVoiceMap) -> Option<String> {
    if piece.is_dialogue {
        if let Some(speaker) = &piece.speaker {
            if let Some(voice) = map.roles.get(speaker) {
                return Some(voice.clone());
            }
        }
        map.dialogue_default.clone()
    } else {
        map.narrator.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 对话切分_基本() {
        let pieces = split_dialogue("他推门而入。「你来了。」她说。");
        assert_eq!(pieces.len(), 3);
        assert!(!pieces[0].is_dialogue);
        assert!(pieces[1].is_dialogue);
        assert_eq!(pieces[1].text, "「你来了。」");
        assert!(!pieces[2].is_dialogue);
    }

    #[test]
    fn 对话切分_弯引号() {
        let pieces = split_dialogue("“你好。”他回答，“请坐。”");
        let dialogues: Vec<_> = pieces.iter().filter(|p| p.is_dialogue).collect();
        assert_eq!(dialogues.len(), 2);
        assert_eq!(dialogues[0].text, "“你好。”");
    }

    #[test]
    fn 对话切分_直引号交替() {
        let pieces = split_dialogue("他说：\"快走。\"然后离开了。");
        let dialogues: Vec<_> = pieces.iter().filter(|p| p.is_dialogue).collect();
        assert_eq!(dialogues.len(), 1);
        assert_eq!(dialogues[0].text, "\"快走。\"");
    }

    #[test]
    fn 对话切分_未闭合容错() {
        let pieces = split_dialogue("叙述。「未闭合的台词");
        assert_eq!(pieces.len(), 2);
        assert!(pieces[1].is_dialogue);
    }

    #[test]
    fn 说话人识别_说道() {
        let name = detect_speaker_from_context(Some("王大妈喊道"));
        assert_eq!(name, Some("王大妈".to_string()));
    }

    #[test]
    fn 说话人识别_冒号() {
        let name = detect_speaker_from_context(Some("张三说"));
        assert_eq!(name, Some("张三".to_string()));
    }

    #[test]
    fn 说话人识别_无名() {
        assert_eq!(detect_speaker_from_context(Some("天色渐晚，远处传来钟声")), None);
    }

    #[test]
    fn 说话人识别_双字以下不识别() {
        assert_eq!(detect_speaker_from_context(Some("他说")), None);
    }

    #[test]
    fn 音色解析_命中角色() {
        let map = RoleVoiceMap::from_json_str(
            r#"{"narrator":"zf_xiaoxiao","dialogue_default":"zm_yunjian","roles":{"李四":"zf_xiaobei"}}"#,
        )
        .unwrap();
        let pieces = split_dialogue("李四说：「我不去。」");
        assert_eq!(pieces.len(), 2);
        assert_eq!(pieces[1].speaker.as_deref(), Some("李四"));
        assert_eq!(resolve_voice(&pieces[1], &map).as_deref(), Some("zf_xiaobei"));
        assert_eq!(resolve_voice(&pieces[0], &map).as_deref(), Some("zf_xiaoxiao"));
    }

    #[test]
    fn 音色解析_默认台词音色() {
        let map = RoleVoiceMap::from_json_str(
            r#"{"dialogue_default":"zm_yunjian","roles":{}}"#,
        )
        .unwrap();
        let pieces = split_dialogue("路人甲说：「借过。」");
        assert_eq!(resolve_voice(&pieces[1], &map).as_deref(), Some("zm_yunjian"));
    }

    #[test]
    fn 音色映射_空表() {
        let map = RoleVoiceMap::from_json_str("{}").unwrap();
        assert!(map.is_empty());
    }
}
