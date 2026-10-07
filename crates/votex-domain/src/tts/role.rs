//! 多角色配音：对话/叙述切分 + 说话人识别 + 角色音色映射
//!
//! 竞品对标（Readify「多角色配音」）：把小说中的引号台词与旁白区分开，
//! 按说话人分配不同音色。纯规则实现（引号配对 + 动词启发式），
//! 覆盖网文 90% 的「XX说：『……』」场景，不依赖 LLM。

use crate::tts::value_object::{VoiceGender, VoiceLocale, VoiceMeta};
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

/// 名字尾部的副词/助词噪声（「张重**又**说」→ 应识为「张重」）
///
/// 动词前的粘连副词会被 `trailing_name` 一并吞进人名，导致同一人被拆成
/// 「张重」/「张重又」两条候选，并使 `RoleVoiceMap` 的角色名匹配失效。
///
/// 只收**几乎不可能出现在人名末尾**的单字虚词。「了/着/过」等高度可能被
/// 用作人名，故不收——宁可漏剥离，也不要把真名字削没。
const NAME_TAIL_NOISE: &[char] = &['又', '也', '就', '才', '便', '即', '则', '还', '啊', '呀', '吧', '呢'];

/// 语气/神态/动作修饰语（「赶忙说」「张重点头」）——必须先于单字剥离，
/// 否则修饰语会因 4 字上限被整体当作人名
///
/// 动作词的词频来自真实书稿统计（1083 章）：
/// 点头 1301 / 摇头 628 / 摆手 229 / 皱眉 71；其中
/// `张重点头` 作为噪声候选出现 114 次、`点头道` 293 次。
const NAME_TAIL_NOISE_WORDS: &[&str] = &[
    "赶忙", "连忙", "急忙", "赶紧", "随即", "顿时", "忽然", "突然", "接着", "于是",
    "慢慢", "缓缓", "淡淡", "轻声", "低声", "沉声", "冷笑", "苦笑", "微笑着", "笑着",
    "哭着", "喊着", "叹着", "念着", "嘀咕着", "嚷着",
    "笑呵呵", "乐呵呵", "黑着脸", "皱着眉", "点点头", "摇摇头", "摆摆手",
    // 动作词：点头/摇头/摆手/抬头/转头/低头/侧头 等
    "点头", "摇头", "摆手", "抬手", "伸手", "握手", "耸肩", "皱眉", "挑眉", "愣住",
    "抬头", "转头", "低头", "侧头", "张嘴", "闭嘴", "眨眼", "瞪眼", "后退", "转身",
];

/// 语气/神态修饰的**截断标记**（「张重_**着**说道」的「着」）
///
/// 网文里说话人常被副词或动作修饰：`张重**笑着**说道`、`笑呵呵地**说道**、
/// `李经理**赶忙**开口`。这些修饰语紧贴动词之前，若不截断会被 4 字上限
/// 连人名一起吞掉（表现为 Top 候选里出现「张重笑着」这类噪声）。
///
/// 规则：遇到这些标记即**截断其后的全部内容**，只取标记之前的部分作为人名。
/// 之所以用「截断」而非「逐字剥离」，是因为修饰语长度不定（1~4 字），
/// 逐字剥离会在「笑着说」这类形态上残留「笑」。
const MODIFIER_TAIL_CUT: &[char] = &['着', '地', '的', '了'];

/// 截断后残留的动词字（「张重笑**着**」→「张重笑」须再去掉「笑」）
///
/// 截断点落在助词上时，其前一个字往往已是动词（`笑着说`→截「着」得「笑」）。
/// 这些单字动词几乎不可能是人名末字，故可安全剥离。
const VERB_TAIL_NOISE: &[char] = &[
    '说', '道', '问', '答', '喊', '叫', '骂', '叹', '吼', '嚷', '念', '笑', '哭', '想',
    '看', '听', '望', '瞥', '盯', '冲', '朝', '对', '向', '跟', '让', '把',
];

/// **整体即副词**的形态，命中则直接判定不是人名
///
/// 长度保护（≥2 字）挡不住「悄悄」「慢慢」「淡淡」这类双字副词——
/// 它们长度合法但显然不是人名。逐字剥离同样无解（剥完就没了），
/// 故对整串做黑名单匹配。
const ADVERB_ONLY_WORDS: &[&str] = &[
    "悄悄", "慢慢", "缓缓", "淡淡", "轻轻", "深深", "默默", "偷偷", "悄悄", "径自", "随即",
    "顿时", "忽然", "突然", "接着", "于是", "赶忙", "连忙", "急忙", "赶紧", "再次", "直接",
    "笑着", "哭着", "喊着", "叫着", "叹着", "念着", "想着", "问着", "看着", "听着",
    "笑呵呵", "乐呵呵", "一路上", "一边", "一面",
];

/// 取字符串尾部的连续人名字符（CJK/字母/数字/·），最长 4 字（常见中文姓名上限）
fn trailing_name(s: &str) -> Option<String> {
    // 第一步：在语气/神态修饰标记处截断，只保留其前的部分
    //（「张重笑着」→ 「张重笑」→ 第二步再退「笑」→ 「张重」；
    //   「笑呵呵地」→ 全截断 → None）
    let cut = s
        .char_indices()
        .find(|(_, c)| MODIFIER_TAIL_CUT.contains(c))
        .map(|(i, _)| &s[..i])
        .unwrap_or(s);
    // 第二步：剥离截断后残留的动词字（保证剥离后仍 ≥2 字）
    let cut = {
        let trimmed = cut.trim_end_matches(|c: char| VERB_TAIL_NOISE.contains(&c));
        if trimmed.chars().count() >= 2 {
            trimmed
        } else {
            cut
        }
    };
    // 第三步：剥掉尾部标点与双字副词（「李经理**赶忙**」→ 4 字上限内才装得下「李经理」）
    let mut s = cut.trim_end_matches(|c: char| {
        matches!(c, '，' | '。' | '：' | ':' | '、' | ' ' | '　' | '"' | '“' | '”' | '…')
    });
    // 整串即副词 → 直接拒绝（「悄悄」「慢慢」长度合法但不是人名）
    if ADVERB_ONLY_WORDS.contains(&s) {
        return None;
    }
    for word in NAME_TAIL_NOISE_WORDS {
        if s.ends_with(word) {
            s = &s[..s.len() - word.len()];
            break;
        }
    }

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
    // 剥离尾部单字副词（「张重又」→「张重」），保证剥离后仍≥2 字（避免削没正常名字）
    //
    // 注意：`name` 是从字符串尾部往前收集的**逆序**缓冲，
    // 因此真正的「尾部」是 `name[0]`，不是 `name.last()`。
    while name.len() > 2 && NAME_TAIL_NOISE.contains(&name[0]) {
        name.remove(0);
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

// ============================================================================
// 角色自动提取（docs/26 模块一）
//
// 背景：此前多角色配音需要用户读完全书、手工统计说话人、编写 role_map JSON。
// 本节提供**纯规则**的全书扫描，自动产出角色候选表供 GUI/CLI 呈现。
//
// 之所以坚持纯规则而非 LLM：
//   1. 成本——100 万字逐章送 LLM 费用不可接受，规则方案零调用零成本；
//   2. 离线——AGENTS.md 硬约束「用户文本内容不上传网络」；
//   3. 延迟——全书扫描是纯字符串操作，可做到秒级返回。
// 代价是覆盖不到无引号的间接引语，但那条路径本就走 dialogue_default 兜底。
// ============================================================================

/// 角色候选及其证据
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RoleCandidate {
    /// 角色名（说话人识别结果）
    pub name: String,
    /// 作为说话人被识别的次数（置信度代理指标）
    pub mentions: usize,
    /// 规则确认的性别；`Neutral` 表示**未识别出**，不等于「中性音色」
    ///
    /// 注意与 `suggested_gender` 的区别：`gender` 是有称谓证据的结论，
    /// `suggested_gender` 只是频次推断的猜测。
    pub gender: VoiceGender,
    /// 性别推测值（仅当`gender` 为 `Neutral` 时可能有值）
    ///
    /// 叠字名（「芃芃」）、单字名等无称谓线索的角色，规则无法确定性别。
    /// 按方案（docs/26§4.2）取频次 Top N无称谓角色给出推测，供 GUI 提示
    /// 用户确认。**不参与 `build_role_map` 的自动分配**——猜错性别会让
    /// 全书角色配音越听越违和，交由用户确认更安全。
    pub suggested_gender: Option<VoiceGender>,
    /// 首次出现的章节序号（从 0 起，便于用户定位核对）
    pub first_chapter: Option<usize>,
    /// 该角色的首段真实台词（供 GUI 试听，避免用通用样句听不出音色是否贴合角色）
    pub sample_line: Option<String>,
}

impl RoleCandidate {
    /// 该候选是否有称谓证据支撑的性别
    pub fn gender_confirmed(&self) -> bool {
        self.gender != VoiceGender::Neutral
    }
}

/// 称谓 → 性别（覆盖网文高频称谓）
///
/// 命中即定性别；女性称谓优先（网文中「妈妈」也可能是「妈妈」角色本身，
/// 而「女儿」几乎必然是女性角色）。
const FEMALE_TITLES: &[&str] = &[
    "女儿", "妈妈", "母亲", "娘", "奶奶", "姥姥", "外婆", "阿姨", "婶婶", "姑姑", "姨妈",
    "姐姐", "妹妹", "妹妹们", "姑娘", "小姐", "美女", "夫人", "太太", "老婆", "妻子", "女友",
    "女人", "女子", "少女", "女孩", "女王", "女皇", "公主", "皇后", "婆婆", "老板娘", "护士",
    "老师", "姐姐们", "女人们",
];

const MALE_TITLES: &[&str] = &[
    "儿子", "爸爸", "父亲", "爹", "爷爷", "姥爷", "外公", "叔叔", "伯伯", "舅舅", "姑父",
    "哥哥", "弟弟", "兄弟", "小伙", "先生", "少爷", "大叔", "老爷", "公公", "老板", "徒弟",
    "男人", "男子", "少年", "男孩", "国王", "皇帝", "太子", "和尚", "道士", "医生", "警察",
];

/// 由称谓推断性别（无称谓时返回 None）
fn gender_by_title(name: &str) -> Option<VoiceGender> {
    // 先查女性表再查男性表：「老板娘」含「老板」不应被判成男声
    if FEMALE_TITLES.iter().any(|t| name.contains(t)) {
        Some(VoiceGender::Female)
    } else if MALE_TITLES.iter().any(|t| name.contains(t)) {
        Some(VoiceGender::Male)
    } else {
        None
    }
}

/// 章节/编号类噪声模式：「第三章」「第3章」「十二回」等
///
/// 形态为「第 + 数字 + 可选结尾单位字」，与 `chapter::is_chapter_line` 保持同一套
/// 语义（复用其 `CHAPTER_UNITS`）。要求单位字若出现则必须是最后一个字符，
/// 因此「第三章风」这类人名不会被误杀。
fn looks_like_chapter_marker(name: &str) -> bool {
    let mut chars = name.chars();
    if chars.next() != Some('第') {
        return false;
    }
    // 「第」之后必须至少含一个数字，且其后不得出现数字/单位字之外的字符
    let mut has_digit = false;
    for c in chars {
        if c.is_ascii_digit() || "零一二三四五六七八九十百千万亿两〇○".contains(c) {
            has_digit = true;
        } else if crate::tts::chapter::CHAPTER_UNITS.contains(&c) {
            has_digit = true;
        } else {
            return false;
        }
    }
    has_digit
}

/// 噪声说话人过滤
///
/// `detect_speaker_from_context` 已排除纯数字与单字符，但仍有大量误命中：
/// - 纯标点/空白
/// - 含引号、括号等结构性字符（说明切分错位）
/// - 过长的「短语」（>8 字，通常是把叙述片段整个当成了人名）
/// - 章节编号（见 `looks_like_chapter_marker`）
fn is_valid_speaker_name(name: &str) -> bool {
    let n = name.chars().count();
    if n < 2 || n > 8 {
        return false;
    }
    if name.chars().any(|c| {
        matches!(c, '"' | '“' | '”' | '「' | '」' | '『' | '』' | '(' | ')' | '（' | '）' | '[' | ']')
    }) {
        return false;
    }
    if name
        .chars()
        .any(|c| c.is_whitespace() || c.is_ascii_punctuation())
    {
        return false;
    }
    if looks_like_chapter_marker(name) {
        return false;
    }
    // 全为数字（含中文数字）→ 大概率是「第3章」类误命中
    !name.chars().all(|c| c.is_ascii_digit() || "零一二三四五六七八九十百千万亿〇".contains(c))
}

/// 全书扫描，汇总角色候选（按出现频次降序）
///
/// 实现要点：
/// 1. **复用** `split_dialogue` + 其内部的说话人识别，逐段累计 `speaker` 频次；
/// 2. 按章节切片以记录 `first_chapter` 与该章内首段台词；
/// 3. 噪声过滤 + 称谓启发性别；
/// 4. 频次 Top `suggest_top_n` 内无称谓线索的角色给出 `suggested_gender`，
///    按「网文主角与配角多为男声、女主角常高频」的弱先验，
///    且**仅作GUI 提示，不参与自动分配**（见 `RoleCandidate::suggested_gender`）。
///
/// 复杂度：O(文本长度)，100 万字全本实测 < 2s（见 tests/e2e_novel_parse.rs）。
///
/// `top_n`：最多返回的候选数（网文常有数十个说话人，前 N 个即覆盖 95% 台词）。
pub fn extract_role_candidates(text: &str, top_n: usize) -> Vec<RoleCandidate> {
    extract_role_candidates_with(text, top_n, 3)
}

/// 同 `extract_role_candidates`，但可指定推测性别的前N 名数量
pub fn extract_role_candidates_with(
    text: &str,
    top_n: usize,
    suggest_top_n: usize,
) -> Vec<RoleCandidate> {
    // name → (频次, 称谓性别, 首章下标, 首段台词)
    let mut acc: BTreeMap<String, (usize, Option<VoiceGender>, Option<usize>, Option<String>)> =
        BTreeMap::new();
    let chapters = crate::tts::chapter::split_into_chapters(text);
    // 无章节结构时退化为单块扫描
    let chapters: Vec<crate::tts::chapter::Chapter> = if chapters.is_empty() {
        vec![crate::tts::chapter::Chapter { title: None, body: text.to_string() }]
    } else {
        chapters
    };

    for (chapter_idx, chapter) in chapters.iter().enumerate() {
        for piece in split_dialogue(&chapter.body) {
            let Some(speaker) = piece.speaker.as_deref() else {
                continue;
            };
            if !is_valid_speaker_name(speaker) {
                continue;
            }
            let entry = acc
                .entry(speaker.to_string())
                .or_insert((0, gender_by_title(speaker), Some(chapter_idx), None));
            entry.0 += 1;
            if entry.3.is_none() {
                // 去掉引号本体，只留台词文字供试听
                entry.3 = Some(piece.text.trim_matches(|c| c == '"' || c == '“' || c == '”').to_string());
            }
        }
    }

    let mut candidates: Vec<RoleCandidate> = acc
        .into_iter()
        .map(|(name, (mentions, gender, first_chapter, sample_line))| RoleCandidate {
            name,
            mentions,
            gender: gender.unwrap_or(VoiceGender::Neutral),
            suggested_gender: None,
            first_chapter,
            sample_line,
        })
        .collect();

    // 频次降序；同频次时有称谓推断性别的优先（性别更可信、用户更需确认）
    candidates.sort_by(|a, b| {
        b.mentions
            .cmp(&a.mentions)
            .then_with(|| gender_rank(b.gender).cmp(&gender_rank(a.gender)))
            .then_with(|| a.name.cmp(&b.name))
    });

    // 前 N 名中无称谓线索的角色给一个弱先验推测（仅提示，不参与分配）
    for c in candidates.iter_mut().take(suggest_top_n) {
        if !c.gender_confirmed() {
            c.suggested_gender = Some(VoiceGender::Male);
        }
    }

    candidates.truncate(top_n);
    candidates
}

/// 性别排序权重：女声 > 男声 > 童声 > 中性（中性排最后，优先让用户确认有性别的角色）
fn gender_rank(g: VoiceGender) -> u8 {
    match g {
        VoiceGender::Female => 0,
        VoiceGender::Male => 1,
        VoiceGender::Child => 2,
        VoiceGender::Neutral => 3,
    }
}

/// 性别未确认角色的轮转音色池（中文优先，男女交错）
///
/// # 为什么需要它
///
/// 真实书稿的说话人**几乎不带称谓线索**（1083 章 E2E：12 个主要角色中
/// 0 个能由称谓定性），若这类角色全落 `dialogue_default`，
/// 会得到「所有角色同一副嗓子」——多角色配音最不能接受的失败形态。
///
/// # 选取策略
///
/// - **中文优先**：英文音色用于中文小说本就违和（见 `pick_for_chinese`）；
/// - **男女交错**：`女,男,女,男…`。相邻角色分到不同性别，比连续 8 个角色
///   都是女声更容易分辨说话人；
/// - **跳过相邻编号**：Kokoro 音色按编号连续（zf_001/002/003…），
///   相邻编号音色极为相似，故同性别内每隔若干个取一个。
///
/// 某个性别的音色用尽时仅继续另一性别（不回头补齐），避免退化成单性别长串。
fn neutral_rotation_pool(voices: &[VoiceMeta]) -> Vec<String> {
    // 同性别内跳号步长：真实池 103 个音色（约 58 女 / 45 男），
    // 步长 8 可让池子足够长，同时避开相似音色
    const STEP: usize = 8;
    // 取不重复的等距样本：`0, STEP, 2*STEP, …`，越界后截断。
    // 不能用 `i*STEP % len`——那会绕回产生重复项，
    // 重复音色等于没分配（正是本函数要解决的问题）。
    fn spread(items: &[&VoiceMeta], step: usize) -> Vec<String> {
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < items.len() {
            out.push(items[i].voice_id.clone());
            i += step;
        }
        out
    }

    // 必须**同时**按性别与语种过滤：真实池里英文女声 `af_maple` 字典序排在
    // `zf_` 之前，只按 gender 过滤会给中文小说配上英文音色
    // （CLI 首次真实验证即出现 `张重 → af_maple`）。
    let chinese_of = |g: VoiceGender| {
        voices
            .iter()
            .filter(|v| v.gender == g && v.locale == VoiceLocale::Chinese)
            .collect::<Vec<_>>()
    };
    let females = spread(&chinese_of(VoiceGender::Female), STEP);
    let males = spread(&chinese_of(VoiceGender::Male), STEP);

    // 男女交错合并；某性别用尽后仅继续另一性别（不回头补齐）
    let mut pool: Vec<String> = Vec::with_capacity(females.len() + males.len());
    let (mut i, mut j) = (0usize, 0usize);
    loop {
        let before = pool.len();
        if i < females.len() {
            pool.push(females[i].clone());
            i += 1;
        }
        if j < males.len() {
            pool.push(males[j].clone());
            j += 1;
        }
        if pool.len() == before {
            break;
        }
    }
    pool
}

/// 角色音色的分配输入（避免用字符串键承载保留角色）
#[derive(Debug, Clone, Default)]
pub struct RoleAssignments {
    /// 角色名 → 音色 ID（用户显式指定，最高优先）
    pub roles: BTreeMap<String, String>,
    /// 旁白音色；缺省时取音色池中第一个女声
    pub narrator: Option<String>,
    /// 未匹配到音色的台词兜底音色；缺省时取第一个中性音色
    pub dialogue_default: Option<String>,
}

/// 按性别自动为角色分配音色
///
/// 三级策略（见 docs/26 §4.3）：
/// 1. `assignments.roles` 显式指定 → 最高优先；
/// 2. 按候选性别从 `voices` 池取第一个匹配音色；
/// 3. 落 `dialogue_default`。
///
/// `assignments.roles` 中出现的角色即使不在候选表内也会写入，
/// 允许用户手工补充规则未覆盖的角色。
pub fn build_role_map(
    candidates: &[RoleCandidate],
    voices: &[VoiceMeta],
    assignments: &RoleAssignments,
) -> RoleVoiceMap {
    let mut roles = BTreeMap::new();
    let mut leftovers: Vec<&RoleCandidate> = Vec::new();

    // 未确认性别角色的音色轮转游标（见下方「声线多样性」说明）
    let mut rotate_idx = 0usize;

    for c in candidates {
        let voice = match assignments.roles.get(&c.name) {
            Some(v) => Some(v.clone()),
            None if c.gender == VoiceGender::Neutral => {
                // 性别未确认：按角色出现顺序轮转不同音色，避免所有角色同一副嗓子。
                //
                // 背景（真实 1083 章 E2E）：本书 12 个主要角色中**有称谓证据的为 0**，
                // 若全部落兜底会得到「12 个角色共用 zm_009」——听感上角色之间
                // 完全无法区分，正是多角色配音最不能接受的失败形态。
                //
                // 这里只保证「听得出来不同」，不保证性别正确；
                // 正确性别须由用户试听确认后显式指定（见 RoleAssignments::roles）。
                let pool = neutral_rotation_pool(voices);
                if pool.is_empty() {
                    None
                } else {
                    let v = pool[rotate_idx % pool.len()].clone();
                    rotate_idx += 1;
                    Some(v)
                }
            }
            None => VoiceMeta::pick_for_chinese(voices, c.gender).map(|v| v.voice_id.clone()),
        };
        match voice {
            Some(v) => {
                roles.insert(c.name.clone(), v);
            }
            None => leftovers.push(c),
        }
    }

    // 兜底音色：中性优先，退而用男声。
    //
    // 真实 Kokoro 池114 个音色里**没有中性音色**（全为 zf_/zm_/af_/bf_），
    // 若严格要求 Neutral 会让 dialogue_default=None，
    // 结果是全部台词段无音色可路由（E2E 实测 29006 段丢失）。
    // 因此必须逐级回退，宁可用男声也不留空。
    let fallback = assignments.dialogue_default.clone().or_else(|| {
        // 兜底优先中文男声；中文池无中性音色（E2E 实测），
        // 严格要求 Neutral 会让 dialogue_default=None，
        // 结果是全部台词段无音色可路由（实测丢 29006 段）
        VoiceMeta::pick_any_for_chinese(voices, VoiceGender::Neutral)
            .map(|v| v.voice_id.clone())
            .or_else(|| voices.first().map(|v| v.voice_id.clone()))
    });
    for c in leftovers {
        if let Some(v) = &fallback {
            roles.insert(c.name.clone(), v.clone());
        }
    }
    // 补齐候选表外的显式指定角色
    for (name, voice) in &assignments.roles {
        roles.entry(name.clone()).or_insert_with(|| voice.clone());
    }

    RoleVoiceMap {
        narrator: assignments.narrator.clone().or_else(|| {
            // 旁白优先中文女声；池中无对应音色时逐级回退，仍不返回 None
            VoiceMeta::pick_any_for_chinese(voices, VoiceGender::Female)
                .map(|v| v.voice_id.clone())
                .or_else(|| voices.first().map(|v| v.voice_id.clone()))
        }),
        dialogue_default: fallback,
        roles,
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
    fn 说话人识别_神态修饰截断() {
        // 真实书稿高频形态：`X 笑着 说道`。动词先被剥掉，
        // 剩下前缀里的修饰语「笑着」须被截断，否则「张重笑着」连人名一起被吞
        assert_eq!(
            detect_speaker_from_context(Some("张重")),
            None,
            "无说话动词的行不该识别出人名"
        );
        assert_eq!(trailing_name("张重笑着").as_deref(), Some("张重"));
        assert_eq!(trailing_name("张重地说道").as_deref(), Some("张重"));
        assert_eq!(trailing_name("李经理赶忙").as_deref(), Some("李经理"));
        // 全是修饰语时不应产出人名
        assert_eq!(trailing_name("笑呵呵地"), None);
        assert_eq!(trailing_name("悄悄地"), None);
        assert_eq!(trailing_name("哭着"), None);
    }

    #[test]
    fn 角色提取_真实书稿噪声形态() {
        // 回归：这些形态曾在 1083 章真实全本 Top5 中占据 2~5 名
        let sample = "张重说道：“甲。”\n张重笑道：“乙。”\n张重说道：“丙。”\n爷爷张行军笑呵呵地说道：“丁。”";
        let cands = extract_role_candidates(sample, 10);
        let names: Vec<&str> = cands.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"张重"), "应提取出张重: {:?}", names);
        assert!(
            !names.iter().any(|n| n.contains("笑") || n.contains("地")),
            "神态修饰语不应进入人名: {:?}",
            names
        );
    }

    #[test]
    fn 说话人识别_剥离尾部副词() {
        // 「张重又说」应识为「张重」而非「张重又」——
        // 否则同一人被拆成两条候选，且 role_map 匹配失效
        assert_eq!(detect_speaker_from_context(Some("张重又说")), Some("张重".to_string()));
        assert_eq!(detect_speaker_from_context(Some("王大妈又笑道")), Some("王大妈".to_string()));
        assert_eq!(detect_speaker_from_context(Some("李经理赶忙说")), Some("李经理".to_string()));
        // 剥离后不足 2 字则放弃，不返回残缺名字
        assert_eq!(detect_speaker_from_context(Some("张三说")), Some("张三".to_string()));
        // 人名本身以「又/也」结尾属罕见但存在，不应误削成 2 字以外的形态
        assert_eq!(detect_speaker_from_context(Some("张又")), None, "单字名本就不该识别");
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

    // ------------------------------------------------------------------
    // 角色自动提取（docs/26 模块一）
    // ------------------------------------------------------------------

    /// 构造测试音色池
    fn test_voices() -> Vec<VoiceMeta> {
        use crate::model::value_object::EngineKind;
        vec![
            VoiceMeta::new("zf_xiaoxiao", "晓晓-女声", EngineKind::Kokoro, VoiceGender::Female),
            VoiceMeta::new("zm_yunjian", "云健-男声", EngineKind::Kokoro, VoiceGender::Male),
            VoiceMeta::new("xf_child", "童声", EngineKind::Kokoro, VoiceGender::Child),
            VoiceMeta::new("default", "默认", EngineKind::Kokoro, VoiceGender::Neutral),
        ]
    }

    /// 构造接近真实规模的音色池（真实 Kokoro 池 103 个：58女/ 45 男）
    ///
    /// 轮转分配需要足够大的池子才能体现「男女交错+ 跳号」；
    /// 4 个音色的小夹具会让所有角色落到同一音色，测不出真实行为。
    fn realistic_voices() -> Vec<VoiceMeta> {
        use crate::model::value_object::EngineKind;
        let mut v = Vec::new();
        for i in 0..58 {
            v.push(VoiceMeta::new(
                &format!("zf_{:03}", i),
                "中文女声",
                EngineKind::Kokoro,
                VoiceGender::Female,
            ));
        }
        for i in 0..45 {
            v.push(VoiceMeta::new(
                &format!("zm_{:03}", i),
                "中文男声",
                EngineKind::Kokoro,
                VoiceGender::Male,
            ));
        }
        v
    }

    #[test]
    fn 角色提取_统计频次与降序() {
        let text = "张重说道：“爸爸。”\n芃芃说：“爸爸。”\n张重又说：“嗯。”\n芃芃喊道：“好。”";
        let cands = extract_role_candidates(text, 10);
        assert_eq!(cands.len(), 2, "应提取出张重与芃芃两个角色: {:?}", cands);
        // 张重 2 次（「张重说」+「张重又说」已按副词剥离合并）= 芃芃 2 次，
        // 同频次按 Unicode 码位升序：「张」(U+5F20) < 「芃」(U+82DF)，故张重在前
        assert_eq!(cands[0].name, "张重");
        assert_eq!(cands[1].name, "芃芃");
        assert!(cands.iter().all(|c| c.mentions == 2), "两人各 2 次");
    }

    #[test]
    fn 角色提取_称谓启发性别() {
        let text = "女人说道：“来了。”\n爸爸喊道：“嗯。”\n张重说：“走。”";
        let cands = extract_role_candidates(text, 10);
        let g = |n: &str| cands.iter().find(|c| c.name == n).map(|c| c.gender);
        assert_eq!(g("女人"), Some(VoiceGender::Female));
        assert_eq!(g("爸爸"), Some(VoiceGender::Male));
        // 无称谓 → 中性别名，交由用户确认
        assert_eq!(g("张重"), Some(VoiceGender::Neutral));
    }

    #[test]
    fn 角色提取_女声表优先于男声表() {
        // 「老板娘」含「老板」，若男声表先查会被误判
        assert_eq!(gender_by_title("老板娘"), Some(VoiceGender::Female));
        assert_eq!(gender_by_title("老板"), Some(VoiceGender::Male));
    }

    #[test]
    fn 角色提取_噪声过滤() {
        assert!(!is_valid_speaker_name("1"));
        assert!(!is_valid_speaker_name("123"));
        assert!(!is_valid_speaker_name("第三章"), "中文数字章节号须过滤");
        assert!(!is_valid_speaker_name("第3章"), "混排章节号须过滤");
        assert!(!is_valid_speaker_name("第十二回"), "须过滤");
        assert!(!is_valid_speaker_name("这\"张"));
        assert!(!is_valid_speaker_name("很长很长很长的人名超过八字"));
        assert!(!is_valid_speaker_name("a b"));
        assert!(is_valid_speaker_name("张重"));
        assert!(is_valid_speaker_name("王大妈"));
        assert!(is_valid_speaker_name("第三章风"), "非纯编号形态不误杀");
    }

    #[test]
    fn 角色提取_首段台词供试听() {
        let text = "芃芃说道：“爸爸，我饿了。”";
        let cands = extract_role_candidates(text, 5);
        assert_eq!(cands[0].name, "芃芃");
        let sample = cands[0].sample_line.as_deref().unwrap();
        assert!(sample.contains("爸爸，我饿了"), "首段台词应可试听: {:?}", sample);
        assert!(!sample.contains('“'), "试听文本应去掉引号本体");
    }

    #[test]
    fn 角色提取_章节定位() {
        let text = "第1章 开始\n\n张重说道：“甲。”\n\n第2章 后续\n\n芃芃说道：“乙。”";
        let cands = extract_role_candidates(text, 10);
        let zhang = cands.iter().find(|c| c.name == "张重").unwrap();
        let peng = cands.iter().find(|c| c.name == "芃芃").unwrap();
        assert_eq!(zhang.first_chapter, Some(0));
        assert_eq!(peng.first_chapter, Some(1));
    }

    #[test]
    fn 角色提取_topN截断() {
        let mut text = String::new();
        for i in 0..10 {
            let name = format!("角色{}", i);
            for _ in 0..(10 - i) {
                text.push_str(&format!("{}说：“话。”\n", name));
            }
        }
        let cands = extract_role_candidates(&text, 3);
        assert_eq!(cands.len(), 3, "应按top_n 截断");
        assert_eq!(cands[0].mentions, 10, "最高频角色应在首位");
    }

    #[test]
    fn 音色自动分配_按性别分池() {
        let text = "女人说道：“甲。”\n爸爸说道：“乙。”\n女儿说道：“丙。”";
        let cands = extract_role_candidates(text, 10);
        let map = build_role_map(&cands, &test_voices(), &RoleAssignments::default());
        assert_eq!(map.roles.get("女人").map(String::as_str), Some("zf_xiaoxiao"));
        assert_eq!(map.roles.get("爸爸").map(String::as_str), Some("zm_yunjian"));
        assert_eq!(map.narrator.as_deref(), Some("zf_xiaoxiao"), "旁白缺省取女声");
        assert_eq!(map.dialogue_default.as_deref(), Some("default"));
    }

    #[test]
    fn 角色提取_叠字名无法确认性别只给推测() {
        // 「芃芃」是叠字名，规则拿不到称谓证据：gender 必须是 Neutral，
        // 推测值只落在 suggested_gender 上，且不参与自动分配
        let text = "芃芃说道：“爸爸。”";
        let cands = extract_role_candidates(text, 10);
        let c = &cands[0];
        assert_eq!(c.gender, VoiceGender::Neutral);
        assert!(!c.gender_confirmed());
        assert_eq!(c.suggested_gender, Some(VoiceGender::Male));
        // 自动分配时走轮转池，绝不臆测成童声
        let map = build_role_map(&cands, &realistic_voices(), &RoleAssignments::default());
        assert_ne!(
            map.roles.get("芃芃").map(String::as_str),
            Some("xf_child"),
            "性别未确认时不应误用童声"
        );
    }

    #[test]
    fn 音色自动分配_无中性音色池时逐级回退() {
        // 真实 Kokoro 池无中性音色：无称谓角色应走轮转池，
        // 而旁白/兜底这两个「必须有音色」的位置仍需逐级回退
        use crate::model::value_object::EngineKind;
        let voices = vec![
            VoiceMeta::new("zf_001", "女声", EngineKind::Kokoro, VoiceGender::Female),
            VoiceMeta::new("zm_009", "男声", EngineKind::Kokoro, VoiceGender::Male),
        ];
        let cands = extract_role_candidates("张重说道：“甲。”", 10);
        let map = build_role_map(&cands, &voices, &RoleAssignments::default());
        // 旁白必须有音色（女声池有则用女声）
        assert_eq!(map.narrator.as_deref(), Some("zf_001"));
        // 台词兜底必须有音色：中性池空→ 退到男声，不能为 None
        assert!(
            map.dialogue_default.is_some(),
            "台词兜底不得为 None（池无中性音色时应回退）"
        );
        // 角色至少分到一个真实存在的音色
        let v = map.roles.get("张重").expect("角色应分到音色");
        assert!(v == "zf_001" || v == "zm_009", "实际 {}", v);
    }

    #[test]
    fn 音色自动分配_空音色池不崩溃() {
        let cands = extract_role_candidates("张重说道：“甲。”", 10);
        let map = build_role_map(&cands, &[], &RoleAssignments::default());
        // 无池时不应 panic；角色仍写入（值为 None 时 resolve_voice回落）
        assert!(map.narrator.is_none());
        assert!(map.dialogue_default.is_none());
    }

    #[test]
    fn 角色提取_有称谓者不给推测值() {
        let text = "爸爸说道：“甲。”";
        let cands = extract_role_candidates(text, 10);
        assert!(cands[0].gender_confirmed());
        assert_eq!(cands[0].suggested_gender, None);
    }

    #[test]
    fn 音色自动分配_童声可由显式指定() {
        let text = "芃芃说道：“甲。”";
        let cands = extract_role_candidates(text, 10);
        let mut roles = BTreeMap::new();
        roles.insert("芃芃".to_string(), "xf_child".to_string());
        let assign = RoleAssignments { roles, ..Default::default() };
        let map = build_role_map(&cands, &test_voices(), &assign);
        assert_eq!(map.roles.get("芃芃").map(String::as_str), Some("xf_child"));
    }

    #[test]
    fn 音色自动分配_显式指定最高优先() {
        let text = "爸爸说道：“甲。”";
        let cands = extract_role_candidates(text, 10);
        let mut roles = BTreeMap::new();
        roles.insert("爸爸".to_string(), "zf_xiaoxiao".to_string());
        let assign = RoleAssignments {
            roles,
            narrator: Some("zm_yunjian".to_string()),
            dialogue_default: Some("default".to_string()),
        };
        let map = build_role_map(&cands, &test_voices(), &assign);
        assert_eq!(map.roles.get("爸爸").map(String::as_str), Some("zf_xiaoxiao"));
        assert_eq!(map.narrator.as_deref(), Some("zm_yunjian"));
    }

    #[test]
    fn 音色自动分配_中性角色落兜底不猜性别() {
        // 「张重」无称谓 → 中性别名，不应被猜成男声或童声。
        // 注意：会走轮转池（见下一个测试），此处验证「不臆测性别」——
        // 分到的音色不得是童声
        let text = "张重说道：“甲。”";
        let cands = extract_role_candidates(text, 10);
        let map = build_role_map(&cands, &test_voices(), &RoleAssignments::default());
        let v = map.roles.get("张重").unwrap();
        assert_ne!(
            v, "xf_child",
            "性别未确认时不应误用童声（真实池亦无童声可用）"
        );
    }

    #[test]
    fn 音色自动分配_中性角色轮转不同音色() {
        // 回归：真实书稿中称谓线索几乎为零（1083 章 E2E 实测 0/12），
        // 若全部落兜底会得到「所有角色同一副嗓子」
        let mut text = String::new();
        for name in ["张重", "许雨涵", "芃芃", "胡慧芳", "庄语", "陈青"] {
            text.push_str(&format!("{}说道：“台词。”\n", name));
        }
        let cands = extract_role_candidates(&text, 10);
        // 前置条件：本例 6 个角色都无称谓
        assert!(cands.iter().all(|c| !c.gender_confirmed()));

        let map = build_role_map(&cands, &realistic_voices(), &RoleAssignments::default());
        let assigned: Vec<&String> = cands.iter().filter_map(|c| map.roles.get(&c.name)).collect();

        // 6 个角色应得到 6 个不同音色，否则角色之间听感无差别
        let mut uniq: Vec<&String> = assigned.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(
            uniq.len(),
            cands.len(),
            "中性角色应各自获得不同音色，实际 {:?}",
            uniq
        );
    }

    #[test]
    fn 轮转池_男女交错且跳过相邻编号() {
        // 用真实规模的池：58 中文女声 + 45 中文男声 + 英文音色（字典序最前）
        use crate::model::value_object::EngineKind;
        let mut voices = realistic_voices();
        voices.insert(0, VoiceMeta::new("af_maple", "英文女声", EngineKind::Kokoro, VoiceGender::Female));
        voices.insert(1, VoiceMeta::new("am_adam", "英文男声", EngineKind::Kokoro, VoiceGender::Male));
        let pool = neutral_rotation_pool(&voices);

        // 全部为**中文**音色（回归：曾出现 `张重 → af_maple` 英文女声）
        assert!(
            pool.iter().all(|v| v.starts_with("zf_") || v.starts_with("zm_")),
            "轮转池不得含英文音色: {:?}",
            &pool[..4.min(pool.len())]
        );
        // 男女交错
        assert!(
            pool[0].starts_with("zf_") && pool[1].starts_with("zm_"),
            "应女声起始并男女交错: {:?}",
            &pool[..4.min(pool.len())]
        );
        // 非相邻编号：相邻两个同性别条目的编号差不为 1，且不得重复
        let females: Vec<usize> = pool
            .iter()
            .filter(|v| v.starts_with("zf_"))
            .map(|v| v[3..].parse().unwrap())
            .collect();
        for w in females.windows(2) {
            assert_ne!(w[0], w[1], "同性别条目不应重复: {:?}", w);
            assert!(w[1].saturating_sub(w[0]) > 1, "同性别相邻条目应跳号: {:?}", w);
        }
    }

    #[test]
    fn 角色提取_动作词修饰剥离() {
        // 回归：真实书稿「张重点头没再问」曾产生噪声候选「张重点头」(114 次)
        assert_eq!(trailing_name("张重点头").as_deref(), Some("张重"));
        assert_eq!(trailing_name("张重摇头").as_deref(), Some("张重"));
        assert_eq!(trailing_name("李经理摆手").as_deref(), Some("李经理"));
        assert_eq!(trailing_name("张重点头").as_deref(), Some("张重"));

        let sample = "张重点头没再问下去。\n张重说道：“甲。”\n张重摇头道：“乙。”";
        let cands = extract_role_candidates(sample, 10);
        let names: Vec<&str> = cands.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["张重"], "动作词不应产生额外角色: {:?}", names);
    }

    #[test]
    fn 音色自动分配_候选表外的角色也被写入() {
        let mut roles = BTreeMap::new();
        roles.insert("路人".to_string(), "zm_yunjian".to_string());
        let assign = RoleAssignments { roles, ..Default::default() };
        let map = build_role_map(&[], &test_voices(), &assign);
        assert_eq!(map.roles.get("路人").map(String::as_str), Some("zm_yunjian"));
    }

    #[test]
    fn 角色候选_可序列化往返() {
        let text = "张重说道：“甲。”";
        let cands = extract_role_candidates(text, 5);
        let json = serde_json::to_string(&cands[0]).unwrap();
        let back: RoleCandidate = serde_json::from_str(&json).unwrap();
        assert_eq!(back, cands[0]);
    }
}
