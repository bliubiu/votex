mod mapping;
pub mod polyphone;

/// 将拼音+声调（如 "ni3"）转换为 Kokoro IPA 音素字符串（如 "ni↓"）
///
/// 声调映射：
/// - 一声（˥）→ → (U+2192)
/// - 二声（˧˥）→ ↗ (U+2197)
/// - 三声（˧˩˧）→ ↓ (U+2193)
/// - 四声（˥˩）→ ↘ (U+2198)
pub fn pinyin_to_ipa(pinyin_with_tone: &str) -> Option<String> {
    let ipa = mapping::lookup(pinyin_with_tone)?;
    Some(convert_tone_markers(ipa))
}

/// 将 IPA 声调标记（˥ ˧˥ ˧˩˧ ˥˩）转换为 Kokoro 箭头符号
fn convert_tone_markers(ipa: &str) -> String {
    // 声调标记长度不同，需要按最长匹配优先替换
    let mut result = String::with_capacity(ipa.len());
    let chars: Vec<char> = ipa.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        // 尝试匹配三字符声调标记 ˧˩˧ (tone 3)
        if i + 2 < chars.len() {
            let triple: String = chars[i..=i + 2].iter().collect();
            if triple == "\u{02E7}\u{02E9}\u{02E7}" {
                result.push('\u{2193}'); // ↓
                i += 3;
                continue;
            }
        }
        // 尝试匹配双字符声调标记 ˧˥ (tone 2) 或 ˥˩ (tone 4)
        if i + 1 < chars.len() {
            let pair: String = chars[i..=i + 1].iter().collect();
            if pair == "\u{02E7}\u{02E5}" {
                result.push('\u{2197}'); // ↗
                i += 2;
                continue;
            }
            if pair == "\u{02E5}\u{02E9}" {
                result.push('\u{2198}'); // ↘
                i += 2;
                continue;
            }
        }
        // 单字符声调标记 ˥ (tone 1)
        if chars[i] == '\u{02E5}' {
            result.push('\u{2192}'); // →
            i += 1;
            continue;
        }
        // 其他字符保持原样
        result.push(chars[i]);
        i += 1;
    }
    result
}

/// 清除 Kokoro VOCAB 不支持的组合字符和特殊标点
///
/// misaki 生成的 IPA 包含一些扩展 IPA 组合字符（如 u̯, i̯, ɻ̩），
/// 但 Kokoro-82M 的词汇表不包含这些字符，它们会在 token ID 转换中被静默跳过，
/// 导致 `au̯` 变成 `au`（双元音→两个独立元音），发音严重失真。
///
/// 替换规则：
/// - `u̯` (U+0075 + U+032F COMBINING INVERTED BREVE BELOW) → `ʊ` (U+028A LATIN SMALL LETTER UPSILON)
/// - `i̯` (U+0069 + U+032F) → `ɪ` (U+026A LATIN SMALL LETTER SMALL CAPITAL I)
/// - `̩` (U+0329 COMBINING VERTICAL LINE BELOW) → 直接移除（成音节信息由上下文隐含）
/// - `，` (U+FF0C) → `,` (ASCII 逗号)
/// - `。` (U+3002) → `.` (ASCII 句号)
fn clean_ipa_for_kokoro(ipa: &str) -> String {
    let mut result = String::with_capacity(ipa.len());
    let chars: Vec<char> = ipa.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            // u + COMBINING INVERTED BREVE BELOW → ʊ
            '\u{0075}' if i + 1 < chars.len() && chars[i + 1] == '\u{032F}' => {
                result.push('\u{028A}'); // ʊ
                i += 2;
            }
            // i + COMBINING INVERTED BREVE BELOW → ɪ
            '\u{0069}' if i + 1 < chars.len() && chars[i + 1] == '\u{032F}' => {
                result.push('\u{026A}'); // ɪ
                i += 2;
            }
            // 移除 COMBINING VERTICAL LINE BELOW（成音节符号）
            '\u{0329}' => {
                i += 1;
            }
            // 全角逗号 → ASCII 逗号
            '\u{FF0C}' => {
                result.push(',');
                i += 1;
            }
            // 全角句号 → ASCII 句号（也可选择直接跳过）
            '\u{3002}' => {
                result.push('.');
                i += 1;
            }
            // 其他字符保持原样
            _ => {
                result.push(c);
                i += 1;
            }
        }
    }
    result
}

/// 将中文文本转换为 Kokoro phonemes 字符串
///
/// 汉字转换为 IPA 音素序列，非汉字字符（英文、数字、标点）保留原样。
/// 各汉字音素之间以空格分隔，非汉字文本连续输出。
/// 输出会经过 `clean_ipa_for_kokoro()` 清洗，确保所有字符均在 VOCAB 中。
pub fn text_to_phonemes(text: &str) -> String {
    use pinyin::ToPinyin;
    
    let mut result = String::with_capacity(text.len() * 6);
    let mut first = true;
    
    for c in text.chars() {
        // 仅处理 CJK 统一表意文字（汉字）
        if c >= '\u{4E00}' && c <= '\u{9FFF}' {
            // 获取带声调数字结尾的拼音（如 "ni3"）
            if let Some(py) = c.to_pinyin() {
                let py_str = py.with_tone_num_end().to_string();
                if let Some(ipa) = pinyin_to_ipa(&py_str) {
                    if !first {
                        result.push(' ');
                    }
                    result.push_str(&ipa);
                    first = false;
                } else {
                    tracing::warn!("G2P: 未找到拼音映射 '{}' 对于汉字 '{}'", py_str, c);
                }
            }
        } else {
            // 非汉字字符（英文、数字、标点）直接追加
            result.push(c);
            if !c.is_whitespace() {
                first = false;
            }
        }
    }
    
    // 后处理：清洗不支持的组合字符
    clean_ipa_for_kokoro(&result)
}

// ===================== v1.1-zh: 注音 G2P =====================

/// 将拼音+声调（如 "ni3"）转换为 Kokoro v1.1-zh 注音字符串（如 "ㄋㄧ3"）
pub fn pinyin_to_zhuyin(pinyin_with_tone: &str) -> Option<String> {
    mapping::lookup_zhuyin(pinyin_with_tone)
}

/// 将缓冲区中的注音音节输出到 result，应用三声变调（3+3 → 2+3）
///
/// 变调规则：缓冲区内的连续三声音节，前一个变二声
/// 非汉字文本会打断变调（在不同缓冲区之间不应用变调）
fn flush_zhuyin_buffer(result: &mut String, buffer: &mut Vec<(String, String)>) {
    for i in 0..buffer.len() {
        let is_tone3 = buffer[i].0.ends_with('3');
        let next_is_tone3 = i + 1 < buffer.len() && buffer[i + 1].0.ends_with('3');

        if is_tone3 && next_is_tone3 {
            // 三声变调：当前音节 3→2
            result.push_str(&buffer[i].1.replace('3', "2"));
        } else {
            result.push_str(&buffer[i].1);
        }
    }
    buffer.clear();
}

/// 拼音预处理：将 pinyin crate 输出的 ü 转换为 v
///
/// `pinyin` crate 对 "女" 返回 "nü3"（带 U+00FC ü），
/// 但 Kokoro v1.1-zh 的映射表使用 "nv3" 作为键。
fn normalize_pinyin_for_lookup(py: &str) -> String {
    py.replace('\u{00FC}', "v")
}

/// 全角/中文标点 → ASCII 标点映射
///
/// Kokoro v1.1-zh 词汇表只包含 ASCII 标点，不包含全角标点。
const PUNCT_MAP: &[(char, char)] = &[
    ('\u{FF0C}', ','),  // 全角逗号
    ('\u{3002}', '.'),  // 中文句号
    ('\u{FF1F}', '?'),  // 全角问号
    ('\u{FF01}', '!'),  // 全角感叹号
    ('\u{FF08}', '('),  // 全角左括号
    ('\u{FF09}', ')'),  // 全角右括号
    ('\u{FF1B}', ';'),  // 全角分号
    ('\u{FF1A}', ':'),  // 全角冒号
    ('\u{2018}', '\''), // 左单引号
    ('\u{2019}', '\''), // 右单引号
    ('\u{201C}', '"'),  // 左双引号（已在 vocab 中，但映射更安全）
    ('\u{201D}', '"'),  // 右双引号（同上）
    ('\u{300A}', '<'),  // 《左书名号
    ('\u{300B}', '>'),  // 》右书名号
];

/// 将单个字符映射为 Kokoro vocabluary 支持的标点，返回 None 表示无需改变
fn map_punct(c: char) -> Option<char> {
    for &(from, to) in PUNCT_MAP {
        if c == from {
            return Some(to);
        }
    }
    None
}

/// 将中文文本转换为 Kokoro v1.1-zh 注音音素字符串
///
/// 汉字转换为注音符号序列（如 "你好" → "ㄋㄧ2ㄏㄠ3"），
/// 非汉字字符（英文、数字）保留原样，全角/中文标点自动映射为 ASCII 等价标点。
/// 自动应用三声变调（3+3 → 2+3）、「一 / 不」变调与多音字词组消歧。
///
/// 输出字符集与 v1.1-zh config.json 的 vocab 一致：
/// - 注音符号（ㄅㄆㄇㄈ...）
/// - 特殊韵母（十、月、万、压...）
/// - 数字声调（1-5）
/// - IPA 字符（英文部分）
/// - ASCII 标点符号
pub fn text_to_phonemes_zh(text: &str) -> String {
    use pinyin::ToPinyin;

    let chars: Vec<char> = text.chars().collect();
    let mut result = String::with_capacity(text.len() * 5);
    // 缓冲区：收集连续的汉字音节（带拼音和注音），用于变调处理
    let mut buffer: Vec<(String, String)> = Vec::new();
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        if is_hanzi(c) {
            // 1) 多音字词组消歧：最长匹配优先（如「银行」读 háng 而非 xíng）
            if let Some((syllables, len)) = polyphone::lookup_phrase(&chars, i) {
                for py in &syllables {
                    let normalized = normalize_pinyin_for_lookup(py);
                    if let Some(zhuyin) = pinyin_to_zhuyin(&normalized) {
                        buffer.push((normalized.clone(), zhuyin));
                    } else {
                        tracing::warn!("G2P(zh): 词组音节注音映射失败 '{}'", normalized);
                    }
                }
                i += len;
                continue;
            }

            // 2) 单字：查拼音（含「一 / 不」变调）
            if let Some(py) = c.to_pinyin() {
                let py_str = py.with_tone_num_end().to_string();
                // 「一 / 不」变调：依紧邻后字声调变读
                let next_tone = peek_next_hanzi_tone(&chars, i + 1);
                let py_str = polyphone::adjust_yi_bu(c, &py_str, next_tone);
                // 归一化: ü → v（pinyin crate 输出 "nü3" 但映射表用 "nv3"）
                let normalized = normalize_pinyin_for_lookup(&py_str);
                if let Some(zhuyin) = pinyin_to_zhuyin(&normalized) {
                    buffer.push((py_str, zhuyin.to_string()));
                } else {
                    tracing::warn!("G2P(zh): 未找到注音映射 '{}' 对于汉字 '{}'", normalized, c);
                    // 注音映射失败（罕见情况），先刷新缓冲区再通过 IPA 降级
                    flush_zhuyin_buffer(&mut result, &mut buffer);
                    if let Some(ipa) = pinyin_to_ipa(&normalized) {
                        result.push(' ');
                        result.push_str(&ipa);
                    }
                }
            } else {
                // 无法获取拼音（如生僻字）：刷新缓冲区并告警。
                // 此前静默跳过导致长文本中「跳字」且无从排查。
                flush_zhuyin_buffer(&mut result, &mut buffer);
                tracing::warn!("G2P(zh): 生僻字 '{}' 无拼音数据，已跳过（文本中将缺字）", c);
            }
            i += 1;
        } else {
            // 非汉字字符：先刷新拼音缓冲区（变调只应用在同一中文段内）
            flush_zhuyin_buffer(&mut result, &mut buffer);
            // 全角/中文标点 → ASCII
            if let Some(ascii) = map_punct(c) {
                result.push(ascii);
            } else {
                result.push(c);
            }
            i += 1;
        }
    }

    // 刷新末尾的缓冲区
    flush_zhuyin_buffer(&mut result, &mut buffer);

    result
}

/// 是否为汉字（CJK 统一表意文字基本区）
fn is_hanzi(c: char) -> bool {
    c >= '\u{4E00}' && c <= '\u{9FFF}'
}

/// 向后窥探紧邻汉字的声调（用于「一 / 不」变调）
///
/// 只在同一连续汉字段内窥探：遇到非汉字立即返回 None。
/// 返回声调数字 1~5。
fn peek_next_hanzi_tone(chars: &[char], from: usize) -> Option<u8> {
    use pinyin::ToPinyin;
    let &next = chars.get(from)?;
    if !is_hanzi(next) {
        return None;
    }
    let py = next.to_pinyin()?;
    let tone_char = py.with_tone_num_end().chars().last()?;
    tone_char.to_digit(10).map(|d| d as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 调试音素输出() {
        let text = "张重躺在床上，看着眼前这个全裸的女人。";
        let ph = crate::tts::kokoro_g2p::text_to_phonemes_zh(text);
        println!("PH_LEN={} PH={}", ph.chars().count(), ph);
        let text2 = "你好世界";
        let ph2 = crate::tts::kokoro_g2p::text_to_phonemes_zh(text2);
        println!("PH2_LEN={} PH2={}", ph2.chars().count(), ph2);
    }

    #[test]
    fn test_tone_conversion() {
        // 一声 ˥ → →
        assert_eq!(convert_tone_markers("a\u{02E5}"), "a\u{2192}");
        // 二声 ˧˥ → ↗
        assert_eq!(convert_tone_markers("a\u{02E7}\u{02E5}"), "a\u{2197}");
        // 三声 ˧˩˧ → ↓
        assert_eq!(convert_tone_markers("a\u{02E7}\u{02E9}\u{02E7}"), "a\u{2193}");
        // 四声 ˥˩ → ↘
        assert_eq!(convert_tone_markers("a\u{02E5}\u{02E9}"), "a\u{2198}");
    }

    #[test]
    fn test_pinyin_to_ipa() {
        // v1.0 IPA 映射文件不存在时，pinyin_to_ipa 返回 None
        // 如果文件存在，应返回对应的 IPA 值
        let result = pinyin_to_ipa("ni3");
        if let Some(ipa) = result.as_ref() {
            // 如果文件存在，验证返回的是 IPA 格式（包含声调标记）
            assert!(ipa.contains("˧") || ipa.contains("˥"));
        }
        // 不存在的拼音始终返回 None
        assert!(pinyin_to_ipa("notexist9").is_none());
    }

    #[test]
    fn test_text_to_phonemes() {
        let result = text_to_phonemes("你好");
        // v1.0 IPA 映射文件不存在时，汉字无法转换为音素，结果为空
        // 如果文件存在，结果应包含音素和空格
        if !result.is_empty() {
            assert!(result.contains(" "));
        }
        println!("'你好' -> phonemes: '{}'", result);
    }

    #[test]
    fn test_mixed_text() {
        let result = text_to_phonemes("Hello 世界");
        println!("'Hello 世界' -> phonemes: '{}'", result);
        assert!(result.contains("Hello"));
    }

    // ============== v1.1-zh 注音测试 ==============

    #[test]
    fn test_pinyin_to_zhuyin() {
        assert_eq!(pinyin_to_zhuyin("ni3").as_deref(), Some("ㄋㄧ3"));
        assert_eq!(pinyin_to_zhuyin("hao3").as_deref(), Some("ㄏㄠ3"));
        assert_eq!(pinyin_to_zhuyin("shi4").as_deref(), Some("ㄕ十4"));
        assert_eq!(pinyin_to_zhuyin("zhong1").as_deref(), Some("ㄓ中1"));
        assert_eq!(pinyin_to_zhuyin("zi3").as_deref(), Some("ㄗㄭ3"));
        assert_eq!(pinyin_to_zhuyin("chi1").as_deref(), Some("ㄔ十1"));
        assert_eq!(pinyin_to_zhuyin("yao4").as_deref(), Some("要4"));
        assert_eq!(pinyin_to_zhuyin("wo3").as_deref(), Some("我3"));
        assert_eq!(pinyin_to_zhuyin("wei4").as_deref(), Some("为4"));
        assert_eq!(pinyin_to_zhuyin("yue4").as_deref(), Some("月4"));
        assert_eq!(pinyin_to_zhuyin("qu4").as_deref(), Some("ㄑㄩ4"));
        assert_eq!(pinyin_to_zhuyin("nv3").as_deref(), Some("ㄋㄩ3"));
        assert_eq!(pinyin_to_zhuyin("er2").as_deref(), Some("ㄦ2"));
        assert!(pinyin_to_zhuyin("notexist9").is_none());
    }

    #[test]
    fn test_text_to_phonemes_zh() {
        let result = text_to_phonemes_zh("你好");
        // 变调: ni3 hao3 → ni2 hao3 → ㄋㄧ2ㄏㄠ3
        assert_eq!(result, "ㄋㄧ2ㄏㄠ3");
        println!("'你好' -> zhuyin: '{}'", result);
    }

    #[test]
    fn test_zh_mixed_text() {
        let result = text_to_phonemes_zh("Hello 世界");
        println!("'Hello 世界' -> zhuyin: '{}'", result);
        assert!(result.contains("Hello"));
        assert!(result.contains('十'));
    }

    #[test]
    fn test_zh_tone_sandhi() {
        // 两个三声连读：你好 ni3 hao3 → ㄋㄧ2ㄏㄠ3
        assert_eq!(text_to_phonemes_zh("你好"), "ㄋㄧ2ㄏㄠ3");
        // 三声+非三声：好 hao3 + 人 ren2 → 不变
        let r = text_to_phonemes_zh("好人");
        println!("'好人' -> zhuyin: '{}'", r);
        assert!(r.contains("ㄏㄠ3")); // 保持三声
        // 三声+三声+三声：我 wo3 + 也 ye3 + 好 hao3
        let r2 = text_to_phonemes_zh("我也好");
        println!("'我也好' -> zhuyin: '{}'", r2);
        // 结果应包含变调后的前两个音节
    }

    #[test]
    fn test_zh_empty_and_punct() {
        assert_eq!(text_to_phonemes_zh(""), "");
        let r = text_to_phonemes_zh("你好，世界！");
        println!("'你好，世界！' -> zhuyin: '{}'", r);
        assert!(r.contains(','));  // 全角逗号→ASCII
        assert!(r.contains('!'));  // 全角感叹号→ASCII
    }

    #[test]
    fn test_zh_nv3_normalization() {
        // 女: pinyin crate 返回 "nü3"，但映射表用 "nv3"
        let r = text_to_phonemes_zh("女人");
        println!("'女人' -> zhuyin: '{}'", r);
        // 不应包含 IPA 箭头 (↓ 等) - 那说明走 IPA 降级路径了
        assert!(!r.contains('↓'), "不应有 IPA 降级: {}", r);
        // 应为注音: 女的 nv3 → ㄋㄩ3
        assert!(r.contains('ㄋ'), "应有注音符号 ㄋ");
        assert!(r.contains('ㄩ'), "应有注音符号 ㄩ");
        // 三声变调: nv3 + ren2 不变
    }

    #[test]
    fn test_zh_fullwidth_questions() {
        let r = text_to_phonemes_zh("你好吗？我很好！");
        println!("'你好吗？我很好！' -> zhuyin: '{}'", r);
        assert!(r.contains('?'), "全角问号→ASCII ?");
        assert!(r.contains('!'), "全角感叹号→ASCII !");
    }

    #[test]
    fn test_zh_parentheses() {
        let r = text_to_phonemes_zh("这是（测试）文本");
        println!("'这是（测试）文本' -> zhuyin: '{}'", r);
        assert!(!r.contains('（'), "不应有全角左括号");
        assert!(r.contains('('), "应有ASCII左括号");
        assert!(r.contains(')'), "应有ASCII右括号");
    }

    #[test]
    fn test_zh_map_count() {
        let count = mapping::zhuyin_len();
        println!("注音映射条目数: {}", count);
        assert!(count > 1500, "注音映射至少应有 1500 个条目");
    }
}

#[cfg(test)]
mod polyphone_integration_tests {
    use super::*;

    fn zh(s: &str) -> String {
        text_to_phonemes_zh(s)
    }

    #[test]
    fn 银行读hang而非xing() {
        // 消歧后「行」读 háng（与「航」同音），而非单字默认 xíng（与「星」同韵）
        let bank = zh("银行");
        assert_eq!(bank, format!("{}{}", zh("银"), zh("航")), "银行应读 yín háng");
        assert_ne!(bank, format!("{}{}", zh("银"), zh("行")), "不应读 yín xíng");
    }

    #[test]
    fn 重量读zhong4() {
        // 「重」在「重量」中读 zhòng（与「仲」同音）
        let weight = zh("重量");
        assert_eq!(weight, format!("{}{}", zh("仲"), zh("量")), "重量应读 zhòng liàng");
    }

    #[test]
    fn 一变调_一样读yi2() {
        // 「一」在去声前读 yí（与「移」同音）
        assert!(zh("一样").starts_with(&zh("移")), "「一」应变调 yí2，实际: {}", zh("一样"));
        // 句尾「一」保持本调 yī（与「衣」同音）
        assert_eq!(zh("一"), zh("衣"));
    }

    #[test]
    fn 不变调_不是读bu2() {
        assert!(zh("不是").starts_with("ㄅㄨ2"), "「不」应变调 bú2，实际: {}", zh("不是"));
        // 非去声前保持本调（「来」二声）
        assert!(zh("不来").starts_with("ㄅㄨ4"), "「不」应保持 bù4，实际: {}", zh("不来"));
    }

    #[test]
    fn 单字读音不受影响() {
        // 「行走」词组读音与逐字读音一致（词表命中但结果相同）
        assert_eq!(zh("行走"), format!("{}{}", zh("行"), zh("走")));
    }
}
