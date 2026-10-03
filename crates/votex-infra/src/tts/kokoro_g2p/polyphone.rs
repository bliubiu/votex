//! 多音字词组消歧
//!
//! 单字查拼音无法处理多音字：「银行」的「行」单字读 xíng，实际读 háng。
//! 本模块维护一张高频多音字词组表，在 G2P 转换时按「最长匹配优先」命中词组，
//! 用整词读音覆盖单字读音；另含「一 / 不」变调规则（依后字声调变读）。
//!
//! 设计取舍：不引入分词器（jieba 体积大、与单二进制约束冲突），
//! 只对「多音字 + 邻字」的确定性组合做覆盖——未命中的字仍走单字拼音。

/// 词组读音表：词组 → 逐音节拼音（声调数字结尾，空格分隔）
///
/// 收录标准：两个字（或多字）中至少一个为多音字，且词组读音与
/// 单字默认读音不同的高频词。
const POLYPHONE_PHRASES: &[(&str, &str)] = &[
    // 行 háng / xíng
    ("银行", "yin2 hang2"),
    ("行业", "hang2 ye4"),
    ("行走", "xing2 zou3"),
    ("行程", "xing2 cheng2"),
    ("银行家", "yin2 hang2 jia1"),
    // 重 zhòng / chóng
    ("重量", "zhong4 liang4"),
    ("重要", "zhong4 yao4"),
    ("重心", "zhong4 xin1"),
    ("重新", "chong2 xin1"),
    ("重复", "chong2 fu4"),
    ("重叠", "chong2 die2"),
    // 长 zhǎng / cháng
    ("长大", "zhang3 da4"),
    ("长度", "chang2 du4"),
    ("长江", "chang2 jiang1"),
    ("成长", "cheng2 zhang3"),
    ("长处", "chang2 chu4"),
    // 乐 yuè / lè
    ("音乐", "yin1 yue4"),
    ("快乐", "kuai4 le4"),
    ("乐器", "yue4 qi4"),
    ("乐团", "yue4 tuan2"),
    ("乐曲", "yue4 qu3"),
    // 传 chuán / zhuàn
    ("传记", "zhuan4 ji4"),
    ("传播", "chuan2 bo1"),
    ("传递", "chuan2 di4"),
    // 曲 qǔ / qū
    ("歌曲", "ge1 qu3"),
    ("弯曲", "wan1 qu1"),
    ("曲线", "qu1 xian4"),
    // 都 dōu / dū
    ("都是", "dou1 shi4"),
    ("首都", "shou3 du1"),
    ("都市", "du1 shi4"),
    // 数 shù / shǔ
    ("数学", "shu4 xue2"),
    ("数字", "shu4 zi4"),
    // 降 jiàng / xiáng
    ("下降", "xia4 jiang4"),
    ("投降", "tou2 xiang2"),
    // 弹 dàn / tán
    ("子弹", "zi3 dan4"),
    ("弹琴", "tan2 qin2"),
    ("弹奏", "tan2 zou4"),
    // 干 gān / gàn
    ("干净", "gan1 jing4"),
    ("干活", "gan4 huo2"),
    // 相 xiāng / xiàng
    ("相信", "xiang1 xin4"),
    ("相片", "xiang4 pian4"),
    // 兴 xìng / xīng
    ("高兴", "gao1 xing4"),
    ("兴趣", "xing4 qu4"),
    ("兴奋", "xing1 fen4"),
    // 处 chǔ / chù
    ("处理", "chu3 li3"),
    ("到处", "dao4 chu4"),
    // 还 hái / huán
    ("还有", "hai2 you3"),
    ("归还", "gui1 huan2"),
    // 系 xì / jì
    ("关系", "guan1 xi4"),
    ("系统", "xi4 tong3"),
    // 教 jiào / jiāo
    ("教育", "jiao4 yu4"),
    ("教书", "jiao1 shu1"),
    // 假 jiǎ / jià
    ("假如", "jia3 ru2"),
    ("放假", "fang4 jia4"),
    ("假期", "jia4 qi1"),
    // 看 kàn / kān
    ("看见", "kan4 jian4"),
    ("看守", "kan1 shou3"),
    // 参 cān / shēn
    ("参加", "can1 jia1"),
    ("人参", "ren2 shen1"),
    // 佛 fú / fó
    ("仿佛", "fang3 fu2"),
    ("佛像", "fo2 xiang4"),
    ("佛教", "fo2 jiao4"),
    // 藏 cáng / zàng
    ("西藏", "xi1 zang4"),
    ("收藏", "shou1 cang2"),
    ("躲藏", "duo3 cang2"),
    // 供 gōng / gòng
    ("提供", "ti2 gong1"),
    ("口供", "kou3 gong4"),
    // 横 héng / hèng
    ("横竖", "heng2 shu4"),
    ("蛮横", "man2 heng4"),
    // 应 yīng / yìng
    ("应该", "ying1 gai1"),
    ("答应", "da1 ying4"),
    ("反应", "fan3 ying4"),
    ("供应", "gong1 ying4"),
    // 种 zhǒng / zhòng
    ("种子", "zhong3 zi3"),
    ("种地", "zhong4 di4"),
    // 好 hǎo / hào
    ("好处", "hao3 chu4"),
    ("爱好", "ai4 hao4"),
    ("好奇", "hao4 qi2"),
    // 难 nán / nàn
    ("困难", "kun4 nan2"),
    ("遇难", "yu4 nan4"),
    // 奇 qí / jī
    ("奇怪", "qi2 guai4"),
    ("奇数", "ji1 shu4"),
    // 圈 quān / juàn
    ("圆圈", "yuan2 quan1"),
    ("猪圈", "zhu1 juan4"),
    // 卡 kǎ / qiǎ
    ("卡片", "ka3 pian4"),
    ("关卡", "guan1 qia3"),
    // 载 zǎi / zài
    ("记载", "ji4 zai3"),
    ("载重", "zai4 zhong4"),
    ("装载", "zhuang1 zai4"),
    // 便 biàn / pián
    ("便宜", "pian2 yi2"),
    ("方便", "fang1 bian4"),
    // 朝 cháo / zhāo
    ("朝代", "chao2 dai4"),
    ("朝气", "zhao1 qi4"),
    // 发 fā / fà
    ("理发", "li3 fa4"),
    // 得 dé / děi
    ("得到", "de2 dao4"),
    // 缝 féng / fèng
    ("缝隙", "feng4 xi4"),
    ("缝纫", "feng2 ren4"),
];

/// 判断是否为汉字（与 mod.rs 的判定一致）
fn is_hanzi(c: char) -> bool {
    c >= '\u{4E00}' && c <= '\u{9FFF}'
}

/// 在 `chars[pos..]` 处尝试词组匹配（最长匹配优先，≤3 字）
///
/// 返回 (逐音节拼音, 匹配的字符数)。仅当整段都是汉字时才可能命中。
pub fn lookup_phrase(chars: &[char], pos: usize) -> Option<(Vec<String>, usize)> {
    const MAX_LEN: usize = 3;
    for len in (2..=MAX_LEN).rev() {
        if pos + len > chars.len() {
            continue;
        }
        let window: String = chars[pos..pos + len].iter().collect();
        // 词组内必须全为汉字（防止把标点/英文卷进来）
        if !window.chars().all(is_hanzi) {
            continue;
        }
        for (phrase, pinyin) in POLYPHONE_PHRASES {
            if *phrase == window {
                return Some((
                    pinyin.split(' ').map(|s| s.to_string()).collect(),
                    len,
                ));
            }
        }
    }
    None
}

/// 「一 / 不」变调
///
/// - 「一」：本调 yī。后字去声（4 调）→ yí2；后字非去声 → yì4；无后字 → 本调。
/// - 「不」：本调 bù。后字去声 → bú2；其余 → 本调。
///
/// `next_tone` 为紧邻后字（同一连续汉字段内）的声调。
pub fn adjust_yi_bu(c: char, base: &str, next_tone: Option<u8>) -> String {
    match (c, base) {
        ('一', "yi1") => match next_tone {
            Some(4) => "yi2".to_string(),
            Some(_) => "yi4".to_string(),
            None => base.to_string(),
        },
        ('不', "bu4") => match next_tone {
            Some(4) => "bu2".to_string(),
            _ => base.to_string(),
        },
        _ => base.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chars(s: &str) -> Vec<char> {
        s.chars().collect()
    }

    #[test]
    fn 词组匹配_银行() {
        let cs = chars("去银行取钱");
        let (syllables, len) = lookup_phrase(&cs, 1).expect("应命中「银行」");
        assert_eq!(syllables, vec!["yin2", "hang2"]);
        assert_eq!(len, 2);
    }

    #[test]
    fn 词组匹配_最长优先() {
        // 「银行家」应整词命中（3 字），而不是先命中「银行」
        let cs = chars("银行家");
        let (syllables, len) = lookup_phrase(&cs, 0).expect("应命中「银行家」");
        assert_eq!(syllables, vec!["yin2", "hang2", "jia1"]);
        assert_eq!(len, 3);
    }

    #[test]
    fn 词组匹配_同字异词() {
        // 「相信」xiang1 vs 「相片」xiang4
        let cs = chars("相片");
        let (s, _) = lookup_phrase(&cs, 0).unwrap();
        assert_eq!(s[0], "xiang4");
        let cs = chars("相信");
        let (s, _) = lookup_phrase(&cs, 0).unwrap();
        assert_eq!(s[0], "xiang1");
    }

    #[test]
    fn 词组不命中普通词() {
        let cs = chars("你好世界");
        assert!(lookup_phrase(&cs, 0).is_none());
    }

    #[test]
    fn 词组不跨非汉字() {
        // 「，行」不构成词组
        let cs = chars("，行业");
        assert!(lookup_phrase(&cs, 0).is_none());
        assert!(lookup_phrase(&cs, 1).is_some());
    }

    #[test]
    fn 一变调_前去声() {
        // 一样：样 4 声 → yí2
        assert_eq!(adjust_yi_bu('一', "yi1", Some(4)), "yi2");
        // 一直：直 2 声 → yì4
        assert_eq!(adjust_yi_bu('一', "yi1", Some(2)), "yi4");
        // 句尾「一」保持本调
        assert_eq!(adjust_yi_bu('一', "yi1", None), "yi1");
    }

    #[test]
    fn 不变调_前去声() {
        assert_eq!(adjust_yi_bu('不', "bu4", Some(4)), "bu2");
        assert_eq!(adjust_yi_bu('不', "bu4", Some(2)), "bu4");
    }

    #[test]
    fn 非一字不受影响() {
        assert_eq!(adjust_yi_bu('行', "xing2", Some(4)), "xing2");
    }
}
