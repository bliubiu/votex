//! 数字→中文读法转换
//!
//! 解决 TTS 引擎对数字字符逐字符怪读/吞字的问题：
//! 「1998 年」→「一九九八年」、「3.14」→「三点一四」、「50%」→「百分之五十」
//!
//! 纯手写扫描实现，不引入 regex 依赖（domain 层保持轻量）。

/// 将文本中的数字序列转换为中文读法
///
/// 规则：
/// - 4 位数字后跟「年」→ 逐位读（一九九八年）
/// - 含小数点 → 整数部分按数值读 + 「点」+ 逐位读
/// - 数字后跟 `%` → 「百分之 + 数值读法」
/// - 其余整数 → 按数值读法（一千九百九十八），组间正确补「零」
pub fn normalize_numbers(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() * 2);
    let mut i = 0;

    while i < chars.len() {
        let c = chars[i];
        if c.is_ascii_digit() {
            // 收集完整数字序列（含小数点）
            let start = i;
            let mut has_dot = false;
            while i < chars.len() && (chars[i].is_ascii_digit() || (chars[i] == '.' && !has_dot)) {
                if chars[i] == '.' {
                    // 「3.14」的小数点：前后都是数字才算小数
                    let prev_digit = i > 0 && chars[i - 1].is_ascii_digit();
                    let next_digit = i + 1 < chars.len() && chars[i + 1].is_ascii_digit();
                    if prev_digit && next_digit {
                        has_dot = true;
                    } else {
                        break;
                    }
                }
                i += 1;
            }
            let token: String = chars[start..i].iter().collect();

            // 百分号：数字 + 可选空格 + %
            if i < chars.len() && chars[i] == '%' {
                out.push_str("百分之");
                out.push_str(&read_number_token(&token));
                i += 1; // 跳过 %
                continue;
            }

            // 4 位纯数字 + 「年」→ 逐位读（1998年 → 一九九八年）
            if token.len() == 4
                && token.bytes().all(|b| b.is_ascii_digit())
                && chars.get(i) == Some(&'年')
            {
                for b in token.bytes() {
                    out.push_str(DIGITS[(b - b'0') as usize]);
                }
                continue;
            }

            out.push_str(&read_number_token(&token));
        } else {
            out.push(c);
            i += 1;
        }
    }

    out
}

/// 读取一个数字 token（可能含小数点）
fn read_number_token(token: &str) -> String {
    if let Some((int_part, frac_part)) = token.split_once('.') {
        let mut out = integer_to_chinese_str(int_part);
        out.push_str("点");
        for c in frac_part.chars() {
            if let Some(d) = c.to_digit(10) {
                out.push_str(DIGITS[d as usize]);
            }
        }
        out
    } else {
        integer_to_chinese_str(token)
    }
}

const DIGITS: [&str; 10] = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];

/// 整数字符串 → 中文读法
fn integer_to_chinese_str(s: &str) -> String {
    // 去前导零
    let trimmed = s.trim_start_matches('0');
    let digits: Vec<usize> = trimmed
        .bytes()
        .map(|b| (b - b'0') as usize)
        .take_while(|&d| d <= 9)
        .collect();
    if digits.is_empty() {
        return DIGITS[0].to_string();
    }

    // 4 位 + 「年」上下文由调用方处理；这里统一按数值读
    // 超长数字（>12 位，如编号/电话）逐位读，避免生成荒谬量级
    if digits.len() > 12 {
        return digits.iter().map(|&d| DIGITS[d as usize]).collect();
    }

    integer_to_chinese(&digits)
}

/// 按中文数值规则读整数（支持到千亿级）
fn integer_to_chinese(digits: &[usize]) -> String {
    let n = digits.len();

    // ≤4 位：万以内
    if n <= 4 {
        return read_under_10k(digits);
    }

    // >4 位：按 4 位一组从低位切，组间加「万」「亿」
    let mut groups: Vec<&[usize]> = Vec::new();
    let mut rest = digits;
    while rest.len() > 4 {
        let split = rest.len() - 4;
        groups.push(&rest[split..]);
        rest = &rest[..split];
    }
    groups.push(rest);
    groups.reverse(); // 高位组在前

    let units = ["", "万", "亿", "兆"];
    let mut out = String::new();
    let group_count = groups.len();
    let mut prev_group_zero = false;

    for (gi, group) in groups.iter().enumerate() {
        let unit_idx = group_count - 1 - gi;
        let is_zero_group = group.iter().all(|&d| d == 0);
        if is_zero_group {
            prev_group_zero = true;
            continue;
        }
        // 组间衔接补零：跳过的全零组之后（12_0000_0034），或本组有前导零
        // （1234_0567 的 0567、12_0600 的 0600）——只补一个「零」
        if !out.is_empty() && (prev_group_zero || group[0] == 0) {
            out.push('零');
        }
        // 万组只有千位非零时（如 12,0345），读作「十二万」而非「一万二万」——
        // 简化处理：直接按组内读法 + 单位
        out.push_str(&read_under_10k(group));
        out.push_str(units[unit_idx.min(units.len() - 1)]);
        prev_group_zero = false;
    }
    out
}

/// 读 0~9999
fn read_under_10k(digits: &[usize]) -> String {
    let units = ["", "十", "百", "千"];
    let n = digits.len();
    let mut out = String::new();
    let mut zero_pending = false;

    let start = digits.iter().position(|&d| d != 0);
    let Some(start) = start else {
        return DIGITS[0].to_string();
    };

    // 特例：10~19 读「十X」不读「一十X」
    let skip_leading_one = n == 2 && digits[0] == 1;

    for (pos, &d) in digits.iter().enumerate().skip(start) {
        let unit_idx = n - 1 - pos;
        if d == 0 {
            zero_pending = true;
            continue;
        }
        if zero_pending && !out.is_empty() {
            out.push('零');
        }
        zero_pending = false;
        if !(skip_leading_one && pos == 0) {
            out.push_str(DIGITS[d as usize]);
        }
        out.push_str(units[unit_idx]);
    }

    if out.is_empty() {
        out.push_str(DIGITS[0]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 数字_整数读法() {
        assert_eq!(normalize_numbers("我有 3 个苹果"), "我有 三 个苹果");
        assert_eq!(normalize_numbers("共 120 元"), "共 一百二十 元");
        // 1998 单独出现按数值读——一千九百九十八
        assert_eq!(normalize_numbers("编号1998号"), "编号一千九百九十八号");
    }

    #[test]
    fn 数字_小数读法() {
        assert_eq!(normalize_numbers("圆周率是3.14"), "圆周率是三点一四");
        assert_eq!(normalize_numbers("3.0"), "三点零");
    }

    #[test]
    fn 数字_百分号() {
        assert_eq!(normalize_numbers("50%"), "百分之五十");
        assert_eq!(normalize_numbers("增长12.5%了"), "增长百分之十二点五了");
    }

    #[test]
    fn 数字_长编号逐位() {
        assert_eq!(normalize_numbers("123456789012345"), "一二三四五六七八九零一二三四五");
    }

    #[test]
    fn 数字_万级() {
        assert_eq!(normalize_numbers("10万人口"), "十万人口");
        assert_eq!(normalize_numbers("12345678"), "一千二百三十四万五千六百七十八");
    }

    #[test]
    fn 数字_年份逐位读() {
        assert_eq!(normalize_numbers("1998年"), "一九九八年");
        assert_eq!(normalize_numbers("2026年春天"), "二零二六年春天");
        // 非 4 位数字 + 年 仍按数值读
        assert_eq!(normalize_numbers("365年"), "三百六十五年");
    }

    #[test]
    fn 数字_组间补零() {
        // 低组有前导零
        assert_eq!(normalize_numbers("12000034"), "一千二百万零三十四");
        assert_eq!(normalize_numbers("12340567"), "一千二百三十四万零五百六十七");
        assert_eq!(normalize_numbers("100600"), "十万零六百");
        assert_eq!(normalize_numbers("10200304"), "一千零二十万零三百零四");
        // 全零组被跳过后衔接
        assert_eq!(normalize_numbers("1200000034"), "十二亿零三十四");
        // 低组无前导零不补零
        assert_eq!(normalize_numbers("12005678"), "一千二百万五千六百七十八");
        // 全零尾组不补尾零
        assert_eq!(normalize_numbers("12000000"), "一千二百万");
    }

    #[test]
    fn 数字_不干扰非数字文本() {
        let text = "今天天气很好。！？…；";
        assert_eq!(normalize_numbers(text), text);
    }

    #[test]
    fn 数字_版本号风格小数() {
        assert_eq!(normalize_numbers("v1.2.3"), "v一点二.三");
    }
}
