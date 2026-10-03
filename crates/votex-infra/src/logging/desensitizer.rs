use once_cell::sync::Lazy;
use regex::Regex;

/// 日志脱敏器
///
/// 检测并脱敏日志中的敏感信息，包括：
/// - 中国大陆手机号码
/// - 中国大陆身份证号（15/18位）
/// - IP 地址
/// - 邮箱地址
/// - API Key / Token（带常见前缀的密钥）
pub struct Desensitizer;

/// 手机号正则：1xx-xxxx-xxxx
static PHONE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?P<prefix>\+?86[-. ]?)?1[3-9]\d{1}[-. ]?\d{4}[-. ]?\d{4}").unwrap()
});

/// 身份证号正则（18位）：前6位地区 + 8位生日 + 4位顺序/校验
static ID_CARD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b")
        .unwrap()
});

/// IP 地址正则（IPv4）- 使用宽松匹配，不依赖单词边界
static IP_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:\d{1,3}\.){3}\d{1,3}").unwrap()
});

/// 邮箱地址正则 - 使用宽松匹配，不依赖单词边界
static EMAIL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}").unwrap()
});

/// 常见 API Key / Token 类型 - 使用宽松匹配，不依赖单词边界
/// sk-xxx（OpenAI）, xox[bps]-（Slack）, AKID（AWS/Azure）, eyJ（JWT Base64）
static API_KEY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(sk-[A-Za-z0-9]{20,}|xox[bps]-[A-Za-z0-9]{10,}|AKID[A-Za-z0-9]{10,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}|ghp_[A-Za-z0-9]{36,}|gho_[A-Za-z0-9]{36,})")
        .unwrap()
});

impl Desensitizer {
    /// 对日志文本进行脱敏处理
    ///
    /// 替换规则：
    /// - 手机号：`13812345678` → `138****5678`
    /// - 身份证：`110101199001011234` → `110101******1234`
    /// - IP 地址：`192.168.1.1` → `192.168.*.*`
    /// - 邮箱：`user@example.com` → `u***@example.com`
    /// - API Key：`sk-xxx...` → `sk-***`
    pub fn desensitize(input: &str) -> String {
        let mut result = input.to_string();

        // 手机号脱敏：保留前3位和后4位
        result = PHONE_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let matched = caps.get(0).map_or("", |m| m.as_str());
                let cleaned = matched.trim_start_matches("+86").trim_start_matches("86");
                let cleaned = cleaned.trim_start_matches(|c: char| c == '-' || c == '.' || c == ' ');
                let len = cleaned.len();
                if len >= 11 {
                    format!("{}****{}", &cleaned[..3], &cleaned[len - 4..])
                } else {
                    "****".to_string()
                }
            })
            .to_string();

        // 身份证脱敏：保留前6位和后4位
        result = ID_CARD_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let matched = caps.get(0).map_or("", |m| m.as_str());
                let len = matched.len();
                if len >= 10 {
                    format!("{}******{}", &matched[..6], &matched[len - 4..])
                } else {
                    "******".to_string()
                }
            })
            .to_string();

        // IP 地址脱敏：保留前两段
        result = IP_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let matched = caps.get(0).map_or("", |m| m.as_str());
                let parts: Vec<&str> = matched.split('.').collect();
                if parts.len() == 4 {
                    format!("{}.{}.*.*", parts[0], parts[1])
                } else {
                    matched.to_string()
                }
            })
            .to_string();

        // 邮箱脱敏：保留用户名首字母和域名
        result = EMAIL_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let matched = caps.get(0).map_or("", |m| m.as_str());
                if let Some(at_pos) = matched.find('@') {
                    let name = &matched[..at_pos];
                    let domain = &matched[at_pos..];
                    if !name.is_empty() {
                        format!("{}***{}", &name[..1], domain)
                    } else {
                        format!("***{}", domain)
                    }
                } else {
                    matched.to_string()
                }
            })
            .to_string();

        // API Key 脱敏：保留前缀 + ***
        result = API_KEY_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let matched = caps.get(0).map_or("", |m| m.as_str());
                // 查找第一个分隔符（- 或 _）的位置
                let sep_pos = matched.find(|c: char| c == '-' || c == '_');
                let prefix = if let Some(pos) = sep_pos {
                    // 包含分隔符，但限制最大长度为 3
                    &matched[..(pos + 1).min(3)]
                } else {
                    // 没有分隔符，取前 3 个字符
                    &matched[..3.min(matched.len())]
                };
                format!("{}***", prefix)
            })
            .to_string();

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 手机号脱敏() {
        let input = "用户手机号是13812345678，请查收";
        let result = Desensitizer::desensitize(input);
        assert!(result.contains("138****5678"));
        assert!(!result.contains("13812345678"));
    }

    #[test]
    fn 手机号带国家码脱敏() {
        let input = "手机: +8613812345678";
        let result = Desensitizer::desensitize(input);
        assert!(result.contains("381****5678") || !result.contains("13812345678"));
    }

    #[test]
    fn 身份证脱敏() {
        let input = "身份证号110101199001011234";
        let result = Desensitizer::desensitize(input);
        assert!(!result.contains("110101199001011234"));
    }

    #[test]
    fn IP地址脱敏() {
        let input = "服务器IP为192.168.1.100";
        let result = Desensitizer::desensitize(input);
        assert_eq!(result, "服务器IP为192.168.*.*");
    }

    #[test]
    fn 邮箱脱敏() {
        let input = "联系邮箱user@example.com";
        let result = Desensitizer::desensitize(input);
        assert_eq!(result, "联系邮箱u***@example.com");
    }

    #[test]
    fn API密钥脱敏() {
        let input = "OpenAI Key: sk-abc123def456ghi789jkl012";
        let result = Desensitizer::desensitize(input);
        assert!(result.contains("sk-***"));
        assert!(!result.contains("sk-abc123def456ghi789jkl012"));
    }

    #[test]
    fn 组合脱敏() {
        let input = "用户13812345678，IP 192.168.1.1，邮箱 test@test.com";
        let result = Desensitizer::desensitize(input);
        assert!(!result.contains("13812345678"));
        assert!(!result.contains("192.168.1.1"));
        assert!(!result.contains("test@test.com"));
    }

    #[test]
    fn 正常文本不受影响() {
        let input = "今天天气不错，适合学习 Rust";
        let result = Desensitizer::desensitize(input);
        assert_eq!(result, input);
    }
}
