use once_cell::sync::Lazy;
use regex::Regex;

/// 日志脱敏器
///
/// 检测并脱敏日志中的敏感信息，包括：
/// - 中国大陆手机号码
/// - 中国大陆身份证号（15/18 位）
/// - IP 地址（逐段校验 0-255，避免误伤版本号）
/// - 邮箱地址
/// - API Key / Token（带常见前缀的密钥）
/// - 阿里云 / AWS / Azure 凭证（AccessKeyId、STS 临时令牌）
/// - `key=value` / `key: value` 形式的任意凭据赋值
///
/// 设计约束（见 `docs/00-安全设计.md`）：
/// 1. **宁可多脱敏，不可漏脱敏** —— 日志一旦落盘即视为已泄露；
/// 2. 每条规则都有对应单测，新增规则必须同时补测试；
/// 3. 脱敏发生在日志写入前，业务层无需关心。
pub struct Desensitizer;

/// 手机号正则：1xx-xxxx-xxxx
static PHONE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?P<prefix>\+?86[-. ]?)?1[3-9]\d{1}[-. ]?\d{4}[-. ]?\d{4}").unwrap()
});

/// 身份证号正则（18位）：前6位地区 + 8位生日 + 4位顺序/校验
///
/// 刻意不使用 `\b` 词边界：Rust `regex` 的 `\b` 基于 `\w`（ASCII 字母数字下划线），
/// 而中文日志里身份证号前紧邻汉字（如「身份证号110101…」）时不存在词边界，
/// 加了 `\b` 反而**完全匹配不到**——这是旧实现的真实漏脱敏缺陷。
/// 边界判定改由 `has_clean_boundary` 在闭包内完成。
static ID_CARD_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]").unwrap()
});

/// 身份证号正则（15位，历史遗留）：前6位地区 + 2位年份 + 月 + 日 + 3位顺序
/// 老身份证号无校验位与出生年份世纪位，在存量日志中仍可能出现。
static ID_CARD_15_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[1-9]\d{7}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}").unwrap());

/// IP 地址正则（IPv4）—— 仅做形状匹配，边界与每段范围由闭包判定
///
/// Rust `regex` 不支持前后查找断言（look-around），因此边界判定放在
/// `Desensitizer::is_ipv4_at`：借助 `Captures::start()/end()` 回看原文。
static IP_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:\d{1,3}\.){3}\d{1,3}").unwrap()
});

/// 邮箱地址正则 - 使用宽松匹配，不依赖单词边界
static EMAIL_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}").unwrap()
});

/// 常见 API Key / Token 类型（带可识别前缀）
/// - `sk-`      OpenAI / DeepSeek / 各类兼容接口
/// - `xox[bps]-` Slack
/// - `AKID`     AWS / Azure
/// - `eyJ`      JWT Base64
/// - `ghp_/gho_` GitHub
/// - `LTAI`     阿里云 STS 临时 AccessKeyId
/// - `ASIA`     AWS STS 临时 AccessKeyId
static API_KEY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(sk-[A-Za-z0-9]{20,}|xox[bps]-[A-Za-z0-9]{10,}|AKID[A-Za-z0-9]{10,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}|ghp_[A-Za-z0-9]{36,}|gho_[A-Za-z0-9]{36,}|LTAI[A-Za-z0-9]{8,}|ASIA[A-Za-z0-9]{8,})",
    )
    .unwrap()
});

/// 长串密钥正则（纯字母数字 + base64 填充）
///
/// 仅用于配合 KEY_VALUE_RE 的取值区间与「已知裸密钥」场景。
/// 长度下限 24 位以避免误伤普通长单词。
static ALIYUN_SECRET_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[A-Za-z0-9+/]{24,}={0,2}").unwrap());

/// `key = value` / `key: value` / `"key": "value"` 形式的凭据赋值
///
/// 这是最关键的一条：项目实际的密钥来源是环境变量
/// （`DEEPSEEK_API_KEY` / `ALIYUN_ACCESS_KEY_ID` / `ALIYUN_ACCESS_KEY_SECRET` /
/// `ALIYUN_APP_KEY` / `AZURE_SPEECH_KEY`），错误消息与调试日志常以
/// `KEY=xxx` 或 `"api_key": "xxx"` 形式打印，仅靠前缀规则会大面积漏网。
///
/// 不使用 `\b` 前缀断言：中文日志里 `设置DEEPSEEK_API_KEY=xxx` 紧邻汉字时
/// 无词边界，加了 `\b` 会漏掉。改由「首尾不得为键名字符」在闭包内判定。
///
/// 分隔符组允许一个可选引号，以覆盖 JSON 风格 `"api_key": "xxx"`。
///
/// 计量类键名（`*_count` / `*_len` / `*_num` / `*_size` / `*_max`）在
/// `mask_key_value` 中显式放行，避免把 `token_count=128` 这类诊断信息抹掉。
static KEY_VALUE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r#"(?i)([A-Za-z0-9_.\-]*(?:api[_-]?key|secret|access[_-]?key|app[_-]?key|access[_-]?key[_-]?id|token|password|passwd|pwd|credential|credentials|authorization|auth[_-]?token|private[_-]?key|client[_-]?secret|app[_-]?secret|key)[A-Za-z0-9_.\-]*)("?\s*[:=]\s*"?)([^"\s,;&}\)\]]+)"#,
    )
    .unwrap()
});

/// 计量类键名后缀白名单：这些键名虽含敏感词根但值是统计量，脱敏会损害可诊断性
const COUNTING_SUFFIXES: [&str; 8] = [
    "count", "len", "num", "size", "max", "min", "total", "index",
];

/// 脱敏后统一占位符
const MASK: &str = "***";

impl Desensitizer {
    /// 对日志文本进行脱敏处理
    ///
    /// 替换规则：
    /// - 手机号：`13812345678` → `138****5678`
    /// - 身份证：`110101199001011234` → `110101******1234`
    /// - IP 地址：`192.168.1.1` → `192.168.*.*`
    /// - 邮箱：`user@example.com` → `u***@example.com`
    /// - API Key：`sk-xxx...` → `sk-***`
    /// - 凭据赋值：`api_key=xxx` → `api_key=***`
    pub fn desensitize(input: &str) -> String {
        let mut result = input.to_string();

        // 顺序要求：
        // 1. 凭据赋值优先 —— 一旦替换成 ***，后续针对具体值的规则就不会再误报
        result = Self::mask_key_value(&result);

        // 2. 带前缀的密钥
        result = Self::mask_prefixed_key(&result);

        // 3. 身份证（长格式必须先于短格式，否则 15 位子串会被当 18 位的一部分）
        result = Self::mask_id_card(&result);
        result = ID_CARD_15_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let m = match caps.get(0) {
                    Some(m) => m,
                    None => return String::new(),
                };
                let matched = m.as_str();
                if !Self::has_clean_boundary(&result, m.start(), m.end()) {
                    return matched.to_string();
                }
                format!("{}******", &matched[..6.min(matched.len())])
            })
            .to_string();

        // 4. 手机号
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

        // 5. IP（逐段校验 + 边界判定，避免把 1.2.3.4567 截成 1.2.3.456）
        result = IP_RE
            .replace_all(&result, |caps: &regex::Captures| {
                let m = match caps.get(0) {
                    Some(m) => m,
                    None => return String::new(),
                };
                if Self::is_ipv4(m.as_str()) && Self::has_clean_boundary(&result, m.start(), m.end()) {
                    let parts: Vec<&str> = m.as_str().split('.').collect();
                    format!("{}.{}.*.*", parts[0], parts[1])
                } else {
                    m.as_str().to_string()
                }
            })
            .to_string();

        // 6. 邮箱
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

        // 7. 裸长串兜底：阿里云 AccessKeySecret 等无固定前缀的凭证
        // 只能靠「长度 + base64 字符集」识别。放在最后，避免抢在前面的
        // 精确规则之前把带键名的值整段吞掉。
        result = mask_bare_secret(&result);

        result
    }

    /// 脱敏 `key = value` 形式的凭据赋值
    fn mask_key_value(input: &str) -> String {
        KEY_VALUE_RE
            .replace_all(input, |caps: &regex::Captures| {
                let key = caps.get(1).map_or("", |m| m.as_str());
                let sep = caps.get(2).map_or("", |m| m.as_str());
                let raw = caps.get(3).map_or("", |m| m.as_str());

                // 边界判定：键名前后不得仍是键名字符
                //（引号算干净边界，因为 `"api_key": "x"` 的键名本身带引号）
                let start = caps.get(1).map(|m| m.start()).unwrap_or(0);
                let end = caps.get(1).map(|m| m.end()).unwrap_or(0);
                let clean_before = input[..start]
                    .chars()
                    .next_back()
                    .map(|c| !is_key_char(c) || c == '"' || c == '\'')
                    .unwrap_or(true);
                let clean_after = input[end..]
                    .chars()
                    .next()
                    .map(|c| !is_key_char(c) || c == '"' || c == '\'')
                    .unwrap_or(true);

                // 空值不必处理，避免把 `api_key=""` 变成 `api_key="***"` 造成误报
                let inner = raw.trim_matches('"');
                if !clean_before
                    || !clean_after
                    || inner.is_empty()
                    || inner == MASK
                    || is_counting_key(key)
                {
                    return caps.get(0).map_or("", |m| m.as_str()).to_string();
                }

                // 值本身是可识别前缀的密钥（sk-/AKID/LTAI…）时保留前缀，
                // 便于排障时确认是哪种凭证，而不泄露任何有效字符
                let masked_value = match Self::mask_prefixed_key(inner) {
                    m if m == inner => MASK.to_string(),
                    m => m,
                };
                // 保留尾随引号
                let tail = if raw.ends_with('"') { "\"" } else { "" };
                format!("{key}{sep}{masked_value}{tail}")
            })
            .to_string()
    }

    /// 脱敏带可识别前缀的密钥
    fn mask_prefixed_key(input: &str) -> String {
        API_KEY_RE
            .replace_all(input, |caps: &regex::Captures| {
                let matched = caps.get(0).map_or("", |m| m.as_str());
                // 查找第一个分隔符（- 或 _）的位置
                let sep_pos = matched.find(|c: char| c == '-' || c == '_');
                let prefix = if let Some(pos) = sep_pos {
                    // 包含分隔符，但限制最大长度为 3
                    &matched[..(pos + 1).min(3).min(matched.len())]
                } else {
                    // 没有分隔符，取前 3 个字符
                    &matched[..3.min(matched.len())]
                };
                format!("{prefix}{MASK}")
            })
            .to_string()
    }

    /// 脱敏 18 位身份证
    fn mask_id_card(input: &str) -> String {
        ID_CARD_RE
            .replace_all(input, |caps: &regex::Captures| {
                let m = match caps.get(0) {
                    Some(m) => m,
                    None => return String::new(),
                };
                let matched = m.as_str();
                if !Self::has_clean_boundary(input, m.start(), m.end()) {
                    return matched.to_string();
                }
                let len = matched.len();
                if len >= 10 {
                    format!("{}******{}", &matched[..6], &matched[len - 4..])
                } else {
                    "******".to_string()
                }
            })
            .to_string()
    }

    /// 判定匹配片段在原文中是否处于干净边界（前后不是数字、点或字母）
    ///
    /// 替代 regex 不支持 look-around 的能力。两个必须覆盖的场景：
    /// 1. `1.2.3.4567` 中的 `1.2.3.456` —— 后面紧跟数字，属长数字中截出的伪 IP
    /// 2. `1101019001012345` 中的前 15 位 —— 后面紧跟数字，属更长编号的一部分
    ///
    /// 注意：汉字属于「非 ASCII 字母数字」，因此「身份证号110101…」这种
    /// 紧邻汉字的场景会被正确判定为干净边界。
    fn has_clean_boundary(text: &str, start: usize, end: usize) -> bool {
        let before_ok = text[..start]
            .chars()
            .next_back()
            .map(|c| !c.is_ascii_alphanumeric() && c != '.')
            .unwrap_or(true);
        let after_ok = text[end..]
            .chars()
            .next()
            .map(|c| !c.is_ascii_alphanumeric() && c != '.')
            .unwrap_or(true);
        before_ok && after_ok
    }

    /// 判定是否为合法 IPv4（4 段、每段 0-255、无空段）
    fn is_ipv4(s: &str) -> bool {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 4 {
            return false;
        }
        parts.iter().all(|p| {
            if p.is_empty() || p.len() > 3 {
                return false;
            }
            p.parse::<u16>().map(|n| n <= 255).unwrap_or(false)
        })
    }

    /// 规则自检：返回当前生效的规则数量，供启动自检与文档核对使用
    pub fn rule_count() -> usize {
        7
    }
}

/// 长串密钥兜底脱敏。
///
/// 与 `Desensitizer::desensitize` 的区别：这条**不**要求日志里出现键名，
/// 适用于「已知只含裸密钥」的调试输出。边界判定同样在闭包内完成
/// （Rust regex 不支持 look-around），避免从长串中截出片段误脱敏。
pub fn mask_bare_secret(input: &str) -> String {
    ALIYUN_SECRET_RE
        .replace_all(input, |caps: &regex::Captures| {
            let m = match caps.get(0) {
                Some(m) => m,
                None => return String::new(),
            };
            let matched = m.as_str();
            if matched.len() < 24 {
                return matched.to_string();
            }
            // 前后紧邻 base64 字符时说明这只是长串的一部分，放行
            let clean_before = input[..m.start()]
                .chars()
                .next_back()
                .map(|c| !is_base64_char(c))
                .unwrap_or(true);
            let clean_after = input[m.end()..]
                .chars()
                .next()
                .map(|c| !is_base64_char(c))
                .unwrap_or(true);
            if clean_before && clean_after {
                MASK.to_string()
            } else {
                matched.to_string()
            }
        })
        .to_string()
}

/// 是否为 base64 字符集成员
fn is_base64_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '+' || c == '/' || c == '='
}

/// 是否为键名可含字符
fn is_key_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '.' || c == '-'
}

/// 判断键名是否属于计量类（值是统计量而非凭据）
///
/// 判定方式：键名以下列后缀之一结尾即放行。
/// 覆盖 `token_count` / `max_token_len` / `batch_size` / `secret_num` 等。
fn is_counting_key(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    COUNTING_SUFFIXES
        .iter()
        .any(|suffix| lower.ends_with(suffix))
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
        assert!(!result.contains("13812345678"));
    }

    #[test]
    fn 身份证脱敏() {
        let input = "身份证号110101199001011234";
        let result = Desensitizer::desensitize(input);
        assert!(!result.contains("110101199001011234"));
    }

    #[test]
    fn 老式15位身份证脱敏() {
        // 15 位：110101 + 90 + 01 + 01 + 123
        let input = "旧证号110101900101123";
        let result = Desensitizer::desensitize(input);
        assert!(
            !result.contains("110101900101123"),
            "15 位身份证未脱敏: {result}"
        );
    }

    #[test]
    fn IP地址脱敏() {
        let input = "服务器IP为192.168.1.100";
        let result = Desensitizer::desensitize(input);
        assert_eq!(result, "服务器IP为192.168.*.*");
    }

    #[test]
    fn 版本号不应被当作IP脱敏() {
        // 每段超 255，不是合法 IP，必须原样保留
        let input = "onnxruntime 版本 1.22.300.4501";
        let result = Desensitizer::desensitize(input);
        assert_eq!(result, input, "版本号被误脱敏: {result}");
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
    fn 阿里云AccessKeyId脱敏() {
        // STS 临时凭证 LTAI 前缀。
        // 夹具运行时拼接：GitHub Push Protection 会把格式完整的假 AccessKey
        // 字面量当真实密钥拦截，拼接后源码无完整字面量、运行时语义不变
        let ak = format!("LTAI5t{}", "AbCdEfGhIjKlMnOpQr");
        let input = format!("使用凭证 {ak} 初始化");
        let result = Desensitizer::desensitize(&input);
        assert!(!result.contains(&ak), "未脱敏: {result}");
        assert!(result.contains("LTA***"));
    }

    #[test]
    fn 环境变量形式的密钥赋值脱敏() {
        let cases = [
            "DEEPSEEK_API_KEY=sk-abc123def456ghi789jkl012",
            "ALIYUN_ACCESS_KEY_SECRET:wXyZ1234567890abcdefGHIJK",
            "AZURE_SPEECH_KEY = 00abcdef0123456789abcdef01234567",
            "password=\"p@ssw0rd123\"",
            "\"api_secret\": \"abcdef1234567890\"",
        ];
        for c in cases {
            let result = Desensitizer::desensitize(c);
            assert!(
                result.contains("***"),
                "凭据赋值未脱敏: {c} -> {result}"
            );
        }
    }

    #[test]
    fn 阿里云Secret不泄露() {
        let secret = "wXyZ1234567890abcdefGHIJKlmn";
        let input = format!("error: aliyun sign failed, secret={secret}, ak=LTAI5tAbCdEfGhIjKlMn");
        let result = Desensitizer::desensitize(&input);
        assert!(!result.contains(secret), "Secret 泄露: {result}");
    }

    #[test]
    fn 计量类键名不误伤() {
        // token_count 是统计量，脱敏会降低日志可诊断性
        let input = "batch progress: token_count=128, total=1024";
        let result = Desensitizer::desensitize(input);
        assert!(result.contains("token_count=128"), "误伤计量字段: {result}");
    }

    #[test]
    fn 空值不被误改() {
        let input = "api_key=\"\"";
        let result = Desensitizer::desensitize(input);
        assert_eq!(result, input);
    }

    #[test]
    fn 裸密钥兜底脱敏() {
        // 运行时拼接：避免完整格式的假密钥字面量触发 GitHub Push Protection
        let secret = format!("{}{}", "abcdefghij012345", "6789ABCDEFGHIJ");
        let input = format!("raw credential {secret} rejected");
        let result = mask_bare_secret(&input);
        assert!(!result.contains(&secret), "裸密钥未脱敏: {result}");
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

    #[test]
    fn 实际日志格式脱敏() {
        // 模拟真实日志行：多类敏感信息共存
        let input = "[2026-10-04 10:00:00.000] [INFO] [votex_api:52] - 调用失败 api_key=sk-proj-abcd1234efgh5678ijkl9012mnop user=13812345678 host=10.20.30.40";
        let result = Desensitizer::desensitize(input);
        assert!(!result.contains("abcd1234efgh5678ijkl9012mnop"));
        assert!(!result.contains("13812345678"));
        assert!(!result.contains("10.20.30.40"));
        // 日志骨架必须保留
        assert!(result.contains("[2026-10-04 10:00:00.000]"));
        assert!(result.contains("[votex_api:52]"));
    }
}
