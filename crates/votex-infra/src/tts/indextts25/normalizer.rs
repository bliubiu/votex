//! IndexTTS-2.5 文本归一化（Rust 移植，Task #49）。
//!
//! 三层结构，与参考实现 `index_tts_2_5_onnx/core/frontend.py::TextNormalizer`
//! + PyPI `wetext==0.1.8`（nbest=1 快路径）逐层对齐：
//!
//! 1. [`FstTextNormalizer`]：kaldifst.TextNormalizer 等价物——UTF-8 字节
//!    acceptor → compose → shortest_path(1) → 输出标签按字节解码。FST 引擎
//!    为 rustfst（OpenFST 二进制格式），FST 文件**逐字节取自 wetext 0.1.8
//!    wheel**（zh/tn + en/tn 的 tagger/verbalizer，编译期 `include_bytes!`
//!    内嵌，首次使用物化到系统临时目录）。该部分流程移植自
//!    SpenserCai/wetext-rs（Apache-2.0），TokenParser 补齐了其缺失的
//!    `escape_value`（Python `Token.string` 的值转义语义）。
//! 2. [`wetext_tn_normalize`]：wetext `utils.normalize` nbest=1 快路径移植：
//!    `preprocess(strip) → should_normalize 门控 → tag → TokenParser.reorder
//!    → verbalize → postprocess(strip)`。
//!    **门控语义**（`should_normalize`，operator=tn、remove_erhua=false）：
//!    zh 仅当含 ASCII 数字才触发 FST；en 非空即触发（wetext-rs 上游对 en
//!    无数字文本会跳过 FST，此处已修正为 Python 语义）。
//! 3. [`TextNormalizer`]：参考实现包装层——`use_chinese` 判定（含汉字 / 无
//!    字母 / 邮箱 / 拼音声调）、英文收缩式 `X's → X is`、术语（TECH_TERM）/
//!    拼音声调 / 中文连字符人名的暂存-恢复、归一化后 `ZH_CHAR_REP_MAP`
//!    （含 `$`→`.`）/ `CHAR_REP_MAP` 二次映射。
//!
//! 有意为之的差异（均不影响金标对拍）：
//! - 术语表（glossary）：参考实现 `term_glossary` 默认为空字典，
//!   `apply_glossary_terms` 恒等返回，故不移植；
//! - Python `set` 迭代顺序受随机化哈希影响，暂存占位符编号不确定，但恢复
//!   使用同一映射，最终输出与占位符编号无关；Rust 侧统一用**首次出现顺序**
//!   （术语按长度降序、首现序破平），确定可复现。子串互相重叠的拼音集合
//!   属病态输入，Python 自身亦不确定，不追求对齐；
//! - compose 结果为空时 Python 的 kaldifst 抛异常（en 路径被 `except` 捕获
//!   回退到收缩式展开后的原文；zh 路径直接冒泡），Rust 同语义：
//!   [`wetext_tn_normalize`] 返回 `Err`，由 [`TextNormalizer::normalize`]
//!   的分支逻辑决定回退或冒泡。
//!
//! 金标：`tmp/make_normalizer_refs.py`（Python wetext 0.1.8 + 参考包装层）
//! 生成 `tmp/indextts25_refs/normalizer_golden.json`，见文末对拍测试。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use rustfst::algorithms::compose::compose;
use rustfst::algorithms::shortest_path;
use rustfst::fst_impls::VectorFst;
use rustfst::fst_traits::SerializableFst;
use rustfst::prelude::*;
use rustfst::semirings::TropicalWeight;
use rustfst::utils::{acceptor, decode_linear_fst};
use rustfst::{Label, EPS_LABEL};

use super::frontend::CHAR_REP_MAP;

// ---------------------------------------------------------------------------
// 1. 内嵌 FST 资产（逐字节来自 wetext 0.1.8 wheel wetext/fsts/）
// ---------------------------------------------------------------------------

const ZH_TN_TAGGER: &[u8] = include_bytes!("fsts/zh/tn/tagger.fst");
const ZH_TN_VERBALIZER: &[u8] = include_bytes!("fsts/zh/tn/verbalizer.fst");
const EN_TN_TAGGER: &[u8] = include_bytes!("fsts/en/tn/tagger.fst");
const EN_TN_VERBALIZER: &[u8] = include_bytes!("fsts/en/tn/verbalizer.fst");

/// 将内嵌 FST 物化到临时目录（幂等），返回 FST 根目录。
fn materialize_fsts() -> Result<PathBuf> {
    let dir = std::env::temp_dir().join("votex-wetext-fsts-0.1.8");
    let entries: &[(&str, &[u8])] = &[
        ("zh/tn/tagger.fst", ZH_TN_TAGGER),
        ("zh/tn/verbalizer.fst", ZH_TN_VERBALIZER),
        ("en/tn/tagger.fst", EN_TN_TAGGER),
        ("en/tn/verbalizer.fst", EN_TN_VERBALIZER),
    ];
    for (rel, bytes) in entries {
        let path = dir.join(rel);
        let fresh = std::fs::metadata(&path)
            .map(|m| m.len() == bytes.len() as u64)
            .unwrap_or(false);
        if !fresh {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&path, bytes)
                .map_err(|e| anyhow!("物化 wetext FST 失败（{:?}）: {e}", path))?;
        }
    }
    Ok(dir)
}

// ---------------------------------------------------------------------------
// 2. FstTextNormalizer（kaldifst.TextNormalizer 等价物，移植自 wetext-rs）
// ---------------------------------------------------------------------------

struct FstTextNormalizer {
    fst: VectorFst<TropicalWeight>,
}

impl FstTextNormalizer {
    fn load(path: &std::path::Path) -> Result<Self> {
        let fst = VectorFst::<TropicalWeight>::read(path)
            .map_err(|e| anyhow!("FST 加载失败（{:?}）: {e}", path))?;
        Ok(Self { fst })
    }

    /// 字节 acceptor → compose → shortest_path(1) → 输出字节解码。
    /// compose 为空时返回 `Err`（调用方按语言分支决定回退语义）。
    fn normalize(&self, input: &str) -> Result<String> {
        if input.is_empty() {
            return Ok(String::new());
        }
        // wetext FST 的输入/输出标签均为 UTF-8 字节（Python acceptor 同源）
        let labels: Vec<Label> = input.as_bytes().iter().map(|&b| b as Label).collect();
        let input_fst: VectorFst<TropicalWeight> = acceptor(&labels, TropicalWeight::one());
        let composed: VectorFst<TropicalWeight> = compose::<
            TropicalWeight,
            VectorFst<TropicalWeight>,
            VectorFst<TropicalWeight>,
            VectorFst<TropicalWeight>,
            _,
            _,
        >(&input_fst, &self.fst)
        .map_err(|e| anyhow!("FST compose 失败: {e}"))?;
        if composed.num_states() == 0 {
            return Err(anyhow!("FST compose 结果为空（无匹配路径）"));
        }
        let best: VectorFst<TropicalWeight> =
            shortest_path(&composed).map_err(|e| anyhow!("FST shortest_path 失败: {e}"))?;
        if best.num_states() == 0 {
            return Err(anyhow!("FST 最短路径为空"));
        }
        let path = decode_linear_fst(&best).map_err(|e| anyhow!("解码线性路径失败: {e}"))?;
        // zh/en TN FST 输出标签为 UTF-8 字节（Python 侧 bytes(...).decode('utf-8') 同源）
        let bytes: Vec<u8> = path
            .olabels
            .iter()
            .filter(|&&l| l != EPS_LABEL)
            .map(|&l| l as u8)
            .collect();
        String::from_utf8(bytes).map_err(|e| anyhow!("FST 输出非法 UTF-8: {e}"))
    }
}

// ---------------------------------------------------------------------------
// 3. wetext 快路径（utils.normalize nbest=1，operator=tn）
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WetextLang {
    Zh,
    En,
}

/// wetext 0.1.8 TN 常量：`TN_ORDERS` / `EN_TN_ORDERS`（TokenParser 字段排序）。
fn token_orders(lang: WetextLang) -> &'static [(&'static str, &'static [&'static str])] {
    const ZH: &[(&str, &[&str])] = &[
        ("date", &["year", "month", "day"]),
        ("fraction", &["denominator", "numerator"]),
        ("measure", &["denominator", "numerator", "value"]),
        ("money", &["value", "currency"]),
        ("time", &["noon", "hour", "minute", "second"]),
    ];
    const EN: &[(&str, &[&str])] = &[
        ("date", &["preserve_order", "text", "day", "month", "year"]),
        (
            "money",
            &["integer_part", "fractional_part", "quantity", "currency_maj"],
        ),
    ];
    match lang {
        WetextLang::Zh => ZH,
        WetextLang::En => EN,
    }
}

/// Python `escape_value`：值输出时转义反斜杠与双引号。
fn escape_value(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Python `TokenParser`（TN 分支）忠实移植：解析 `name { key: "value" }`
/// token 流并按语言字段序重排。解析失败返回 `Err`（tagger 输出恒为
/// 全 token 流，该分支仅防御退化输入）。
fn token_reorder(tagged: &str, lang: WetextLang) -> Result<String> {
    if tagged.trim().is_empty() {
        return Ok(String::new());
    }
    let chars: Vec<char> = tagged.chars().collect();

    fn parse_key(chars: &[char], index: &mut usize) -> Result<String> {
        let start = *index;
        while *index < chars.len() && (chars[*index].is_ascii_alphabetic() || chars[*index] == '_')
        {
            *index += 1;
        }
        Ok(chars[start..*index].iter().collect())
    }

    /// Python `parse_chars(exp)`：逐个尝试消费 exp 中的单字符（ok |= parse_char(x)）。
    fn parse_chars(chars: &[char], index: &mut usize, exp: &str) {
        for x in exp.chars() {
            if *index < chars.len() && chars[*index] == x {
                *index += 1;
            }
        }
    }

    /// Python `parse_value`：读至未转义的 `"`；`\\` 时连带下一字符原样保留。
    fn parse_value(chars: &[char], index: &mut usize) -> Result<String> {
        let mut value = String::new();
        while *index < chars.len() && chars[*index] != '"' {
            value.push(chars[*index]);
            let escape = chars[*index] == '\\';
            *index += 1;
            if escape {
                match chars.get(*index) {
                    Some(&c) => {
                        value.push(c);
                        *index += 1;
                    }
                    None => return Err(anyhow!("token 值以反斜杠结尾（EOS）")),
                }
            }
        }
        Ok(value)
    }

    struct Tok {
        name: String,
        order: Vec<String>,
        members: HashMap<String, String>,
    }

    // ---- parse（Python TokenParser.parse）----
    let mut tokens: Vec<Tok> = Vec::new();
    let mut index = 0usize;
    loop {
        // parse_ws：仅跳过 ASCII 空格
        while index < chars.len() && chars[index] == ' ' {
            index += 1;
        }
        if index >= chars.len() {
            break; // EOS
        }
        let name = parse_key(&chars, &mut index)?;
        if name.is_empty() {
            return Err(anyhow!("token 名缺失（非 token 流输入）"));
        }
        parse_chars(&chars, &mut index, " { ");
        let mut token = Tok {
            name,
            order: Vec::new(),
            members: HashMap::new(),
        };
        loop {
            while index < chars.len() && chars[index] == ' ' {
                index += 1;
            }
            if index >= chars.len() {
                break; // EOS：token 无 '}' 结束
            }
            if chars[index] == '}' {
                index += 1;
                break;
            }
            let key = parse_key(&chars, &mut index)?;
            if key.is_empty() {
                return Err(anyhow!("token 键缺失"));
            }
            parse_chars(&chars, &mut index, ": \"");
            let value = parse_value(&chars, &mut index)?;
            if index < chars.len() && chars[index] == '"' {
                index += 1; // parse_char('"')
            }
            token.order.push(key.clone());
            token.members.insert(key, value);
        }
        tokens.push(token);
    }
    if tokens.is_empty() {
        return Err(anyhow!("token 流为空"));
    }

    // ---- reorder_with_spans → " ".join(serialized) ----
    let orders = token_orders(lang);
    let mut out_parts: Vec<String> = Vec::with_capacity(tokens.len());
    for token in &tokens {
        let mut output = format!("{} {{", token.name);
        // Python Token.string：若 token 名有 canonical order 且未标记
        // preserve_order=true，则按 canonical 序输出
        let order: Vec<String> = match orders.iter().find(|(n, _)| *n == token.name) {
            Some((_, fields)) => {
                if token.members.get("preserve_order").map(String::as_str) == Some("true") {
                    token.order.clone()
                } else {
                    let mut o: Vec<String> = fields.iter().map(|s| s.to_string()).collect();
                    for key in &token.order {
                        if !o.contains(key) {
                            o.push(key.clone());
                        }
                    }
                    o
                }
            }
            None => token.order.clone(),
        };
        for key in &order {
            if let Some(value) = token.members.get(key) {
                output.push_str(&format!(" {}: \"{}\"", key, escape_value(value)));
            }
        }
        output.push_str(" }");
        out_parts.push(output);
    }
    Ok(out_parts.join(" "))
}

/// wetext 0.1.8 `utils.normalize` nbest=1 快路径（operator=tn，
/// traditional_to_simple / full_to_half / remove_interjections / remove_puncts /
/// tag_oov / fix_contractions / remove_erhua 均为 false——与参考实现的调用配置
/// 一致，故预处理/后处理只剩 strip）。
fn wetext_tn_normalize(fsts: &WetextFsts, text: &str, lang: WetextLang) -> Result<String> {
    // preprocess（无 traditional_to_simple）：strip
    let prepared = text.trim();
    // should_normalize 门控：zh 仅含 ASCII 数字才触发；en 非空即触发
    let should = match lang {
        WetextLang::Zh => prepared.chars().any(|c| c.is_ascii_digit()),
        WetextLang::En => !prepared.is_empty(),
    };
    let output = if should {
        // tag（.strip()）
        let tagger = match lang {
            WetextLang::Zh => &fsts.zh_tagger,
            WetextLang::En => &fsts.en_tagger,
        };
        let tagged = tagger.normalize(prepared)?.trim().to_string();
        // reorder
        let reordered = token_reorder(&tagged, lang)?;
        // verbalize（.strip()）
        let verbalizer = match lang {
            WetextLang::Zh => &fsts.zh_verbalizer,
            WetextLang::En => &fsts.en_verbalizer,
        };
        verbalizer.normalize(&reordered)?.trim().to_string()
    } else {
        prepared.to_string()
    };
    // postprocess（无后处理 FST）：strip
    Ok(output.trim().to_string())
}

// ---------------------------------------------------------------------------
// 4. 参考实现包装层（frontend.py::TextNormalizer）
// ---------------------------------------------------------------------------

/// Python `PINYIN_TONE_PATTERN`（IGNORECASE；`(?<![a-z])` 在 IGNORECASE 下
/// 同样屏蔽大写前缀字符，fancy-regex 语义一致）。
const PINYIN_TONE_PATTERN: &str = "(?i)(?<![a-z])((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])";

/// Python `NAME_PATTERN`。
const NAME_PATTERN: &str = "(?i)[\u{4e00}-\u{9fff}]+(?:[-·—][\u{4e00}-\u{9fff}]+){1,2}";

/// Python `TECH_TERM_PATTERN`。
const TECH_TERM_PATTERN: &str = "[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+";

/// Python `ENGLISH_CONTRACTION_PATTERN`（IGNORECASE，替换为 `\1 is`）。
const ENGLISH_CONTRACTION_PATTERN: &str = "(?i)(what|where|who|which|how|t?here|it|s?he|that|this)'s";

/// Python `match_email`（re.match = 锚定开头，配合 `$` 即整串匹配）。
const EMAIL_PATTERN: &str = "^[a-zA-Z0-9]+@[a-zA-Z0-9]+\\.[a-zA-Z]+$";

struct WetextFsts {
    zh_tagger: FstTextNormalizer,
    zh_verbalizer: FstTextNormalizer,
    en_tagger: FstTextNormalizer,
    en_verbalizer: FstTextNormalizer,
}

impl WetextFsts {
    fn load() -> Result<Self> {
        let dir = materialize_fsts()?;
        Ok(Self {
            zh_tagger: FstTextNormalizer::load(&dir.join("zh/tn/tagger.fst"))?,
            zh_verbalizer: FstTextNormalizer::load(&dir.join("zh/tn/verbalizer.fst"))?,
            en_tagger: FstTextNormalizer::load(&dir.join("en/tn/tagger.fst"))?,
            en_verbalizer: FstTextNormalizer::load(&dir.join("en/tn/verbalizer.fst"))?,
        })
    }
}

/// IndexTTS-2.5 参考实现 `TextNormalizer` 的 Rust 移植（术语表不支持，恒空）。
pub struct TextNormalizer {
    fsts: Arc<WetextFsts>,
    zh_char_map_pattern: regex::Regex,
    zh_char_map: HashMap<&'static str, &'static str>,
    char_map_pattern: regex::Regex,
    char_map: HashMap<&'static str, &'static str>,
    pinyin_pattern: fancy_regex::Regex,
    name_pattern: fancy_regex::Regex,
    tech_pattern: regex::Regex,
    contraction_pattern: fancy_regex::Regex,
    email_pattern: regex::Regex,
}

impl TextNormalizer {
    /// 构建并加载内嵌 FST（约 13.6MB，冷加载 ~1s）。
    pub fn new() -> Result<Self> {
        let fsts = Arc::new(WetextFsts::load()?);
        // ZH_CHAR_REP_MAP = {"$": ".", **CHAR_REP_MAP} —— "$" 在交替序首位
        let mut zh_keys: Vec<&'static str> = vec!["$"];
        let mut zh_map: HashMap<&'static str, &'static str> = HashMap::new();
        zh_map.insert("$", ".");
        for (k, v) in CHAR_REP_MAP {
            zh_keys.push(k);
            zh_map.insert(k, v);
        }
        let mut keys: Vec<&'static str> = Vec::new();
        let mut map: HashMap<&'static str, &'static str> = HashMap::new();
        for (k, v) in CHAR_REP_MAP {
            keys.push(k);
            map.insert(k, v);
        }
        Ok(Self {
            fsts,
            zh_char_map_pattern: build_map_regex(&zh_keys)?,
            zh_char_map: zh_map,
            char_map_pattern: build_map_regex(&keys)?,
            char_map: map,
            pinyin_pattern: fancy_regex::Regex::new(PINYIN_TONE_PATTERN)
                .map_err(|e| anyhow!("拼音声调正则编译失败: {e}"))?,
            name_pattern: fancy_regex::Regex::new(NAME_PATTERN)
                .map_err(|e| anyhow!("人名正则编译失败: {e}"))?,
            tech_pattern: regex::Regex::new(TECH_TERM_PATTERN)
                .map_err(|e| anyhow!("术语正则编译失败: {e}"))?,
            contraction_pattern: fancy_regex::Regex::new(ENGLISH_CONTRACTION_PATTERN)
                .map_err(|e| anyhow!("收缩式正则编译失败: {e}"))?,
            email_pattern: regex::Regex::new(EMAIL_PATTERN)
                .map_err(|e| anyhow!("邮箱正则编译失败: {e}"))?,
        })
    }

    /// Python `use_chinese`。
    fn use_chinese(&self, s: &str) -> bool {
        let has_chinese = s.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c));
        let has_alpha = s.chars().any(|c| c.is_ascii_alphabetic());
        let is_email = self.email_pattern.is_match(s);
        if has_chinese || !has_alpha || is_email {
            return true;
        }
        matches!(self.pinyin_pattern.is_match(s), Ok(true))
    }

    /// Python `correct_pinyin`。
    fn correct_pinyin(&self, pinyin: &str) -> String {
        let Some(first) = pinyin.chars().next() else {
            return pinyin.to_string();
        };
        if !"jqxJQX".contains(first) {
            return pinyin.to_string();
        }
        let re = fancy_regex::Regex::new("(?i)([jqx])[uü](n|e|an)*(\\d)").expect("correct_pinyin 正则");
        let replaced = re
            .replace_all(pinyin, "${1}v${2}${3}")
            .into_owned();
        replaced.to_uppercase()
    }

    /// Python `save_tech_terms`：术语命中后把 `-` 改写为 `<H>` 防止 FST 误归一。
    fn save_tech_terms(&self, text: &str) -> (String, bool) {
        let mut terms: Vec<&str> = self.tech_pattern.find_iter(text).map(|m| m.as_str()).collect();
        if terms.is_empty() {
            return (text.to_string(), false);
        }
        // Python: sorted(set(list), key=len, reverse=True)；此处按长度降序 +
        // 首现序破平（set 顺序不可复现，恢复用同一折叠正则与编号无关）
        terms.sort_by_key(|t| std::cmp::Reverse(t.len()));
        terms.dedup();
        let mut transformed = text.to_string();
        for term in &terms {
            transformed = transformed.replace(term, &term.replace('-', "<H>"));
        }
        (transformed, true)
    }

    /// Python `save_pinyin_tones`（占位符按首次出现顺序编号）。
    fn save_pinyin_tones(&self, text: &str) -> Result<(String, Vec<String>)> {
        let mut found: Vec<String> = Vec::new();
        for caps in self.pinyin_pattern.captures_iter(text) {
            let caps = caps.map_err(|e| anyhow!("拼音正则执行失败: {e}"))?;
            // Python findall 返回两组拼接（"".join(p)）
            let joined: String = (1..caps.len())
                .filter_map(|i| caps.get(i))
                .map(|m| m.as_str())
                .collect();
            if !found.contains(&joined) {
                found.push(joined);
            }
        }
        if found.is_empty() {
            return Ok((text.to_string(), Vec::new()));
        }
        let mut transformed = text.to_string();
        for (i, pinyin) in found.iter().enumerate() {
            let number = char::from(b'a' + i as u8);
            transformed = transformed.replace(pinyin.as_str(), &format!("<pinyin_{number}>"));
        }
        Ok((transformed, found))
    }

    /// Python `restore_pinyin_tones`（恢复时应用 correct_pinyin）。
    fn restore_pinyin_tones(&self, text: &str, pinyins: &[String]) -> String {
        if pinyins.is_empty() {
            return text.to_string();
        }
        let mut transformed = text.to_string();
        for (i, pinyin) in pinyins.iter().enumerate() {
            let number = char::from(b'a' + i as u8);
            let corrected = self.correct_pinyin(pinyin);
            transformed = transformed.replace(&format!("<pinyin_{number}>"), &corrected);
        }
        transformed
    }

    /// Python `save_names`（占位符按首次出现顺序编号）。
    fn save_names(&self, text: &str) -> Result<(String, Vec<String>)> {
        let mut found: Vec<String> = Vec::new();
        for caps in self.name_pattern.captures_iter(text) {
            let caps = caps.map_err(|e| anyhow!("人名正则执行失败: {e}"))?;
            let whole = caps.get(0).expect("whole match").as_str().to_string();
            if !found.contains(&whole) {
                found.push(whole);
            }
        }
        if found.is_empty() {
            return Ok((text.to_string(), Vec::new()));
        }
        let mut transformed = text.to_string();
        for (i, name) in found.iter().enumerate() {
            let number = char::from(b'a' + i as u8);
            transformed = transformed.replace(name.as_str(), &format!("<n_{number}>"));
        }
        Ok((transformed, found))
    }

    /// Python `restore_names`。
    fn restore_names(&self, text: &str, names: &[String]) -> String {
        if names.is_empty() {
            return text.to_string();
        }
        let mut transformed = text.to_string();
        for (i, name) in names.iter().enumerate() {
            let number = char::from(b'a' + i as u8);
            transformed = transformed.replace(&format!("<n_{number}>"), name);
        }
        transformed
    }

    /// Python `restore_tech_terms`：忽略原列表，统一折叠 `\s*<H>\s*` → `-`。
    fn restore_tech_terms(&self, text: &str) -> String {
        let re = regex::Regex::new(r"\s*<H>\s*").expect("<H> 折叠正则");
        re.replace_all(text, "-").into_owned()
    }

    fn map_sub(
        &self,
        pattern: &regex::Regex,
        map: &HashMap<&'static str, &'static str>,
        text: &str,
    ) -> String {
        pattern
            .replace_all(text, |caps: &regex::Captures| {
                let m = caps.get(0).expect("whole match").as_str();
                map.get(m).copied().unwrap_or(m).to_string()
            })
            .into_owned()
    }

    /// 全流程归一化（等价 `TextNormalizer.normalize`；术语表恒空故略去）。
    pub fn normalize(&self, text: &str) -> Result<String> {
        if self.use_chinese(text) {
            let text = self
                .contraction_pattern
                .replace_all(text, "${1} is")
                .into_owned();
            // zh 分支：rstrip()（仅右侧空白）
            let (replaced, has_tech) = self.save_tech_terms(text.trim_end());
            let (replaced, pinyins) = self.save_pinyin_tones(&replaced)?;
            let (replaced, names) = self.save_names(&replaced)?;
            let mut result = wetext_tn_normalize(&self.fsts, &replaced, WetextLang::Zh)?;
            result = self.restore_names(&result, &names);
            result = self.restore_pinyin_tones(&result, &pinyins);
            if has_tech {
                result = self.restore_tech_terms(&result);
            }
            Ok(self.map_sub(&self.zh_char_map_pattern, &self.zh_char_map, &result))
        } else {
            // en 分支：FST 异常回退到收缩式展开后的文本（try/except 同语义）
            let contracted = self
                .contraction_pattern
                .replace_all(text, "${1} is")
                .into_owned();
            let (replaced, has_tech) = self.save_tech_terms(&contracted);
            let result = match wetext_tn_normalize(&self.fsts, &replaced, WetextLang::En) {
                Ok(r) => {
                    if has_tech {
                        self.restore_tech_terms(&r)
                    } else {
                        r
                    }
                }
                Err(_) => contracted,
            };
            Ok(self.map_sub(&self.char_map_pattern, &self.char_map, &result))
        }
    }
}

fn build_map_regex(keys: &[&'static str]) -> Result<regex::Regex> {
    let pattern = keys
        .iter()
        .map(|k| regex::escape(k))
        .collect::<Vec<_>>()
        .join("|");
    regex::Regex::new(&pattern).map_err(|e| anyhow!("映射正则编译失败: {e}"))
}

// ---------------------------------------------------------------------------
// 5. 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// zh 门控：无数字不触发 FST，但走 zh CHAR_REP_MAP 二次映射（含 `$`→`.`）。
    #[test]
    fn zh_无数字文本_透传加映射() {
        let n = TextNormalizer::new().unwrap();
        assert_eq!(
            n.normalize("今日天气几好，我哋一齐去饮茶啦。").unwrap(),
            "今日天气几好,我哋一齐去饮茶啦."
        );
        assert_eq!(n.normalize("你好……再见").unwrap(), "你好…再见");
        assert_eq!(
            n.normalize("价格$一百").unwrap(),
            "价格.一百",
            "zh 无数字时 $ 经 ZH_CHAR_REP_MAP 映射为句点"
        );
    }

    #[test]
    fn use_chinese_判定() {
        let n = TextNormalizer::new().unwrap();
        assert!(n.use_chinese("你好"));
        assert!(!n.use_chinese("abc123"), "纯 ASCII 字母数字无拼音声调 → en");
        assert!(!n.use_chinese("hello world"));
        assert!(n.use_chinese("user@example.com"), "邮箱 → zh");
        assert!(n.use_chinese("peng2"), "拼音声调 → zh");
        assert!(n.use_chinese("LV3"));
        // 无字母（纯数字/标点）→ zh（not has_alpha）
        assert!(n.use_chinese("123,456"));
    }

    #[test]
    fn 收缩式展开() {
        let n = TextNormalizer::new().unwrap();
        let out = n.normalize("what's the time?").unwrap();
        assert!(out.contains("what is"), "收缩式必须展开: {out}");
    }

    #[test]
    fn correct_pinyin_语义() {
        let n = TextNormalizer::new().unwrap();
        assert_eq!(n.correct_pinyin("ju4"), "JV4");
        assert_eq!(n.correct_pinyin("juan4"), "JVAN4");
        assert_eq!(n.correct_pinyin("lv3"), "lv3", "非 jqx 声母不动");
        assert_eq!(n.correct_pinyin("pin1"), "pin1");
    }

    /// zh 数字归一冒烟（真实 FST）。
    #[test]
    fn zh_数字归一化冒烟() {
        let n = TextNormalizer::new().unwrap();
        let out = n.normalize("今天2024年1月15日").unwrap();
        eprintln!("2024年1月15日 → {out}");
        assert!(
            !out.chars().any(|c| c.is_ascii_digit()),
            "阿拉伯数字应被展开: {out}"
        );
    }

    #[test]
    fn en_数字归一化冒烟() {
        let n = TextNormalizer::new().unwrap();
        let out = n.normalize("I have 3 apples and $100").unwrap();
        eprintln!("en → {out}");
        assert!(
            !out.chars().any(|c| c.is_ascii_digit()),
            "阿拉伯数字应被展开: {out}"
        );
    }

    /// 金标对拍：`tmp/make_normalizer_refs.py` 生成（Python wetext 0.1.8 +
    /// 参考包装层），逐例精确相等。金标缺失时跳过（refs 不入库）。
    #[test]
    fn normalizer_与python金标逐例一致() {
        let root = crate::shared::workspace_paths::WorkspacePaths::workspace_root();
        let path = root
            .join("tmp")
            .join("indextts25_refs")
            .join("normalizer_golden.json");
        let Ok(data) = std::fs::read_to_string(&path) else {
            eprintln!("跳过：金标不存在（{:?}）", path);
            return;
        };
        #[derive(serde::Deserialize)]
        struct Case {
            text: String,
            out: String,
        }
        let cases: Vec<Case> = serde_json::from_str(&data).expect("金标 JSON 解析");
        let n = TextNormalizer::new().unwrap();
        let mut failed = 0usize;
        for case in &cases {
            let got = n
                .normalize(&case.text)
                .unwrap_or_else(|e| panic!("用例 {:?} 归一化失败: {e}", case.text));
            if got != case.out {
                failed += 1;
                eprintln!(
                    "不一致:\n  text: {:?}\n  want: {:?}\n  got:  {:?}",
                    case.text, case.out, got
                );
            }
        }
        assert_eq!(failed, 0, "{failed}/{cases_len} 例不一致", cases_len = cases.len());
        println!("normalizer 金标 {} 例全对", cases.len());
    }
}
