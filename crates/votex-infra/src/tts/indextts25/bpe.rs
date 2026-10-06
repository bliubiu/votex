//! IndexTTS-2.5 tiktoken BPE 编码器（Rust 移植）。
//!
//! 对照蓝本：`index_tts_2_5_onnx/core/frontend.py::get_encoding`（tiktoken.Encoding），
//! 逐级对拍目标见 `docs/25-IndexTTS25-Rust移植蓝本.md` §2/§7。
//!
//! 与参考实现的有意差异：
//! - 参考实现 `get_encoding(num_languages=99)` 存在 off-by-one 缺陷：语言表共 106 项，
//!   截取前 99 项止于 `su`，导致 `<|yue|>` 等方言 token 未注册、被当作普通文本念出
//!   （实测听感为音频开头多念「粤语」二字）。本实现注册 **100** 种语言（en..yue），
//!   `<|yue|>` id = 58937，与 GPT 文本嵌入表 [60510, 1280] 严格吻合。
//! - 强制断言 `n_vocab == 60510`：嵌入表行数是硬性契约，minnan/wuyu 等 6 项语言
//!   在嵌入表之外，不可注册（越界）。

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;

/// 语言表（顺序即 LANGUAGE_DICT 序号，106 项；与参考实现 frontend.py 逐项一致）。
pub const LANGUAGES: [&str; 106] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
    "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
    "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
    "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su", "yue", "minnan", "wuyu", "dialect", "zh/en", "en/zh",
    "common",
];

/// 注册的语言数：修正参考实现 `num_languages=99` 的 off-by-one（en..yue 恰为前 100 项）。
pub const N_REGISTERED_LANGUAGES: usize = 100;

/// 词表规模硬性契约：mergeable ranks(58836) + special tokens(1674) 必须与
/// GPT 文本嵌入表行数 [60510] 严格一致。
pub const EXPECTED_N_VOCAB: usize = 60510;

/// 音频事件标签（11 项）。
const AUDIO_EVENT: [&str; 11] = [
    "ASR", "AED", "SER", "Speech", "/Speech", "BGM", "/BGM", "Laughter",
    "/Laughter", "Applause", "/Applause",
];

/// 情绪标签（4 项）。
const EMOTION: [&str; 4] = ["HAPPY", "SAD", "ANGRY", "NEUTRAL"];

/// TTS 发音控制标签（7 项固定 + SP01..SP13）。
const TTS_VOCAL: [&str; 20] = [
    "TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H",
    "TTS/SP01", "TTS/SP02", "TTS/SP03", "TTS/SP04", "TTS/SP05", "TTS/SP06",
    "TTS/SP07", "TTS/SP08", "TTS/SP09", "TTS/SP10", "TTS/SP11", "TTS/SP12",
    "TTS/SP13",
];

/// GPT-2/o200k 风格预分词模式（与参考实现 PAT_STR 逐字符一致）。
/// 含 `(?!\S)` 负向前瞻，regex crate 不支持，故使用 fancy-regex。
const PAT_STR: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

/// tiktoken 编码器：mergeable ranks + special tokens + 预分词正则。
pub struct Tiktoken {
    ranks: HashMap<Vec<u8>, u32>,
    special_tokens: HashMap<String, u32>,
    pretokenize: fancy_regex::Regex,
}

impl Tiktoken {
    /// 从词表文件加载（行格式：`base64 token + 空白 + rank`）。
    pub fn load(vocab_path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(vocab_path)
            .with_context(|| format!("读取 tiktoken 词表失败: {}", vocab_path.display()))?;
        let mut ranks: HashMap<Vec<u8>, u32> = HashMap::with_capacity(EXPECTED_N_VOCAB);
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut fields = line.split_whitespace();
            let (tok_b64, rank_str) = match (fields.next(), fields.next()) {
                (Some(t), Some(r)) => (t, r),
                _ => bail!("词表第 {} 行格式非法: {line}", lineno + 1),
            };
            let token = lenient_b64_decode(tok_b64, lineno + 1)?;
            let rank: u32 = rank_str
                .parse()
                .with_context(|| format!("词表第 {} 行 rank 非法: {rank_str}", lineno + 1))?;
            ranks.insert(token, rank);
        }
        let encoder = Self::from_parts(ranks, build_special_token_list())?;
        if encoder.n_vocab() != EXPECTED_N_VOCAB {
            bail!(
                "词表规模不符：实际 n_vocab={}（mergeable={} + special={}），期望 {EXPECTED_N_VOCAB}（GPT 文本嵌入表行数）",
                encoder.n_vocab(),
                encoder.ranks.len(),
                encoder.special_tokens.len(),
            );
        }
        Ok(encoder)
    }

    /// 从 ranks 与 special token 列表构建（special id 按列表顺序从 ranks 之后分配）。
    /// 供测试注入合成词表；`load` 负责真实词表的规模契约校验。
    pub fn from_parts(ranks: HashMap<Vec<u8>, u32>, specials: Vec<String>) -> Result<Self> {
        // BPE 前置条件：256 个单字节必须全部在词表中（tiktoken 词表的通用保证）
        for b in 0..=255u8 {
            if !ranks.contains_key(&[b][..]) {
                bail!("词表缺少单字节 token 0x{b:02X}，无法进行 BPE 编码");
            }
        }
        let mut special_tokens = HashMap::with_capacity(specials.len());
        let next_id = ranks.len() as u32;
        for (id, special) in (next_id..).zip(specials) {
            if special_tokens.insert(special.clone(), id).is_some() {
                bail!("special token 重复: {special}");
            }
        }
        let pretokenize = fancy_regex::Regex::new(PAT_STR)
            .map_err(|e| anyhow!("预分词正则编译失败: {e}"))?;
        Ok(Self {
            ranks,
            special_tokens,
            pretokenize,
        })
    }

    /// 词表总规模（mergeable + special）。
    pub fn n_vocab(&self) -> usize {
        self.ranks.len() + self.special_tokens.len()
    }

    /// 查询 special token 的 id（如 `<|yue|>` -> 58937）。
    pub fn special_id(&self, token: &str) -> Option<u32> {
        self.special_tokens.get(token).copied()
    }

    /// 编码（等价 Python `encode(text, allowed_special='all')`：special token 全部生效）。
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        let mut rest = text;
        // special token 一律形如 `<|...|>`：先切出 special，再对间隙做普通编码
        while let Some((pos, len, id)) = self.find_next_special(rest) {
            self.encode_ordinary_into(&rest[..pos], &mut out)?;
            out.push(id);
            rest = &rest[pos + len..];
        }
        self.encode_ordinary_into(rest, &mut out)?;
        Ok(out)
    }

    /// 普通编码（忽略 special token，等价 `encode_ordinary`）。
    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        self.encode_ordinary_into(text, &mut out)?;
        Ok(out)
    }

    fn encode_ordinary_into(&self, text: &str, out: &mut Vec<u32>) -> Result<()> {
        for mat in self.pretokenize.find_iter(text) {
            let m = mat.map_err(|e| anyhow!("预分词正则执行失败: {e}"))?;
            let piece = m.as_str().as_bytes();
            match self.ranks.get(piece) {
                Some(&rank) => out.push(rank),
                None => self.byte_pair_encode(piece, out)?,
            }
        }
        Ok(())
    }

    /// tiktoken `_byte_pair_merge` 的忠实移植：贪心合并当前 rank 最小的相邻对
    /// （同 rank 取最左，rank 即合并结果 token id）。
    fn byte_pair_encode(&self, piece: &[u8], out: &mut Vec<u32>) -> Result<()> {
        debug_assert!(piece.len() >= 2, "单字节 piece 必然命中 ranks（加载时已校验）");
        if piece.len() < 2 {
            bail!("BPE 输入异常：piece 长度 {} < 2", piece.len());
        }
        // parts[i] = (字节起点, 该段与下一段合并后的 rank；MAX 表示不可合并)
        let mut parts: Vec<(usize, u32)> = Vec::with_capacity(piece.len() + 1);
        let mut min_rank = (u32::MAX, usize::MAX);
        for i in 0..piece.len() - 1 {
            let rank = self
                .ranks
                .get(&piece[i..i + 2])
                .copied()
                .unwrap_or(u32::MAX);
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
            parts.push((i, rank));
        }
        parts.push((piece.len() - 1, u32::MAX));
        parts.push((piece.len(), u32::MAX));

        let get_rank = |parts: &[(usize, u32)], i: usize| -> u32 {
            if i + 3 < parts.len() {
                self.ranks
                    .get(&piece[parts[i].0..parts[i + 3].0])
                    .copied()
                    .unwrap_or(u32::MAX)
            } else {
                u32::MAX
            }
        };

        while min_rank.0 != u32::MAX {
            let i = min_rank.1;
            if i > 0 {
                parts[i - 1].1 = get_rank(&parts, i - 1);
            }
            parts[i].1 = get_rank(&parts, i);
            parts.remove(i + 1);

            min_rank = (u32::MAX, usize::MAX);
            for (j, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
                if rank < min_rank.0 {
                    min_rank = (rank, j);
                }
            }
        }

        for w in parts.windows(2) {
            let rank = self
                .ranks
                .get(&piece[w[0].0..w[1].0])
                .copied()
                .ok_or_else(|| anyhow!("BPE 合并结果缺失 rank（词表损坏？）"))?;
            out.push(rank);
        }
        Ok(())
    }

    /// 在文本中找最早的 special token 出现位置 `(起点, 字节长, id)`。
    ///
    /// 本编码器的 special token 全部形如 `<|...|>`（内部不含 `>`），因此按 `<|`
    /// 前缀定位 + 截取到首个 `>` 查表即可，等价且远快于逐 token 全文查找
    /// （1674 个 special，token_len 在分段循环中被高频调用）。
    fn find_next_special(&self, text: &str) -> Option<(usize, usize, u32)> {
        let bytes = text.as_bytes();
        let mut i = 0;
        while i + 1 < bytes.len() {
            if bytes[i] == b'<' && bytes[i + 1] == b'|' {
                if let Some(close) = text[i..].find('>') {
                    let end = i + close + 1;
                    if let Some(&id) = self.special_tokens.get(&text[i..end]) {
                        return Some((i, end - i, id));
                    }
                    i += 1;
                } else {
                    break;
                }
            } else {
                i += 1;
            }
        }
        None
    }
}

/// tiktoken 词表的宽容 base64 解码（对齐 Python `base64.b64decode` 默认 validate=False 语义）。
///
/// 实测本词表（58836 行）仅有 1 个非规范行：第 48475 行 token 为裸 `"="`，
/// Python 返回空字节串（仅占 rank 计数；编码查找的 piece 永远非空，该键不会被命中，
/// 但必须保留以维持 n_vocab=60510 的规模契约）。除纯 padding 外的解码失败仍然报错。
fn lenient_b64_decode(token: &str, lineno: usize) -> Result<Vec<u8>> {
    // validate=False：先丢弃标准字母表与 '=' 之外的字符
    let cleaned: String = token
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '/' | '='))
        .collect();
    if cleaned.is_empty() {
        bail!("词表第 {lineno} 行 token 为空（无有效 base64 字符）");
    }
    match BASE64.decode(cleaned.as_bytes()) {
        Ok(bytes) => Ok(bytes),
        Err(_) if cleaned.bytes().all(|b| b == b'=') => {
            // Python binascii 宽容语义：纯 padding 解码为空字节串
            Ok(Vec::new())
        }
        Err(e) => bail!("词表第 {lineno} 行 base64 解码失败: {e}"),
    }
}

/// 构建 special token 列表（顺序决定 id 分配，须与参考实现逐项一致）：
/// 2 + 100 语言 + 11 音频事件 + 4 情绪 + 6 固定 + 30 SPECIAL_TOKEN + 20 TTS + 1501 时间戳 = 1674。
fn build_special_token_list() -> Vec<String> {
    let mut specials = Vec::with_capacity(2048);
    specials.push("<|endoftext|>".to_string());
    specials.push("<|startoftranscript|>".to_string());
    for lang in LANGUAGES.iter().take(N_REGISTERED_LANGUAGES) {
        specials.push(format!("<|{lang}|>"));
    }
    for ae in AUDIO_EVENT {
        specials.push(format!("<|{ae}|>"));
    }
    for e in EMOTION {
        specials.push(format!("<|{e}|>"));
    }
    for fixed in [
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ] {
        specials.push(fixed.to_string());
    }
    for i in 1..=30 {
        specials.push(format!("<|SPECIAL_TOKEN_{i}|>"));
    }
    for t in TTS_VOCAL {
        specials.push(format!("<|{t}|>"));
    }
    // Python `f"<|{i * 0.02:.2f}|>"`：两侧均为 IEEE f64 乘法 + 十进制正确舍入，结果一致
    for i in 0..1501 {
        specials.push(format!("<|{:.2}|>", i as f64 * 0.02));
    }
    specials
}

/// 语言 -> LANGUAGE_DICT 序号（gpt_prefill 的 `lang_id` 输入；未知语言回退 common）。
pub fn lang_id(lang: &str) -> u32 {
    let lang = lang.to_ascii_lowercase();
    match LANGUAGES.iter().position(|l| *l == lang) {
        Some(i) => i as u32,
        None => LANGUAGES
            .iter()
            .position(|l| *l == "common")
            .unwrap_or(0) as u32,
    }
}

/// 定位真实词表（仅测试用）：优先 registry 约定的模型目录，其次 HF 缓存（PoC 阶段落点）。
#[cfg(test)]
pub(crate) fn find_real_vocab() -> Option<std::path::PathBuf> {
    let name = "multilingual_zh_ja_yue_char_del.tiktoken";
    let snapshots = std::env::var_os("USERPROFILE").map(|p| {
        std::path::PathBuf::from(p)
            .join(".cache/huggingface/hub/models--yunfengwang--IndexTTS-2.5-onnx/snapshots")
    });
    let candidates = [
        Some(
            crate::shared::workspace_paths::WorkspacePaths::models_dir()
                .join("tts")
                .join("indextts25")
                .join(name),
        ),
        snapshots,
    ];
    for cand in candidates.into_iter().flatten() {
        if cand.is_file() {
            return Some(cand);
        }
        // 快照目录：遍历 <commit>/ 一层
        if cand.is_dir() {
            if let Ok(entries) = std::fs::read_dir(&cand) {
                for entry in entries.flatten() {
                    let p = entry.path().join(name);
                    if p.is_file() {
                        return Some(p);
                    }
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 合成词表：256 单字节 + ab=256 / bc=257 / abc=258
    fn synthetic_ranks() -> HashMap<Vec<u8>, u32> {
        let mut ranks = HashMap::new();
        for b in 0..=255u8 {
            ranks.insert(vec![b], b as u32);
        }
        ranks.insert(b"ab".to_vec(), 256);
        ranks.insert(b"bc".to_vec(), 257);
        ranks.insert(b"abc".to_vec(), 258);
        ranks
    }

    fn synthetic_encoder() -> Tiktoken {
        Tiktoken::from_parts(synthetic_ranks(), vec!["<|x|>".to_string()]).unwrap()
    }

    #[test]
    fn bpe合成词表_整体合并为最低rank链() {
        let enc = synthetic_encoder();
        // abc: ab(256) 优先于 bc(257) 合并，随后合并出 abc(258)
        assert_eq!(enc.encode("abc").unwrap(), vec![258]);
        // abcd: abc(258) + d
        assert_eq!(enc.encode("abcd").unwrap(), vec![258, 100]);
        // abab: 左优先合并 ab -> [256, 256]
        assert_eq!(enc.encode("abab").unwrap(), vec![256, 256]);
        // 单字节直查
        assert_eq!(enc.encode("a").unwrap(), vec![97]);
        // 整段命中 rank 时不做合并
        assert_eq!(enc.encode("bc").unwrap(), vec![257]);
    }

    #[test]
    fn bpe_special_token按字面切分() {
        let enc = synthetic_encoder();
        let x_id = enc.special_id("<|x|>").unwrap();
        assert_eq!(enc.encode("a<|x|>b").unwrap(), vec![97, x_id, 98]);
        // 未闭合的 <| 不是 special
        let ids = enc.encode("<|x").unwrap();
        assert!(!ids.contains(&x_id));
        // 未知 <|...|> 按普通文本编码
        assert!(!enc.encode("x<|nope|>y").unwrap().contains(&x_id));
    }

    #[test]
    fn bpe_special_id按列表顺序分配() {
        let enc = synthetic_encoder();
        assert_eq!(enc.special_id("<|x|>"), Some(259)); // 259 个 rank 之后
        assert_eq!(enc.special_id("<|nope|>"), None);
    }

    #[test]
    fn 语言id_序号与回退() {
        assert_eq!(lang_id("en"), 0);
        assert_eq!(lang_id("zh"), 1);
        assert_eq!(lang_id("yue"), 99);
        assert_eq!(lang_id("YUE"), 99, "大写输入应归一化");
        assert_eq!(lang_id("common"), 105);
        assert_eq!(lang_id("xx"), 105, "未知语言回退 common");
    }

    #[test]
    fn 真实词表_规模契约与关键special锚点() {
        let Some(vocab) = find_real_vocab() else {
            eprintln!("跳过：未找到真实 tiktoken 词表");
            return;
        };
        let enc = Tiktoken::load(&vocab).unwrap();
        assert_eq!(enc.n_vocab(), EXPECTED_N_VOCAB);
        assert_eq!(enc.special_id("<|endoftext|>"), Some(58836));
        assert_eq!(enc.special_id("<|startoftranscript|>"), Some(58837));
        assert_eq!(enc.special_id("<|en|>"), Some(58838));
        assert_eq!(enc.special_id("<|yue|>"), Some(58937), "100 语言修正的锚点");
        assert_eq!(enc.special_id("<|SPECIAL_TOKEN_1|>"), Some(58959));
        assert_eq!(enc.special_id("<|SPECIAL_TOKEN_2|>"), Some(58960));
        assert_eq!(enc.special_id("<|TTS/B|>"), Some(58989));
        assert_eq!(enc.special_id("<|TTS/SP01|>"), Some(58996));
        assert_eq!(enc.special_id("<|0.00|>"), Some(59009));
        assert_eq!(enc.special_id("<|0.02|>"), Some(59010));
        assert_eq!(enc.special_id("<|30.00|>"), Some(60509));
        // 越界语言不可注册（嵌入表之外）
        assert_eq!(enc.special_id("<|minnan|>"), None);
        assert_eq!(enc.special_id("<|common|>"), None);
    }

    /// Python 金标（tmp/bpe_golden.py 生成，tiktoken 0.14.0，num_languages=100）。
    #[test]
    fn 真实词表_与python金标逐id一致() {
        let Some(vocab) = find_real_vocab() else {
            eprintln!("跳过：未找到真实 tiktoken 词表");
            return;
        };
        let enc = Tiktoken::load(&vocab).unwrap();
        let cases: &[(&str, &[u32])] = &[
            ("你好世界", &[48934, 50371, 48721, 53743]),
            (
                "哈喽，今日天气几好！",
                &[49732, 49881, 58827, 48838, 51930, 50324, 52692, 49263, 50371, 58824],
            ),
            (
                "It's a test... 2023年10月5日, price: $12.50\nNew  line ",
                &[
                    3493, 311, 257, 1500, 485, 42832, 50937, 3254, 52075, 20, 51930, 11, 3194,
                    25, 1844, 4714, 13, 2789, 198, 17831, 220, 1620, 220,
                ],
            ),
            ("<|yue|> 哈喽", &[58937, 220, 49732, 49881]),
            ("a  b   c", &[64, 220, 272, 220, 220, 269]),
            (
                "<|SPECIAL_TOKEN_2|>SHENZHEN<|SPECIAL_TOKEN_2|>测试",
                &[58960, 16900, 2188, 57, 44476, 58960, 52879, 56443],
            ),
            ("ん hello", &[48609, 7627]),
            ("emoji 😀 mixed 42", &[35021, 3978, 20205, 222, 7351, 13733]),
            ("<|0.02|>", &[59010]),
            ("readme", &[2527, 1398]),
            (
                "广东话配音：今日气温二十六度。",
                &[
                    50947, 48727, 56451, 57200, 57967, 58829, 48838, 51930, 52692, 53003, 48793,
                    49462, 49188, 50969, 1542,
                ],
            ),
        ];
        for (text, expected) in cases {
            let actual = enc.encode(text).unwrap();
            assert_eq!(&actual, expected, "编码不一致: {text:?}");
        }
    }
}
