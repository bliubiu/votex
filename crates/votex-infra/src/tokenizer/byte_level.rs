//! ByteLevel BPE 分词器（GPT-2 / Qwen2 风格）
//!
//! 基于 HuggingFace `tokenizers` crate，使用 ByteLevel BPE 算法。
//! 用于 CosyVoice3（基于 Qwen2-0.5B 的 text embedding 层）。
//!
//! # Qwen2 官方配置
//!
//! 与官方 `Qwen/Qwen2-0.5B` 的 `AutoTokenizer.from_pretrained()` 保持一致：
//! - ByteLevel pre-tokenizer: `add_prefix_space=true, trim_offsets=false, use_regex=true`
//! - 从 `tokenizer.json` 加载（包含完整配置），优先于手写参数

use std::path::Path;
use std::sync::Arc;

use votex_domain::tts::tokenizer::{TextTokenizer, TokenizerError};

/// ByteLevel BPE 分词器
///
/// 使用 HuggingFace `tokenizers` crate，支持：
/// - 从 `tokenizer.json` 加载（官方推荐，配置完整）
/// - 从 `vocab.json` + `merges.txt` + 手动配置加载（兼容旧版）
///
/// # 线程安全
///
/// 内部使用 `tokenizers::Tokenizer`（Arc 包装），支持多线程并发。
pub struct ByteLevelBpeTokenizer {
    inner: Arc<tokenizers::Tokenizer>,
    vocab_size: usize,
}

impl ByteLevelBpeTokenizer {
    /// 从 `tokenizer.json` 加载完整的 ByteLevel BPE 分词器（推荐）
    ///
    /// 这是与官方 HuggingFace tokenizer 完全一致的标准加载方式，
    /// 配置（pre-tokenizer/decoder/post-processor）已固化在文件中。
    ///
    /// # 参数
    /// - `path`: `tokenizer.json` 文件路径
    pub fn from_tokenizer_json(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let path = path.as_ref();
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| {
                TokenizerError::VocabLoadFailed(format!("加载 tokenizer.json 失败 ({}): {}", path.display(), e))
            })?;
        let vocab_size = inner.get_vocab_size(false);
        Ok(Self {
            inner: Arc::new(inner),
            vocab_size,
        })
    }

    /// 从 `vocab.json` + `merges.txt` 手动构建 ByteLevel BPE 分词器
    ///
    /// 仅在 `tokenizer.json` 不存在时使用。
    /// Qwen2 官方参数：`add_prefix_space=false, trim_offsets=false, use_regex=true`
    /// （add_prefix_space=false 对齐 transformers GPT2Tokenizer 默认行为：
    /// 加前导空格会把首字节与 Ġ 合并，如「收」被拆成 ĠæĶ+¶，产生分词偏差）
    ///
    /// # 参数
    /// - `vocab_path`: `vocab.json` 文件路径
    /// - `merges_path`: `merges.txt` 文件路径
    pub fn from_vocab_merges(
        vocab_path: impl AsRef<Path>,
        merges_path: impl AsRef<Path>,
    ) -> Result<Self, TokenizerError> {
        let vocab_str = vocab_path
            .as_ref()
            .to_str()
            .ok_or_else(|| TokenizerError::VocabLoadFailed("vocab.json 路径无效".into()))?;
        let merges_str = merges_path
            .as_ref()
            .to_str()
            .ok_or_else(|| TokenizerError::VocabLoadFailed("merges.txt 路径无效".into()))?;

        let bpe = tokenizers::models::bpe::BPE::from_file(vocab_str, merges_str)
            .build()
            .map_err(|e| TokenizerError::VocabLoadFailed(format!("加载 BPE 模型失败: {}", e)))?;

        let mut inner = tokenizers::Tokenizer::new(bpe);

        // Qwen2 官方 ByteLevel 配置（三个布尔参数依次为 add_prefix_space / trim_offsets / use_regex）：
        // add_prefix_space=false: **不加**前导空格，与 transformers `GPT2Tokenizer` 默认行为对齐
        //   （加前导空格会把首字节与 Ġ 合并，如「收」被拆成 ĠæĶ+¶，产生分词偏差）
        // trim_offsets=false: 不裁剪偏移量
        // use_regex=true: 使用 GPT-2 风格的正则 pre-tokenization
        let byte_level = tokenizers::pre_tokenizers::byte_level::ByteLevel::new(false, false, true);
        inner.with_pre_tokenizer(Some(byte_level.clone()));
        inner.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::new(
            false, false, true,
        )));
        inner.with_post_processor(Some(tokenizers::processors::byte_level::ByteLevel::new(
            false, false, true,
        )));

        let vocab_size = inner.get_vocab_size(false);
        Ok(Self {
            inner: Arc::new(inner),
            vocab_size,
        })
    }
}

impl TextTokenizer for ByteLevelBpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i64>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| TokenizerError::EncodeFailed(e.to_string()))?;
        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
    }

    fn decode(&self, ids: &[i64]) -> Result<String, TokenizerError> {
        let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner
            .decode(&u32_ids, false)
            .map_err(|e| TokenizerError::DecodeFailed(e.to_string()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 定位 CosyVoice 模型目录（仅测试用；模型缺失则跳过）
    fn cosyvoice_dir() -> std::path::PathBuf {
        crate::shared::WorkspacePaths::models_dir()
            .join("tts")
            .join("cosyvoice")
    }

    /// docs/20 F67 取证：CosyVoice3 对任意输入文本都产出 prompt 文本的音频
    /// （7 次不同文本的 LLM 初始 logits 完全一致），因此必须先排除
    /// 「分词器把不同文本压成近似 token 序列」这一候选根因。
    ///
    /// 判据（客观、不依赖官方 token id 记忆）：
    /// ① 往返一致：decode(encode(s)) == s
    /// ② 两段差异极大的中文文本，其 token 序列必须有实质差异
    /// ③ 分词粒度合理：远少于「逐字符 1 token」
    #[test]
    fn docs20_f67_中文分词往返一致且区分不同文本() {
        let dir = cosyvoice_dir();
        let (vocab, merges) = (dir.join("vocab.json"), dir.join("merges.txt"));
        if !vocab.is_file() || !merges.is_file() {
            eprintln!("跳过：CosyVoice 分词器文件不在 {}", dir.display());
            return;
        }

        let tok = ByteLevelBpeTokenizer::from_vocab_merges(&vocab, &merges)
            .expect("加载 CosyVoice 分词器失败");

        let a = "夜色落下，林远推开了那扇木门。";
        let b = "请用普通话朗读这段完全不同的技术文档。";

        let ids_a = tok.encode(a).expect("encode a 失败");
        let ids_b = tok.encode(b).expect("encode b 失败");

        // ① 往返一致
        let back_a = tok.decode(&ids_a).expect("decode a 失败");
        let back_b = tok.decode(&ids_b).expect("decode b 失败");
        assert_eq!(
            back_a.trim(),
            a.trim(),
            "中文分词往返不一致，encode/decode 有损 —— 分词器配置错误"
        );
        assert_eq!(back_b.trim(), b.trim(), "中文分词往返不一致");

        // ② 两段文本的 token 序列必须实质不同
        assert_ne!(
            ids_a, ids_b,
            "两段差异极大的中文文本得到完全相同的 token 序列 —— 分词器失效"
        );

        // ③ 粒度：不应逐字符成 token（留 4 倍余量，只捕捉退化情形）
        let len_a = a.chars().filter(|c| !c.is_whitespace()).count();
        assert!(
            ids_a.len() < len_a * 4,
            "分词粒度异常：{} 字文本产生 {} 个 token",
            len_a,
            ids_a.len()
        );

        eprintln!(
            "[F67 取证] '{}' -> {} tokens; '{}' -> {} tokens; vocab_size={}",
            a,
            ids_a.len(),
            b,
            ids_b.len(),
            tok.vocab_size()
        );
    }
}
