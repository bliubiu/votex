//! 单字查表分词器
//!
//! 基于 `char → token ID` 的简单哈希表查找，用于 Kokoro 等模型的音素编码。
//! 不涉及 BPE/子词算法，直接按字符映射。
//!
//! # 使用场景
//!
//! 对于输入已经是离散符号序列（如音素）的模型，无需 BPE 编码，
//! 只需将每个符号映射到对应的 token ID，并添加 BOS/EOS。

use std::collections::HashMap;
use std::sync::Arc;

use votex_domain::tts::tokenizer::{TextTokenizer, TokenizerError};

/// 单字查表分词器
///
/// 输入文本中的每个字符独立映射到词汇表中的 token ID。
/// 支持可选的 BOS（开头）和 EOS（结尾）特殊 token。
///
/// # 示例
///
/// ```ignore
/// let mut vocab = HashMap::new();
/// vocab.insert('a', 3);
/// vocab.insert('b', 4);
/// let tokenizer = CharLookupTokenizer::new(vocab, Some(0), Some(0));
/// let ids = tokenizer.encode("ab")?; // → [0, 3, 4, 0]
/// ```
#[derive(Debug)]
pub struct CharLookupTokenizer {
    /// 字符 → token ID 映射
    vocab: Arc<HashMap<char, i64>>,
    /// BOS token ID（`Some(id)` 启用，`None` 禁用）
    bos: Option<i64>,
    /// EOS token ID
    eos: Option<i64>,
    /// 未知字符的默认替换 ID
    unk_id: i64,
    vocab_size: usize,
}

impl CharLookupTokenizer {
    /// 创建新的单字查表分词器
    ///
    /// # 参数
    /// - `vocab`: 字符 → token ID 映射
    /// - `bos`: BOS token ID（可选）
    /// - `eos`: EOS token ID（可选）
    /// - `unk_id`: 未知字符的替换 ID（默认 0）
    pub fn new(
        vocab: HashMap<char, i64>,
        bos: Option<i64>,
        eos: Option<i64>,
    ) -> Self {
        let vocab_size = vocab.len()
            + bos.map(|_| 1).unwrap_or(0)
            + eos.map(|_| 1).unwrap_or(0);
        Self {
            vocab: Arc::new(vocab),
            bos,
            eos,
            unk_id: 0,
            vocab_size,
        }
    }

    /// 设置未知字符替换 ID
    pub fn with_unk(mut self, unk_id: i64) -> Self {
        self.unk_id = unk_id;
        self
    }

    /// 获取词汇表引用（方便外部访问）
    pub fn vocab(&self) -> &HashMap<char, i64> {
        &self.vocab
    }

    /// 获取 BOS ID
    pub fn bos_id(&self) -> Option<i64> {
        self.bos
    }

    /// 获取 EOS ID
    pub fn eos_id(&self) -> Option<i64> {
        self.eos
    }
}

impl TextTokenizer for CharLookupTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i64>, TokenizerError> {
        let mut ids = Vec::with_capacity(text.len() + 2);

        // BOS
        if let Some(bos) = self.bos {
            ids.push(bos);
        }

        // 逐字符映射
        for c in text.chars() {
            match self.vocab.get(&c) {
                Some(&token_id) => ids.push(token_id),
                None => ids.push(self.unk_id),
            }
        }

        // EOS
        if let Some(eos) = self.eos {
            ids.push(eos);
        }

        Ok(ids)
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
