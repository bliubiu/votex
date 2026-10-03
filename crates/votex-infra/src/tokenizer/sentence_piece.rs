//! SentencePiece BPE 分词器
//!
//! 基于 `bpe_tokenizer.rs` 中的通用 SentencePiece BPE 实现，
//! 适配 `TextTokenizer` trait，用于 IndexTTS2、翻译模型等。

use std::path::Path;
use std::sync::Arc;

use votex_domain::tts::tokenizer::{TextTokenizer, TokenizerError};

use crate::bpe_tokenizer::SentencePieceBpe;

/// SentencePiece BPE 分词器
///
/// 从 SentencePiece 的 `.model` 文件加载词汇表，
/// 实现基于字节的 BPE 编码/解码。
///
/// # 示例
///
/// ```ignore
/// let tokenizer = SentencePieceBpeTokenizer::load("models/indextts2/bpe.model")?;
/// let ids = tokenizer.encode("你好世界")?;
/// ```
pub struct SentencePieceBpeTokenizer {
    inner: Arc<SentencePieceBpe>,
    vocab_size: usize,
}

impl SentencePieceBpeTokenizer {
    /// 从 `.model` 文件加载 SentencePiece 模型
    pub fn load(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let inner = SentencePieceBpe::load(path.as_ref())
            .map_err(|e| TokenizerError::VocabLoadFailed(e.to_string()))?;
        let vocab_size = inner.vocab_size();
        Ok(Self {
            inner: Arc::new(inner),
            vocab_size,
        })
    }
}

impl TextTokenizer for SentencePieceBpeTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<i64>, TokenizerError> {
        self.inner
            .encode(text)
            .map_err(|e| TokenizerError::EncodeFailed(e.to_string()))
    }

    fn decode(&self, ids: &[i64]) -> Result<String, TokenizerError> {
        let u32_ids: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
        self.inner
            .decode_piece_ids(&u32_ids)
            .map_err(|e| TokenizerError::DecodeFailed(e.to_string()))
    }

    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
