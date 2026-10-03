//! 分词器抽象接口
//!
//! 定义统一的 `TextTokenizer` trait，所有模型的分词器通过此接口实现。
//! 遵循 DDD 分层：接口定义在 domain 层，实现在 infra 层。
//!
//! # 设计思路
//!
//! 不同模型使用不同分词方案（SentencePiece BPE、ByteLevel BPE、字符查表等），
//! 但它们都遵循相同的使用模式：文本 → token ID。统一 trait 后：
//!
//! - 各 Provider 无需关心分词实现细节
//! - 新增模型只需实现 trait，无需从零搭建
//! - 可替换实现（如用 `tokenizers` crate 替换手写 BPE）

use thiserror::Error;

/// 分词器错误
#[derive(Debug, Error)]
pub enum TokenizerError {
    /// 编码失败
    #[error("编码失败: {0}")]
    EncodeFailed(String),

    /// 解码失败
    #[error("解码失败: {0}")]
    DecodeFailed(String),

    /// 词汇表加载失败
    #[error("词汇表加载失败: {0}")]
    VocabLoadFailed(String),

    /// 词汇表错误
    #[error("词汇表错误: {0}")]
    VocabError(String),
}

/// 统一分词器接口
///
/// 所有文本模型（TTS/ASR/翻译/LLM）的分词器统一实现此 trait。
/// `encode` 将文本转为 token ID 序列，是每个分词器的核心能力。
/// `decode` 可选实现，用于需要 ID→文本转换的场景（翻译、ASR 解码）。
///
/// # 线程安全
///
/// 要求 `Send + Sync`，因为 tokenizer 通常在 Provider 内部共享。
pub trait TextTokenizer: Send + Sync {
    /// 编码文本为 token ID 序列
    ///
    /// # 参数
    /// - `text`: 输入文本（原始字符串，或预处理后的字符串如音素串）
    ///
    /// # 返回
    /// token ID 序列（包含 BOS/EOS 等特殊 token 由实现决定）
    fn encode(&self, text: &str) -> Result<Vec<i64>, TokenizerError>;

    /// 解码 token ID 序列为文本（可选）
    ///
    /// 默认返回错误。需要解码能力的实现应覆盖此方法。
    fn decode(&self, ids: &[i64]) -> Result<String, TokenizerError> {
        let _ = ids;
        Err(TokenizerError::DecodeFailed(
            "该分词器未实现解码功能".into(),
        ))
    }

    /// 词汇表大小
    fn vocab_size(&self) -> usize;
}
