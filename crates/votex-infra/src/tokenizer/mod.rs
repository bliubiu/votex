//! 分词器实现
//!
//! 统一的文本分词器实现，均实现 `votex_domain::tts::tokenizer::TextTokenizer` trait。
//!
//! # 实现列表
//!
//! - `SentencePieceBpeTokenizer`：SentencePiece BPE 分词器，用于 M2M-100、NLLB 等翻译模型
//! - `ByteLevelBpeTokenizer`：ByteLevel BPE 分词器（GPT-2/Qwen2 风格），用于 CosyVoice3
//! - `CharLookupTokenizer`：单字查表分词器，用于 Kokoro
//!
//! # 新增分词器
//!
//! 1. 本模块下新建文件，实现 `TextTokenizer` trait
//! 2. 在 `mod.rs` 中注册子模块
//! 3. 在 Provider 的 `load()` 中初始化并使用

mod sentence_piece;
mod byte_level;
mod char_lookup;

pub use sentence_piece::SentencePieceBpeTokenizer;
pub use byte_level::ByteLevelBpeTokenizer;
pub use char_lookup::CharLookupTokenizer;
