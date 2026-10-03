//! SentencePiece BPE 分词器（纯 Rust 实现，零 protobuf 依赖）
//!
//! 手动解码 SentencePiece .model 文件的 protobuf 格式，
//! 提供 BPE 编码（文本 → token IDs）和解码（token IDs → 文本）。
//!
//! 为什么用纯 Rust 实现而非 sentencepiece-sys：
//! - 避免 sentencepiece-sys 的 protobuf-lite 与 ort/sherpa 的 protobuf 符号冲突
//! - 减少 C++ 依赖，加快编译速度
//! - 满足完全离线、单二进制的分发要求

use std::collections::HashMap;
use std::path::Path;

use votex_domain::model::registry::TokenizerConfig;

/// BPE 分词器错误
#[derive(Debug)]
pub enum BpeError {
    Io(std::io::Error),
    Parse(String),
}

impl std::fmt::Display for BpeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BpeError::Io(e) => write!(f, "IO 错误: {}", e),
            BpeError::Parse(msg) => write!(f, "解析错误: {}", msg),
        }
    }
}

impl std::error::Error for BpeError {}

impl From<std::io::Error> for BpeError {
    fn from(e: std::io::Error) -> Self {
        BpeError::Io(e)
    }
}

/// 前缀树节点
struct TrieNode {
    /// 匹配到的 piece ID（-1 表示非叶子节点）
    piece_id: i32,
    /// 子节点：字节 → 子节点
    children: HashMap<u8, Box<TrieNode>>,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode { piece_id: -1, children: HashMap::new() }
    }
}

/// SentencePiece BPE 分词器
///
/// 从 SentencePiece 的 .model 文件（protobuf 格式）加载词汇表，
/// 实现基于字节的 BPE 编码和解码。
/// 可选从 tokenizer.json 加载 BPE 词汇表 ID 映射（用于 M2M-100 等模型）。
pub struct SentencePieceBpe {
    /// piece 字节串 → (score)
    pieces: Vec<(Vec<u8>, f32)>,
    /// 字符级回退映射（单字节/单字符 → ID）
    char_fallback: HashMap<Vec<u8>, i32>,
    /// ID → piece 字节串的映射（用于解码）
    id_to_piece: HashMap<i32, Vec<u8>>,
    /// BPE 词汇表映射（来自 tokenizer.json）
    /// piece 字节串 → BPE 词汇表 ID
    bpe_piece_to_id: Option<HashMap<Vec<u8>, i64>>,
    /// BPE 词汇表 ID → piece 字节串
    bpe_id_to_piece: Option<HashMap<i64, Vec<u8>>>,
    /// SentencePiece 最多有 128000 个 token，额外的 ID 被视为特殊 token
    bpe_vocab_size: usize,
    /// 前缀树（用于 O(V) → O(max_piece_len) 的编码加速）
    trie: TrieNode,
}

impl SentencePieceBpe {
    /// 从 .model 文件加载 SentencePiece 模型
    pub fn load(path: &Path) -> Result<Self, BpeError> {
        let data = std::fs::read(path)?;
        Self::decode_protobuf(&data)
    }

    /// 从字节数据加载 SentencePiece 模型
    pub fn from_bytes(data: &[u8]) -> Result<Self, BpeError> {
        Self::decode_protobuf(data)
    }

    /// 手动解码 SentencePiece protobuf
    ///
    /// SentencePiece 存储格式（protobuf）：
    /// ```protobuf
    /// message SentencePieceProcessor {
    ///   repeated Piece pieces = 1;
    /// }
    /// message Piece {
    ///   string piece = 1;
    ///   float score = 2;
    ///   int32 id = 3;  // 可选，默认按顺序编号
    /// }
    /// ```
    fn decode_protobuf(data: &[u8]) -> Result<Self, BpeError> {
        let mut pos = 0;
        let mut pieces: Vec<(Vec<u8>, f32)> = Vec::new();

        while pos < data.len() {
            let tag = decode_varint(data, &mut pos)?;
            let field_number = tag >> 3;
            let wire_type = (tag & 0x07) as u8;

            match (field_number, wire_type) {
                (1, 2) => {
                    // Field 1: repeated Piece pieces = 1
                    let len = decode_varint(data, &mut pos)? as usize;
                    let sub_end = pos + len;

                    let mut piece_bytes: Option<Vec<u8>> = None;
                    let mut score: f32 = 0.0;
                    let mut _custom_id: Option<i32> = None;

                    while pos < sub_end {
                        let sub_tag = decode_varint(data, &mut pos)?;
                        let sub_field = sub_tag >> 3;
                        let sub_wire = (sub_tag & 0x07) as u8;

                        match (sub_field, sub_wire) {
                            (1, 2) => {
                                // string piece = 1
                                let str_len = decode_varint(data, &mut pos)? as usize;
                                if pos + str_len <= data.len() {
                                    piece_bytes = Some(data[pos..pos + str_len].to_vec());
                                    pos += str_len;
                                } else {
                                    return Err(BpeError::Parse("SentencePiece 数据截断".into()));
                                }
                            }
                            (2, 5) => {
                                // float score = 2 (fixed32)
                                if pos + 4 <= data.len() {
                                    score = f32::from_le_bytes([
                                        data[pos],
                                        data[pos + 1],
                                        data[pos + 2],
                                        data[pos + 3],
                                    ]);
                                    pos += 4;
                                }
                            }
                            (3, 0) => {
                                // int32 score = 3 (varint) - 自定义 ID
                                let id = decode_varint(data, &mut pos)? as i32;
                                _custom_id = Some(id);
                            }
                            _ => {
                                skip_field(data, &mut pos, sub_wire)?;
                            }
                        }
                    }

                    if let Some(bytes) = piece_bytes {
                        pieces.push((bytes, score));
                    }

                    pos = sub_end;
                }
                _ => {
                    skip_field(data, &mut pos, wire_type)?;
                }
            }
        }

        if pieces.is_empty() {
            return Err(BpeError::Parse("SentencePiece 模型为空或无有效片段".into()));
        }

        // 构建字符级回退映射和 ID→piece 映射
        let mut char_fallback = HashMap::new();
        let mut id_to_piece = HashMap::new();
        for (i, (bytes, _)) in pieces.iter().enumerate() {
            let id = i as i32;
            id_to_piece.insert(id, bytes.clone());

            // 单字节 ASCII 字符直接映射
            if bytes.len() == 1 {
                char_fallback.insert(bytes.clone(), id);
            }
            // 多字节 UTF-8 单字符
            if let Ok(s) = std::str::from_utf8(bytes) {
                if s.chars().count() == 1 && bytes.len() > 1 {
                    char_fallback.insert(bytes.clone(), id);
                }
            }
        }

        // 构建前缀树
        let mut trie = TrieNode::new();
        for (id, (bytes, _)) in pieces.iter().enumerate() {
            let mut node = &mut trie;
            for &b in bytes {
                node = node.children.entry(b).or_insert_with(|| Box::new(TrieNode::new()));
            }
            node.piece_id = id as i32;
        }

        Ok(Self {
            pieces,
            char_fallback,
            id_to_piece,
            bpe_piece_to_id: None,
            bpe_id_to_piece: None,
            bpe_vocab_size: 0,
            trie,
        })
    }

    /// 编码文本为 token ID 序列
    ///
    /// 实现 SentencePiece BPE 编码：
    /// 1. 在文本前添加空格标记 `▁`（U+2581，SentencePiece 约定）
    /// 2. 按 UTF-8 字节分割
    /// 3. 贪婪匹配最长 piece
    /// 4. 如果加载了 BPE 词汇表，将 piece 索引映射到 BPE 词汇表 ID
    pub fn encode(&self, text: &str) -> Result<Vec<i64>, BpeError> {
        // SentencePiece 在文本前加空格标记 U+2581 (▁)
        let marker = "\u{2581}";
        let mut input_bytes = Vec::with_capacity(marker.len() + text.len());
        input_bytes.extend_from_slice(marker.as_bytes());
        input_bytes.extend_from_slice(text.as_bytes());
        let bytes = &input_bytes;

        let mut token_ids: Vec<i64> = Vec::new();
        let mut i = 0;

        while i < bytes.len() {
            // 使用前缀树从当前位置查找最长匹配的 piece
            let mut node = &self.trie;
            let mut best_match: Option<(usize, i32)> = None;
            let max_search = (bytes.len() - i).min(20);

            for len in 0..max_search {
                let b = bytes[i + len];
                match node.children.get(&b) {
                    Some(child) => {
                        if child.piece_id >= 0 {
                            best_match = Some((len + 1, child.piece_id));
                        }
                        node = child;
                    }
                    None => break,
                }
            }

            if let Some((match_len, id)) = best_match {
                let piece = &self.pieces[id as usize].0;
                // 如果加载了 BPE 词汇表，映射到 BPE ID
                if let Some(ref map) = self.bpe_piece_to_id {
                    let bpe_id = map.get(piece).copied().unwrap_or(id as i64);
                    token_ids.push(bpe_id);
                } else {
                    token_ids.push(id as i64);
                }
                i += match_len;
            } else {
                // 回退：尝试单个字符
                let byte = bytes[i];
                if byte.is_ascii() {
                    let key = vec![byte];
                    if let Some(&sp_id) = self.char_fallback.get(&key) {
                        let id = sp_id as i64;
                        // 如果加载了 BPE 词汇表，映射到 BPE ID
                        if let Some(ref map) = self.bpe_piece_to_id {
                            let bpe_id = map.get(&key).copied().unwrap_or(id);
                            token_ids.push(bpe_id);
                        } else {
                            token_ids.push(id);
                        }
                    } else {
                        token_ids.push(0i64); // <unk>
                    }
                    i += 1;
                } else {
                    let char_len = utf8_char_length(byte);
                    if i + char_len <= bytes.len() {
                        let key = bytes[i..i + char_len].to_vec();
                        if let Some(&sp_id) = self.char_fallback.get(&key) {
                            let id = sp_id as i64;
                            // 如果加载了 BPE 词汇表，映射到 BPE ID
                            if let Some(ref map) = self.bpe_piece_to_id {
                                let bpe_id = map.get(&key).copied().unwrap_or(id);
                                token_ids.push(bpe_id);
                            } else {
                                token_ids.push(id);
                            }
                        } else {
                            // 逐字节回退
                            for j in 0..char_len {
                                let b = bytes[i + j];
                                let key = vec![b];
                                if let Some(&sp_id) = self.char_fallback.get(&key) {
                                    let id = sp_id as i64;
                                    if let Some(ref map) = self.bpe_piece_to_id {
                                        let bpe_id = map.get(&key).copied().unwrap_or(id);
                                        token_ids.push(bpe_id);
                                    } else {
                                        token_ids.push(id);
                                    }
                                }
                            }
                        }
                        i += char_len;
                    } else {
                        token_ids.push(0i64);
                        i += 1;
                    }
                }
            }
        }

        Ok(token_ids)
    }

    /// 解码 token ID 序列为文本
    ///
    /// 与 SentencePieceProcessor::decode_piece_ids 兼容，
    /// 用于翻译模型的输出解码。
    /// 如果加载了 BPE 词汇表（tokenizer.json），优先使用 BPE ID 映射。
    pub fn decode_piece_ids(&self, ids: &[u32]) -> Result<String, BpeError> {
        // 如果加载了 BPE 词汇表，使用 BPE 映射
        if let Some(ref bpe_id_to_piece) = self.bpe_id_to_piece {
            let mut result = Vec::new();
            for &id in ids {
                let id_i64 = id as i64;
                match bpe_id_to_piece.get(&id_i64) {
                    Some(bytes) => {
                        result.extend_from_slice(bytes);
                    }
                    None => {
                        if id != 0 {
                            result.push(b'?');
                        }
                    }
                }
            }
            let text = String::from_utf8(result)
                .map_err(|e| BpeError::Parse(format!("解码结果不是合法 UTF-8: {}", e)))?;
            let text = text.replace('\u{2581}', " ");
            return Ok(text.trim_start().to_string());
        }

        // 回退：使用 SentencePiece 原始 ID 映射
        let mut result = Vec::new();
        for &id in ids {
            let id_signed = id as i32;
            match self.id_to_piece.get(&id_signed) {
                Some(bytes) => {
                    result.extend_from_slice(bytes);
                }
                None => {
                    if id != 0 {
                        result.push(b'?');
                    }
                }
            }
        }

        let text = String::from_utf8(result)
            .map_err(|e| BpeError::Parse(format!("解码结果不是合法 UTF-8: {}", e)))?;
        let text = text.replace('\u{2581}', " ");
        Ok(text.trim_start().to_string())
    }

    /// 使用配置编码文本，自动处理 ID 偏移
    ///
    /// 某些模型（如 NLLB）的 HuggingFace wrapper 会重排特殊 token，
    /// 导致 token ID 比原始 SentencePiece 模型偏移。此方法根据
    /// `TokenizerConfig.id_offset` 自动应用偏移。
    ///
    /// # 参数
    /// * `text` - 输入文本
    /// * `config` - 分词器配置（含 id_offset 等信息）
    ///
    /// # 返回值
    /// 编码后的 token ID 序列（已偏移）
    pub fn encode_with_config(&self, text: &str, config: &TokenizerConfig) -> Result<Vec<i64>, BpeError> {
        let mut ids = self.encode(text)?;
        if config.id_offset != 0 {
            for id in ids.iter_mut() {
                // 跳过特殊 token（<unk>=0, <s>=1, </s>=2），这些已被 HF 重排
                if *id >= 3 {
                    *id += config.id_offset;
                }
            }
        }
        Ok(ids)
    }

    /// 使用配置解码 token ID 序列，自动还原偏移
    ///
    /// 与 `encode_with_config` 配合使用，将模型输出的已偏移 ID
    /// 还原为原始 SentencePiece ID 后再解码。
    pub fn decode_with_config(&self, ids: &[i64], config: &TokenizerConfig) -> Result<String, BpeError> {
        if config.id_offset != 0 {
            // 还原偏移：HF 空间中内容 token 起始 ID = 3 + id_offset
            let content_start = 3 + config.id_offset;
            let restored: Vec<u32> = ids.iter()
                .map(|&id| {
                    if id >= content_start {
                        (id - config.id_offset) as u32
                    } else {
                        id as u32
                    }
                })
                .collect();
            self.decode_piece_ids(&restored)
        } else {
            let raw: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
            self.decode_piece_ids(&raw)
        }
    }

    /// 从 tokenizer.json 加载 BPE 词汇表 ID 映射
    ///
    /// tokenizer.json 中的 BPE 词汇表 ID 可能与 SentencePiece 模型不同。
    /// ONNX 模型使用的是 BPE 词汇表的 ID 排序，必须加载此映射才能正确编码/解码。
    ///
    /// 注意：tokenizer.json 的 vocab size 可能大于 SentencePiece 的 128000，
    /// 多出的 ID（如语言 token `__en__`→128022）需要使用 `set_custom_token` 手动注册。
    pub fn load_bpe_vocab_from_json(&mut self, path: &Path) -> Result<(), BpeError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| BpeError::Io(e))?;
        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| BpeError::Parse(format!("解析 tokenizer.json 失败: {}", e)))?;

        let vocab_obj = json["model"]["vocab"].as_object()
            .ok_or_else(|| BpeError::Parse("tokenizer.json 缺少 model.vocab".into()))?;

        let mut bpe_piece_to_id = HashMap::new();
        let mut bpe_id_to_piece = HashMap::new();

        for (piece_str, id_value) in vocab_obj {
            let id = id_value.as_i64()
                .ok_or_else(|| BpeError::Parse(format!("词汇表 ID 非整数: {}", piece_str)))?;
            let piece_bytes = piece_str.as_bytes().to_vec();

            bpe_piece_to_id.insert(piece_bytes.clone(), id);
            bpe_id_to_piece.insert(id, piece_bytes);
        }

        let bpe_vocab_size = bpe_piece_to_id.len();

        self.bpe_piece_to_id = Some(bpe_piece_to_id);
        self.bpe_id_to_piece = Some(bpe_id_to_piece);
        self.bpe_vocab_size = bpe_vocab_size;

        Ok(())
    }

    /// 是否有 BPE 词汇表映射
    pub fn has_bpe_vocab(&self) -> bool {
        self.bpe_piece_to_id.is_some()
    }

    /// 获取 SentencePiece 词汇表大小
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// 获取 BPE 词汇表大小（来自 tokenizer.json）
    pub fn bpe_vocab_size(&self) -> usize {
        self.bpe_vocab_size
    }

    /// 根据 token ID 获取对应的 piece 字节串
    pub fn get_piece(&self, id: i32) -> Option<&[u8]> {
        self.id_to_piece.get(&id).map(|v| v.as_slice())
    }
}

// ===================== Protobuf 解码工具函数 =====================

/// 解码 protobuf varint
pub(crate) fn decode_varint(data: &[u8], pos: &mut usize) -> Result<u64, BpeError> {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        if *pos >= data.len() {
            return Err(BpeError::Parse("protobuf 解码: 意外的数据结束".into()));
        }
        let byte = data[*pos];
        *pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        shift += 7;
        if byte & 0x80 == 0 {
            return Ok(result);
        }
        if shift > 63 {
            return Err(BpeError::Parse("protobuf 解码: varint 过大".into()));
        }
    }
}

/// 跳过 protobuf 字段
fn skip_field(data: &[u8], pos: &mut usize, wire_type: u8) -> Result<(), BpeError> {
    match wire_type {
        0 => {
            // varint
            decode_varint(data, pos)?;
        }
        1 => {
            // fixed64
            if *pos + 8 > data.len() {
                return Err(BpeError::Parse("protobuf 解码: 数据不足".into()));
            }
            *pos += 8;
        }
        2 => {
            // length-delimited
            let len = decode_varint(data, pos)? as usize;
            if *pos + len > data.len() {
                return Err(BpeError::Parse(
                    "protobuf 解码: 数据不足(length-delimited)".into(),
                ));
            }
            *pos += len;
        }
        5 => {
            // fixed32
            if *pos + 4 > data.len() {
                return Err(BpeError::Parse("protobuf 解码: 数据不足(fixed32)".into()));
            }
            *pos += 4;
        }
        _ => {
            return Err(BpeError::Parse(format!(
                "protobuf 解码: 不支持的 wire_type {}",
                wire_type
            )));
        }
    }
    Ok(())
}

/// UTF-8 首字节 → 字符长度
pub(crate) fn utf8_char_length(first_byte: u8) -> usize {
    if first_byte & 0x80 == 0 {
        1
    } else if first_byte & 0xE0 == 0xC0 {
        2
    } else if first_byte & 0xF0 == 0xE0 {
        3
    } else if first_byte & 0xF8 == 0xF0 {
        4
    } else {
        1
    }
}

// ===================== 测试 =====================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_and_encode_decode() {
        // 使用项目中的测试 .model 文件
        let test_path = Path::new("models/indextts2/bpe.model");
        if !test_path.exists() {
            eprintln!("跳过测试: bpe.model 不存在");
            return;
        }

        let bpe = SentencePieceBpe::load(test_path).expect("加载 BPE 模型失败");
        assert!(bpe.vocab_size() > 0);

        // 编码
        let ids = bpe.encode("测试文本").expect("编码失败");
        assert!(!ids.is_empty());

        // 解码
        let ids_u32: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
        // 解码结果可能与原文不同（BPE 的归一化/空格处理），但不应为空
        let decoded = bpe.decode_piece_ids(&ids_u32).expect("解码失败");
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_decode_roundtrip() {
        let test_path = Path::new("models/indextts2/bpe.model");
        if !test_path.exists() {
            return;
        }

        let bpe = SentencePieceBpe::load(test_path).unwrap();
        let ids = bpe.encode("hello world").unwrap();
        let ids_u32: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
        let result = bpe.decode_piece_ids(&ids_u32).unwrap();

        // BPE 编解码应该是无损的（对 ASCII 文本）
        assert_eq!(result, "hello world");
    }
}
