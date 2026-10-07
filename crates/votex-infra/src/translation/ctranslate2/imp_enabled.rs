//! CTranslate2 真实推理形态（`--features ct2` 时编译）
//!
//! 通过 ct2rs 绑定加载 CTranslate2 模型（如 opus-mt-zh-en 的本地转换产物）：
//! int8 量化 + C++ 高度优化的 Transformer 算子与 KV-cache 管理，
//! 相较原生 ONNX Runtime 显著提速并降低内存占用。
//!
//! # 分词器（零 protobuf 方案）
//!
//! ct2rs 的 `sentencepiece` feature 走 C++ sentencepiece-sys，其内嵌 protobuf
//! 会与 sherpa-onnx-sys 的 protobuf 在静态链接时产生 LNK2005 符号冲突。
//! 因此这里用本仓库纯 Rust 的 [`SentencePieceBpe`]（opus_mt 同款）实现
//! [`ct2rs::Tokenizer`] trait：`Translator<T>` 泛型天然支持注入。
//!
//! # 模型目录要求（`scripts/convert_ct2.py` 转换产物）
//! - `model.bin` - CTranslate2 权重（int8 量化）
//! - `source.spm` + `target.spm` - SentencePiece 分词模型（本适配器读取；
//!   官方转换器不会自动复制，由转换脚本补齐）
//!
//! # 方向语义（对齐 opus_mt.rs 的路由模式）
//! 单目录单方向：目录名含 `en-zh` 视为英→中模型，否则视为中→英模型。
//! 请求方向与模型方向不符时返回 `UnsupportedDirection`。

use std::collections::HashMap;
use std::path::Path;

use ct2rs::{ComputeType, Config, TranslationOptions, Translator};
use votex_domain::error::TranslationError;
use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::value_object::TranslationDirection;

use crate::bpe_tokenizer::SentencePieceBpe;
use crate::shared::EngineState;

// ============================================================
// ct2rs 分词适配器（SentencePieceBpe → ct2rs::Tokenizer）
// ============================================================

/// ct2rs 分词适配器：source.spm 编码、target.spm 解码
///
/// 编码语义与 ct2rs 官方 sentencepiece 分词器一致：
/// piece 字符串序列 + 句尾 `</s>` 追加（Marian 约定）。
struct Ct2SpTokenizer {
    /// 源语言分词器（文本 → piece 字符串）
    encoder: SentencePieceBpe,
    /// 目标语言分词器（piece 字符串 → 文本）
    decoder: SentencePieceBpe,
    /// 目标词表反查表：piece 字节串 → piece ID（解码用，加载时构建一次）
    target_piece_to_id: HashMap<Vec<u8>, u32>,
}

impl Ct2SpTokenizer {
    /// 从模型目录加载（读 `source.spm` / `target.spm`）
    fn from_dir(dir: &Path) -> Result<Self, String> {
        let encoder = SentencePieceBpe::load(&dir.join("source.spm"))
            .map_err(|e| format!("加载 source.spm 失败: {}", e))?;
        let decoder = SentencePieceBpe::load(&dir.join("target.spm"))
            .map_err(|e| format!("加载 target.spm 失败: {}", e))?;

        // 构建目标词表反查表（opus_mt.rs 同款「piece 字节串 → ID」思路）
        let mut target_piece_to_id = HashMap::with_capacity(decoder.vocab_size());
        for id in 0..decoder.vocab_size() {
            if let Some(bytes) = decoder.get_piece(id as i32) {
                target_piece_to_id.insert(bytes.to_vec(), id as u32);
            }
        }

        Ok(Self {
            encoder,
            decoder,
            target_piece_to_id,
        })
    }
}

impl ct2rs::Tokenizer for Ct2SpTokenizer {
    /// 文本 → piece 字符串序列（追加句尾 `</s>`）
    fn encode(&self, input: &str) -> anyhow::Result<Vec<String>> {
        let ids = self
            .encoder
            .encode(input)
            .map_err(|e| anyhow::anyhow!("SentencePiece 编码失败: {}", e))?;
        let mut pieces: Vec<String> = ids
            .iter()
            .filter_map(|&id| self.encoder.get_piece(id as i32))
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
            .collect();
        pieces.push("</s>".to_string());
        Ok(pieces)
    }

    /// piece 字符串序列 → 文本（经 target.spm 反查后解码）
    fn decode(&self, tokens: Vec<String>) -> anyhow::Result<String> {
        let ids: Vec<u32> = tokens
            .iter()
            .filter_map(|t| self.target_piece_to_id.get(t.as_bytes()).copied())
            .collect();
        self.decoder
            .decode_piece_ids(&ids)
            .map_err(|e| anyhow::anyhow!("SentencePiece 解码失败: {}", e))
    }
}

// ============================================================
// 引擎实现
// ============================================================

/// ct2rs 翻译器（注入自定义分词适配器）
type Ct2Translator = Translator<Ct2SpTokenizer>;

/// 已加载的翻译器 + 语言对
struct LoadedModel {
    translator: Ct2Translator,
    /// 源语言 / 目标语言（如 `("zh", "en")`）
    pair: (String, String),
}

impl LoadedModel {
    /// 从目录名解析语言对
    ///
    /// 目录名含 `en-zh` → 英→中；其余（含 `zh-en`）→ 中→英。
    fn parse_pair(dir_name: &str) -> (String, String) {
        if dir_name.contains("en-zh") {
            ("en".to_string(), "zh".to_string())
        } else {
            ("zh".to_string(), "en".to_string())
        }
    }

    /// 翻译单段文本（先校验方向，再推理）
    fn translate(
        &self,
        text: &str,
        direction: &TranslationDirection,
    ) -> Result<String, TranslationError> {
        let (src, tgt) = &self.pair;
        match direction {
            // Auto：跟随已加载模型的方向
            TranslationDirection::Auto => {}
            TranslationDirection::ZhToEn => {
                if src != "zh" || tgt != "en" {
                    return Err(TranslationError::UnsupportedDirection);
                }
            }
            TranslationDirection::EnToZh => {
                if src != "en" || tgt != "zh" {
                    return Err(TranslationError::UnsupportedDirection);
                }
            }
            TranslationDirection::ByLanguagePair { source, target } => {
                if source != src || target != tgt {
                    return Err(TranslationError::UnsupportedDirection);
                }
            }
        }

        // 返回 Vec<(译文, 可选句级评分)>，单条输入取首个结果
        let results = self
            .translator
            .translate_batch(&[text], &TranslationOptions::default(), None)
            .map_err(|e| TranslationError::ApiError(format!("CTranslate2 推理失败: {}", e)))?;

        results
            .into_iter()
            .next()
            .map(|(output, _score)| output)
            .ok_or_else(|| TranslationError::ApiError("CTranslate2 未返回译文".to_string()))
    }
}

/// CTranslate2 翻译引擎（真实形态）
pub struct CTranslate2Provider {
    state: EngineState<LoadedModel>,
}

impl CTranslate2Provider {
    pub fn new() -> Self {
        Self {
            state: EngineState::new(),
        }
    }

    /// 从 CTranslate2 模型目录加载（int8 量化）
    ///
    /// `model.bin` 缺失时返回 `ModelNotLoaded`；加载失败（目录不完整等）返回 `ApiError`。
    pub fn load_from_dir(&self, model_dir: &Path) -> Result<(), TranslationError> {
        if !model_dir.join("model.bin").exists() {
            return Err(TranslationError::ModelNotLoaded);
        }

        let dir_name = model_dir
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        let pair = LoadedModel::parse_pair(&dir_name);

        let tokenizer = Ct2SpTokenizer::from_dir(model_dir)
            .map_err(TranslationError::ApiError)?;

        // int8 量化：与模型转换时的量化级别一致，兼顾速度与内存
        let config = Config {
            compute_type: ComputeType::INT8,
            ..Default::default()
        };
        let translator = Translator::with_tokenizer(model_dir, tokenizer, &config)
            .map_err(|e| TranslationError::ApiError(format!("加载 CTranslate2 模型失败: {}", e)))?;

        tracing::info!(
            "CTranslate2 模型加载完成（int8 量化）: {:?}（方向 {}→{}）",
            model_dir,
            pair.0,
            pair.1
        );
        self.state.load(LoadedModel { translator, pair });
        Ok(())
    }
}

impl TranslationProvider for CTranslate2Provider {
    fn name(&self) -> &str {
        "ctranslate2"
    }

    fn translate(
        &self,
        text: &str,
        direction: TranslationDirection,
    ) -> Result<String, TranslationError> {
        if text.trim().is_empty() {
            return Err(TranslationError::EmptyText);
        }

        // 外层 Err 是引擎未加载，内层 Err 才是翻译错误；
        // `?` 解包外层后，整体表达式直接返回内层 Result。
        self.state
            .with(|model| model.translate(text.trim(), &direction))
            .map_err(|_| TranslationError::ModelNotLoaded)?
    }

    fn load(&self, model_dir: &Path) -> Result<(), TranslationError> {
        self.load_from_dir(model_dir)
    }

    fn unload(&self) {
        self.state.unload();
    }

    fn is_loaded(&self) -> bool {
        self.state.is_loaded()
    }

    fn supported_pairs(&self) -> Vec<(String, String)> {
        self.state
            .with(|m| vec![(m.pair.0.clone(), m.pair.1.clone())])
            .unwrap_or_default()
    }

    fn max_input_chars(&self) -> usize {
        // CTranslate2 默认解码上限 256 token，输入侧对齐保守限制
        256
    }
}
