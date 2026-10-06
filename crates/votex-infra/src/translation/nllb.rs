//! NLLB-200 翻译引擎（ONNX 推理 + 自回归解码）
//!
//! 使用 Meta 的 NLLB-200-distilled-600M 模型通过 ONNX Runtime 实现完全离线的多语言翻译。
//! 支持 200+ 语言对，模型约 2.4GB，适合 CPU 运行。
//!
//! # 模型文件
//! - `models/translation/nllb-200-distilled-600m/encoder_model.onnx` - 编码器 (FP32)
//! - `models/translation/nllb-200-distilled-600m/decoder_model.onnx` - 解码器 (FP32)
//! - `models/translation/nllb-200-distilled-600m/encoder_model_int8.onnx` - 编码器 (INT8, 优先)
//! - `models/translation/nllb-200-distilled-600m/decoder_model_int8.onnx` - 解码器 (INT8, 优先)
//! - `models/translation/nllb-200-distilled-600m/sentencepiece.bpe.model` - 分词器
//!
//! # 语言方向
//! NLLB 使用 FLORES-200 语言代码 token（如 `eng_Latn`、`zho_Hans`）作为 added tokens。
//! 这些 token 不在 SentencePiece 词汇表中，需手动处理：
//! - 编码器输入：源语言 token + 原文
//! - 解码器初始 token：目标语言 token

use std::path::Path;

use ndarray::{Array2, ArrayD};
use ort::session::Session;
use ort::value::Value;
use votex_domain::error::TranslationError;
use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::value_object::TranslationDirection;

use std::collections::HashMap;

use crate::bpe_tokenizer::SentencePieceBpe;
use crate::shared::{EngineState, OrtSessionFactory};
use votex_domain::model::registry::TokenizerConfig;

// ===================== 默认 Token ID =====================

const PAD_TOKEN_ID: i64 = 1;
const EOS_TOKEN_ID: i64 = 2;      // </s>
const UNK_TOKEN_ID: i64 = 3;      // <unk>
const MAX_LENGTH: usize = 200;

/// 默认 NLLB 分词器配置（当未设置自定义配置时使用）
fn default_nllb_config() -> TokenizerConfig {
    let mut added = HashMap::new();
    added.insert("eng_Latn".into(), 256047i64);
    added.insert("zho_Hans".into(), 256200i64);
    added.insert("zho_Hant".into(), 256201i64);
    TokenizerConfig {
        tokenizer_type: Some("sentencepiece".into()),
        id_offset: 1,
        eos_on_input: true,
        decoder_start_token: Some("eos".into()),
        added_tokens: added,
    }
}

/// 语言对（源语言 token ID → 目标语言 token ID）
struct LangPair {
    src: i64,
    tgt: i64,
}

// ===================== NLLB Provider =====================

/// NLLB-200 ONNX 翻译引擎
pub struct NllbProvider {
    encoder: EngineState<Session>,
    decoder: EngineState<Session>,
    /// SentencePiece 分词器
    ///
    /// 走 `EngineState`：`TranslationProvider::load` 是 `&self`，
    /// 裸 `Option` 无法在共享实例上写入。
    tokenizer: EngineState<SentencePieceBpe>,
    tokenizer_config: TokenizerConfig,
    max_length: usize,
}

impl NllbProvider {
    pub fn new() -> Self {
        Self {
            encoder: EngineState::new(),
            decoder: EngineState::new(),
            tokenizer: EngineState::new(),
            tokenizer_config: default_nllb_config(),
            max_length: MAX_LENGTH,
        }
    }

    /// 设置最大生成长度
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// 设置分词汇配置（覆盖默认值）
    pub fn with_tokenizer_config(mut self, config: TokenizerConfig) -> Self {
        self.tokenizer_config = config;
        self
    }

    /// 根据方向（和汇总配置）查找语言代码 ID
    fn direction_to_lang_pair(&self, dir: &TranslationDirection) -> Result<LangPair, TranslationError> {
        let added = &self.tokenizer_config.added_tokens;
        match dir {
            TranslationDirection::ZhToEn | TranslationDirection::Auto => {
                let src = *added.get("zho_Hans").ok_or_else(|| TranslationError::ApiError("配置中缺少 zho_Hans".into()))?;
                let tgt = *added.get("eng_Latn").ok_or_else(|| TranslationError::ApiError("配置中缺少 eng_Latn".into()))?;
                Ok(LangPair { src, tgt })
            }
            TranslationDirection::EnToZh => {
                let src = *added.get("eng_Latn").ok_or_else(|| TranslationError::ApiError("配置中缺少 eng_Latn".into()))?;
                let tgt = *added.get("zho_Hans").ok_or_else(|| TranslationError::ApiError("配置中缺少 zho_Hans".into()))?;
                Ok(LangPair { src, tgt })
            }
            TranslationDirection::ByLanguagePair { .. } => {
                let src = *added.get("zho_Hans").ok_or_else(|| TranslationError::ApiError("配置中缺少 zho_Hans".into()))?;
                let tgt = *added.get("eng_Latn").ok_or_else(|| TranslationError::ApiError("配置中缺少 eng_Latn".into()))?;
                Ok(LangPair { src, tgt })
            }
        }
    }

    /// 从模型目录加载
    ///
    /// 加载策略：
    /// 1. 优先尝试 `encoder_model_int8.onnx` / `decoder_model_int8.onnx`
    /// 2. 回退到 `encoder_model.onnx` / `decoder_model.onnx`
    pub fn load_from_dir(&self, model_dir: &Path) -> Result<(), TranslationError> {
        let tokenizer_path = model_dir.join("sentencepiece.bpe.model");

        // 尝试 INT8 量化模型（优先）
        let int8_enc = model_dir.join("encoder_model_int8.onnx");
        let int8_dec = model_dir.join("decoder_model_int8.onnx");

        if int8_enc.exists() && int8_dec.exists() {
            let enc = OrtSessionFactory::create_raw(&int8_enc)
                .map_err(|e| TranslationError::ApiError(format!("加载 INT8 编码器失败: {}", e)))?;
            self.encoder.load(enc);

            let dec = OrtSessionFactory::create_raw(&int8_dec)
                .map_err(|e| TranslationError::ApiError(format!("加载 INT8 解码器失败: {}", e)))?;
            self.decoder.load(dec);

            tracing::info!("NLLB-200 已加载 INT8 量化模型");
        } else {
            // 回退到 FP32 模型
            let encoder_path = model_dir.join("encoder_model.onnx");
            let decoder_path = model_dir.join("decoder_model.onnx");

            if !encoder_path.exists() || !decoder_path.exists() {
                return Err(TranslationError::ModelNotLoaded);
            }

            let enc = OrtSessionFactory::create_raw(&encoder_path)
                .map_err(|e| TranslationError::ApiError(format!("加载编码器失败: {}", e)))?;
            self.encoder.load(enc);

            let dec = OrtSessionFactory::create_raw(&decoder_path)
                .map_err(|e| TranslationError::ApiError(format!("加载解码器失败: {}", e)))?;
            self.decoder.load(dec);

            tracing::info!("NLLB-200 已加载 FP32 模型");
        }

        if tokenizer_path.exists() {
            let sp = SentencePieceBpe::load(&tokenizer_path)
                .map_err(|e| TranslationError::ApiError(format!("加载分词器失败: {}", e)))?;
            self.tokenizer.load(sp);
        } else {
            return Err(TranslationError::ModelNotLoaded);
        }

        tracing::info!("NLLB-200 翻译引擎加载完成: {:?}", model_dir);
        Ok(())
    }

    /// 翻译文本（内部方法）
    fn translate_internal(
        &self,
        text: &str,
        direction: TranslationDirection,
    ) -> Result<String, TranslationError> {
        // 守卫必须活到整个翻译流程结束（编码 → 推理 → 解码都用到 tokenizer）。
        let tokenizer_guard = self.tokenizer.get();
        let tokenizer = tokenizer_guard
            .as_ref()
            .and_then(|g| g.as_ref())
            .ok_or(TranslationError::ModelNotLoaded)?;

        let pair = self.direction_to_lang_pair(&direction)?;
        let config = &self.tokenizer_config;

        // 预计算 added_tokens 集合用于过滤
        let added_token_ids: std::collections::HashSet<i64> =
            config.added_tokens.values().copied().collect();

        // ===== 1. 编码：源语言 token + 原文 =====
        // encode_with_config 自动处理 id_offset（NLLB 偏移 +1）
        let mut sp_ids = tokenizer
            .encode_with_config(text, config)
            .map_err(|e| TranslationError::ApiError(format!("分词失败: {}", e)))?;
        // 在开头插入源语言 token（语言代码不偏移，直接在 config 中配置最终 ID）
        sp_ids.insert(0, pair.src);
        // 末尾追加 EOS（与 HuggingFace tokenizer 行为一致）
        if config.eos_on_input {
            sp_ids.push(EOS_TOKEN_ID);
        }

        let input_tensor = Array2::from_shape_vec((1, sp_ids.len()), sp_ids.clone())
            .map_err(|e| TranslationError::ApiError(format!("创建输入张量失败: {}", e)))?;

        let attention_mask: Array2<i64> = Array2::ones((1, sp_ids.len()));
        let input_value = Value::from_array(input_tensor)
            .map_err(|e| TranslationError::ApiError(format!("创建输入值失败: {}", e)))?;
        let mask_value = Value::from_array(attention_mask)
            .map_err(|e| TranslationError::ApiError(format!("创建 mask 值失败: {}", e)))?;

        // ===== 2. 编码器推理 =====
        let encoder_hidden: ArrayD<f32> = self
            .encoder
            .with_mut(|session| {
                let outputs = session
                    .run(ort::inputs![
                        "input_ids" => input_value,
                        "attention_mask" => mask_value
                    ])
                    .map_err(|e| {
                        TranslationError::ApiError(format!("编码器推理失败: {}", e))
                    })?;
                let arr = outputs[0]
                    .try_extract_array::<f32>()
                    .map_err(|e| {
                        TranslationError::ApiError(format!("提取编码器输出失败: {}", e))
                    })?;
                Ok(arr.to_owned())
            })
            .map_err(|_| TranslationError::ApiError("编码器引擎未加载".to_string()))?
            .map_err(|e| e)?;

        // ===== 3. 自回归解码 =====
        // 解码器以 EOS（ID 2）开始，强制第一个生成 token 为目标语言代码
        let mut gen_ids: Vec<i64> = vec![EOS_TOKEN_ID];
        let enc_attention_mask: Array2<i64> = Array2::ones((1, sp_ids.len()));
        let mut is_first_step = true;

        for _step in 0..self.max_length {
            let dec_arr = Array2::from_shape_vec((1, gen_ids.len()), gen_ids.clone())
                .map_err(|e| TranslationError::ApiError(format!("创建解码输入失败: {}", e)))?;
            let dec_val: Value = Value::from_array(dec_arr)
                .map_err(|e| {
                    TranslationError::ApiError(format!("创建解码输入值失败: {}", e))
                })?
                .into();
            let enc_hid_val: Value = Value::from_array(encoder_hidden.clone())
                .map_err(|e| {
                    TranslationError::ApiError(format!("创建 hidden 值失败: {}", e))
                })?
                .into();
            let enc_mask_val: Value = Value::from_array(enc_attention_mask.clone())
                .map_err(|e| {
                    TranslationError::ApiError(format!("创建 enc mask 值失败: {}", e))
                })?
                .into();

            let raw_logits = self
                .decoder
                .with_mut(|session| {
                    let outputs = session
                        .run(ort::inputs![
                            "input_ids" => dec_val,
                            "encoder_hidden_states" => enc_hid_val,
                            "encoder_attention_mask" => enc_mask_val,
                        ])
                        .map_err(|e| {
                            TranslationError::ApiError(format!("解码推理失败: {}", e))
                        })?;
                    let logits = outputs[0]
                        .try_extract_array::<f32>()
                        .map_err(|e| {
                            TranslationError::ApiError(format!("提取 logits 失败: {}", e))
                        })?;
                    Ok(logits.to_owned())
                })
                .map_err(|_| TranslationError::ApiError("解码器引擎未加载".to_string()))?
                .map_err(|e| e)?;

            // 第一步：强制生成目标语言 token
            let token = if is_first_step {
                is_first_step = false;
                pair.tgt as u32
            } else {
                greedy_argmax_decoder(&raw_logits.view())
            };

            if token == EOS_TOKEN_ID as u32 {
                break;
            }
            if token == UNK_TOKEN_ID as u32 {
                break;
            }

            gen_ids.push(token as i64);
        }

        // ===== 4. 解码输出 =====
        // Debug 统计
        let lang_count = gen_ids
            .iter()
            .filter(|&&id| added_token_ids.contains(&id))
            .count();
        let special_count = gen_ids
            .iter()
            .filter(|&&id| {
                id == PAD_TOKEN_ID || id == EOS_TOKEN_ID || id == UNK_TOKEN_ID
            })
            .count();
        let content_count = gen_ids.len() - lang_count - special_count;
        tracing::debug!(
            "NLLB 生成 tokens: 总数={}, 语言代码={}, 特殊={}, 内容={}",
            gen_ids.len(),
            lang_count,
            special_count,
            content_count
        );
        if !gen_ids.is_empty() {
            tracing::debug!(
                "NLLB 前 10 个生成 token: {:?}",
                &gen_ids[..gen_ids.len().min(10)]
            );
        }

        // 过滤辅助 token + 语言代码 token
        let content: Vec<i64> = gen_ids
            .iter()
            .filter(|&&id| {
                id != PAD_TOKEN_ID
                    && id != EOS_TOKEN_ID
                    && id != UNK_TOKEN_ID
                    && !added_token_ids.contains(&id)
            })
            .copied()
            .collect();

        // decode_with_config 自动还原 id_offset
        let translated = tokenizer
            .decode_with_config(&content, config)
            .map_err(|e| TranslationError::ApiError(format!("解码失败: {}", e)))?;

        Ok(translated.trim().to_string())
    }
}

impl TranslationProvider for NllbProvider {
    fn name(&self) -> &str {
        "nllb"
    }

    fn translate(
        &self,
        text: &str,
        direction: TranslationDirection,
    ) -> Result<String, TranslationError> {
        if text.trim().is_empty() {
            return Err(TranslationError::EmptyText);
        }

        match direction {
            TranslationDirection::ZhToEn
            | TranslationDirection::EnToZh
            | TranslationDirection::Auto
            | TranslationDirection::ByLanguagePair { .. } => self.translate_internal(text, direction),
        }
    }

    fn load(&self, model_dir: &Path) -> Result<(), TranslationError> {
        self.load_from_dir(model_dir)
    }

    fn unload(&self) {
        self.encoder.unload();
        self.decoder.unload();
    }

    fn is_loaded(&self) -> bool {
        self.encoder.is_loaded() && self.decoder.is_loaded() && self.tokenizer.is_loaded()
    }

    fn max_input_chars(&self) -> usize {
        // NLLB 最长 200 token，中文按 ~1.5 字/token 保守折算
        self.max_length * 2
    }
}

/// 贪心解码：从 decoder logits 的最后一个位置选出概率最高的 token
fn greedy_argmax_decoder(logits: &ndarray::ArrayViewD<f32>) -> u32 {
    let shape = logits.shape();
    // logits shape: (1, seq_len, vocab_size)
    let seq_pos = shape[1].saturating_sub(1);
    let vocab_sz = shape[2];
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for v in 0..vocab_sz {
        let val = logits[[0, seq_pos, v]];
        if val > max_val {
            max_val = val;
            max_idx = v;
        }
    }
    max_idx as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_argmax_decoder() {
        let logits = ndarray::array![[[0.1, 0.2, 0.3, 0.4]]].into_dyn();
        let token = greedy_argmax_decoder(&logits.view());
        assert_eq!(token, 3);
    }

    #[test]
    fn test_direction_mapping() {
        let provider = NllbProvider::new();
        // 默认配置使用 eng_Latn=256047, zho_Hans=256200
        let pair = provider.direction_to_lang_pair(&TranslationDirection::ZhToEn).unwrap();
        assert_eq!(pair.src, 256200);
        assert_eq!(pair.tgt, 256047);

        let pair = provider.direction_to_lang_pair(&TranslationDirection::EnToZh).unwrap();
        assert_eq!(pair.src, 256047);
        assert_eq!(pair.tgt, 256200);

        let pair = provider.direction_to_lang_pair(&TranslationDirection::Auto).unwrap();
        assert_eq!(pair.src, 256200);
        assert_eq!(pair.tgt, 256047);
    }

    #[test]
    fn test_default_config() {
        let config = default_nllb_config();
        assert_eq!(config.id_offset, 1);
        assert!(config.eos_on_input);
        assert_eq!(config.added_tokens.len(), 3);
        assert_eq!(*config.added_tokens.get("eng_Latn").unwrap(), 256047);
        assert_eq!(*config.added_tokens.get("zho_Hans").unwrap(), 256200);
    }
}
