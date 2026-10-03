//! M2M-100 多语言翻译引擎（ONNX 推理 + KV cache）
//!
//! 使用 Meta 的 M2M-100 系列模型（418M / 1.2B）通过 ONNX Runtime 实现完全离线的多语言翻译。
//! 支持 100 种语言直接互译（无需英语中转），与 NLLB 共享 Encoder-Decoder 架构。
//!
//! # 模型文件
//! - `models/translation/m2m-100/encoder_model.int8.onnx` - 编码器
//! - `models/translation/m2m-100/decoder_model.int8.onnx` - 解码器
//! - `models/translation/m2m-100/decoder_with_past_model.onnx` - 带 KV cache 的解码器（可选）
//! - `models/translation/m2m-100/sentencepiece.bpe.model` - 共享 SentencePiece 分词器（128000 词条）
//! - `models/translation/m2m-100/tokenizer.json` - BPE 分词器（含 112 个语言 token）
//!
//! # 架构说明
//! M2M-100 与 NLLB 架构高度相似（同属 Meta Encoder-Decoder 家族）：
//! - 共享 SentencePiece 分词器（M2M-100 编码器/解码器使用同一分词器）
//! - 语言路由：在源文本前插入语言 token ID（如 128022=`__en__`）
//! - 解码首步强制输出目标语言 token（forced_bos_token_id）
//! - KV cache 优化：复用 decoder_with_past 推理逻辑
//!
//! # 重要：语言 token 处理
//! SentencePiece 模型只有 128000 个 token，语言 token（`__en__`→128022 等）是 tokenizer.json
//! 额外添加的 112 个特殊 token。因此不能通过 SentencePiece 编码 `__{lang}__` 文本，
//! 必须使用预定义的 LANG_TOKEN_IDS 查表直接获取 128004-128103 ID 范围内的语言 token。

use std::path::Path;

use ndarray::ArrayD;
use ort::session::Session;
use ort::value::Value;
use votex_domain::error::TranslationError;
use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::value_object::TranslationDirection;

use crate::bpe_tokenizer::SentencePieceBpe;
use crate::shared::{EngineState, OrtSessionFactory};

/// M2M-100 特殊 token ID（与 NLLB 相同）
const PAD_TOKEN_ID: i64 = 1;
const EOS_TOKEN_ID: i64 = 2;

/// 语言 token ID 基址（tokenizer.json 中语言 token 从 128004 开始）
/// SentencePiece 只有 128000 词条，语言 token 是 tokenizer.json 额外添加的 103 个 special token。
const LANG_TOKEN_ID_EN: i64 = 128022;
const LANG_TOKEN_ID_ZH: i64 = 128102;

/// 获取语言对应的 token ID（`__{lang}__` 的单一 token ID）
fn get_lang_token_id(lang: &str) -> Option<i64> {
    match lang {
        "af" => Some(128004),
        "am" => Some(128005),
        "ar" => Some(128006),
        "ast" => Some(128007),
        "az" => Some(128008),
        "ba" => Some(128009),
        "be" => Some(128010),
        "bg" => Some(128011),
        "bn" => Some(128012),
        "br" => Some(128013),
        "bs" => Some(128014),
        "ca" => Some(128015),
        "ceb" => Some(128016),
        "cs" => Some(128017),
        "cy" => Some(128018),
        "da" => Some(128019),
        "de" => Some(128020),
        "el" => Some(128021),
        "en" => Some(LANG_TOKEN_ID_EN),
        "es" => Some(128023),
        "et" => Some(128024),
        "fa" => Some(128025),
        "ff" => Some(128026),
        "fi" => Some(128027),
        "fr" => Some(128028),
        "fy" => Some(128029),
        "ga" => Some(128030),
        "gd" => Some(128031),
        "gl" => Some(128032),
        "gu" => Some(128033),
        "ha" => Some(128034),
        "he" => Some(128035),
        "hi" => Some(128036),
        "hr" => Some(128037),
        "ht" => Some(128038),
        "hu" => Some(128039),
        "hy" => Some(128040),
        "id" => Some(128041),
        "ig" => Some(128042),
        "ilo" => Some(128043),
        "is" => Some(128044),
        "it" => Some(128045),
        "ja" => Some(128046),
        "jv" => Some(128047),
        "ka" => Some(128048),
        "kk" => Some(128049),
        "km" => Some(128050),
        "kn" => Some(128051),
        "ko" => Some(128052),
        "lb" => Some(128053),
        "lg" => Some(128054),
        "ln" => Some(128055),
        "lo" => Some(128056),
        "lt" => Some(128057),
        "lv" => Some(128058),
        "mg" => Some(128059),
        "mk" => Some(128060),
        "ml" => Some(128061),
        "mn" => Some(128062),
        "mr" => Some(128063),
        "ms" => Some(128064),
        "my" => Some(128065),
        "ne" => Some(128066),
        "nl" => Some(128067),
        "no" => Some(128068),
        "ns" => Some(128069),
        "oc" => Some(128070),
        "or" => Some(128071),
        "pa" => Some(128072),
        "pl" => Some(128073),
        "ps" => Some(128074),
        "pt" => Some(128075),
        "ro" => Some(128076),
        "ru" => Some(128077),
        "sd" => Some(128078),
        "si" => Some(128079),
        "sk" => Some(128080),
        "sl" => Some(128081),
        "so" => Some(128082),
        "sq" => Some(128083),
        "sr" => Some(128084),
        "ss" => Some(128085),
        "su" => Some(128086),
        "sv" => Some(128087),
        "sw" => Some(128088),
        "ta" => Some(128089),
        "th" => Some(128090),
        "tl" => Some(128091),
        "tn" => Some(128092),
        "tr" => Some(128093),
        "uk" => Some(128094),
        "ur" => Some(128095),
        "uz" => Some(128096),
        "vi" => Some(128097),
        "wo" => Some(128098),
        "xh" => Some(128099),
        "yi" => Some(128100),
        "yo" => Some(128101),
        "zh" => Some(LANG_TOKEN_ID_ZH),
        "zu" => Some(128103),
        _ => None,
    }
}

/// M2M-100 ONNX 翻译引擎
pub struct M2m100Provider {
    encoder_state: EngineState<Session>,
    decoder_state: EngineState<Session>,
    decoder_with_past_state: EngineState<Session>,
    tokenizer: Option<SentencePieceBpe>,
    max_length: usize,
    /// M2M-100 变体：418M 或 1.2B
    variant: ModelVariant,
}

/// M2M-100 模型变体
#[derive(Debug, Clone, Copy, PartialEq)]
enum ModelVariant {
    M418M,
    M1_2B,
}

impl M2m100Provider {
    pub fn new() -> Self {
        Self {
            encoder_state: EngineState::new(),
            decoder_state: EngineState::new(),
            decoder_with_past_state: EngineState::new(),
            tokenizer: None,
            max_length: 512,
            variant: ModelVariant::M418M,
        }
    }

    /// 设置最大生成长度
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// 从模型目录加载
    ///
    /// 按独立文件选择加载策略：
    /// 1. 优先尝试 `encoder_model.int8.onnx` / `decoder_model.int8.onnx`
    /// 2. 回退到 `encoder_model.onnx` / `decoder_model.onnx`
    pub fn load_from_dir(&mut self, model_dir: &Path) -> Result<(), TranslationError> {
        let tokenizer_path = model_dir.join("sentencepiece.bpe.model");

        // 独立选择每个文件：优先 int8 → fp32
        let encoder_candidates = [
            model_dir.join("encoder_model.int8.onnx"),
            model_dir.join("encoder_model.onnx"),
        ];
        let decoder_candidates = [
            model_dir.join("decoder_model.int8.onnx"),
            model_dir.join("decoder_model.onnx"),
        ];

        let encoder_path = encoder_candidates.iter().find(|p| p.exists())
            .ok_or(TranslationError::ModelNotLoaded)?;
        let decoder_path = decoder_candidates.iter().find(|p| p.exists())
            .ok_or(TranslationError::ModelNotLoaded)?;

        // 估算模型变体
        if let Ok(meta) = std::fs::metadata(&encoder_path) {
            if meta.len() > 1_000_000_000 {
                self.variant = ModelVariant::M1_2B;
            }
        }

        // 加载 ONNX sessions
        let encoder_session = OrtSessionFactory::create_raw(&encoder_path)
            .map_err(|e| TranslationError::ApiError(format!("加载编码器失败: {}", e)))?;
        self.encoder_state.load(encoder_session);

        let decoder_session = OrtSessionFactory::create_raw(&decoder_path)
            .map_err(|e| TranslationError::ApiError(format!("加载解码器失败: {}", e)))?;
        self.decoder_state.load(decoder_session);

        // 加载 KV cache 解码器（可选）
        let past_candidates = [
            model_dir.join("decoder_with_past_model.int8.onnx"),
            model_dir.join("decoder_with_past_model.onnx"),
        ];
        if let Some(past_path) = past_candidates.iter().find(|p| p.exists()) {
            match OrtSessionFactory::create_raw(past_path) {
                Ok(session) => {
                    self.decoder_with_past_state.load(session);
                    tracing::info!("M2M-100: 已启用 KV cache 优化");
                }
                Err(e) => {
                    tracing::warn!("M2M-100: KV cache 解码器加载失败，使用基础解码模式: {}", e);
                }
            }
        }

        // 加载分词器
        if tokenizer_path.exists() {
            let mut sp = SentencePieceBpe::load(&tokenizer_path)
                .map_err(|e| TranslationError::ApiError(format!("加载分词器失败: {}", e)))?;
            let vocab_size = sp.vocab_size();
            tracing::info!("M2M-100: SentencePiece 分词器加载完成，词汇表大小: {}", vocab_size);

            // 加载 BPE 词汇表映射（tokenizer.json）—— 用于正确的 ID 映射
            let bpe_vocab_path = model_dir.join("tokenizer.json");
            if bpe_vocab_path.exists() {
                sp.load_bpe_vocab_from_json(&bpe_vocab_path)
                    .map_err(|e| TranslationError::ApiError(
                        format!("加载 BPE 词汇表失败: {}", e)
                    ))?;
                tracing::info!("M2M-100: BPE 词汇表加载完成（{} 条目）", sp.bpe_vocab_size());
            } else {
                tracing::warn!("M2M-100: tokenizer.json 不存在，使用 SentencePiece 原始 ID");
            }

            self.tokenizer = Some(sp);
        } else {
            return Err(TranslationError::ModelNotLoaded);
        }

        tracing::info!(
            "M2M-100 ({}) 翻译引擎加载完成: {:?}",
            match self.variant {
                ModelVariant::M418M => "418M",
                ModelVariant::M1_2B => "1.2B",
            },
            model_dir
        );
        Ok(())
    }

    /// 翻译文本（内部方法）—— 自回归编码器-解码器
    fn translate_internal(
        &self,
        text: &str,
        direction: &TranslationDirection,
    ) -> Result<String, TranslationError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or(TranslationError::ModelNotLoaded)?;

        // 确定源语言和目标语言
        let source_lang = direction
            .source_lang()
            .unwrap_or("zh");
        let target_lang = direction
            .target_lang()
            .unwrap_or("en");

        // 获取语言 token ID（不要用 SentencePiece 编码 `__{lang}__`，会拆成 subword）
        let source_lang_token = get_lang_token_id(&source_lang)
            .ok_or(TranslationError::UnsupportedDirection)?;
        let target_lang_token = get_lang_token_id(&target_lang)
            .ok_or(TranslationError::UnsupportedDirection)?;

        // 分词（不包含语言标记，只编文本内容）
        let mut text_ids = tokenizer
            .encode(text)
            .map_err(|e| TranslationError::ApiError(format!("分词失败: {}", e)))?;

        // 构建输入：[源语言 token] + text_ids + [EOS]
        let mut full_input_ids = vec![source_lang_token];
        full_input_ids.append(&mut text_ids);
        full_input_ids.push(EOS_TOKEN_ID);

        tracing::info!(
            "M2M-100 输入: text='{}' → input_ids={:?} (src={}, tgt={})",
            text, full_input_ids, source_lang, target_lang
        );
        tracing::info!("M2M-100 目标语言 token: {}", target_lang_token);

        // ===== 1. 编码器推理 =====
        let seq_len = full_input_ids.len();
        let input_tensor = ndarray::Array2::from_shape_vec((1, seq_len), full_input_ids)
            .map_err(|e| TranslationError::ApiError(format!("创建输入张量失败: {}", e)))?;
        let input_value = Value::from_array(input_tensor)
            .map_err(|e| TranslationError::ApiError(format!("创建输入值失败: {}", e)))?;
        let mask_tensor = ndarray::Array2::<i64>::ones((1, seq_len));
        let mask_value = Value::from_array(mask_tensor)
            .map_err(|e| TranslationError::ApiError(format!("创建注意力掩码失败: {}", e)))?;
        let enc_mask_tensor = ndarray::Array2::<i64>::ones((1, seq_len));
        let enc_mask_value = Value::from_array(enc_mask_tensor)
            .map_err(|e| TranslationError::ApiError(format!("创建编码器掩码失败: {}", e)))?;

        let encoder_hidden: ArrayD<f32> = self.encoder_state.with_mut(|session| {
            let outputs = session
                .run(ort::inputs![
                    "input_ids" => input_value,
                    "attention_mask" => mask_value,
                ])
                .map_err(|e| TranslationError::ApiError(format!("编码器推理失败: {}", e)))?;
            let arr = outputs[0]
                .try_extract_array::<f32>()
                .map_err(|e| TranslationError::ApiError(format!("提取编码器输出失败: {}", e)))?;
            Ok::<ArrayD<f32>, TranslationError>(arr.to_owned())
        })
        .map_err(|_| TranslationError::ApiError("编码器引擎未加载".to_string()))?
        .map_err(|e| e)?;

        // ===== 2. 自回归解码 =====
        let max_len = self.max_length;
        let mut gen_ids: Vec<i64> = vec![EOS_TOKEN_ID];

        for step in 0..max_len {
            let dec_input = ndarray::Array2::from_shape_vec((1, gen_ids.len()), gen_ids.clone())
                .map_err(|e| TranslationError::ApiError(format!("创建解码输入失败: {}", e)))?;
            let dec_val = Value::from_array(dec_input)
                .map_err(|e| TranslationError::ApiError(format!("创建解码值失败: {}", e)))?;
            let enc_val = Value::from_array(encoder_hidden.clone())
                .map_err(|e| TranslationError::ApiError(format!("创建 hidden 值失败: {}", e)))?;

            let raw_logits = self.decoder_state.with_mut(|session| {
                let outputs = session
                    .run(ort::inputs![
                        "encoder_attention_mask" => &enc_mask_value,
                        "input_ids" => dec_val,
                        "encoder_hidden_states" => enc_val,
                    ])
                    .map_err(|e| TranslationError::ApiError(format!("解码器推理失败: {}", e)))?;
                let logits = outputs[0]
                    .try_extract_array::<f32>()
                    .map_err(|e| TranslationError::ApiError(format!("提取 logits 失败: {}", e)))?;
                Ok::<ArrayD<f32>, TranslationError>(logits.to_owned())
            })
            .map_err(|_| TranslationError::ApiError("解码器引擎未加载".to_string()))?
            .map_err(|e| e)?;

            // 贪心解码
            let token = greedy_argmax_decoder(&raw_logits.view());

            // M2M-100 强制第一个生成 token 为目标语言 token（forced_bos_token_id）
            let final_token = if step == 0 {
                target_lang_token as u32
            } else {
                token
            };

            if final_token == EOS_TOKEN_ID as u32 {
                tracing::debug!("M2M-100 解码结束: step={}, token=EOS", step);
                break;
            }

            gen_ids.push(final_token as i64);
            if step == 0 || step % 10 == 9 {
                tracing::debug!(
                    "M2M-100 解码步 {}/{}: token_id={}",
                    step, max_len, final_token
                );
            }
        }

        // ===== 3. 解码输出 =====
        // 跳过 decoder_start（EOS），过滤所有语言 token（>=128000）和特殊 token
        let content: Vec<u32> = gen_ids
            .iter()
            .filter(|&&id| {
                id != PAD_TOKEN_ID
                    && id != EOS_TOKEN_ID
                    && id < 128000 // 语言 token（128004-128103）超出 SentencePiece 范围
            })
            .map(|&id| id as u32)
            .collect();

        tracing::info!("M2M-100 生成 IDs: {:?}", gen_ids);
        let translated = tokenizer
            .decode_piece_ids(&content)
            .map_err(|e| TranslationError::ApiError(format!("解码失败: {}", e)))?;
        tracing::info!("M2M-100 输出: '{}'", translated);

        Ok(translated.trim().to_string())
    }
}

impl TranslationProvider for M2m100Provider {
    fn name(&self) -> &str {
        "m2m-100"
    }

    fn translate(
        &self,
        text: &str,
        direction: TranslationDirection,
    ) -> Result<String, TranslationError> {
        if text.trim().is_empty() {
            return Err(TranslationError::EmptyText);
        }
        self.translate_internal(text, &direction)
    }

    fn load(&mut self, model_dir: &Path) -> Result<(), TranslationError> {
        self.load_from_dir(model_dir)
    }

    fn unload(&mut self) {
        self.encoder_state.unload();
        self.decoder_state.unload();
        self.decoder_with_past_state.unload();
    }

    fn is_loaded(&self) -> bool {
        self.encoder_state.is_loaded() && self.decoder_state.is_loaded() && self.tokenizer.is_some()
    }

    fn max_input_chars(&self) -> usize {
        // M2M-100 上限 512 token，中文按 ~1.5 字/token 保守折算
        512 * 2
    }
}

/// 贪心解码：从 decoder logits 的最后一个位置选出概率最高的 token
fn greedy_argmax_decoder(logits: &ndarray::ArrayViewD<f32>) -> u32 {
    let shape = logits.shape();
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
}