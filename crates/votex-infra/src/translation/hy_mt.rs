//! HY-MT1.5 翻译引擎（ONNX 推理 · 自回归解码）
//!
//! 使用腾讯混元 HY-MT1.5 模型通过 ONNX Runtime 实现完全离线翻译。
//! HY-MT1.5 是 LLaMA-like Decoder-only 架构（CausalLM），支持 33 种语言 + 5 种少数民族语言。
//!
//! # 推理方式
//! - Decoder-only 自回归文本生成（逐个 token 解码）
//! - Prompt 模板：`Translate {source_lang} to {target_lang}: {text}`
//! - 贪心解码（argmax），支持 int4 量化推理
//!
//! # ONNX 模型文件
//! 从 HuggingFace onnx-community 获取预导出 ONNX 模型：
//! - `models/translation/hy-mt-1.5/model_q4.onnx` — int4 量化版（推荐 CPU，~1.4GB）
//! - `models/translation/hy-mt-1.5/model.onnx` — FP32 完整版
//! - `models/translation/hy-mt-1.5/model_q4.onnx_data` — Q4 权重数据
//! - `models/translation/hy-mt-1.5/tokenizer.json` — HuggingFace Tokenizer
//! - `models/translation/hy-mt-1.5/config.json` — 模型配置
//!
//! # 特殊 Token
//! | Token | ID |
//! |-------|----|
//! | BOS   | 120000 |
//! | EOS   | 120020 |
//! | PAD   | 120002 |
//!
//! # 链接
//! - 原始模型：<https://huggingface.co/tencent/HY-MT1.5-1.8B>
//! - 社区 ONNX：<https://huggingface.co/onnx-community/HY-MT1.5-1.8B-ONNX>

use std::path::Path;

use ndarray::{Array2, ArrayD, IxDyn};
use ort::session::Session;
use ort::value::Value;
use votex_domain::error::TranslationError;
use votex_domain::translation::options::TranslationHints;
use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::value_object::TranslationDirection;

use crate::shared::{EngineState, OrtSessionFactory};

// ===================== 特殊 Token ID（HY-MT1.5 配置值） =====================

const EOS_TOKEN_ID: i64 = 120_020;
const PAD_TOKEN_ID: i64 = 120_002;
const VOCAB_SIZE: usize = 120_818;
const MAX_LENGTH: usize = 512;
const NUM_LAYERS: usize = 32;
const NUM_KV_HEADS: usize = 4;
const HEAD_DIM: usize = 128;

// ===================== HY-MT1.5 Provider =====================

/// HY-MT1.5 Decoder-only ONNX 翻译引擎
pub struct HyMtProvider {
    state: EngineState<Session>,
    /// HuggingFace tokenizers 分词器
    ///
    /// 走 `EngineState`：`TranslationProvider::load` 是 `&self`，
    /// 裸 `Option` 无法在共享实例上写入。
    tokenizer: EngineState<tokenizers::Tokenizer>,
    max_length: usize,
    /// 少数民族语言支持（藏/维/蒙/壮/彝）
    minority_langs: Vec<String>,
}

impl HyMtProvider {
    pub fn new() -> Self {
        Self {
            state: EngineState::new(),
            tokenizer: EngineState::new(),
            max_length: MAX_LENGTH,
            minority_langs: vec![
                "bo".into(), // 藏语
                "ug".into(), // 维吾尔语
                "mn".into(), // 蒙古语
                "za".into(), // 壮语
                "ii".into(), // 彝语
            ],
        }
    }

    /// 设置最大生成长度
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// 从模型目录加载
    ///
    /// 加载策略：
    /// 1. 优先尝试 `model_q4.onnx`（int4 量化，最小体积）
    /// 2. 回退到 `model.onnx`（FP32 完整版）
    pub fn load_from_dir(&self, model_dir: &Path) -> Result<(), TranslationError> {
        // 优先加载 Q4 量化模型（CPU 友好）
        let model_path = if model_dir.join("model_q4.onnx").exists() {
            tracing::info!("HY-MT1.5: 使用 Q4 量化模型");
            model_dir.join("model_q4.onnx")
        } else if model_dir.join("model.onnx").exists() {
            tracing::info!("HY-MT1.5: 使用 FP32 ONNX 模型");
            model_dir.join("model.onnx")
        } else {
            return Err(TranslationError::ModelNotLoaded);
        };

        let session = OrtSessionFactory::create_raw(&model_path)
            .map_err(|e| TranslationError::ApiError(format!("加载 ONNX 模型失败: {}", e)))?;
        self.state.load(session);

        // 加载 Tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        if tokenizer_path.exists() {
            let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| TranslationError::ApiError(format!("加载 tokenizer 失败: {}", e)))?;
            self.tokenizer.load(tokenizer);
        } else {
            return Err(TranslationError::ModelNotLoaded);
        }

        tracing::info!("HY-MT1.5 翻译引擎加载完成: {:?}", model_dir);
        Ok(())
    }

    /// 构建翻译 prompt（HY-MT1.5 Chat 模板格式）
    ///
    /// 模型使用 ChatML 风格的对话模板，必须包含特殊 token 标记角色边界：
    /// `<｜hy_begin▁of▁sentence｜><｜hy_User｜>指令<｜hy_Assistant｜>`
    /// `<｜hy_Assistant｜>` token 标识助手开始生成的起始位置。
    #[allow(dead_code)] // 保留便捷重载，供测试与调试使用
    fn build_prompt(source_lang: &str, target_lang: &str, text: &str) -> String {
        Self::build_prompt_with_glossary(source_lang, target_lang, text, None)
    }

    /// 构建带术语表的翻译 prompt
    ///
    /// HY-MT1.5 是 Chat 模型，能理解「参考术语表」这类指令。
    /// 术语表以 `原文->译文 #备注` 多行形式给出；为空时退化为普通指令。
    fn build_prompt_with_glossary(
        source_lang: &str,
        target_lang: &str,
        text: &str,
        glossary_text: Option<&str>,
    ) -> String {
        let glossary = glossary_text.map(str::trim).unwrap_or("");
        let instruction = if glossary.is_empty() {
            format!("Translate {} to {}: {}", source_lang, target_lang, text)
        } else {
            format!(
                "参考以下术语表（格式为 原文->译文 #备注）：\n{}\n\nTranslate {} to {}: {}",
                glossary, source_lang, target_lang, text
            )
        };
        format!(
            "<｜hy_begin▁of▁sentence｜><｜hy_User｜>{}<｜hy_Assistant｜>",
            instruction
        )
    }

    /// 翻译文本（内部方法）
    ///
    /// # 自回归解码流程
    /// 1. 编码 prompt → token IDs
    /// 2. 循环：ONNX 推理 → 取最后位置 logits → argmax → 检查 EOS
    /// 3. 解码生成的 token（跳过 prompt 部分）
    fn translate_internal(
        &self,
        text: &str,
        direction: &TranslationDirection,
    ) -> Result<String, TranslationError> {
        self.translate_internal_with_glossary(text, direction, None)
    }

    /// 翻译文本（可携带术语表指令）
    fn translate_internal_with_glossary(
        &self,
        text: &str,
        direction: &TranslationDirection,
        glossary_text: Option<&str>,
    ) -> Result<String, TranslationError> {
        // 守卫必须活到推理与解码结束。
        let tokenizer_guard = self.tokenizer.get();
        let tokenizer = tokenizer_guard
            .as_ref()
            .and_then(|g| g.as_ref())
            .ok_or(TranslationError::ModelNotLoaded)?;

        // 确定源语言和目标语言
        let source = direction.source_lang().unwrap_or("auto");
        let target = direction.target_lang().unwrap_or("en");

        // 构建 prompt（术语表不为空时一并写入指令）
        let prompt = Self::build_prompt_with_glossary(source, target, text.trim(), glossary_text);

        // 编码 prompt
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| TranslationError::ApiError(format!("编码失败: {}", e)))?;

        let mut tokens: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let prompt_len = tokens.len();

        // ===== 自回归生成循环 =====
        // HY-MT1.5 ONNX 模型使用 GroupQueryAttention + KV Cache，
        // 需要为 32 层每层传入空的 past_key_values 张量（首次调用）。
        // 每次推理传入完整序列（全量 prompt + 已生成 tokens），
        // KV Cache 为空（形状 [1, 4, 0, 128]）让模型自行计算全量 attention。
        for _step in 0..self.max_length {
            let seq_len = tokens.len();

            // --- 准备 3 个必需输入 ---
            let input_tensor = Array2::from_shape_vec((1, seq_len), tokens.clone())
                .map_err(|e| TranslationError::ApiError(format!("创建输入张量失败: {}", e)))?;
            let attention_mask: Array2<i64> = Array2::ones((1, seq_len));
            let position_ids: Vec<i64> = (0..seq_len as i64).collect();
            let pos_tensor = Array2::from_shape_vec((1, seq_len), position_ids)
                .map_err(|e| TranslationError::ApiError(format!("创建 position_ids 失败: {}", e)))?;

            let input_val = Value::from_array(input_tensor)
                .map_err(|e| TranslationError::ApiError(format!("创建输入值失败: {}", e)))?;
            let mask_val = Value::from_array(attention_mask)
                .map_err(|e| TranslationError::ApiError(format!("创建 mask 值失败: {}", e)))?;
            let pos_val = Value::from_array(pos_tensor)
                .map_err(|e| TranslationError::ApiError(format!("创建 position_ids 值失败: {}", e)))?;

            // --- 构建 67 个输入的动态列表 ---
            // 模型要求：input_ids + attention_mask + position_ids + 64 × past_key_values
            let mut session_inputs: Vec<(String, ort::value::Value)> = Vec::with_capacity(67);
            session_inputs.push(("input_ids".into(), input_val.into()));
            session_inputs.push(("attention_mask".into(), mask_val.into()));
            session_inputs.push(("position_ids".into(), pos_val.into()));

            // 为 32 层每层创建空的 KV Cache（past_seq_len = 0）
            let empty_kv_shape = IxDyn(&[1, NUM_KV_HEADS, 0, HEAD_DIM]);
            for layer in 0..NUM_LAYERS {
                let k_cache = ArrayD::<f32>::from_shape_vec(empty_kv_shape.clone(), vec![])
                    .map_err(|e| TranslationError::ApiError(format!("创建 K cache 失败: {}", e)))?;
                let v_cache = ArrayD::<f32>::from_shape_vec(empty_kv_shape.clone(), vec![])
                    .map_err(|e| TranslationError::ApiError(format!("创建 V cache 失败: {}", e)))?;
                let k_val = Value::from_array(k_cache)
                    .map_err(|e| TranslationError::ApiError(format!("创建 K cache Value 失败: {}", e)))?;
                let v_val = Value::from_array(v_cache)
                    .map_err(|e| TranslationError::ApiError(format!("创建 V cache Value 失败: {}", e)))?;
                session_inputs.push((format!("past_key_values.{}.key", layer), k_val.into()));
                session_inputs.push((format!("past_key_values.{}.value", layer), v_val.into()));
            }

            // --- ONNX 推理 ---
            let next_token = self
                .state
                .with_mut(|session| {
                    let outputs = session
                        .run(session_inputs)
                        .map_err(|e| {
                            TranslationError::ApiError(format!("ONNX 推理失败: {}", e))
                        })?;

                    let logits = outputs[0]
                        .try_extract_array::<f32>()
                        .map_err(|e| {
                            TranslationError::ApiError(format!("提取 logits 失败: {}", e))
                        })?;

                    // 贪心解码：取最后位置的 argmax
                    let last_pos = seq_len - 1;
                    let mut max_idx = 0i64;
                    let mut max_val = f32::NEG_INFINITY;
                    for v in 0..VOCAB_SIZE {
                        let val = logits[[0, last_pos, v]];
                        if val > max_val {
                            max_val = val;
                            max_idx = v as i64;
                        }
                    }

                    Ok::<i64, TranslationError>(max_idx)
                })
                .map_err(|_| TranslationError::ApiError("引擎未加载".to_string()))?
                .map_err(|e| e)?;

            // 检查 EOS/PAD 终止条件
            if next_token == EOS_TOKEN_ID || next_token == PAD_TOKEN_ID {
                break;
            }

            tokens.push(next_token);
        }

        // ===== 解码生成的 token（跳过 prompt） =====
        let generated: Vec<u32> = tokens[prompt_len..]
            .iter()
            .map(|&x| x as u32)
            .collect();

        if generated.is_empty() {
            return Err(TranslationError::ApiError("生成结果为空".into()));
        }

        let translated = tokenizer
            .decode(&generated, true)
            .map_err(|e| TranslationError::ApiError(format!("解码失败: {}", e)))?;

        Ok(translated.trim().to_string())
    }

    /// 检查是否支持指定语言
    pub fn supports_language(&self, lang: &str) -> bool {
        // HY-MT1.5 支持 33 种主要语言 + 5 种少数民族语言
        let supported: &[&str] = &[
            "zh", "en", "ja", "ko", "fr", "de", "es", "pt", "ru", "ar",
            "vi", "th", "id", "ms", "tl", "my", "lo", "km", "mn", "bo",
            "ug", "za", "ii", "ne", "si", "hi", "bn", "ur", "pa", "ta",
            "te", "mr", "gu",
        ];
        supported.contains(&lang) || self.minority_langs.contains(&lang.to_string())
    }
}

impl TranslationProvider for HyMtProvider {
    fn name(&self) -> &str {
        "hy-mt-1.5"
    }

    fn translate(
        &self,
        text: &str,
        direction: TranslationDirection,
    ) -> Result<String, TranslationError> {
        if text.trim().is_empty() {
            return Err(TranslationError::EmptyText);
        }

        // 检查语言支持
        if let Some(target) = direction.target_lang() {
            if !self.supports_language(target) {
                return Err(TranslationError::UnsupportedDirection);
            }
        }

        self.translate_internal(text, &direction)
    }

    fn load(&self, model_dir: &Path) -> Result<(), TranslationError> {
        self.load_from_dir(model_dir)
    }

    fn unload(&self) {
        self.state.unload();
    }

    fn is_loaded(&self) -> bool {
        self.state.is_loaded() && self.tokenizer.is_loaded()
    }

    fn max_input_chars(&self) -> usize {
        // 输入侧 prompt 会额外占用若干 token，这里留出余量
        self.max_length.saturating_mul(2).max(64)
    }

    fn supports_glossary_prompt(&self) -> bool {
        // HY-MT1.5 是 Chat 模型，能理解术语表指令
        true
    }

    fn translate_with_hints(
        &self,
        text: &str,
        direction: TranslationDirection,
        hints: &TranslationHints,
    ) -> Result<String, TranslationError> {
        if text.trim().is_empty() {
            return Err(TranslationError::EmptyText);
        }
        if let Some(target) = direction.target_lang() {
            if !self.supports_language(target) {
                return Err(TranslationError::UnsupportedDirection);
            }
        }
        let glossary = if hints.glossary_text.is_empty() {
            None
        } else {
            Some(hints.glossary_text.as_str())
        };
        self.translate_internal_with_glossary(text, &direction, glossary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_language() {
        let provider = HyMtProvider::new();
        assert!(provider.supports_language("zh"));
        assert!(provider.supports_language("en"));
        assert!(provider.supports_language("bo"));
        assert!(!provider.supports_language("xx"));
    }

    #[test]
    fn test_build_prompt() {
        let prompt = HyMtProvider::build_prompt("Chinese", "English", "你好世界");
        assert_eq!(
            prompt,
            "<｜hy_begin▁of▁sentence｜><｜hy_User｜>Translate Chinese to English: 你好世界<｜hy_Assistant｜>"
        );
    }

    #[test]
    fn test_new_empty_engine() {
        let provider = HyMtProvider::new();
        assert!(!provider.state.is_loaded());
        assert_eq!(provider.max_length, MAX_LENGTH);
    }
}