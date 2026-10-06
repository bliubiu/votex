//! Opus-MT 离线翻译引擎（ONNX 推理）
//!
//! 使用 Helsinki-NLP 的 MarianMT ONNX 模型通过 ONNX Runtime 实现完全离线翻译。
//! 支持双向翻译（zh↔en），需分别加载 opus-mt-zh-en 和 opus-mt-en-zh 模型。
//! 模型约 350MB/方向（encoder + decoder + decoder_with_past），适合 CPU 低资源运行。
//!
//! # 模型文件（每方向）
//! - `encoder_model.onnx` - 编码器
//! - `decoder_model.onnx` - 解码器
//! - `decoder_with_past_model.onnx` - 带 KV cache 的解码器
//! - `source.spm` - 源语言 SentencePiece 分词器
//! - `target.spm` - 目标语言 SentencePiece 分词器
//! - `vocab.json` - 词汇表映射（SentencePiece piece → 模型 ID）
//!
//! # 方向路由
//! - zh→en: 加载 opus-mt-zh-en 模型，用 source.spm（中文）编码，target.spm（英文）解码
//! - en→zh: 加载 opus-mt-en-zh 模型，用 source.spm（英文）编码，target.spm（中文）解码

use std::collections::HashMap;
use std::path::Path;

use ndarray::{Array2, ArrayD};
use ort::session::Session;
use ort::value::Value;
use votex_domain::error::TranslationError;
use votex_domain::translation::provider::TranslationProvider;
use votex_domain::translation::value_object::TranslationDirection;

use crate::bpe_tokenizer::SentencePieceBpe;
use crate::shared::{EngineState, OrtSessionFactory};

/// KV cache 类型：present 输出名 → 张量数据
type KvCache = HashMap<String, ArrayD<f32>>;

const DECODER_START_TOKEN: i64 = 65000; // <pad>
const EOS_TOKEN_ID: i64 = 0;            // </s>

/// 单个方向的模型数据（zh→en 或 en→zh）
struct DirectionalModel {
    encoder: EngineState<Session>,
    decoder: EngineState<Session>,
    decoder_past: Option<EngineState<Session>>,
    source_spm: SentencePieceBpe,
    target_spm: SentencePieceBpe,
    sp_to_model_id: HashMap<Vec<u8>, i64>,
    model_to_target: HashMap<i64, i32>,
}

impl DirectionalModel {
    fn load_from_dir(model_dir: &Path) -> Result<Self, TranslationError> {
        let encoder_path = model_dir.join("encoder_model.onnx");
        let decoder_path = model_dir.join("decoder_model.onnx");
        let decoder_past_path = model_dir.join("decoder_with_past_model.onnx");

        if !encoder_path.exists() || !decoder_path.exists() {
            return Err(TranslationError::ModelNotLoaded);
        }

        let enc = OrtSessionFactory::create_raw(&encoder_path)
            .map_err(|e| TranslationError::ApiError(format!("加载编码器失败: {}", e)))?;
        let encoder = EngineState::new();
        encoder.load(enc);

        let dec = OrtSessionFactory::create_raw(&decoder_path)
            .map_err(|e| TranslationError::ApiError(format!("加载解码器失败: {}", e)))?;
        let decoder = EngineState::new();
        decoder.load(dec);

        let mut decoder_past = None;
        if decoder_past_path.exists() {
            let dp = OrtSessionFactory::create_raw(&decoder_past_path)
                .map_err(|e| TranslationError::ApiError(format!("加载 KV cache 解码器失败: {}", e)))?;
            let dp_state = EngineState::new();
            dp_state.load(dp);
            decoder_past = Some(dp_state);
        }

        let source_spm = {
            let p = model_dir.join("source.spm");
            if p.exists() {
                SentencePieceBpe::load(&p)
                    .map_err(|e| TranslationError::ApiError(format!("加载源语言分词器失败: {}", e)))?
            } else {
                return Err(TranslationError::ModelNotLoaded);
            }
        };
        let target_spm = {
            let p = model_dir.join("target.spm");
            if p.exists() {
                SentencePieceBpe::load(&p)
                    .map_err(|e| TranslationError::ApiError(format!("加载目标语言分词器失败: {}", e)))?
            } else {
                return Err(TranslationError::ModelNotLoaded);
            }
        };

        // 加载 vocab.json 并构建映射表
        let vocab_path = model_dir.join("vocab.json");
        let content = std::fs::read_to_string(&vocab_path)
            .map_err(|e| TranslationError::ApiError(format!("读取 vocab.json 失败: {}", e)))?;
        let vocab: HashMap<String, i64> = serde_json::from_str(&content)
            .map_err(|e| TranslationError::ApiError(format!("解析 vocab.json 失败: {}", e)))?;

        // SP piece 字节串 → 模型输入 ID（编码映射）
        let mut sp2model = HashMap::new();
        for sid in 0..source_spm.vocab_size() as i32 {
            if let Some(bytes) = source_spm.get_piece(sid) {
                let s = String::from_utf8_lossy(bytes);
                if let Some(&mid) = vocab.get(s.as_ref()) {
                    sp2model.insert(bytes.to_vec(), mid);
                }
            }
        }

        // 模型输出 ID → target.spm piece ID（解码映射）
        let mut m2tgt = HashMap::new();
        for sid in 0..target_spm.vocab_size() as i32 {
            if let Some(bytes) = target_spm.get_piece(sid) {
                let s = String::from_utf8_lossy(bytes);
                if let Some(&mid) = vocab.get(s.as_ref()) {
                    m2tgt.insert(mid, sid);
                }
            }
        }

        tracing::info!("Opus-MT 方向模型加载完成: {:?}", model_dir);
        Ok(Self {
            encoder,
            decoder,
            decoder_past,
            source_spm,
            target_spm,
            sp_to_model_id: sp2model,
            model_to_target: m2tgt,
        })
    }

    /// 翻译文本（自回归解码）
    fn translate(&self, text: &str) -> Result<String, TranslationError> {
        // ===== 1. 编码文本 → 模型输入 ID =====
        let sp_ids = self.source_spm.encode(text)
            .map_err(|e| TranslationError::ApiError(format!("分词失败: {}", e)))?;

        let mut input_ids: Vec<i64> = Vec::with_capacity(sp_ids.len() + 1);
        for &sid in &sp_ids {
            if let Some(bytes) = self.source_spm.get_piece(sid as i32) {
                if let Some(&mid) = self.sp_to_model_id.get(bytes) {
                    input_ids.push(mid);
                } else {
                    input_ids.push(1); // <unk>
                }
            }
        }
        input_ids.push(EOS_TOKEN_ID);

        // ===== 2. 编码器推理 =====
        let enc_seq_len = input_ids.len();
        let inp_arr = Array2::from_shape_vec((1, enc_seq_len), input_ids.clone())
            .map_err(|e| TranslationError::ApiError(format!("创建输入张量失败: {}", e)))?;
        let msk_arr = Array2::<i64>::from_shape_vec((1, enc_seq_len), vec![1i64; enc_seq_len])
            .map_err(|e| TranslationError::ApiError(format!("创建 mask 张量失败: {}", e)))?;
        let inp_val: Value = Value::from_array(inp_arr)
            .map_err(|e| TranslationError::ApiError(format!("创建 input_ids 值失败: {}", e)))?.into();
        let msk_val: Value = Value::from_array(msk_arr)
            .map_err(|e| TranslationError::ApiError(format!("创建 mask 值失败: {}", e)))?.into();

        let encoder_hidden: ArrayD<f32> = self.encoder
            .with_mut(|session| {
                let outputs = session.run(ort::inputs![inp_val, msk_val])
                    .map_err(|e| TranslationError::ApiError(format!("编码器推理失败: {}", e)))?;
                let arr = outputs[0].try_extract_array::<f32>()
                    .map_err(|e| TranslationError::ApiError(format!("提取编码器输出失败: {}", e)))?;
                Ok(arr.to_owned())
            })
            .map_err(|_| TranslationError::ApiError("编码器引擎未加载".to_string()))?
            .map_err(|e| e)?;

        // ===== 3. 自回归解码 =====
        // 第一阶段：首步解码（始终使用 decoder_model，产生 24 个 presents 构建 KV cache）
        let mut gen_ids = vec![DECODER_START_TOKEN];
        let mut kv_cache: KvCache = HashMap::new();

        let seq_len = gen_ids.len();
        let dec_inp_arr = Array2::from_shape_vec((1, seq_len), gen_ids.clone())
            .map_err(|e| TranslationError::ApiError(format!("创建解码输入失败: {}", e)))?;
        let dec_inp_val: Value = Value::from_array(dec_inp_arr)
            .map_err(|e| TranslationError::ApiError(format!("创建解码输入值失败: {}", e)))?.into();
        let enc_hid_val: Value = Value::from_array(encoder_hidden.clone())
            .map_err(|e| TranslationError::ApiError(format!("创建 hidden 值失败: {}", e)))?.into();
        let dec_msk_arr = Array2::<i64>::from_shape_vec((1, enc_seq_len), vec![1i64; enc_seq_len])
            .map_err(|e| TranslationError::ApiError(format!("创建 decoder mask 失败: {}", e)))?;
        let dec_msk_val: Value = Value::from_array(dec_msk_arr)
            .map_err(|e| TranslationError::ApiError(format!("创建 decoder mask 值失败: {}", e)))?.into();

        let token = Self::decode_first_step(&self.decoder, &dec_inp_val, &enc_hid_val, &dec_msk_val, &mut kv_cache)?;
        gen_ids.push(token as i64);

        // 第二阶段：后续步解码
        let has_past = self.decoder_past.is_some();
        for _step in 1..64 {
            let dec_slice: Vec<i64> = vec![gen_ids[gen_ids.len() - 1]]; // 仅最新 token
            let dec_inp_arr = Array2::from_shape_vec((1, 1), dec_slice)
                .map_err(|e| TranslationError::ApiError(format!("创建解码输入失败: {}", e)))?;
            let dec_inp_val: Value = Value::from_array(dec_inp_arr)
                .map_err(|e| TranslationError::ApiError(format!("创建解码输入值失败: {}", e)))?.into();
            let enc_hid_val: Value = Value::from_array(encoder_hidden.clone())
                .map_err(|e| TranslationError::ApiError(format!("创建 hidden 值失败: {}", e)))?.into();
            let dec_msk_arr = Array2::<i64>::from_shape_vec((1, enc_seq_len), vec![1i64; enc_seq_len])
                .map_err(|e| TranslationError::ApiError(format!("创建 decoder mask 失败: {}", e)))?;
            let dec_msk_val: Value = Value::from_array(dec_msk_arr)
                .map_err(|e| TranslationError::ApiError(format!("创建 decoder mask 值失败: {}", e)))?.into();

            let token = if has_past {
                Self::decode_next_step(
                    self.decoder_past.as_ref().unwrap(), &dec_inp_val, &enc_hid_val, &dec_msk_val, &mut kv_cache,
                )?
            } else {
                Self::decode_first_step(
                    &self.decoder, &dec_inp_val, &enc_hid_val, &dec_msk_val, &mut kv_cache,
                )?
            };

            gen_ids.push(token as i64);
            if token as i64 == EOS_TOKEN_ID {
                break;
            }
        }

        // ===== 4. 解码输出 =====
        let content: Vec<i64> = gen_ids.iter()
            .filter(|&&id| id != DECODER_START_TOKEN && id != EOS_TOKEN_ID)
            .copied()
            .collect();

        let mut tgt_ids: Vec<u32> = Vec::with_capacity(content.len());
        for &mid in &content {
            if let Some(&tid) = self.model_to_target.get(&mid) {
                tgt_ids.push(tid as u32);
            }
        }

        let translated = self.target_spm
            .decode_piece_ids(&tgt_ids)
            .map_err(|e| TranslationError::ApiError(format!("解码失败: {}", e)))?;

        Ok(translated.trim().to_string())
    }

    /// 首步解码：使用 decoder_model（3 输入，25 输出：logits + 24 presents）
    /// 返回 token，同时构建完整的 KV cache
    fn decode_first_step(
        decoder: &EngineState<Session>,
        dec_input: &Value,
        encoder_hidden: &Value,
        encoder_mask: &Value,
        kv_cache: &mut KvCache,
    ) -> Result<u32, TranslationError> {
        decoder.with_mut(|session| {
            let outputs = session.run(ort::inputs![encoder_mask, dec_input, encoder_hidden])
                .map_err(|e| TranslationError::ApiError(format!("首步解码推理失败: {}", e)))?;

            let logits = outputs[0].try_extract_array::<f32>()
                .map_err(|e| TranslationError::ApiError(format!("提取 logits 失败: {}", e)))?;
            let token = greedy_argmax(&logits);

            // 提取 24 个 presents 构建 KV cache
            let present_names: [&str; 24] = [
                "present.0.decoder.key", "present.0.decoder.value",
                "present.0.encoder.key", "present.0.encoder.value",
                "present.1.decoder.key", "present.1.decoder.value",
                "present.1.encoder.key", "present.1.encoder.value",
                "present.2.decoder.key", "present.2.decoder.value",
                "present.2.encoder.key", "present.2.encoder.value",
                "present.3.decoder.key", "present.3.decoder.value",
                "present.3.encoder.key", "present.3.encoder.value",
                "present.4.decoder.key", "present.4.decoder.value",
                "present.4.encoder.key", "present.4.encoder.value",
                "present.5.decoder.key", "present.5.decoder.value",
                "present.5.encoder.key", "present.5.encoder.value",
            ];
            for (i, name) in present_names.iter().enumerate() {
                let arr = outputs[1 + i].try_extract_array::<f32>()
                    .map_err(|e| TranslationError::ApiError(format!("提取 {} 失败: {}", name, e)))?;
                kv_cache.insert(name.to_string(), arr.to_owned());
            }

            Ok(token)
        })
        .map_err(|_| TranslationError::ApiError("解码器引擎未加载".to_string()))?
        .map_err(|e| e)
    }

    /// 后续步解码：使用 decoder_with_past_model（26 输入，13 输出：logits + 12 decoder presents）
    /// 返回 token，同时更新 KV cache 中的 decoder 部分
    fn decode_next_step(
        decoder_past: &EngineState<Session>,
        dec_input: &Value,
        _encoder_hidden: &Value,
        encoder_mask: &Value,
        kv_cache: &mut KvCache,
    ) -> Result<u32, TranslationError> {
        const NUM_LAYERS: usize = 6;

        decoder_past.with_mut(|session| {
            // 构建 26 个输入：2 个命名输入 + 24 个 past_key_values
            let mut inputs = ort::inputs![
                "encoder_attention_mask" => encoder_mask,
                "input_ids" => dec_input,
            ];

            for layer in 0..NUM_LAYERS {
                for suffix in ["decoder.key", "decoder.value", "encoder.key", "encoder.value"] {
                    let present_name = format!("present.{}.{}", layer, suffix);
                    let past_name = format!("past_key_values.{}.{}", layer, suffix);
                    if let Some(arr) = kv_cache.get(&present_name) {
                        let val = Value::from_array(arr.clone())
                            .map_err(|e| TranslationError::ApiError(
                                format!("创建 {} 值失败: {}", past_name, e)))?;
                        inputs.push((past_name.into(), val.into()));
                    }
                }
            }

            let outputs = session.run(inputs)
                .map_err(|e| TranslationError::ApiError(format!("KV cache 解码推理失败: {}", e)))?;

            let logits = outputs[0].try_extract_array::<f32>()
                .map_err(|e| TranslationError::ApiError(format!("提取 logits 失败: {}", e)))?;
            let token = greedy_argmax(&logits);

            // 更新 decoder 部分的 KV cache（6 层 × 2 = 12 个）
            for layer in 0..NUM_LAYERS {
                for (offset, suffix) in ["decoder.key", "decoder.value"].iter().enumerate() {
                    let name = format!("present.{}.{}", layer, suffix);
                    let idx = 1 + layer * 2 + offset;
                    let arr = outputs[idx].try_extract_array::<f32>()
                        .map_err(|e| TranslationError::ApiError(
                            format!("提取 {} 失败: {}", name, e)))?;
                    kv_cache.insert(name, arr.to_owned());
                }
            }

            Ok(token)
        })
        .map_err(|_| TranslationError::ApiError("解码器引擎未加载".to_string()))?
        .map_err(|e| e)
    }
}

/// Opus-MT ONNX 翻译引擎（支持双向 zh↔en）
pub struct OpusMtProvider {
    /// zh→en 方向模型
    ///
    /// 走 `EngineState`：`TranslationProvider::load` 是 `&self`，
    /// 裸 `Option` 无法在共享实例上写入。
    /// 内层 `DirectionalModel` 本身已全部由 `EngineState` 承载，
    /// 因此整个模型对象可安全跨线程共享。
    zh_en: EngineState<DirectionalModel>,
    /// en→zh 方向模型
    en_zh: EngineState<DirectionalModel>,
}

impl OpusMtProvider {
    pub fn new() -> Self {
        Self {
            zh_en: EngineState::new(),
            en_zh: EngineState::new(),
        }
    }

    /// 加载 zh→en 模型
    pub fn load_zh_en_from_dir(&self, model_dir: &Path) -> Result<(), TranslationError> {
        let model = DirectionalModel::load_from_dir(model_dir)?;
        self.zh_en.load(model);
        Ok(())
    }

    /// 加载 en→zh 模型
    pub fn load_en_zh_from_dir(&self, model_dir: &Path) -> Result<(), TranslationError> {
        let model = DirectionalModel::load_from_dir(model_dir)?;
        self.en_zh.load(model);
        Ok(())
    }

    /// 兼容旧接口（默认加载 zh→en）
    pub fn load_from_dir(&self, model_dir: &Path) -> Result<(), TranslationError> {
        self.load_zh_en_from_dir(model_dir)
    }
}

impl TranslationProvider for OpusMtProvider {
    fn name(&self) -> &str {
        "opus-mt"
    }

    fn translate(
        &self,
        text: &str,
        direction: TranslationDirection,
    ) -> Result<String, TranslationError> {
        if text.trim().is_empty() {
            return Err(TranslationError::EmptyText);
        }

        // 守卫必须活到 `model.translate()` 结束：
        // `and_then(|g| g.as_ref())` 会让守卫在闭包返回时析构，
        // 借出的 `&DirectionalModel` 随之失效。
        let (zh_en_guard, en_zh_guard) = (self.zh_en.get(), self.en_zh.get());
        let model = match direction {
            TranslationDirection::ZhToEn
            | TranslationDirection::Auto
            | TranslationDirection::ByLanguagePair { source: _, target: _ } => {
                zh_en_guard
                    .as_ref()
                    .and_then(|g| g.as_ref())
                    .ok_or(TranslationError::ModelNotLoaded)?
            }
            TranslationDirection::EnToZh => {
                en_zh_guard
                    .as_ref()
                    .and_then(|g| g.as_ref())
                    .ok_or(TranslationError::UnsupportedDirection)?
            }
        };

        // 如果有指定的语言对，检查是否能处理
        if let TranslationDirection::ByLanguagePair { ref source, ref target } = direction {
            if source != "zh" || target != "en" {
                return Err(TranslationError::UnsupportedDirection);
            }
        }

        model.translate(text)
    }

    fn load(&self, model_dir: &Path) -> Result<(), TranslationError> {
        // 单个目录只对应一个方向；默认按 zh→en 加载，保持与旧接口一致
        self.load_from_dir(model_dir)
    }

    fn unload(&self) {
        self.zh_en.unload();
        self.en_zh.unload();
    }

    fn is_loaded(&self) -> bool {
        self.zh_en.is_loaded() || self.en_zh.is_loaded()
    }

    fn supported_pairs(&self) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        if self.zh_en.is_loaded() {
            pairs.push(("zh".to_string(), "en".to_string()));
        }
        if self.en_zh.is_loaded() {
            pairs.push(("en".to_string(), "zh".to_string()));
        }
        pairs
    }

    fn max_input_chars(&self) -> usize {
        // Opus-MT 解码上限 64 步，输入侧同样不宜过长
        128
    }
}

/// 贪心解码：从 logits 中选择概率最高的 token
fn greedy_argmax(logits: &ndarray::ArrayViewD<f32>) -> u32 {
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
    fn test_opus_mt_fallback_decoder() {
        // 验证无 KV cache 时的回退解码路径是否工作
        // 这个测试不需要实际模型，只验证 greedy_argmax
        let logits = ndarray::array![[[0.1, 0.2, 0.3, 0.4]]].into_dyn();
        let token = greedy_argmax(&logits.view());
        assert_eq!(token, 3);
    }
}
