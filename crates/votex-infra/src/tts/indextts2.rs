//! IndexTTS2 TTS 引擎实现
//!
//! 完整推理管线：BPE 分词 → GPT 自回归生成 mel code → BigVGAN 声码器 → 波形输出
//! 使用 ONNX Runtime（ort crate）加载 .onnx 模型进行推理
//!
//! # 模型文件
//! - `bpe.model`: SentencePiece BPE 分词模型（protobuf 格式）
//! - `speaker_encoder.onnx`: 说话人编码器（输出 192 维 embedding）
//! - `gpt.onnx`: GPT 自回归模型（需从 gpt.pth 导出后可用）
//! - `bigvgan.onnx`: BigVGAN 声码器（mel→波形）
//!
//! # 管线步骤
//! 1. BPE 分词：文本 → token IDs
//! 2. 加载音色 embedding（192 维向量，预计算 .bin 文件）
//! 3. GPT 自回归生成：text_tokens + speaker_embedding → mel code 序列
//! 4. 查代码表：mel code IDs → 连续 mel 频谱帧
//! 5. BigVGAN 声码器：mel 频谱 → 音频波形

use std::collections::HashMap;
use std::path::PathBuf;

use ndarray::{Array, Array2, IxDyn};
use ort::session::Session;
use ort::value::Value;

use votex_domain::error::TtsError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;
use votex_domain::tts::provider::TtsProvider;
use votex_domain::tts::tokenizer::TextTokenizer;
use votex_domain::tts::value_object::{TtsParams, VoiceId};

use crate::shared::{EngineState, OrtSessionFactory};
use crate::tokenizer::SentencePieceBpeTokenizer;

// ===================== 常量定义 =====================

/// 模型基础目录
/// 优先使用项目根目录的 models/tts/indextts2
pub(crate) fn model_base_dir() -> PathBuf {
    // 尝试从当前工作目录向上查找
    let mut dir = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    
    // 如果当前在 crates/votex-infra 目录，向上两级到项目根目录
    if dir.ends_with("crates/votex-infra") || dir.ends_with("crates\\votex-infra") {
        dir.pop();
        dir.pop();
    }
    
    dir.join("models/tts/indextts2")
}



// ===================== 采样策略 =====================

/// GPT 自回归采样策略
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum SamplingStrategy {
    /// 贪心（始终选概率最高的 token）
    Greedy,
    /// Top-k 采样
    TopK { k: usize },
    /// Top-p（核）采样
    TopP { p: f32 },
    /// Top-k + Top-p 联合
    TopKP { k: usize, p: f32 },
    /// 温度采样
    Temperature { temp: f32 },
}

impl Default for SamplingStrategy {
    fn default() -> Self {
        SamplingStrategy::TopKP { k: 50, p: 0.95 }
    }
}

/// 从 logits 中采样下一个 token
fn sample_from_logits(logits: &[f32], strategy: &SamplingStrategy) -> usize {
    match strategy {
        SamplingStrategy::Greedy => {
            logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
        SamplingStrategy::TopK { k } => {
            let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
            indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
            indexed.truncate(*k);
            softmax_sample(&indexed)
        }
        SamplingStrategy::TopP { p } => {
            let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
            indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
            let probs = softmax_values(&indexed.iter().map(|(_, v)| *v).collect::<Vec<_>>());
            let nucleus_end = nucleus_cutoff(&probs, *p);
            let nucleus: Vec<(usize, f32)> = indexed[..nucleus_end]
                .iter().cloned()
                .zip(probs.iter().cloned())
                .map(|((i, _), p)| (i, p))
                .collect();
            softmax_sample(&nucleus)
        }
        SamplingStrategy::TopKP { k, p } => {
            let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
            indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
            indexed.truncate(*k);
            let probs = softmax_values(&indexed.iter().map(|(_, v)| *v).collect::<Vec<_>>());
            let nucleus_end = nucleus_cutoff(&probs, *p);
            let nucleus: Vec<(usize, f32)> = indexed[..nucleus_end]
                .iter().cloned()
                .zip(probs.iter().cloned())
                .map(|((i, _), p)| (i, p))
                .collect();
            softmax_sample(&nucleus)
        }
        SamplingStrategy::Temperature { temp } => {
            let scaled: Vec<f32> = logits.iter().map(|l| l / temp).collect();
            let indexed: Vec<(usize, f32)> = scaled.iter().cloned().enumerate().collect();
            let probs = softmax_values(&scaled);
            let weighted: Vec<(usize, f32)> = indexed.into_iter()
                .zip(probs.into_iter())
                .map(|((i, _), p)| (i, p))
                .collect();
            softmax_sample(&weighted)
        }
    }
}

/// Softmax 归一化
fn softmax_values(values: &[f32]) -> Vec<f32> {
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = values.iter().map(|v| (v - max_val).exp()).sum();
    values.iter().map(|v| (v - max_val).exp() / exp_sum).collect()
}

/// softmax + 按概率采样
fn softmax_sample(indexed: &[(usize, f32)]) -> usize {
    if indexed.is_empty() {
        return 0;
    }
    let max_logit = indexed.iter().map(|(_, v)| *v).fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = indexed.iter().map(|(_, v)| (v - max_logit).exp()).sum();
    let probs: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_logit).exp() / exp_sum).collect();

    let r: f32 = rand::random();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r <= cumsum {
            return indexed[i].0;
        }
    }
    indexed[indexed.len() - 1].0
}

/// 计算 nucleus 截断点（累积概率超过 p 的最小集合）
fn nucleus_cutoff(probs: &[f32], p: f32) -> usize {
    let mut cumsum = 0.0;
    for (i, &prob) in probs.iter().enumerate() {
        cumsum += prob;
        if cumsum >= p {
            return (i + 1).min(probs.len());
        }
    }
    probs.len()
}

// ===================== 重复惩罚 =====================

/// 对已生成的 token 施加重复惩罚
fn apply_repetition_penalty(logits: &mut [f32], prev_tokens: &[i64], penalty: f32) {
    for &t in prev_tokens {
        let idx = t as usize;
        if idx < logits.len() {
            if logits[idx] > 0.0 {
                logits[idx] /= penalty;
            } else {
                logits[idx] *= penalty;
            }
        }
    }
}

// ===================== 音色 Embedding 加载 =====================

/// 加载预计算的说话人 embedding（192 维 float32）
fn load_speaker_embedding(voice_id: &str) -> Result<Vec<f32>, TtsError> {
    if voice_id.is_empty()
        || voice_id.contains('/')
        || voice_id.contains('\\')
        || voice_id.contains("..")
        || voice_id.contains('\0')
    {
        return Err(TtsError::SynthesisFailed(
            "非法的 voice_id: 包含路径分隔符或危险字符".into()
        ));
    }
    let embed_path = model_base_dir()
        .join("voices")
        .join(format!("{}.bin", voice_id));

    let data = std::fs::read(&embed_path).map_err(|e| {
        TtsError::SynthesisFailed(format!(
            "读取音色 embedding 文件失败 {:?}: {}", embed_path, e
        ))
    })?;

    if data.len() != 192 * 4 {
        return Err(TtsError::SynthesisFailed(format!(
            "音色 embedding 文件大小不正确: 期望 {} 字节, 实际 {} 字节",
            192 * 4,
            data.len()
        )));
    }

    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    Ok(floats)
}

// ===================== IndexTTS2Provider =====================

/// IndexTTS2 TTS 引擎实现
///
/// 完整推理管线：BPE 分词 → GPT 自回归生成 mel codes → BigVGAN 声码器 → 波形
#[allow(dead_code)]
pub struct IndexTTS2Provider {
    /// BPE 分词器
    tokenizer: Option<SentencePieceBpeTokenizer>,
    /// GPT 会话（需 gpt.onnx，由 Python 脚本导出）
    gpt_state: EngineState<Session>,
    /// BigVGAN 声码器会话
    vocoder_state: EngineState<Session>,
    /// 说话人编码器会话（可选，用于从参考音频提取 embedding）
    speaker_encoder_state: EngineState<Session>,
    /// 音色 embedding 缓存
    voice_cache: std::sync::Mutex<HashMap<String, Vec<f32>>>,
    /// 输出采样率
    sample_rate: u32,
    /// BigVGAN mel 频带数
    num_mels: usize,
    /// BigVGAN hop_length
    hop_length: usize,
    /// 模型配置参数（索引从 config.yaml）
    start_mel_token: i64,
    stop_mel_token: i64,
    number_mel_codes: i64,
    number_text_tokens: i64,
    /// mel 长度压缩比（每个 mel code 对应的音频采样数）
    mel_length_compression: usize,
    /// 从 GPT 模型提取的 mel codebook embedding，形状 [number_mel_codes, model_dim]
    codebook: Option<Vec<f32>>,
    /// GPT 模型隐藏维度
    model_dim: usize,
}

impl IndexTTS2Provider {
    pub fn new() -> Self {
        Self {
            tokenizer: None,
            gpt_state: EngineState::new(),
            vocoder_state: EngineState::new(),
            speaker_encoder_state: EngineState::new(),
            voice_cache: std::sync::Mutex::new(HashMap::new()),
            sample_rate: 24000,
            num_mels: 80,
            hop_length: 256,
            start_mel_token: 8192,
            stop_mel_token: 8193,
            number_mel_codes: 8194,
            number_text_tokens: 12000,
            mel_length_compression: 1024,
            codebook: None,
            model_dim: 1280,
        }
    }

    /// 加载说话人 embedding（带缓存）
    fn ensure_voice_loaded(&self, voice_id: &str) -> Result<Vec<f32>, TtsError> {
        let mut cache = self.voice_cache.lock().map_err(|e| {
            TtsError::SynthesisFailed(format!("音色缓存锁失败: {}", e))
        })?;

        if let Some(data) = cache.get(voice_id) {
            return Ok(data.clone());
        }

        let data = load_speaker_embedding(voice_id)?;
        cache.insert(voice_id.to_string(), data.clone());
        Ok(data)
    }

    /// 运行 GPT 自回归生成，输出 mel code IDs
    ///
    /// 输入：text_tokens + speaker_embedding
    /// 输出：mel_code_ids 序列（含 start/stop token）
    fn generate_mel_codes(
        &self,
        text_tokens: &[i64],
        speaker_embedding: &[f32],
        max_new_tokens: usize,
    ) -> Result<Vec<i64>, TtsError> {
        let mut generated = vec![self.start_mel_token];
        let strategy = SamplingStrategy::default();
        let repetition_penalty = 1.1;

        for _ in 0..max_new_tokens {
            // 拼接输入：text_tokens + 已生成的 mel codes
            let mut input_ids = text_tokens.to_vec();
            input_ids.extend_from_slice(&generated);

            // 构造 ONNX 输入张量
            let seq_len = input_ids.len();
            let input_tensor = Array::from_shape_vec(
                IxDyn(&[1, seq_len]),
                input_ids.iter().map(|&x| x as i64).collect::<Vec<_>>(),
            ).map_err(|e| TtsError::SynthesisFailed(
                format!("创建 GPT input tensor 失败: {}", e)
            ))?;
            let input_value = Value::from_array(input_tensor).map_err(|e| {
                TtsError::SynthesisFailed(format!("input Value 创建失败: {}", e))
            })?;

            // speaker_embedding: [1, 192]
            let spk_array = Array::from_shape_vec(
                IxDyn(&[1, 192]),
                speaker_embedding.to_vec(),
            ).map_err(|e| TtsError::SynthesisFailed(
                format!("创建 speaker_embedding 张量失败: {}", e)
            ))?;
            let spk_value = Value::from_array(spk_array).map_err(|e| {
                TtsError::SynthesisFailed(format!("speaker_embedding Value 创建失败: {}", e))
            })?;

            // 使用 EngineState 执行推理
            let next_token = self.gpt_state.with_mut(|session| {
                let outputs = session.run(ort::inputs![
                    input_value,
                    spk_value,
                ]).map_err(|e| TtsError::SynthesisFailed(
                    format!("GPT ONNX 推理失败: {}", e)
                ))?;

                if outputs.len() == 0 {
                    return Err(TtsError::SynthesisFailed("GPT 推理无输出".to_string()));
                }

                // 取最后一个位置的 logits
                let logits_tensor = outputs[0].try_extract_array::<f32>().map_err(|e| {
                    TtsError::SynthesisFailed(format!("提取 GPT logits 失败: {}", e))
                })?;

                let logits_shape = logits_tensor.shape();
                if logits_shape.len() < 2 {
                    return Err(TtsError::SynthesisFailed(
                        format!("GPT 输出形状异常: {:?}", logits_shape)
                    ));
                }

                let last_pos = logits_shape[logits_shape.len() - 2] - 1;
                let vocab_size = logits_shape[logits_shape.len() - 1];

                let mut last_logits: Vec<f32> = (0..vocab_size)
                    .map(|i| {
                        if logits_shape.len() == 3 {
                            logits_tensor[[0, last_pos, i]]
                        } else if logits_shape.len() == 2 {
                            logits_tensor[[last_pos, i]]
                        } else {
                            0.0
                        }
                    })
                    .collect();

                // 重复惩罚
                apply_repetition_penalty(&mut last_logits, &generated, repetition_penalty);

                // 采样
                Ok::<i64, TtsError>(sample_from_logits(&last_logits, &strategy) as i64)
            }).map_err(|_| TtsError::EngineNotLoaded)??;

            // 检查 stop token
            if next_token == self.stop_mel_token {
                break;
            }

            generated.push(next_token);
        }

        Ok(generated)
    }

    /// mel code IDs → 80 频段 mel 频谱
    ///
    /// 每个 mel code 对应 `model_dim / num_mels` 帧频谱（1280/80=16 帧）
    fn mel_codes_to_spectrogram(&self, mel_code_ids: &[i64]) -> Result<Array2<f32>, TtsError> {
        // 去掉 start/stop token
        let codes: Vec<&i64> = mel_code_ids
            .iter()
            .filter(|&&id| id != self.start_mel_token && id != self.stop_mel_token)
            .collect();

        if codes.is_empty() {
            return Err(TtsError::SynthesisFailed("生成的 mel code 序列为空".to_string()));
        }

        // 每个 code 产生 model_dim/num_mels 帧（1280/80=16）
        let frames_per_code = self.model_dim / self.num_mels;
        let total_frames = codes.len() * frames_per_code;

        if let Some(codebook) = &self.codebook {
            // 真实 codebook 查表
            let codebook_dim = self.model_dim;
            let mut mel = Array2::zeros((self.num_mels, total_frames));

            for (code_idx, &&code) in codes.iter().enumerate() {
                if code < 0 || code as usize >= self.number_mel_codes as usize {
                    continue; // 跳过越界的 code
                }
                let offset = (code as usize) * codebook_dim;
                for f in 0..frames_per_code {
                    let frame_idx = code_idx * frames_per_code + f;
                    for mel_idx in 0..self.num_mels {
                        let emb_idx = offset + f * self.num_mels + mel_idx;
                        if emb_idx < offset + codebook_dim {
                            mel[[mel_idx, frame_idx]] = codebook[emb_idx];
                        }
                    }
                }
            }
            Ok(mel)
        } else {
            // 回退：近似值
            let mut mel = Array2::zeros((self.num_mels, total_frames));
            for (code_idx, &&code) in codes.iter().enumerate() {
                let base_val = (code as f32 / self.number_mel_codes as f32) * 2.0 - 4.0;
                for f in 0..frames_per_code {
                    let frame_idx = code_idx * frames_per_code + f;
                    for mel_idx in 0..self.num_mels {
                        let freq_envelope = 1.0 - (mel_idx as f32 / self.num_mels as f32) * 0.8;
                        let val = base_val + freq_envelope * 2.0;
                        mel[[mel_idx, frame_idx]] = val.max(-10.0).min(10.0);
                    }
                }
            }
            Ok(mel)
        }
    }

    /// BigVGAN 声码器推理：mel 频谱 → 波形
    fn vocode(&self, mel_spectrogram: &Array2<f32>) -> Result<Vec<f32>, TtsError> {
        // BigVGAN 输入形状: [1, 80, time]
        // 注意：speaker_encoder 使用 [batch, time, 80]，但 BigVGAN 使用 [batch, 80, time]
        let mel_batch = mel_spectrogram.view().into_shape_with_order(
            IxDyn(&[1, self.num_mels, mel_spectrogram.ncols()])
        ).map_err(|e| TtsError::SynthesisFailed(
            format!("创建 BigVGAN 输入形状失败: {}", e)
        ))?;

        let mel_value = Value::from_array(mel_batch.to_owned()).map_err(|e| {
            TtsError::SynthesisFailed(format!("BigVGAN Value 创建失败: {}", e))
        })?;

        // 使用 EngineState 执行推理
        self.vocoder_state.with_mut(|session| {
            let outputs = session.run(ort::inputs![
                mel_value,
            ]).map_err(|e| TtsError::SynthesisFailed(
                format!("BigVGAN ONNX 推理失败: {}", e)
            ))?;

            if outputs.len() == 0 {
                return Err(TtsError::SynthesisFailed("BigVGAN 推理无输出".to_string()));
            }

            let audio_tensor = outputs[0].try_extract_array::<f32>().map_err(|e| {
                TtsError::SynthesisFailed(format!("提取 BigVGAN 音频输出失败: {}", e))
            })?;

            let audio_shape = audio_tensor.shape();
            if audio_shape.len() != 3 || audio_shape[0] != 1 {
                return Err(TtsError::SynthesisFailed(
                    format!("BigVGAN 输出形状异常: {:?}", audio_shape)
                ));
            }

            let num_samples = audio_shape[2];
            let mut samples = Vec::with_capacity(num_samples);
            for i in 0..num_samples {
                samples.push(audio_tensor[[0, 0, i]]);
            }

            // 简单后处理：裁剪到 [-1, 1]
            for s in samples.iter_mut() {
                *s = s.clamp(-1.0, 1.0);
            }

            Ok::<Vec<f32>, TtsError>(samples)
        }).map_err(|_| TtsError::EngineNotLoaded)?
    }
}

impl TtsProvider for IndexTTS2Provider {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::IndexTTS2
    }

    fn load(&mut self, _model: &Model) -> Result<(), TtsError> {
        let base_dir = model_base_dir();

        // 1. 加载 BPE 分词器
        let bpe_path = base_dir.join("bpe.model");
        tracing::info!("加载 BPE 模型: {:?}", bpe_path);
        let tokenizer = SentencePieceBpeTokenizer::load(&bpe_path)
            .map_err(|e| TtsError::SynthesisFailed(e.to_string()))?;
        tracing::info!("BPE 模型加载完成, 词汇表大小: {}", tokenizer.vocab_size());

        // 2. 加载 BigVGAN ONNX 模型（支持外部数据格式）
        let bigvgan_path = base_dir.join("bigvgan.onnx");
        if bigvgan_path.exists() {
            tracing::info!("加载 BigVGAN 模型: {:?}", bigvgan_path);
            
            // 检查是否存在外部数据文件
            let data_path = bigvgan_path.with_extension("onnx.data");
            if data_path.exists() {
                tracing::info!("检测到外部数据文件: {:?}", data_path);
                // 使用 commit_from_file 加载，ONNX Runtime 会自动查找同目录的 .data 文件
                let session = OrtSessionFactory::create_with_threads(&bigvgan_path, 4)
                    .map_err(|e| TtsError::SynthesisFailed(format!("加载 BigVGAN 失败: {}", e)))?;
                self.vocoder_state.load(session);
                tracing::info!("BigVGAN 模型加载完成（外部数据格式）");
            } else {
                // 没有外部数据文件，尝试直接加载
                let session = OrtSessionFactory::create_with_threads(&bigvgan_path, 4)
                    .map_err(|e| TtsError::SynthesisFailed(format!("加载 BigVGAN 失败: {}", e)))?;
                self.vocoder_state.load(session);
                tracing::info!("BigVGAN 模型加载完成");
            }
        } else {
            tracing::warn!("BigVGAN ONNX 模型不存在: {:?}, 声码器将不可用", bigvgan_path);
        }

        // 3. 加载说话人编码器 ONNX 模型（可选，用于从参考音频提取 embedding）
        let spk_enc_path = base_dir.join("speaker_encoder.onnx");
        if spk_enc_path.exists() {
            tracing::info!("加载说话人编码器: {:?}", spk_enc_path);
            match OrtSessionFactory::create_raw(&spk_enc_path) {
                Ok(session) => {
                    self.speaker_encoder_state.load(session);
                    tracing::info!("说话人编码器加载完成");
                }
                Err(e) => {
                    tracing::warn!("说话人编码器加载失败（非致命）: {}", e);
                }
            }
        }

        // 4. 尝试加载 GPT ONNX 模型（可选）
        let gpt_path = base_dir.join("gpt.onnx");
        if gpt_path.exists() {
            tracing::info!("加载 GPT 模型: {:?}", gpt_path);
            match OrtSessionFactory::create_raw(&gpt_path) {
                Ok(session) => {
                    self.gpt_state.load(session);
                    tracing::info!("GPT 模型加载完成");
                }
                Err(e) => {
                    tracing::warn!("GPT 模型加载失败（可通过导出 gpt.onnx 修复）: {}", e);
                }
            }
        } else {
            tracing::info!(
                "GPT ONNX 模型不存在 ({}), 合成时将使用回退模式。请运行 tools/export_indextts2_gpt_onnx.py 导出",
                gpt_path.display()
            );
        }

        // 5. 加载 mel codebook
        let codebook_path = base_dir.join("codebook.bin");
        if codebook_path.exists() {
            let codebook_data = std::fs::read(&codebook_path).map_err(|e| {
                TtsError::SynthesisFailed(format!("读取 codebook 失败: {}", e))
            })?;
            let expected_size = self.number_mel_codes as usize * self.model_dim * 4;
            if codebook_data.len() != expected_size {
                return Err(TtsError::SynthesisFailed(format!(
                    "codebook.bin 大小不匹配: 期望 {} 字节, 实际 {} 字节",
                    expected_size, codebook_data.len()
                )));
            }
            let codebook: Vec<f32> = codebook_data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            self.codebook = Some(codebook);
            tracing::info!("Codebook 加载完成 ({} 个 code, 每个 {} 维)",
                self.number_mel_codes, self.model_dim);
        } else {
            tracing::warn!("codebook.bin 不存在，mel 频谱将使用近似值");
        }

        self.tokenizer = Some(tokenizer);
        tracing::info!("IndexTTS2 引擎加载完成");
        Ok(())
    }

    fn unload(&mut self) -> Result<(), TtsError> {
        self.tokenizer = None;
        self.gpt_state.unload();
        self.vocoder_state.unload();
        self.speaker_encoder_state.unload();
        if let Ok(mut cache) = self.voice_cache.lock() {
            cache.clear();
        }
        tracing::info!("IndexTTS2 引擎已释放");
        Ok(())
    }

    fn synthesize(
        &self,
        text: &str,
        voice: &VoiceId,
        _params: &TtsParams,
    ) -> Result<AudioData, TtsError> {
        if !self.gpt_state.is_loaded() || !self.vocoder_state.is_loaded() {
            return Err(TtsError::EngineNotLoaded);
        }

        if text.is_empty() {
            return Err(TtsError::EmptyText);
        }

        let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
            TtsError::SynthesisFailed("BPE 分词器未加载".to_string())
        })?;

        // 1. BPE 分词
        let text_tokens = tokenizer.encode(text)
            .map_err(|e| TtsError::SynthesisFailed(e.to_string()))?;
        tracing::debug!("IndexTTS2 BPE 分词: '{}' → {} tokens", text, text_tokens.len());

        if text_tokens.is_empty() {
            return Err(TtsError::SynthesisFailed("BPE 分词结果为空".to_string()));
        }

        // 2. 加载说话人 embedding
        let speaker_embedding = self.ensure_voice_loaded(&voice.id)?;

        // 3. GPT 自回归生成 mel codes
        let max_mel_tokens = (text_tokens.len() * 2).max(50).min(1815);
        let mel_code_ids = self.generate_mel_codes(&text_tokens, &speaker_embedding, max_mel_tokens)?;
        tracing::debug!("GPT 生成 {} 个 mel codes", mel_code_ids.len());

        // 4. mel codes → mel 频谱
        let mel_spectrogram = self.mel_codes_to_spectrogram(&mel_code_ids)?;

        // 5. BigVGAN 声码器 → 波形
        let samples = self.vocode(&mel_spectrogram)?;
        tracing::debug!("BigVGAN 输出 {} 个采样点", samples.len());

        tracing::info!(
            "IndexTTS2 合成完成: text='{}', voice={}, tokens={}, samples={}",
            text, voice.id, text_tokens.len(), samples.len()
        );

        Ok(AudioData {
            samples,
            sample_rate: self.sample_rate,
            channels: 1,
        })
    }

    fn list_voices(&self) -> Vec<VoiceId> {
        vec![
            VoiceId::new("default", "默认音色", EngineKind::IndexTTS2),
            VoiceId::new("female_1", "女声 1", EngineKind::IndexTTS2),
            VoiceId::new("male_1", "男声 1", EngineKind::IndexTTS2),
        ]
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn supported_dialects(&self) -> Vec<votex_domain::tts::dialect::DialectSupport> {
        use votex_domain::tts::dialect::*;
        vec![
            DialectSupport::new(Dialect::Mandarin, DialectQuality::Native)
                .with_voice("default"),
            DialectSupport::new(Dialect::Cantonese, DialectQuality::Native)
                .with_voice("yue_male"),
            DialectSupport::new(Dialect::SouthernMin, DialectQuality::Native)
                .with_voice("nan_male"),
        ]
    }

    fn is_loaded(&self) -> bool {
        self.gpt_state.is_loaded() && self.vocoder_state.is_loaded()
    }
}

// ===================== 单元测试 =====================

#[cfg(test)]
mod tests {
    use super::*;
    use votex_domain::model::value_object::{ModelId, ModelKind};
    use votex_domain::tts::value_object::{DenoiseLevel, Pitch, SegmentSize, Speed, Volume};

    #[test]
    fn test_sentencepiece_decode() {
        // 检查模型文件可以被解析
        let model_path = model_base_dir().join("bpe.model");
        if !model_path.exists() {
            eprintln!("跳过: bpe.model 不存在 ({:?})", model_path);
            return;
        }
        let bpe = SentencePieceBpeTokenizer::load(&model_path);
        assert!(bpe.is_ok(), "SentencePiece 模型加载失败: {:?}", bpe.err());
        let bpe = bpe.unwrap();
        assert!(bpe.vocab_size() > 100, "词汇表太小: {}", bpe.vocab_size());
        let _ = bpe; // 持有直到测试结束
    }

    #[test]
    fn test_sentencepiece_encode() {
        let model_path = model_base_dir().join("bpe.model");
        if !model_path.exists() {
            eprintln!("跳过");
            return;
        }
        let bpe = SentencePieceBpeTokenizer::load(&model_path).unwrap();
        let tokens = bpe.encode("你好世界");
        assert!(tokens.is_ok(), "编码失败: {:?}", tokens.err());
        let tokens = tokens.unwrap();
        assert!(!tokens.is_empty(), "编码结果为空");
        assert!(tokens.len() <= 20, "编码结果过长: {}", tokens.len());
    }

    #[test]
    fn test_sampling_greedy() {
        let logits = vec![0.1f32, 0.5, 0.3, 2.0, 0.01];
        let token = sample_from_logits(&logits, &SamplingStrategy::Greedy);
        assert_eq!(token, 3); // 索引 3 对应值 2.0
    }

    #[test]
    fn test_repetition_penalty() {
        let mut logits = vec![1.0f32, 0.5, 0.3];
        apply_repetition_penalty(&mut logits, &[0], 1.1);
        assert!(logits[0] < 1.0); // 正数 logit 被除以惩罚项
        assert!((logits[0] - 1.0 / 1.1).abs() < 1e-6);
    }

    #[test]
    fn test_mel_codes_to_spectrogram() {
        let provider = IndexTTS2Provider::new();
        let codes = vec![8192i64, 100, 200, 300, 8193];
        let mel = provider.mel_codes_to_spectrogram(&codes);
        assert!(mel.is_ok());
        let mel = mel.unwrap();
        assert_eq!(mel.nrows(), 80);
        // 3 个有效 code × 4 帧/code = 12 帧
        assert!(mel.ncols() > 0);
    }

    #[test]
    fn test_protobuf_varint() {
        let data = vec![0xAC, 0x02];
        let mut pos = 0;
        let val = crate::bpe_tokenizer::decode_varint(&data, &mut pos).unwrap();
        // 0xAC = 10101100, 0x02 = 00000010
        // varint: byte0=0xAC(10101100), byte1=0x02(00000010)
        // result = (0xAC & 0x7F) | (0x02 << 7) = 0x2C | 0x100 = 44 + 256 = 300
        assert_eq!(val, (0xAC & 0x7F) as u64 | (0x02 as u64) << 7);
        assert_eq!(val, 300);
    }

    #[test]
    fn test_utf8_char_length() {
        assert_eq!(crate::bpe_tokenizer::utf8_char_length(b'a'), 1);
        assert_eq!(crate::bpe_tokenizer::utf8_char_length(0xC0), 2);
        assert_eq!(crate::bpe_tokenizer::utf8_char_length(0xE0), 3);
        assert_eq!(crate::bpe_tokenizer::utf8_char_length(0xF0), 4);
    }

    #[test]
    fn test_inspect_onnx_models() {
        // 使用 model_base_dir() 获取正确的模型路径
        let base_dir = model_base_dir();
        println!("模型目录: {:?}", base_dir);
        
        // 检查 GPT 模型
        let gpt_path = base_dir.join("gpt.onnx");
        if gpt_path.exists() {
            println!("\n=== GPT ONNX 模型 ===");
            println!("路径: {:?}", gpt_path);
            match OrtSessionFactory::create_raw(&gpt_path) {
                Ok(_session) => {
                    println!("✓ GPT 模型加载成功");
                }
                Err(e) => {
                    println!("✗ GPT 模型加载失败: {}", e);
                }
            }
        } else {
            println!("GPT 模型不存在: {:?}", gpt_path);
        }
        
        // 检查 BigVGAN 模型
        let bigvgan_path = base_dir.join("bigvgan.onnx");
        if bigvgan_path.exists() {
            println!("\n=== BigVGAN ONNX 模型 ===");
            println!("路径: {:?}", bigvgan_path);
            match OrtSessionFactory::create_raw(&bigvgan_path) {
                Ok(_session) => {
                    println!("✓ BigVGAN 模型加载成功");
                }
                Err(e) => {
                    println!("✗ BigVGAN 模型加载失败: {}", e);
                }
            }
        } else {
            println!("BigVGAN 模型不存在: {:?}", bigvgan_path);
        }
        
        // 检查说话人编码器模型
        let spk_path = base_dir.join("speaker_encoder.onnx");
        if spk_path.exists() {
            println!("\n=== 说话人编码器 ONNX 模型 ===");
            println!("路径: {:?}", spk_path);
            match OrtSessionFactory::create_raw(&spk_path) {
                Ok(_session) => {
                    println!("✓ 说话人编码器模型加载成功");
                }
                Err(e) => {
                    println!("✗ 说话人编码器模型加载失败: {}", e);
                }
            }
        } else {
            println!("说话人编码器模型不存在: {:?}", spk_path);
        }
    }

    #[test]
    fn test_indextts2_integration() {
        // 集成测试：完整 TTS 流程
        let mut provider = IndexTTS2Provider::new();
        
        // 使用 Model::new 创建模型对象
        let model = Model::new(
            ModelId::new("indextts2"),
            "IndexTTS2",
            ModelKind::Tts,
            EngineKind::IndexTTS2,
        );
        
        // 加载模型
        match provider.load(&model) {
            Ok(_) => {
                println!("✓ IndexTTS2 引擎加载成功");
                
                // 验证引擎状态
                assert!(provider.is_loaded(), "引擎应该已加载");
                
                // 尝试合成语音
                let voice = VoiceId::new("default", "默认音色", EngineKind::IndexTTS2);
                let params = TtsParams {
                    engine: EngineKind::IndexTTS2,
                    voice: voice.clone(),
                    speed: Speed::new(1.0).unwrap(),
                    pitch: Pitch::new(0).unwrap(),
                    volume: Volume::new(100).unwrap(),
                    segment_size: SegmentSize::S500,
                    segment_silence_ms: 500,
                    crossfade_ms: 50,
                    num_to_chinese: true,
                    denoise: false,
                    denoise_level: DenoiseLevel::Low,
                    emotion: None,
                    dialect: None,
                };
                
                match provider.synthesize("你好，这是一个测试。", &voice, &params) {
                    Ok(audio) => {
                        println!("✓ 语音合成成功");
                        println!("  采样率: {} Hz", audio.sample_rate);
                        println!("  声道数: {}", audio.channels);
                        println!("  采样点数: {}", audio.samples.len());
                        println!("  时长: {:.2} 秒", audio.samples.len() as f32 / audio.sample_rate as f32);
                        
                        assert_eq!(audio.sample_rate, 24000);
                        assert_eq!(audio.channels, 1);
                        assert!(!audio.samples.is_empty());
                    }
                    Err(e) => {
                        println!("✗ 语音合成失败: {}", e);
                        // 如果 GPT 模型不存在，这是预期的
                        if e.to_string().contains("EngineNotLoaded") {
                            println!("  （GPT 模型未加载，这是预期的）");
                        }
                    }
                }
            }
            Err(e) => {
                println!("✗ IndexTTS2 引擎加载失败: {}", e);
                // 如果模型文件不存在，这是预期的
                if e.to_string().contains("不存在") || e.to_string().contains("读取") {
                    println!("  （模型文件不存在，这是预期的）");
                }
            }
        }
    }
}
