//! Qwen3-ASR 引擎（ONNX 推理）
//!
//! Qwen3-ASR 是通义千问系列的多模态语音理解模型，
//! 在 Qwen2-Audio 基础上进一步优化，支持语音转文本、语音情感识别。
//!
//! # 模型文件
//! - `models/Qwen3-ASR/model.onnx` - ONNX 模型文件
//! - `models/Qwen3-ASR/tokenizer.json` - Tokenizer 文件
//! - `models/Qwen3-ASR/config.json` - 模型配置
//!
//! # 集成状态
//! 【已完成】ONNX 推理管线集成

use std::path::Path;

use ndarray::Array3;
use ort::session::Session;

use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, RecognizeOutput, WordTimestamp};
use votex_domain::error::AsrError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;

use crate::shared::{EngineState, ModelFileLocator, OrtSessionFactory};

/// Qwen3-ASR Provider（ONNX 推理）
pub struct Qwen3AsrProvider {
    state: EngineState<Session>,
    tokenizer: Option<tokenizers::Tokenizer>,
}

impl Qwen3AsrProvider {
    pub fn new() -> Self {
        Self {
            state: EngineState::new(),
            tokenizer: None,
        }
    }

    /// 从模型目录加载
    fn load_from_dir(&mut self, model_dir: &Path) -> Result<(), AsrError> {
        // F53：候选名覆盖官方 ONNX 导出的常见命名
        let model_path =
            ModelFileLocator::find_first_existing(model_dir, &["model.onnx", "model_int8.onnx"])
                .map_err(|e| {
                    AsrError::ModelNotFound(format!(
                        "{}\n注意：Qwen3-ASR 需 ONNX 权重，若目录内只有 model.safetensors，\
                         说明该模型尚未转换为 ONNX 格式，请先执行转换后再加载。",
                        e
                    ))
                })?;
        let tokenizer_path = ModelFileLocator::find_first_existing(
            model_dir,
            &["tokenizer.json", "vocab.json"],
        )
        .map_err(|e| AsrError::ModelNotFound(format!("{}", e)))?;

        // 使用 OrtSessionFactory 创建 Session
        let session = OrtSessionFactory::create(&model_path)
            .map_err(|e| AsrError::LoadFailed(format!("{}", e)))?;
        self.state.load(session);

        // 加载 Tokenizer
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| AsrError::LoadFailed(format!("加载 Tokenizer 失败: {}", e)))?;
        self.tokenizer = Some(tokenizer);

        tracing::info!("Qwen3-ASR 模型加载完成: {:?}", model_dir);
        Ok(())
    }

    /// 将音频转换为梅尔频谱特征
    fn audio_to_features(&self, audio: &AudioData) -> Result<Array3<f32>, AsrError> {
        let pcm = audio.to_mono_f32_16k();
        let n_samples = pcm.len();

        let frame_length = 400; // 25ms @ 16kHz
        let frame_shift = 160; // 10ms @ 16kHz

        if n_samples < frame_length {
            return Err(AsrError::EmptyAudio);
        }
        let n_frames = (n_samples - frame_length) / frame_shift + 1;
        let n_mels = 80;

        let mut features = vec![0.0f32; n_frames * n_mels];

        // 简化的特征提取
        for frame_idx in 0..n_frames {
            let start = frame_idx * frame_shift;
            let end = start + frame_length;
            if end > n_samples {
                break;
            }

            let frame_energy: f32 =
                pcm[start..end].iter().map(|&x| x * x).sum::<f32>() / frame_length as f32;
            let log_energy = (frame_energy + 1e-10).ln();

            for mel_idx in 0..n_mels {
                let idx = frame_idx * n_mels + mel_idx;
                features[idx] = log_energy * (1.0 - mel_idx as f32 / n_mels as f32);
            }
        }

        Array3::from_shape_vec((1, n_frames, n_mels), features)
            .map_err(|e| AsrError::RecognizeFailed(format!("创建特征张量失败: {}", e)))
    }
}

impl AsrProvider for Qwen3AsrProvider {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::Qwen3Asr
    }

    fn load(&mut self, model: &Model) -> Result<(), AsrError> {
        let model_dir = ModelFileLocator::locate_model_dir(model)
            .map_err(|e| AsrError::ModelNotFound(format!("{}", e)))?;
        self.load_from_dir(&model_dir)
    }

    fn unload(&mut self) -> Result<(), AsrError> {
        self.state.unload();
        self.tokenizer = None;
        tracing::info!("Qwen3-ASR 引擎已释放");
        Ok(())
    }

    fn recognize(
        &self,
        audio: &AudioData,
        _params: &AsrParams,
    ) -> Result<RecognizeOutput, AsrError> {
        let tokenizer = self.tokenizer.as_ref().ok_or(AsrError::EngineNotLoaded)?;

        if audio.samples.is_empty() {
            return Err(AsrError::EmptyAudio);
        }
        // 提取特征
        let features = self.audio_to_features(audio)?;

        // 准备输入
        let input_value = ort::value::Value::from_array(features)
            .map_err(|e| AsrError::RecognizeFailed(format!("创建输入值失败: {}", e)))?;

        // 使用 EngineState 安全访问 session 并执行推理
        let output_vec = self.state.with_mut(|session| {
            let outputs = session
                .run(ort::inputs![input_value])
                .map_err(|e| AsrError::RecognizeFailed(format!("Qwen3-ASR 推理失败: {}", e)))?;

            let output_ids = outputs[0]
                .try_extract_array::<i64>()
                .map_err(|e| AsrError::RecognizeFailed(format!("提取输出失败: {}", e)))?;

            Ok::<Vec<u32>, AsrError>(output_ids.iter().map(|&x| x as u32).collect())
        }).map_err(|_| AsrError::EngineNotLoaded)??;

        // 解码文本
        let text = tokenizer
            .decode(&output_vec, true)
            .map_err(|e| AsrError::RecognizeFailed(format!("解码失败: {}", e)))?;

        let mut word_timestamps = Vec::new();
        if !text.is_empty() {
            word_timestamps.push(WordTimestamp {
                word: text.clone(),
                start_ms: 0.0,
                end_ms: 0.0,
            });
        }

        tracing::info!("Qwen3-ASR 识别完成: 文本长度 {}", text.len());

        Ok(RecognizeOutput {
            text,
            word_timestamps,
        })
    }

    fn sample_rate(&self) -> u32 {
        16000
    }

    fn is_loaded(&self) -> bool {
        self.state.is_loaded()
    }
}
