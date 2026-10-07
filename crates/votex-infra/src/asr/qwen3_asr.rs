//! Qwen3-ASR 引擎实现（sherpa-onnx 绑定）
//!
//! 通过 sherpa-onnx Rust 绑定加载 [Wasser1462/Qwen3-ASR-onnx] 导出的
//! 分图 ONNX（conv_frontend / encoder / decoder + tokenizer 目录），
//! 不再依赖自研 ort 单模型推理（旧实现为无效 stub，已废弃）。
//!
//! # 模型文件结构（sherpa-onnx 官方预转换包）
//!
//! ```text
//! models/asr/qwen3-asr/
//! ├── conv_frontend.onnx   # 卷积前端（fbank 特征）
//! ├── encoder.int8.onnx    # 编码器（int8 量化）
//! ├── decoder.int8.onnx    # LLM 解码器（int8 量化）
//! └── tokenizer/           # HF tokenizer 目录（vocab.json/merges.txt/tokenizer_config.json）
//! ```
//!
//! # 能力
//!
//! Qwen3-ASR-0.6B 支持 29 种语言（中文/英语/粤语/日语/韩语…）+ 22 种中国方言
//! （粤语（广东/香港口音）、吴语、闽南语…），并支持歌词/说唱识别。
//!
//! # 集成状态
//! 【已完成】sherpa-onnx 绑定推理管线（与 firered_asr.rs 同范式）

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sherpa_onnx::{
    OfflineQwen3ASRModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
};
use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, RecognizeOutput};
use votex_domain::error::AsrError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;

/// Qwen3-ASR Provider（sherpa-onnx 分图 ONNX 推理）
pub struct Qwen3AsrProvider {
    recognizer: Mutex<Option<OfflineRecognizer>>,
    sample_rate: u32,
}

impl Qwen3AsrProvider {
    pub fn new() -> Self {
        Self {
            recognizer: Mutex::new(None),
            sample_rate: 16000,
        }
    }

    /// 从模型目录加载（三 ONNX + tokenizer 目录）
    fn load_from_dir(&self, model_dir: &Path) -> Result<(), AsrError> {
        let conv_frontend = Self::find_file(model_dir, &["conv_frontend.onnx"])?;
        let encoder = Self::find_file(
            model_dir,
            &["encoder.int8.onnx", "encoder.onnx"],
        )?;
        let decoder = Self::find_file(
            model_dir,
            &["decoder.int8.onnx", "decoder.onnx"],
        )?;
        let tokenizer_dir = model_dir.join("tokenizer");
        if !tokenizer_dir.is_dir() {
            return Err(AsrError::ModelNotFound(format!(
                "Qwen3-ASR tokenizer 目录不存在: {:?}（sherpa-onnx 分图包必须包含 tokenizer/ 子目录）",
                tokenizer_dir
            )));
        }

        tracing::info!(
            "加载 Qwen3-ASR 分图 ONNX 模型: {:?}（encoder: {:?}）",
            model_dir,
            encoder
        );

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.qwen3_asr = OfflineQwen3ASRModelConfig {
            conv_frontend: Some(conv_frontend.to_string_lossy().to_string()),
            encoder: Some(encoder.to_string_lossy().to_string()),
            decoder: Some(decoder.to_string_lossy().to_string()),
            tokenizer: Some(tokenizer_dir.to_string_lossy().to_string()),
            // 解码参数保持 crate Default（temperature≈0 贪心、top_p=0.8、seed=42）
            ..Default::default()
        };
        // qwen3_asr 后端使用内置 HF tokenizer，tokens 置空串（与官方 rust 示例一致）
        config.model_config.tokens = Some(String::new());
        config.model_config.num_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| AsrError::LoadFailed("创建 Qwen3-ASR 识别器失败".to_string()))?;

        *self.recognizer.lock().unwrap_or_else(|e| e.into_inner()) = Some(recognizer);

        tracing::info!("Qwen3-ASR ONNX 引擎加载完成 (目录: {:?})", model_dir);
        Ok(())
    }

    /// 按候选名查找文件
    fn find_file(model_dir: &Path, candidates: &[&str]) -> Result<PathBuf, AsrError> {
        for name in candidates {
            let p = model_dir.join(name);
            if p.exists() {
                return Ok(p);
            }
        }
        Err(AsrError::ModelNotFound(format!(
            "Qwen3-ASR 模型文件缺失: {:?} 中未找到 {}。\
             请从 sherpa-onnx 官方预转换包落位（models/registry/qwen3-asr.yaml）",
            model_dir,
            candidates.join(" / ")
        )))
    }

    /// 查找模型目录
    fn find_model_dir(model: &Model) -> Result<PathBuf, AsrError> {
        // 优先检查 models/asr/qwen3-asr/ 目录
        let models_dir = Path::new("models").join("asr").join("qwen3-asr");
        if models_dir.is_dir() {
            return Ok(models_dir);
        }

        // 回退：检查 models/<model_name>/ 目录
        let models_dir = Path::new("models").join(&model.name);
        if models_dir.is_dir() {
            return Ok(models_dir);
        }

        // 检查 model.name 是否为目录路径
        let model_path = Path::new(&model.name);
        if model_path.is_dir() {
            return Ok(model_path.to_path_buf());
        }

        Err(AsrError::ModelNotFound(format!(
            "未找到 Qwen3-ASR 模型目录，请将模型文件放在 models/asr/qwen3-asr/ 目录下。\
             需要文件: conv_frontend.onnx, encoder.int8.onnx, decoder.int8.onnx, tokenizer/"
        )))
    }
}

impl AsrProvider for Qwen3AsrProvider {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::Qwen3Asr
    }

    fn load(&self, model: &Model) -> Result<(), AsrError> {
        let model_dir = Self::find_model_dir(model)?;
        self.load_from_dir(&model_dir)
    }

    fn unload(&self) -> Result<(), AsrError> {
        *self.recognizer.lock().unwrap_or_else(|e| e.into_inner()) = None;
        tracing::info!("Qwen3-ASR 引擎已释放");
        Ok(())
    }

    fn recognize(
        &self,
        audio: &AudioData,
        _params: &AsrParams,
    ) -> Result<RecognizeOutput, AsrError> {
        let guard = self.recognizer.lock().unwrap_or_else(|e| e.into_inner());
        let recognizer = guard.as_ref().ok_or(AsrError::EngineNotLoaded)?;

        if audio.samples.is_empty() {
            return Err(AsrError::EmptyAudio);
        }

        // 转换为 16kHz 单声道 f32 PCM
        let pcm_data = audio.to_mono_f32_16k();

        // 创建流并输入音频
        let stream = recognizer.create_stream();
        stream.accept_waveform(self.sample_rate as i32, &pcm_data);

        // 执行推理
        recognizer.decode(&stream);

        // 获取结果
        let result = stream.get_result().ok_or_else(|| {
            AsrError::RecognizeFailed("Qwen3-ASR 推理未返回结果".to_string())
        })?;

        tracing::info!("Qwen3-ASR 识别完成: 文本长度 {}", result.text.len());

        // 统一走公共组装：模型带 token 时间戳时产出词级，否则伪整段
        Ok(super::recognize_output_from(
            result.text,
            &result.tokens,
            result.timestamps.as_deref(),
        ))
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn is_loaded(&self) -> bool {
        self.recognizer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_kind() {
        let provider = Qwen3AsrProvider::new();
        assert_eq!(provider.engine_kind(), EngineKind::Qwen3Asr);
    }

    #[test]
    fn test_sample_rate() {
        let provider = Qwen3AsrProvider::new();
        assert_eq!(provider.sample_rate(), 16000);
    }

    #[test]
    fn test_is_loaded_initially_false() {
        let provider = Qwen3AsrProvider::new();
        assert!(!provider.is_loaded());
    }

    #[test]
    fn test_unload_when_not_loaded() {
        let provider = Qwen3AsrProvider::new();
        assert!(provider.unload().is_ok());
    }

    #[test]
    fn test_find_file_candidates() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("encoder.int8.onnx");
        std::fs::write(&p, b"fake").unwrap();
        assert_eq!(
            Qwen3AsrProvider::find_file(dir.path(), &["encoder.int8.onnx", "encoder.onnx"])
                .unwrap(),
            p
        );
        assert!(Qwen3AsrProvider::find_file(dir.path(), &["decoder.int8.onnx"]).is_err());
    }

    #[test]
    fn test_recognize_requires_loaded_engine() {
        let provider = Qwen3AsrProvider::new();
        let audio = AudioData {
            samples: vec![0.0; 16000],
            sample_rate: 16000,
            channels: 1,
        };
        let params = AsrParams {
            model: votex_domain::model::value_object::ModelId::new("qwen3-asr"),
            language: votex_domain::asr::value_object::Language::Zh,
            auto_punctuation: false,
            auto_slice: false,
            slice_length: votex_domain::asr::value_object::SliceLength::S30,
            denoise: false,
            denoise_level: votex_domain::asr::value_object::DenoiseLevel::Low,
            output_format: votex_domain::asr::value_object::SubtitleFormat::Txt,
        };
        assert!(matches!(
            provider.recognize(&audio, &params),
            Err(AsrError::EngineNotLoaded)
        ));
    }
}
