use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sherpa_onnx::{
    OfflineFireRedAsrCtcModelConfig, OfflineFireRedAsrModelConfig,
    OfflineRecognizer, OfflineRecognizerConfig,
};
use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, RecognizeOutput};
use votex_domain::error::AsrError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;

/// FireRedASR CTC ASR 引擎实现（通过 sherpa-onnx ONNX 推理）
///
/// 使用 sherpa-onnx 加载 ONNX 格式的 FireRedASR v2 CTC 模型，
/// 支持中文、英文及 20+ 种方言识别（普通话、粤语、四川话、上海话、吴语、闽南话等）。
///
/// ## 模型文件结构
///
/// ```text
/// models/asr/firered-asr-ctc/
/// ├── model.int8.onnx    # CTC ONNX 模型（仅编码器+CTC 头，不含注意力解码器）
/// └── tokens.txt         # 词汇表文件
/// ```
pub struct FireRedAsrCtcProvider {
    recognizer: Mutex<Option<OfflineRecognizer>>,
    sample_rate: u32,
}

impl FireRedAsrCtcProvider {
    pub fn new() -> Self {
        Self {
            recognizer: Mutex::new(None),
            sample_rate: 16000,
        }
    }

    /// 从模型目录加载 ONNX 模型
    fn load_from_dir(&self, model_dir: &Path) -> Result<(), AsrError> {
        let model_path = Self::find_model_file(model_dir)?;
        let tokens_path = model_dir.join("tokens.txt");
        if !tokens_path.exists() {
            return Err(AsrError::ModelNotFound(format!(
                "FireRedASR CTC 词汇表文件不存在: {:?}",
                tokens_path
            )));
        }

        tracing::info!("加载 FireRedASR CTC ONNX 模型: {:?}", model_path);

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.fire_red_asr_ctc = OfflineFireRedAsrCtcModelConfig {
            model: Some(model_path.to_string_lossy().to_string()),
        };
        config.model_config.tokens = Some(tokens_path.to_string_lossy().to_string());
        config.model_config.num_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        let recognizer = OfflineRecognizer::create(&config).ok_or_else(|| {
            AsrError::LoadFailed("创建 FireRedASR CTC 识别器失败".to_string())
        })?;

        *self.recognizer.lock().unwrap_or_else(|e| e.into_inner()) = Some(recognizer);

        tracing::info!("FireRedASR CTC ONNX 引擎加载完成 (目录: {:?})", model_dir);
        Ok(())
    }

    /// 查找 ONNX 模型文件
    fn find_model_file(model_dir: &Path) -> Result<PathBuf, AsrError> {
        let candidates = [
            model_dir.join("model.int8.onnx"),
            model_dir.join("model.onnx"),
        ];
        for p in &candidates {
            if p.exists() {
                return Ok(p.clone());
            }
        }

        // 回退：扫描目录下任意 .onnx 文件
        if model_dir.exists() {
            for entry in
                std::fs::read_dir(model_dir)
                    .map_err(|e| AsrError::LoadFailed(format!("读取模型目录失败: {}", e)))?
            {
                let entry = entry
                    .map_err(|e| AsrError::LoadFailed(format!("读取目录项失败: {}", e)))?;
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with(".onnx") {
                        return Ok(entry.path());
                    }
                }
            }
        }

        Err(AsrError::ModelNotFound(format!(
            "FireRedASR CTC ONNX 模型文件不存在于: {:?}。需要 model.int8.onnx 或 model.onnx",
            model_dir
        )))
    }

    /// 查找模型目录
    fn find_model_dir(model: &Model) -> Result<PathBuf, AsrError> {
        // 优先检查 models/asr/firered-asr-ctc/ 目录
        let models_dir = Path::new("models").join("asr").join("firered-asr-ctc");
        if models_dir.exists() && models_dir.is_dir() {
            return Ok(models_dir);
        }

        // 回退：检查 models/firered-asr-ctc/ 目录
        let models_dir = Path::new("models").join("firered-asr-ctc");
        if models_dir.exists() && models_dir.is_dir() {
            return Ok(models_dir);
        }

        // 检查 models/<model_name>/ 目录
        let models_dir = Path::new("models").join(&model.name);
        if models_dir.exists() && models_dir.is_dir() {
            return Ok(models_dir);
        }

        // 检查 model.name 是否为目录路径
        let model_path = Path::new(&model.name);
        if model_path.exists() && model_path.is_dir() {
            return Ok(model_path.to_path_buf());
        }

        Err(AsrError::ModelNotFound(format!(
            "未找到 FireRedASR CTC 模型目录，请将模型文件放在 models/asr/firered-asr-ctc/ 目录下。\
             需要文件: model.int8.onnx, tokens.txt"
        )))
    }
}

impl AsrProvider for FireRedAsrCtcProvider {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::FireRedAsr
    }

    fn load(&self, model: &Model) -> Result<(), AsrError> {
        let model_dir = Self::find_model_dir(model)?;
        self.load_from_dir(&model_dir)
    }

    fn unload(&self) -> Result<(), AsrError> {
        *self.recognizer.lock().unwrap_or_else(|e| e.into_inner()) = None;
        tracing::info!("FireRedASR CTC 引擎已释放");
        Ok(())
    }

    fn recognize(
        &self,
        audio: &AudioData,
        _params: &AsrParams,
    ) -> Result<RecognizeOutput, AsrError> {
        let guard = self.recognizer.lock().unwrap_or_else(|e| e.into_inner());
        let recognizer = guard.as_ref().ok_or(AsrError::EngineNotLoaded)?;

        // 转换为 16kHz 单声道 f32 PCM
        let pcm_data = audio.to_mono_f32_16k();

        // 创建流并输入音频
        let stream = recognizer.create_stream();
        stream.accept_waveform(self.sample_rate as i32, &pcm_data);

        // 执行推理
        recognizer.decode(&stream);

        // 获取结果
        let result = stream
            .get_result()
            .ok_or_else(|| AsrError::RecognizeFailed("FireRedASR CTC 推理未返回结果".to_string()))?;

        tracing::info!(
            "FireRedASR CTC 识别完成: 文本长度 {}",
            result.text.len()
        );

        // 统一走公共组装：CTC 类模型带 token 时间戳时产出词级，否则伪整段
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
        self.recognizer.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }
}

// ============================================================
// FireRedASR AED Provider
// ============================================================

/// FireRedASR AED ASR 引擎实现（通过 sherpa-onnx ONNX 推理）
///
/// 使用 sherpa-onnx 加载 ONNX 格式的 FireRedASR v2 AED（Attention Encoder-Decoder）模型，
/// 支持中文、英文及 20+ 种方言识别（普通话、粤语、四川话、上海话、吴语、闽南话等）。
/// AED 版本使用 encoder + decoder 双模型架构，相比 CTC 精度更高、解码更灵活。
///
/// ## 模型文件结构
///
/// ```text
/// models/asr/firered-asr-aed/
/// ├── encoder.int8.onnx    # AED 编码器模型
/// ├── decoder.int8.onnx   # AED 解码器模型
/// └── tokens.txt          # 词汇表文件
/// ```
pub struct FireRedAsrAedProvider {
    recognizer: Mutex<Option<OfflineRecognizer>>,
    sample_rate: u32,
}

impl FireRedAsrAedProvider {
    pub fn new() -> Self {
        Self {
            recognizer: Mutex::new(None),
            sample_rate: 16000,
        }
    }

    /// 从模型目录加载 ONNX 模型
    fn load_from_dir(&self, model_dir: &Path) -> Result<(), AsrError> {
        let encoder_path = Self::find_encoder_file(model_dir)?;
        let decoder_path = Self::find_decoder_file(model_dir)?;
        let tokens_path = model_dir.join("tokens.txt");
        if !tokens_path.exists() {
            return Err(AsrError::ModelNotFound(format!(
                "FireRedASR AED 词汇表文件不存在: {:?}",
                tokens_path
            )));
        }

        tracing::info!("加载 FireRedASR AED ONNX 模型: {:?}", model_dir);

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.fire_red_asr = OfflineFireRedAsrModelConfig {
            encoder: Some(encoder_path.to_string_lossy().to_string()),
            decoder: Some(decoder_path.to_string_lossy().to_string()),
        };
        config.model_config.tokens = Some(tokens_path.to_string_lossy().to_string());
        config.model_config.num_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        let recognizer = OfflineRecognizer::create(&config).ok_or_else(|| {
            AsrError::LoadFailed("创建 FireRedASR AED 识别器失败".to_string())
        })?;

        *self.recognizer.lock().unwrap_or_else(|e| e.into_inner()) = Some(recognizer);

        tracing::info!("FireRedASR AED ONNX 引擎加载完成 (目录: {:?})", model_dir);
        Ok(())
    }

    /// 查找编码器 ONNX 文件
    fn find_encoder_file(model_dir: &Path) -> Result<PathBuf, AsrError> {
        let candidates = [
            model_dir.join("encoder.int8.onnx"),
            model_dir.join("encoder.onnx"),
        ];
        for p in &candidates {
            if p.exists() {
                return Ok(p.clone());
            }
        }

        Err(AsrError::ModelNotFound(format!(
            "FireRedASR AED 编码器模型文件不存在于: {:?}。需要 encoder.int8.onnx 或 encoder.onnx",
            model_dir
        )))
    }

    /// 查找解码器 ONNX 文件
    fn find_decoder_file(model_dir: &Path) -> Result<PathBuf, AsrError> {
        let candidates = [
            model_dir.join("decoder.int8.onnx"),
            model_dir.join("decoder.onnx"),
        ];
        for p in &candidates {
            if p.exists() {
                return Ok(p.clone());
            }
        }

        Err(AsrError::ModelNotFound(format!(
            "FireRedASR AED 解码器模型文件不存在于: {:?}。需要 decoder.int8.onnx 或 decoder.onnx",
            model_dir
        )))
    }

    /// 查找模型目录
    fn find_model_dir(model: &Model) -> Result<PathBuf, AsrError> {
        // 优先检查 models/asr/firered-asr-aed/ 目录
        let models_dir = Path::new("models").join("asr").join("firered-asr-aed");
        if models_dir.exists() && models_dir.is_dir() {
            return Ok(models_dir);
        }

        // 回退：检查 models/firered-asr-aed/ 目录
        let models_dir = Path::new("models").join("firered-asr-aed");
        if models_dir.exists() && models_dir.is_dir() {
            return Ok(models_dir);
        }

        // 检查 models/<model_name>/ 目录
        let models_dir = Path::new("models").join(&model.name);
        if models_dir.exists() && models_dir.is_dir() {
            return Ok(models_dir);
        }

        // 检查 model.name 是否为目录路径
        let model_path = Path::new(&model.name);
        if model_path.exists() && model_path.is_dir() {
            return Ok(model_path.to_path_buf());
        }

        Err(AsrError::ModelNotFound(format!(
            "未找到 FireRedASR AED 模型目录，请将模型文件放在 models/asr/firered-asr-aed/ 目录下。\
             需要文件: encoder.int8.onnx, decoder.int8.onnx, tokens.txt"
        )))
    }
}

impl AsrProvider for FireRedAsrAedProvider {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::FireRedAsr
    }

    fn load(&self, model: &Model) -> Result<(), AsrError> {
        let model_dir = Self::find_model_dir(model)?;
        self.load_from_dir(&model_dir)
    }

    fn unload(&self) -> Result<(), AsrError> {
        *self.recognizer.lock().unwrap_or_else(|e| e.into_inner()) = None;
        tracing::info!("FireRedASR AED 引擎已释放");
        Ok(())
    }

    fn recognize(
        &self,
        audio: &AudioData,
        _params: &AsrParams,
    ) -> Result<RecognizeOutput, AsrError> {
        let guard = self.recognizer.lock().unwrap_or_else(|e| e.into_inner());
        let recognizer = guard.as_ref().ok_or(AsrError::EngineNotLoaded)?;

        // 转换为 16kHz 单声道 f32 PCM
        let pcm_data = audio.to_mono_f32_16k();

        // 创建流并输入音频
        let stream = recognizer.create_stream();
        stream.accept_waveform(self.sample_rate as i32, &pcm_data);

        // 执行推理
        recognizer.decode(&stream);

        // 获取结果
        let result = stream
            .get_result()
            .ok_or_else(|| AsrError::RecognizeFailed("FireRedASR AED 推理未返回结果".to_string()))?;

        tracing::info!(
            "FireRedASR AED 识别完成: 文本长度 {}",
            result.text.len()
        );

        // 统一走公共组装：CTC 类模型带 token 时间戳时产出词级，否则伪整段
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
        self.recognizer.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }
}

// ============================================================
// 单元测试
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ─── CTC Provider 测试 ─────────────────────────────

    #[test]
    fn test_ctc_find_model_file_found() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.int8.onnx");
        std::fs::write(&model_path, b"fake onnx").unwrap();
        let result = FireRedAsrCtcProvider::find_model_file(dir.path());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_path);
    }

    #[test]
    fn test_ctc_find_model_file_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("custom.onnx");
        std::fs::write(&model_path, b"fake onnx").unwrap();
        let result = FireRedAsrCtcProvider::find_model_file(dir.path());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_path);
    }

    #[test]
    fn test_ctc_find_model_file_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let result = FireRedAsrCtcProvider::find_model_file(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_ctc_engine_kind() {
        let provider = FireRedAsrCtcProvider::new();
        assert_eq!(provider.engine_kind(), EngineKind::FireRedAsr);
    }

    #[test]
    fn test_ctc_sample_rate() {
        let provider = FireRedAsrCtcProvider::new();
        assert_eq!(provider.sample_rate(), 16000);
    }

    #[test]
    fn test_ctc_is_loaded() {
        let provider = FireRedAsrCtcProvider::new();
        assert!(!provider.is_loaded());
    }

    #[test]
    fn test_ctc_unload_when_not_loaded() {
        let provider = FireRedAsrCtcProvider::new();
        assert!(provider.unload().is_ok());
    }

    // ─── AED Provider 测试 ─────────────────────────────

    #[test]
    fn test_aed_find_encoder_file_found() {
        let dir = tempfile::tempdir().unwrap();
        let enc_path = dir.path().join("encoder.int8.onnx");
        std::fs::write(&enc_path, b"fake onnx").unwrap();
        let result = FireRedAsrAedProvider::find_encoder_file(dir.path());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), enc_path);
    }

    #[test]
    fn test_aed_find_encoder_file_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let enc_path = dir.path().join("encoder.onnx");
        std::fs::write(&enc_path, b"fake onnx").unwrap();
        let result = FireRedAsrAedProvider::find_encoder_file(dir.path());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), enc_path);
    }

    #[test]
    fn test_aed_find_encoder_file_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let result = FireRedAsrAedProvider::find_encoder_file(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_aed_find_decoder_file_found() {
        let dir = tempfile::tempdir().unwrap();
        let dec_path = dir.path().join("decoder.int8.onnx");
        std::fs::write(&dec_path, b"fake onnx").unwrap();
        let result = FireRedAsrAedProvider::find_decoder_file(dir.path());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), dec_path);
    }

    #[test]
    fn test_aed_find_decoder_file_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let result = FireRedAsrAedProvider::find_decoder_file(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_aed_engine_kind() {
        let provider = FireRedAsrAedProvider::new();
        assert_eq!(provider.engine_kind(), EngineKind::FireRedAsr);
    }

    #[test]
    fn test_aed_sample_rate() {
        let provider = FireRedAsrAedProvider::new();
        assert_eq!(provider.sample_rate(), 16000);
    }

    #[test]
    fn test_aed_is_loaded() {
        let provider = FireRedAsrAedProvider::new();
        assert!(!provider.is_loaded());
    }

    #[test]
    fn test_aed_unload_when_not_loaded() {
        let provider = FireRedAsrAedProvider::new();
        assert!(provider.unload().is_ok());
    }
}
