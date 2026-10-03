use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OfflineWenetCtcModelConfig,
};
use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, RecognizeOutput, WordTimestamp};
use votex_domain::error::AsrError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;

/// WeNet Conformer CTC ASR 引擎实现（通过 sherpa-onnx ONNX 推理）
///
/// 使用 sherpa-onnx 加载 ONNX 格式的 WeNet 模型（Conformer U2++ 架构），
/// 基于 WenetSpeech 数据集训练，支持中文普通话识别。
///
/// ## 模型文件结构
///
/// ```text
/// models/asr/wenet/
/// ├── model.onnx       # WeNet CTC ONNX 模型（Conformer U2++ 架构）
/// └── tokens.txt       # 词汇表文件
/// ```
pub struct WeNetProvider {
    recognizer: Mutex<Option<OfflineRecognizer>>,
    sample_rate: u32,
}

impl WeNetProvider {
    pub fn new() -> Self {
        Self {
            recognizer: Mutex::new(None),
            sample_rate: 16000,
        }
    }

    /// 从模型目录加载 ONNX 模型
    fn load_from_dir(&mut self, model_dir: &Path) -> Result<(), AsrError> {
        let model_path = Self::find_model_file(model_dir)?;
        let tokens_path = model_dir.join("tokens.txt");
        if !tokens_path.exists() {
            return Err(AsrError::ModelNotFound(format!(
                "WeNet 词汇表文件不存在: {:?}",
                tokens_path
            )));
        }

        tracing::info!("加载 WeNet Conformer ONNX 模型: {:?}", model_path);

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.wenet_ctc = OfflineWenetCtcModelConfig {
            model: Some(model_path.to_string_lossy().to_string()),
        };
        config.model_config.tokens = Some(tokens_path.to_string_lossy().to_string());
        config.model_config.num_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        let recognizer = OfflineRecognizer::create(&config).ok_or_else(|| {
            AsrError::LoadFailed("创建 WeNet 识别器失败".to_string())
        })?;

        *self.recognizer.lock().unwrap() = Some(recognizer);

        tracing::info!("WeNet Conformer ONNX 引擎加载完成 (目录: {:?})", model_dir);
        Ok(())
    }

    /// 查找 ONNX 模型文件
    fn find_model_file(model_dir: &Path) -> Result<PathBuf, AsrError> {
        let candidates = [
            model_dir.join("model.onnx"),
            model_dir.join("model.int8.onnx"),
            model_dir.join("encoder.onnx"),
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
            "WeNet ONNX 模型文件不存在于: {:?}。需要 model.onnx 或 model.int8.onnx",
            model_dir
        )))
    }

    /// 查找模型目录
    fn find_model_dir(model: &Model) -> Result<PathBuf, AsrError> {
        // 优先检查 models/asr/wenet/ 目录
        let models_dir = Path::new("models").join("asr").join("wenet");
        if models_dir.exists() && models_dir.is_dir() {
            return Ok(models_dir);
        }

        // 回退：检查 models/wenet/ 目录
        let models_dir = Path::new("models").join("wenet");
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
            "未找到 WeNet 模型目录，请将模型文件放在 models/asr/wenet/ 目录下。\
             需要文件: model.onnx, tokens.txt"
        )))
    }
}

impl AsrProvider for WeNetProvider {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::WeNet
    }

    fn load(&mut self, model: &Model) -> Result<(), AsrError> {
        let model_dir = Self::find_model_dir(model)?;
        self.load_from_dir(&model_dir)
    }

    fn unload(&mut self) -> Result<(), AsrError> {
        *self.recognizer.lock().unwrap() = None;
        tracing::info!("WeNet 引擎已释放");
        Ok(())
    }

    fn recognize(
        &self,
        audio: &AudioData,
        _params: &AsrParams,
    ) -> Result<RecognizeOutput, AsrError> {
        let guard = self.recognizer.lock().unwrap();
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
            .ok_or_else(|| AsrError::RecognizeFailed("WeNet 推理未返回结果".to_string()))?;

        tracing::info!("WeNet 识别完成: 文本长度 {}", result.text.len());

        // 将整段文本作为一个时间戳项
        let word_timestamps = if result.text.is_empty() {
            Vec::new()
        } else {
            vec![WordTimestamp {
                word: result.text.clone(),
                start_ms: 0.0,
                end_ms: 0.0,
            }]
        };

        Ok(RecognizeOutput {
            text: result.text,
            word_timestamps,
        })
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn is_loaded(&self) -> bool {
        self.recognizer.lock().unwrap().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_model_file_found() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.onnx");
        std::fs::write(&model_path, b"fake onnx").unwrap();
        let result = WeNetProvider::find_model_file(dir.path());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_path);
    }

    #[test]
    fn test_find_model_file_fallback_int8() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.int8.onnx");
        std::fs::write(&model_path, b"fake onnx").unwrap();
        let result = WeNetProvider::find_model_file(dir.path());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_path);
    }

    #[test]
    fn test_find_model_file_scan() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("custom.onnx");
        std::fs::write(&model_path, b"fake onnx").unwrap();
        let result = WeNetProvider::find_model_file(dir.path());
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), model_path);
    }

    #[test]
    fn test_find_model_file_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let result = WeNetProvider::find_model_file(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_engine_kind() {
        let provider = WeNetProvider::new();
        assert_eq!(provider.engine_kind(), EngineKind::WeNet);
    }

    #[test]
    fn test_sample_rate() {
        let provider = WeNetProvider::new();
        assert_eq!(provider.sample_rate(), 16000);
    }

    #[test]
    fn test_is_loaded() {
        let provider = WeNetProvider::new();
        assert!(!provider.is_loaded());
    }

    #[test]
    fn test_unload_when_not_loaded() {
        let mut provider = WeNetProvider::new();
        assert!(provider.unload().is_ok());
    }
}
