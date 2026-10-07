use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig,
};
use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, RecognizeOutput};
use votex_domain::error::AsrError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;

/// SenseVoice ASR 引擎（通过 sherpa-onnx ONNX 推理）
///
/// 使用 sherpa-onnx 加载 ONNX 格式的 SenseVoice 模型，
/// 支持中文、英文、粤语、日语、韩语识别。
///
/// ## 模型文件结构
///
/// ```text
/// models/SenseVoice/
/// ├── model.int8.onnx    # 量化 ONNX 模型（sherpa-onnx 导出）
/// ├── model_quant.onnx   # 备选：FunASR 导出的量化 ONNX 模型
/// └── tokens.txt         # 词汇表文件
/// ```
pub struct SenseVoiceProvider {
    recognizer: Mutex<Option<OfflineRecognizer>>,
    sample_rate: u32,
}

impl SenseVoiceProvider {
    pub fn new() -> Self {
        Self {
            recognizer: Mutex::new(None),
            sample_rate: 16000,
        }
    }

    /// 从模型目录加载 ONNX 模型
    fn load_from_dir(&self, model_dir: &Path) -> Result<(), AsrError> {
        // 查找 ONNX 模型文件（优先 sherpa-onnx 导出的 int8 版本，其次 FunASR 导出的量化版本）
        let candidates = [
            model_dir.join("model.int8.onnx"),    // sherpa-onnx 导出
            model_dir.join("model_quant.onnx"),   // FunASR 导出
            model_dir.join("model.onnx"),          // float32 版本
        ];
        let model_path = candidates.iter().find(|p| p.exists()).cloned();
        let model_path = match model_path {
            Some(p) => p,
            None => {
                // 最终回退：扫描目录下任意 .onnx 文件
                let mut found: Option<PathBuf> = None;
                if model_dir.exists() {
                    for entry in std::fs::read_dir(model_dir)
                        .map_err(|e| AsrError::LoadFailed(format!("读取模型目录失败: {}", e)))?
                    {
                        let entry = entry
                            .map_err(|e| AsrError::LoadFailed(format!("读取目录项失败: {}", e)))?;
                        if let Some(name) = entry.file_name().to_str() {
                            if name.ends_with(".onnx") && !name.contains("speech_tokenizer") {
                                found = Some(entry.path());
                                break;
                            }
                        }
                    }
                }
                found.ok_or_else(|| {
                    AsrError::ModelNotFound(format!(
                        "SenseVoice ONNX 模型文件不存在于: {:?}。\
                         需要 model.int8.onnx 或 model_quant.onnx",
                        model_dir
                    ))
                })?
            }
        };

        let tokens_path = model_dir.join("tokens.txt");
        if !tokens_path.exists() {
            return Err(AsrError::ModelNotFound(format!(
                "SenseVoice 词汇表文件不存在: {:?}",
                tokens_path
            )));
        }

        tracing::info!("加载 SenseVoice ONNX 模型: {:?}", model_path);

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.sense_voice = OfflineSenseVoiceModelConfig {
            model: Some(model_path.to_string_lossy().to_string()),
            language: Some("auto".to_string()),
            use_itn: true,
        };
        config.model_config.tokens = Some(tokens_path.to_string_lossy().to_string());
        config.model_config.num_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| AsrError::LoadFailed("创建 SenseVoice 识别器失败".to_string()))?;

        *self.recognizer.lock().unwrap_or_else(|e| e.into_inner()) = Some(recognizer);

        tracing::info!("SenseVoice ONNX 引擎加载完成 (目录: {:?})", model_dir);
        Ok(())
    }

    /// 查找模型目录
    fn find_model_dir(model: &Model) -> Result<PathBuf, AsrError> {
        // 优先检查 models/asr/sensevoice/ 目录（新路径）
        let models_dir = Path::new("models").join("asr").join("sensevoice");
        if models_dir.exists() && models_dir.is_dir() {
            return Ok(models_dir);
        }

        // 回退：检查 models/SenseVoice/ 目录（旧路径）
        let models_dir = Path::new("models").join("SenseVoice");
        if models_dir.exists() && models_dir.is_dir() {
            return Ok(models_dir);
        }

        // 回退：检查 models/sensevoice/ 目录
        let models_dir = Path::new("models").join("sensevoice");
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
            "未找到 SenseVoice 模型目录，请将模型文件放在 models/asr/sensevoice/ 目录下。\
             需要文件: model.int8.onnx 或 model_quant.onnx, tokens.txt"
        )))
    }
}

impl AsrProvider for SenseVoiceProvider {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::SenseVoice
    }

    fn load(&self, model: &Model) -> Result<(), AsrError> {
        let model_dir = Self::find_model_dir(model)?;
        self.load_from_dir(&model_dir)
    }

    fn unload(&self) -> Result<(), AsrError> {
        *self.recognizer.lock().unwrap_or_else(|e| e.into_inner()) = None;
        tracing::info!("SenseVoice 引擎已释放");
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
            .ok_or_else(|| AsrError::RecognizeFailed("SenseVoice 推理未返回结果".to_string()))?;

        tracing::info!(
            "SenseVoice 识别完成: 文本长度 {}",
            result.text.len()
        );

        // SenseVoice 无 token 时间戳 → 统一走公共组装（回退整段伪时间戳）
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
