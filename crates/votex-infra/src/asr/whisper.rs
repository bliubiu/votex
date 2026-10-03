use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OfflineWhisperModelConfig,
};
use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, Language, RecognizeOutput, WordTimestamp};
use votex_domain::error::AsrError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;

/// Whisper ASR 引擎实现（通过 sherpa-onnx ONNX 推理）
///
/// 使用 sherpa-onnx 加载 ONNX 格式的 Whisper 模型，
/// 支持中文、英文、中英混合识别。
///
/// ## 模型文件结构
///
/// ```text
/// models/asr/whisper-base/
/// ├── encoder.onnx    # 编码器模型
/// ├── decoder.onnx   # 解码器模型
/// └── tokens.txt     # 词汇表文件
/// ```
pub struct WhisperProvider {
    recognizer: Mutex<Option<OfflineRecognizer>>,
}

impl WhisperProvider {
    pub fn new() -> Self {
        Self {
            recognizer: Mutex::new(None),
        }
    }

    /// 从模型目录加载 ONNX 模型
    fn load_from_dir(&mut self, model_dir: &Path) -> Result<(), AsrError> {
        // F53：候选名必须覆盖三类导出命名，否则已有资产无法加载——
        //  1. 规范命名：`encoder.onnx` / `decoder.onnx` / `tokens.txt`
        //  2. sherpa-onnx 官方命名：`<size>-encoder.onnx` / `<size>-decoder.onnx` / `<size>-tokens.txt`
        //  3. HuggingFace 命名：`encoder_model.onnx` / `decoder_model.onnx`（需 HF 词表转换，
        //     目录中须另有 sherpa 词表，缺失时报错信息会列出目录实况）
        let encoder_path = Self::find_model_file(model_dir, &["encoder.onnx", "encoder_model.onnx"], "encoder")?;
        let decoder_path = Self::find_model_file(model_dir, &["decoder.onnx", "decoder_model.onnx"], "decoder")?;
        let tokens_path = Self::find_aux_file(model_dir, &["tokens.txt"], "tokens")?;

        tracing::info!("加载 Whisper ONNX 模型: {:?}", model_dir);

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.whisper = OfflineWhisperModelConfig {
            encoder: Some(encoder_path.to_string_lossy().to_string()),
            decoder: Some(decoder_path.to_string_lossy().to_string()),
            language: Some("zh".to_string()),
            task: Some("transcribe".to_string()),
            tail_paddings: 0,
            enable_token_timestamps: false,
            enable_segment_timestamps: false,
        };
        config.model_config.tokens = Some(tokens_path.to_string_lossy().to_string());
        config.model_config.num_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| AsrError::LoadFailed("创建 Whisper 识别器失败".to_string()))?;

        *self.recognizer.lock().unwrap() = Some(recognizer);

        tracing::info!("Whisper ONNX 引擎加载完成 (目录: {:?})", model_dir);
        Ok(())
    }

    /// 按 docs/05-模型规范.md 查找模型目录
    fn find_model_dir(model: &Model) -> Result<PathBuf, AsrError> {
        // 1. 标准路径：models/asr/<model_id>/
        let standard_path = Path::new("models").join("asr").join(model.id.as_str());
        if standard_path.exists() && standard_path.is_dir() {
            return Ok(standard_path);
        }

        // 2. 回退：models/<model.name>/
        let name_path = Path::new("models").join(&model.name);
        if name_path.exists() && name_path.is_dir() {
            return Ok(name_path);
        }

        // 3. model.name 作为绝对路径
        let abs_path = Path::new(&model.name);
        if abs_path.exists() && abs_path.is_dir() {
            return Ok(abs_path.to_path_buf());
        }

        Err(AsrError::ModelNotFound(format!(
            "未找到 Whisper 模型目录，请将模型文件放在 models/asr/{}/ 目录下。\
             需要文件: encoder.onnx, decoder.onnx, tokens.txt",
            model.id
        )))
    }

    /// 在模型目录中按候选名查找文件（F53）
    ///
    /// 探测顺序：
    /// 1. 调用方给出的规范候选名
    /// 2. `<目录名>-<候选名>` —— 覆盖 `whisper-small/small-encoder.onnx` 这类官方导出
    /// 3. 目录扫描 —— 文件名包含 role 且以 `.onnx` 结尾（如 `encoder_model_int8.onnx`）
    ///
    /// `role` 只用于第 3 步的匹配与错误提示，不参与前两步。
    fn find_model_file(
        model_dir: &Path,
        candidates: &[&str],
        role: &str,
    ) -> Result<PathBuf, AsrError> {
        let mut tried: Vec<String> = candidates.iter().map(|s| s.to_string()).collect();

        for name in candidates {
            let path = model_dir.join(name);
            if path.exists() {
                return Ok(path);
            }
        }

        // 目录名前缀变体：whisper-small → small-encoder.onnx
        if let Some(dir_name) = model_dir.file_name().and_then(|s| s.to_str()) {
            for name in candidates {
                let prefixed = format!("{}-{}", dir_name, name);
                let path = model_dir.join(&prefixed);
                tried.push(prefixed);
                if path.exists() {
                    return Ok(path);
                }
            }
        }

        if let Some(hit) = Self::scan_by_role(model_dir, role, &["onnx"]) {
            return Ok(hit);
        }

        Err(AsrError::ModelNotFound(format!(
            "未找到 Whisper {} 模型文件于 {:?}；已尝试: {:?}；目录实况: {}",
            role,
            model_dir,
            tried,
            crate::shared::ModelFileLocator::list_dir(model_dir)
        )))
    }

    /// 在模型目录中按候选名查找辅助文件（F53，探测规则同 `find_model_file`）
    fn find_aux_file(
        model_dir: &Path,
        candidates: &[&str],
        role: &str,
    ) -> Result<PathBuf, AsrError> {
        let mut tried: Vec<String> = candidates.iter().map(|s| s.to_string()).collect();

        for name in candidates {
            let path = model_dir.join(name);
            if path.exists() {
                return Ok(path);
            }
        }

        if let Some(dir_name) = model_dir.file_name().and_then(|s| s.to_str()) {
            for name in candidates {
                let prefixed = format!("{}-{}", dir_name, name);
                let path = model_dir.join(&prefixed);
                tried.push(prefixed);
                if path.exists() {
                    return Ok(path);
                }
            }
        }

        if let Some(hit) = Self::scan_by_role(model_dir, role, &["txt"]) {
            return Ok(hit);
        }

        Err(AsrError::ModelNotFound(format!(
            "未找到 Whisper {} 辅助文件于 {:?}；已尝试: {:?}；目录实况: {}",
            role,
            model_dir,
            tried,
            crate::shared::ModelFileLocator::list_dir(model_dir)
        )))
    }

    /// 目录扫描回退：文件名包含 `role` 且扩展名在 `exts` 内
    ///
    /// 排在最后一步，避免误取到无关文件（如 whisper 的 `speech_tokenizer` 一类辅助产物）。
    fn scan_by_role(model_dir: &Path, role: &str, exts: &[&str]) -> Option<PathBuf> {
        let entries = std::fs::read_dir(model_dir).ok()?;
        let mut hits: Vec<PathBuf> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .filter(|e| {
                let name = e.file_name().to_str().unwrap_or_default().to_lowercase();
                name.contains(role) && exts.iter().any(|x| name.ends_with(x))
            })
            .map(|e| e.path())
            .collect();
        // 名字短的优先（model.onnx 优于 encoder_model_int8.onnx），保证结果稳定
        hits.sort_by_key(|p| {
            p.file_name()
                .map(|n| n.to_string_lossy().len())
                .unwrap_or(usize::MAX)
        });
        hits.into_iter().next()
    }

    /// 语言枚举转 whisper 语言代码
    #[allow(dead_code)]
    fn language_code(lang: &Language) -> String {
        match lang {
            Language::Zh => "zh".to_string(),
            Language::ZhEn => "zh".to_string(),
            Language::En => "en".to_string(),
        }
    }
}

impl AsrProvider for WhisperProvider {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::Whisper
    }

    fn load(&mut self, model: &Model) -> Result<(), AsrError> {
        let model_dir = Self::find_model_dir(model)?;
        self.load_from_dir(&model_dir)
    }

    fn unload(&mut self) -> Result<(), AsrError> {
        *self.recognizer.lock().unwrap() = None;
        tracing::info!("Whisper 引擎已释放");
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
        stream.accept_waveform(16000, &pcm_data);

        // 执行推理
        recognizer.decode(&stream);

        // 获取结果
        let result = stream
            .get_result()
            .ok_or_else(|| AsrError::RecognizeFailed("Whisper 推理未返回结果".to_string()))?;

        tracing::info!("Whisper 识别完成: 文本长度 {}", result.text.len());

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
        16000
    }

    fn is_loaded(&self) -> bool {
        self.recognizer.lock().unwrap().is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    /// 造一个模型目录：目录名决定前缀候选，files 为实际文件名
    fn make_dir(dir_name: &str, files: &[&str]) -> tempfile::TempDir {
        let root = tempdir().unwrap();
        let dir = root.path().join(dir_name);
        fs::create_dir_all(&dir).unwrap();
        for f in files {
            fs::write(dir.join(f), b"x").unwrap();
        }
        root
    }

    #[test]
    fn 候选名_规范命名优先() {
        let t = make_dir("whisper-base", &["encoder.onnx", "decoder.onnx", "tokens.txt"]);
        let dir = t.path().join("whisper-base");
        let e = WhisperProvider::find_model_file(&dir, &["encoder.onnx", "encoder_model.onnx"], "encoder").unwrap();
        assert!(e.ends_with("encoder.onnx"));
        assert!(!e.to_string_lossy().contains("encoder_model"));
    }

    #[test]
    fn 候选名_命中目录前缀变体() {
        // 回归 F53：whisper-small 的 `small-encoder.onnx` 曾因候选名缺前缀而无法加载
        let t = make_dir(
            "whisper-small",
            &["small-encoder.onnx", "small-decoder.onnx", "small-tokens.txt"],
        );
        let dir = t.path().join("whisper-small");
        assert!(WhisperProvider::find_model_file(&dir, &["encoder.onnx", "encoder_model.onnx"], "encoder")
            .unwrap()
            .ends_with("small-encoder.onnx"));
        assert!(WhisperProvider::find_model_file(&dir, &["decoder.onnx", "decoder_model.onnx"], "decoder")
            .unwrap()
            .ends_with("small-decoder.onnx"));
        assert!(WhisperProvider::find_aux_file(&dir, &["tokens.txt"], "tokens")
            .unwrap()
            .ends_with("small-tokens.txt"));
    }

    #[test]
    fn 候选名_回退扫描命中hf命名() {
        let t = make_dir(
            "whisper-hf",
            &["encoder_model.onnx", "decoder_model.onnx", "extra.onnx"],
        );
        let dir = t.path().join("whisper-hf");
        assert!(WhisperProvider::find_model_file(&dir, &["encoder.onnx", "encoder_model.onnx"], "encoder")
            .unwrap()
            .ends_with("encoder_model.onnx"));
    }

    #[test]
    fn 候选名_扫描不误取无关文件() {
        // 无关 .onnx 不能被当作 encoder
        let t = make_dir("whisper-x", &["unrelated.onnx"]);
        let dir = t.path().join("whisper-x");
        assert!(WhisperProvider::find_model_file(&dir, &["encoder.onnx"], "encoder").is_err());
    }

    #[test]
    fn 候选名_缺失时错误信息含目录实况() {
        let t = make_dir("whisper-y", &["notes.txt"]);
        let dir = t.path().join("whisper-y");
        let err = WhisperProvider::find_model_file(&dir, &["encoder.onnx"], "encoder").unwrap_err();
        let msg = format!("{:?}", err);
        assert!(msg.contains("notes.txt"), "错误信息须列出目录实况: {}", msg);
    }
}
