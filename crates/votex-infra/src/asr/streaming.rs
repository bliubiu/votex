//! 流式（实时）语音识别引擎——流式 Zipformer（sherpa-onnx 绑定）
//!
//! 通过 sherpa-onnx `OnlineRecognizer` 加载流式 Zipformer transducer 模型
//! （中英混合），支持：
//! - 增量解码：`accept_waveform` 即时吐出中间文本
//! - 端点检测：静音超阈值后自动定稿分句
//!
//! # 模型文件结构（sherpa-onnx 官方预转换包）
//!
//! ```text
//! models/asr/streaming-zipformer/
//! ├── tokens.txt                      # 词汇表
//! ├── encoder-epoch-99-avg-1.onnx     # 编码器（含 .int8.onnx 量化版）
//! ├── decoder-epoch-99-avg-1.onnx     # 解码器
//! └── joiner-epoch-99-avg-1.onnx      # 连接器
//! ```
//!
//! 清单见 `models/registry/streaming-zipformer-zh-en.yaml`。

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use sherpa_onnx::{
    OnlineRecognizer, OnlineRecognizerConfig, OnlineStream, OnlineTransducerModelConfig,
};
use votex_domain::asr::streaming::{StreamingAsrProvider, StreamingAsrSession, StreamingUpdate};
use votex_domain::error::AsrError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;

/// 会话间共享的识别器
///
/// `OnlineRecognizer` 原生侧仅 `Send`；所有原生调用都被
/// 内层 `Mutex` 串行化，因此跨线程共享是安全的。
struct SharedRecognizer(Arc<Mutex<OnlineRecognizer>>);

// 安全性：OnlineRecognizer 的全部原生方法调用都经由 `Arc<Mutex<..>>`
// 串行化，同一时刻至多一个线程触碰原生对象。
unsafe impl Sync for SharedRecognizer {}
unsafe impl Send for SharedRecognizer {}

/// 流式 Zipformer 引擎 Provider
pub struct StreamingZipformerProvider {
    recognizer: Mutex<Option<SharedRecognizer>>,
    sample_rate: u32,
}

impl StreamingZipformerProvider {
    pub fn new() -> Self {
        Self {
            recognizer: Mutex::new(None),
            sample_rate: 16000,
            // 与 sherpa-onnx 官方默认一致：允许稍长句尾停顿，避免频繁截句
        }
    }

    /// 构建识别器（模型文件解析 + OnlineRecognizer 创建）
    fn build_recognizer(model_dir: &Path) -> Result<OnlineRecognizer, AsrError> {
        let tokens_path = model_dir.join("tokens.txt");
        if !tokens_path.exists() {
            return Err(AsrError::ModelNotFound(format!(
                "流式模型词汇表不存在: {:?}",
                tokens_path
            )));
        }

        // transducer 三件套：encoder/decoder/joiner，int8 量化版优先
        let encoder = find_component(model_dir, "encoder")?;
        let decoder = find_component(model_dir, "decoder")?;
        let joiner = find_component(model_dir, "joiner")?;

        tracing::info!(
            "加载流式 Zipformer 模型: encoder={:?}, decoder={:?}, joiner={:?}",
            encoder,
            decoder,
            joiner
        );

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(2);

        let mut config = OnlineRecognizerConfig::default();
        config.model_config.transducer = OnlineTransducerModelConfig {
            encoder: Some(encoder.to_string_lossy().to_string()),
            decoder: Some(decoder.to_string_lossy().to_string()),
            joiner: Some(joiner.to_string_lossy().to_string()),
        };
        config.model_config.tokens = Some(tokens_path.to_string_lossy().to_string());
        config.model_config.num_threads = num_threads;
        config.model_config.provider = Some("cpu".to_string());

        // 解码方式与端点检测
        config.decoding_method = Some("greedy_search".to_string());
        config.enable_endpoint = true;
        config.rule1_min_trailing_silence = 2.4;
        config.rule2_min_trailing_silence = 1.4;
        config.rule3_min_utterance_length = 30.0;

        let recognizer = OnlineRecognizer::create(&config)
            .ok_or_else(|| AsrError::LoadFailed("创建流式识别器失败".to_string()))?;

        tracing::info!("流式 Zipformer 引擎加载完成 (目录: {:?})", model_dir);
        Ok(recognizer)
    }

    /// 查找模型目录：优先 `models/asr/streaming-zipformer/`（registry 路径）
    pub fn find_model_dir(_model: &Model) -> Result<PathBuf, AsrError> {
        let candidates = [
            Path::new("models").join("asr").join("streaming-zipformer"),
            Path::new("models").join("streaming-zipformer"),
            Path::new("models").join("StreamingZipformer"),
        ];
        for dir in &candidates {
            if dir.exists() && dir.is_dir() {
                return Ok(dir.clone());
            }
        }
        Err(AsrError::ModelNotFound(
            "未找到流式听写模型目录，请将 tokens.txt 与 encoder/decoder/joiner \
             ONNX 文件放在 models/asr/streaming-zipformer/ 目录下，\
             或通过模型管理下载 streaming-zipformer-zh-en"
                .to_string(),
        ))
    }
}

impl Default for StreamingZipformerProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// 在模型目录中定位 transducer 组件 ONNX（int8 版优先，兼容解包子目录）
fn find_component(model_dir: &Path, stem: &str) -> Result<PathBuf, AsrError> {
    let direct_int8 = model_dir.join(format!("{}.int8.onnx", stem));
    let direct = model_dir.join(format!("{}.onnx", stem));
    if direct_int8.exists() {
        return Ok(direct_int8);
    }
    if direct.exists() {
        return Ok(direct);
    }
    // 子目录一层扫描（tar.bz2 解包可能保留目录前缀）
    if let Ok(entries) = std::fs::read_dir(model_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if !p.is_dir() {
                continue;
            }
            let int8 = p.join(format!("{}.int8.onnx", stem));
            if int8.exists() {
                return Ok(int8);
            }
            let plain = p.join(format!("{}.onnx", stem));
            if plain.exists() {
                return Ok(plain);
            }
        }
    }
    Err(AsrError::ModelNotFound(format!(
        "流式模型组件 {} 缺失: {:?} 下未找到 {}.onnx 或 {}.int8.onnx",
        stem, model_dir, stem, stem
    )))
}

impl StreamingAsrProvider for StreamingZipformerProvider {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::StreamingZipformer
    }

    fn load(&self, model: &Model) -> Result<(), AsrError> {
        let model_dir = Self::find_model_dir(model)?;
        let recognizer = Self::build_recognizer(&model_dir)?;
        *self
            .recognizer
            .lock()
            .map_err(|e| AsrError::LoadFailed(format!("锁竞争失败: {}", e)))? =
            Some(SharedRecognizer(Arc::new(Mutex::new(recognizer))));
        Ok(())
    }

    fn unload(&self) -> Result<(), AsrError> {
        *self
            .recognizer
            .lock()
            .map_err(|e| AsrError::LoadFailed(format!("锁竞争失败: {}", e)))? = None;
        tracing::info!("流式 Zipformer 引擎已释放");
        Ok(())
    }

    fn is_loaded(&self) -> bool {
        self.recognizer
            .lock()
            .map(|g| g.is_some())
            .unwrap_or(false)
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn create_session(&self) -> Result<Box<dyn StreamingAsrSession + Send>, AsrError> {
        let shared = {
            let guard = self
                .recognizer
                .lock()
                .map_err(|_| AsrError::EngineNotLoaded)?;
            guard
                .as_ref()
                .ok_or(AsrError::EngineNotLoaded)?
                .0
                .clone()
        };
        let stream = shared.lock().map_err(|e| {
            AsrError::RecognizeFailed(format!("锁竞争失败: {}", e))
        })?.create_stream();
        Ok(Box::new(SharedSession {
            recognizer: shared,
            stream,
            samples_pushed: 0,
            finished: false,
        }))
    }
}

/// 流式识别会话
///
/// 持有共享识别器 + 会话私有流。所有 `OnlineRecognizer` 原生调用
/// 经互斥锁串行化。
struct SharedSession {
    recognizer: Arc<Mutex<OnlineRecognizer>>,
    stream: OnlineStream,
    /// 已送入的采样数（用于换算音频时长）
    samples_pushed: u64,
    finished: bool,
}

impl StreamingAsrSession for SharedSession {
    fn accept_waveform(&mut self, samples: &[f32]) -> Result<StreamingUpdate, AsrError> {
        if self.finished {
            return Err(AsrError::RecognizeFailed("会话已结束".to_string()));
        }
        if samples.is_empty() {
            return Ok(self.current_update("", None));
        }
        self.samples_pushed += samples.len() as u64;

        let recognizer = self.recognizer.lock().map_err(|e| {
            AsrError::RecognizeFailed(format!("锁竞争失败: {}", e))
        })?;

        self.stream.accept_waveform(self.sample_rate_of(), samples);

        // 解码至特征缓冲耗尽
        while recognizer.is_ready(&self.stream) {
            recognizer.decode(&self.stream);
        }

        // 当前已解码文本（未定稿，可能被后续结果覆盖）
        let text = recognizer
            .get_result(&self.stream)
            .map(|r| r.text)
            .unwrap_or_default();

        // 端点检测：静音足够久 → 定稿 + 重置流（空结果仅重置）
        let mut final_text = None;
        if recognizer.is_endpoint(&self.stream) {
            if !text.trim().is_empty() {
                final_text = Some(text.clone());
            }
            recognizer.reset(&self.stream);
        }

        Ok(self.current_update(&text, final_text))
    }

    fn finish(&mut self) -> Result<String, AsrError> {
        if self.finished {
            return Ok(String::new());
        }
        self.finished = true;
        let recognizer = self.recognizer.lock().map_err(|e| {
            AsrError::RecognizeFailed(format!("锁竞争失败: {}", e))
        })?;

        // 结束输入：引擎冲刷尾部特征，把残余音频解完
        self.stream.input_finished();
        while recognizer.is_ready(&self.stream) {
            recognizer.decode(&self.stream);
        }
        let text = recognizer
            .get_result(&self.stream)
            .map(|r| r.text)
            .unwrap_or_default();
        Ok(text)
    }
}

impl SharedSession {
    fn sample_rate_of(&self) -> i32 {
        16000
    }

    /// 组装当前更新：`partial` 为未定稿文本，`final_text` 为端点定稿文本
    fn current_update(&self, partial: &str, final_text: Option<String>) -> StreamingUpdate {
        StreamingUpdate {
            partial: partial.to_string(),
            final_text,
            elapsed_ms: self.samples_pushed * 1000 / 16000,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use votex_domain::model::value_object::{ModelId, ModelKind};

    fn plain_model() -> Model {
        Model::new(
            ModelId::new("streaming-zipformer"),
            "streaming-zipformer",
            ModelKind::Asr,
            EngineKind::StreamingZipformer,
        )
    }

    #[test]
    fn 流式引擎_未加载时会话创建失败() {
        let provider = StreamingZipformerProvider::new();
        assert!(matches!(
            provider.create_session(),
            Err(AsrError::EngineNotLoaded)
        ));
    }

    #[test]
    fn 流式引擎_模型目录缺失时明确报错() {
        let err = StreamingZipformerProvider::find_model_dir(&plain_model());
        // 本机若恰好已下载模型则跳过断言
        if let Err(e) = err {
            assert!(e.to_string().contains("流式"));
        }
    }

    #[test]
    fn 流式引擎_组件查找_缺失时明确报错() {
        let dir = std::env::temp_dir().join(format!("votex_stream_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let err = find_component(&dir, "encoder");
        std::fs::remove_dir_all(&dir).ok();
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("encoder"));
    }

    #[test]
    fn 流式引擎_默认采样率16k() {
        let provider = StreamingZipformerProvider::new();
        assert_eq!(provider.sample_rate(), 16000);
        assert_eq!(provider.engine_kind(), EngineKind::StreamingZipformer);
        assert!(!provider.is_loaded());
    }
}
