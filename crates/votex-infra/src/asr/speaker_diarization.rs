//! 说话人分离（Speaker Diarization）
//!
//! 基于 sherpa-onnx 实现：
//! - Pyannote 分割模型 → 语音活动检测 + 说话人分割
//! - x-vector / ERes2Net 嵌入提取 → 说话人向量
//! - 快速聚类（AHC） → 说话人分组
//!
//! 输出：带说话人标签的时间轴片段，供转写标注/多角色 TTS 使用。
//!
//! # 模型文件结构
//!
//! ```text
//! models/asr/speaker-diarization/
//! ├── segmentation.onnx          # Pyannote 分割模型
//! └── embedding.onnx             # x-vector/ERes2Net 嵌入模型
//! ```
//!
//! 两个 ONNX 文件可从 sherpa-onnx 官方发布包获取
//! （`sherpa-onnx-pyannote-segmentation-3-0` 与 3D-Speaker ERes2Net 嵌入模型），
//! 清单见 `models/registry/speaker-diarization.yaml`。

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{bail, Result};

use sherpa_onnx::{
    OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    SpeakerEmbeddingExtractorConfig, FastClusteringConfig,
};
use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, RecognizeOutput};
use votex_domain::error::AsrError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;

/// 说话人分离选项
#[derive(Debug, Clone, PartialEq)]
pub struct DiarizeOptions {
    /// 期望说话人数（0 = 由聚类阈值自动决定）
    pub num_speakers: u32,
    /// 聚类相似度阈值（0~1，越小分得越粗；仅在 num_speakers=0 时生效）
    pub threshold: f32,
}

impl Default for DiarizeOptions {
    fn default() -> Self {
        Self {
            num_speakers: 0,
            threshold: 0.5,
        }
    }
}

/// 说话人分离结果片段
#[derive(Debug, Clone, PartialEq)]
pub struct DiarizationSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub speaker: usize,
}

/// 说话人分离提供器
pub struct SpeakerDiarizationProvider {
    diarizer: Mutex<Option<OfflineSpeakerDiarization>>,
    sample_rate: u32,
}

impl SpeakerDiarizationProvider {
    pub fn new() -> Self {
        Self {
            diarizer: Mutex::new(None),
            sample_rate: 16000,
        }
    }

    /// 从模型目录加载分离模型
    fn load_from_dir(&self, model_dir: &Path) -> Result<(), AsrError> {
        let segmentation_path = find_file(model_dir, "segmentation").ok_or_else(|| {
            AsrError::ModelNotFound(format!(
                "说话人分割模型不存在: {:?}（需要 segmentation.onnx）",
                model_dir
            ))
        })?;
        let embedding_path = find_file(model_dir, "embedding").ok_or_else(|| {
            AsrError::ModelNotFound(format!(
                "说话人嵌入模型不存在: {:?}（需要 embedding.onnx）",
                model_dir
            ))
        })?;

        tracing::info!(
            "加载说话人分离模型: 分割={:?}, 嵌入={:?}",
            segmentation_path,
            embedding_path
        );

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        let mut config = OfflineSpeakerDiarizationConfig::default();

        // 分割模型配置（num_threads 等参数在 segmentation 配置上）
        config.segmentation = OfflineSpeakerSegmentationModelConfig {
            pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                model: Some(segmentation_path.to_string_lossy().to_string()),
            },
            num_threads,
            debug: false,
            provider: None,
        };

        // 嵌入提取器配置
        config.embedding = SpeakerEmbeddingExtractorConfig {
            model: Some(embedding_path.to_string_lossy().to_string()),
            num_threads,
            debug: false,
            provider: None,
        };

        // 聚类配置（默认参数；具体阈值在 diarize 时按选项覆盖）
        config.clustering = FastClusteringConfig::default();

        // VAD 阈值：最小开启/关闭时长（秒）
        config.min_duration_on = 0.5;
        config.min_duration_off = 0.5;

        let diarizer = OfflineSpeakerDiarization::create(&config)
            .ok_or_else(|| AsrError::LoadFailed("创建说话人分离器失败".to_string()))?;

        *self.diarizer.lock().unwrap_or_else(|e| e.into_inner()) = Some(diarizer);

        tracing::info!("说话人分离模型加载完成 (目录: {:?})", model_dir);
        Ok(())
    }

    /// 执行说话人分离
    ///
    /// `audio`：16kHz 单声道音频（与 sherpa-onnx 要求一致）
    /// 返回：按时间排序的分片段
    pub fn diarize(&self, audio: &AudioData) -> Result<Vec<DiarizationSegment>> {
        self.diarize_with(audio, &DiarizeOptions::default())
    }

    /// 按选项执行说话人分离
    pub fn diarize_with(
        &self,
        audio: &AudioData,
        options: &DiarizeOptions,
    ) -> Result<Vec<DiarizationSegment>> {
        let guard = self
            .diarizer
            .lock()
            .map_err(|e| anyhow::anyhow!("锁竞争失败: {}", e))?;
        let diarizer = guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("说话人分离模型未加载，请先调用 load()"))?;

        // sherpa-onnx 要求 16kHz 单声道
        if audio.sample_rate != 16000 {
            bail!("说话人分离要求 16kHz 音频，当前 {}Hz", audio.sample_rate);
        }
        if audio.channels != 1 {
            bail!("说话人分离要求单声道音频，当前 {} 声道", audio.channels);
        }

        // 聚类参数每次调用前覆盖：num_speakers>0 走固定簇数，否则按阈值自动聚类
        let mut clustering = FastClusteringConfig::default();
        if options.num_speakers > 0 {
            clustering.num_clusters = options.num_speakers as i32;
            clustering.threshold = 0.0;
        } else {
            clustering.num_clusters = -1;
            clustering.threshold = options.threshold.clamp(0.05, 0.95);
        }
        diarizer.set_config(&OfflineSpeakerDiarizationConfig {
            segmentation: Default::default(),
            embedding: Default::default(),
            clustering,
            min_duration_on: 0.5,
            min_duration_off: 0.5,
        });

        let pcm = audio.to_mono_f32_16k();
        let result = diarizer
            .process(&pcm)
            .ok_or_else(|| anyhow::anyhow!("说话人分离处理失败"))?;

        let segments = result.sort_by_start_time();
        let mut out = Vec::with_capacity(segments.len());
        for seg in segments {
            out.push(DiarizationSegment {
                start_ms: (seg.start * 1000.0) as u64,
                end_ms: (seg.end * 1000.0) as u64,
                speaker: seg.speaker.max(0) as usize,
            });
        }

        tracing::info!(
            "说话人分离完成：{} 个片段，{} 个说话人",
            out.len(),
            result.num_speakers()
        );
        Ok(out)
    }

    /// 获取采样率
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// 检查是否已加载
    pub fn is_loaded(&self) -> bool {
        self.diarizer.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    /// 查找模型目录：优先 `models/asr/speaker-diarization/`（registry 路径），
    /// 回退旧路径
    pub fn find_model_dir(_model: &Model) -> Result<PathBuf, AsrError> {
        let candidates = [
            Path::new("models").join("asr").join("speaker-diarization"),
            Path::new("models").join("speaker-diarization"),
            Path::new("models").join("SpeakerDiarization"),
        ];
        for dir in &candidates {
            if dir.exists() && dir.is_dir() {
                return Ok(dir.clone());
            }
        }
        Err(AsrError::ModelNotFound(
            "未找到说话人分离模型目录，请将 segmentation.onnx 与 embedding.onnx \
             放在 models/asr/speaker-diarization/ 目录下，或通过模型管理下载"
                .to_string(),
        ))
    }
}

impl Default for SpeakerDiarizationProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// 在模型目录中查找 `<stem>.onnx`（兼容解包后保留的子目录前缀）
fn find_file(dir: &Path, stem: &str) -> Option<PathBuf> {
    if !dir.exists() {
        return None;
    }
    let mut dirs = vec![dir.to_path_buf()];
    let mut candidates: Vec<PathBuf> = Vec::new();
    let mut visited = 0usize;
    while let Some(d) = dirs.pop() {
        visited += 1;
        if visited > 64 {
            break; // 目录扫描上限，防御异常深的解包目录
        }
        let entries = match std::fs::read_dir(&d) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                dirs.push(p);
            } else if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                let lower = name.to_ascii_lowercase();
                if lower == format!("{}.onnx", stem) {
                    return Some(p);
                }
                if lower.contains(stem) && lower.ends_with(".onnx") {
                    candidates.push(p);
                }
            }
        }
    }
    candidates.into_iter().next()
}

impl AsrProvider for SpeakerDiarizationProvider {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::SpeakerDiarization
    }

    fn load(&self, model: &Model) -> Result<(), AsrError> {
        let model_dir = Self::find_model_dir(model)?;
        self.load_from_dir(&model_dir)
    }

    fn unload(&self) -> Result<(), AsrError> {
        *self
            .diarizer
            .lock()
            .map_err(|e| AsrError::LoadFailed(format!("锁竞争失败: {}", e)))? = None;
        tracing::info!("说话人分离引擎已释放");
        Ok(())
    }

    /// 说话人分离只输出「谁在何时说话」，不产出文本。
    /// 文本识别请选择 ASR 引擎；分离结果通过
    /// [`SpeakerDiarizationProvider::diarize_with`] 获取。
    fn recognize(
        &self,
        _audio: &AudioData,
        _params: &AsrParams,
    ) -> Result<RecognizeOutput, AsrError> {
        Err(AsrError::RecognizeFailed(
            "说话人分离引擎不产生文本，请使用 diarize_with() 获取说话人片段，\
             文本识别请选择 ASR 引擎"
                .into(),
        ))
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn is_loaded(&self) -> bool {
        self.is_loaded()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use votex_domain::model::value_object::{ModelId, ModelKind};

    fn plain_model() -> Model {
        Model::new(
            ModelId::new("speaker-diarization"),
            "speaker-diarization",
            ModelKind::Asr,
            EngineKind::SpeakerDiarization,
        )
    }

    #[test]
    fn 说话人分离_未加载时返回错误() {
        let provider = SpeakerDiarizationProvider::new();
        let audio = AudioData::silence(16000, 1000);
        assert!(provider.diarize(&audio).is_err());
    }

    #[test]
    fn 说话人分离_采样率检查() {
        let provider = SpeakerDiarizationProvider::new();
        let audio = AudioData::silence(24000, 1000); // 非 16kHz
        assert!(provider.diarize(&audio).is_err());
    }

    #[test]
    fn 说话人分离_声道检查() {
        let provider = SpeakerDiarizationProvider::new();
        let audio = AudioData {
            samples: vec![0.0; 16000],
            sample_rate: 16000,
            channels: 2,
        };
        assert!(provider.diarize(&audio).is_err());
    }

    #[test]
    fn 说话人分离_recognize_不支持() {
        let provider = SpeakerDiarizationProvider::new();
        let audio = AudioData::silence(16000, 100);
        let params = AsrParams {
            model: votex_domain::model::value_object::ModelId::new("speaker-diarization"),
            language: votex_domain::asr::value_object::Language::Zh,
            auto_punctuation: false,
            auto_slice: false,
            slice_length: votex_domain::asr::value_object::SliceLength::S30,
            denoise: false,
            denoise_level: votex_domain::asr::value_object::DenoiseLevel::Low,
            output_format: votex_domain::asr::value_object::SubtitleFormat::Srt,
        };
        let out = provider.recognize(&audio, &params);
        assert!(out.is_err(), "分离引擎不应产出文本识别结果");
    }

    #[test]
    fn 说话人分离_模型目录不存在时明确报错() {
        let err = SpeakerDiarizationProvider::find_model_dir(&plain_model());
        // 若本机恰好存在模型目录则跳过该断言（CI 可能预装）
        if let Err(e) = err {
            assert!(e.to_string().contains("说话人分离"));
        }
    }

    #[test]
    fn 说话人分离_文件查找_精确名优先() {
        let dir = std::env::temp_dir().join(format!("votex_diar_test_{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("segmentation.onnx");
        std::fs::write(&p, b"stub").unwrap();
        let found = find_file(&dir, "segmentation");
        std::fs::remove_dir_all(&dir).ok();
        assert!(found.is_some(), "segmentation.onnx 应被找到");
    }

    #[test]
    fn 分离选项_默认自动聚类() {
        let opts = DiarizeOptions::default();
        assert_eq!(opts.num_speakers, 0);
        assert!((opts.threshold - 0.5).abs() < f32::EPSILON);
    }
}
