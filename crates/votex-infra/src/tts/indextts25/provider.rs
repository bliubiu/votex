//! IndexTTS-2.5 Provider —— `TtsProvider` trait 接入
//!
//! 音色模型：零样本音色克隆，`voice.id` 对应 `models/tts/indextts25/prompts/<id>.wav`
//! 参考音频（≤15s，超出自动截断）。方言映射：普通话→`zh`，粤语→`yue`
//! （词表 100 语言注册内；闽南语/吴语 token 越界不可注册，见蓝本 §3）。
//!
//! 归一化：参考实现的 wetext 数字/日期归一化未移植（见 frontend.rs），构造时警告，
//! 金级对拍须走 `--no-normalization` 基线。

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use votex_domain::error::TtsError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;
use votex_domain::tts::dialect::{Dialect, DialectQuality, DialectSupport};
use votex_domain::tts::provider::TtsProvider;
use votex_domain::tts::value_object::{TtsParams, VoiceId};

use super::engines::IndexTts25Engines;
use super::frontend::Frontend;
use super::pipeline::{self, SpeakerContext, SpkProj, SynthParams, SAMPLING_RATE};

/// 参考音频目录名（registry 目录下）
const PROMPTS_DIR: &str = "prompts";
/// 参考音频时长上限（秒），与 pipeline MAX_REF_SECONDS 一致
const MAX_REF_SECONDS: usize = pipeline::MAX_REF_SECONDS;

/// IndexTTS-2.5 引擎 Provider
pub struct IndexTts25Provider {
    engines: IndexTts25Engines,
    frontend: OnceLock<Frontend>,
    spk_proj: OnceLock<SpkProj>,
    /// 音色缓存：voice id → build_speaker 产物（首次合成后缓存）
    speakers: Mutex<HashMap<String, Arc<SpeakerContext>>>,
    /// ORT 线程数
    threads: usize,
}

impl Default for IndexTts25Provider {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexTts25Provider {
    pub fn new() -> Self {
        Self {
            engines: IndexTts25Engines::new(),
            frontend: OnceLock::new(),
            spk_proj: OnceLock::new(),
            speakers: Mutex::new(HashMap::new()),
            threads: 4,
        }
    }

    /// 模型基目录（registry 约定：models/tts/indextts25/）
    fn model_base_dir() -> PathBuf {
        pipeline::default_model_dir()
    }

    /// voice id → 参考音频路径
    ///
    /// 查找顺序：引擎自带 `prompts/<id>.wav` → 统一音色库
    /// `models/voices/refs/<id>.wav`（`voice add` 入库音色，两个引擎共用，
    /// 免去手动复制 wav 到 prompts/ 目录）。
    fn prompt_path(&self, voice_id: &str) -> PathBuf {
        let builtin = Self::model_base_dir()
            .join(PROMPTS_DIR)
            .join(format!("{voice_id}.wav"));
        if builtin.is_file() {
            return builtin;
        }
        if let Ok(lib_ref) = crate::tts::voice_library::reference_path(voice_id) {
            return lib_ref;
        }
        builtin
    }

    /// 加载（或取缓存）指定音色的 speaker 上下文
    fn ensure_speaker(&self, voice_id: &str) -> Result<Arc<SpeakerContext>, TtsError> {
        if let Some(spk) = self.speakers.lock().map_err(|_| TtsError::EngineNotLoaded)?.get(voice_id) {
            return Ok(Arc::clone(spk));
        }

        let prompt = self.prompt_path(voice_id);
        if !prompt.is_file() {
            return Err(TtsError::UnsupportedVoice(format!(
                "音色 '{}' 不存在（可经 `voice add --reference <wav>` 入库，\
                 或放置 15s 内清晰人声 wav 到 prompts/ 目录）",
                voice_id
            )));
        }

        let (audio_22k, audio_16k) = load_ref_audio(&prompt)
            .map_err(|e| TtsError::SynthesisFailed(format!("读取参考音频失败: {e}")))?;

        let spk_proj = self
            .spk_proj
            .get()
            .ok_or(TtsError::EngineNotLoaded)?;
        let spk = pipeline::build_speaker(&self.engines, &audio_22k, &audio_16k, spk_proj)
            .map_err(|e| TtsError::SynthesisFailed(format!("build_speaker 失败: {e}")))?;
        let spk = Arc::new(spk);

        self.speakers
            .lock()
            .map_err(|_| TtsError::EngineNotLoaded)?
            .insert(voice_id.to_string(), Arc::clone(&spk));
        Ok(spk)
    }

    /// 方言 → 前端 lang 标签
    fn dialect_to_lang(dialect: Option<Dialect>) -> &'static str {
        match dialect {
            Some(Dialect::Cantonese) => "yue",
            _ => "zh",
        }
    }
}

impl TtsProvider for IndexTts25Provider {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::IndexTTS25
    }

    fn load(&self, _model: &Model) -> Result<(), TtsError> {
        let base = Self::model_base_dir();
        if !base.is_dir() {
            return Err(TtsError::SynthesisFailed(format!(
                "IndexTTS-2.5 模型目录不存在: {:?}（请先经模型管理下载 indextts-2.5-onnx）",
                base
            )));
        }

        // 1. 8 个 ONNX session（gpt_prefill/gpt_step/semantic_model/cfm/bigvgan/…）
        self.engines
            .load(&base, self.threads)
            .map_err(|e| TtsError::SynthesisFailed(format!("加载引擎失败: {e}")))?;

        // 2. tiktoken 前端（归一化未移植 → false，构造时打警告）
        let vocab_path = base.join(super::frontend::VOCAB_FILE_NAME);
        let frontend = Frontend::new(&vocab_path, false)
            .map_err(|e| TtsError::SynthesisFailed(format!("加载 tiktoken 词表失败: {e}")))?;
        let _ = self.frontend.set(frontend);

        // 3. spk_proj.npz（说话人投影矩阵）
        let spk_proj = SpkProj::load(&base.join("spk_proj.npz"))
            .map_err(|e| TtsError::SynthesisFailed(format!("加载 spk_proj 失败: {e}")))?;
        let _ = self.spk_proj.set(spk_proj);

        tracing::info!("IndexTTS-2.5 引擎加载完成（模型目录 {:?}）", base);
        Ok(())
    }

    fn unload(&self) -> Result<(), TtsError> {
        self.engines.unload_all();
        if let Ok(mut cache) = self.speakers.lock() {
            cache.clear();
        }
        tracing::info!("IndexTTS-2.5 引擎已释放");
        Ok(())
    }

    fn synthesize(
        &self,
        text: &str,
        voice: &VoiceId,
        params: &TtsParams,
    ) -> Result<AudioData, TtsError> {
        // 情感参数能力边界：IndexTTS-2.5 无情感 token（不支持语种 token 会越界）
        static EMOTION_WARN: crate::tts::EmotionWarnOnce = crate::tts::EmotionWarnOnce::new();
        EMOTION_WARN.warn_if_unsupported("IndexTTS-2.5", params);

        if !self.engines.is_loaded() {
            return Err(TtsError::EngineNotLoaded);
        }
        if text.trim().is_empty() {
            return Err(TtsError::EmptyText);
        }
        let frontend = self.frontend.get().ok_or(TtsError::EngineNotLoaded)?;
        let spk = self.ensure_speaker(&voice.id)?;

        // 方言：params.dialect 优先（Mandarin→zh / Cantonese→yue）
        let lang = Self::dialect_to_lang(params.dialect);

        let synth = SynthParams::default();
        let wav_i16 = pipeline::synthesize(&self.engines, frontend, &spk, text, lang, &synth)
            .map_err(|e| TtsError::SynthesisFailed(format!("IndexTTS-2.5 合成失败: {e}")))?;

        let samples: Vec<f32> = wav_i16.into_iter().map(|v| v as f32 / 32768.0).collect();
        Ok(AudioData {
            samples,
            sample_rate: SAMPLING_RATE as u32,
            channels: 1,
        })
    }

    fn list_voices(&self) -> Vec<VoiceId> {
        let prompts = Self::model_base_dir().join(PROMPTS_DIR);
        let mut voices = Vec::new();
        let mut seen: Vec<String> = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&prompts) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("wav") {
                    let id = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or_default()
                        .to_string();
                    if !seen.contains(&id) {
                        seen.push(id.clone());
                        voices.push(VoiceId::new(&id, &id, EngineKind::IndexTTS25));
                    }
                }
            }
        }
        // 统一音色库（voice add 入库）对 IndexTTS-2.5 同样可用
        for meta in crate::tts::voice_library::list() {
            if !seen.contains(&meta.name) {
                seen.push(meta.name.clone());
                voices.push(VoiceId::new(&meta.name, &meta.name, EngineKind::IndexTTS25));
            }
        }
        voices.sort_by(|a, b| a.id.cmp(&b.id));
        voices
    }

    fn sample_rate(&self) -> u32 {
        SAMPLING_RATE as u32
    }

    fn supported_dialects(&self) -> Vec<DialectSupport> {
        // 词表注册 100 语言：zh/yue 可用；闽南语/吴语等 6 项 token 越界（>60510）不可注册
        vec![
            DialectSupport::new(Dialect::Mandarin, DialectQuality::Native),
            DialectSupport::new(Dialect::Cantonese, DialectQuality::Native),
        ]
    }

    fn is_loaded(&self) -> bool {
        self.engines.is_loaded()
    }
}

/// 读取参考音频：返回 (22k, 16k) 两路 f32
///
/// 与参考实现 `load_ref_audio` 一致：源读入后统一按需重采样。
/// 22k 非标准采样率先 sinc-hann 重采样，再派生 16k。
fn load_ref_audio(path: &std::path::Path) -> anyhow::Result<(Vec<f32>, Vec<f32>)> {
    let audio = crate::audio::wav::read_wav(path)?;
    anyhow::ensure!(
        !audio.samples.is_empty(),
        "参考音频为空: {:?}",
        path
    );

    // 抽取单声道（多声道取均值）
    let mono: Vec<f32> = if audio.channels <= 1 {
        audio.samples
    } else {
        audio
            .samples
            .chunks(audio.channels as usize)
            .map(|c| c.iter().sum::<f32>() / c.len() as f32)
            .collect()
    };

    let sr = audio.sample_rate as usize;
    // 参考实现 load_ref_audio 语义：非 22k 源先 sinc-hann 到 22k，再派生 16k
    // （默认参数 lowpass=6 / rolloff=0.99，见 dsp.py 签名）
    let (audio_22k, audio_16k) = if sr == SAMPLING_RATE {
        let a16 = super::dsp::resample_sinc_hann(&mono, SAMPLING_RATE, 16000, 6, 0.99)?;
        (mono, a16)
    } else if sr == 16000 {
        let a22 = super::dsp::resample_sinc_hann(&mono, 16000, SAMPLING_RATE, 6, 0.99)?;
        (a22, mono)
    } else {
        let a22 = super::dsp::resample_sinc_hann(&mono, sr, SAMPLING_RATE, 6, 0.99)?;
        let a16 = super::dsp::resample_sinc_hann(&a22, SAMPLING_RATE, 16000, 6, 0.99)?;
        (a22, a16)
    };

    // 截断到 15s（build_speaker 内部亦会截，此处提前省算力）
    let cap = MAX_REF_SECONDS * SAMPLING_RATE;
    let audio_22k = if audio_22k.len() > cap { audio_22k[..cap].to_vec() } else { audio_22k };
    let cap16 = MAX_REF_SECONDS * 16000;
    let audio_16k = if audio_16k.len() > cap16 { audio_16k[..cap16].to_vec() } else { audio_16k };
    Ok((audio_22k, audio_16k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 方言到lang映射() {
        assert_eq!(IndexTts25Provider::dialect_to_lang(None), "zh");
        assert_eq!(IndexTts25Provider::dialect_to_lang(Some(Dialect::Mandarin)), "zh");
        assert_eq!(IndexTts25Provider::dialect_to_lang(Some(Dialect::Cantonese)), "yue");
    }

    #[test]
    fn 引擎类型与采样率() {
        let provider = IndexTts25Provider::new();
        assert_eq!(provider.engine_kind(), EngineKind::IndexTTS25);
        assert_eq!(provider.sample_rate(), 22050);
        assert!(!provider.is_loaded());
        // 未加载时 synthesize 报 EngineNotLoaded 而非 panic
        let voice = VoiceId::new("default", "默认", EngineKind::IndexTTS25);
        let params = default_params();
        assert!(matches!(
            provider.synthesize("你好", &voice, &params),
            Err(TtsError::EngineNotLoaded)
        ));
    }

    #[test]
    fn 音色列表_目录存在时枚举wav() {
        let provider = IndexTts25Provider::new();
        // 不依赖本地 prompts 目录内容，只验证不 panic 且元素引擎标记正确
        for voice in provider.list_voices() {
            assert_eq!(voice.engine, EngineKind::IndexTTS25);
        }
    }

    fn default_params() -> votex_domain::tts::value_object::TtsParams {
        votex_domain::tts::value_object::TtsParams {
            engine: EngineKind::IndexTTS25,
            voice: VoiceId::new("default", "默认", EngineKind::IndexTTS25),
            speed: votex_domain::tts::value_object::Speed::default(),
            pitch: votex_domain::tts::value_object::Pitch::default(),
            volume: votex_domain::tts::value_object::Volume::default(),
            segment_size: votex_domain::tts::value_object::SegmentSize::default(),
            segment_silence_ms: 200,
            crossfade_ms: 0,
            num_to_chinese: true,
            denoise: false,
            denoise_level: votex_domain::tts::value_object::DenoiseLevel::Low,
            emotion: None,
            dialect: None,
        }
    }
}
