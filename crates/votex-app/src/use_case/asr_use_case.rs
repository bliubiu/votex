use anyhow::Result;
use std::path::Path;
use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{
    AsrParams, AsrResult, DenoiseLevel, Language, SliceLength, SubtitleEntry, SubtitleFormat,
    Timestamp,
};
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{EngineKind, ModelId, ModelKind};
use votex_domain::shared::value_object::AudioData;
use votex_infra::asr::sensevoice::SenseVoiceProvider;
use votex_infra::asr::whisper::WhisperProvider;
use votex_infra::asr::paraformer::ParaformerProvider;
use votex_infra::asr::qwen3_asr::Qwen3AsrProvider;
use votex_infra::asr::firered_asr::FireRedAsrCtcProvider;
use votex_infra::asr::wenet::WeNetProvider;
use votex_infra::subtitle::SubtitleWriter;

/// ASR 用例——语音识别
pub struct AsrUseCase {
    whisper: WhisperProvider,
    sensevoice: SenseVoiceProvider,
    paraformer: ParaformerProvider,
    qwen3_asr: Qwen3AsrProvider,
    firered_asr: FireRedAsrCtcProvider,
    wenet: WeNetProvider,
}

impl AsrUseCase {
    pub fn new() -> Self {
        Self {
            whisper: WhisperProvider::new(),
            sensevoice: SenseVoiceProvider::new(),
            paraformer: ParaformerProvider::new(),
            qwen3_asr: Qwen3AsrProvider::new(),
            firered_asr: FireRedAsrCtcProvider::new(),
            wenet: WeNetProvider::new(),
        }
    }

    /// 根据引擎类型获取对应 Provider 的不可变引用
    fn get_provider(&self, engine: EngineKind) -> Result<&dyn AsrProvider> {
        match engine {
            EngineKind::Whisper => Ok(&self.whisper),
            EngineKind::SenseVoice => Ok(&self.sensevoice),
            EngineKind::Paraformer => Ok(&self.paraformer),
            EngineKind::Qwen3Asr => Ok(&self.qwen3_asr),
            EngineKind::FireRedAsr => Ok(&self.firered_asr),
            EngineKind::WeNet => Ok(&self.wenet),
            _ => anyhow::bail!("不支持的 ASR 引擎: {:?}", engine),
        }
    }

    /// 根据引擎类型获取对应 Provider 的可变引用
    fn get_provider_mut(&mut self, engine: EngineKind) -> Result<&mut dyn AsrProvider> {
        match engine {
            EngineKind::Whisper => Ok(&mut self.whisper),
            EngineKind::SenseVoice => Ok(&mut self.sensevoice),
            EngineKind::Paraformer => Ok(&mut self.paraformer),
            EngineKind::Qwen3Asr => Ok(&mut self.qwen3_asr),
            EngineKind::FireRedAsr => Ok(&mut self.firered_asr),
            EngineKind::WeNet => Ok(&mut self.wenet),
            _ => anyhow::bail!("不支持的 ASR 引擎: {:?}", engine),
        }
    }

    /// 执行 ASR 识别
    pub fn recognize(
        &mut self,
        input_path: &Path,
        output_path: &Path,
        model_id: &str,
        language: &str,
        format: &str,
    ) -> Result<AsrResult> {
        if !input_path.exists() {
            anyhow::bail!("输入文件不存在: {:?}", input_path);
        }

        // 解析模型和参数
        let model = Self::parse_model(model_id)?;
        let engine = model.engine;
        let lang = Self::parse_language(language);
        let subtitle_format = Self::parse_format(format);

        let params = AsrParams {
            model: model.id.clone(),
            language: lang,
            auto_punctuation: true,
            auto_slice: true,
            slice_length: SliceLength::S30,
            denoise: false,
            denoise_level: DenoiseLevel::Low,
            output_format: subtitle_format,
        };

        // 加载模型
        self.ensure_loaded(&model)?;

        // 读取音频
        let audio = Self::load_audio(input_path)?;

        // 音频分段
        let slices = Self::slice_audio(&audio, params.slice_length.seconds());
        tracing::info!("音频分段完成: {} 段", slices.len());

        let provider = self.get_provider(engine)?;
        let mut all_entries = Vec::new();
        let mut offset_ms: u64 = 0;

        for (i, slice) in slices.iter().enumerate() {
            tracing::info!("识别第 {}/{} 段", i + 1, slices.len());
            let output = provider.recognize(slice, &params)?;

            if !output.text.is_empty() {
                let duration_ms = slice.duration_ms() as u64;
                all_entries.push(SubtitleEntry {
                    index: all_entries.len() + 1,
                    start_time: Timestamp::from_millis(offset_ms),
                    end_time: Timestamp::from_millis(offset_ms + duration_ms),
                    text: output.text,
                });
                offset_ms += duration_ms;
            }
        }

        // 写入字幕文件
        SubtitleWriter::write(&all_entries, subtitle_format, output_path)?;
        tracing::info!("字幕文件已保存: {:?}", output_path);

        let full_text: String = all_entries
            .iter()
            .map(|e| e.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");

        Ok(AsrResult {
            text: full_text,
            subtitles: all_entries,
            output_path: output_path.to_path_buf(),
        })
    }

    /// 解析模型 ID 为 (Model, EngineKind)
    fn parse_model(model_id: &str) -> Result<Model> {
        let (id, engine) = match model_id {
            "whisper-base" => (ModelId::new("whisper-base"), EngineKind::Whisper),
            "whisper-small" => (ModelId::new("whisper-small"), EngineKind::Whisper),
            "sensevoice" => (ModelId::new("sensevoice"), EngineKind::SenseVoice),
            "paraformer" => (ModelId::new("paraformer"), EngineKind::Paraformer),
            "qwen3-asr" => (ModelId::new("qwen3-asr"), EngineKind::Qwen3Asr),
            "firered-asr" => (ModelId::new("firered-asr"), EngineKind::FireRedAsr),
            "wenet" => (ModelId::new("wenet"), EngineKind::WeNet),
            _ => anyhow::bail!(
                "不支持的 ASR 模型: {}，可选: whisper-base, whisper-small, sensevoice, paraformer, qwen3-asr, firered-asr, wenet",
                model_id
            ),
        };
        Ok(Model::new(id, model_id, ModelKind::Asr, engine))
    }

    /// 解析语言参数
    fn parse_language(lang: &str) -> Language {
        match lang {
            "zh" => Language::Zh,
            "zhen" => Language::ZhEn,
            "en" => Language::En,
            _ => Language::Zh,
        }
    }

    /// 解析输出格式
    fn parse_format(format: &str) -> SubtitleFormat {
        match format {
            "srt" => SubtitleFormat::Srt,
            "lrc" => SubtitleFormat::Lrc,
            _ => SubtitleFormat::Txt,
        }
    }

    /// 加载音频文件
    fn load_audio(path: &Path) -> Result<AudioData> {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        tracing::info!("加载音频文件: {:?} (格式: {})", path, ext);

        match ext {
            "wav" => {
                let audio = votex_infra::audio::wav::read_wav(path)?;
                tracing::info!(
                    "音频信息: {}Hz, {}声道, {}ms",
                    audio.sample_rate,
                    audio.channels,
                    audio.duration_ms()
                );
                Ok(audio)
            }
            _ => {
                tracing::warn!("不支持的格式 {}，返回静音", ext);
                Ok(AudioData::silence(16000, 5000))
            }
        }
    }

    /// 将音频按指定秒数分片
    fn slice_audio(audio: &AudioData, slice_seconds: u32) -> Vec<AudioData> {
        let samples_per_slice =
            (audio.sample_rate as usize * slice_seconds as usize) * audio.channels as usize;
        if audio.samples.len() <= samples_per_slice {
            return vec![audio.clone()];
        }

        audio
            .samples
            .chunks(samples_per_slice)
            .map(|chunk| AudioData {
                samples: chunk.to_vec(),
                sample_rate: audio.sample_rate,
                channels: audio.channels,
            })
            .collect()
    }

    /// 确保模型已加载
    fn ensure_loaded(&mut self, model: &Model) -> Result<()> {
        let provider = self.get_provider_mut(model.engine)?;
        if !provider.is_loaded() {
            provider.load(model)?;
        }
        Ok(())
    }
}
