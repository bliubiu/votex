use anyhow::Result;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::{EngineKind, ModelId, ModelKind};
use votex_domain::tts::chapter;
use votex_domain::tts::number::normalize_numbers;
use votex_domain::tts::pause::{self, TextPiece};
use votex_domain::tts::provider::TtsProvider;
use votex_domain::tts::role::{self, RoleVoiceMap};
use votex_domain::tts::service::TextSegmenter;
use votex_domain::tts::value_object::{
    AudioFormat, DenoiseLevel, Pitch, SegmentSize, Speed, TtsParams, VoiceId, Volume,
};
use votex_domain::shared::value_object::AudioData;
#[cfg(feature = "ffmpeg")]
use votex_infra::audio::mp3::{self as mp3_encoder};
use votex_infra::audio::wav::WavWriter;
use votex_infra::shared::InferenceGate;
use votex_infra::tts::kokoro::KokoroProvider;
use votex_infra::tts::indextts2::IndexTTS2Provider;
use votex_infra::tts::qwen3_tts::Qwen3TtsProvider;
use votex_infra::tts::cosyvoice::CosyVoiceProvider;

/// TTS 进度回调：参数 (当前段索引, 总段数, 进度描述)
pub type TtsProgressCallback = Option<Box<dyn Fn(usize, usize, &str) + Send>>;

fn resolve_kokoro_model(voice_id: &str, lang: Option<&str>) -> ModelId {
    if let Some(lang_val) = lang {
        match lang_val {
            "zh" | "cn" => return ModelId::new("kokoro-82m-v1.1-zh"),
            "en" => return ModelId::new("kokoro-82m"),
            _ => {}
        }
    }
    if voice_id.starts_with("zf_") || voice_id.starts_with("zm_") {
        ModelId::new("kokoro-82m-v1.1-zh")
    } else {
        ModelId::new("kokoro-82m")
    }
}

/// 引擎内存预估（MB），供推理闸门做提交前预检
pub fn engine_memory_estimate_mb(engine: EngineKind, model_override: Option<&str>) -> u64 {
    match engine {
        EngineKind::Kokoro => 512,
        EngineKind::IndexTTS2 => 2048,
        EngineKind::Qwen3Tts => {
            if model_override.map(|m| m.contains("1.7b")).unwrap_or(false) {
                4096
            } else {
                2048
            }
        }
        EngineKind::CosyVoice3 => 2048,
        _ => 1024,
    }
}

/// 合成计划中的单个段
#[derive(Debug, Clone)]
pub struct PlanSegment {
    /// 待合成文本（已做数字读法转换、已剔除停顿标签）
    pub text: String,
    /// 所属章节标题（无章节标记的文本为 None）
    pub chapter_title: Option<String>,
    /// 是否为新章节的第一段（用于 m4b 章节时间戳）
    pub is_new_chapter: bool,
    /// 本段合成完之后追加的静音毫秒（内联 [[pause:N]] 标签值；0 = 无）
    ///
    /// 有内联停顿时替代默认段间静音，保证「停顿值即用户听到的停顿」。
    pub pause_after_ms: u32,
    /// 音色覆盖（多角色配音：按说话人路由；None = 使用全局音色）
    pub voice_override: Option<String>,
    /// 是否为引号台词段（供支持情感参数的引擎切换语气）
    pub is_dialogue: bool,
}

/// 合成附加选项（T2 多角色 / T5 后处理）
#[derive(Debug, Clone, Default)]
pub struct SynthesisOptions {
    /// 角色音色映射（Some 且非空时启用多角色配音）
    pub role_map: Option<RoleVoiceMap>,
    /// EBU R128 响度归一化（输出编码阶段执行）
    pub loudnorm: bool,
    /// 输出语速（atempo 变速，0.5~2.0；与引擎级 speed 独立）
    pub output_tempo: Option<f32>,
}

impl SynthesisOptions {
    /// 构建后处理编码选项
    pub fn encode_opts(&self) -> votex_infra::audio::mp3::EncodeOpts {
        votex_infra::audio::mp3::EncodeOpts {
            loudnorm: self.loudnorm,
            tempo: self.output_tempo,
        }
    }
}

/// 构建合成计划：停顿标签提取 → 数字读法 → 章节切分 → 逐章节分段
///
/// 章节边界强制开新段（标题不与正文粘连）；内联停顿挂在上一段末尾。
/// 启用 `role_map`（多角色配音）时：先按引号切分台词/叙述，
/// 台词与叙述不合并进同一段，并按说话人路由音色。
pub fn build_synthesis_plan(
    text: &str,
    segment_size: SegmentSize,
    num_to_chinese: bool,
    role_map: Option<&RoleVoiceMap>,
) -> Vec<PlanSegment> {
    let pieces = pause::split_pause_tags(text);
    if pieces.is_empty() {
        return Vec::new();
    }

    let multi_voice = role_map.map(|m| !m.is_empty()).unwrap_or(false);

    let mut plan: Vec<PlanSegment> = Vec::new();
    // 待应用到下一段之前的停顿（文本片段之间的 Pause）
    let mut pending_pause_ms: u32 = 0;

    for piece in &pieces {
        match piece {
            TextPiece::Pause(ms) => {
                pending_pause_ms = pending_pause_ms.saturating_add(*ms);
            }
            TextPiece::Text(raw) => {
                // 先把先前积累的停顿挂到上一段末尾（停顿位于两段文本之间）
                if pending_pause_ms > 0 {
                    if let Some(last) = plan.last_mut() {
                        last.pause_after_ms = pending_pause_ms;
                    }
                    // 若停顿前没有任何段（文本以停顿开头），前置停顿无听感意义，丢弃
                    pending_pause_ms = 0;
                }
                let normalized = if num_to_chinese {
                    normalize_numbers(raw)
                } else {
                    raw.to_string()
                };
                for ch in chapter::split_into_chapters(&normalized) {
                    if multi_voice {
                        // 多角色路径：台词/叙述分别分段（不跨片段合并）
                        append_chapter_multi_voice(&mut plan, &ch, segment_size, role_map);
                    } else {
                        // 常规路径
                        let segs = TextSegmenter::segment(&ch.body, segment_size);
                        if segs.is_empty() {
                            // docs/20 F74 回归修复：纯标题输入（有标题无正文，如管线把
                            // 章节标题独立成段后逐段合成）——标题自身成段，否则整章
                            // 被静默丢弃、上游报「没有可输出的音频段」。
                            if let Some(title) = &ch.title {
                                plan.push(PlanSegment {
                                    text: title.clone(),
                                    chapter_title: Some(title.clone()),
                                    is_new_chapter: true,
                                    pause_after_ms: 0,
                                    voice_override: None,
                                    is_dialogue: false,
                                });
                            }
                            continue;
                        }
                        for (j, seg) in segs.into_iter().enumerate() {
                            // 章节标题行单独成段（不与正文合并），保持标题独立韵律
                            if let Some(title) = &ch.title {
                                if j == 0 {
                                    plan.push(PlanSegment {
                                        text: title.clone(),
                                        chapter_title: Some(title.clone()),
                                        is_new_chapter: true,
                                        pause_after_ms: 0,
                                        voice_override: None,
                                        is_dialogue: false,
                                    });
                                }
                            }
                            let _ = j;
                            plan.push(PlanSegment {
                                text: seg.text,
                                chapter_title: ch.title.clone(),
                                is_new_chapter: false,
                                pause_after_ms: 0,
                                voice_override: None,
                                is_dialogue: false,
                            });
                        }
                    }
                }
            }
        }
    }

    // 尾部停顿挂到最后一段
    if pending_pause_ms > 0 {
        if let Some(last) = plan.last_mut() {
            last.pause_after_ms = last.pause_after_ms.max(pending_pause_ms);
        }
    }

    plan
}

/// 多角色路径：对单个章节按台词/叙述片段分别分段并分配音色
fn append_chapter_multi_voice(
    plan: &mut Vec<PlanSegment>,
    ch: &chapter::Chapter,
    segment_size: SegmentSize,
    role_map: Option<&RoleVoiceMap>,
) {
    let map = role_map.expect("multi_voice 已启用");
    // 章节标题行单独成段（叙述音色），先于正文片段落位
    if let Some(title) = &ch.title {
        plan.push(PlanSegment {
            text: title.clone(),
            chapter_title: Some(title.clone()),
            is_new_chapter: true,
            pause_after_ms: 0,
            voice_override: map.narrator.clone(),
            is_dialogue: false,
        });
    }
    // docs/20 F74 回归修复：纯标题输入（正文为空）时上面已落标题段，
    // 不会因台词/正文为空而整章丢失
    if ch.body.trim().is_empty() {
        return;
    }
    for piece in role::split_dialogue(&ch.body) {
        let voice_override = role::resolve_voice(&piece, map);
        let segs = TextSegmenter::segment(&piece.text, segment_size);
        for seg in segs {
            plan.push(PlanSegment {
                text: seg.text,
                chapter_title: ch.title.clone(),
                is_new_chapter: false,
                pause_after_ms: 0,
                voice_override: voice_override.clone(),
                is_dialogue: piece.is_dialogue,
            });
        }
    }
}

/// 章节时间戳（毫秒），用于 m4b 章节元数据
#[derive(Debug, Clone)]
pub struct ChapterTiming {
    pub title: String,
    pub start_ms: u64,
}

/// 写出章节表 JSON（供 GUI 播放器章节跳转/高亮）
///
/// 文件名：`{音频文件}.chapters.json`；无章节时不写、已有旧文件则删除。
///
/// docs/20 F64：原为模块私有，只有直连合成路径会调用；听书管线
/// （`pipeline --kind audiobook`）走「逐段 synthesize + write_concat」，
/// 从不经过这里，导致管线产物**没有章节表**，GUI 播放器的章节跳转/高亮
/// 在管线产物上直接失效（E2E-PL-05 明确要求产出 `*.chapters.json`）。
/// 现改为 `pub(crate)` 供 `pipeline_use_case` 复用。
pub(crate) fn write_chapters_json(audio_path: &Path, chapters: &[ChapterTiming], total_ms: u64) {
    let json_path = audio_path.with_extension("chapters.json");
    if chapters.is_empty() {
        let _ = std::fs::remove_file(&json_path);
        return;
    }
    let entries: Vec<serde_json::Value> = chapters
        .iter()
        .enumerate()
        .map(|(i, c)| {
            let end_ms = chapters.get(i + 1).map(|x| x.start_ms).unwrap_or(total_ms);
            serde_json::json!({
                "title": c.title,
                "start_ms": c.start_ms,
                "end_ms": end_ms,
            })
        })
        .collect();
    let _ = std::fs::write(
        &json_path,
        serde_json::to_string_pretty(&entries).unwrap_or_default(),
    );
    tracing::info!("章节表已写出: {:?}", json_path);
}

pub struct TtsUseCase {
    kokoro: KokoroProvider,
    indextts2: IndexTTS2Provider,
    qwen3tts: Qwen3TtsProvider,
    cosyvoice: CosyVoiceProvider,
}

impl TtsUseCase {
    pub fn new() -> Self {
        Self {
            kokoro: KokoroProvider::new(),
            indextts2: IndexTTS2Provider::new(),
            qwen3tts: Qwen3TtsProvider::new(),
            cosyvoice: CosyVoiceProvider::new(),
        }
    }

    /// 执行 TTS 合成（兼容旧签名，不可取消、无断点续转）
    ///
    /// `model_override` — 可选，强制使用指定模型 ID（如 "qwen3-tts-0.6b"）。
    pub fn synthesize(
        &mut self,
        text: &str,
        output_path: &Path,
        engine: EngineKind,
        voice_id: &str,
        speed: f32,
        format: AudioFormat,
        lang: Option<&str>,
        on_progress: TtsProgressCallback,
        model_override: Option<&str>,
    ) -> Result<()> {
        self.synthesize_ext(
            text, output_path, engine, voice_id, speed, format, lang,
            on_progress, model_override, None, None,
            &SynthesisOptions::default(),
        )
    }

    /// 扩展版合成：支持取消令牌与断点续转会话目录
    ///
    /// - `cancel`：每段合成前检查，置 true 即中断（返回错误「任务已取消」）
    /// - `session_dir`：断点续转目录。每段音频落盘为 `seg_NNNNN.wav`，
    ///   恢复时已存在的段直接复用（配合进度文件校验文本一致性）。
    ///   传 None 则不落盘 session（纯流式拼接，内存占用仍为单段级）。
    /// - `opts`：多角色配音 / 响度归一 / 输出变速等附加选项
    pub fn synthesize_ext(
        &mut self,
        text: &str,
        output_path: &Path,
        engine: EngineKind,
        voice_id: &str,
        speed: f32,
        format: AudioFormat,
        lang: Option<&str>,
        on_progress: TtsProgressCallback,
        model_override: Option<&str>,
        cancel: Option<Arc<AtomicBool>>,
        session_dir: Option<&Path>,
        opts: &SynthesisOptions,
    ) -> Result<()> {
        if text.trim().is_empty() {
            anyhow::bail!("文本内容不能为空");
        }

        // 模型加载进度：Qwen3 加载可能耗时 30s+
        if engine == EngineKind::Qwen3Tts && !self.qwen3tts.is_loaded() {
            if let Some(ref cb) = on_progress {
                cb(0, 0, "正在加载 Qwen3-TTS 模型（首次加载需 10~60 秒）...");
            }
        }

        self.ensure_loaded(engine, voice_id, lang, model_override)?;

        let voice = self.find_voice(engine, voice_id)?;
        let segment_size = match engine {
            EngineKind::Kokoro => {
                let model_id = resolve_kokoro_model(voice_id, lang);
                if model_id.as_str() == "kokoro-82m-v1.1-zh" {
                    SegmentSize::S120  // 中文 G2P 每字 ~3 token, 120字约 420 token < 512上限
                } else {
                    SegmentSize::S500   // IPA 每字符约 1 token, 500字符约 500 token
                }
            }
            _ => SegmentSize::S500,
        };
        let params = TtsParams {
            engine,
            voice: voice.clone(),
            speed: Speed::new(speed).unwrap_or_default(),
            pitch: Pitch::default(),
            volume: Volume::default(),
            segment_size,
            segment_silence_ms: 300,
            crossfade_ms: 50,
            num_to_chinese: true,
            denoise: false,
            denoise_level: DenoiseLevel::Low,
            emotion: None,
            dialect: None,
        };

        // 数字→中文读法（此前该配置无任何实现，「1998 年」被逐字符怪读）
        // 注意：停顿标签先于数字转换提取，避免标签内的数字被误转成中文读法
        let plan = build_synthesis_plan(
            text,
            params.segment_size,
            params.num_to_chinese,
            opts.role_map.as_ref(),
        );
        let segments: Vec<votex_domain::tts::entity::Segment> = plan
            .iter()
            .enumerate()
            .map(|(i, s)| votex_domain::tts::entity::Segment::new(i, s.text.clone()))
            .collect();
        tracing::info!(
            "合成计划完成: {} 段 ({} 章节, {} 个内联停顿)",
            plan.len(),
            plan.iter().filter(|s| s.is_new_chapter).count(),
            plan.iter().filter(|s| s.pause_after_ms > 0).count()
        );

        // 断点续转会话初始化（哈希覆盖完整计划：段文本 + 停顿 + 章节 + 音色）
        let session = session_dir.map(|dir| {
            let canonical: String = plan
                .iter()
                .map(|s| {
                    format!(
                        "{}\u{1F}{}\u{1F}{}\u{1F}{}\u{1E}",
                        s.text,
                        s.pause_after_ms,
                        s.chapter_title.as_deref().unwrap_or(""),
                        s.voice_override.as_deref().unwrap_or("")
                    )
                })
                .collect();
            SynthesisSession::init_from_canonical(dir, &canonical, &segments)
        });
        if let Some(s) = &session {
            tracing::info!(
                "断点续转: 会话目录 {:?}, 已完成 {}/{} 段",
                s.dir, s.completed_count(), segments.len()
            );
        }

        // Qwen3：设置段内进度回调（decode 帧级别），让 GUI 在长段中也能看到进度
        let shared_progress: Arc<std::sync::Mutex<Option<Box<dyn Fn(usize, usize, &str) + Send>>>> =
            Arc::new(std::sync::Mutex::new(on_progress));
        if engine == EngineKind::Qwen3Tts {
            let sp = shared_progress.clone();
            let tts_cb: Arc<dyn Fn(u32, u32, &str) + Send + Sync> = Arc::new(move |step, total, msg| {
                if let Ok(guard) = sp.lock() {
                    if let Some(ref cb) = *guard {
                        cb(step as usize, total as usize, msg);
                    }
                }
            });
            self.qwen3tts.set_progress_callback(tts_cb);
        }

        // 流式输出：边合成边写盘，长文本不再全量驻留内存
        // （写入器在拿到首段真实采样率后初始化，兼容不同引擎的输出规格）
        let mut streamer: Option<Streamer> = None;
        let mut chapters: Vec<ChapterTiming> = Vec::new();
        let estimate_mb = engine_memory_estimate_mb(engine, model_override);
        let gate = InferenceGate::global();

        let total = segments.len();

        let result = (|| -> Result<()> {
            for (i, seg) in segments.iter().enumerate() {
                // 取消检查：每段合成前执行
                if let Some(ref token) = cancel {
                    if token.load(Ordering::SeqCst) {
                        anyhow::bail!("任务已取消");
                    }
                }

                let msg = format!("合成第 {}/{} 段 ({} 字符)", i + 1, total, seg.text.len());
                tracing::info!("{}", msg);
                if let Ok(guard) = shared_progress.lock() {
                    if let Some(ref cb) = *guard {
                        cb(i + 1, total, &msg);
                    }
                }

                // 断点续转：已完成的段直接从 session 读取
                let cached = session
                    .as_ref()
                    .and_then(|s| s.load_segment(i));
                let _was_cached = cached.is_some();

                let audio = match cached {
                    Some(a) => a,
                    None => {
                        // 内存预检 + 并发限流（RAII 许可，段完成即释放）
                        let _permit = gate.acquire(estimate_mb);
                        // 多角色配音：按段解析音色（未命中回退默认音色）
                        let seg_voice = match plan[i].voice_override.as_deref() {
                            Some(vid) if vid != voice.id => self
                                .find_voice(engine, vid)
                                .unwrap_or_else(|e| {
                                    tracing::warn!("音色 {} 未找到（{}），回退默认音色", vid, e);
                                    voice.clone()
                                }),
                            _ => voice.clone(),
                        };
                        let mut seg_params = params.clone();
                        seg_params.voice = seg_voice.clone();
                        let audio =
                            self.synthesize_segment(&seg.text, &seg_voice, &seg_params)?;
                        if let Some(s) = &session {
                            s.save_segment(i, &audio)?;
                        }
                        audio
                    }
                };

                // 首段：用真实采样率/声道数初始化流式写入器
                if streamer.is_none() {
                    streamer = Some(Streamer::begin(
                        output_path,
                        format,
                        audio.sample_rate,
                        audio.channels,
                        opts,
                    )?);
                }
                let stream = streamer.as_mut().expect("streamer 已初始化");

                // 章节起始时间戳（供 m4b 章节元数据）
                if plan[i].is_new_chapter {
                    if let Some(title) = &plan[i].chapter_title {
                        chapters.push(ChapterTiming {
                            title: title.clone(),
                            start_ms: stream.current_ms(),
                        });
                    }
                }

                // 段间静音：内联停顿标签优先，否则用默认段间静音
                // （段 0 无前置静音；内联停顿挂在其宿主段之后，落到下一段之前生效）
                if i > 0 {
                    let pause = plan[i - 1].pause_after_ms;
                    let ms = if pause > 0 { pause } else { params.segment_silence_ms };
                    if ms > 0 {
                        stream.append_silence_ms(ms)?;
                    }
                }
                stream.append_f32(&audio.samples)?;
                drop(audio); // 立即释放本段内存
            }
            // 尾部内联停顿（挂在最后一段之后）
            if let Some(last) = plan.last() {
                if last.pause_after_ms > 0 {
                    if let Some(stream) = streamer.as_mut() {
                        stream.append_silence_ms(last.pause_after_ms)?;
                    }
                }
            }
            Ok(())
        })();

        // 清除段内回调
        if engine == EngineKind::Qwen3Tts {
            self.qwen3tts.clear_progress_callback();
        }

        // 取消/失败时也要正确收尾流式写入器
        match result {
            Ok(()) => {
                let streamer = streamer
                    .ok_or_else(|| anyhow::anyhow!("没有可输出的音频段"))?;
                let total_ms = streamer.finish_all(&chapters, opts)?;
                tracing::info!(
                    "合成完成: {} 段, 总时长 {:.1}s, {} 章节, 输出 {:?}",
                    total,
                    total_ms as f64 / 1000.0,
                    chapters.len(),
                    output_path
                );
                Ok(())
            }
            Err(e) => {
                if let Some(s) = streamer {
                    let _ = s.abort();
                }
                Err(e)
            }
        }
    }

    fn synthesize_segment(
        &self,
        text: &str,
        voice: &VoiceId,
        params: &TtsParams,
    ) -> Result<AudioData> {
        let provider = self.get_provider(params.engine)?;
        let audio = provider.synthesize(text, voice, params)?;
        Ok(audio)
    }

    fn get_provider(&self, engine: EngineKind) -> Result<&dyn TtsProvider> {
        match engine {
            EngineKind::Kokoro => Ok(&self.kokoro),
            EngineKind::IndexTTS2 => Ok(&self.indextts2),
            EngineKind::Qwen3Tts => Ok(&self.qwen3tts),
            EngineKind::CosyVoice3 => Ok(&self.cosyvoice),
            _ => anyhow::bail!("不支持的 TTS 引擎: {:?}", engine),
        }
    }

    #[allow(dead_code)]
    fn get_provider_mut(&mut self, engine: EngineKind) -> Result<&mut dyn TtsProvider> {
        match engine {
            EngineKind::Kokoro => Ok(&mut self.kokoro),
            EngineKind::IndexTTS2 => Ok(&mut self.indextts2),
            EngineKind::Qwen3Tts => Ok(&mut self.qwen3tts),
            EngineKind::CosyVoice3 => Ok(&mut self.cosyvoice),
            _ => anyhow::bail!("不支持的 TTS 引擎: {:?}", engine),
        }
    }

    fn find_voice(&self, engine: EngineKind, voice_id: &str) -> Result<VoiceId> {
        let provider = self.get_provider(engine)?;
        for voice in provider.list_voices() {
            if voice.id == voice_id {
                return Ok(voice);
            }
        }
        let voices = provider.list_voices();
        voices
            .into_iter()
            .next()
            .ok_or_else(|| anyhow::anyhow!("no voice found for engine {:?}", engine))
    }

    #[allow(dead_code)]
    fn list_voices(&self, engine: EngineKind) -> Vec<VoiceId> {
        let provider = self.get_provider(engine).ok();
        match provider {
            Some(p) => p.list_voices(),
            None => Vec::new(),
        }
    }

    fn ensure_loaded(&mut self, engine: EngineKind, voice_id: &str, lang: Option<&str>, model_override: Option<&str>) -> Result<()> {
        match engine {
            EngineKind::Kokoro => {
                if !self.kokoro.is_loaded() {
                    let model_id = resolve_kokoro_model(voice_id, lang);
                    let model_name = if model_id.as_str() == "kokoro-82m-v1.1-zh" {
                        "Kokoro-82M-v1.1-zh"
                    } else {
                        "Kokoro-82M"
                    };
                    self.kokoro.load(&Model::new(
                        model_id, model_name, ModelKind::Tts, EngineKind::Kokoro,
                    ))?;
                }
            }
            EngineKind::IndexTTS2 => {
                if !self.indextts2.is_loaded() {
                    self.indextts2.load(&Model::new(
                        ModelId::new("indextts2"), "IndexTTS2", ModelKind::Tts, EngineKind::IndexTTS2,
                    ))?;
                }
            }
            EngineKind::Qwen3Tts => {
                let model_id_str = model_override.unwrap_or("qwen3-tts-0.6b");
                let model_name = if model_id_str.contains("0.6b") {
                    "Qwen3-TTS-0.6B-CustomVoice"
                } else {
                    "Qwen3-TTS-1.7B-VoiceDesign"
                };
                if !self.qwen3tts.is_loaded() {
                    self.qwen3tts.load(&Model::new(
                        ModelId::new(model_id_str), model_name, ModelKind::Tts, EngineKind::Qwen3Tts,
                    ))?;
                } else if let Some(override_id) = model_override {
                    // 已加载但用户手动切换了变体 → 先卸载再重载
                    let current = ModelId::new(override_id);
                    self.qwen3tts.unload()?;
                    self.qwen3tts.load(&Model::new(
                        current, model_name, ModelKind::Tts, EngineKind::Qwen3Tts,
                    ))?;
                }
            }
            EngineKind::CosyVoice3 => {
                if !self.cosyvoice.is_loaded() {
                    self.cosyvoice.load(&Model::new(
                        ModelId::new("cosyvoice3"), "CosyVoice 3.0", ModelKind::Tts, EngineKind::CosyVoice3,
                    ))?;
                }
            }
            _ => anyhow::bail!("不支持的 TTS 引擎: {:?}", engine),
        }
        Ok(())
    }
}

/// 断点续转会话
///
/// 目录结构：
/// ```text
/// {session_dir}/
/// ├── progress.json   # 文本哈希 + 段文本列表（校验一致性）
/// └── seg_00000.wav   # 每段已合成音频
/// ```
struct SynthesisSession {
    dir: std::path::PathBuf,
    text_hash: String,
}

impl SynthesisSession {
    /// 初始化（或恢复）会话（基于计划规范串哈希）
    ///
    /// 若目录中已有 progress.json 且哈希一致 → 复用已有段文件；
    /// 否则清空目录重新开始。
    fn init_from_canonical(dir: &Path, canonical: &str, segments: &[votex_domain::tts::entity::Segment]) -> Self {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        let text_hash = format!("{:x}", hasher.finalize());

        let _ = std::fs::create_dir_all(dir);
        let progress_path = dir.join("progress.json");

        let valid = std::fs::read_to_string(&progress_path)
            .ok()
            .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
            .map(|v| v.get("text_hash").and_then(|h| h.as_str()) == Some(text_hash.as_str()))
            .unwrap_or(false);

        if !valid {
            // 会话失效（文本变化/首次）：清空旧段
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    if entry.path().extension().map(|e| e == "wav").unwrap_or(false) {
                        let _ = std::fs::remove_file(entry.path());
                    }
                }
            }
        }

        // 写入/刷新进度文件（段列表仅供调试查看）
        let segment_texts: Vec<&str> = segments.iter().map(|s| s.text.as_str()).collect();
        let progress = serde_json::json!({
            "text_hash": text_hash,
            "total_segments": segments.len(),
            "segments_preview": segment_texts.iter().take(5).collect::<Vec<_>>(),
        });
        let _ = std::fs::write(&progress_path, progress.to_string());

        Self { dir: dir.to_path_buf(), text_hash }
    }

    fn seg_path(&self, index: usize) -> std::path::PathBuf {
        self.dir.join(format!("seg_{:05}.wav", index))
    }

    /// 读取已完成的段（不存在返回 None）
    fn load_segment(&self, index: usize) -> Option<AudioData> {
        let path = self.seg_path(index);
        if path.exists() {
            votex_infra::audio::wav::read_wav(&path).ok()
        } else {
            None
        }
    }

    /// 保存段音频
    fn save_segment(&self, index: usize, audio: &AudioData) -> Result<()> {
        WavWriter::write(audio, &self.seg_path(index))
    }

    /// 已完成的段数（用于日志）
    fn completed_count(&self) -> usize {
        std::fs::read_dir(&self.dir)
            .map(|entries| {
                entries
                    .flatten()
                    .filter(|e| {
                        e.path()
                            .file_name()
                            .and_then(|n| n.to_str())
                            .map(|n| n.starts_with("seg_") && n.ends_with(".wav"))
                            .unwrap_or(false)
                    })
                    .count()
            })
            .unwrap_or(0)
    }

    #[allow(dead_code)]
    fn hash(&self) -> &str {
        &self.text_hash
    }
}

/// 流式输出封装
///
/// - WAV：直接流式写目标文件（.tmp 中转，完成后原子重命名）
/// - MP3/M4A/FLAC：流式写临时 WAV → ffmpeg 编码为真实格式
///   （修复此前 M4A/FLAC 写出实为 PCM WAV 的假格式问题）
enum Streamer {
    /// WAV 直出：先写同目录临时文件，成功后再原子改名到最终产物
    WavDirect {
        stream: votex_infra::audio::wav::StreamingWav,
        temp_path: std::path::PathBuf,
        output_path: std::path::PathBuf,
    },
    ViaTempWav {
        temp: votex_infra::audio::wav::StreamingWav,
        temp_path: std::path::PathBuf,
        output_path: std::path::PathBuf,
        format: AudioFormat,
    },
}

/// 由最终输出路径推导同目录的流式临时文件路径。
///
/// **不能用 `with_extension` / `set_extension` 拼临时名（F3）**：
/// `Path::with_extension` 会替换而非追加扩展名，于是
/// `"out.wav".with_extension("wav.tmp")` → `"out.wav.tmp"`，
/// 收尾时再 `with_extension("wav")` 就变成 `"out.wav.wav"` —— 扩展名被追加两次。
///
/// 改为基于 `file_stem` 拼一个独立文件名，并带进程 id 后缀避免并发任务互相覆盖：
/// `out.wav` → `out.<pid>.tmp.wav`，收尾时直接 rename 到 `out.wav`。
fn stream_temp_path(output_path: &Path) -> std::path::PathBuf {
    let stem = output_path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "votex".to_string());
    output_path.with_file_name(format!("{}.{}.tmp.wav", stem, std::process::id()))
}

impl Streamer {
    fn begin(
        output_path: &Path,
        format: AudioFormat,
        sample_rate: u32,
        channels: u16,
        opts: &SynthesisOptions,
    ) -> Result<Self> {
        let _ = opts; // 后处理在 finish_all 阶段生效
        let temp_path = stream_temp_path(output_path);
        match format {
            AudioFormat::Wav => {
                let stream =
                    votex_infra::audio::wav::StreamingWav::create(&temp_path, sample_rate, channels)?;
                Ok(Streamer::WavDirect {
                    stream,
                    temp_path,
                    output_path: output_path.to_path_buf(),
                })
            }
            AudioFormat::Mp3 | AudioFormat::M4A | AudioFormat::M4B | AudioFormat::Flac => {
                let temp =
                    votex_infra::audio::wav::StreamingWav::create(&temp_path, sample_rate, channels)?;
                Ok(Streamer::ViaTempWav {
                    temp,
                    temp_path,
                    output_path: output_path.to_path_buf(),
                    format,
                })
            }
        }
    }

    fn append_f32(&mut self, samples: &[f32]) -> Result<()> {
        match self {
            Streamer::WavDirect { stream, .. } => stream.append_f32(samples),
            Streamer::ViaTempWav { temp, .. } => temp.append_f32(samples),
        }
    }

    fn append_silence_ms(&mut self, ms: u32) -> Result<()> {
        match self {
            Streamer::WavDirect { stream, .. } => stream.append_silence_ms(ms),
            Streamer::ViaTempWav { temp, .. } => temp.append_silence_ms(ms),
        }
    }

    /// 当前已写入的总时长（毫秒）——章节时间戳基准
    fn current_ms(&self) -> u64 {
        match self {
            Streamer::WavDirect { stream, .. } => stream.duration_ms() as u64,
            Streamer::ViaTempWav { temp, .. } => temp.duration_ms() as u64,
        }
    }

    /// 完成流式写入并产出最终文件，返回总时长（毫秒）
    ///
    /// `chapters`：章节时间戳（按 start 升序），m4b 输出消费，
    /// 同时写出 `{输出名}.chapters.json` 供播放器做章节跳转/高亮。
    /// `opts`：响度归一 / 输出变速等后处理（作用于 ffmpeg 编码阶段）。
    fn finish_all(self, chapters: &[ChapterTiming], opts: &SynthesisOptions) -> Result<u64> {
        match self {
            Streamer::WavDirect {
                stream,
                temp_path,
                output_path,
            } => {
                let ms = stream.duration_ms();
                stream.finish()?;
                // 原子改名到调用方指定的路径（不再二次追加扩展名，F3）
                std::fs::rename(&temp_path, &output_path)?;
                write_chapters_json(&output_path, chapters, ms);
                Ok(ms)
            }
            Streamer::ViaTempWav {
                temp,
                temp_path,
                output_path,
                format,
            } => {
                let ms = temp.duration_ms();
                temp.finish()?;
                let chapter_pairs: Vec<(String, u64)> = chapters
                    .iter()
                    .map(|c| (c.title.clone(), c.start_ms))
                    .collect();
                let encode_opts = opts.encode_opts();
                let result = match format {
                    AudioFormat::Mp3 => {
                        #[cfg(feature = "ffmpeg")]
                        {
                            let encoder = mp3_encoder::FfmpegEncoder::new()?;
                            encoder.wav_to_mp3_opts(&temp_path, &output_path, None, encode_opts)
                        }
                        #[cfg(not(feature = "ffmpeg"))]
                        {
                            anyhow::bail!("MP3 输出需要 ffmpeg 支持（编译时启用 ffmpeg feature）")
                        }
                    }
                    AudioFormat::M4A | AudioFormat::Flac => {
                        #[cfg(feature = "ffmpeg")]
                        {
                            let encoder = mp3_encoder::FfmpegEncoder::new()?;
                            encoder.wav_to_format_opts(&temp_path, &output_path, encode_opts)
                        }
                        #[cfg(not(feature = "ffmpeg"))]
                        {
                            anyhow::bail!("M4A/FLAC 输出需要 ffmpeg 支持（编译时启用 ffmpeg feature）")
                        }
                    }
                    AudioFormat::M4B => {
                        #[cfg(feature = "ffmpeg")]
                        {
                            let encoder = mp3_encoder::FfmpegEncoder::new()?;
                            encoder.wav_to_m4b_opts(
                                &temp_path,
                                &output_path,
                                &chapter_pairs,
                                ms,
                                encode_opts,
                            )
                        }
                        #[cfg(not(feature = "ffmpeg"))]
                        {
                            anyhow::bail!("M4B 输出需要 ffmpeg 支持（编译时启用 ffmpeg feature）")
                        }
                    }
                    AudioFormat::Wav => unreachable!("WAV 不走临时文件路径"),
                };
                let _ = std::fs::remove_file(&temp_path);
                write_chapters_json(&output_path, chapters, ms);
                result.map(|_| ms)
            }
        }
    }

    /// 中止：清理临时文件
    fn abort(self) -> Result<()> {
        match self {
            Streamer::WavDirect { stream, temp_path, .. } => {
                let _ = stream; // drop 即可，不回填头（文件无效但无害）
                let _ = std::fs::remove_file(&temp_path);
                Ok(())
            }
            Streamer::ViaTempWav { temp_path, .. } => {
                let _ = std::fs::remove_file(&temp_path);
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod stream_path_tests {
    use super::*;

    #[test]
    fn 临时路径_不重复追加扩展名() {
        // 回归 F3：`out.wav` 不得推导出 `out.wav.wav` 或 `out.wav.tmp`
        let out = Path::new("/tmp/out.wav");
        let tmp = stream_temp_path(out);
        let name = tmp.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.starts_with("out."), "临时文件名应以 stem 开头: {name}");
        assert!(name.ends_with(".tmp.wav"), "临时文件须为 .wav 以便 ffmpeg 识别: {name}");
        assert!(
            !name.contains(".wav.wav") && !name.contains(".wav.tmp"),
            "扩展名被重复追加: {name}"
        );
        // 与最终产物同目录，保证 rename 是同卷原子操作
        assert_eq!(tmp.parent(), out.parent());
    }

    #[test]
    fn 临时路径_多扩展名只取第一段() {
        // `chapter01.mp3` 的 file_stem 是 `chapter01`，临时名 `chapter01.<pid>.tmp.wav`
        let tmp = stream_temp_path(Path::new("/tmp/chapter01.mp3"));
        let name = tmp.file_name().unwrap().to_string_lossy().to_string();
        assert!(name.starts_with("chapter01."), "实际: {name}");
        assert!(name.ends_with(".tmp.wav"), "实际: {name}");
    }

    #[test]
    fn 临时路径_无扩展名与中文名可用() {
        for input in ["无扩展名", "有声书 第一章", "a.b.c.wav"] {
            let tmp = stream_temp_path(Path::new(input));
            assert!(tmp.is_absolute() == false, "相对路径应保持相对: {tmp:?}");
            assert!(tmp.to_string_lossy().ends_with(".tmp.wav"), "实际: {tmp:?}");
        }
    }
}

#[cfg(test)]
mod plan_tests {
    use super::*;
    use votex_domain::tts::value_object::SegmentSize;

    #[test]
    fn 计划_纯文本无章节无停顿() {
        let plan = build_synthesis_plan("第一句。第二句。", SegmentSize::S500, false, None);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].text, "第一句。第二句。");
        assert_eq!(plan[0].pause_after_ms, 0);
        assert!(plan[0].chapter_title.is_none());
        assert!(!plan[0].is_new_chapter);
    }

    /// docs/20 F74 回归：纯标题输入（有标题行、无正文）不得返回空计划——
    /// 管线把章节标题独立成段后逐段合成，标题段进入 synthesize_ext 时
    /// body 为空，此前整段被静默丢弃 → 上游报「没有可输出的音频段」。
    #[test]
    fn 计划_纯标题输入不丢段() {
        let plan = build_synthesis_plan("第0001章 荒诞的机器", SegmentSize::S500, false, None);
        assert_eq!(plan.len(), 1, "纯标题输入应产出 1 段");
        assert_eq!(plan[0].text, "第0001章 荒诞的机器");
        assert!(plan[0].is_new_chapter);
        assert_eq!(plan[0].chapter_title.as_deref(), Some("第0001章 荒诞的机器"));

        // 多角色路径同样不丢段
        let mut map = RoleVoiceMap::default();
        map.narrator = Some("zf_001".to_string());
        let plan_mv =
            build_synthesis_plan("第0001章 荒诞的机器", SegmentSize::S500, false, Some(&map));
        assert_eq!(plan_mv.len(), 1, "多角色路径纯标题输入也应产出 1 段");
        assert_eq!(plan_mv[0].text, "第0001章 荒诞的机器");
    }

    #[test]
    fn 计划_章节边界独立分段() {
        let text = "正文开头。\n第一章 起点\n正文一。\n第二章 转折\n正文二。";
        let plan = build_synthesis_plan(text, SegmentSize::S500, false, None);
        // 前置块1段 + (标题1+正文1) + (标题1+正文1) = 5 段
        assert_eq!(plan.len(), 5);
        assert_eq!(plan[1].text, "第一章 起点");
        assert!(plan[1].is_new_chapter);
        assert_eq!(plan[1].chapter_title.as_deref(), Some("第一章 起点"));
        // 正文段不标记新章节（章节时间戳由标题段记录）
        assert!(!plan[2].is_new_chapter);
        assert_eq!(plan[3].text, "第二章 转折");
        assert!(plan[3].is_new_chapter);
        // 章节标题不与正文粘连
        assert!(!plan[2].text.contains("第一章"));
    }

    #[test]
    fn 计划_内联停顿挂在宿主段() {
        let plan = build_synthesis_plan("甲句。[[pause:500]]乙句。", SegmentSize::S500, false, None);
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].pause_after_ms, 500);
        assert_eq!(plan[1].pause_after_ms, 0);
    }

    #[test]
    fn 计划_尾部停顿() {
        let plan = build_synthesis_plan("只有一句。[[pause:1s]]", SegmentSize::S500, false, None);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].pause_after_ms, 1000);
    }

    #[test]
    fn 计划_停顿标签内数字不被转换() {
        let plan = build_synthesis_plan("第 3 条。[[pause:300]]完毕。", SegmentSize::S500, true, None);
        // 数字读法只作用于朗读文本；停顿标签已被剔除且值保留为毫秒
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].pause_after_ms, 300);
        assert!(plan.iter().all(|s| !s.text.contains("pause")));
    }

    #[test]
    fn 计划_空文本() {
        let plan = build_synthesis_plan("", SegmentSize::S500, true, None);
        assert!(plan.is_empty());
    }

    #[test]
    fn 计划_多角色音色路由() {
        let map = votex_domain::tts::role::RoleVoiceMap::from_json_str(
            r#"{"narrator":"zf_xiaoxiao","dialogue_default":"zm_yunjian","roles":{"张三":"zm_yunjian"}}"#,
        )
        .unwrap();
        let text = "清晨。张三说：「我不去。」李四走了。";
        let plan = build_synthesis_plan(text, SegmentSize::S500, false, Some(&map));
        // 叙述、台词、叙述分别成段（台词与叙述不合并）
        assert_eq!(plan.len(), 3);
        assert_eq!(plan[0].voice_override.as_deref(), Some("zf_xiaoxiao"));
        assert!(!plan[0].is_dialogue);
        assert!(plan[1].is_dialogue);
        assert!(plan[1].text.contains('「'));
        assert_eq!(plan[1].voice_override.as_deref(), Some("zm_yunjian"));
        assert_eq!(plan[2].voice_override.as_deref(), Some("zf_xiaoxiao"));
    }
}
