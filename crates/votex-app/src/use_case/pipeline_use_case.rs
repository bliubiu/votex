use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use votex_domain::model::value_object::EngineKind;
use votex_domain::pipeline::entity::{Pipeline, PipelineInput};
use votex_domain::pipeline::value_object::{PipelineKind, StageKind, StageStatus, PipelineStatus};
use votex_domain::repository::PipelineRepository;
use votex_domain::shared::value_object::{AudioData, ProgressEvent};
use votex_domain::tts::service::TextSegmenter;
use votex_domain::tts::value_object::{parse_tts_engine, AudioFormat, TTS_ENGINE_HINT};
use votex_infra::audio::wav::WavWriter;

use crate::use_case::asr_use_case::AsrUseCase;
use crate::use_case::tts_use_case::{build_synthesis_plan, write_chapters_json, ChapterTiming, TtsUseCase};

/// 段间静音毫秒（拼接时插在相邻两段之间，章节起点计算依赖此值）
const SEGMENT_PAUSE_MS: u32 = 300;

/// docs/20 F74：由章节边界标记与各段时长构建章节时间表（纯函数，便于测试）
///
/// 章节只认真实章节边界段（`build_synthesis_plan` 的 `is_new_chapter` 标记）；
/// 全文无任何边界时给单一「全文」章节兜底，保证 GUI 播放器章节数据源不为空。
fn build_chapter_timings(
    chapter_starts: &[Option<String>],
    durations_ms: &[u64],
    silence_ms: u64,
) -> Vec<ChapterTiming> {
    let mut chapters: Vec<ChapterTiming> = Vec::new();
    let mut cursor_ms: u64 = 0;
    for (i, &dur) in durations_ms.iter().enumerate() {
        if let Some(Some(title)) = chapter_starts.get(i) {
            let title = title.trim();
            if !title.is_empty() {
                chapters.push(ChapterTiming {
                    title: title.to_string(),
                    start_ms: cursor_ms,
                });
            }
        }
        cursor_ms += dur;
        if i + 1 < durations_ms.len() {
            cursor_ms += silence_ms;
        }
    }
    if chapters.is_empty() {
        chapters.push(ChapterTiming {
            title: "全文".to_string(),
            start_ms: 0,
        });
    }
    chapters
}

/// 流水线上下文，在阶段之间传递数据
#[derive(Clone)]
pub struct PipelineContext {
    /// 原始文本
    pub raw_text: String,
    /// 预处理后的文本
    pub processed_text: String,
    /// 分段后的文本段
    pub segments: Vec<String>,
    /// docs/20 F74：各分段的真实章节边界标题——仅章节首段 `Some(标题)`，
    /// 其余段 `None`（与 `build_synthesis_plan` 的 `is_new_chapter` 对齐）。
    /// 与 `segments` 一一对应；空 Vec = 分段阶段未运行。
    pub segment_chapter_starts: Vec<Option<String>>,
    /// 各段合成的音频
    pub audio_segments: Vec<AudioData>,
    /// 最终输出音频路径
    pub audio_path: Option<std::path::PathBuf>,
    /// 字幕文件路径
    pub subtitle_path: Option<std::path::PathBuf>,
    /// 基础文件名
    pub base_name: String,
    /// 输出目录
    pub output_dir: std::path::PathBuf,
    /// TTS 引擎
    pub engine_kind: EngineKind,
    /// TTS 音色
    pub voice: String,
    /// TTS 语速
    pub speed: f32,
    /// ASR 模型
    pub asr_model: String,
    /// 字幕格式
    pub subtitle_format: String,
}

/// 阶段执行函数
type StageFn = Arc<dyn Fn(&mut PipelineContext, &dyn Fn(ProgressEvent)) -> Result<()> + Send + Sync>;

/// 流水线用例（配置驱动）
pub struct PipelineUseCase {
    repo: Option<Arc<dyn PipelineRepository>>,
    /// 阶段处理器注册表：将 StageKind 映射到对应的执行函数
    stage_registry: HashMap<StageKind, StageFn>,
}

impl PipelineUseCase {
    pub fn new() -> Self {
        let registry = Self::default_stage_registry();
        Self {
            repo: None,
            stage_registry: registry,
        }
    }

    /// 注册自定义阶段处理器
    pub fn register_stage(&mut self, kind: StageKind, handler: StageFn) {
        self.stage_registry.insert(kind, handler);
    }

    /// 构建默认阶段处理器注册表
    fn default_stage_registry() -> HashMap<StageKind, StageFn> {
        let mut map: HashMap<StageKind, StageFn> = HashMap::new();
        map.insert(StageKind::TextPreprocess, Arc::new(Self::run_text_preprocess));
        map.insert(StageKind::TextSegment, Arc::new(Self::run_text_segment));
        map.insert(StageKind::TtsSynthesize, Arc::new(Self::run_tts_synthesize));
        map.insert(StageKind::AudioConcat, Arc::new(Self::run_audio_concat));
        map.insert(StageKind::AudioExtract, Arc::new(Self::run_audio_extract));
        map.insert(StageKind::AudioSlice, Arc::new(Self::run_audio_slice));
        map.insert(StageKind::AsrRecognize, Arc::new(Self::run_asr_recognize));
        map.insert(StageKind::SubtitleAlign, Arc::new(Self::run_subtitle_align));
        map.insert(StageKind::Export, Arc::new(Self::run_export));
        map.insert(StageKind::OcrRecognize, Arc::new(Self::run_ocr_recognize));
        map
    }

    /// 设置持久化仓储
    pub fn with_repo(mut self, repo: Arc<dyn PipelineRepository>) -> Self {
        self.repo = Some(repo);
        self
    }

    /// 保存流水线状态（忽略仓储未设置的场景）
    fn save_pipeline(&self, pipeline: &Pipeline) {
        if let Some(ref repo) = self.repo {
            if let Err(e) = repo.save(pipeline) {
                tracing::warn!("保存流水线状态失败: {}", e);
            }
        }
    }

    /// 执行流水线（配置驱动版本）
    ///
    /// 不依赖硬编码的 match 分支，而是通过 stage_registry 动态查找并执行对应的阶段处理函数。
    pub fn execute(
        &self,
        kind: PipelineKind,
        input_path: &Path,
        output_dir: &Path,
        engine: &str,
        voice: &str,
        speed: f32,
        asr_model: &str,
        subtitle_format: &str,
        progress: Option<Box<dyn Fn(ProgressEvent) + Send>>,
    ) -> Result<()> {
        std::fs::create_dir_all(output_dir)?;

        // docs/20 F62：此前这里只映射 kokoro/indextts2，导致 `--engine cosyvoice3`
        // 与 `--engine qwen3` 直接 `不支持的 TTS 引擎`；现统一走 domain 的
        // `parse_tts_engine`，新增 TTS 引擎只需改那一处。
        //
        // `subtitle` 管线不含 TTS 阶段（阶段为 AudioExtract→AudioSlice→AsrRecognize
        // →SubtitleAlign→Export），`--engine` 对它毫无作用，因此**不参与校验**：
        // 否则 `--kind subtitle --engine <任意无效值>` 会报"不支持的 TTS 引擎"，
        // 而这条管线根本不会合成语音。占位用 kokoro，仅为满足 `PipelineContext` 字段。
        let subtitle_only = kind == PipelineKind::Subtitle;
        let engine_kind = if subtitle_only {
            parse_tts_engine(engine).unwrap_or(EngineKind::Kokoro)
        } else {
            parse_tts_engine(engine).ok_or_else(|| {
                anyhow::anyhow!("不支持的 TTS 引擎: {}，可选: {}", engine, TTS_ENGINE_HINT)
            })?
        };

        // docs/20 F63：此前这里对所有流水线类型都无条件 `read_to_string(input_path)`，
        // 但 `--kind subtitle` 的输入是**音频**：传入 .wav 会直接
        // `stream did not contain valid UTF-8`（exit 1），该管线完全不可用。
        // 现按流水线类型区分输入：文本型读文本，字幕型登记音频路径交由音频提取阶段。
        // docs/20 F73：文本型入口改用 `EncodingDetector::read_text_file`（UTF-8/GBK
        // 自动检测），与 TTS 单命令行为一致；裸 `read_to_string` 遇 GBK 文件直接
        // `stream did not contain valid UTF-8`（exit 1）。
        let text = if subtitle_only {
            String::new()
        } else {
            let t = votex_infra::encoding::detector::EncodingDetector::read_text_file(input_path)?;
            if t.trim().is_empty() {
                anyhow::bail!("输入文本为空");
            }
            t
        };

        if subtitle_only && !input_path.is_file() {
            anyhow::bail!("字幕流水线的输入音频不存在: {:?}", input_path);
        }

        let base_name = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");

        // 构建流水线
        let pipeline_input = PipelineInput {
            text_file: if subtitle_only { None } else { Some(input_path.to_path_buf()) },
            media_file: if subtitle_only { Some(input_path.to_path_buf()) } else { None },
            tts_params: None,
            asr_params: None,
        };
        let mut pipeline = Pipeline::new(kind, pipeline_input);

        let mut ctx = PipelineContext {
            raw_text: text,
            processed_text: String::new(),
            segments: Vec::new(),
            segment_chapter_starts: Vec::new(),
            audio_segments: Vec::new(),
            // docs/20 F63：字幕管线没有 TTS 阶段产出音频，ASR 阶段依赖
            // `audio_path` 定位输入音频，此处直接登记（run_audio_extract 会校验）。
            audio_path: if subtitle_only { Some(input_path.to_path_buf()) } else { None },
            subtitle_path: None,
            base_name: base_name.to_string(),
            output_dir: output_dir.to_path_buf(),
            engine_kind,
            voice: voice.to_string(),
            speed,
            asr_model: asr_model.to_string(),
            subtitle_format: subtitle_format.to_string(),
        };

        tracing::info!("流水线启动: {:?}, 阶段数: {}", kind, pipeline.stages.len());
        pipeline.status = PipelineStatus::Running(0);
        self.save_pipeline(&pipeline);

        // 进度回调包装
        let progress_wrapper = |event: ProgressEvent| {
            if let Some(ref cb) = progress {
                cb(event);
            }
        };

        let stage_count = pipeline.stages.len();
        let mut stage_statuses: Vec<StageStatus> = pipeline.stages.iter().map(|s| s.status.clone()).collect();
        let stage_kinds: Vec<StageKind> = pipeline.stages.iter().map(|s| s.kind).collect();

        // 配置驱动：通过 stage_registry 查找并执行对应阶段的处理函数
        let mut failed_stage: Option<(usize, String)> = None;
        for stage_idx in 0..stage_count {
            stage_statuses[stage_idx] = StageStatus::Running;
            pipeline.status = PipelineStatus::Running(stage_idx);
            self.save_pipeline(&pipeline);

            progress_wrapper(ProgressEvent::PhaseChanged {
                phase: format!("{:?}", stage_kinds[stage_idx]),
            });

            let result = match self.stage_registry.get(&stage_kinds[stage_idx]) {
                Some(handler) => handler(&mut ctx, &progress_wrapper),
                None => {
                    tracing::warn!("未注册的阶段处理器: {:?}，跳过", stage_kinds[stage_idx]);
                    Ok(())
                }
            };

            match result {
                Ok(()) => {
                    stage_statuses[stage_idx] = StageStatus::Completed;
                    tracing::info!("[阶段 {}] {:?} 完成", stage_idx, stage_kinds[stage_idx]);
                }
                Err(e) => {
                    stage_statuses[stage_idx] = StageStatus::Failed(format!("{}", e));
                    failed_stage = Some((stage_idx, format!("{}", e)));
                    tracing::error!("[阶段 {}] {:?} 失败: {}", stage_idx, stage_kinds[stage_idx], e);
                    break;
                }
            }
        }

        // 将状态同步回 pipeline
        for (i, status) in stage_statuses.into_iter().enumerate() {
            if i < pipeline.stages.len() {
                pipeline.stages[i].status = status;
            }
        }

        // 保存最终状态
        if let Some((idx, msg)) = failed_stage {
            pipeline.status = PipelineStatus::Failed(idx, msg.clone());
            self.save_pipeline(&pipeline);
            anyhow::bail!("流水线在第 {} 阶段失败: {}", idx, msg);
        } else {
            pipeline.status = PipelineStatus::Completed;
            self.save_pipeline(&pipeline);
            progress_wrapper(ProgressEvent::PhaseChanged {
                phase: "完成".to_string(),
            });
            tracing::info!("流水线执行完成: {:?}", kind);
        }
        Ok(())
    }

    // ======== 阶段处理器实现 ========

    /// 阶段：文本预处理
    fn run_text_preprocess(ctx: &mut PipelineContext, _progress: &dyn Fn(ProgressEvent)) -> Result<()> {
        tracing::info!("[阶段] 文本预处理");
        ctx.processed_text = TextSegmenter::preprocess(&ctx.raw_text);
        if ctx.processed_text.is_empty() {
            anyhow::bail!("预处理后文本为空");
        }
        tracing::info!("  原文 {} 字符 → 处理后 {} 字符", ctx.raw_text.len(), ctx.processed_text.len());
        Ok(())
    }

    /// 阶段：文本分段
    fn run_text_segment(ctx: &mut PipelineContext, _progress: &dyn Fn(ProgressEvent)) -> Result<()> {
        tracing::info!("[阶段] 文本分段");
        // docs/20 F74：改用 `build_synthesis_plan`（与 tts 直连路径同一条计划链路），
        // 章节标题由 `chapter::split_into_chapters` 真实检测并单独成段，
        // 每段的 `is_new_chapter` 即章节边界——替代此前「每个 TextSegmenter 分段
        // 都当成章节、正文截 20 字当标题」的伪章节行为。
        // 数字读法：管线预处理阶段未做数字转换，此处沿用 tts 路径默认（关闭），
        // 与修复前管线的数字行为一致。
        let plan = build_synthesis_plan(
            &ctx.processed_text,
            votex_domain::tts::value_object::SegmentSize::S500,
            false,
            None,
        );
        ctx.segments = plan.iter().map(|s| s.text.clone()).collect();
        ctx.segment_chapter_starts = plan
            .iter()
            .map(|s| if s.is_new_chapter { s.chapter_title.clone() } else { None })
            .collect();
        tracing::info!("  分段完成: {} 段", ctx.segments.len());
        if ctx.segments.is_empty() {
            anyhow::bail!("文本分段后为空");
        }
        Ok(())
    }

    /// 阶段：TTS 合成
    fn run_tts_synthesize(ctx: &mut PipelineContext, _progress: &dyn Fn(ProgressEvent)) -> Result<()> {
        tracing::info!("[阶段] TTS 合成 (引擎: {:?}, 音色: {})", ctx.engine_kind, ctx.voice);

        if ctx.segments.is_empty() {
            tracing::info!("  没有文本分段，跳过 TTS 合成");
            return Ok(());
        }

        let mut tts = TtsUseCase::new();
        let total = ctx.segments.len();
        let seg_dir = ctx.output_dir.join("segments");
        std::fs::create_dir_all(&seg_dir)?;

        for (i, seg_text) in ctx.segments.iter().enumerate() {
            tracing::info!("  合成第 {}/{} 段 ({} 字)", i + 1, total, seg_text.len());

            let seg_path = seg_dir.join(format!("seg_{:04}.wav", i));
            tts.synthesize(
                seg_text,
                &seg_path,
                ctx.engine_kind,
                &ctx.voice,
                ctx.speed,
                AudioFormat::Wav,
                None,
                None,
                None,
            )?;

            let audio = votex_infra::audio::wav::read_wav(&seg_path)?;
            ctx.audio_segments.push(audio);
        }

        Ok(())
    }

    /// 阶段：音频拼接
    ///
    /// docs/20 F64：此前只拼音频、**不写章节表**。章节 JSON 是 GUI 播放器
    /// 章节跳转/高亮的唯一数据来源（`gui/widgets/audio_player.rs:280`），
    /// E2E-PL-05 也明确要求产出 `*.chapters.json`；直连 `tts` 路径会写，
    /// 管线路径绕过 `write_chapters_json`，于是听书产物在 GUI 里没有章节。
    fn run_audio_concat(ctx: &mut PipelineContext, _progress: &dyn Fn(ProgressEvent)) -> Result<()> {
        tracing::info!("[阶段] 音频拼接");

        if ctx.audio_segments.is_empty() {
            tracing::warn!("  没有音频段可拼接");
            return Ok(());
        }

        let output_path = ctx.output_dir.join(format!("{}.wav", ctx.base_name));

        let silence = AudioData::silence(ctx.audio_segments[0].sample_rate, SEGMENT_PAUSE_MS);
        let mut segments_with_pause = Vec::new();
        for (i, seg) in ctx.audio_segments.iter().enumerate() {
            segments_with_pause.push(seg.clone());
            if i < ctx.audio_segments.len() - 1 {
                segments_with_pause.push(silence.clone());
            }
        }

        WavWriter::write_concat(&segments_with_pause, &output_path)?;
        ctx.audio_path = Some(output_path.clone());

        // 章节起点必须与 write_concat 的拼接顺序严格一致：
        // 段 i 起点 = 前 i 段时长之和 + i × 段间静音
        // docs/20 F74：章节只认真实章节边界段（`segment_chapter_starts` 中
        // `Some(标题)` 的分段，来自 `build_synthesis_plan` 的章节检测）。
        // 此前每个分段都被当成章节、非标题段截前 20 字正文当标题，
        // 导致 chapters.json 出现「女人面目狰狞…」这类伪章节。
        let silence_ms = silence.duration_ms() as u64;
        let durations_ms: Vec<u64> = ctx
            .audio_segments
            .iter()
            .map(|a| a.duration_ms() as u64)
            .collect();
        let chapters = build_chapter_timings(&ctx.segment_chapter_starts, &durations_ms, silence_ms);
        let total_ms: u64 =
            durations_ms.iter().sum::<u64>() + silence_ms * (durations_ms.len().saturating_sub(1)) as u64;
        write_chapters_json(&output_path, &chapters, total_ms);

        tracing::info!(
            "  拼接完成: {} 段, 总时长 {:.1}s, 章节 {} 个",
            ctx.audio_segments.len(),
            total_ms as f64 / 1000.0,
            chapters.len()
        );

        Ok(())
    }

    /// 阶段：音频提取
    ///
    /// docs/20 F63：此前是纯占位（只打一条 warn、从不设置 `ctx.audio_path`），
    /// 于是 `--kind subtitle` 即使文本读取侥幸通过，也必然在 ASR 阶段以
    /// 「找不到音频文件用于 ASR 识别」失败。现做两件实事：
    /// 1. 校验输入音频确实存在且可读，输出采样率/声道/时长等事实信息；
    /// 2. 非 WAV 格式**直接报错**，不再让错误静默传播（见下）。
    ///
    /// 非 WAV（mp3/m4a/flac）暂不支持解码：`votex-infra::audio` 只有 ffmpeg
    /// **编码**方向（wav→mp3/m4a），没有解码器；`AsrUseCase::load_audio`
    /// 遇到非 WAV 会返回 5 秒静音并继续跑完流程（docs/20 F60/F32），
    /// 产出与输入完全无关的垃圾字幕。这里提前失败，把问题暴露在提取阶段。
    fn run_audio_extract(ctx: &mut PipelineContext, _progress: &dyn Fn(ProgressEvent)) -> Result<()> {
        tracing::info!("[阶段] 音频提取");

        let Some(audio_path) = ctx.audio_path.clone() else {
            anyhow::bail!("音频提取阶段缺少音频输入：`--kind subtitle` 需要传入音频文件路径");
        };

        if !audio_path.is_file() {
            anyhow::bail!("音频文件不存在: {:?}", audio_path);
        }

        let ext = audio_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if ext != "wav" {
            anyhow::bail!(
                "暂不支持的音频格式 `{}`（当前仅支持 wav）：解码 mp3/m4a/flac 需要 ffmpeg 解码器，\
                 当前项目只实现了编码方向（docs/20 F60）。请先转为 wav 后重试: {:?}",
                ext,
                audio_path
            );
        }

        let audio = votex_infra::audio::wav::read_wav(&audio_path)?;
        tracing::info!(
            "  音频校验通过: {}Hz, {}声道, {}ms ({} 采样点)",
            audio.sample_rate,
            audio.channels,
            audio.duration_ms(),
            audio.samples.len()
        );
        Ok(())
    }

    /// 阶段：音频切片
    fn run_audio_slice(ctx: &mut PipelineContext, _progress: &dyn Fn(ProgressEvent)) -> Result<()> {
        tracing::info!("[阶段] 音频切片");

        if let Some(ref audio_path) = ctx.audio_path {
            if audio_path.exists() {
                let audio = votex_infra::audio::wav::read_wav(audio_path)?;
                let slice_dir = ctx.output_dir.join("slices");
                std::fs::create_dir_all(&slice_dir)?;

                let slice_seconds = 30u32;
                let samples_per_slice = (audio.sample_rate as usize * slice_seconds as usize) * audio.channels as usize;
                let total_slices = (audio.samples.len() + samples_per_slice - 1) / samples_per_slice;
                tracing::info!("  切片: {} 段", total_slices);
            }
        }

        Ok(())
    }

    /// 阶段：ASR 识别
    fn run_asr_recognize(ctx: &mut PipelineContext, _progress: &dyn Fn(ProgressEvent)) -> Result<()> {
        tracing::info!("[阶段] ASR 识别 (模型: {}, 格式: {})", ctx.asr_model, ctx.subtitle_format);

        let audio_path = match ctx.audio_path.as_ref().filter(|p| p.exists()) {
            Some(p) => p.clone(),
            None => {
                let fallback = ctx.output_dir.join(format!("{}.wav", ctx.base_name));
                if fallback.exists() {
                    fallback
                } else {
                    anyhow::bail!("找不到音频文件用于 ASR 识别");
                }
            }
        };

        let subtitle_ext = match ctx.subtitle_format.as_str() {
            "srt" => "srt",
            "lrc" => "lrc",
            _ => "txt",
        };
        let subtitle_path = ctx.output_dir.join(format!("{}.{}", ctx.base_name, subtitle_ext));

        let mut asr = AsrUseCase::new();
        asr.recognize(&audio_path, &subtitle_path, &ctx.asr_model, "zh", &ctx.subtitle_format)?;

        ctx.subtitle_path = Some(subtitle_path);
        tracing::info!("  字幕输出完成");
        Ok(())
    }

    /// 阶段：字幕对齐（占位）
    fn run_subtitle_align(_ctx: &mut PipelineContext, _progress: &dyn Fn(ProgressEvent)) -> Result<()> {
        tracing::info!("[阶段] 字幕对齐");
        tracing::info!("  字幕已在 ASR 阶段完成对齐");
        Ok(())
    }

    /// 阶段：导出
    fn run_export(ctx: &mut PipelineContext, _progress: &dyn Fn(ProgressEvent)) -> Result<()> {
        tracing::info!("[阶段] 导出");

        if let Some(ref audio_path) = ctx.audio_path {
            if audio_path.exists() {
                let size = std::fs::metadata(audio_path).map(|m| m.len()).unwrap_or(0);
                tracing::info!("  📄 音频: {:?} ({} 字节)", audio_path, size);
            }
        }

        if let Some(ref subtitle_path) = ctx.subtitle_path {
            if subtitle_path.exists() {
                let size = std::fs::metadata(subtitle_path).map(|m| m.len()).unwrap_or(0);
                tracing::info!("  📄 字幕: {:?} ({} 字节)", subtitle_path, size);
            }
        }

        tracing::info!("  输出目录: {:?}", ctx.output_dir);
        Ok(())
    }

    /// 阶段：OCR 识别（占位）
    fn run_ocr_recognize(_ctx: &mut PipelineContext, _progress: &dyn Fn(ProgressEvent)) -> Result<()> {
        tracing::info!("[阶段] OCR 识别（略过：流水线暂不包含 OCR）");
        Ok(())
    }
}

#[cfg(test)]
mod audio_extract_tests {
    use super::*;
    use std::path::PathBuf;
    use votex_domain::shared::value_object::AudioData;

    /// 构造一个只关心音频提取阶段的最小上下文
    fn ctx_with_audio(audio_path: Option<PathBuf>) -> PipelineContext {
        PipelineContext {
            raw_text: String::new(),
            processed_text: String::new(),
            segments: Vec::new(),
            segment_chapter_starts: Vec::new(),
            audio_segments: Vec::new(),
            audio_path,
            subtitle_path: None,
            base_name: "t".to_string(),
            output_dir: std::env::temp_dir(),
            engine_kind: EngineKind::Kokoro,
            voice: "zf_001".to_string(),
            speed: 1.0,
            asr_model: "sensevoice".to_string(),
            subtitle_format: "srt".to_string(),
        }
    }

    /// 写入一个 0.1 秒的合法 WAV，返回其路径
    fn write_test_wav(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        let mut audio = AudioData::silence(24000, 100);
        // 非全零，避免退化成静音文件
        for (i, s) in audio.samples.iter_mut().enumerate() {
            *s = (i as f32 * 0.01).sin() * 0.5;
        }
        WavWriter::write(&audio, &path).expect("写入测试 WAV 失败");
        path
    }

    /// docs/20 F63：WAV 输入应通过校验
    #[test]
    fn 音频提取_wav输入校验通过() {
        let dir = std::env::temp_dir().join("votex_pl_audio_extract_ok");
        std::fs::create_dir_all(&dir).unwrap();
        let wav = write_test_wav(&dir, "ok.wav");

        let mut ctx = ctx_with_audio(Some(wav));
        PipelineUseCase::run_audio_extract(&mut ctx, &|_| {}).expect("合法 WAV 应通过校验");
    }

    /// docs/20 F63：占位实现时期此路径静默 Ok(())
    #[test]
    fn 音频提取_文件不存在时报错() {
        let mut ctx = ctx_with_audio(Some(PathBuf::from("definitely_not_exist_9d3f.wav")));
        let err = PipelineUseCase::run_audio_extract(&mut ctx, &|_| {})
            .expect_err("不存在的音频应报错");
        assert!(
            err.to_string().contains("音频文件不存在"),
            "错误信息应说明文件不存在，实际: {}",
            err
        );
    }

    /// docs/20 F60/F63：非 WAV 必须在提取阶段就失败，
    /// 不能流到 `AsrUseCase::load_audio` 那里变成「返回静音」并产出垃圾字幕
    #[test]
    fn 音频提取_非wav格式明确报错() {
        let dir = std::env::temp_dir().join("votex_pl_audio_extract_ext");
        std::fs::create_dir_all(&dir).unwrap();
        let fake_mp3 = dir.join("fake.mp3");
        std::fs::write(&fake_mp3, b"ID3 not really an mp3").unwrap();

        let mut ctx = ctx_with_audio(Some(fake_mp3));
        let err = PipelineUseCase::run_audio_extract(&mut ctx, &|_| {})
            .expect_err("非 WAV 输入应报错");
        let msg = err.to_string();
        assert!(msg.contains("仅支持 wav"), "错误信息应指明仅支持 wav，实际: {}", msg);
        assert!(
            msg.contains("F60"),
            "错误信息应指向已登记缺陷编号，实际: {}",
            msg
        );
    }

    /// docs/20 F63：缺少音频输入时明确报错，而不是静默放行
    #[test]
    fn 音频提取_缺少音频输入时报错() {
        let mut ctx = ctx_with_audio(None);
        let err = PipelineUseCase::run_audio_extract(&mut ctx, &|_| {})
            .expect_err("无音频输入应报错");
        assert!(
            err.to_string().contains("缺少音频输入"),
            "错误信息应说明缺少音频输入，实际: {}",
            err
        );
    }
}

#[cfg(test)]
mod chapter_tests {
    use super::*;

    /// docs/20 F74：章节只认真实章节边界段，正文段不得成为章节
    #[test]
    fn 章节表_只认真实章节边界正文段不产生伪章节() {
        let chapter_starts = vec![
            Some("第一章 归乡".to_string()),
            None, // 正文段（修复前会被截 20 字当伪章节标题）
            Some("第二章 旧事".to_string()),
            None,
        ];
        // 修复前的问题形态：正文行「　　“你！”女人面目狰狞…」被当章边界
        let durations_ms = [1000u64, 1000, 1000, 1000];
        let chapters = build_chapter_timings(&chapter_starts, &durations_ms, 300);

        assert_eq!(chapters.len(), 2, "2 个真实章节边界 → 2 个章节");
        assert_eq!(chapters[0].title, "第一章 归乡");
        assert_eq!(chapters[0].start_ms, 0);
        assert_eq!(chapters[1].title, "第二章 旧事");
        // 第 2 章起点 = 前 2 段时长 + 2 段间静音
        assert_eq!(chapters[1].start_ms, 2600);
        for ch in &chapters {
            assert!(
                !ch.title.contains('，'),
                "章节名不应是正文截断（伪章节），实际: {}",
                ch.title
            );
        }
    }

    /// docs/20 F74：全文无章节标记时，给单一「全文」章节兜底（F64 语义保留）
    #[test]
    fn 章节表_无章节标记时兜底全文单章() {
        let chapter_starts = vec![None::<String>, None, None];
        let durations_ms = [1000u64, 1000, 1000];
        let chapters = build_chapter_timings(&chapter_starts, &durations_ms, 300);

        assert_eq!(chapters.len(), 1);
        assert_eq!(chapters[0].title, "全文");
        assert_eq!(chapters[0].start_ms, 0);
    }

    /// docs/20 F74：边界标题为空白串时跳过，不产出空标题章节
    #[test]
    fn 章节表_空白标题边界被跳过() {
        let chapter_starts = vec![Some("   ".to_string()), Some("第二章 旧事".to_string())];
        let durations_ms = [1000u64, 1000];
        let chapters = build_chapter_timings(&chapter_starts, &durations_ms, 300);

        assert_eq!(chapters.len(), 1);
        assert_eq!(chapters[0].title, "第二章 旧事");
        assert_eq!(chapters[0].start_ms, 1300);
    }

    /// docs/20 F64：拼接后必须写出 `{音频名}.chapters.json`，
    /// 且章节起点与实际拼接顺序一致（含 300ms 段间静音偏移）
    #[test]
    fn 音频拼接_写出章节表且起点含段间静音偏移() {
        let dir = std::env::temp_dir().join("votex_pl_chapters");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();

        let mut ctx = PipelineContext {
            raw_text: String::new(),
            processed_text: String::new(),
            segments: vec![
                "第一章 归乡".to_string(),
                "夜色像一层薄薄的霜，落在村口那棵老槐树上。林远站在原地，看着那扇熟悉的木门，门上的漆已经斑驳。".to_string(),
                "第二章 旧事".to_string(),
            ],
            segment_chapter_starts: vec![
                Some("第一章 归乡".to_string()),
                None,
                Some("第二章 旧事".to_string()),
            ],
            audio_segments: vec![
                AudioData::silence(24000, 1000),
                AudioData::silence(24000, 1000),
                AudioData::silence(24000, 1000),
            ],
            audio_path: None,
            subtitle_path: None,
            base_name: "book".to_string(),
            output_dir: dir.clone(),
            engine_kind: EngineKind::Kokoro,
            voice: "zf_001".to_string(),
            speed: 1.0,
            asr_model: "sensevoice".to_string(),
            subtitle_format: "srt".to_string(),
        };

        PipelineUseCase::run_audio_concat(&mut ctx, &|_| {}).expect("拼接应成功");

        let wav = dir.join("book.wav");
        let chapters_json = wav.with_extension("chapters.json");
        assert!(wav.is_file(), "应产出拼接音频");
        assert!(
            chapters_json.is_file(),
            "docs/20 F64：应产出章节表 {}",
            chapters_json.display()
        );

        let raw = std::fs::read_to_string(&chapters_json).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&raw).unwrap();
        let arr = parsed.as_array().expect("章节表应为数组");
        // docs/20 F74：3 段中只有 2 个真实章节边界 → 2 个章节（正文段不再是章节）
        assert_eq!(arr.len(), 2, "正文段不应产生伪章节");

        // 起点 = 前序段时长 + 段间静音 300ms
        assert_eq!(arr[0]["start_ms"], 0);
        assert_eq!(arr[1]["start_ms"], 2600);
        // 末章 end_ms = 总长（3×1000 + 2×300 = 3600）
        assert_eq!(arr[1]["end_ms"], 3600);
        // 章节名即真实章节标题行
        assert_eq!(arr[0]["title"], "第一章 归乡");
        assert_eq!(arr[1]["title"], "第二章 旧事");

        // 拼接音频实际时长应与章节表总长一致（防止时间戳与音频脱节）
        let audio = votex_infra::audio::wav::read_wav(&wav).unwrap();
        assert_eq!(audio.duration_ms() as u64, 3600, "音频时长应与章节表总长一致");

        let _ = std::fs::remove_dir_all(&dir);
    }
}
