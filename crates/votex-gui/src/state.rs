use std::path::PathBuf;
use std::sync::{mpsc, Arc};

use crate::widgets::audio_player::AudioPlayerState;
use crate::widgets::subtitle_editor::SubtitleEntry;

/// 后台任务事件接收器
pub type TaskEventReceiver = mpsc::Receiver<(Page, crate::task_runner::TaskEvent)>;
/// 后台任务事件发送器
pub type TaskEventSender = mpsc::Sender<(Page, crate::task_runner::TaskEvent)>;

/// 页面路由
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Page {
    Dashboard,
    Tts,
    Asr,
    Dictation,
    Voices,
    Ocr,
    Translation,
    Batch,
    Pipeline,
    Video,
    Settings,
}

impl Page {
    pub fn title(&self) -> &str {
        match self {
            Page::Dashboard => "首页",
            Page::Tts => "语音合成",
            Page::Asr => "语音识别",
            Page::Dictation => "实时听写",
            Page::Voices => "音色库",
            Page::Ocr => "图文识别",
            Page::Translation => "文本翻译",
            Page::Batch => "批量任务",
            Page::Pipeline => "流水线",
            Page::Video => "视频生成",
            Page::Settings => "系统设置",
        }
    }

    pub fn all() -> &'static [Page] {
        &[
            Page::Dashboard,
            Page::Tts,
            Page::Asr,
            Page::Dictation,
            Page::Voices,
            Page::Ocr,
            Page::Translation,
            Page::Batch,
            Page::Pipeline,
            Page::Video,
            Page::Settings,
        ]
    }
}

/// TTS 页面状态
#[derive(Debug, Clone)]
pub struct TtsState {
    pub input_text: String,
    pub output_path: String,
    pub engine: String,
    pub voice: String,
    pub speed: f32,
    pub format: String,
    pub language: String,
    pub is_running: bool,
    pub progress: f32,
    pub progress_text: String,
    pub result_message: String,
    /// Qwen3-TTS 模型变体选择结果（"0.6b" / "1.7b" / ""=未选择）
    pub qwen3_variant: String,
    /// 是否显示 Qwen3 模型选择弹窗
    pub show_qwen3_selector: bool,
    /// 是否显示资源建议弹窗
    pub show_resource_recommendation: bool,
    /// 用户已关闭资源建议弹窗（不再自动弹出）
    pub resource_tip_dismissed: bool,
    /// 当前任务的取消令牌（合成按钮创建，取消按钮触发）
    pub cancel_token: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl Default for TtsState {
    fn default() -> Self {
        Self {
            input_text: String::new(),
            output_path: "tmp/output.wav".to_string(),
            engine: "kokoro".to_string(),
            voice: "zf_001".to_string(),
            speed: 1.0,
            format: "wav".to_string(),
            language: "zh".to_string(),
            is_running: false,
            progress: 0.0,
            progress_text: String::new(),
            result_message: String::new(),
            qwen3_variant: String::new(),
            show_qwen3_selector: false,
            show_resource_recommendation: false,
            resource_tip_dismissed: false,
            cancel_token: None,
        }
    }
}

/// ASR 页面状态
#[derive(Debug, Clone)]
pub struct AsrState {
    pub input_path: String,
    pub output_path: String,
    pub model: String,
    pub language: String,
    pub format: String,
    /// 词级时间戳对齐（精准字幕 + words.json）
    pub word_timestamps: bool,
    /// 说话人分离（说话人标注字幕 + diarization.json）
    pub diarize: bool,
    /// 期望说话人数（0 = 自动聚类）
    pub num_speakers: u32,
    pub is_running: bool,
    pub progress: f32,
    pub progress_text: String,
    pub result_message: String,
    pub result_text: String,
    /// 当前任务的取消令牌（启动按钮创建，取消按钮触发）
    ///
    /// 必须由页面持有并复用，不能在启动时临时 `Arc::new` 后丢弃——
    /// 那样取消按钮拿不到同一个令牌，点了没反应，
    /// 后台任务仍会跑完（此前的 ASR / OCR 页面就是这个 bug）。
    pub cancel_token: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl Default for AsrState {
    fn default() -> Self {
        Self {
            input_path: String::new(),
            output_path: "output.srt".to_string(),
            model: "whisper-base".to_string(),
            language: "zh".to_string(),
            format: "srt".to_string(),
            word_timestamps: false,
            diarize: false,
            num_speakers: 0,
            is_running: false,
            progress: 0.0,
            progress_text: String::new(),
            result_message: String::new(),
            result_text: String::new(),
            cancel_token: None,
        }
    }
}

/// 实时听写页面状态
///
/// 不派生 Clone/Debug：事件接收器与听写句柄不可复制。
/// 事件不走全局 task_tx（TaskEvent 无流式增量语义），
/// 使用页面专属 channel，每帧在页面 show() 里 try_recv 拉取。
pub struct DictationState {
    /// 听写模型（当前支持 streaming-zipformer）
    pub model: String,
    /// 转写保存路径（空 = 不落盘）
    pub output_path: String,
    pub is_running: bool,
    /// 当前未定稿文本（灰色展示，随识别覆盖）
    pub partial_text: String,
    /// 已定稿全文（可编辑）
    pub final_text: String,
    /// 录音累计时长（毫秒，用于显示）
    pub elapsed_ms: u64,
    /// 状态提示（就绪/错误/统计）
    pub status_message: String,
    /// 听写事件接收端（start 时创建）
    pub event_rx: Option<std::sync::mpsc::Receiver<votex_app::use_case::dictation_use_case::DictationEvent>>,
    /// 听写句柄（start 时创建，停止按钮触发）
    pub handle: Option<Arc<std::sync::Mutex<votex_app::use_case::dictation_use_case::DictationHandle>>>,
}

impl Default for DictationState {
    fn default() -> Self {
        Self {
            model: "streaming-zipformer".to_string(),
            output_path: String::new(),
            is_running: false,
            partial_text: String::new(),
            final_text: String::new(),
            elapsed_ms: 0,
            status_message: String::new(),
            event_rx: None,
            handle: None,
        }
    }
}

/// OCR 页面状态（单个页面在任务列表中的表示）
#[derive(Debug, Clone)]
pub struct OcrPageState {
    pub index: usize,
    pub image_path: String,
    pub status: String,
    pub confidence: f32,
    pub block_count: usize,
}

/// OCR 页面状态
#[derive(Debug, Clone)]
pub struct OcrState {
    pub engine: String,
    pub input_path: String,
    pub output_path: String,
    pub output_format: String,
    pub no_cls: bool,
    pub is_running: bool,
    pub progress: f32,
    pub progress_text: String,
    pub result_message: String,
    pub pages: Vec<OcrPageState>,
    pub task_id: String,
    pub use_batch: bool,
    pub batch_inputs: Vec<String>,
    /// 当前任务的取消令牌（启动按钮创建，取消按钮触发）
    ///
    /// 必须由页面持有并复用，不能在启动时临时 `Arc::new` 后丢弃——
    /// 那样取消按钮拿不到同一个令牌，点了没反应，
    /// 后台任务仍会跑完（此前的 ASR / OCR 页面就是这个 bug）。
    pub cancel_token: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    pub batch_output_dir: String,
    pub max_concurrency: usize,
}

impl Default for OcrState {
    fn default() -> Self {
        Self {
            engine: "paddleocr-v6-medium".to_string(),
            input_path: String::new(),
            output_path: String::new(),
            output_format: "txt".to_string(),
            no_cls: false,
            is_running: false,
            progress: 0.0,
            progress_text: String::new(),
            result_message: String::new(),
            pages: Vec::new(),
            task_id: String::new(),
            use_batch: false,
            batch_inputs: Vec::new(),
            batch_output_dir: "ocr_output".to_string(),
            max_concurrency: 2,
            cancel_token: None,
        }
    }
}

/// 批量任务状态
#[derive(Debug, Clone)]
pub struct BatchState {
    pub task_list: Vec<BatchTaskItem>,
    pub is_running: bool,
    pub input_dir: String,
    pub output_dir: String,
    pub engine: String,
    pub model: String,
    pub voice: String,
    pub speed: f32,
    pub format: String,
    pub language: String,
    pub concurrency: usize,
    pub progress: f32,
    pub progress_text: String,
    pub result_message: String,
    /// 当前批量任务的取消令牌
    pub cancel_token: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl Default for BatchState {
    fn default() -> Self {
        Self {
            task_list: Vec::new(),
            is_running: false,
            input_dir: String::new(),
            output_dir: "batch_output".to_string(),
            engine: "kokoro".to_string(),
            model: "whisper-base".to_string(),
            voice: "zf_001".to_string(),
            speed: 1.0,
            format: "wav".to_string(),
            language: "zh".to_string(),
            concurrency: 1,
            progress: 0.0,
            progress_text: String::new(),
            result_message: String::new(),
            cancel_token: None,
        }
    }
}

/// 批量任务条目
#[derive(Debug, Clone)]
pub struct BatchTaskItem {
    pub name: String,
    pub kind: String,
    pub status: String,
    pub progress: f32,
}

/// 设置页面状态
#[derive(Debug, Clone)]
pub struct SettingsState {
    pub models_dir: String,
    pub log_level: String,
    pub mirror: String,
    pub model_list: Vec<ModelItem>,
    pub model_filter: String,
    // -------- 推理资源限制 --------
    /// intra-op CPU 线程数（0=自动）
    pub cpu_threads: u32,
    /// 内存上限（MB），0=不限
    pub memory_limit_mb: u32,
    /// 执行模式
    pub execution_mode: String,
    /// 当前选中的 TTS 引擎（用于资源建议提示）
    pub selected_tts_engine: String,
    /// 是否显示资源建议弹窗
    pub show_resource_tip: bool,
    // -------- 模型删除二次确认 --------
    /// 待二次确认删除的模型 ID
    ///
    /// AGENTS.md 约束「禁止自动删除模型文件，必须二次确认操作」：
    /// 点击删除按钮只登记此项，真正的文件删除在确认弹窗中完成
    pub pending_delete: Option<String>,
    /// 删除操作结果提示（成功或失败原因），用于界面反馈
    pub delete_result: Option<String>,
    /// 配置保存结果提示
    pub config_message: Option<String>,
}

impl Default for SettingsState {
    fn default() -> Self {
        Self {
            models_dir: "models".to_string(),
            log_level: "info".to_string(),
            mirror: "cn".to_string(),
            model_list: Vec::new(),
            model_filter: "全部".to_string(),
            cpu_threads: 0,
            memory_limit_mb: 0,
            execution_mode: "sequential".to_string(),
            selected_tts_engine: String::new(),
            show_resource_tip: false,
            pending_delete: None,
            delete_result: None,
            config_message: None,
        }
    }
}

/// 模型条目
#[derive(Debug, Clone)]
pub struct ModelItem {
    pub id: String,
    pub name: String,
    pub kind: String,
    pub engine: String,
    pub status: String,
}

/// 流水线页面状态
#[derive(Debug, Clone)]
pub struct PipelineState {
    pub template_name: String,
    pub steps: Vec<PipelineStep>,
    pub is_running: bool,
    pub progress: f32,
    pub progress_text: String,
    pub result_message: String,
    pub input_path: String,
    pub output_dir: String,
    pub engine: String,
    pub voice: String,
    pub speed: f32,
    /// 当前任务的取消令牌（启动按钮创建，取消按钮触发）
    pub cancel_token: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl Default for PipelineState {
    fn default() -> Self {
        Self {
            template_name: String::new(),
            steps: Vec::new(),
            is_running: false,
            progress: 0.0,
            progress_text: String::new(),
            result_message: String::new(),
            input_path: "input.txt".to_string(),
            output_dir: "output".to_string(),
            engine: "kokoro".to_string(),
            voice: "zf_001".to_string(),
            speed: 1.0,
            cancel_token: None,
        }
    }
}

/// 流水线步骤
#[derive(Debug, Clone)]
pub struct PipelineStep {
    pub name: String,
    pub kind: String,
    pub enabled: bool,
    pub status: String,
}

/// 视频生成页面状态
#[derive(Debug, Clone)]
pub struct VideoState {
    /// 页面模式："generate"（AI 短视频生成）/ "dub"（视频配音）
    pub mode: String,
    pub script: String,
    pub output_path: String,
    pub tts_engine: String,
    pub tts_voice: String,
    pub material_source: String,
    pub material_keyword: String,
    pub is_running: bool,
    pub progress: f32,
    pub progress_text: String,
    pub result_message: String,
    pub translated_keyword: String,
    pub material_preview_url: String,
    // ---- 配音模式（dub）----
    /// 源视频路径
    pub dub_video_path: String,
    /// ASR 引擎（识别原声）
    pub dub_asr_engine: String,
    /// 配音 TTS 引擎
    pub dub_tts_engine: String,
    /// 配音音色
    pub dub_voice: String,
    /// 配音语速
    pub dub_speed: f32,
    /// 翻译方向（空 = 不翻译，直接用原声文本配音）
    pub dub_translate: String,
    /// 单段音频最大加速倍率（时长拟合上限）
    pub dub_max_tempo: f32,
    /// 二次 ASR 质检（对配音结果再识别，验证可懂度）
    pub dub_verify: bool,
    /// 保留背景音（原音轨按低音量与配音混音）；false = 替换原音轨
    pub dub_keep_bg: bool,
    /// 输出目录
    pub dub_output_dir: String,
    /// 当前任务的取消令牌（启动按钮创建，取消按钮触发）
    pub cancel_token: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl Default for VideoState {
    fn default() -> Self {
        Self {
            mode: "generate".to_string(),
            script: String::new(),
            output_path: "output.mp4".to_string(),
            tts_engine: "kokoro".to_string(),
            tts_voice: "zf_001".to_string(),
            material_source: "pexels".to_string(),
            material_keyword: String::new(),
            is_running: false,
            progress: 0.0,
            progress_text: String::new(),
            result_message: String::new(),
            translated_keyword: String::new(),
            material_preview_url: String::new(),
            dub_video_path: String::new(),
            dub_asr_engine: "paraformer".to_string(),
            dub_tts_engine: "kokoro".to_string(),
            dub_voice: "zf_001".to_string(),
            dub_speed: 1.0,
            dub_translate: String::new(),
            dub_max_tempo: 1.5,
            dub_verify: true,
            dub_keep_bg: false,
            dub_output_dir: "output".to_string(),
            cancel_token: None,
        }
    }
}

/// 音色库页面状态（克隆音色管理）
///
/// 列表/添加/删除是毫秒级文件操作，直接在 UI 线程同步执行，
/// 不走 task_runner。
#[derive(Debug, Clone)]
pub struct VoicesState {
    /// 音色列表（进入页面/操作后刷新）
    pub voices: Vec<votex_app::platform::tts::CloneVoiceMeta>,
    /// 新音色名
    pub new_name: String,
    /// 参考音频路径
    pub new_reference: String,
    /// 参考音频转写（CosyVoice 必需）
    pub new_transcript: String,
    /// 入库时降噪
    pub new_denoise: bool,
    /// 引擎参考音频上限（秒，0 = 不截取）
    pub new_max_ref_seconds: u64,
    /// 待确认删除的音色名（二次确认弹层）
    pub pending_remove: Option<String>,
    /// 操作结果消息
    pub message: Option<String>,
    /// 当前试听的音色名
    pub previewing: Option<String>,
}

impl Default for VoicesState {
    fn default() -> Self {
        Self {
            voices: Vec::new(),
            new_name: String::new(),
            new_reference: String::new(),
            new_transcript: String::new(),
            new_denoise: true,
            new_max_ref_seconds: 15,
            pending_remove: None,
            message: None,
            previewing: None,
        }
    }
}

/// 首次启动引导状态
#[derive(Debug, Clone)]
pub struct OnboardingState {
    pub step: usize,
    pub total_steps: usize,
    pub downloading: bool,
    pub download_progress: f32,
    pub download_model: String,
    pub selected_kokoro: bool,
    pub selected_whisper_base: bool,
    pub selected_whisper_small: bool,
}

impl Default for OnboardingState {
    fn default() -> Self {
        Self {
            step: 0,
            total_steps: 3,
            downloading: false,
            download_progress: 0.0,
            download_model: String::new(),
            selected_kokoro: true,
            selected_whisper_base: true,
            selected_whisper_small: false,
        }
    }
}

/// 翻译页面状态
#[derive(Debug, Clone)]
pub struct TranslationState {
    pub source_text: String,
    pub target_text: String,
    pub engine: String,
    pub direction: String,
    pub is_running: bool,
    pub progress: f32,
    pub progress_text: String,
    pub result_message: String,
    /// 当前任务的取消令牌（启动按钮创建，取消按钮触发）
    pub cancel_token: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

impl Default for TranslationState {
    fn default() -> Self {
        Self {
            source_text: String::new(),
            target_text: String::new(),
            engine: "dict".to_string(),
            direction: "zh-en".to_string(),
            is_running: false,
            progress: 0.0,
            progress_text: String::new(),
            result_message: String::new(),
            cancel_token: None,
        }
    }
}

/// GUI 应用全局状态
pub struct AppState {
    pub current_page: Page,
    /// 模型文件存储根目录（从 CLI 或配置文件传入）
    pub models_dir: String,
    /// 配置文件路径（由 bootstrap 传入；设置页读写它，
    /// 不能写死相对路径——GUI 双击启动时 cwd 任意）
    pub config_path: String,
    pub tts: TtsState,
    pub asr: AsrState,
    pub dictation: DictationState,
    pub voices: VoicesState,
    pub ocr: OcrState,
    pub translation: TranslationState,
    pub batch: BatchState,
    pub pipeline: PipelineState,
    pub video: VideoState,
    pub settings: SettingsState,
    pub onboarding: OnboardingState,
    pub audio_player: AudioPlayerState,
    pub subtitle_entries: Vec<SubtitleEntry>,
    pub initialized: bool,
    pub show_onboarding: bool,
    pub dropped_file: Option<PathBuf>,
    /// 后台任务事件发送器
    pub task_tx: Option<TaskEventSender>,
    /// 后台任务事件接收器
    pub task_rx: Option<TaskEventReceiver>,
    /// 流水线持久化仓储（可选，用于故障恢复）
    pub pipeline_repo: Option<Arc<dyn votex_domain::repository::PipelineRepository>>,
    /// 下载记录持久化仓储（可选，用于断点续传追踪）
    pub download_repo: Option<Arc<dyn votex_domain::repository::DownloadRepository>>,
}

impl AppState {
    /// 创建应用状态，使用传入的模型目录
    pub fn new(models_dir: String) -> Self {
        let (task_tx, task_rx) = mpsc::channel();
        Self {
            current_page: Page::Dashboard,
            models_dir: models_dir.clone(),
            config_path: "application.yml".to_string(),
            tts: TtsState::default(),
            asr: AsrState::default(),
            dictation: DictationState::default(),
            voices: VoicesState::default(),
            ocr: OcrState::default(),
            translation: TranslationState::default(),
            batch: BatchState::default(),
            pipeline: PipelineState::default(),
            video: VideoState::default(),
            settings: SettingsState {
                models_dir,
                ..Default::default()
            },
            onboarding: OnboardingState::default(),
            audio_player: AudioPlayerState::default(),
            subtitle_entries: Vec::new(),
            initialized: false,
            show_onboarding: false,
            dropped_file: None,
            task_tx: Some(task_tx),
            task_rx: Some(task_rx),
            pipeline_repo: None,
            download_repo: None,
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        // 走统一路径解析，不写死 "models" 字面量
        Self::new(
            votex_app::platform::paths::models_dir()
                .display()
                .to_string(),
        )
    }
}
