use crate::model::value_object::EngineKind;
use serde::{Deserialize, Serialize};

/// TTS 引擎名提示（供 CLI 报错信息复用，避免各子命令各写一份）
pub const TTS_ENGINE_HINT: &str = "kokoro, indextts2, qwen3, cosyvoice3";

/// 解析 TTS 引擎名（CLI / 配置的统一入口）
///
/// 输入域覆盖 CLI 历代在用的 4 个串：`kokoro` / `indextts2` / `qwen3` / `cosyvoice3`，
/// 同时接受 `EngineKind::as_str()` 的规范名与两个在线引擎（azure-tts / aliyun-tts）。
///
/// 不直接复用 `EngineKind::from_str`：它的输入域是全量引擎名且**不含 `qwen3`**
/// （只有 `qwen3-tts`/`qwen3tts`），而 `qwen` 会被解析成 `QwenLlm`，
/// 用在 TTS 场景会静默拿到错误的引擎。
///
/// **新增 TTS 引擎时只改这一处**。此前 CLI 的 `tts` / `batch tts` / `video` /
/// `pipeline` 各自写了一份 match，其中 `pipeline` 只映射了 2 个引擎，
/// 导致 `--engine cosyvoice3` 直接 `不支持的 TTS 引擎`（docs/20 F62）。
pub fn parse_tts_engine(s: &str) -> Option<EngineKind> {
    match s.to_lowercase().as_str() {
        "kokoro" => Some(EngineKind::Kokoro),
        "indextts2" | "indextts" => Some(EngineKind::IndexTTS2),
        "qwen3" | "qwen3-tts" | "qwen3tts" => Some(EngineKind::Qwen3Tts),
        "cosyvoice3" | "cosyvoice" => Some(EngineKind::CosyVoice3),
        "azure-tts" | "azuretts" => Some(EngineKind::AzureTts),
        "aliyun-tts" | "aliyuntts" => Some(EngineKind::AliyunTts),
        _ => None,
    }
}

/// 输入来源
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputSource {
    DirectInput,
    FileImport,
}

/// 文件编码
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileEncoding {
    Utf8,
    Gbk,
    Unknown,
}

/// TTS 输入
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsInput {
    pub source: InputSource,
    pub raw_text: Option<String>,
    pub file_path: Option<std::path::PathBuf>,
    pub encoding: FileEncoding,
}

/// 音色标识
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VoiceId {
    pub id: String,
    pub display_name: String,
    pub engine: EngineKind,
}

impl VoiceId {
    pub fn new(id: &str, display_name: &str, engine: EngineKind) -> Self {
        Self {
            id: id.to_string(),
            display_name: display_name.to_string(),
            engine,
        }
    }
}

/// 语速（0.5~2.0）
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Speed(f32);

impl Speed {
    pub fn new(value: f32) -> Option<Self> {
        if (0.5..=2.0).contains(&value) {
            Some(Self(value))
        } else {
            None
        }
    }

    pub fn value(&self) -> f32 {
        self.0
    }

    pub fn default_value() -> Self {
        Self(1.0)
    }
}

impl Default for Speed {
    fn default() -> Self {
        Self::default_value()
    }
}

/// 音调（-20~+20）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pitch(i32);

impl Pitch {
    pub fn new(value: i32) -> Option<Self> {
        if (-20..=20).contains(&value) {
            Some(Self(value))
        } else {
            None
        }
    }

    pub fn value(&self) -> i32 {
        self.0
    }
}

impl Default for Pitch {
    fn default() -> Self {
        Self(0)
    }
}

/// 音量（0~100）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Volume(u8);

impl Volume {
    pub fn new(value: u8) -> Option<Self> {
        if value <= 100 {
            Some(Self(value))
        } else {
            None
        }
    }

    pub fn value(&self) -> u8 {
        self.0
    }
}

impl Default for Volume {
    fn default() -> Self {
        Self(80)
    }
}

/// 分段字数
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentSize {
    /// 120 字（中文安全值：v1.1-zh 注音 ~3 token/字，120字 ≈ 420 token < 512 限制）
    S120,
    /// 300 字（英文安全值）
    S300,
    /// 500 字
    S500,
    /// 800 字
    S800,
}

impl SegmentSize {
    pub fn max_chars(&self) -> usize {
        match self {
            SegmentSize::S120 => 120,
            SegmentSize::S300 => 300,
            SegmentSize::S500 => 500,
            SegmentSize::S800 => 800,
        }
    }
}

impl Default for SegmentSize {
    fn default() -> Self {
        SegmentSize::S500
    }
}

/// 降噪强度
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DenoiseLevel {
    Low,
    Medium,
    High,
}

impl Default for DenoiseLevel {
    fn default() -> Self {
        DenoiseLevel::Low
    }
}

/// 音频格式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioFormat {
    Wav,
    Mp3,
    M4A,
    Flac,
    /// 有声书容器（AAC 编码 + 章节元数据，播放器可按章节跳转）
    M4B,
}

impl AudioFormat {
    pub fn extension(&self) -> &str {
        match self {
            AudioFormat::Wav => "wav",
            AudioFormat::Mp3 => "mp3",
            AudioFormat::M4A => "m4a",
            AudioFormat::Flac => "flac",
            AudioFormat::M4B => "m4b",
        }
    }

    /// 是否为基于 ffmpeg 编码的容器格式（走临时 WAV 中转）
    pub fn needs_ffmpeg(&self) -> bool {
        !matches!(self, AudioFormat::Wav)
    }
}

/// TTS 情感/情绪标签（参考 PilotTTS 11 种情感分类）
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Emotion {
    /// 中性 / 无情感
    Neutral,
    /// 高兴/愉快
    Happy,
    /// 悲伤/难过
    Sad,
    /// 愤怒/生气
    Angry,
    /// 恐惧/害怕
    Fearful,
    /// 惊讶/吃惊
    Surprised,
    /// 厌恶/反感
    Disgusted,
    /// 轻蔑/蔑视
    Contempt,
    /// 严肃/庄重
    Serious,
    /// 忧郁/低落
    Blue,
    /// 关切/关心
    Concern,
    /// 心理/悬疑氛围
    Psychology,
    /// 耳语/悄悄话
    Whisper,
}

impl Emotion {
    /// 返回情感标签的英文标识
    pub fn as_str(&self) -> &str {
        match self {
            Emotion::Neutral => "neutral",
            Emotion::Happy => "happy",
            Emotion::Sad => "sad",
            Emotion::Angry => "angry",
            Emotion::Fearful => "fearful",
            Emotion::Surprised => "surprised",
            Emotion::Disgusted => "disgusted",
            Emotion::Contempt => "contempt",
            Emotion::Serious => "serious",
            Emotion::Blue => "blue",
            Emotion::Concern => "concern",
            Emotion::Psychology => "psychology",
            Emotion::Whisper => "whisper",
        }
    }

    /// 返回情感的中文名称
    pub fn display_cn(&self) -> &'static str {
        match self {
            Emotion::Neutral => "中性",
            Emotion::Happy => "高兴",
            Emotion::Sad => "悲伤",
            Emotion::Angry => "愤怒",
            Emotion::Fearful => "恐惧",
            Emotion::Surprised => "惊讶",
            Emotion::Disgusted => "厌恶",
            Emotion::Contempt => "轻蔑",
            Emotion::Serious => "严肃",
            Emotion::Blue => "忧郁",
            Emotion::Concern => "关切",
            Emotion::Psychology => "悬疑",
            Emotion::Whisper => "耳语",
        }
    }

    /// 从字符串解析情感标签
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "neutral" | "中" | "中性" => Some(Emotion::Neutral),
            "happy" | "高兴" | "愉快" => Some(Emotion::Happy),
            "sad" | "悲伤" | "难过" => Some(Emotion::Sad),
            "angry" | "愤怒" | "生气" => Some(Emotion::Angry),
            "fearful" | "fear" | "恐惧" | "害怕" => Some(Emotion::Fearful),
            "surprised" | "surprise" | "惊讶" | "吃惊" => Some(Emotion::Surprised),
            "disgusted" | "disgust" | "厌恶" | "反感" => Some(Emotion::Disgusted),
            "contempt" | "轻蔑" | "蔑视" => Some(Emotion::Contempt),
            "serious" | "严肃" | "庄重" => Some(Emotion::Serious),
            "blue" | "忧郁" | "低落" => Some(Emotion::Blue),
            "concern" | "关切" | "关心" => Some(Emotion::Concern),
            "psychology" | "悬疑" | "心理" => Some(Emotion::Psychology),
            "whisper" | "耳语" => Some(Emotion::Whisper),
            _ => None,
        }
    }

    /// 列出所有情感
    pub fn all() -> Vec<Emotion> {
        vec![
            Emotion::Neutral,
            Emotion::Happy,
            Emotion::Sad,
            Emotion::Angry,
            Emotion::Fearful,
            Emotion::Surprised,
            Emotion::Disgusted,
            Emotion::Contempt,
            Emotion::Serious,
            Emotion::Blue,
            Emotion::Concern,
            Emotion::Psychology,
            Emotion::Whisper,
        ]
    }
}

impl Default for Emotion {
    fn default() -> Self {
        Emotion::Neutral
    }
}

/// 副语言标签（参考 PilotTTS 4 种副语言分类）
///
/// 副语言指非言语的声音特征，如笑、呼吸、哭、咳嗽等。
/// 可在 TTS 文本中通过 `[laugh]`、`[breath]` 等标签插入。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Paralanguage {
    /// 笑声
    Laugh,
    /// 呼吸声
    Breath,
    /// 哭声
    Cry,
    /// 咳嗽声
    Cough,
}

impl Paralanguage {
    /// 返回副语言标签的英文标识
    pub fn as_str(&self) -> &str {
        match self {
            Paralanguage::Laugh => "laugh",
            Paralanguage::Breath => "breath",
            Paralanguage::Cry => "cry",
            Paralanguage::Cough => "cough",
        }
    }

    /// 返回副语言的中文名称
    pub fn display_cn(&self) -> &'static str {
        match self {
            Paralanguage::Laugh => "笑声",
            Paralanguage::Breath => "呼吸",
            Paralanguage::Cry => "哭声",
            Paralanguage::Cough => "咳嗽",
        }
    }

    /// 从字符串解析副语言标签
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "laugh" | "笑" | "笑声" => Some(Paralanguage::Laugh),
            "breath" | "呼吸" => Some(Paralanguage::Breath),
            "cry" | "哭" | "哭声" => Some(Paralanguage::Cry),
            "cough" | "咳嗽" => Some(Paralanguage::Cough),
            _ => None,
        }
    }

    /// 列出所有副语言
    pub fn all() -> Vec<Paralanguage> {
        vec![
            Paralanguage::Laugh,
            Paralanguage::Breath,
            Paralanguage::Cry,
            Paralanguage::Cough,
        ]
    }

    /// 返回文本标签表示（用于在文本中插入）
    pub fn to_tag(&self) -> String {
        format!("[{}]", self.as_str())
    }
}

/// 情感合成参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionConfig {
    pub emotion: Emotion,
    pub intensity: f32,
    /// 副语言标签（可选）
    pub paralanguage: Option<Paralanguage>,
}

impl EmotionConfig {
    pub fn new(emotion: Emotion, intensity: f32) -> Self {
        Self {
            emotion,
            intensity,
            paralanguage: None,
        }
    }

    /// 创建含副语言的配置
    pub fn with_paralanguage(mut self, pl: Paralanguage) -> Self {
        self.paralanguage = Some(pl);
        self
    }
}

impl Default for EmotionConfig {
    fn default() -> Self {
        Self {
            emotion: Emotion::Neutral,
            intensity: 1.0,
            paralanguage: None,
        }
    }
}

/// 文本标签解析器
///
/// 从文本中提取 `[laugh]`、`[breath]`、`[emotion:happy]` 等标签，
/// 用于在 TTS 合成时控制情感和副语言表现。
///
/// # 标签格式
///
/// | 格式 | 说明 | 示例 |
/// |------|------|------|
/// | `[laugh]` | 副语言标签 | `[laugh]`, `[breath]`, `[cry]`, `[cough]` |
/// | `[emotion:xxx]` | 情感标签 | `[emotion:happy]`, `[emotion:sad]` |
pub struct TagParser;

impl TagParser {
    /// 解析文本，分离标签和纯净文本（不使用正则，保持轻量）
    ///
    /// # 返回值
    /// - `clean_text`: 去除所有标签后的文本
    /// - `tags`: 提取到的标签列表（按出现顺序）
    pub fn parse(text: &str) -> ParsedResult {
        let mut clean = String::new();
        let mut tags = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            if chars[i] == '[' {
                // 查找匹配的 ]
                let mut end = None;
                for j in i + 1..len {
                    if chars[j] == ']' {
                        end = Some(j);
                        break;
                    }
                }

                if let Some(end_idx) = end {
                    let inner: String = chars[i + 1..end_idx].iter().collect();
                    let tag_len = end_idx - i + 1;

                    // [laugh], [breath], [cry], [cough]
                    if let Some(pl) = Paralanguage::from_str(&inner) {
                        tags.push(TagItem {
                            tag_type: TagType::Paralanguage(pl),
                            offset: clean.chars().count(),
                        });
                        i += tag_len;
                        continue;
                    }

                    // [emotion:happy]
                    if let Some(val) = inner.strip_prefix("emotion:") {
                        if let Some(emotion) = Emotion::from_str(val) {
                            tags.push(TagItem {
                                tag_type: TagType::Emotion(emotion),
                                offset: clean.chars().count(),
                            });
                            i += tag_len;
                            continue;
                        }
                    }

                    // 未知标签，保留原样
                    clean.push(chars[i]);
                    i += 1;
                } else {
                    // 没有匹配的 ]，保留 [
                    clean.push(chars[i]);
                    i += 1;
                }
            } else {
                clean.push(chars[i]);
                i += 1;
            }
        }

        ParsedResult { clean_text: clean, tags }
    }

    /// 移除文本中的所有标签
    pub fn strip_tags(text: &str) -> String {
        Self::parse(text).clean_text
    }

    /// 从文本中提取第一个情感标签
    pub fn extract_emotion(text: &str) -> Option<Emotion> {
        for tag in Self::parse(text).tags {
            if let TagType::Emotion(e) = tag.tag_type {
                return Some(e);
            }
        }
        None
    }

    /// 从文本中提取第一个副语言标签
    pub fn extract_paralanguage(text: &str) -> Option<Paralanguage> {
        for tag in Self::parse(text).tags {
            if let TagType::Paralanguage(p) = tag.tag_type {
                return Some(p);
            }
        }
        None
    }
}

/// 标签解析结果
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedResult {
    /// 去除标签后的纯净文本
    pub clean_text: String,
    /// 提取到的标签列表
    pub tags: Vec<TagItem>,
}

/// 解析出的标签条目
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagItem {
    pub tag_type: TagType,
    /// 标签在纯净文本中的字符偏移
    pub offset: usize,
}

/// 标签类型
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TagType {
    Emotion(Emotion),
    Paralanguage(Paralanguage),
}

/// TTS 合成参数
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TtsParams {
    pub engine: EngineKind,
    pub voice: VoiceId,
    pub speed: Speed,
    pub pitch: Pitch,
    pub volume: Volume,
    pub segment_size: SegmentSize,
    pub segment_silence_ms: u32,
    pub crossfade_ms: u32,
    pub num_to_chinese: bool,
    pub denoise: bool,
    pub denoise_level: DenoiseLevel,
    /// 情感合成配置
    #[serde(default)]
    pub emotion: Option<EmotionConfig>,
    /// 目标方言（None 表示使用默认）
    #[serde(default)]
    pub dialect: Option<super::dialect::Dialect>,
}

/// 音频输出
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioOutput {
    pub format: AudioFormat,
    pub path: std::path::PathBuf,
    pub duration_ms: u32,
    pub sample_rate: u32,
    pub bitrate_kbps: Option<u32>,
    pub sample_depth: Option<u16>,
}

/// TTS 合成阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TtsPhase {
    Idle,
    Preprocessing,
    Synthesizing,
    Concatenating,
    Exporting,
    Completed,
    Failed,
    Paused,
    Cancelled,
}

/// TTS 进度
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TtsProgress {
    pub current_segment: usize,
    pub total_segments: usize,
    pub phase: TtsPhase,
}

/// 段落状态
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentStatus {
    Pending,
    Synthesizing,
    Completed,
    Failed(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_tts_engine_四个CLI引擎串全部可解析() {
        // docs/20 F62：pipeline 曾只映射 kokoro/indextts2，
        // 这里锁定 CLI 帮助文本里对外承诺的 4 个引擎串。
        assert_eq!(parse_tts_engine("kokoro"), Some(EngineKind::Kokoro));
        assert_eq!(parse_tts_engine("indextts2"), Some(EngineKind::IndexTTS2));
        assert_eq!(parse_tts_engine("qwen3"), Some(EngineKind::Qwen3Tts));
        assert_eq!(parse_tts_engine("cosyvoice3"), Some(EngineKind::CosyVoice3));
    }

    #[test]
    fn parse_tts_engine_规范名与别名() {
        assert_eq!(parse_tts_engine("CosyVoice"), Some(EngineKind::CosyVoice3));
        assert_eq!(parse_tts_engine("COSYVOICE3"), Some(EngineKind::CosyVoice3));
        assert_eq!(parse_tts_engine("qwen3-tts"), Some(EngineKind::Qwen3Tts));
        assert_eq!(parse_tts_engine("indextts"), Some(EngineKind::IndexTTS2));
        assert_eq!(parse_tts_engine("azure-tts"), Some(EngineKind::AzureTts));
        assert_eq!(parse_tts_engine("aliyun-tts"), Some(EngineKind::AliyunTts));
    }

    #[test]
    fn parse_tts_engine_拒绝非TTS引擎() {
        // ASR/OCR/翻译/LLM 引擎不得被当作 TTS 引擎使用。
        assert_eq!(parse_tts_engine("whisper-base"), None);
        assert_eq!(parse_tts_engine("sensevoice"), None);
        assert_eq!(parse_tts_engine("paddleocr-v6-tiny"), None);
        assert_eq!(parse_tts_engine("opus-mt"), None);
        // `qwen` 是 LLM（TTS 场景下解析成它会静默用错引擎）
        assert_eq!(parse_tts_engine("qwen"), None);
        assert_eq!(parse_tts_engine("notexist"), None);
    }

    #[test]
    fn parse_tts_engine_与as_str往返一致() {
        for kind in [
            EngineKind::Kokoro,
            EngineKind::IndexTTS2,
            EngineKind::CosyVoice3,
            EngineKind::Qwen3Tts,
            EngineKind::AzureTts,
            EngineKind::AliyunTts,
        ] {
            assert_eq!(
                parse_tts_engine(kind.as_str()),
                Some(kind),
                "as_str() 产出的名称必须能被 parse_tts_engine 解析回自身"
            );
        }
    }
    use super::*;

    // ---- 原有测试 ----

    #[test]
    fn speed_范围校验() {
        assert!(Speed::new(1.0).is_some());
        assert!(Speed::new(0.5).is_some());
        assert!(Speed::new(2.0).is_some());
        assert!(Speed::new(0.4).is_none());
        assert!(Speed::new(2.1).is_none());
    }

    #[test]
    fn pitch_范围校验() {
        assert!(Pitch::new(0).is_some());
        assert!(Pitch::new(-20).is_some());
        assert!(Pitch::new(20).is_some());
        assert!(Pitch::new(-21).is_none());
        assert!(Pitch::new(21).is_none());
    }

    #[test]
    fn volume_范围校验() {
        assert!(Volume::new(0).is_some());
        assert!(Volume::new(100).is_some());
        assert!(Volume::new(101).is_none());
    }

    #[test]
    fn segment_size_字数映射() {
        assert_eq!(SegmentSize::S300.max_chars(), 300);
        assert_eq!(SegmentSize::S500.max_chars(), 500);
        assert_eq!(SegmentSize::S800.max_chars(), 800);
    }

    // ---- 情感扩展测试 ----

    #[test]
    fn emotion_列表长度() {
        assert_eq!(Emotion::all().len(), 13);
    }

    #[test]
    fn emotion_中文显示() {
        assert_eq!(Emotion::Happy.display_cn(), "高兴");
        assert_eq!(Emotion::Contempt.display_cn(), "轻蔑");
        assert_eq!(Emotion::Psychology.display_cn(), "悬疑");
    }

    #[test]
    fn emotion_从字符串解析() {
        assert_eq!(Emotion::from_str("happy"), Some(Emotion::Happy));
        assert_eq!(Emotion::from_str("恐惧"), Some(Emotion::Fearful));
        assert_eq!(Emotion::from_str("轻蔑"), Some(Emotion::Contempt));
        assert_eq!(Emotion::from_str("耳语"), Some(Emotion::Whisper));
        assert_eq!(Emotion::from_str("未知情感"), None);
    }

    // ---- 副语言测试 ----

    #[test]
    fn paralanguage_列表长度() {
        assert_eq!(Paralanguage::all().len(), 4);
    }

    #[test]
    fn paralanguage_标签格式() {
        assert_eq!(Paralanguage::Laugh.to_tag(), "[laugh]");
        assert_eq!(Paralanguage::Cry.to_tag(), "[cry]");
    }

    #[test]
    fn paralanguage_从标签解析() {
        assert_eq!(Paralanguage::from_str("laugh"), Some(Paralanguage::Laugh));
        assert_eq!(Paralanguage::from_str("breath"), Some(Paralanguage::Breath));
        assert_eq!(Paralanguage::from_str("咳嗽"), Some(Paralanguage::Cough));
    }

    // ---- TagParser 测试 ----

    #[test]
    fn tag_parser_提取副语言标签() {
        let result = TagParser::parse("哈哈[laugh]这真好笑");
        assert_eq!(result.clean_text, "哈哈这真好笑");
        assert_eq!(result.tags.len(), 1);
        assert!(matches!(result.tags[0].tag_type, TagType::Paralanguage(Paralanguage::Laugh)));
    }

    #[test]
    fn tag_parser_提取情感标签() {
        let result = TagParser::parse("他[emotion:angry]非常愤怒地说");
        assert_eq!(result.clean_text, "他非常愤怒地说");
        assert_eq!(result.tags.len(), 1);
        assert!(matches!(result.tags[0].tag_type, TagType::Emotion(Emotion::Angry)));
    }

    #[test]
    fn tag_parser_多个标签() {
        let result = TagParser::parse("[emotion:happy]你好[laugh]世界");
        assert_eq!(result.clean_text, "你好世界");
        assert_eq!(result.tags.len(), 2);
    }

    #[test]
    fn tag_parser_无标签文本不动() {
        let result = TagParser::parse("正常文本");
        assert_eq!(result.clean_text, "正常文本");
        assert!(result.tags.is_empty());
    }

    #[test]
    fn tag_parser_提取情感() {
        assert_eq!(TagParser::extract_emotion("[emotion:sad]测试"), Some(Emotion::Sad));
        assert_eq!(TagParser::extract_emotion("[laugh]测试"), None);
    }

    #[test]
    fn tag_parser_提取副语言() {
        assert_eq!(TagParser::extract_paralanguage("[cry]测试"), Some(Paralanguage::Cry));
        assert_eq!(TagParser::extract_paralanguage("测试"), None);
    }

    #[test]
    fn tag_parser_去除标签() {
        assert_eq!(TagParser::strip_tags("[laugh]哈哈"), "哈哈");
        assert_eq!(TagParser::strip_tags("无标签"), "无标签");
    }

    // ---- EmotionConfig 测试 ----

    #[test]
    fn emotion_config_默认中性() {
        let cfg = EmotionConfig::default();
        assert_eq!(cfg.emotion, Emotion::Neutral);
        assert_eq!(cfg.intensity, 1.0);
        assert!(cfg.paralanguage.is_none());
    }

    #[test]
    fn emotion_config_含副语言() {
        let cfg = EmotionConfig::new(Emotion::Sad, 0.8)
            .with_paralanguage(Paralanguage::Cry);
        assert_eq!(cfg.emotion, Emotion::Sad);
        assert_eq!(cfg.intensity, 0.8);
        assert_eq!(cfg.paralanguage, Some(Paralanguage::Cry));
    }

    // ---- TtsParams 默认测试 ----

    #[test]
    fn tts_params_默认无情感方言() {
        let params = TtsParams {
            engine: crate::model::value_object::EngineKind::Kokoro,
            voice: VoiceId::new("test", "测试", crate::model::value_object::EngineKind::Kokoro),
            speed: Speed::default(),
            pitch: Pitch::default(),
            volume: Volume::default(),
            segment_size: SegmentSize::default(),
            segment_silence_ms: 300,
            crossfade_ms: 50,
            num_to_chinese: true,
            denoise: false,
            denoise_level: DenoiseLevel::Low,
            emotion: None,
            dialect: None,
        };
        assert!(params.emotion.is_none());
        assert!(params.dialect.is_none());
    }
}
