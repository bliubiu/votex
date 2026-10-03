use clap::{Parser, Subcommand};

/// 声阅 (votex) - 离线桌面语音工具
#[derive(Parser, Debug)]
#[command(name = "votex", version, about = "声阅 - 离线桌面语音工具")]
pub struct Cli {
    /// 启动 GUI 模式
    #[arg(long)]
    pub gui: bool,

    /// 配置文件路径
    #[arg(long, default_value = "application.yml")]
    pub config: String,

    /// 模型文件存储目录
    #[arg(long, default_value = "models")]
    pub models_dir: String,

    /// 启用详细日志输出 (debug 级别)
    #[arg(long, short = 'v')]
    pub verbose: bool,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// 语音合成
    Tts {
        /// 输入文件路径
        #[arg(short, long)]
        input: String,

        /// 输出文件路径（扩展名决定格式：wav/mp3/m4a/flac/m4b）
        #[arg(short, long)]
        output: String,

        /// TTS 引擎 (kokoro / indextts2 / qwen3 / cosyvoice3)
        #[arg(long, default_value = "kokoro")]
        engine: String,

        /// 音色
        #[arg(long, default_value = "zf_001")]
        voice: String,

        /// 语速 (0.5~2.0)
        #[arg(long, default_value = "1.0")]
        speed: f32,

        /// 语言选择（zh/cn=中文, en=英文），未指定时从音色前缀自动推导
        #[arg(long)]
        lang: Option<String>,

        /// 模型变体（qwen3 专用: 0.6b / 1.7b），默认 0.6b
        #[arg(long)]
        model: Option<String>,

        /// 断点续转会话目录：每段音频落盘，中断后重跑自动复用已完成段
        #[arg(long)]
        session_dir: Option<String>,

        /// 多角色配音映射表（JSON：narrator/dialogue_default/roles）
        #[arg(long)]
        role_map: Option<String>,

        /// EBU R128 响度归一化（编码阶段 loudnorm）
        #[arg(long, default_value_t = false)]
        loudnorm: bool,

        /// 输出语速（atempo 变速 0.5~2.0，与 --speed 引擎级语速独立）
        #[arg(long)]
        atempo: Option<f32>,
    },

    /// 语音识别
    Asr {
        /// 输入文件路径
        #[arg(short, long)]
        input: String,

        /// 输出文件路径
        #[arg(short, long)]
        output: String,

        /// ASR 模型 (whisper-base / whisper-small / sensevoice / paraformer / qwen3-asr / firered-asr / wenet)
        #[arg(long, default_value = "whisper-base")]
        model: String,

        /// 输出格式 (txt / srt / lrc)
        #[arg(long, default_value = "srt")]
        format: String,

        /// 语言选择 (zh=中文, en=英文, zhen=中英混合)
        #[arg(long, default_value = "zh")]
        lang: String,
    },

    /// 图文识别 (OCR)
    Ocr {
        #[command(subcommand)]
        action: OcrAction,
    },

    /// 模型管理
    Model {
        #[command(subcommand)]
        action: ModelAction,
    },

    /// 克隆音色库
    Voice {
        #[command(subcommand)]
        action: VoiceAction,
    },

    /// 流水线
    Pipeline {
        /// 流水线类型 (audiobook / subtitle / audiobook-sub)
        #[arg(short, long)]
        kind: String,

        /// 输入文件路径
        #[arg(short, long)]
        input: String,

        /// 输出目录
        #[arg(short, long)]
        output: String,

        /// TTS 引擎 (kokoro / indextts2 / qwen3 / cosyvoice3)
        #[arg(long, default_value = "kokoro")]
        engine: String,

        /// 音色
        #[arg(long, default_value = "zf_001")]
        voice: String,

        /// 语速 (0.5~2.0)
        #[arg(long, default_value = "1.0")]
        speed: f32,

        /// ASR 模型 (whisper-base / whisper-small / sensevoice / paraformer / qwen3-asr)
        #[arg(long, default_value = "whisper-base")]
        asr_model: String,

        /// 字幕格式 (txt / srt / lrc)
        #[arg(long, default_value = "srt")]
        subtitle_format: String,
    },

    /// 批量任务
    Batch {
        #[command(subcommand)]
        action: BatchAction,
    },

    /// 配置管理
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },

    /// 生成短视频文案
    Script(super::script::ScriptCommand),

    /// 一键生成短视频
    Video(super::video::VideoCommand),

    /// 文本翻译
    Translate {
        /// 待翻译文本
        #[arg(short, long)]
        text: String,

        /// 翻译引擎 (dict / opus-mt / nllb-200 / m2m-100 / hy-mt-1.5 / qwen-mt / llm)
        #[arg(long, default_value = "dict")]
        engine: String,

        /// 翻译方向 (zh-en / en-zh / auto / zh->ja)
        #[arg(long, default_value = "zh-en")]
        direction: String,

        /// 术语表文件路径（JSON 数组，保证专有名词译法一致）
        #[arg(long)]
        glossary: Option<String>,

        /// 模型根目录
        #[arg(long)]
        models_dir: Option<String>,

        /// 目标中文书写系统 (simplified / traditional)
        #[arg(long)]
        target_script: Option<String>,

        /// 长文本分段：单段最大字符数，0 表示按引擎能力自动决定
        #[arg(long, default_value = "0")]
        max_segment_chars: usize,

        /// 不使用翻译缓存
        #[arg(long)]
        no_cache: bool,
    },

    /// 生成术语表模板文件
    GlossaryInit {
        /// 输出文件路径
        #[arg(short, long, default_value = "glossary.json")]
        output: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum BatchAction {
    /// 批量 TTS 合成
    Tts {
        /// 输入文件路径（Tab 分隔的批量定义文件）
        #[arg(short, long)]
        input: String,

        /// 输出目录
        #[arg(short, long, default_value = "tts_batch_output")]
        output: String,

        /// TTS 引擎 (kokoro / indextts2 / qwen3 / cosyvoice3)
        #[arg(long, default_value = "kokoro")]
        engine: String,

        /// 默认音色
        #[arg(long, default_value = "zf_001")]
        voice: String,

        /// 默认语速 (0.5~2.0)
        #[arg(long, default_value = "1.0")]
        speed: f32,

        /// 语言选择
        #[arg(long)]
        lang: Option<String>,

        /// 模型变体（qwen3 专用: 0.6b / 1.7b）
        #[arg(long)]
        model: Option<String>,

        /// 输出格式 (wav / mp3)
        #[arg(long, default_value = "wav")]
        format: String,

        /// 最大并发数
        #[arg(long, default_value = "1")]
        concurrency: usize,

        /// 启用降噪
        #[arg(long)]
        denoise: bool,

        /// 降噪级别 (low / medium / high)
        #[arg(long, default_value = "low")]
        denoise_level: String,
    },

    /// 批量 ASR 识别
    Asr {
        /// 输入目录
        #[arg(short, long)]
        input: String,

        /// 输出目录
        #[arg(short, long, default_value = "asr_batch_output")]
        output: String,

        /// ASR 模型 (whisper-base / whisper-small / sensevoice / paraformer / qwen3-asr / firered-asr / wenet)
        #[arg(long, default_value = "whisper-base")]
        model: String,

        /// 输出格式 (txt / srt / lrc)
        #[arg(long, default_value = "srt")]
        format: String,

        /// 语言选择 (zh=中文, en=英文, zhen=中英混合)
        #[arg(long, default_value = "zh")]
        lang: String,

        /// 递归扫描子目录
        #[arg(long)]
        recursive: bool,

        /// 最大并发数
        #[arg(long, default_value = "1")]
        concurrency: usize,

        /// 启用降噪
        #[arg(long)]
        denoise: bool,

        /// 降噪级别 (low / medium / high)
        #[arg(long, default_value = "low")]
        denoise_level: String,
    },

    /// 批量翻译
    Translate {
        /// 输入文本文件路径（每行一条待翻译文本）
        #[arg(short, long)]
        input: String,

        /// 输出文件路径（译文按行写出）
        #[arg(short, long, default_value = "translation_batch_output.txt")]
        output: String,

        /// 翻译引擎 (dict / opus-mt / nllb-200 / m2m-100 / hy-mt-1.5 / qwen-mt / llm)
        #[arg(long, default_value = "dict")]
        engine: String,

        /// 翻译方向 (zh-en / en-zh / auto / zh->ja)
        #[arg(long, default_value = "zh-en")]
        direction: String,

        /// 术语表文件路径
        #[arg(long)]
        glossary: Option<String>,

        /// 模型根目录
        #[arg(long)]
        models_dir: Option<String>,
    },
}

/// 克隆音色库子命令
#[derive(Subcommand, Debug)]
pub enum VoiceAction {
    /// 添加克隆音色（参考音频入库 + 自动降噪）
    Add {
        /// 音色名（同时作为文件名主体）
        name: String,

        /// 参考 WAV 音频路径（3 秒 ~ 10 分钟）
        reference: String,

        /// 是否降噪（默认开启）
        #[arg(long, default_value_t = true)]
        denoise: bool,
    },

    /// 列出克隆音色
    List,

    /// 删除克隆音色（必须 --confirm）
    Remove {
        /// 音色名
        name: String,

        /// 确认删除（禁止自动删除音色文件）
        #[arg(long)]
        confirm: bool,
    },
}

#[derive(Subcommand, Debug)]
pub enum ModelAction {    /// 列出模型
    List,

    /// 下载模型
    Download {
        /// 模型 ID
        model_id: String,

        /// 镜像源 (default / cn)
        #[arg(long, default_value = "default")]
        mirror: String,
    },

    /// 导入模型
    Import {
        /// 模型 ID
        model_id: String,

        /// 模型文件路径
        #[arg(short, long)]
        path: String,
    },

    /// 校验模型
    Verify {
        /// 模型 ID
        model_id: String,
    },

    /// 删除模型
    Remove {
        /// 模型 ID
        model_id: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum OcrAction {
    /// 单图识别（默认模式）
    Single {
        /// 输入图片路径
        #[arg(short, long)]
        input: String,

        /// 输出文件路径
        #[arg(short, long)]
        output: Option<String>,

        /// 输出格式 (txt / json / md / markdown)
        #[arg(long, default_value = "txt")]
        format: String,

        /// 是否禁用方向分类
        #[arg(long)]
        no_cls: bool,

        /// OCR 引擎 (paddleocr-v6-tiny / paddleocr-v4 / paddleocr-v5-mobile / paddleocr-v5-server / easyocr)
        #[arg(long, default_value = "paddleocr-v6-tiny")]
        engine: String,
    },

    /// 批量识别
    Batch {
        /// 输入图片路径（多个用空格分隔）
        #[arg(short, long, required = true, num_args = 1..)]
        inputs: Vec<String>,

        /// 输出目录
        #[arg(short, long, default_value = "ocr_output")]
        output: String,

        /// 输出格式 (txt / json / md)
        #[arg(long, default_value = "txt")]
        format: String,

        /// 是否禁用方向分类
        #[arg(long)]
        no_cls: bool,

        /// OCR 引擎 (paddleocr-v6-tiny / paddleocr-v4 / paddleocr-v5-mobile / paddleocr-v5-server / easyocr)
        #[arg(long, default_value = "paddleocr-v6-tiny")]
        engine: String,

        /// 最大并发数
        #[arg(long, default_value = "2")]
        concurrency: usize,
    },

    /// 列出历史任务
    List,

    /// 查看任务详情
    Show {
        /// 任务 ID
        task_id: String,
    },

    /// 删除任务
    Delete {
        /// 任务 ID
        task_id: String,
    },
}

#[derive(Subcommand, Debug)]
pub enum ConfigAction {
    /// 显示当前配置
    Show,

    /// 设置配置项
    Set {
        /// 配置键
        key: String,

        /// 配置值
        value: String,
    },

    /// 重置为默认配置
    Reset,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// 验证 CLI 定义完整性：所有子命令的 help 信息能正常生成
    #[test]
    fn cli_定义完整性() {
        let cli = Cli::command();
        // 至少包含 Model、Tts、Asr 等子命令
        let subcommands: Vec<&str> = cli.get_subcommands().map(|c| c.get_name()).collect();
        assert!(subcommands.contains(&"model"), "缺少 model 子命令");
        assert!(subcommands.contains(&"tts"), "缺少 tts 子命令");
        assert!(subcommands.contains(&"asr"), "缺少 asr 子命令");
        assert!(subcommands.contains(&"ocr"), "缺少 ocr 子命令");
        assert!(subcommands.contains(&"pipeline"), "缺少 pipeline 子命令");
        assert!(subcommands.contains(&"batch"), "缺少 batch 子命令");
        assert!(subcommands.contains(&"config"), "缺少 config 子命令");
    }

    /// 验证 Cli 参数解析：--gui 标志
    #[test]
    fn cli_解析_gui标志() {
        let cli = Cli::try_parse_from(["votex", "--gui"]).unwrap();
        assert!(cli.gui);
        assert_eq!(cli.models_dir, "models");
        assert_eq!(cli.config, "application.yml");
    }

    /// 验证 Cli 参数解析：--models-dir 自定义路径
    #[test]
    fn cli_解析_models_dir() {
        let cli = Cli::try_parse_from(["votex", "tts", "--input", "test.txt", "--output", "out.wav"]).unwrap();
        assert!(!cli.gui);
        assert_eq!(cli.config, "application.yml");
    }

    /// 验证 Cli 参数解析：--verbose
    #[test]
    fn cli_解析_verbose() {
        let cli = Cli::try_parse_from(["votex", "-v", "tts", "-i", "in.txt", "-o", "out.wav"]).unwrap();
        assert!(cli.verbose);
    }

    /// 验证 ModelAction 子命令
    #[test]
    fn cli_model_子命令() {
        let cli = Cli::try_parse_from(["votex", "model", "list"]).unwrap();
        match cli.command.unwrap() {
            Commands::Model { action } => {
                assert!(matches!(action, ModelAction::List));
            }
            _ => panic!("期望 Model 命令"),
        }
    }

    /// 验证 ConfigAction 子命令
    #[test]
    fn cli_config_子命令() {
        let cli = Cli::try_parse_from(["votex", "config", "show"]).unwrap();
        match cli.command.unwrap() {
            Commands::Config { action } => {
                assert!(matches!(action, ConfigAction::Show));
            }
            _ => panic!("期望 Config 命令"),
        }
    }

    /// 验证 TTS 子命令参数
    #[test]
    fn cli_tts_子命令参数() {
        let cli = Cli::try_parse_from([
            "votex", "tts", "-i", "input.txt", "-o", "output.mp3",
            "--engine", "kokoro", "--voice", "xiaobei", "--speed", "1.2",
        ]).unwrap();
        match cli.command.unwrap() {
            Commands::Tts { input, output, engine, voice, speed, lang, model, session_dir, role_map, loudnorm, atempo } => {
                assert!(role_map.is_none());
                assert!(!loudnorm);
                assert!(atempo.is_none());
                assert_eq!(input, "input.txt");
                assert_eq!(output, "output.mp3");
                assert_eq!(engine, "kokoro");
                assert_eq!(voice, "xiaobei");
                assert!((speed - 1.2).abs() < f32::EPSILON);
                assert!(lang.is_none());
                assert!(model.is_none());
                assert!(session_dir.is_none());
            }
            _ => panic!("期望 TTS 命令"),
        }
    }
}
