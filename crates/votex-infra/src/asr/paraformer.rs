//! Paraformer ASR 引擎（sherpa-onnx 绑定）
//!
//! Paraformer 是阿里达摩院（FunASR）的非自回归端到端语音识别模型，
//! 支持中文、英文、中英混合识别，推理速度快于传统自回归模型。
//!
//! # 实现说明
//!
//! 通过 sherpa-onnx Rust 绑定推理（与 `firered_asr.rs` / `qwen3_asr.rs` 同范式）。
//! 特征提取（kaldi fbank 80 维 + LFR 7×80=560 维堆叠 + CMVN 归一）由
//! sherpa-onnx 运行时完成，本项目不再自行实现特征前端。
//!
//! ⚠️ 历史教训：此前的"自行实现"只把帧能量乘线性系数填满 80 维——
//! 不是 mel 特征，识别输出是无效数据，但注释却标注「已完成」。
//! 模型输入分布必须与训练分布一致，没有金标对拍前不要手写特征前端。
//!
//! # 模型文件（models/asr/paraformer/）
//! - `model_quant.onnx` - 主模型（FunASR 导出，CMVN 已内嵌图中）
//! - `tokens.json` - 词表（FunASR 导出为 JSON 数组；sherpa-onnx 只认
//!   `token id` 格式的 tokens.txt，加载时派生到系统临时目录）
//! - `config.yaml` - 前端参数（fs=16k, hamming, 80 mel, LFR m=7 n=6）

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use sherpa_onnx::{
    OfflineParaformerModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
};
use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, RecognizeOutput, WordTimestamp};
use votex_domain::error::AsrError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;

/// 读取 Paraformer 词表，兼容两种磁盘格式（F53）
///
/// - `tokens.txt`：每行一个 token（sherpa-onnx / 规范导出）
/// - `tokens.json`：`["<blank>", "<s>", ...]` 的 JSON 字符串数组（FunASR 导出）
///
/// 两种格式语义等价（索引即 token id）。JSON 走 `serde_json` 解析，
/// 解析失败再退化为"逐行剥引号"，以兼容缩进换行与紧凑单行两种写法。
fn load_tokens(path: &Path) -> Result<Vec<String>, AsrError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| AsrError::LoadFailed(format!("读取词表文件失败: {}", e)))?;

    let tokens = if content.trim_start().starts_with('[') {
        match serde_json::from_str::<Vec<String>>(&content) {
            Ok(v) => v,
            Err(json_err) => strip_json_array_lines(&content).ok_or_else(|| {
                AsrError::LoadFailed(format!(
                    "词表 JSON 解析失败且无法按行降级解析: {:?}（原始错误: {}）",
                    path, json_err
                ))
            })?,
        }
    } else {
        content
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect()
    };

    if tokens.is_empty() {
        return Err(AsrError::LoadFailed(format!(
            "词表为空或格式无法解析: {:?}",
            path
        )));
    }
    Ok(tokens)
}

/// 逐行剥壳解析 JSON 字符串数组（serde_json 失败时的降级路径）
fn strip_json_array_lines(content: &str) -> Option<Vec<String>> {
    let tokens: Vec<String> = content
        .lines()
        .filter_map(|line| {
            let mut t = line.trim();
            if t.is_empty() || t == "[" || t == "]" {
                return None;
            }
            t = t.strip_suffix(',').unwrap_or(t).trim();
            if t.len() >= 2 && t.starts_with('"') && t.ends_with('"') {
                t = &t[1..t.len() - 1];
            }
            if t.is_empty() {
                None
            } else {
                Some(t.to_string())
            }
        })
        .collect();
    if tokens.is_empty() {
        None
    } else {
        Some(tokens)
    }
}

/// Paraformer ASR Provider（sherpa-onnx 分图 ONNX 推理）
pub struct ParaformerProvider {
    recognizer: Mutex<Option<OfflineRecognizer>>,
    sample_rate: u32,
}

impl ParaformerProvider {
    pub fn new() -> Self {
        Self {
            recognizer: Mutex::new(None),
            sample_rate: 16000,
        }
    }

    /// 从模型目录加载
    fn load_from_dir(&self, model_dir: &Path) -> Result<(), AsrError> {
        // ⚠️ 进程安全护栏：sherpa-onnx 的 Paraformer 加载器要求模型 ONNX
        // 元数据含 `vocab_size`（官方转换包有，FunASR 原始导出没有），
        // 元数据缺失时 sherpa C++ 端直接 abort——**整个进程崩溃**，
        // 无法用 Result 捕获。因此必须在调用 sherpa 前拒绝已知不兼容的
        // FunASR 导出（model_quant.onnx，来自 modelscope iic/...-onnx 仓库）。
        // 需要换用 sherpa-onnx 官方发布包（sherpa-onnx-paraformer-zh-*，
        // 含 model.int8.onnx + tokens.txt），届时同步更新 registry 源。
        let model_path = ModelFileLocatorHelper::find_first_existing(
            model_dir,
            &["model.int8.onnx", "model.onnx"],
        );
        let model_path = match model_path {
            Ok(p) => p,
            Err(_) if model_dir.join("model_quant.onnx").exists() => {
                return Err(AsrError::LoadFailed(
                    "当前 Paraformer 模型为 FunASR 原始导出（model_quant.onnx），\
                     缺少 sherpa-onnx 所需的 vocab_size 元数据，直接加载会崩溃进程，已拒绝。\
                     请改用 sherpa-onnx 官方转换包（sherpa-onnx-paraformer-zh-*，\
                     含 model.int8.onnx 与 tokens.txt）"
                        .to_string(),
                ));
            }
            Err(e) => return Err(AsrError::ModelNotFound(format!("{}", e))),
        };
        let tokens_path =
            ModelFileLocatorHelper::find_first_existing(model_dir, &["tokens.txt", "tokens.json"])
                .map_err(|e| AsrError::ModelNotFound(format!("{}", e)))?;

        // sherpa-onnx 只认 `token id` 行格式的 tokens.txt；
        // FunASR 导出是 JSON 数组 → 解析后派生一份到系统临时目录（创建识别器后即删）
        let tokens_for_sherpa = if tokens_path.extension().and_then(|e| e.to_str()) == Some("json") {
            Some(Self::derive_tokens_txt(&tokens_path)?)
        } else {
            None
        };
        let tokens_ref_path = tokens_for_sherpa
            .as_deref()
            .unwrap_or(tokens_path.as_path());

        let mut config = OfflineRecognizerConfig::default();
        config.model_config.paraformer = OfflineParaformerModelConfig {
            model: Some(model_path.to_string_lossy().to_string()),
        };
        config.model_config.tokens = Some(tokens_ref_path.to_string_lossy().to_string());
        config.model_config.num_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);

        let recognizer = OfflineRecognizer::create(&config)
            .ok_or_else(|| AsrError::LoadFailed("创建 Paraformer 识别器失败".to_string()))?;

        // tokens.txt 派生文件已读完，立即清理
        if let Some(p) = tokens_for_sherpa {
            let _ = std::fs::remove_file(&p);
        }

        *self.recognizer.lock().unwrap_or_else(|e| e.into_inner()) = Some(recognizer);

        tracing::info!(
            "Paraformer ONNX 引擎加载完成 (模型: {:?}, 词表: {:?})",
            model_path.file_name().unwrap_or_default(),
            tokens_path.file_name().unwrap_or_default()
        );
        Ok(())
    }

    /// 从 FunASR tokens.json 派生 sherpa-onnx 行格式的 tokens.txt
    ///
    /// 写入系统临时目录（模型目录可能是只读的），返回文件路径；
    /// 调用方在识别器创建完成后负责删除。
    fn derive_tokens_txt(tokens_json: &Path) -> Result<PathBuf, AsrError> {
        let tokens = load_tokens(tokens_json)?;
        let path = std::env::temp_dir().join(format!(
            "votex_paraformer_tokens_{}.txt",
            std::process::id()
        ));
        let mut content = String::with_capacity(tokens.len() * 12);
        for (id, token) in tokens.iter().enumerate() {
            // token 内含空白会让 sherpa 词表错位，必须替换（词表中此类 token 不参与解码）
            let safe = token.replace([' ', '\t', '\r', '\n'], "_");
            content.push_str(&format!("{} {}\n", safe, id));
        }
        std::fs::write(&path, content)
            .map_err(|e| AsrError::LoadFailed(format!("写入派生词表失败: {}", e)))?;
        Ok(path)
    }
}

/// 模型文件定位（局部辅助，避免与其他引擎的实现耦合）
struct ModelFileLocatorHelper;

impl ModelFileLocatorHelper {
    fn find_first_existing(dir: &Path, names: &[&str]) -> Result<PathBuf, anyhow::Error> {
        for name in names {
            let p = dir.join(name);
            if p.exists() {
                return Ok(p);
            }
        }
        Err(anyhow::anyhow!(
            "{:?} 中未找到以下任一文件: {}",
            dir,
            names.join(" / ")
        ))
    }
}

impl AsrProvider for ParaformerProvider {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::Paraformer
    }

    fn load(&self, model: &Model) -> Result<(), AsrError> {
        let model_dir = ModelFileLocatorHelper::find_first_existing_dir(model)
            .map_err(|e| AsrError::ModelNotFound(format!("{}", e)))?;
        self.load_from_dir(&model_dir)
    }

    fn unload(&self) -> Result<(), AsrError> {
        *self.recognizer.lock().unwrap_or_else(|e| e.into_inner()) = None;
        tracing::info!("Paraformer 引擎已释放");
        Ok(())
    }

    fn recognize(
        &self,
        audio: &AudioData,
        _params: &AsrParams,
    ) -> Result<RecognizeOutput, AsrError> {
        let guard = self.recognizer.lock().unwrap_or_else(|e| e.into_inner());
        let recognizer = guard.as_ref().ok_or(AsrError::EngineNotLoaded)?;

        // 转换为 16kHz 单声道 f32 PCM
        let pcm_data = audio.to_mono_f32_16k();
        if pcm_data.is_empty() {
            return Err(AsrError::EmptyAudio);
        }

        // 创建流并输入音频
        let stream = recognizer.create_stream();
        stream.accept_waveform(self.sample_rate as i32, &pcm_data);

        // 执行推理
        recognizer.decode(&stream);

        // 获取结果
        let result = stream
            .get_result()
            .ok_or_else(|| AsrError::RecognizeFailed("Paraformer 推理未返回结果".to_string()))?;

        tracing::info!("Paraformer 识别完成: 文本长度 {}", result.text.len());

        // 将整段文本作为一个时间戳项（sherpa Paraformer 无词级时间戳）
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
        self.sample_rate
    }

    fn is_loaded(&self) -> bool {
        self.recognizer
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }
}

impl ModelFileLocatorHelper {
    /// 定位 Paraformer 模型目录（models/asr/paraformer/ 等）
    fn find_first_existing_dir(model: &Model) -> Result<PathBuf, anyhow::Error> {
        for dir in [
            PathBuf::from("models").join("asr").join("paraformer"),
            PathBuf::from("models").join("paraformer"),
        ] {
            if dir.is_dir() {
                return Ok(dir);
            }
        }
        let by_name = PathBuf::from("models").join(&model.name);
        if by_name.is_dir() {
            return Ok(by_name);
        }
        let as_path = PathBuf::from(&model.name);
        if as_path.is_dir() {
            return Ok(as_path);
        }
        Err(anyhow::anyhow!(
            "未找到 Paraformer 模型目录，请将模型文件放在 models/asr/paraformer/ 目录下。\
             需要文件: model_quant.onnx, tokens.json"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn 词表_读取纯文本格式() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tokens.txt");
        fs::write(&path, "<blank>\n<s>\n</s>\nand@@\nprice\n").unwrap();

        let tokens = load_tokens(&path).unwrap();
        assert_eq!(tokens, vec!["<blank>", "<s>", "</s>", "and@@", "price"]);
    }

    #[test]
    fn 词表_读取json数组格式() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tokens.json");
        fs::write(
            &path,
            "[\n    \"<blank>\",\n    \"<s>\",\n    \"and@@\",\n    \"price\"\n]\n",
        )
        .unwrap();

        // 关键回归：不得把 `    "and@@",` 整行当成一个 token
        let tokens = load_tokens(&path).unwrap();
        assert_eq!(tokens, vec!["<blank>", "<s>", "and@@", "price"]);
    }

    #[test]
    fn 词表_两种格式产出完全一致() {
        let dir = tempdir().unwrap();
        let txt = dir.path().join("tokens.txt");
        let json = dir.path().join("tokens.json");
        fs::write(&txt, "a\nb\nc\n").unwrap();
        fs::write(&json, "[\"a\", \"b\", \"c\"]").unwrap();

        assert_eq!(load_tokens(&txt).unwrap(), load_tokens(&json).unwrap());
    }

    #[test]
    fn 词表_保留带引号的合法token() {
        let dir = tempdir().unwrap();
        // tokens.txt 中的 token 本身带引号时不得被剥壳（只有 JSON 数组才剥）
        let path = dir.path().join("tokens.txt");
        fs::write(&path, "\"quoted\"\nplain\n").unwrap();

        let tokens = load_tokens(&path).unwrap();
        assert_eq!(tokens, vec!["\"quoted\"", "plain"]);
    }

    #[test]
    fn 词表_空文件报错() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("tokens.txt");
        fs::write(&path, "").unwrap();
        assert!(load_tokens(&path).is_err(), "空词表必须报错而非静默通过");
    }

    #[test]
    fn 词表_json派生tokens_txt行格式() {
        let dir = tempdir().unwrap();
        let json = dir.path().join("tokens.json");
        fs::write(&json, "[\"<blank>\", \"你\", \"好\"]").unwrap();

        let derived = ParaformerProvider::derive_tokens_txt(&json).unwrap();
        let content = fs::read_to_string(&derived).unwrap();
        fs::remove_file(&derived).ok();

        assert_eq!(content, "<blank> 0\n你 1\n好 2\n");
    }
}
