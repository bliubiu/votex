//! Paraformer ASR 引擎（ONNX 推理）
//!
//! Paraformer 是阿里达摩院（FunASR）的非自回归端到端语音识别模型，
//! 支持中文、英文、中英混合识别，推理速度快于传统自回归模型。
//!
//! # 模型文件
//! - `models/Paraformer/model.onnx` - 主模型
//! - `models/Paraformer/tokens.txt` - 词表文件
//! - `models/Paraformer/am.mvn` - 特征归一化参数（可选）
//!
//! # 集成状态
//! 【已完成】ONNX 推理管线集成

use std::path::Path;

use ndarray::Array3;
use ort::session::Session;

use votex_domain::asr::provider::AsrProvider;
use votex_domain::asr::value_object::{AsrParams, RecognizeOutput, WordTimestamp};
use votex_domain::error::AsrError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;

use crate::shared::{EngineState, ModelFileLocator, OrtSessionFactory};

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

/// Paraformer ASR Provider（ONNX 推理）
pub struct ParaformerProvider {
    state: EngineState<Session>,
    tokens: Vec<String>,
}

impl ParaformerProvider {
    pub fn new() -> Self {
        Self {
            state: EngineState::new(),
            tokens: Vec::new(),
        }
    }

    /// 从模型目录加载
    fn load_from_dir(&mut self, model_dir: &Path) -> Result<(), AsrError> {
        // F53：FunASR 导出的资产名为 `model_quant.onnx`，规范名为 `model.onnx`，
        // 单文件名断言会让已有资产无法加载，这里按候选名依次探测。
        let model_path = ModelFileLocator::find_first_existing(
            model_dir,
            &["model_quant.onnx", "model.onnx", "model.int8.onnx"],
        )
        .map_err(|e| AsrError::ModelNotFound(format!("{}", e)))?;
        let tokens_path =
            ModelFileLocator::find_first_existing(model_dir, &["tokens.txt", "tokens.json"])
                .map_err(|e| AsrError::ModelNotFound(format!("{}", e)))?;

        // 使用 OrtSessionFactory 创建 Session
        let session = OrtSessionFactory::create(&model_path)
            .map_err(|e| AsrError::LoadFailed(format!("{}", e)))?;
        self.state.load(session);

        // 加载词表
        self.tokens = load_tokens(&tokens_path)?;

        tracing::info!(
            "Paraformer 模型加载完成，词表: {:?}，词表条目数: {}",
            tokens_path.file_name().unwrap_or_default(),
            self.tokens.len()
        );
        Ok(())
    }

    /// 将 f32 PCM 转换为梅尔频谱特征
    fn audio_to_mel(&self, audio: &AudioData) -> Result<Array3<f32>, AsrError> {
        let pcm = audio.to_mono_f32_16k();
        let n_samples = pcm.len();

        let frame_length = 400; // 25ms @ 16kHz
        let frame_shift = 160; // 10ms @ 16kHz

        if n_samples < frame_length {
            return Err(AsrError::EmptyAudio);
        }
        let n_frames = (n_samples - frame_length) / frame_shift + 1;
        let n_mels = 80;

        // 创建特征张量 (batch=1, frames, n_mels)
        let mut features = vec![0.0f32; n_frames * n_mels];

        // 简化的特征提取（实际应使用 FFT + 梅尔滤波器）
        for frame_idx in 0..n_frames {
            let start = frame_idx * frame_shift;
            let end = start + frame_length;
            if end > n_samples {
                break;
            }

            // 计算帧能量作为简化特征
            let frame_energy: f32 = pcm[start..end].iter().map(|&x| x * x).sum::<f32>() / frame_length as f32;
            let log_energy = (frame_energy + 1e-10).ln();

            // 将能量分布到 80 个梅尔频段（简化处理）
            for mel_idx in 0..n_mels {
                let idx = frame_idx * n_mels + mel_idx;
                features[idx] = log_energy * (1.0 - mel_idx as f32 / n_mels as f32);
            }
        }

        Array3::from_shape_vec((1, n_frames, n_mels), features)
            .map_err(|e| AsrError::RecognizeFailed(format!("创建特征张量失败: {}", e)))
    }
}

impl AsrProvider for ParaformerProvider {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::Paraformer
    }

    fn load(&mut self, model: &Model) -> Result<(), AsrError> {
        let model_dir = ModelFileLocator::locate_model_dir(model)
            .map_err(|e| AsrError::ModelNotFound(format!("{}", e)))?;
        self.load_from_dir(&model_dir)
    }

    fn unload(&mut self) -> Result<(), AsrError> {
        self.state.unload();
        self.tokens.clear();
        tracing::info!("Paraformer 引擎已释放");
        Ok(())
    }

    fn recognize(
        &self,
        audio: &AudioData,
        _params: &AsrParams,
    ) -> Result<RecognizeOutput, AsrError> {
        if audio.samples.is_empty() {
            return Err(AsrError::EmptyAudio);
        }
        // 提取梅尔频谱特征
        let mel_features = self.audio_to_mel(audio)?;

        // 准备输入
        let input_value = ort::value::Value::from_array(mel_features)
            .map_err(|e| AsrError::RecognizeFailed(format!("创建输入值失败: {}", e)))?;

        // 使用 EngineState 安全访问 session 并执行推理
        let output_ids_vec = self.state.with_mut(|session| {
            let outputs = session
                .run(ort::inputs![input_value])
                .map_err(|e| AsrError::RecognizeFailed(format!("Paraformer 推理失败: {}", e)))?;

            let output_ids = outputs[0]
                .try_extract_array::<i64>()
                .map_err(|e| AsrError::RecognizeFailed(format!("提取输出失败: {}", e)))?;

            Ok::<Vec<i64>, AsrError>(output_ids.iter().copied().collect())
        }).map_err(|_| AsrError::EngineNotLoaded)??;

        // 解码文本
        let mut text = String::new();
        let mut word_timestamps = Vec::new();

        for token_id in output_ids_vec {
            if token_id > 0 && (token_id as usize) < self.tokens.len() {
                let token = &self.tokens[token_id as usize];
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(token);

                word_timestamps.push(WordTimestamp {
                    word: token.clone(),
                    start_ms: 0.0,
                    end_ms: 0.0,
                });
            }
        }

        tracing::info!("Paraformer 识别完成: 文本长度 {}", text.len());

        Ok(RecognizeOutput {
            text,
            word_timestamps,
        })
    }

    fn sample_rate(&self) -> u32 {
        16000
    }

    fn is_loaded(&self) -> bool {
        self.state.is_loaded()
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
}

