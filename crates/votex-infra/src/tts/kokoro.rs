use std::collections::HashMap;
use std::path::{Path, PathBuf};

use ndarray::Array2;
use ort::session::Session;
use ort::value::Value;

use votex_domain::error::TtsError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;
use votex_domain::tts::provider::TtsProvider;
use votex_domain::tts::tokenizer::TextTokenizer;
use votex_domain::tts::value_object::{TtsParams, VoiceId};

use super::kokoro_g2p;
use crate::shared::{EngineState, ModelFileLocator, OrtSessionFactory};
use crate::tokenizer::CharLookupTokenizer;

/// Kokoro BERT 位置编码最大支持 512 tokens，预留安全余量
const MAX_SEQ_LEN: usize = 500;
/// 每个分片的最大内容 token 数（除去 BOS/EOS）
const CHUNK_CONTENT_LEN: usize = MAX_SEQ_LEN - 2;

// ===================== 词汇表（静态嵌入） =====================

/// 解析 config.json 中的 vocab 字段
fn parse_vocab_json(json_str: &str) -> HashMap<char, i64> {
    let cfg: serde_json::Value = serde_json::from_str(json_str)
        .expect("config.json 格式无效");
    let vocab = cfg["vocab"].as_object()
        .expect("config.json 缺少 vocab 字段");
    let mut map = HashMap::with_capacity(vocab.len());
    for (ch, id) in vocab {
        let chars: Vec<char> = ch.chars().collect();
        if chars.len() == 1 {
            map.insert(chars[0], id.as_i64().unwrap_or(0));
        }
    }
    map
}

/// 从模型目录加载词汇表 JSON（不再使用硬编码的静态全局变量）
fn load_vocab_for_model(base_dir: &Path) -> HashMap<char, i64> {
    let config_path = base_dir.join("config.json");
    match std::fs::read_to_string(&config_path) {
        Ok(content) => parse_vocab_json(&content),
        Err(e) => {
            tracing::warn!("加载词汇表失败 ({}): {}", config_path.display(), e);
            HashMap::new()
        }
    }
}

// ===================== 基础路径 =====================

/// 从模型的 file_paths 推导基础目录
///
/// 优先使用 file_paths 中的第一条路径推导基础目录；
/// 如果 file_paths 为空，则回退到 ModelFileLocator 的统一查找逻辑。
fn resolve_model_base_dir(model: &Model) -> Result<PathBuf, TtsError> {
    if let Some(first_path) = model.file_paths.first() {
        if let Some(parent) = first_path.parent() {
            return Ok(parent.to_path_buf());
        }
    }
    // 回退：使用 ModelFileLocator 按 models/<kind_dir>/<model_id>/ 查找
    ModelFileLocator::locate_model_dir(model)
        .map_err(|e| TtsError::SynthesisFailed(
            format!("无法定位模型目录: {}。请将模型文件放在 models/tts/ 目录下", e)
        ))
}

/// 测试用：从 manifest 路径推导工作区根
#[cfg(test)]
fn workspace_root_for_test() -> PathBuf {
    let mut dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.pop();
    dir.pop();
    dir
}

// ===================== G2P 策略 =====================

/// G2P（文字→音素）策略
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum G2pStrategy {
    /// v1.0 IPA 音素（英语/欧洲语言）
    Ipa,
    /// v1.1-zh 注音音素（中文 + 英文）
    Zhuyin,
}

impl G2pStrategy {
    fn convert(&self, text: &str) -> String {
        match self {
            G2pStrategy::Ipa => kokoro_g2p::text_to_phonemes(text),
            G2pStrategy::Zhuyin => kokoro_g2p::text_to_phonemes_zh(text),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            G2pStrategy::Ipa => "IPA",
            G2pStrategy::Zhuyin => "Zhuyin",
        }
    }
}

// ===================== 模型配置 =====================

/// 单个 Kokoro 模型的运行时配置（`load()` 时根据 `ModelId` 解析得到）
#[derive(Debug)]
struct KokoroModelConfig {
    base_dir: PathBuf,
    g2p: G2pStrategy,
    /// speed 输入是否为 int32（v1.0 为 float32）
    speed_is_int: bool,
    /// waveform 输出是否为 1D（v1.0 为 [1, T]）
    output_is_1d: bool,
    /// 分词器
    tokenizer: CharLookupTokenizer,
}

impl KokoroModelConfig {
    /// 创建 speed 输入张量
    /// 构造 speed 输入张量
    ///
    /// int32 导出（2026-08 之前的 onnx-community 版本）的 speed 语义是
    /// 「原始倍速取整」：1 = 正常语速（官方 kokoro-onnx 传 round(speed)）。
    /// 曾错误地按「十分之一编码」发送 speed×10 —— 1.0 倍速被模型当成
    /// 10 倍速，时长预测整体塌缩为 1/10，音频急促且音节残缺、内容不可辨。
    fn create_speed_value(&self, speed_val: f32) -> Result<Value, TtsError> {
        if self.speed_is_int {
            // 四舍五入到整数倍速（int 导出不支持小数倍速，官方同款处理）
            let speed_int = speed_val.round() as i64;
            let speed_int = speed_int.max(1).min(20);
            let arr = ndarray::arr1::<i32>(&[speed_int as i32]);
            let v: Value = Value::from_array(arr).map_err(|e| {
                TtsError::SynthesisFailed(format!("speed (int32) Value 创建失败: {}", e))
            })?.into();
            Ok(v)
        } else {
            let arr = ndarray::arr1::<f32>(&[speed_val]);
            let v: Value = Value::from_array(arr).map_err(|e| {
                TtsError::SynthesisFailed(format!("speed (float32) Value 创建失败: {}", e))
            })?.into();
            Ok(v)
        }
    }

    /// 从 ONNX 输出中提取音频样本
    fn extract_audio(&self, output_value: &Value) -> Result<(usize, Vec<f32>), TtsError> {
        let audio_output = output_value.try_extract_array::<f32>().map_err(|e| {
            TtsError::SynthesisFailed(format!("提取音频输出失败: {}", e))
        })?;

        let audio_shape = audio_output.shape();

        if self.output_is_1d {
            if audio_shape.len() != 1 {
                return Err(TtsError::SynthesisFailed(
                    format!("音频输出形状异常（期望 1D）: {:?}", audio_shape)
                ));
            }
            let num_samples = audio_shape[0];
            let samples: Vec<f32> = (0..num_samples).map(|i| audio_output[i]).collect();
            Ok((num_samples, samples))
        } else {
            if audio_shape.len() != 2 || audio_shape[0] != 1 {
                return Err(TtsError::SynthesisFailed(
                    format!("音频输出形状异常（期望 [1, T]）: {:?}", audio_shape)
                ));
            }
            let num_samples = audio_shape[1];
            let samples: Vec<f32> = (0..num_samples).map(|i| audio_output[[0, i]]).collect();
            Ok((num_samples, samples))
        }
    }
}

/// 根据 ModelId 解析模型配置
///
/// 基础目录从 `model.file_paths` 的第一个路径推导（必须已配置）。
/// 词汇表按模型目录下的 `config.json` 动态加载。
fn resolve_model_config(model: &Model) -> Result<KokoroModelConfig, TtsError> {
    let base_dir = resolve_model_base_dir(model)?;

    // 动态加载词汇表（路径由 model 配置驱动）
    let vocab = load_vocab_for_model(&base_dir);
    let tokenizer = CharLookupTokenizer::new(vocab, Some(0), Some(0));

    match model.id.as_str() {
        "kokoro-82m" => {
            Ok(KokoroModelConfig {
                base_dir,
                g2p: G2pStrategy::Ipa,
                speed_is_int: true,
                output_is_1d: true,
                tokenizer,
            })
        }
        "kokoro-82m-v1.1-zh" => {
            Ok(KokoroModelConfig {
                base_dir,
                g2p: G2pStrategy::Zhuyin,
                speed_is_int: true,
                output_is_1d: true,
                tokenizer,
            })
        }
        other => Err(TtsError::SynthesisFailed(
            format!("不支持的 Kokoro 模型: {}", other)
        )),
    }
}

// ===================== 音色数据加载 =====================

/// 从 .bin 文件加载音色向量（510 × 256 = 130560 float32）
fn load_voice_vector_from(base_dir: &PathBuf, voice_id: &str) -> Result<Vec<f32>, TtsError> {
    if voice_id.is_empty()
        || voice_id.contains('/')
        || voice_id.contains('\\')
        || voice_id.contains("..")
        || voice_id.contains('\0')
    {
        return Err(TtsError::SynthesisFailed(
            "非法的 voice_id: 包含路径分隔符或危险字符".into()
        ));
    }
    let voice_path = base_dir.join("voices").join(format!("{}.bin", voice_id));
    let data = std::fs::read(&voice_path)
        .map_err(|e| TtsError::SynthesisFailed(
            format!("读取音色文件失败 {:?}: {}", voice_path, e)
        ))?;
    if data.len() != 130560 * 4 {
        return Err(TtsError::SynthesisFailed(
            format!("音色文件大小不正确: 期望 {} 字节, 实际 {} 字节", 130560 * 4, data.len())
        ));
    }
    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(floats)
}

/// 根据 phonemes 长度选取样式向量（索引为 len-1，范围 0..510）
fn select_style_vector(voice_data: &[f32], phonemes_len: usize) -> [f32; 256] {
    let idx = (phonemes_len.saturating_sub(1)).min(509).max(0);
    let offset = idx * 256;
    let mut style = [0.0f32; 256];
    style.copy_from_slice(&voice_data[offset..offset + 256]);
    style
}

// ===================== KokoroProvider =====================

/// Kokoro-82M TTS 引擎（ONNX Runtime 推理）
///
/// 一个实例始终只加载一个模型版本，通过 `load()` 时传入的 `ModelId` 选择:
/// - `kokoro-82m` → v1.0（英语 IPA）
/// - `kokoro-82m-v1.1-zh` → v1.1-zh（中文注音 + 英文）
pub struct KokoroProvider {
    state: EngineState<Session>,
    config: std::sync::Mutex<Option<KokoroModelConfig>>,
    sample_rate: u32,
    voice_cache: std::sync::Mutex<HashMap<String, Vec<f32>>>,
}

impl KokoroProvider {
    pub fn new() -> Self {
        Self {
            state: EngineState::new(),
            config: std::sync::Mutex::new(None),
            sample_rate: 24000,
            voice_cache: std::sync::Mutex::new(HashMap::new()),
        }
    }
}

impl TtsProvider for KokoroProvider {
    fn engine_kind(&self) -> EngineKind {
        EngineKind::Kokoro
    }

    fn load(&mut self, model: &Model) -> Result<(), TtsError> {
        let config = resolve_model_config(model)?;

        // 查找 ONNX 模型文件：按模型版本分组探测，优先匹配当前模型的权重。
        // 此前 v1.0 int8 排在最前，若被误放入 v1.1-zh 目录会被静默选中，
        // 导致中文质量下降且难以察觉——现按 G2P 策略隔离两组探测列表。
        let model_paths: Vec<PathBuf> = match config.g2p {
            // v1.1-zh（中文注音）：只探测 v1.1-zh 变体，绝不回落到 v1.0 权重
            G2pStrategy::Zhuyin => vec![
                config.base_dir.join("kokoro-v1.1-zh.fixed.onnx"),
                config.base_dir.join("kokoro-v1.1-zh.onnx"),
                config.base_dir.join("kokoro-v1.1-zh.os18.onnx"),
                config.base_dir.join("kokoro.onnx"),
                config.base_dir.join("model.onnx"),
            ],
            // v1.0（英语 IPA）：int8 体积小、CPU 推理快，保持优先
            G2pStrategy::Ipa => vec![
                config.base_dir.join("kokoro-v1.0.int8.onnx"),
                config.base_dir.join("kokoro.onnx"),
                config.base_dir.join("kokoro-fp16.onnx"),
                config.base_dir.join("model.onnx"),
                config.base_dir.join("model_uint8.onnx"),
                config.base_dir.join("model_fp16.onnx"),
            ],
        };
        
        // 依次尝试加载 ONNX 模型文件（跳过无法解析的文件）
        let mut last_error: Option<anyhow::Error> = None;
        let mut loaded_model_path: Option<std::path::PathBuf> = None;
        
        for path in model_paths.iter().filter(|p| p.exists()) {
            match OrtSessionFactory::create_raw(path) {
                Ok(session) => {
                    self.state.load(session);
                    loaded_model_path = Some(path.clone());
                    break;
                }
                Err(e) => {
                    tracing::warn!("ONNX 模型加载失败 (跳过): {:?} - {}", path, e);
                    last_error = Some(e.into());
                }
            }
        }
        
        let model_path = loaded_model_path.ok_or_else(|| {
            let msg = last_error
                .as_ref()
                .map(|e| format!("{}", e))
                .unwrap_or_else(|| "未知错误".to_string());
            TtsError::SynthesisFailed(format!("所有 ONNX 模型均加载失败: {}", msg))
        })?;

        tracing::info!(
            "Kokoro ({}) 引擎加载完成 (模型: {:?})",
            config.g2p.name(), model_path
        );

        *self.config.lock().map_err(|e| {
            TtsError::SynthesisFailed(format!("config 锁失败: {}", e))
        })? = Some(config);

        Ok(())
    }

    fn unload(&mut self) -> Result<(), TtsError> {
        self.state.unload();
        *self.config.lock().unwrap_or_else(|e| e.into_inner()) = None;
        self.voice_cache.lock().unwrap_or_else(|e| e.into_inner()).clear();
        tracing::info!("Kokoro 引擎已释放");
        Ok(())
    }

    fn synthesize(
        &self,
        text: &str,
        voice: &VoiceId,
        params: &TtsParams,
    ) -> Result<AudioData, TtsError> {
        let config = self.config.lock().map_err(|e| {
            TtsError::SynthesisFailed(format!("config 锁失败: {}", e))
        })?;
        let cfg = config.as_ref().ok_or(TtsError::EngineNotLoaded)?;

        if text.is_empty() {
            return Err(TtsError::EmptyText);
        }

        // 1. G2P
        let phonemes = cfg.g2p.convert(text);
        tracing::debug!("G2P({}): '{}' → phonemes: '{}'", cfg.g2p.name(), text, phonemes);

        if phonemes.is_empty() {
            return Err(TtsError::SynthesisFailed("G2P 转换结果为空".to_string()));
        }

        // 2. 加载音色数据（内联以避免双重锁定 config）
        let voice_data = {
            let mut cache = self.voice_cache.lock().map_err(|e| {
                TtsError::SynthesisFailed(format!("音色缓存锁失败: {}", e))
            })?;
            if let Some(data) = cache.get(&voice.id) {
                data.clone()
            } else {
                let data = load_voice_vector_from(&cfg.base_dir, &voice.id)?;
                cache.insert(voice.id.clone(), data.clone());
                data
            }
        };

        // 3. 选择样式向量
        // 注意用字符数而非字节数：注音符号占 3 字节，字节数会使 style
        // 行索引偏大 ~2 倍，选到错误长度档位的韵律向量
        let style = select_style_vector(&voice_data, phonemes.chars().count());

        // 4. Tokenize（通过统一分词器接口）
        let token_ids = cfg.tokenizer.encode(&phonemes)
            .map_err(|e| TtsError::SynthesisFailed(format!("分词失败: {}", e)))?;
        let seq_len = token_ids.len();

        tracing::debug!(
            "推理参数: g2p={}, phonemes_len={}, seq_len={}, voice={}",
            cfg.g2p.name(), phonemes.len(), seq_len, voice.id
        );

        // 5. 构建 ONNX 输入张量并推理
        // 当序列超过 MAX_SEQ_LEN 时自动分片合成，避免 /bert/embeddings/Add 位置编码越界
        let style_value: Value = {
            let style_array = Array2::from_shape_vec((1, 256), style.to_vec())
                .map_err(|e| TtsError::SynthesisFailed(
                    format!("创建 style 张量失败: {}", e)
                ))?;
            Value::from_array(style_array).map_err(|e| {
                TtsError::SynthesisFailed(format!("style Value 创建失败: {}", e))
            })?.into()
        };
        let speed_value: Value = cfg.create_speed_value(params.speed.value())?.into();

        // 分片闭包：对单组 token_ids 执行推理
        let run_chunk = |chunk_ids: Vec<i64>, style: &Value, speed: &Value, cfg: &KokoroModelConfig| -> Result<Vec<f32>, TtsError> {
            let chunk_len = chunk_ids.len();
            let input_ids = Array2::from_shape_vec((1, chunk_len), chunk_ids)
                .map_err(|e| TtsError::SynthesisFailed(
                    format!("创建 input_ids 张量失败: {}", e)
                ))?;
            let input_ids_value = Value::from_array(input_ids).map_err(|e| {
                TtsError::SynthesisFailed(format!("input_ids Value 创建失败: {}", e))
            })?;

            // 使用 EngineState 执行推理
            self.state.with_mut(|session| {
                let outputs = session.run(ort::inputs![
                    input_ids_value,
                    style,
                    speed,
                ]).map_err(|e| TtsError::SynthesisFailed(
                    format!("ONNX 推理失败: {}", e)
                ))?;

                let (_, samples) = cfg.extract_audio(&outputs[0])?;
                Ok::<Vec<f32>, TtsError>(samples)
            }).map_err(|_| TtsError::EngineNotLoaded)?
        };

        let samples = if seq_len <= MAX_SEQ_LEN {
            // 快速路径：单段合成
            run_chunk(token_ids, &style_value, &speed_value, cfg)?
        } else {
            // 分片路径：按 CHUNK_CONTENT_LEN 切分内容 token，逐段合成后拼接
            let content = &token_ids[1..token_ids.len() - 1]; // 去掉 BOS/EOS
            let mut all_samples: Vec<f32> = Vec::new();
            let n_chunks = content.len().div_ceil(CHUNK_CONTENT_LEN);

            tracing::info!(
                "Kokoro 分段合成: seq_len={}, chunks={}, chunk_content_len={}",
                seq_len, n_chunks, CHUNK_CONTENT_LEN
            );

            for (i, chunk) in content.chunks(CHUNK_CONTENT_LEN).enumerate() {
                let mut chunk_ids = Vec::with_capacity(chunk.len() + 2);
                chunk_ids.push(0);  // BOS
                chunk_ids.extend_from_slice(chunk);
                chunk_ids.push(0);  // EOS

                tracing::debug!("  分片 {}: {} tokens", i + 1, chunk_ids.len());
                let samples = run_chunk(chunk_ids, &style_value, &speed_value, cfg)?;
                all_samples.extend(samples);
            }

            tracing::info!(
                "Kokoro 分段合成完成: {} 段 → {} 个采样点",
                n_chunks, all_samples.len()
            );

            all_samples
        };

        tracing::info!(
            "Kokoro ({}) 合成完成: text='{}', voice={}, samples={}",
            cfg.g2p.name(), text, voice.id, samples.len()
        );

        Ok(AudioData {
            samples,
            sample_rate: self.sample_rate,
            channels: 1,
        })
    }

    fn list_voices(&self) -> Vec<VoiceId> {
        let guard = match self.config.lock() {
            Ok(g) => g,
            _ => return vec![],
        };
        let cfg = match guard.as_ref() {
            Some(c) => c,
            None => return vec![],
        };

        let voices_dir = cfg.base_dir.join("voices");
        let mut voices = Vec::new();

        if let Ok(entries) = std::fs::read_dir(&voices_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("bin") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        let desc = if stem.starts_with("zf_") {
                            format!("中文女声{}", &stem[3..])
                        } else if stem.starts_with("zm_") {
                            format!("中文男声{}", &stem[3..])
                        } else if stem.starts_with("af_") {
                            format!("英文女声{}", &stem[3..])
                        } else if stem.starts_with("am_") {
                            format!("英文男声{}", &stem[3..])
                        } else if stem.starts_with("bf_") {
                            format!("英式女声{}", &stem[3..])
                        } else if stem.starts_with("bm_") {
                            format!("英式男声{}", &stem[3..])
                        } else {
                            stem.to_string()
                        };
                        voices.push(VoiceId::new(stem, &desc, EngineKind::Kokoro));
                    }
                }
            }
        }

        voices.sort_by(|a, b| a.id.cmp(&b.id));
        voices
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn supported_dialects(&self) -> Vec<votex_domain::tts::dialect::DialectSupport> {
        vec![
            votex_domain::tts::dialect::DialectSupport::new(
                votex_domain::tts::dialect::Dialect::Mandarin,
                votex_domain::tts::dialect::DialectQuality::Native,
            ),
        ]
    }

    fn is_loaded(&self) -> bool {
        self.state.is_loaded()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::LazyLock;

    // ============================================================
    // 测试用词汇表（从模型文件动态加载，缓存避免重复 I/O）
    // ============================================================
    static TEST_VOCAB: LazyLock<HashMap<char, i64>> = LazyLock::new(|| {
        let base = workspace_root_for_test().join("models/tts/kokoro-82m-v1.1-zh");
        load_vocab_for_model(&base)
    });

    fn test_vocab() -> HashMap<char, i64> {
        TEST_VOCAB.clone()
    }


    #[test]
    fn speed编码_int导出发送原始倍速() {
        // e2e 回归：int32 speed 的语义是原始倍速（1 = 正常语速），
        // 曾错误地按十分之一编码发送 speed*10 —— 1.0 倍速被模型当成
        // 10 倍速，时长预测塌缩为 1/10，音频急促、内容不可辨
        let cfg = KokoroModelConfig {
            base_dir: std::path::PathBuf::new(),
            g2p: G2pStrategy::Zhuyin,
            speed_is_int: true,
            output_is_1d: true,
            tokenizer: CharLookupTokenizer::new(HashMap::new(), Some(0), Some(0)),
        };
        let v = cfg.create_speed_value(1.0).unwrap();
        let (_, arr) = v.try_extract_tensor::<i32>().unwrap();
        assert_eq!(arr[0], 1, "1.0 倍速应发送 1 而非 10");

        let v = cfg.create_speed_value(1.5).unwrap();
        let (_, arr) = v.try_extract_tensor::<i32>().unwrap();
        assert_eq!(arr[0], 2, "1.5 倍速四舍五入为 2");

        // float 导出走原始值
        let cfg_float = KokoroModelConfig {
            speed_is_int: false,
            ..cfg
        };
        let v = cfg_float.create_speed_value(1.5).unwrap();
        let (_, arr) = v.try_extract_tensor::<f32>().unwrap();
        assert_eq!(arr[0], 1.5);
    }

    #[test]
    fn style选择_注音按字符数索引() {
        // style 行索引必须用字符数：注音占 3 字节，字节数会偏大 ~2 倍
        let voice = vec![0.5f32; 510 * 256];
        let ph = "ㄋㄧ2ㄏㄠ3"; // 6 字符 / 14 字节
        let s_char = select_style_vector(&voice, ph.chars().count());
        let s_byte = select_style_vector(&voice, ph.len());
        // 两路都应选到合法行且无 panic；字符路径选第 5 行，字节路径选第 13 行
        assert_eq!(s_char[0], 0.5);
        assert_eq!(s_byte[0], 0.5);
        // 字符计数语义：第 6 行起始偏移 = 5*256
        let expect = &voice[5 * 256..6 * 256];
        assert_eq!(s_char.as_slice(), expect);
    }

    #[test]
    fn 调试token序列() {
        let base = workspace_root_for_test().join("models/tts/kokoro-82m-v1.1-zh");
        let vocab = load_vocab_for_model(&base);
        let tokenizer = CharLookupTokenizer::new(vocab, Some(0), Some(0));
        for text in ["你好世界。", "张重躺在床上。"] {
            let ph = kokoro_g2p::text_to_phonemes_zh(text);
            let ids = tokenizer.encode(&ph).unwrap();
            println!("TEXT={} PH={} PH_LEN={}", text, ph, ph.chars().count());
            let chars: Vec<char> = ph.chars().collect();
            for (c, id) in chars.iter().zip(ids.iter()) {
                println!("  {:?} -> {}", c, id);
            }
            println!("  ids_len={} zeros={}", ids.len(), ids.iter().filter(|&&i| i == 0).count());
        }
    }


    #[test]
    fn test_vocab_loaded() {
        let vocab = test_vocab();
        assert!(!vocab.is_empty());
        assert!(vocab.contains_key(&'a'));
        assert!(vocab.contains_key(&'n'));
        assert!(vocab.contains_key(&'i'));
        assert!(vocab.contains_key(&'1'));
        assert!(vocab.contains_key(&'3'));
        assert!(vocab.contains_key(&'十'));
        assert!(vocab.contains_key(&'月'));
        assert_eq!(vocab.get(&';'), Some(&1));
    }

    #[test]
    fn test_tokenize_v10() {
        let tokenizer = CharLookupTokenizer::new(test_vocab(), Some(0), Some(0));
        let cfg = KokoroModelConfig {
            base_dir: PathBuf::new(),
            g2p: G2pStrategy::Ipa,
            speed_is_int: false,
            output_is_1d: false,
            tokenizer,
        };
        let ids = cfg.tokenizer.encode("a").unwrap();
        assert_eq!(ids.first(), Some(&0));  // BOS
        assert_eq!(ids.last(), Some(&0));   // EOS
        assert!(ids.len() >= 3);
    }

    #[test]
    fn test_vocab_v11_loaded() {
        let vocab = test_vocab();
        assert!(!vocab.is_empty());
        assert!(vocab.contains_key(&'ㄅ'));
        assert!(vocab.contains_key(&'ㄆ'));
        assert!(vocab.contains_key(&'ㄇ'));
        assert!(vocab.contains_key(&'1'));
        assert!(vocab.contains_key(&'3'));
        assert!(vocab.contains_key(&'十'));
        assert!(vocab.contains_key(&'月'));
        assert_eq!(vocab.get(&';'), Some(&1));
    }

    #[test]
    fn test_tokenize_v11() {
        let tokenizer = CharLookupTokenizer::new(test_vocab(), Some(0), Some(0));
        let cfg = KokoroModelConfig {
            base_dir: PathBuf::new(),
            g2p: G2pStrategy::Zhuyin,
            speed_is_int: true,
            output_is_1d: true,
            tokenizer,
        };
        let ids = cfg.tokenizer.encode("a").unwrap();
        assert_eq!(ids.first(), Some(&0));
        assert_eq!(ids.last(), Some(&0));
        assert!(ids.len() >= 3);
    }

    #[test]
    fn test_select_style_vector() {
        let mut voice = vec![0.0f32; 130560];
        voice[0] = 1.0;
        voice[256] = 2.0;
        voice[509 * 256] = 3.0;
        assert_eq!(select_style_vector(&voice, 1)[0], 1.0);
        assert_eq!(select_style_vector(&voice, 2)[0], 2.0);
        assert_eq!(select_style_vector(&voice, 510)[0], 3.0);
        assert_eq!(select_style_vector(&voice, 999)[0], 3.0);
    }

    #[test]
    fn test_load_voice_vector_file_not_found() {
        let base_dir = workspace_root_for_test().join("models/kokoro-82m");
        let result = load_voice_vector_from(&base_dir, "nonexistent_voice");
        assert!(result.is_err());
    }

    fn make_test_model(id: &str, name: &str, file_path: Option<&str>) -> Model {
        let mut model = Model::new(
            votex_domain::model::value_object::ModelId::new(id),
            name,
            votex_domain::model::value_object::ModelKind::Tts,
            EngineKind::Kokoro,
        );
        if let Some(path) = file_path {
            model.file_paths.push(PathBuf::from(path));
        }
        model
    }

    fn test_model_path_v10() -> &'static str {
        // 注意：此路径仅用于测试 resolve_model_base_dir 的路径推导逻辑
        // 实际的模型文件可以不存在——我们只验证路径计算正确
        Box::leak(
            workspace_root_for_test()
                .join("models/tts/kokoro-82m-v1.1-zh/kokoro-v1.0.onnx")
                .to_string_lossy()
                .to_string()
                .into_boxed_str()
        )
    }

    fn test_model_path_v11() -> &'static str {
        Box::leak(
            workspace_root_for_test()
                .join("models/tts/kokoro-82m-v1.1-zh/kokoro-v1.1-zh.onnx")
                .to_string_lossy()
                .to_string()
                .into_boxed_str()
        )
    }

    #[test]
    fn test_resolve_model_config_v10() {
        let model = make_test_model("kokoro-82m", "Kokoro-82M", Some(test_model_path_v10()));
        let cfg = resolve_model_config(&model).unwrap();
        assert_eq!(cfg.g2p, G2pStrategy::Ipa);
        assert!(cfg.speed_is_int);
        assert!(cfg.output_is_1d);
    }

    #[test]
    fn test_resolve_model_config_v11() {
        let model = make_test_model("kokoro-82m-v1.1-zh", "Kokoro-82M-v1.1-zh", Some(test_model_path_v11()));
        let cfg = resolve_model_config(&model).unwrap();
        assert_eq!(cfg.g2p, G2pStrategy::Zhuyin);
        assert!(cfg.speed_is_int);
        assert!(cfg.output_is_1d);
    }

    #[test]
    fn test_resolve_model_config_invalid() {
        let model = make_test_model("invalid-model", "Invalid", Some(test_model_path_v11()));
        assert!(resolve_model_config(&model).is_err());
    }

    #[test]
    fn test_resolve_model_config_without_filepath() {
        let model = make_test_model("kokoro-82m-v1.1-zh", "Kokoro-82M-v1.1-zh", None);
        let err = resolve_model_config(&model).unwrap_err();
        assert!(err.to_string().contains("file_paths"));
    }

    #[test]
    fn test_resolve_model_config_with_custom_path() {
        // 验证 file_paths 中的路径被优先使用
        let mut model = make_test_model("kokoro-82m-v1.1-zh", "Kokoro-82M-v1.1-zh", None);
        model.file_paths.push(
            workspace_root_for_test().join("models/tts/kokoro-82m-v1.1-zh/kokoro-v1.1-zh.onnx")
        );
        let cfg = resolve_model_config(&model).unwrap();
        assert_eq!(
            cfg.base_dir,
            workspace_root_for_test().join("models/tts/kokoro-82m-v1.1-zh")
        );
    }
}
