//! Qwen3-TTS 1.7B VoiceDesign 引擎（ONNX Runtime 推理）
//!
//! 四阶段流水线：
//! 1. Talker Prefill：处理完整输入序列 → logits + KV 缓存
//! 2. Talker Decode：单步自回归解码 + 代码预测
//! 3. Code Predictor：预测 16 个码本组
//! 4. Vocoder：码本序列 → 24kHz 波形
//!
//! 参考 generate_onnx.py (wavekat/Qwen3-TTS-1.7B-VoiceDesign-ONNX)

    use std::collections::HashMap;
    use std::io::Read;
    use std::io::Write;
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::Mutex;

use ndarray::{Array1, Array2, Array3, ArrayD, Axis, IxDyn};
use ort::session::Session;
use ort::value::Value;
use rand::Rng;

use votex_domain::error::TtsError;
use votex_domain::model::entity::Model;
use votex_domain::model::value_object::EngineKind;
use votex_domain::shared::value_object::AudioData;
use votex_domain::tts::provider::TtsProvider;
use votex_domain::tts::value_object::{TtsParams, VoiceId};

use crate::shared::{EngineState, ExecutionProvider, ModelFileLocator, OrtSessionFactory};

// ============================================================
// 常量定义
// ============================================================

/// Qwen3-TTS 采样率：24kHz
const SAMPLE_RATE: u32 = 24000;
/// 最大生成帧数
const MAX_NEW_TOKENS: usize = 2048;
/// 默认采样参数
const DEFAULT_TEMPERATURE: f32 = 0.9;
const DEFAULT_TOP_K: usize = 50;

// ============================================================
// 自定义 .npy 文件读取器（避免额外依赖）
// ============================================================

/// 读取 .npy 文件中的 float32 数据，返回 ArrayD<f32>
fn read_npy_f32(path: &Path) -> Result<ArrayD<f32>, TtsError> {
    let data = std::fs::read(path)
        .map_err(|e| TtsError::SynthesisFailed(format!("读取 NPY 文件失败 {:?}: {}", path, e)))?;

    if data.len() < 11 || &data[0..6] != b"\x93NUMPY" {
        return Err(TtsError::SynthesisFailed(
            format!("无效的 NPY 文件（魔数错误）: {:?}", path)
        ));
    }

    let version_major = data[6];
    let header_len = if version_major == 1 {
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else if version_major == 2 {
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    } else if version_major == 3 {
        u64::from_le_bytes([
            data[8], data[9], data[10], data[11],
            data[12], data[13], data[14], data[15],
        ]) as usize
    } else {
        return Err(TtsError::SynthesisFailed(
            format!("不支持的 NPY 版本: v{}", version_major)
        ));
    };

    let hdr_off = if version_major == 1 { 10 } else { 12 + 4 * (version_major as usize - 1) };
    let hdr_end = hdr_off + header_len;
    let hdr_str = std::str::from_utf8(&data[hdr_off..hdr_end])
        .map_err(|e| TtsError::SynthesisFailed(format!("NPY 头解析失败: {}", e)))?;

    // 解析 shape: {'descr': '<f4', 'fortran_order': False, 'shape': (a, b, ...), }
    let shape_str = hdr_str.split("'shape': ")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .map(|s| s.trim_start().trim_start_matches('('))
        .ok_or_else(|| TtsError::SynthesisFailed(
            format!("NPY 头中找不到 shape: {}", hdr_str)
        ))?;

    let shape: Vec<usize> = if shape_str.is_empty() {
        vec![1]
    } else {
        shape_str.split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| s.parse::<usize>().unwrap_or(1))
            .filter(|&d| d > 0)
            .collect()
    };

    let data_start = hdr_end + (64 - (hdr_end % 64)) % 64;
    let raw_data = &data[data_start..];
    let num_floats = shape.iter().product::<usize>();

    if raw_data.len() < num_floats * 4 {
        return Err(TtsError::SynthesisFailed(
            format!("NPY 数据不足: 期望 {} float32, 实际 {} 字节",
                num_floats, raw_data.len())
        ));
    }

    let bytes = &raw_data[..num_floats * 4];
    let floats: Vec<f32> = bytes.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    match shape.len() {
        2 => {
            let arr = Array2::from_shape_vec((shape[0], shape[1]), floats)
                .map_err(|e| TtsError::SynthesisFailed(format!("NPY 形状: {}", e)))?;
            Ok(arr.into_dyn())
        }
        _ => Ok(ArrayD::from_shape_vec(IxDyn(&shape), floats)
            .map_err(|e| TtsError::SynthesisFailed(format!("NPY 形状: {}", e)))?),
    }
}

// ============================================================
// 嵌入数据
// ============================================================

/// 内存映射的 text_embedding（避免加载 ~1.18GB 到堆内存）
///
/// Qwen3 基础 LLM 词表非常大（151936 tokens × 2048 维），
/// 用内存映射文件按需读取 token 行，避免 OOM。
struct MmapTextEmbedding {
    /// 文件的内存映射
    _mmap: memmap2::Mmap,
    /// 行数（token 数）
    rows: usize,
    /// 列数（embedding 维度）
    cols: usize,
    /// 数据起始偏移（NPY header 后的 64 字节对齐偏移）
    data_start: usize,
}

impl MmapTextEmbedding {
    /// 从 NPY 文件加载内存映射
    fn load(path: &Path) -> Result<Self, TtsError> {
        let mut file = std::fs::File::open(path)
            .map_err(|e| TtsError::SynthesisFailed(format!("打开 NPY 失败 {:?}: {}", path, e)))?;

        // 读前 512 字节解析 header
        let mut header_buf = [0u8; 512];
        let n = file.read(&mut header_buf)
            .map_err(|e| TtsError::SynthesisFailed(format!("读取 NPY 头失败 {:?}: {}", path, e)))?;
        if n < 11 || &header_buf[0..6] != b"\x93NUMPY" {
            return Err(TtsError::SynthesisFailed(
                format!("无效的 NPY 文件（魔数错误）: {:?}", path)
            ));
        }

        let version_major = header_buf[6];
        let header_len = if version_major == 1 {
            u16::from_le_bytes([header_buf[8], header_buf[9]]) as usize
        } else if version_major == 2 {
            u32::from_le_bytes([header_buf[8], header_buf[9], header_buf[10], header_buf[11]]) as usize
        } else if version_major == 3 {
            u64::from_le_bytes([
                header_buf[8], header_buf[9], header_buf[10], header_buf[11],
                header_buf[12], header_buf[13], header_buf[14], header_buf[15],
            ]) as usize
        } else {
            return Err(TtsError::SynthesisFailed(
                format!("不支持的 NPY 版本: v{}", version_major)
            ));
        };

        let hdr_off = if version_major == 1 { 10 } else { 12 + 4 * (version_major as usize - 1) };
        let hdr_end = hdr_off + header_len;

        // header 必须能完全读入 512 字节缓冲区
        if hdr_end > 512 {
            return Err(TtsError::SynthesisFailed(
                format!("NPY header 过大 ({})", header_len)
            ));
        }

        let hdr_str = std::str::from_utf8(&header_buf[hdr_off..hdr_end])
            .map_err(|e| TtsError::SynthesisFailed(format!("NPY 头解析失败: {}", e)))?;

        // 解析 shape
        let shape_str = hdr_str.split("'shape': ")
            .nth(1)
            .and_then(|s| s.split(')').next())
            .map(|s| s.trim_start().trim_start_matches('('))
            .ok_or_else(|| TtsError::SynthesisFailed(
                format!("NPY 头中找不到 shape: {}", hdr_str)
            ))?;

        let shape: Vec<usize> = if shape_str.is_empty() {
            vec![1]
        } else {
            shape_str.split(',')
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .map(|s| s.parse::<usize>().unwrap_or(1))
                .filter(|&d| d > 0)
                .collect()
        };

        if shape.len() != 2 {
            return Err(TtsError::SynthesisFailed(
                format!("text_embedding 期望 2D shape，实际 {}D", shape.len())
            ));
        }

        // 数据起始偏移（64 字节对齐）
        let data_start = hdr_end + (64 - (hdr_end % 64)) % 64;

        // 内存映射整个文件
        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| TtsError::SynthesisFailed(format!("映射 NPY 文件失败 {:?}: {}", path, e)))?
        };

        let rows = shape[0];
        let cols = shape[1];
        let expected_bytes = rows * cols * 4;

        // 验证文件完整性
        if mmap.len() < data_start + expected_bytes {
            return Err(TtsError::SynthesisFailed(
                format!("NPY 数据不足: 期望 {} float32, 实际 {} 字节",
                    rows * cols, mmap.len() - data_start)
            ));
        }

        Ok(Self { _mmap: mmap, rows, cols, data_start })
    }

    /// 返回行数
    fn nrows(&self) -> usize {
        self.rows
    }

    /// 返回列数（embedding 维度）
    fn ncols(&self) -> usize {
        self.cols
    }

    /// 返回第 i 行的 f32 切片
    fn row(&self, i: usize) -> &[f32] {
        debug_assert!(i < self.rows, "text_embedding 行索引越界: {} >= {}", i, self.rows);
        let start = self.data_start + i * self.cols * 4;
        let end = start + self.cols * 4;
        let bytes = &self._mmap[start..end];
        // 安全：内存映射文件是 f32 的小端表示，已验证 shape 正确
        unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const f32,
                self.cols,
            )
        }
    }
}

/// 从 .npy 加载的嵌入权重
struct Embeddings {
    text_embedding: MmapTextEmbedding,
    text_projection_fc1_weight: Array2<f32>,
    text_projection_fc1_bias: Array1<f32>,
    text_projection_fc2_weight: Array2<f32>,
    text_projection_fc2_bias: Array1<f32>,
    talker_codec_embedding: Array2<f32>,
    cp_codec_embeddings: Vec<Array2<f32>>,
}

impl Embeddings {
    fn load(model_dir: &Path) -> Result<Self, TtsError> {
        let ed = model_dir.join("embeddings");
        let load = |name: &str| read_npy_f32(&ed.join(name));

        // text_embedding.npy 使用内存映射加载（文件 ~1.18GB，避免堆内存溢出）
        let text_embedding = Self::load_text_embedding(&ed)?;
        let fc1_w = load("text_projection_fc1_weight.npy")?;
        let fc1_b = load("text_projection_fc1_bias.npy")?;
        let fc2_w = load("text_projection_fc2_weight.npy")?;
        let fc2_b = load("text_projection_fc2_bias.npy")?;
        let codec_emb = load("talker_codec_embedding.npy")?;

        // 动态加载 cp_codec_embedding_{i}.npy（文件数由模型架构决定）
        let mut cp_embs = Vec::new();
        let mut i = 0;
        loop {
            let path = ed.join(format!("cp_codec_embedding_{}.npy", i));
            if !path.exists() { break; }
            let a = read_npy_f32(&path)?;
            cp_embs.push(a.into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| TtsError::SynthesisFailed(format!("cp_{} 形状: {}", i, e)))?);
            i += 1;
        }
        tracing::info!("加载了 {} 个 cp_codec_embedding 文件", cp_embs.len());

        fn to2(a: ArrayD<f32>, name: &str) -> Result<Array2<f32>, TtsError> {
            a.into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| TtsError::SynthesisFailed(format!("{} 形状: {}", name, e)))
        }
        fn to1(a: ArrayD<f32>, name: &str) -> Result<Array1<f32>, TtsError> {
            a.into_dimensionality::<ndarray::Ix1>()
                .map_err(|e| TtsError::SynthesisFailed(format!("{} 形状: {}", name, e)))
        }

        Ok(Self {
            text_embedding,
            text_projection_fc1_weight: to2(fc1_w, "fc1_w")?,
            text_projection_fc1_bias: to1(fc1_b, "fc1_b")?,
            text_projection_fc2_weight: to2(fc2_w, "fc2_w")?,
            text_projection_fc2_bias: to1(fc2_b, "fc2_b")?,
            talker_codec_embedding: to2(codec_emb, "codec_emb")?,
            cp_codec_embeddings: cp_embs,
        })
    }

    /// 加载 text_embedding.npy（内存映射 + fallback 到兄弟模型）
    fn load_text_embedding(ed: &Path) -> Result<MmapTextEmbedding, TtsError> {
        let path = ed.join("text_embedding.npy");
        match MmapTextEmbedding::load(&path) {
            Ok(te) => Ok(te),
            Err(e) => {
                // text_embedding.npy 在 0.6B / 1.7B 模型中共享相同的 Qwen3 LLM 词表。
                // 如果当前模型的 embedding 文件损坏/截断（如下载不完整），
                // 自动 fallback 到兄弟模型目录下的完整文件。
                let tts_dir = ed.parent().and_then(|p| p.parent()); // tts/
                let fallback_paths = [
                    tts_dir.map(|p| p.join("qwen3-tts-1.7b").join("embeddings").join("text_embedding.npy")),
                    tts_dir.map(|p| p.join("qwen3-tts").join("embeddings").join("text_embedding.npy")),
                ];
                let last_err = e;
                for fp_opt in &fallback_paths {
                    if let Some(fp) = fp_opt {
                        if fp.exists() {
                            tracing::warn!(
                                "text_embedding.npy 加载失败 ({}), 尝试 fallback: {:?}",
                                last_err, fp,
                            );
                            return MmapTextEmbedding::load(fp);
                        }
                    }
                }
                Err(last_err)
            }
        }
    }

    /// 文本投影：SiLU 门控 MLP
    ///
    /// text_embedding 是 Qwen3 基础 LLM 词表，维度固定为 2048。
    /// talker_hidden_size 是 talker 模型的隐藏层维度（0.6B=1024, 1.7B=2048），
    /// 两者可能不同，所以 embeds 的列数用 text_embedding.ncols()。
    /// 投影权重（fc1/fc2）会自动映射到正确的输出维度。
    fn text_project(&self, token_ids: &[i64]) -> Array2<f32> {
        let n = token_ids.len();
        let text_dim = self.text_embedding.ncols();
        let mut embeds = Array2::zeros((n, text_dim));
        for (i, &tid) in token_ids.iter().enumerate() {
            let tidx = tid as usize;
            if tidx < self.text_embedding.nrows() {
                let row_slice = self.text_embedding.row(tidx);
                let row_view = ndarray::ArrayView1::from(row_slice);
                embeds.row_mut(i).assign(&row_view);
            }
        }
        // hidden = embeds @ fc1_w.T + fc1_b
        let hidden = embeds.dot(&self.text_projection_fc1_weight.t()) + &self.text_projection_fc1_bias;
        // SiLU
        let sig = hidden.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let activated = hidden * sig;
        // return activated @ fc2_w.T + fc2_b
        activated.dot(&self.text_projection_fc2_weight.t()) + &self.text_projection_fc2_bias
    }

    fn text_project_single(&self, token_id: i64) -> Array1<f32> {
        let r = self.text_project(&[token_id]);
        let len = r.len();
        r.clone().into_shape_with_order(len).unwrap()
    }
}

// ============================================================
// 采样函数
// ============================================================

fn sample_top_k(logits: &[f32], top_k: usize, temperature: f32) -> i64 {
    let mut rng = rand::thread_rng();
    let mut scores = logits.to_vec();
    if (temperature - 1.0).abs() > f32::EPSILON {
        for s in scores.iter_mut() { *s /= temperature; }
    }
    if top_k > 0 && top_k < scores.len() {
        let mut sorted = scores.clone();
        sorted.select_nth_unstable_by(scores.len() - top_k, |a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[scores.len() - top_k];
        for s in scores.iter_mut() { if *s < threshold { *s = f32::NEG_INFINITY; } }
    }
    let max_v = scores.iter().cloned().max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)).unwrap_or(0.0);
    for s in scores.iter_mut() { *s = (*s - max_v).exp(); }
    let sum: f32 = scores.iter().sum();
    if sum <= 0.0 || !sum.is_finite() { return 0; }
    for s in scores.iter_mut() { *s /= sum; }
    let r: f32 = rng.gen();
    let mut cum = 0.0;
    for (i, &p) in scores.iter().enumerate() {
        cum += p;
        if r < cum { return i as i64; }
    }
    (scores.len() - 1) as i64
}

// ============================================================
// Tokenizer：使用 tokenizers crate v0.21 加载 Qwen2 tokenizer.json
// ============================================================

fn load_tokenizer(model_dir: &Path) -> Result<tokenizers::Tokenizer, TtsError> {
    // 优先加载 tokenizer.json（tokenizers 格式）
    let tj_path = model_dir.join("tokenizer/tokenizer.json");
    if tj_path.exists() {
        return tokenizers::Tokenizer::from_file(&tj_path)
            .map_err(|e| TtsError::SynthesisFailed(format!("加载 tokenizer.json 失败: {}", e)));
    }
    // 回退到 BPE 格式（vocab.json + merges.txt，如 elbruno 0.6B 模型）
    let vocab_path = model_dir.join("tokenizer/vocab.json");
    let merges_path = model_dir.join("tokenizer/merges.txt");
    if vocab_path.exists() && merges_path.exists() {
        let vocab_str = vocab_path.to_str().unwrap_or("");
        let merges_str = merges_path.to_str().unwrap_or("");
        let bpe = tokenizers::models::bpe::BPE::from_file(vocab_str, merges_str)
            .build()
            .map_err(|e| TtsError::SynthesisFailed(format!("构建 BPE tokenizer 失败: {}", e)))?;
        let tok = tokenizers::Tokenizer::new(bpe);
        // Qwen2 BPE tokenizer 不需要特殊的 decoder，保持默认即可
        return Ok(tok);
    }
    Err(TtsError::SynthesisFailed(
        "未找到 tokenizer 文件（tokenizer.json 或 vocab.json+merges.txt）".into()
    ))
}

fn tokenize_text(tok: &tokenizers::Tokenizer, text: &str) -> Vec<i64> {
    // Python 参考设置：
    //   input_ids = tokenizer.encode(chat_text, add_special_tokens=False)
    // 所以 Rust 也必须用 add_special_tokens=false，
    // 因为 chat 模板已手动包含 <|im_start|>/<|im_end|> 标记。
    match tok.encode(text, false) {
        Ok(enc) => enc.get_ids().iter().map(|&id| id as i64).collect(),
        Err(_) => Vec::new(),
    }
}

// ============================================================
// 配置结构
// ============================================================

#[derive(Clone)]
struct Qwen3TtsConfig {
    // Token IDs
    tts_pad_token_id: i64,
    tts_bos_token_id: i64,
    tts_eos_token_id: i64,
    codec_eos_token_id: i64,
    codec_think_id: i64,
    codec_think_bos_id: i64,
    codec_think_eos_id: i64,
    codec_pad_id: i64,
    codec_bos_id: i64,
    language_ids: HashMap<String, i64>,

    // 架构参数 — 从 config.json 读取，支持不同规模的模型变体
    talker_hidden_size: usize,
    talker_num_layers: usize,
    num_code_groups: usize,
    cp_num_layers: usize,
    cp_num_kv_heads: usize,
    cp_head_dim: usize,

    variant: String,
}

impl Qwen3TtsConfig {
    /// 从模型目录加载配置，支持两种布局：
    /// - wavekat 格式：根目录 config.json（扁平键）
    /// - elbruno 格式：embeddings/config.json（嵌套结构）
    fn load(model_dir: &Path) -> Result<Self, TtsError> {
        // 尝试根目录或 embeddings/ 子目录
        let config_paths = [
            model_dir.join("config.json"),
            model_dir.join("embeddings/config.json"),
        ];
        let mut parsed = None;
        for p in &config_paths {
            if p.exists() {
                let s = std::fs::read_to_string(p)
                    .map_err(|e| TtsError::SynthesisFailed(format!("读取 {:?} 失败: {}", p, e)))?;
                let c: serde_json::Value = serde_json::from_str(&s)
                    .map_err(|e| TtsError::SynthesisFailed(format!("解析 {:?} 失败: {}", p, e)))?;
                parsed = Some(c);
                break;
            }
        }
        let c = parsed.ok_or_else(|| TtsError::SynthesisFailed(
            "未找到 config.json（根目录或 embeddings/ 下）".into()
        ))?;

        // 辅助函数：从扁平键或嵌套的 talker 段取值（优先 talker）
        let get_talker_u64 = |flat_key: &str, nested_key: &str, default: u64| -> u64 {
            c[flat_key].as_u64()
                .or_else(|| c["talker"][nested_key].as_u64())
                .or_else(|| c["code_predictor"][nested_key].as_u64())
                .unwrap_or(default)
        };
        // 辅助函数：从扁平键或嵌套的 code_predictor 段取值（优先 code_predictor）
        let get_cp_u64 = |flat_key: &str, nested_key: &str, default: u64| -> u64 {
            c[flat_key].as_u64()
                .or_else(|| c["code_predictor"][nested_key].as_u64())
                .or_else(|| c["talker"][nested_key].as_u64())
                .unwrap_or(default)
        };
        let get_i64 = |keys: &[&str], default: i64| -> i64 {
            for &k in keys {
                // 尝试扁平键: c["key"]
                if let Some(v) = c[k].as_i64() { return v; }
                // 尝试 talker/key
                if let Some(v) = c["talker"][k].as_i64() { return v; }
                // 尝试 tts/key
                if let Some(v) = c["tts"][k].as_i64() { return v; }
            }
            default
        };

        // 语言 ID
        let lang_ids = c["codec_language_id"].as_object()
            .or_else(|| c["language_ids"].as_object())
            .map(|o| o.iter().map(|(k, v)| (k.clone(), v.as_i64().unwrap_or(2050))).collect())
            .unwrap_or_default();

        let talker_hidden = get_talker_u64("talker_hidden_size", "hidden_size", 2048) as usize;
        let talker_layers = get_talker_u64("talker_num_layers", "num_hidden_layers", 28) as usize;
        let num_cg = get_talker_u64("talker_num_code_groups", "num_code_groups", 16) as usize;
        let cp_layers = get_cp_u64("cp_num_layers", "num_hidden_layers", 5) as usize;
        let cp_kv_heads = get_cp_u64("cp_num_kv_heads", "num_key_value_heads", 8) as usize;
        let cp_hdim = get_cp_u64("cp_head_dim", "head_dim", 128) as usize;

        tracing::info!("  Qwen3-TTS 架构: talker_hidden={talker_hidden}, layers={talker_layers}, code_groups={num_cg}");
        tracing::info!("  Code Predictor: layers={cp_layers}, kv_heads={cp_kv_heads}, head_dim={cp_hdim}");

        Ok(Self {
            tts_pad_token_id: get_i64(&["tts_pad_token_id", "pad_token_id"], 151671),
            tts_bos_token_id: get_i64(&["tts_bos_token_id", "bos_token_id"], 151672),
            tts_eos_token_id: get_i64(&["tts_eos_token_id", "eos_token_id"], 151673),
            codec_eos_token_id: get_i64(&["codec_eos_token_id"], 2150),
            codec_think_id: get_i64(&["codec_think_id"], 2154),
            codec_think_bos_id: get_i64(&["codec_think_bos_id"], 2156),
            codec_think_eos_id: get_i64(&["codec_think_eos_id"], 2157),
            codec_pad_id: get_i64(&["codec_pad_id"], 2148),
            codec_bos_id: get_i64(&["codec_bos_id"], 2149),
            language_ids: lang_ids,
            talker_hidden_size: talker_hidden,
            talker_num_layers: talker_layers,
            num_code_groups: num_cg,
            cp_num_layers: cp_layers,
            cp_num_kv_heads: cp_kv_heads,
            cp_head_dim: cp_hdim,
            variant: String::new(),
        })
    }
}

// ============================================================
// ONNX 会话集合
// ============================================================

struct Qwen3TtsSessions {
    prefill: Session,
    decode: Session,
    code_predictor: Session,
    vocoder: Session,
}

// ============================================================
// Builder 辅助：将 ort::Error 转为 TtsError
// ============================================================

fn map_ort_err(e: ort::Error) -> TtsError {
    TtsError::SynthesisFailed(format!("ONNX 运算失败: {}", e))
}

// ============================================================
// Qwen3TtsProvider
// ============================================================

pub struct Qwen3TtsProvider {
    sessions: EngineState<Qwen3TtsSessions>,
    config: Mutex<Option<Qwen3TtsConfig>>,
    embeddings: Mutex<Option<Embeddings>>,
    tokenizer: Mutex<Option<tokenizers::Tokenizer>>,
    voices: Vec<VoiceId>,
    /// true=0.6B（elbruno，预设 speaker embedding），
    /// false=1.7B（wavekat，VoiceDesign instruct）
    is_small: bool,
    /// 段内进度回调：参数 (current_step, total_steps, message)
    /// 使用 Arc 方便在闭包间共享
    progress_cb: Mutex<Option<Arc<dyn Fn(u32, u32, &str) + Send + Sync>>>,
}

impl Qwen3TtsProvider {
    pub fn new() -> Self {
        let voices = vec![
            VoiceId::new("default", "默认音色（中性女声）", EngineKind::Qwen3Tts),
            VoiceId::new("warm_female", "温暖亲切女声", EngineKind::Qwen3Tts),
            VoiceId::new("professional_male", "专业沉稳男声", EngineKind::Qwen3Tts),
            VoiceId::new("cheerful_female", "欢快活泼女声", EngineKind::Qwen3Tts),
            VoiceId::new("deep_male", "低沉浑厚男声", EngineKind::Qwen3Tts),
            VoiceId::new("soft_female", "轻柔舒缓女声", EngineKind::Qwen3Tts),
        ];
        Self {
            sessions: EngineState::new(),
            config: Mutex::new(None),
            embeddings: Mutex::new(None),
            tokenizer: Mutex::new(None),
            voices,
            is_small: false,
            progress_cb: Mutex::new(None),
        }
    }

    /// 设置段内进度回调（可选，用于 GUI 进度条实时更新）
    pub fn set_progress_callback(&self, cb: Arc<dyn Fn(u32, u32, &str) + Send + Sync>) {
        *self.progress_cb.lock().unwrap() = Some(cb);
    }

    /// 清除段内进度回调
    pub fn clear_progress_callback(&self) {
        *self.progress_cb.lock().unwrap() = None;
    }

    /// 报告段内进度（内部方法）
    fn report_progress(&self, current_step: u32, total_steps: u32, msg: &str) {
        if let Some(ref cb) = *self.progress_cb.lock().unwrap() {
            cb(current_step, total_steps, msg);
        }
    }

    /// 返回音色对应的 instruct 文本。
    /// - 1.7B（wavekat）模型：VoiceDesign 风格，通过 instruct 文本控制音色
    /// - 0.6B（elbruno）模型：预训练 speaker embedding，无需 instruct
    fn voice_to_instruct(&self, voice_id: &str) -> Option<String> {
        // 0.6B 模型使用预设 speaker embedding，instruct 文本无效
        if self.is_small {
            return None;
        }
        match voice_id {
            "default" => Some("Speak in a natural and neutral female voice.".to_string()),
            "warm_female" => Some("Speak in a warm and friendly female voice.".to_string()),
            "professional_male" => Some("Speak in a professional and steady male voice.".to_string()),
            "cheerful_female" => Some("Speak in a cheerful and lively female voice.".to_string()),
            "deep_male" => Some("Speak slowly in a deep and rich male voice.".to_string()),
            "soft_female" => Some("Speak in a soft and gentle female voice.".to_string()),
            _ => None,
        }
    }
}

impl TtsProvider for Qwen3TtsProvider {
    fn engine_kind(&self) -> EngineKind { EngineKind::Qwen3Tts }

    fn load(&mut self, model: &Model) -> Result<(), TtsError> {
        let model_dir = ModelFileLocator::locate_model_dir(model)
            .map_err(|e| TtsError::SynthesisFailed(format!("查找模型目录失败: {}", e)))?;

        tracing::info!("Qwen3-TTS 开始加载: {:?}", model_dir);

        // 先加载 config 以确定模型尺寸（避免 junction 路径干扰路径名检测）
        let mut config = Qwen3TtsConfig::load(&model_dir)?;
        // 检测模型变体（0.6B / 1.7B）：talker_hidden_size=1024 为 0.6B, 2048 为 1.7B
        let is_small = config.talker_hidden_size == 1024;
        let ep = OrtSessionFactory::get_global_ep();
        let use_small = if ep == ExecutionProvider::Cpu {
            if is_small {
                tracing::info!("  CPU 模式 → 0.6B 模型: {:?}", model_dir);
            } else {
                tracing::info!("  CPU 模式 → 1.7B 模型（int4 量化）: {:?}", model_dir);
            }
            is_small
        } else {
            if is_small {
                tracing::info!("  GPU/DirectML 模式 → 0.6B 模型: {:?}", model_dir);
            } else {
                tracing::info!("  GPU/DirectML 模式 → 1.7B 模型: {:?}", model_dir);
            }
            is_small
        };

        // 检测 ONNX 文件布局：int4/ → fp32/ → 根目录（elbruno 格式）
        let variant = if model_dir.join("int4/talker_prefill.onnx").exists() { "int4" }
            else if model_dir.join("fp32/talker_prefill.onnx").exists() { "fp32" }
            else if model_dir.join("talker_prefill.onnx").exists() { "direct" }
            else { return Err(TtsError::SynthesisFailed(
                format!("未找到 ONNX 文件（{:?} 下无 int4/fp32/talker_prefill.onnx）", model_dir)
            )); };
        config.variant = variant.to_string();
        let onnx_dir = match variant {
            "direct" => model_dir.clone(),
            v => model_dir.join(v),
        };

        tracing::info!("  ONNX 布局: {}（{:?}）", variant, onnx_dir);

        // Tokenizer
        let tokenizer = load_tokenizer(&model_dir)?;

        // Embeddings
        let embeddings = Embeddings::load(&model_dir)?;

        // int4 量化 ONNX 模型使用了 MatMulNBits 等量化算子，
        // DirectML EP 可能不支持这些算子，强制使用 CPU 提供器以避免崩溃。
        // fp32 变体的数据文件（~5.27 GB）超出可用内存，因此 int4 + CPU 是唯一可行路径。
        // int4 模型已量化，Level3 图优化效果有限且加载极慢（10 分钟+），降为 Level2。
        let onnx_ep = if use_small || variant == "int4" { ExecutionProvider::Cpu } else { ep };
        let create_onnx = |name: &str| {
            if variant == "int4" {
                // int4 模型的 ONNX 导出将权重拆为百万级小 tensor，
                // 开启任何级别图优化（Level1~3）都会触发巨量 BFCArena 分配 → 加载超时。
                // 直接无优化加载，int4 量化图已预优化，不影响推理性能。
                OrtSessionFactory::create_raw_with_ep_and_level(
                    &onnx_dir.join(name),
                    onnx_ep,
                    ort::session::builder::GraphOptimizationLevel::Disable,
                )
                .map_err(|e| TtsError::SynthesisFailed(format!("加载 {} 失败: {}", name, e)))
            } else {
                OrtSessionFactory::create_raw_with_ep(&onnx_dir.join(name), onnx_ep)
                    .map_err(|e| TtsError::SynthesisFailed(format!("加载 {} 失败: {}", name, e)))
            }
        };

        // ONNX 模型（4 个）
        let prefill = create_onnx("talker_prefill.onnx")?;
        let decode = create_onnx("talker_decode.onnx")?;
        let cp = create_onnx("code_predictor.onnx")?;
        let vocoder = create_onnx("vocoder.onnx")?;

        self.sessions.load(Qwen3TtsSessions { prefill, decode, code_predictor: cp, vocoder });
        self.is_small = is_small;
        *self.config.lock().unwrap() = Some(config);
        *self.embeddings.lock().unwrap() = Some(embeddings);
        *self.tokenizer.lock().unwrap() = Some(tokenizer);

        // 根据模型变体设置音色列表
        self.voices = if use_small {
            vec![
                VoiceId::new("serena", "Serena（自然女声）", EngineKind::Qwen3Tts),
                VoiceId::new("vivian", "Vivian（温暖女声）", EngineKind::Qwen3Tts),
                VoiceId::new("uncle_fu", "Uncle Fu（大叔音）", EngineKind::Qwen3Tts),
                VoiceId::new("ryan", "Ryan（青年男声）", EngineKind::Qwen3Tts),
                VoiceId::new("aiden", "Aiden（清亮男声）", EngineKind::Qwen3Tts),
                VoiceId::new("ono_anna", "小野杏（日语女声）", EngineKind::Qwen3Tts),
                VoiceId::new("sohee", "Sohee（韩语女声）", EngineKind::Qwen3Tts),
                VoiceId::new("eric", "Eric（四川话男声）", EngineKind::Qwen3Tts),
                VoiceId::new("dylan", "Dylan（京片子男声）", EngineKind::Qwen3Tts),
            ]
        } else {
            vec![
                VoiceId::new("default", "默认音色（中性女声）", EngineKind::Qwen3Tts),
                VoiceId::new("warm_female", "温暖亲切女声", EngineKind::Qwen3Tts),
                VoiceId::new("professional_male", "专业沉稳男声", EngineKind::Qwen3Tts),
                VoiceId::new("cheerful_female", "欢快活泼女声", EngineKind::Qwen3Tts),
                VoiceId::new("deep_male", "低沉浑厚男声", EngineKind::Qwen3Tts),
                VoiceId::new("soft_female", "轻柔舒缓女声", EngineKind::Qwen3Tts),
            ]
        };

        let model_size = if use_small { "0.6B" } else { "1.7B" };
        tracing::info!("Qwen3-TTS 引擎加载完成（{}, {}，{} 种音色）", model_size, variant, self.voices.len());
        Ok(())
    }

    fn unload(&mut self) -> Result<(), TtsError> {
        self.sessions.unload();
        self.is_small = false;
        *self.config.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *self.embeddings.lock().unwrap_or_else(|e| e.into_inner()) = None;
        *self.tokenizer.lock().unwrap_or_else(|e| e.into_inner()) = None;
        Ok(())
    }

    fn synthesize(
        &self,
        text: &str,
        voice: &VoiceId,
        _params: &TtsParams,
    ) -> Result<AudioData, TtsError> {
        let cfg = self.config.lock().unwrap();
        let cfg = cfg.as_ref().ok_or(TtsError::EngineNotLoaded)?.clone();
        let emb = self.embeddings.lock().unwrap();
        let emb = emb.as_ref().ok_or(TtsError::EngineNotLoaded)?;
        let tok = self.tokenizer.lock().unwrap();
        let tok = tok.as_ref().ok_or(TtsError::EngineNotLoaded)?;
        if text.is_empty() { return Err(TtsError::EmptyText); }

        tracing::info!("Qwen3-TTS 合成: text='{}' ({} chars), voice={}", text, text.chars().count(), voice.id);
        println!("[synthesize] 进入合成方法：text='{}'", text);

        // ---- 1. Tokenize ----
        // Windows CRLF → LF 转换，确保 token 序列与 Python 参考一致
        let clean_text = text.replace("\r\n", "\n").replace('\r', "");
        let chat = format!("<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n", clean_text);
        let input_ids = tokenize_text(tok, &chat);

        // ---- 2. Instruct tokens（音色指令，先于角色前缀） ----
        // 与 Python 参考实现一致：指令文本需作为 user 消息包装再 tokenize
        let instruct_tokens: Vec<i64> = if let Some(instruct_str) = self.voice_to_instruct(&voice.id) {
            let instruct_chat = format!("<|im_start|>user\n{}<|im_end|>\n", instruct_str);
            tokenize_text(tok, &instruct_chat)
        } else {
            Vec::new()
        };

        // ---- 3. Build prefill embeddings ----
        let lang_id = cfg.language_ids.get("chinese").copied().unwrap_or(2055);
        let codec_prefix = [cfg.codec_think_id, cfg.codec_think_bos_id, lang_id, cfg.codec_think_eos_id];

        let pad_e = emb.text_project_single(cfg.tts_pad_token_id);
        let bos_e = emb.text_project_single(cfg.tts_bos_token_id);
        let eos_e = emb.text_project_single(cfg.tts_eos_token_id);
        let codec_pad = emb.talker_codec_embedding.row(cfg.codec_pad_id as usize).to_owned();
        let codec_bos = emb.talker_codec_embedding.row(cfg.codec_bos_id as usize).to_owned();

        let mut rows: Vec<Array1<f32>> = Vec::new();

        // instruct
        if !instruct_tokens.is_empty() {
            let ids: Vec<i64> = instruct_tokens.iter().map(|&id| id as i64).collect();
            for r in emb.text_project(&ids).rows() { rows.push(r.to_owned()); }
        }
        // role prefix
        if input_ids.len() >= 3 {
            let ids: Vec<i64> = input_ids[..3].iter().map(|&id| id as i64).collect();
            for r in emb.text_project(&ids).rows() { rows.push(r.to_owned()); }
        }
        // codec prefix
        for &cid in &codec_prefix {
            let ce = emb.talker_codec_embedding.row(cid as usize).to_owned();
            rows.push(&pad_e + &ce);
        }
        // transition
        rows.push(&bos_e + &codec_pad);
        // text tokens
        if input_ids.len() > 8 {
            let ids: Vec<i64> = input_ids[3..input_ids.len() - 5].iter().map(|&id| id as i64).collect();
            for &tid in &ids { rows.push(emb.text_project_single(tid) + &codec_pad); }
        }
        // end + final
        rows.push(&eos_e + &codec_pad);
        rows.push(&pad_e + &codec_bos);

        let seq_len = rows.len();
        if seq_len == 0 { return Err(TtsError::SynthesisFailed("空嵌入序列".into())); }

        let mut flat = Vec::with_capacity(seq_len * cfg.talker_hidden_size);
        for r in &rows { flat.extend_from_slice(r.as_slice().unwrap()); }
        let prefill_emb_arr = Array3::from_shape_vec((1, seq_len, cfg.talker_hidden_size), flat)
            .map_err(|e| TtsError::SynthesisFailed(format!("prefill 形状: {}", e)))?;
        let mask_arr = Array2::<i64>::ones((1, seq_len));
        let mut pd = Vec::with_capacity(3 * seq_len);
        for _ in 0..3 { for i in 0..seq_len as i64 { pd.push(i); } }
        let pos_arr = Array3::from_shape_vec((3, 1, seq_len), pd)
            .map_err(|e| TtsError::SynthesisFailed(format!("pos 形状: {}", e)))?;

        // Clone embeddings for closure use
        let codec_emb = emb.talker_codec_embedding.clone();
        let cp_codec_embs = emb.cp_codec_embeddings.clone();

        // ---- 4. Run pipeline in with_mut ----
        let audio = self.sessions.with_mut(|sessions| {
            // === Prefill ===
            let pf_v = Value::from_array(prefill_emb_arr.into_dyn()).map_err(map_ort_err)?;
            let mk_v = Value::from_array(mask_arr.into_dyn()).map_err(map_ort_err)?;
            let pp_v = Value::from_array(pos_arr.into_dyn()).map_err(map_ort_err)?;

            println!("[prefill] 开始推理...");
            let pf_out = sessions.prefill.run(ort::inputs![pf_v, mk_v, pp_v])
                .map_err(map_ort_err)?;
            println!("[prefill] 推理完成, {} 个输出", pf_out.len());

            if pf_out.len() < 3 {
                return Err(TtsError::SynthesisFailed(format!("prefill 输出不足: {}", pf_out.len())));
            }

            let pf_logits_raw = pf_out[0].try_extract_array::<f32>().map_err(map_ort_err)?;
            let pf_hidden_raw = pf_out[1].try_extract_array::<f32>().map_err(map_ort_err)?;
            tracing::info!("  prefill output[0] shape: {:?}", pf_logits_raw.shape());
            tracing::info!("  prefill output[1] shape: {:?}", pf_hidden_raw.shape());
            // DEBUG: logit 数值分布——与 Python 参考对比的关键指标
            {
                let s = pf_logits_raw.as_slice().unwrap_or(&[]);
                if !s.is_empty() {
                    let mut sorted: Vec<f32> = s.iter().copied().collect();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let n = sorted.len();
                    let mean: f32 = s.iter().sum::<f32>() / n as f32;
                    tracing::info!("  prefill logits: min={:.4}, max={:.4}, mean={:.4}, p50={:.4}, p90={:.4}, p99={:.4}",
                        sorted[0], sorted[n-1], mean, sorted[n/2], sorted[(n*9)/10], sorted[(n*99)/100]);
                    // 前10个 logit 值（用于与 Python 精确对比）
                    tracing::info!("  prefill logits[0,0,:10] = {:?}", &s[..10.min(s.len())]);
                }
            }
            {
                let s = pf_hidden_raw.as_slice().unwrap_or(&[]);
                if !s.is_empty() {
                    let mean: f32 = s.iter().sum::<f32>() / s.len() as f32;
                    tracing::info!("  prefill hidden: min={:.4}, max={:.4}, mean={:.4}",
                        s.iter().cloned().fold(f32::INFINITY, f32::min),
                        s.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
                        mean);
                }
            }
            if pf_out.len() > 2 {
                let kv_shape = pf_out[2].try_extract_array::<f32>().map_err(map_ort_err)?.shape().to_vec();
                tracing::info!("  prefill output[2] shape: {:?}, count={}", kv_shape, pf_out.len());
            }

            // 维度归一化：某些 0.6B ONNX 导出会输出 4D [1,1,3072,0]（动态 dim 为 0）
            // squeeze 掉大小为 1 的前导/尾部维度，移除大小为 0 的维度
            fn normalize_tensor(mut arr: ArrayD<f32>) -> Result<ArrayD<f32>, TtsError> {
                let orig_ndim = arr.ndim();
                // 移除大小为 0 的维度（ONNX 动态维度未绑定）
                let nonzero_shape: Vec<usize> = arr.shape().iter().copied().filter(|&d| d > 0).collect();
                if nonzero_shape.len() < arr.ndim() {
                    let total: usize = arr.shape().iter().product();
                    if total == 0 {
                        return Err(TtsError::SynthesisFailed(
                            format!("prefill 输出包含零大小维度，shape={:?}", arr.shape())));
                    }
                    arr = arr.into_shape_with_order(IxDyn(&nonzero_shape))
                        .map_err(|e| TtsError::SynthesisFailed(format!("reshape 移除零维: {}", e)))?;
                    tracing::info!("  normalize: 移除了零维, shape={:?}->{:?}", orig_ndim, arr.shape());
                }
                // 反复 squeeze 大小为 1 的维度直到 <= 3D
                let mut cur = arr;
                loop {
                    let s: Vec<usize> = cur.shape().iter().copied().collect();
                    if s.len() <= 3 { break; }
                    // 找到第一个大小为 1 的维度的索引
                    if let Some(idx) = s.iter().position(|&d| d == 1) {
                        let new_shape: Vec<usize> = s.iter().enumerate()
                            .filter(|&(i, _)| i != idx).map(|(_, &d)| d).collect();
                        cur = cur.into_shape_with_order(IxDyn(&new_shape))
                            .map_err(|e| TtsError::SynthesisFailed(format!("squeeze: {}", e)))?;
                    } else {
                        break; // 没有可 squeeze 的维度
                    }
                }
                Ok(cur)
            }

            let pf_logits = normalize_tensor(pf_logits_raw.into_owned())?;
            let pf_hidden = normalize_tensor(pf_hidden_raw.into_owned())?;
            tracing::info!("  normalize 后 logits shape: {:?}, hidden shape: {:?}",
                pf_logits.shape(), pf_hidden.shape());

            // Stack KV cache: prefill outputs 2+ are pairs of (k, v)
            // Stack into (num_layers, ...) tensors for decode
            let num_layers = cfg.talker_num_layers;
            let kv_raw = pf_out[2].try_extract_array::<f32>().map_err(map_ort_err)?;
            let kv_shape = kv_raw.shape().to_vec();
            tracing::info!("  prefill KV shape (原始): {:?}×{}", kv_shape, num_layers);
            let mut kv_dims = vec![num_layers];
            kv_dims.extend_from_slice(&kv_shape);

            let kv_total: usize = kv_shape.iter().product();
            let mut kv_keys = Vec::with_capacity(num_layers * kv_total);
            let mut kv_vals = Vec::with_capacity(num_layers * kv_total);

            for i in 0..num_layers {
                let k = pf_out[2 + i * 2].try_extract_array::<f32>().map_err(map_ort_err)?.into_owned();
                let v = pf_out[2 + i * 2 + 1].try_extract_array::<f32>().map_err(map_ort_err)?.into_owned();
                kv_keys.extend(k.iter().cloned());
                kv_vals.extend(v.iter().cloned());
            }

            // 如果 KV 本身也有多余的维度，在 stack 后 squeeze
            let past_keys_arr = ArrayD::from_shape_vec(IxDyn(&kv_dims), kv_keys)
                .map_err(|e| TtsError::SynthesisFailed(format!("stack keys: {}", e)))?;
            let past_vals_arr = ArrayD::from_shape_vec(IxDyn(&kv_dims), kv_vals)
                .map_err(|e| TtsError::SynthesisFailed(format!("stack vals: {}", e)))?;

            let mut past_keys_5d = past_keys_arr;
            let mut past_vals_5d = past_vals_arr;
            let mut logits_arr = pf_logits;
            let mut hidden_arr = pf_hidden;
            let mut all_codes: Vec<Vec<i64>> = Vec::new();
            let mut current_pos = seq_len as i64;

            tracing::info!("Qwen3-TTS 解码开始（max {} 帧）", MAX_NEW_TOKENS);

            let suppress_start = 3072 - 1024;
            let codec_eos = cfg.codec_eos_token_id as usize;
            const REPETITION_PENALTY: f32 = 1.05;
            let mut generated_tokens: Vec<i64> = Vec::new();

            for step in 0..MAX_NEW_TOKENS {
                let logits_last = if step == 0 {
                    // 安全取 prefill 输出的最后一个序列位置的 logits
                    if logits_arr.ndim() < 2 || seq_len as usize == 0 {
                        return Err(TtsError::SynthesisFailed(format!(
                            "prefill logits 形状异常: {:?}, seq_len={}", logits_arr.shape(), seq_len)));
                    }
                    let axis1_len = logits_arr.shape()[1];
                    let idx = (seq_len - 1).min(axis1_len.saturating_sub(1));
                    logits_arr.index_axis(Axis(1), idx).to_owned()
                } else {
                    // decode 输出形状为 [batch, 1, vocab]，取唯一序列位置
                    if logits_arr.ndim() < 2 {
                        return Err(TtsError::SynthesisFailed(format!(
                            "decode logits 形状异常: {:?}", logits_arr.shape())));
                    }
                    let axis1_len = logits_arr.shape()[1];
                    if axis1_len == 0 {
                        return Err(TtsError::SynthesisFailed(
                            "decode logits 序列维度为 0".to_string()));
                    }
                    logits_arr.index_axis(Axis(1), 0).to_owned()
                };
                let mut masked: Vec<f32> = logits_last.iter().cloned().collect();
                for i in suppress_start..3072 { if i != codec_eos { masked[i] = f32::NEG_INFINITY; } }
                if step < 2 { masked[codec_eos] = f32::NEG_INFINITY; }

                // Repetition penalty（抑制已生成的 token 重复）
                if REPETITION_PENALTY != 1.0 && !generated_tokens.is_empty() {
                    for &seen_id in &generated_tokens {
                        let idx = seen_id as usize;
                        if idx < masked.len() {
                            let s = masked[idx];
                            masked[idx] = if s > 0.0 { s / REPETITION_PENALTY } else { s * REPETITION_PENALTY };
                        }
                    }
                }
                let g0 = sample_top_k(&masked, DEFAULT_TOP_K, DEFAULT_TEMPERATURE);
                if g0 == cfg.codec_eos_token_id {
                    tracing::debug!("EOS at step {}", step);
                    break;
                }
                generated_tokens.push(g0);

                // === Code Predictor ===
                let num_cg = cfg.num_code_groups;
                let mut frame = vec![g0; num_cg];
                let h_pos = if step == 0 { seq_len - 1 } else { 0 };
                if hidden_arr.ndim() < 2 || h_pos as usize >= hidden_arr.shape()[1] {
                    return Err(TtsError::SynthesisFailed(format!(
                        "hidden 形状异常: {:?}, h_pos={}", hidden_arr.shape(), h_pos)));
                }
                let talker_h = hidden_arr.index_axis(Axis(1), h_pos as usize).to_owned();
                let g0_emb = codec_emb.row(g0 as usize).to_owned();

                let cp_in_data: Vec<f32> = talker_h.iter().cloned().chain(g0_emb.iter().cloned()).collect();
                let mut cp_in_arr = Array3::from_shape_vec((1, 2, cfg.talker_hidden_size), cp_in_data)
                    .map_err(|e| TtsError::SynthesisFailed(format!("cp input: {}", e)))?;

                // CP KV cache: empty, shape (num_layers, 1, num_kv_heads, 0, head_dim)
                let cp_kv_shape = IxDyn(&[cfg.cp_num_layers, 1, cfg.cp_num_kv_heads, 0, cfg.cp_head_dim]);
                let mut cp_pk = ArrayD::<f32>::zeros(cp_kv_shape.clone());
                let mut cp_pv = ArrayD::<f32>::zeros(cp_kv_shape);

                let h_size = cfg.talker_hidden_size;
                for g in 0..num_cg - 1 {
                    let v1 = Value::from_array(cp_in_arr.into_dyn()).map_err(map_ort_err)?;
                    let v2 = Value::from_array(ndarray::arr1(&[g as i64]).into_dyn()).map_err(map_ort_err)?;
                    let v3 = Value::from_array(cp_pk.clone()).map_err(map_ort_err)?;
                    let v4 = Value::from_array(cp_pv.clone()).map_err(map_ort_err)?;

                    let cp_out = sessions.code_predictor.run(ort::inputs![v1, v2, v3, v4])
                        .map_err(map_ort_err)?;

                    let cp_logits = cp_out[0].try_extract_array::<f32>().map_err(map_ort_err)?.into_owned();
                    cp_pk = cp_out[1].try_extract_array::<f32>().map_err(map_ort_err)?.into_owned();
                    cp_pv = cp_out[2].try_extract_array::<f32>().map_err(map_ort_err)?.into_owned();

                    // 取最后一个序列位置的 logits（Python 参考: cp_logits[0, -1, :]）
                    let seq_dim = cp_logits.ndim() - 2;
                    let last_pos = cp_logits.shape()[seq_dim] - 1;
                    let cp_logit_last = cp_logits.index_axis(Axis(seq_dim), last_pos).to_owned();
                    let cp_slice: Vec<f32> = cp_logit_last.iter().cloned().collect();
                    let token = sample_top_k(&cp_slice, DEFAULT_TOP_K, DEFAULT_TEMPERATURE);
                    frame[g + 1] = token;

                    let next_e = if (g as usize) < cp_codec_embs.len() {
                        let eg = &cp_codec_embs[g as usize];
                        let t = token as usize;
                        if t < eg.nrows() { eg.row(t).to_owned() } else { Array1::zeros(h_size) }
                    } else { Array1::zeros(h_size) };

                    let nd: Vec<f32> = next_e.iter().cloned().collect();
                    cp_in_arr = Array3::from_shape_vec((1, 1, h_size), nd)
                        .map_err(|e| TtsError::SynthesisFailed(format!("cp next: {}", e)))?;
                }

                all_codes.push(frame.clone());

                // === Build next talker input ===
                let mut next_emb = codec_emb.row(frame[0] as usize).to_owned();
                for g in 0..num_cg - 1 {
                    if (g as usize) < cp_codec_embs.len() {
                        let t = frame[g + 1] as usize;
                        let eg = &cp_codec_embs[g as usize];
                        if t < eg.nrows() { next_emb = next_emb + eg.row(t); }
                    }
                }
                next_emb = next_emb + &pad_e;

                let nd: Vec<f32> = next_emb.iter().cloned().collect();
                let next_3d = Array3::from_shape_vec((1, 1, h_size), nd)
                    .map_err(|e| TtsError::SynthesisFailed(format!("decode input: {}", e)))?;

                // === Talker Decode ===
                let new_pos = current_pos;
                let dk_arr = Array2::<i64>::ones((1, (new_pos + 1) as usize));
                let mut dd = Vec::with_capacity(3);
                for _ in 0..3 { dd.push(new_pos); }
                let dp_arr = Array3::from_shape_vec((3, 1, 1), dd)
                    .map_err(|e| TtsError::SynthesisFailed(format!("decode pos: {}", e)))?;

                let v1 = Value::from_array(next_3d.into_dyn()).map_err(map_ort_err)?;
                let v2 = Value::from_array(dk_arr.into_dyn()).map_err(map_ort_err)?;
                let v3 = Value::from_array(dp_arr.into_dyn()).map_err(map_ort_err)?;
                let v4 = Value::from_array(past_keys_5d.clone()).map_err(map_ort_err)?;
                let v5 = Value::from_array(past_vals_5d.clone()).map_err(map_ort_err)?;

                let dc_out = sessions.decode.run(ort::inputs![v1, v2, v3, v4, v5])
                    .map_err(map_ort_err)?;

                if dc_out.len() < 4 {
                    return Err(TtsError::SynthesisFailed(format!("decode 输出不足: {}", dc_out.len())));
                }

                logits_arr = dc_out[0].try_extract_array::<f32>().map_err(map_ort_err)?.into_owned();
                hidden_arr = dc_out[1].try_extract_array::<f32>().map_err(map_ort_err)?.into_owned();
                past_keys_5d = dc_out[2].try_extract_array::<f32>().map_err(map_ort_err)?.into_owned();
                past_vals_5d = dc_out[3].try_extract_array::<f32>().map_err(map_ort_err)?.into_owned();

                current_pos += 1;
                if (step + 1) % 50 == 0 {
                    tracing::info!("  ... {} 帧", step + 1);
                    self.report_progress((step + 1) as u32, MAX_NEW_TOKENS as u32, &format!("解码中... {} 帧", step + 1));
                }
            }

            let n_frames = all_codes.len();
            tracing::info!("  Generated {} frames", n_frames);
            self.report_progress(MAX_NEW_TOKENS as u32, MAX_NEW_TOKENS as u32, "声码器合成中...");
            if n_frames == 0 { return Err(TtsError::SynthesisFailed("未生成码本帧".into())); }

            // DEBUG: save codes to file for Python vocoder comparison
            {
                let codes_path = std::env::temp_dir().join("rust_tts_codes.txt");
                if let Ok(mut f) = std::fs::File::create(&codes_path) {
                    for frame in &all_codes {
                        let line: String = frame.iter()
                            .map(|c| c.to_string())
                            .collect::<Vec<_>>()
                            .join(",");
                        let _ = writeln!(f, "{}", line);
                    }
                    tracing::info!("  Codes saved to {:?}", codes_path);
                }
            }

            // === Vocoder ===
            let num_cg = cfg.num_code_groups;
            let mut cf = Vec::with_capacity(n_frames * num_cg);
            for f in &all_codes { for &c in f { cf.push(c); } }
            // 先构建 (T, num_cg) 矩阵：每行一帧，每列一个码本组
            let c2d = Array2::from_shape_vec((n_frames, num_cg), cf)
                .map_err(|e| TtsError::SynthesisFailed(format!("codes shape: {}", e)))?;
            // 转置为 (num_cg, T) 并 reshape 为 (1, num_cg, T) 供 vocoder 使用
            let c3d_data: Vec<i64> = c2d.t().iter().cloned().collect();
            let c3d = Array3::from_shape_vec((1, num_cg, n_frames), c3d_data)
                .map_err(|e| TtsError::SynthesisFailed(format!("codes 3d: {}", e)))?;

            let cv = Value::from_array(c3d.into_dyn()).map_err(map_ort_err)?;
            let vc_out = sessions.vocoder.run(ort::inputs![cv]).map_err(map_ort_err)?;
            let wav = vc_out[0].try_extract_array::<f32>().map_err(map_ort_err)?.into_owned();
            let samples: Vec<f32> = wav.iter().cloned().collect();

            tracing::info!("  Vocoder: {} samples ({:.1}s)", samples.len(), samples.len() as f64 / 24000.0);
            Ok(AudioData { samples, sample_rate: SAMPLE_RATE, channels: 1 })
        }).map_err(|_| TtsError::EngineNotLoaded)??;

        Ok(audio)
    }

    fn list_voices(&self) -> Vec<VoiceId> { self.voices.clone() }
    fn sample_rate(&self) -> u32 { SAMPLE_RATE }
    fn is_loaded(&self) -> bool { self.sessions.is_loaded() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use votex_domain::model::value_object::{ModelId, ModelKind};

    #[test]
    fn test_read_npy_f32_文件不存在应报错() {
        assert!(read_npy_f32(Path::new("/nonexistent.npy")).is_err());
    }

    #[test]
    fn test_sample_top_k() {
        let logits = vec![1000.0f32, 0.001, 0.001, 0.001];
        assert_eq!(sample_top_k(&logits, 1, 1.0), 0);
    }

    #[test]
    fn test_voice_to_instruct() {
        let p = Qwen3TtsProvider::new();
        assert!(p.voice_to_instruct("default").is_some());
        assert!(p.voice_to_instruct("nonexistent").is_none());
    }

    #[test]
    fn test_load_models() {
        let mut provider = Qwen3TtsProvider::new();
        let model = Model::new(
            ModelId::new("qwen3-tts"),
            "Qwen3-TTS",
            ModelKind::Tts,
            EngineKind::Qwen3Tts,
        );
        
        match provider.load(&model) {
            Ok(_) => {
                println!("✓ Qwen3-TTS 引擎加载成功");
                assert!(provider.sessions.is_loaded(), "引擎应该已加载");
            }
            Err(e) => {
                println!("✗ Qwen3-TTS 引擎加载失败: {}", e);
                panic!("加载失败: {}", e);
            }
        }
    }

    #[test]
    fn test_load_single_onnx() {
        use crate::shared::OrtSessionFactory;
        use std::path::Path;

        let base = Path::new(r"D:\19-Training\Rust\votex\models\tts\qwen3-tts-0.6b");
        let model_names = [
            "code_predictor.onnx",
            "vocoder.onnx",
            "talker_prefill.onnx",
            "talker_decode.onnx",
        ];

        for name in &model_names {
            let path = base.join(name);
            println!("加载 ONNX 模型: {:?} ({} MB)", path, 
                std::fs::metadata(&path).map(|m| m.len() / 1024 / 1024).unwrap_or(0));

            let data_path = base.join(format!("{}.data", name));
            if data_path.exists() {
                println!("  外部数据文件: {:?} ({} MB)", data_path,
                    std::fs::metadata(&data_path).map(|m| m.len() / 1024 / 1024).unwrap_or(0));
            }

            let start = std::time::Instant::now();
            match OrtSessionFactory::create_raw_with_ep(&path, ExecutionProvider::Cpu) {
                Ok(_session) => {
                    let elapsed = start.elapsed();
                    println!("  ✓ {name} 加载成功 ({:.1}s)", elapsed.as_secs_f64());
                }
                Err(e) => {
                    let elapsed = start.elapsed();
                    println!("  ✗ {name} 加载失败 ({:.1}s): {e}", elapsed.as_secs_f64());
                    panic!("ONNX 模型 {name} 加载失败: {e}");
                }
            }
        }
    }

    #[test]
    fn test_load_all_onnx_simultaneously() {
        use crate::shared::OrtSessionFactory;
        use std::path::Path;

        // prefill 和 decode 共享同一个 .onnx.data 文件（SHA256 相同）。
        // 同时加载两者时，ORT 可能因内存映射冲突导致 STATUS_ACCESS_VIOLATION。
        // 此测试验证这一假设。
        let base = Path::new(r"D:\19-Training\Rust\votex\models\tts\qwen3-tts-0.6b");

        println!("同时加载所有 4 个 ONNX 模型...");
        println!("  prefill.onnx.data 和 decode.onnx.data 完全相同（共享文件）");
        println!("  total RAM 估计: 420MB(cp) + 435MB(vocoder) + 1692MB(prefill) + 1692MB(decode) ≈ 4.3GB");

        let mut sessions = Vec::new();

        let models = [
            "code_predictor.onnx",
            "vocoder.onnx",
            "talker_prefill.onnx",
            "talker_decode.onnx",
        ];

        for (i, name) in models.iter().enumerate() {
            let path = base.join(name);
            print!("[{}/4] {name}... ", i + 1);
            let start = std::time::Instant::now();
            match OrtSessionFactory::create_raw_with_ep(&path, ExecutionProvider::Cpu) {
                Ok(sess) => {
                    sessions.push(sess);
                    let elapsed = start.elapsed();
                    println!("OK ({:.1}s)", elapsed.as_secs_f64());
                }
                Err(e) => {
                    let elapsed = start.elapsed();
                    println!("FAILED ({:.1}s): {e}", elapsed.as_secs_f64());
                    panic!("加载到第 {} 个模型 {name} 时崩溃: {e}", i + 1);
                }
            }
        }

        println!("全部 4 个模型同时加载成功！共 {} 个 session", sessions.len());
    }

    #[test]
    fn test_synthesize() {
        // 初始化 tracing subscriber 以捕获日志
        let _ = tracing_subscriber::fmt()
            .with_env_filter("info")
            .with_test_writer()
            .try_init();

        // 测试环境无 DirectML 支持，强制 CPU 避免崩溃
        OrtSessionFactory::set_global_ep(ExecutionProvider::Cpu);

        let mut provider = Qwen3TtsProvider::new();
        let model = Model::new(
            ModelId::new("qwen3-tts"),
            "Qwen3-TTS",
            ModelKind::Tts,
            EngineKind::Qwen3Tts,
        );
        
        println!("[test_synthesize] 开始加载模型...");
        provider.load(&model).expect("加载模型失败");
        println!("[test_synthesize] 模型加载成功");
        
        // 使用 0.6B 预设 speaker 音色（不生成 instruct tokens，与 Python 参考 `instruct=None` 对齐）
        let voice = VoiceId::new("serena", "Serena（自然女声）", EngineKind::Qwen3Tts);
        let params = TtsParams {
            engine: EngineKind::Qwen3Tts,
            voice: voice.clone(),
            speed: votex_domain::tts::value_object::Speed::default_value(),
            pitch: votex_domain::tts::value_object::Pitch::default(),
            volume: votex_domain::tts::value_object::Volume::default(),
            segment_size: votex_domain::tts::value_object::SegmentSize::default(),
            segment_silence_ms: 300,
            crossfade_ms: 50,
            num_to_chinese: true,
            denoise: false,
            denoise_level: votex_domain::tts::value_object::DenoiseLevel::default(),
            emotion: None,
            dialect: None,
        };
        
        // 从共享测试语料文件读取合成文本
        let test_path = Path::new(r"D:\19-Training\Rust\votex\tmp\tts_test.txt");
        let test_text = std::fs::read_to_string(test_path)
            .expect("读取测试语料文件失败，请先创建 tmp/tts_test.txt");
        println!("[test_synthesize] 合成文本: '{}'", test_text.trim());

        // 读取码本文件路径（与合成方法中保存的路径一致）
        let codes_path = std::env::temp_dir().join("rust_tts_codes.txt");
        // 删除旧码本文件
        let _ = std::fs::remove_file(&codes_path);

        match provider.synthesize(test_text.trim(), &voice, &params) {
            Ok(audio) => {
                println!("✓ 合成成功！音频长度: {} 采样点, 采样率: {} Hz", 
                    audio.samples.len(), audio.sample_rate);
                assert!(audio.samples.len() > 0, "音频不能为空");
                assert_eq!(audio.sample_rate, 24000, "采样率应该是 24000");
                assert_eq!(audio.channels, 1, "应该是单声道");
                
                // 保存为 WAV 文件
                let output_path = Path::new(r"D:\19-Training\Rust\votex\tmp\rust_test_output.wav");
                if let Some(parent) = output_path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                save_wav_file(&output_path, &audio.samples, audio.sample_rate).expect("保存 WAV 失败");
                println!("✓ 音频已保存到: {:?}", output_path);
                
                // 输出码本对比信息
                if codes_path.exists() {
                    if let Ok(content) = std::fs::read_to_string(&codes_path) {
                        let lines: Vec<&str> = content.lines().collect();
                        println!("✓ 码本已保存到: {:?} ({} 帧)", codes_path, lines.len());
                        if !lines.is_empty() {
                            println!("  第一帧码本: {}", lines[0]);
                        }
                    }
                }
            }
            Err(e) => {
                println!("✗ 合成失败: {}", e);
                panic!("合成失败: {}", e);
            }
        }
    }
    
    /// 保存 WAV 文件
    fn save_wav_file(path: &std::path::Path, samples: &[f32], sample_rate: u32) -> Result<(), TtsError> {
        use std::io::Write;
        
        let mut file = std::fs::File::create(path)
            .map_err(|e| TtsError::SynthesisFailed(format!("创建文件失败: {}", e)))?;
        
        let data_size = (samples.len() * 2) as u32;
        
        // WAV 头
        file.write_all(b"RIFF").unwrap();
        file.write_all(&(36 + data_size).to_le_bytes()).unwrap();
        file.write_all(b"WAVE").unwrap();
        file.write_all(b"fmt ").unwrap();
        file.write_all(&16u32.to_le_bytes()).unwrap(); // fmt chunk size
        file.write_all(&1u16.to_le_bytes()).unwrap();  // PCM
        file.write_all(&1u16.to_le_bytes()).unwrap();  // mono
        file.write_all(&sample_rate.to_le_bytes()).unwrap();
        file.write_all(&(sample_rate * 2).to_le_bytes()).unwrap(); // byte rate
        file.write_all(&2u16.to_le_bytes()).unwrap();  // block align
        file.write_all(&16u16.to_le_bytes()).unwrap(); // bits per sample
        file.write_all(b"data").unwrap();
        file.write_all(&data_size.to_le_bytes()).unwrap();
        
        // PCM 数据
        for &sample in samples {
            let pcm = (sample * 32767.0).max(-32768.0).min(32767.0) as i16;
            file.write_all(&pcm.to_le_bytes()).unwrap();
        }
        
        Ok(())
    }
}
