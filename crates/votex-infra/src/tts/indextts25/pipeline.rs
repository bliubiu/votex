//! IndexTTS-2.5 合成流水线（pipeline.py / cfm_solver.py 忠实移植）
//!
//! # 数据流（与参考实现逐行对齐）
//!
//! ```text
//! ref_audio → (audio_22k, audio_16k)
//!   audio_16k → W2vBertFeatures → semantic_model → spk_cond_emb [1,Te,1024]
//!   spk_cond_emb + ilens → emo_vec → emovec [1,1280]
//!   audio_16k → campplus_fbank → campplus → style [1,192]
//!   style @ spk_proj_w.T + b → spk_cond [1,1280]
//!   conds = [(spk_cond+emovec); 0; 0]  → [1,3,1280]
//!   audio_22k → mel_spectrogram → ref_mel [1,80,P]
//!   spk_cond_emb + template(P) → length_regulator → prompt_condition [1,P,512]
//!
//! text → Frontend.prepare → 分段 token
//!   → GPT 循环 → codes [Tc]
//!   → codec_decode → s_infer [1,Tc',1024]
//!   → length_regulator(s_infer, ⌈Tc'·1.72⌉) → cond [1,Tt,512]
//!   → cat(prompt_condition, cond) [1,P+Tt,512]
//!   → CFM euler 25 步（CFG 0.7）→ mel [1,80,Tt]
//!   → bigvgan → wav [Tt·256] @ 22050
//! ```
//!
//! # 确定性
//!
//! - z 噪声可注入（`CfmOptions::z`）用于金标逐位对拍；
//!   生产路径 `standard_normal` 由 StdRng + Box-Muller 生成（与 numpy
//!   PCG64-ziggurat 不同流，统计对拍约定，见蓝本 §7.4）。
//! - GPT greedy 与参考实现逐 token 一致（#45 已验证）。

use std::io::Read as _;
use std::path::Path;

use anyhow::{bail, Context, Result};
use ndarray::{Array1, Array2, Array3, ArrayD, Axis};

use crate::shared::WorkspacePaths;

use super::dsp;
use super::engines::{
    bigvgan_run, campplus_run, codec_decode_run, cfm_run, emo_vec_run, length_regulator_run,
    semantic_model_run, IndexTts25Engines,
};
use super::features::W2vBertFeatures;
use super::gpt::generate_codes;
use super::sampler::GptParams;

/// 采样率（pipeline.py SAMPLING_RATE）
pub const SAMPLING_RATE: usize = 22050;
/// 参考音频最大秒数（pipeline.py MAX_REF_SECONDS）
pub const MAX_REF_SECONDS: usize = 15;
/// 分段间静音毫秒数
pub const INTERVAL_SILENCE_MS: usize = 200;
/// 目标长度系数（pipeline.py duration_factor 默认 1.0 时 1.72）
pub const TARGET_LEN_FACTOR: f64 = 1.72;

// ===================== spk_proj.npz =====================

/// `spk_proj.npz` 的两个成员（weight [1280,192]、bias [1280]）
pub struct SpkProj {
    pub weight: Vec<f32>,
    pub weight_shape: (usize, usize),
    pub bias: Vec<f32>,
}

impl SpkProj {
    pub fn load(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("打开 spk_proj.npz 失败: {}", path.display()))?;
        let mut archive = zip::ZipArchive::new(file)
            .with_context(|| format!("解析 npz(zip) 失败: {}", path.display()))?;
        let weight = read_npz_member_f32(&mut archive, "weight.npy")
            .with_context(|| "读取 weight.npy")?;
        let bias = read_npz_member_f32(&mut archive, "bias.npy").with_context(|| "读取 bias.npy")?;
        if bias.shape.len() != 1 {
            bail!("bias 形状异常: {:?}（期望 1 维）", bias.shape);
        }
        if weight.shape.len() != 2 {
            bail!("weight 形状异常: {:?}（期望 2 维）", weight.shape);
        }
        Ok(Self {
            weight: weight.data,
            weight_shape: (weight.shape[0], weight.shape[1]),
            bias: bias.data,
        })
    }
}

/// 从 npz（zip 容器）读取一个 .npy 成员（f32）
fn read_npz_member_f32<R: std::io::Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    name: &str,
) -> Result<NpyArray> {
    let mut entry = archive
        .by_name(name)
        .with_context(|| format!("npz 中不存在成员 {name}"))?;
    let mut bytes = Vec::with_capacity(entry.size() as usize);
    entry.read_to_end(&mut bytes)?;
    parse_npy_f32(&bytes)
}

struct NpyArray {
    shape: Vec<usize>,
    data: Vec<f32>,
}

/// 最小 .npy 解析（f32，little-endian，C 序）
fn parse_npy_f32(data: &[u8]) -> Result<NpyArray> {
    if data.len() < 10 || &data[0..6] != b"\x93NUMPY" {
        bail!("无效 NPY 魔数");
    }
    let ver = data[6];
    let (hdr_len, hdr_off) = match ver {
        1 => (u16::from_le_bytes([data[8], data[9]]) as usize, 10usize),
        2 | 3 => (
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize,
            12usize,
        ),
        v => bail!("不支持 NPY v{v}"),
    };
    let hdr = std::str::from_utf8(&data[hdr_off..hdr_off + hdr_len])?;
    if !hdr.contains("'<f4'") && !hdr.contains("'<f4,'") {
        bail!("仅支持 <f4 npy（头: {}）", hdr.trim());
    }
    let shape_part = hdr
        .split("'shape': (")
        .nth(1)
        .and_then(|s| s.split(')').next())
        .ok_or_else(|| anyhow::anyhow!("npy 头无 shape"))?;
    let shape: Vec<usize> = shape_part
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.parse().ok())
        .collect();
    // npy 头按 64 字节对齐（v1.0 格式），数据起点 = 头结束后的对齐边界
    let data_start = hdr_off + hdr_len + (64 - ((hdr_off + hdr_len) % 64)) % 64;
    let n: usize = shape.iter().product();
    if data.len() < data_start + n * 4 {
        bail!("npy 数据不足: 期望 {} 字节", n * 4);
    }
    let floats = data[data_start..data_start + n * 4]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(NpyArray { shape, data: floats })
}

// ===================== SpeakerContext =====================

/// 说话人上下文（pipeline.py SpeakerContext）
pub struct SpeakerContext {
    /// [1,Te,1024]
    pub spk_cond_emb: ArrayD<f32>,
    /// [1,1280]
    pub emovec: ArrayD<f32>,
    /// [1,192]
    pub style: ArrayD<f32>,
    /// [1,3,1280] GPT 说话人条件
    pub conds: ArrayD<f32>,
    /// [1,P,512]
    pub prompt_condition: ArrayD<f32>,
    /// [1,80,P]
    pub ref_mel: ArrayD<f32>,
}

/// build_speaker（pipeline.py 逐行移植）
///
/// # 参数
/// - `engines`: 已加载引擎
/// - `audio_22k`: 参考音频 22050 Hz（≤15s，超出截断）
/// - `audio_16k`: 同一音频的 16k 版本（调用方负责重采样，与 `load_ref_audio`
///   语义一致：22k 信号经 sinc-hann 重采样到 16k）
/// - `spk_proj`: spk_proj.npz 内容
pub fn build_speaker(
    engines: &IndexTts25Engines,
    audio_22k: &[f32],
    audio_16k: &[f32],
    spk_proj: &SpkProj,
) -> Result<SpeakerContext> {
    let audio_22k = &audio_22k[..audio_22k.len().min(MAX_REF_SECONDS * SAMPLING_RATE)];
    let audio_16k = &audio_16k[..audio_16k.len().min(MAX_REF_SECONDS * 16000)];

    // 1. w2v-bert 特征 → 语义编码
    let features = W2vBertFeatures::new();
    let feats = features.extract(audio_16k)?;
    let te = feats.te;
    let input_features = Array3::from_shape_vec((1, te, 160), feats.input_features)
        .map_err(|e| anyhow::anyhow!("input_features 形状: {e}"))?
        .into_dyn();
    let attention_mask = Array2::from_shape_vec((1, te), feats.attention_mask)
        .map_err(|e| anyhow::anyhow!("attention_mask 形状: {e}"))?
        .into_dyn();
    let spk_cond_emb = engines
        .semantic_model
        .with_mut(|s| semantic_model_run(s, input_features, attention_mask))??;
    let te = spk_cond_emb.shape()[1];
    let ilens = Array1::from_vec(vec![te as i64]).into_dyn();

    // 2. 情感向量
    let emovec = engines
        .emo_vec
        .with_mut(|s| emo_vec_run(s, spk_cond_emb.clone(), ilens))??;

    // 3. 说话人风格（campplus + 整句均值归一化的 fbank）
    let fbank = dsp::campplus_fbank(audio_16k)?;
    let frames = fbank.len() / 80;
    let fbank_t = Array3::from_shape_vec((1, frames, 80), fbank)
        .expect("fbank 形状")
        .into_dyn();
    let style = engines.campplus.with_mut(|s| campplus_run(s, fbank_t))??;

    // 4. spk_cond = style @ W.T + b；conds = [(spk_cond+emovec); 0; 0]
    let (w_rows, w_cols) = spk_proj.weight_shape;
    let style_row = style
        .as_slice()
        .ok_or_else(|| anyhow::anyhow!("style 非连续"))?;
    let emovec_row = emovec
        .as_slice()
        .ok_or_else(|| anyhow::anyhow!("emovec 非连续"))?;
    let mut spk_cond = spk_proj.bias.clone();
    for (c, sc) in spk_cond.iter_mut().enumerate() {
        // weight 行主序 [1280,192]：第 c 行 = weight[c*192 .. (c+1)*192]
        let row = &spk_proj.weight[c * w_cols..(c + 1) * w_cols];
        let mut acc = 0.0f32;
        for (r, &sv) in style_row.iter().enumerate() {
            acc += sv * row[r];
        }
        *sc += acc + emovec_row[c];
    }
    let mut conds = vec![0.0f32; 3 * w_rows];
    conds[..w_rows].copy_from_slice(&spk_cond);
    let conds = Array3::from_shape_vec((1, 3, w_rows), conds)
        .expect("conds 形状")
        .into_dyn();

    // 5. 参考音频 mel + prompt_condition
    let ref_mel = dsp::mel_spectrogram(audio_22k)?;
    let p_frames = ref_mel.len() / 80;
    let ref_mel = Array3::from_shape_vec((1, 80, p_frames), ref_mel)
        .expect("ref_mel 形状")
        .into_dyn();
    let prompt_condition = engines
        .length_regulator
        .with_mut(|s| length_regulator_run(s, spk_cond_emb.clone(), p_frames))??;

    Ok(SpeakerContext {
        spk_cond_emb,
        emovec,
        style,
        conds,
        prompt_condition,
        ref_mel,
    })
}

// ===================== CFM euler 求解器 =====================

/// CFM 求解参数（cfm_solver.solve 签名）
#[derive(Debug, Clone)]
pub struct CfmOptions {
    pub n_timesteps: usize,
    pub cfg_rate: f64,
    /// 注入的初始噪声 [1,80,T]（金标对拍用）；None 时用 Box-Muller 生成
    pub z: Option<ArrayD<f32>>,
}

impl Default for CfmOptions {
    fn default() -> Self {
        Self {
            n_timesteps: 25,
            cfg_rate: 0.7,
            z: None,
        }
    }
}

/// Box-Muller 标准正态（生产路径 z 生成；numpy ziggurat 不同流，统计对拍）
fn standard_normal_f32(rng: &mut super::sampler::GptRng, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        out.push((r * (2.0 * std::f64::consts::PI * u2).cos()) as f32);
        if out.len() < n {
            out.push((r * (2.0 * std::f64::consts::PI * u2).sin()) as f32);
        }
    }
    out
}

/// CFM euler 求解 + 分类器无关引导（cfm_solver.solve 忠实移植）
///
/// # 参数
/// - `cat_condition`: [1,T,512]（prompt_condition 与 cond 拼接）
/// - `ref_mel`: [1,80,P]
/// - `style`: [1,192]
///
/// 返回生成 mel [1,80,T-P]。
pub fn cfm_solve(
    engines: &IndexTts25Engines,
    cat_condition: &ArrayD<f32>,
    ref_mel: &ArrayD<f32>,
    style: &ArrayD<f32>,
    opts: &CfmOptions,
    seed: u64,
) -> Result<ArrayD<f32>> {
    let shape = cat_condition.shape();
    if shape.len() != 3 || shape[0] != 1 || shape[2] != 512 {
        bail!("cat_condition 形状异常: {:?}（期望 [1,T,512]）", shape);
    }
    let t = shape[1];
    let prompt_len = *ref_mel.shape().last().ok_or_else(|| anyhow::anyhow!("ref_mel 空"))?;
    if ref_mel.shape()[1] != 80 {
        bail!("ref_mel 形状异常: {:?}（期望 [1,80,P]）", ref_mel.shape());
    }

    // z [1,80,T]
    let z: ArrayD<f32> = match &opts.z {
        Some(z) => {
            if z.shape() != [1, 80, t] {
                bail!("注入 z 形状 {:?} 与 [1,80,{t}] 不符", z.shape());
            }
            z.clone()
        }
        None => {
            let mut rng = super::sampler::GptRng::new(seed);
            let data = standard_normal_f32(&mut rng, 80 * t);
            Array3::from_shape_vec((1, 80, t), data)
                .expect("z 形状")
                .into_dyn()
        }
    };

    // prompt_x：前 P 帧为 ref_mel，其余 0；x = z 但前 P 帧置 0
    // （x 用 [80,t] 视角，batch 组装时复制进 [2,80,t]）
    let z3 = z
        .view()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| anyhow::anyhow!("z 维度: {e}"))?;
    let ref3 = ref_mel
        .view()
        .into_dimensionality::<ndarray::Ix3>()
        .map_err(|e| anyhow::anyhow!("ref_mel 维度: {e}"))?;
    let mut prompt_x = Array2::<f32>::zeros((80, t));
    let mut x = Array2::<f32>::zeros((80, t));
    for c in 0..80 {
        for f in 0..t {
            x[[c, f]] = z3[[0, c, f]];
        }
        for f in 0..prompt_len.min(t) {
            prompt_x[[c, f]] = ref3[[0, c, f]];
            x[[c, f]] = 0.0;
        }
    }
    let x_lens = Array1::from_vec(vec![t as i64]).into_dyn();

    // t_span = np.linspace(0, 1, n+1, dtype=f32)：内部 arange*step，末点强制为 1.0
    let n = opts.n_timesteps;
    let step = 1.0f32 / n as f32;
    let mut t_span: Vec<f32> = (0..n).map(|i| i as f32 * step).collect();
    t_span.push(1.0);

    let zeros_style = ndarray::ArrayD::<f32>::zeros(style.shape());
    let zeros_cond = ndarray::ArrayD::<f32>::zeros(cat_condition.shape());

    // CFG 双 batch：batch0 = (x, prompt_x)，batch1 = (x, zeros_prompt)
    let mut batch_x = Array3::<f32>::zeros((2, 80, t));
    let mut batch_px = Array3::<f32>::zeros((2, 80, t));
    batch_px
        .slice_mut(ndarray::s![0, .., ..])
        .assign(&prompt_x.view());
    // batch_px[1] 保持全零

    let mut t_cur = t_span[0];
    for &t_next in t_span.iter().skip(1) {
        let dt = t_next - t_cur;
        batch_x.slice_mut(ndarray::s![0, .., ..]).assign(&x.view());
        batch_x.slice_mut(ndarray::s![1, .., ..]).assign(&x.view());
        let t2 = Array1::from_vec(vec![t_cur, t_cur]).into_dyn();
        let stacked = engines
            .cfm
            .with_mut(|s| {
                cfm_run(
                    s,
                    batch_x.clone().into_dyn(),
                    batch_px.clone().into_dyn(),
                    x_lens.clone(),
                    t2,
                    ndarray::concatenate(Axis(0), &[style.view(), zeros_style.view()])
                        .expect("style 拼接")
                        .into_dyn(),
                    ndarray::concatenate(Axis(0), &[cat_condition.view(), zeros_cond.view()])
                        .expect("cond 拼接")
                        .into_dyn(),
                )
            })??;
        // dphi = stacked[0], dphi_cfg = stacked[1]（固定 Ix3，避免与 x 的维度不匹配）
        let stacked3 = stacked
            .view()
            .into_dimensionality::<ndarray::Ix3>()
            .map_err(|e| anyhow::anyhow!("cfm 输出维度: {e}"))?;
        let dphi = stacked3.index_axis(Axis(0), 0).to_owned();
        let dphi_cfg = stacked3.index_axis(Axis(0), 1).to_owned();
        // x += dt * ((1+cfg)·dphi − cfg·dphi_cfg)
        let cfg = opts.cfg_rate as f32;
        let a = (1.0 + cfg) * dt;
        let b = cfg * dt;
        x = &x + &(&dphi * a - &(&dphi_cfg * b));
        // x 的 prompt 区间置 0
        for c in 0..80 {
            for f in 0..prompt_len.min(t) {
                x[[c, f]] = 0.0;
            }
        }
        t_cur = t_next;
    }
    // 返回 x[:, :, prompt_len:]（恢复 [1,80,T-P] 维度）
    let out = x
        .slice(ndarray::s![.., prompt_len..])
        .to_owned()
        .insert_axis(ndarray::Axis(0));
    Ok(out.into_dyn())
}

// ===================== synthesize =====================

/// 合成参数
#[derive(Debug, Clone)]
pub struct SynthParams {
    pub greedy: bool,
    pub top_k: usize,
    pub top_p: f64,
    pub temperature: f64,
    pub repetition_penalty: f64,
    pub max_mel_tokens: usize,
    pub max_text_tokens_per_segment: usize,
    /// 每段 mel 帧数上限（GPT stop_token 之外的硬上限）
    pub duration_factor: f64,
    pub n_timesteps: usize,
    pub cfg_rate: f64,
    /// 总种子；None 时用系统熵
    pub seed: Option<u64>,
}

impl Default for SynthParams {
    fn default() -> Self {
        Self {
            greedy: false,
            top_k: 30,
            top_p: 0.8,
            temperature: 0.8,
            repetition_penalty: 10.0,
            max_mel_tokens: 1500,
            max_text_tokens_per_segment: 120,
            duration_factor: 1.0,
            n_timesteps: 25,
            cfg_rate: 0.7,
            seed: None,
        }
    }
}

/// 单段合成：codes → mel → wav f32 [-1,1]（synthesize 内层循环，便于对拍）
///
/// 参数与参考实现 `synthesize` 内层调用一一对应，不做参数对象化以便逐项对拍。
#[allow(clippy::too_many_arguments)]
pub fn synthesize_segment(
    engines: &IndexTts25Engines,
    spk: &SpeakerContext,
    text_ids: &[i64],
    lang_id: i64,
    params: &SynthParams,
    seed_gpt: u64,
    seed_cfm: u64,
    z: Option<ArrayD<f32>>,
) -> Result<Vec<f32>> {
    // 1. GPT 生成 mel codes
    let gpt_params = GptParams {
        greedy: params.greedy,
        top_k: params.top_k,
        top_p: params.top_p,
        temperature: params.temperature,
        repetition_penalty: params.repetition_penalty,
        max_mel_tokens: params.max_mel_tokens,
        stop_token: super::sampler::STOP_MEL_TOKEN,
    };
    let codes = generate_codes(engines, text_ids, &spk.conds, lang_id, &gpt_params, seed_gpt)?;
    if codes.is_empty() {
        return Ok(Vec::new());
    }

    // 2. codec 解码 + 长度调节
    let codes_t = Array2::from_shape_vec((1, codes.len()), codes)
        .expect("codes 形状")
        .into_dyn();
    let s_infer = engines
        .codec_decode
        .with_mut(|s| codec_decode_run(s, codes_t))??;
    let s_frames = s_infer.shape()[1];
    let target_len = ((s_frames as f64) * TARGET_LEN_FACTOR * params.duration_factor) as usize;
    let cond = engines
        .length_regulator
        .with_mut(|s| super::engines::length_regulator_run(s, s_infer, target_len))??;

    // 3. 拼接 prompt_condition + cond
    let cat_condition = ndarray::concatenate(Axis(1), &[spk.prompt_condition.view(), cond.view()])
        .map_err(|e| anyhow::anyhow!("cat_condition 拼接失败: {e}"))?
        .into_dyn();

    // 4. CFM + bigvgan
    let opts = CfmOptions {
        n_timesteps: params.n_timesteps,
        cfg_rate: params.cfg_rate,
        z,
    };
    let mel = cfm_solve(engines, &cat_condition, &spk.ref_mel, &spk.style, &opts, seed_cfm)?;
    let mel_len = *mel.shape().last().unwrap();
    let wav = engines.bigvgan.with_mut(|s| bigvgan_run(s, mel))??;
    let _ = mel_len;
    Ok(wav)
}

/// 完整合成：text → int16 pcm [N] @ 22050 Hz
///
/// 分段、分段间静音、int16 转换均与 pipeline.py.synthesize 一致。
pub fn synthesize(
    engines: &IndexTts25Engines,
    frontend: &super::frontend::Frontend,
    spk: &SpeakerContext,
    text: &str,
    lang: &str,
    params: &SynthParams,
) -> Result<Vec<i16>> {
    synthesize_with_cap(engines, frontend, spk, text, lang, params, None).map(|(wav, _)| wav)
}

/// 带段数上限的合成（长文本 e2e 用，Task #51）：`max_segments` 只影响实际
/// 合成的段数，不改变分段与种子派生顺序（前 N 段与全量合成逐位一致）。
/// 返回 (pcm, 实际合成段数)。
pub fn synthesize_with_cap(
    engines: &IndexTts25Engines,
    frontend: &super::frontend::Frontend,
    spk: &SpeakerContext,
    text: &str,
    lang: &str,
    params: &SynthParams,
    max_segments: Option<usize>,
) -> Result<(Vec<i16>, usize)> {
    let segments = frontend.prepare(text, lang, params.max_text_tokens_per_segment)?;
    if segments.is_empty() {
        return Ok((Vec::new(), 0));
    }
    let lang_id = super::bpe::lang_id(lang) as i64;
    let take = max_segments.unwrap_or(segments.len()).min(segments.len());
    // master 种子：各段派生（Python 用 master.integers(0, 2^31-1)，此处统计等价）
    let mut master = super::sampler::GptRng::new(params.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42)
    }));

    let mut wavs: Vec<Vec<f32>> = Vec::new();
    for seg in segments.iter().take(take) {
        let seg_ids: Vec<i64> = seg.iter().map(|&v| v as i64).collect();
        let seed_gpt = master.next_f64() * ((1u64 << 31) - 1) as f64;
        let seed_cfm = master.next_f64() * ((1u64 << 31) - 1) as f64;
        let wav = synthesize_segment(
            engines,
            spk,
            &seg_ids,
            lang_id,
            params,
            seed_gpt as u64,
            seed_cfm as u64,
            None,
        )?;
        if !wav.is_empty() {
            wavs.push(wav);
        }
    }
    if wavs.is_empty() {
        return Ok((Vec::new(), 0));
    }
    // 段间静音
    let silence = vec![0.0f32; SAMPLING_RATE * INTERVAL_SILENCE_MS / 1000];
    let mut joined: Vec<f32> = Vec::new();
    for (i, w) in wavs.iter().enumerate() {
        if i > 0 {
            joined.extend_from_slice(&silence);
        }
        joined.extend_from_slice(w);
    }
    // clip → int16（numpy astype 语义：向零截断，不四舍五入）
    let pcm = joined
        .into_iter()
        .map(|v| (32767.0f32 * v).clamp(-32767.0, 32767.0) as i16)
        .collect();
    Ok((pcm, take))
}

/// 默认模型目录（registry 约定），供 Provider 加载使用
pub fn default_model_dir() -> std::path::PathBuf {
    WorkspacePaths::models_dir().join("tts").join("indextts25")
}

#[allow(clippy::too_many_lines)]
#[cfg(test)]
mod tests {
    use super::*;
    use super::super::engines::find_model_dir;

    fn load_spk_proj_from_hf_cache() -> Option<SpkProj> {
        let snapshots = std::env::var_os("USERPROFILE")?;
        let dir = std::path::PathBuf::from(snapshots)
            .join(".cache/huggingface/hub/models--yunfengwang--IndexTTS-2.5-onnx/snapshots");
        for entry in std::fs::read_dir(dir).ok()?.flatten() {
            let p = entry.path().join("spk_proj.npz");
            if p.is_file() {
                return SpkProj::load(&p).ok();
            }
        }
        None
    }

    #[test]
    fn npz解析_weight_bias形状与数值() {
        let Some(proj) = load_spk_proj_from_hf_cache() else {
            eprintln!("跳过：spk_proj.npz 不在本地");
            return;
        };
        assert_eq!(proj.weight_shape, (1280, 192));
        assert_eq!(proj.bias.len(), 1280);
        // Python: weight.ravel()[:3] = [0.12976073, 0.05247156, -0.06556923]
        assert!((proj.weight[0] - 0.12976073).abs() < 1e-6);
        assert!((proj.weight[1] - 0.05247156).abs() < 1e-6);
        assert!((proj.weight[2] - (-0.06556923)).abs() < 1e-6);
        // bias[:3] = [-0.17965049, 0.17451322, -0.09641936]
        assert!((proj.bias[0] - (-0.17965049)).abs() < 1e-6);
        assert!((proj.bias[1] - 0.17451322).abs() < 1e-6);
        assert!((proj.bias[2] - (-0.09641936)).abs() < 1e-6);
    }

    #[test]
    fn box_muller_统计量合理() {
        let mut rng = super::super::sampler::GptRng::new(7);
        let data = standard_normal_f32(&mut rng, 20000);
        let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / data.len() as f64;
        let var: f64 = data.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / data.len() as f64;
        assert!(mean.abs() < 0.05, "均值异常: {mean}");
        assert!((var - 1.0).abs() < 0.1, "方差异常: {var}");
    }

    #[test]
    fn cfm形状校验_错误输入被拒() {
        let engines = IndexTts25Engines::new();
        let cond = ndarray::Array3::<f32>::zeros((1, 10, 512)).into_dyn();
        let ref_mel = ndarray::Array3::<f32>::zeros((1, 80, 5)).into_dyn();
        let style = ndarray::Array2::<f32>::zeros((1, 192)).into_dyn();
        let err = cfm_solve(&engines, &cond, &ref_mel, &style, &CfmOptions::default(), 0);
        assert!(err.is_err(), "未加载引擎应报错");
        let bad = ndarray::Array3::<f32>::zeros((1, 10, 1024)).into_dyn();
        let err = cfm_solve(&engines, &bad, &ref_mel, &style, &CfmOptions::default(), 0);
        assert!(err.is_err(), "形状错误应被拒绝");
    }

    // ===================== slow-models 金标对拍（Task #46） =====================

    /// 三个对拍测试共享一次引擎加载（11.7GB），OnceLock 阻塞并发首调者。
    fn shared_engines() -> &'static IndexTts25Engines {
        static ENGINES: std::sync::OnceLock<IndexTts25Engines> = std::sync::OnceLock::new();
        ENGINES.get_or_init(|| {
            let engines = IndexTts25Engines::new();
            let dir = crate::tts::indextts25::engines::find_model_dir()
                .expect("未找到 IndexTTS-2.5 模型目录（models/tts/indextts25 或 HF 缓存）");
            engines
                .load(&dir, 4)
                .expect("引擎加载失败");
            engines
        })
    }

    fn refs_dir() -> std::path::PathBuf {
        WorkspacePaths::workspace_root().join("tmp").join("indextts25_refs")
    }

    fn read_f32_file(name: &str) -> Vec<f32> {
        let path = refs_dir().join(name);
        let data = std::fs::read(&path).unwrap_or_else(|e| panic!("金标不存在 {:?}: {}", path, e));
        data.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    /// 读金标 npy → (shape, data)，复用本模块的 parse_npy_f32
    fn read_npy(name: &str) -> (Vec<usize>, Vec<f32>) {
        let path = refs_dir().join(name);
        let data = std::fs::read(&path).unwrap_or_else(|e| panic!("金标不存在 {:?}: {}", path, e));
        let arr = parse_npy_f32(&data).unwrap_or_else(|e| panic!("{name}: {e}"));
        (arr.shape, arr.data)
    }

    fn max_diff(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "长度不一致");
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    fn assert_close(name: &str, got: &[f32], want: &[f32], tol: f32) {
        let d = max_diff(got, want);
        println!("{name}: max_abs_diff = {d:.3e}（容差 {tol}）");
        assert!(d < tol, "{name} 偏差 {d:.3e} 超过容差 {tol}");
    }

    /// build_speaker 六产物逐位对拍（3s chirp 确定性参考音频）。
    #[test]
    #[cfg_attr(
        not(feature = "slow-models"),
        ignore = "加载 11.7GB 引擎，--features slow-models 启用"
    )]
    fn build_speaker_与python金标对拍() {
        let engines = shared_engines();
        let audio22 = read_f32_file("spk_wave.f32");
        let audio16 = read_f32_file("spk_wave16.f32");
        let proj_path = crate::tts::indextts25::engines::find_model_dir()
            .expect("模型目录")
            .join("spk_proj.npz");
        let proj = SpkProj::load(&proj_path).expect("spk_proj.npz");

        let spk = build_speaker(engines, &audio22, &audio16, &proj).expect("build_speaker");

        let (wshape, want) = read_npy("spk_cond_emb.npy");
        assert_eq!(spk.spk_cond_emb.shape(), wshape.as_slice());
        assert_close(
            "spk_cond_emb",
            spk.spk_cond_emb.as_slice().unwrap(),
            &want,
            2e-3,
        );

        let (_, want) = read_npy("spk_emovec.npy");
        assert_close("emovec", spk.emovec.as_slice().unwrap(), &want, 1e-3);

        let (_, want) = read_npy("spk_style.npy");
        // campplus fbank 在低能量 bin 有 kissfft 固有 f32 舍入噪声（见 dsp 测试 5e-4 容差），
        // 经 campplus DNN 传播放大到 ~1e-3 量级，属统计等价而非 bit-exact 链路
        assert_close("style", spk.style.as_slice().unwrap(), &want, 5e-3);

        let (_, want) = read_npy("spk_conds.npy");
        // conds = (style @ W.T + b) + emovec：继承 style 的 fbank 噪声经矩阵乘传播（~2.5e-3）
        assert_close("conds", spk.conds.as_slice().unwrap(), &want, 5e-3);

        let (wshape, want) = read_npy("spk_prompt_condition.npy");
        assert_eq!(spk.prompt_condition.shape(), wshape.as_slice());
        assert_close(
            "prompt_condition",
            spk.prompt_condition.as_slice().unwrap(),
            &want,
            1e-3,
        );

        let (wshape, want) = read_npy("spk_ref_mel.npy");
        assert_eq!(spk.ref_mel.shape(), wshape.as_slice());
        // log-mel 幅值 ~±10，FFT f32 舍入噪声 ~2.4e-4（与 kaldi fbank 同源，见 dsp 测试）
        assert_close("ref_mel", spk.ref_mel.as_slice().unwrap(), &want, 5e-4);
    }

    /// cfm_solve euler+CFG 25 步对拍（确定性输入 + 注入 z，linspace f32 语义一致）。
    #[test]
    #[cfg_attr(
        not(feature = "slow-models"),
        ignore = "加载 11.7GB 引擎，--features slow-models 启用"
    )]
    fn cfm_solve_与python金标对拍() {
        let engines = shared_engines();
        let (scond, cond) = read_npy("cfm_in_cat_condition.npy");
        let (_, style) = read_npy("cfm_in_style.npy");
        let (_, ref_mel) = read_npy("cfm_in_ref_mel.npy");
        let (_, z) = read_npy("cfm_in_z.npy");
        let (wshape, want) = read_npy("cfm_out_mel.npy");
        assert_eq!(scond, vec![1, 100, 512]);

        let cond = Array3::from_shape_vec((1, 100, 512), cond)
            .expect("cond 形状")
            .into_dyn();
        let style = Array2::from_shape_vec((1, 192), style)
            .expect("style 形状")
            .into_dyn();
        let ref_mel = Array3::from_shape_vec((1, 80, 20), ref_mel)
            .expect("ref_mel 形状")
            .into_dyn();
        let z = Array3::from_shape_vec((1, 80, 100), z)
            .expect("z 形状")
            .into_dyn();

        let opts = CfmOptions {
            n_timesteps: 25,
            cfg_rate: 0.7,
            z: Some(z),
        };
        let got = cfm_solve(engines, &cond, &ref_mel, &style, &opts, 0).expect("cfm_solve");
        assert_eq!(got.shape(), wshape.as_slice(), "输出形状不一致");
        assert_close("cfm_out_mel", got.as_slice().unwrap(), &want, 5e-4);
    }

    /// bigvgan 声码对拍（输入用 Python 侧 cfm 输出 mel，纯图推理应逐位接近）。
    #[test]
    #[cfg_attr(
        not(feature = "slow-models"),
        ignore = "加载 11.7GB 引擎，--features slow-models 启用"
    )]
    fn bigvgan_与python金标对拍() {
        let engines = shared_engines();
        let (_, mel) = read_npy("cfm_out_mel.npy");
        let want = read_f32_file("bigvgan_out_wav.f32");

        let mel = Array3::from_shape_vec((1, 80, mel.len() / 80), mel)
            .expect("mel 形状")
            .into_dyn();
        let got = engines
            .bigvgan
            .with_mut(|s| bigvgan_run(s, mel))
            .expect("bigvgan 加载")
            .expect("bigvgan 运行");

        // 振幅 [-1,1]，容差取绝对值
        assert_eq!(got.len(), want.len(), "wav 长度不一致");
        assert_close("bigvgan_out_wav", &got, &want, 1e-4);
    }

    // ===================== 端到端验收（Task #48，蓝本 §7.5） =====================

    /// 端到端：粤语一句话从文本到 wav 的全链路合成（真实 11.7GB 模型）。
    ///
    /// 验收点：
    /// 1. 全链路不 panic/不报错（8 session + tiktoken + spk_proj）；
    /// 2. 输出非空、时长与文本量级匹配（≥1.5s）；
    /// 3. 幅值合法（|x| ≤ 1）；
    /// 4. wav 落盘 `tmp/indextts25_e2e_yue.wav` 供人工试听。
    ///
    /// 与 PoC（uvx）的输出不做逐位对拍：numpy/StdRng RNG 流不同（蓝本 §7.4 约定
    /// 采样路径只对拍统计分布）；确定性逐位对拍由 #43~#46 的分级金标承担。
    #[test]
    #[cfg_attr(
        not(feature = "slow-models"),
        ignore = "加载 11.7GB 引擎，--features slow-models 启用"
    )]
    fn 端到端_粤语合成落盘试听() {
        let engines = shared_engines();
        let vocab = super::super::frontend::VOCAB_FILE_NAME;
        let frontend = super::super::frontend::Frontend::new(&find_model_dir().unwrap().join(vocab), false)
            .expect("tiktoken 加载");

        // 参考音频：PoC 同源（tmp/cosyvoice_official_zhprompt.wav → prompts/default.wav），
        // 无 prompts 时退回 tmp 下 PoC 参考音频
        let prompt = find_prompt_wav();

        let audio_22k_full = crate::audio::wav::read_wav(&prompt).expect("读取参考 wav");
        let mono: Vec<f32> = if audio_22k_full.channels <= 1 {
            audio_22k_full.samples
        } else {
            audio_22k_full
                .samples
                .chunks(audio_22k_full.channels as usize)
                .map(|c| c.iter().sum::<f32>() / c.len() as f32)
                .collect()
        };
        let sr = audio_22k_full.sample_rate as usize;
        let audio_22k = if sr == SAMPLING_RATE {
            mono.clone()
        } else {
            super::dsp::resample_sinc_hann(&mono, sr, SAMPLING_RATE, 6, 0.99).expect("重采样 22k")
        };
        let audio_16k = if sr == 16000 {
            mono.clone()
        } else {
            super::dsp::resample_sinc_hann(&audio_22k, SAMPLING_RATE, 16000, 6, 0.99)
                .expect("重采样 16k")
        };

        let proj_path = find_model_dir().unwrap().join("spk_proj.npz");
        let spk_proj = SpkProj::load(&proj_path).expect("spk_proj 加载");
        let spk = build_speaker(&engines, &audio_22k, &audio_16k, &spk_proj).expect("build_speaker");

        // 粤语验收句（含数字混排走 zhen？——保持纯 yue 文本，无数字）
        let text = "今日天气几好，我哋一齐去饮茶啦。";
        let params = SynthParams {
            greedy: true, // 确定性路径：grepy 避开 RNG 流差异，保证测试可复现
            seed: Some(42),
            ..SynthParams::default()
        };
        let wav = synthesize(&engines, &frontend, &spk, text, "yue", &params).expect("合成失败");

        assert!(!wav.is_empty(), "合成结果为空");
        let duration_s = wav.len() as f64 / SAMPLING_RATE as f64;
        println!("端到端输出: {} 样本, {duration_s:.2}s", wav.len());
        assert!(duration_s >= 1.5, "时长异常: {duration_s:.2}s");
        // i16 幅值恒 <= i16::MAX（合成前已 clamp ±32767），改验信号非静音
        assert!(wav.iter().any(|v| v.abs() > 100), "音频疑似静音");

        // 落盘供试听
        let out_dir = WorkspacePaths::workspace_root().join("tmp");
        std::fs::create_dir_all(&out_dir).ok();
        let audio = votex_domain::shared::value_object::AudioData {
            samples: wav.iter().map(|&v| v as f32 / 32768.0).collect(),
            sample_rate: SAMPLING_RATE as u32,
            channels: 1,
        };
        let out_path = out_dir.join("indextts25_e2e_yue.wav");
        crate::audio::wav::WavWriter::write(&audio, &out_path).expect("写 wav");
        println!("试听文件: {:?}", out_path);
    }

    /// 查找参考音频：prompts/default.wav 优先，退回 tmp 的 PoC 参考音频
    fn find_prompt_wav() -> std::path::PathBuf {
        let root = WorkspacePaths::workspace_root();
        let candidates = [
            root.join("models/tts/indextts25/prompts/default.wav"),
            root.join("tmp/cosyvoice_official_zhprompt.wav"),
        ];
        candidates
            .into_iter()
            .find(|p| p.is_file())
            .expect("未找到参考音频（prompts/default.wav 或 tmp/cosyvoice_official_zhprompt.wav）")
    }

    /// 端到端共用：参考音频 → 重采样 22k/16k → build_speaker。
    fn prepare_test_speaker(engines: &IndexTts25Engines) -> SpeakerContext {
        let prompt = find_prompt_wav();
        let audio_22k_full = crate::audio::wav::read_wav(&prompt).expect("读取参考 wav");
        let mono: Vec<f32> = if audio_22k_full.channels <= 1 {
            audio_22k_full.samples
        } else {
            audio_22k_full
                .samples
                .chunks(audio_22k_full.channels as usize)
                .map(|c| c.iter().sum::<f32>() / c.len() as f32)
                .collect()
        };
        let sr = audio_22k_full.sample_rate as usize;
        let audio_22k = if sr == SAMPLING_RATE {
            mono.clone()
        } else {
            super::dsp::resample_sinc_hann(&mono, sr, SAMPLING_RATE, 6, 0.99).expect("重采样 22k")
        };
        let audio_16k = if sr == 16000 {
            mono.clone()
        } else {
            super::dsp::resample_sinc_hann(&audio_22k, SAMPLING_RATE, 16000, 6, 0.99)
                .expect("重采样 16k")
        };
        let proj_path = find_model_dir().unwrap().join("spk_proj.npz");
        let spk_proj = SpkProj::load(&proj_path).expect("spk_proj 加载");
        build_speaker(engines, &audio_22k, &audio_16k, &spk_proj).expect("build_speaker")
    }

    /// 端到端落盘辅助。
    fn write_e2e_wav(wav: &[i16], filename: &str) -> std::path::PathBuf {
        let out_path = WorkspacePaths::workspace_root().join("tmp").join(filename);
        std::fs::create_dir_all(out_path.parent().unwrap()).ok();
        let audio = votex_domain::shared::value_object::AudioData {
            samples: wav.iter().map(|&v| v as f32 / 32768.0).collect(),
            sample_rate: SAMPLING_RATE as u32,
            channels: 1,
        };
        crate::audio::wav::WavWriter::write(&audio, &out_path).expect("写 wav");
        out_path
    }

    /// 端到端：长文本粤语多段合成（Task #51，tmp/e2e_ch1.txt 三章小说全文）。
    ///
    /// 验证点：多段切分（预算 120 token）、段间静音拼接、长文本无 panic、
    /// 每段时长量级正常。环境变量 `VOTEX_E2E_MAX_SEGMENTS` 控制实际合成段数
    /// （默认 8；全量 ~150 段按 RTF 需数小时，不进常规验收）。
    #[test]
    #[cfg_attr(
        not(feature = "slow-models"),
        ignore = "加载 11.7GB 引擎，--features slow-models 启用"
    )]
    fn 端到端_长文本粤语多段合成() {
        let engines = shared_engines();
        let vocab = super::super::frontend::VOCAB_FILE_NAME;
        let frontend = super::super::frontend::Frontend::new(
            &find_model_dir().unwrap().join(vocab),
            false,
        )
        .expect("tiktoken 加载");
        let root = WorkspacePaths::workspace_root();
        let text =
            std::fs::read_to_string(root.join("tmp/e2e_ch1.txt")).expect("读取长文本 tmp/e2e_ch1.txt");
        let params = SynthParams {
            greedy: true,
            seed: Some(42),
            ..SynthParams::default()
        };

        let total = frontend
            .prepare(&text, "yue", params.max_text_tokens_per_segment)
            .expect("分段")
            .len();
        println!("长文本共 {total} 段");
        assert!(total > 1, "长文本必须产生多段");

        let cap: usize = std::env::var("VOTEX_E2E_MAX_SEGMENTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let spk = prepare_test_speaker(engines);
        let t0 = std::time::Instant::now();
        let (wav, n) =
            synthesize_with_cap(engines, &frontend, &spk, &text, "yue", &params, Some(cap))
                .expect("合成失败");
        let wall = t0.elapsed().as_secs_f64();
        assert_eq!(n, cap.min(total), "实际合成段数不符");
        assert!(!wav.is_empty(), "合成结果为空");
        let duration_s = wav.len() as f64 / SAMPLING_RATE as f64;
        println!(
            "合成 {n} 段：音频 {duration_s:.1}s，墙钟 {wall:.1}s（RTF 墙钟/音频 = {:.1}）",
            wall / duration_s
        );
        assert!(duration_s >= n as f64, "时长量级异常: {duration_s:.2}s");
        // i16 幅值恒 <= i16::MAX（合成前已 clamp ±32767），改验信号非静音
        assert!(wav.iter().any(|v| v.abs() > 100), "音频疑似静音");

        let out_path = write_e2e_wav(&wav, "indextts25_e2e_yue_ch1.wav");
        println!("试听文件: {out_path:?}");
    }
}
