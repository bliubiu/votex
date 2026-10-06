//! IndexTTS-2.5 ONNX 引擎会话封装（engine.py 忠实移植）
//!
//! 8 个 session 的加载与单次推理，输入/输出名与参考实现逐字一致：
//!
//! | 引擎 | 输入 | 输出 |
//! | :--- | :--- | :--- |
//! | `semantic_model` | `input_features [1,Te,160] f32`、`attention_mask [1,Te] i64` | `[1,Te,1024]` |
//! | `emo_vec` | `feat [1,Te,1024]`、`ilens [1] i64` | `[1,1280]` |
//! | `campplus` | `fbank [1,T,80]` | `[1,192]` |
//! | `length_regulator` | `x [1,T,1024]`、`template [1,T_out,1]` | `[1,T_out,512]` |
//! | `codec_decode` | `codes [1,T] i64` | `[1,T',1024]` |
//! | `bigvgan` | `mel [1,80,T]` | `[1,1,T*256]`（截断到 `T*256`） |
//! | `gpt_prefill` | `text_ids [1,T] i64`、`conds [1,3,1280]`、`lang_id [1] i64` | logits + 48 个 KV |
//! | `gpt_step` | `token [1,1] i64`、`pos_idx [1] i64`、48 个 `past_kv_{i}_{k,v}` | logits + 48 个 KV |
//! | `cfm` | `x`、`prompt_x`、`x_lens i64`、`t`、`style`、`cond` | mel 增量 |
//!
//! # 与参考实现的差异
//!
//! - KV state 的生命周期：Python 把 `outs[1:]` 写回 list 原地替换；
//!   Rust 以 `&mut Vec<ArrayD<f32>>` 语义等价替换（每步重新喂入，
//!   存在一次 KV 拷贝，正确性优先，#46 接入时如构成瓶颈再优化为持久 Value）。
//! - 设备选择：仅 CPU（参考实现同样确认 int8/量化不可行，fp32 bit-exact）。

use std::path::Path;

use anyhow::{bail, Result};
use ndarray::{Array1, ArrayD};
use ort::session::Session;
use ort::value::Value;

use crate::shared::{EngineState, OrtSessionFactory};

/// GPT Transformer 层数（KV state = 24 层 × {k, v} = 48 个张量）
pub const N_LAYERS: usize = 24;

/// KV state 张量个数
pub const N_KV_TENSORS: usize = N_LAYERS * 2;

/// IndexTTS-2.5 全部 ONNX 引擎
///
/// 各字段均为 [`EngineState`]（`&self` 加载契约，见 P2-17）。
pub struct IndexTts25Engines {
    pub semantic_model: EngineState<Session>,
    pub emo_vec: EngineState<Session>,
    pub campplus: EngineState<Session>,
    pub length_regulator: EngineState<Session>,
    pub codec_decode: EngineState<Session>,
    pub bigvgan: EngineState<Session>,
    pub gpt_prefill: EngineState<Session>,
    pub gpt_step: EngineState<Session>,
    pub cfm: EngineState<Session>,
}

impl Default for IndexTts25Engines {
    fn default() -> Self {
        Self::new()
    }
}

impl IndexTts25Engines {
    pub fn new() -> Self {
        Self {
            semantic_model: EngineState::new(),
            emo_vec: EngineState::new(),
            campplus: EngineState::new(),
            length_regulator: EngineState::new(),
            codec_decode: EngineState::new(),
            bigvgan: EngineState::new(),
            gpt_prefill: EngineState::new(),
            gpt_step: EngineState::new(),
            cfm: EngineState::new(),
        }
    }

    /// 加载全部 8 个 session
    ///
    /// 目录布局与 engine.py 一致：子目录引擎 `{name}/{name}.onnx`，
    /// 平铺引擎 `{name}.onnx`（外部数据 `.onnx.data` 由 ORT 自动查找同目录文件）。
    ///
    /// # 参数
    /// - `model_dir`: 模型根目录（registry 约定 `models/tts/indextts25/`）
    /// - `threads`: 每个 session 的 intra-op 线程数
    ///
    /// # 错误
    /// 任一模型缺失或加载失败即整体失败（已加载的 session 保留，可 unload）。
    pub fn load(&self, model_dir: &Path, threads: usize) -> Result<()> {
        let sub = |name: &str| model_dir.join(name).join(format!("{name}.onnx"));
        let flat = |name: &str| model_dir.join(format!("{name}.onnx"));

        macro_rules! load {
            ($field:ident, $path:expr) => {{
                let path: std::path::PathBuf = $path;
                let session = OrtSessionFactory::create_with_threads(&path, threads)
                    .map_err(|e| anyhow::anyhow!("加载 {} 失败: {}", path.display(), e))?;
                self.$field.load(session);
            }};
        }

        load!(semantic_model, sub("semantic_model"));
        load!(emo_vec, flat("emo_vec"));
        load!(campplus, flat("campplus"));
        load!(length_regulator, flat("length_regulator"));
        load!(codec_decode, flat("semantic_codec_decode"));
        load!(bigvgan, flat("bigvgan"));
        load!(gpt_prefill, sub("gpt_prefill"));
        load!(gpt_step, sub("gpt_step"));
        load!(cfm, sub("cfm_estimator"));
        Ok(())
    }

    /// 卸载全部 session
    pub fn unload_all(&self) {
        self.semantic_model.unload();
        self.emo_vec.unload();
        self.campplus.unload();
        self.length_regulator.unload();
        self.codec_decode.unload();
        self.bigvgan.unload();
        self.gpt_prefill.unload();
        self.gpt_step.unload();
        self.cfm.unload();
    }

    /// 全部引擎是否已加载
    pub fn is_loaded(&self) -> bool {
        self.semantic_model.is_loaded()
            && self.emo_vec.is_loaded()
            && self.campplus.is_loaded()
            && self.length_regulator.is_loaded()
            && self.codec_decode.is_loaded()
            && self.bigvgan.is_loaded()
            && self.gpt_prefill.is_loaded()
            && self.gpt_step.is_loaded()
            && self.cfm.is_loaded()
    }
}

// ===================== 单引擎推理（自由函数，供 pipeline 在 with 闭包内调用） =====================

/// 提取第 `idx` 个输出为 owned f32 张量
fn out_f32(outputs: &ort::session::SessionOutputs<'_>, idx: usize) -> Result<ArrayD<f32>> {
    let view = outputs[idx]
        .try_extract_array::<f32>()
        .map_err(|e| anyhow::anyhow!("提取输出[{idx}] (f32) 失败: {}", e))?;
    Ok(view.to_owned())
}

/// w2v-bert 语义编码：`input_features [1,Te,160]` + `attention_mask [1,Te]` → `[1,Te,1024]`
pub fn semantic_model_run(
    session: &mut Session,
    input_features: ArrayD<f32>,
    attention_mask: ArrayD<i64>,
) -> Result<ArrayD<f32>> {
    let outputs = session.run(ort::inputs![
        "input_features" => Value::from_array(input_features)
            .map_err(|e| anyhow::anyhow!("构造 input_features 失败: {}", e))?,
        "attention_mask" => Value::from_array(attention_mask)
            .map_err(|e| anyhow::anyhow!("构造 attention_mask 失败: {}", e))?,
    ])?;
    out_f32(&outputs, 0)
}

/// 情感向量：`feat [1,Te,1024]` + `ilens [1]` → `[1,1280]`
pub fn emo_vec_run(session: &mut Session, feat: ArrayD<f32>, ilens: ArrayD<i64>) -> Result<ArrayD<f32>> {
    let outputs = session.run(ort::inputs![
        "feat" => Value::from_array(feat).map_err(|e| anyhow::anyhow!("构造 feat 失败: {}", e))?,
        "ilens" => Value::from_array(ilens).map_err(|e| anyhow::anyhow!("构造 ilens 失败: {}", e))?,
    ])?;
    out_f32(&outputs, 0)
}

/// 说话人风格向量：`fbank [1,T,80]` → `[1,192]`
pub fn campplus_run(session: &mut Session, fbank: ArrayD<f32>) -> Result<ArrayD<f32>> {
    let outputs = session.run(ort::inputs![
        "fbank" => Value::from_array(fbank).map_err(|e| anyhow::anyhow!("构造 fbank 失败: {}", e))?,
    ])?;
    out_f32(&outputs, 0)
}

/// 长度调节：`x [1,T,1024]` + 全零模板 `[1,T_out,1]` → `[1,T_out,512]`
///
/// 模板内容全零（engine.py `np.zeros`），仅提供目标长度。
pub fn length_regulator_run(session: &mut Session, x: ArrayD<f32>, t_out: usize) -> Result<ArrayD<f32>> {
    let template = ndarray::Array3::<f32>::zeros((1, t_out, 1)).into_dyn();
    let outputs = session.run(ort::inputs![
        "x" => Value::from_array(x).map_err(|e| anyhow::anyhow!("构造 x 失败: {}", e))?,
        "template" => Value::from_array(template)
            .map_err(|e| anyhow::anyhow!("构造 template 失败: {}", e))?,
    ])?;
    out_f32(&outputs, 0)
}

/// 语义 codec 解码：`codes [1,T] i64` → `[1,T',1024]`
pub fn codec_decode_run(session: &mut Session, codes: ArrayD<i64>) -> Result<ArrayD<f32>> {
    let outputs = session.run(ort::inputs![
        "codes" => Value::from_array(codes).map_err(|e| anyhow::anyhow!("构造 codes 失败: {}", e))?,
    ])?;
    out_f32(&outputs, 0)
}

/// BigVGAN 声码：`mel [1,80,T]` → 波形 `Vec<f32>`（截断到 `T*256` 样本）
pub fn bigvgan_run(session: &mut Session, mel: ArrayD<f32>) -> Result<Vec<f32>> {
    let t = mel.shape()[mel.ndim() - 1];
    let outputs = session.run(ort::inputs![
        "mel" => Value::from_array(mel).map_err(|e| anyhow::anyhow!("构造 mel 失败: {}", e))?,
    ])?;
    let wav = out_f32(&outputs, 0)?;
    let n = t * 256;
    Ok(wav.iter().copied().take(n).collect())
}

/// CFM 流匹配估计器：euler 单步
///
/// 输入名与 CfmEstimator.run 一致：`x`、`prompt_x`、`x_lens (i64)`、`t`、`style`、`cond`。
#[allow(clippy::too_many_arguments)]
pub fn cfm_run(
    session: &mut Session,
    x: ArrayD<f32>,
    prompt_x: ArrayD<f32>,
    x_lens: ArrayD<i64>,
    t: ArrayD<f32>,
    style: ArrayD<f32>,
    cond: ArrayD<f32>,
) -> Result<ArrayD<f32>> {
    let outputs = session.run(ort::inputs![
        "x" => Value::from_array(x).map_err(|e| anyhow::anyhow!("构造 x 失败: {}", e))?,
        "prompt_x" => Value::from_array(prompt_x)
            .map_err(|e| anyhow::anyhow!("构造 prompt_x 失败: {}", e))?,
        "x_lens" => Value::from_array(x_lens).map_err(|e| anyhow::anyhow!("构造 x_lens 失败: {}", e))?,
        "t" => Value::from_array(t).map_err(|e| anyhow::anyhow!("构造 t 失败: {}", e))?,
        "style" => Value::from_array(style).map_err(|e| anyhow::anyhow!("构造 style 失败: {}", e))?,
        "cond" => Value::from_array(cond).map_err(|e| anyhow::anyhow!("构造 cond 失败: {}", e))?,
    ])?;
    out_f32(&outputs, 0)
}

/// GPT prefill：文本条件 + 说话人条件 → 首 token logits + KV state
///
/// 返回 `(logits[8194], state[48])`，对应 engine.py 的
/// `outs[0][0]` 与 `list(outs[1:])`。
pub fn gpt_prefill_run(
    session: &mut Session,
    text_ids: ArrayD<i64>,
    conds: ArrayD<f32>,
    lang_id: i64,
) -> Result<(Vec<f32>, Vec<ArrayD<f32>>)> {
    let lang_id = Array1::from_vec(vec![lang_id]).into_dyn();
    let outputs = session.run(ort::inputs![
        "text_ids" => Value::from_array(text_ids)
            .map_err(|e| anyhow::anyhow!("构造 text_ids 失败: {}", e))?,
        "conds" => Value::from_array(conds).map_err(|e| anyhow::anyhow!("构造 conds 失败: {}", e))?,
        "lang_id" => Value::from_array(lang_id)
            .map_err(|e| anyhow::anyhow!("构造 lang_id 失败: {}", e))?,
    ])?;
    let logits = out_f32(&outputs, 0)?;
    let state = extract_state(&outputs)?;
    Ok((logits.iter().copied().collect(), state))
}

/// GPT step：单 token 前进 + KV 更新（`pos_idx = i + 2` 的 quirk 由调用方传入）
///
/// `state` 被 `outs[1:]` 整体替换（engine.py `state[:] = outs[1:]`）。
pub fn gpt_step_run(
    session: &mut Session,
    state: &mut Vec<ArrayD<f32>>,
    token: i64,
    pos_idx: i64,
) -> Result<Vec<f32>> {
    if state.len() != N_KV_TENSORS {
        bail!("KV state 数量错误: 期望 {}，实际 {}", N_KV_TENSORS, state.len());
    }
    let token_t = ndarray::Array2::<i64>::from_shape_vec((1, 1), vec![token])
        .expect("token 形状固定 [1,1]")
        .into_dyn();
    let pos_t = Array1::from_vec(vec![pos_idx]).into_dyn();

    let mut feeds = ort::inputs![
        "token" => Value::from_array(token_t).map_err(|e| anyhow::anyhow!("构造 token 失败: {}", e))?,
        "pos_idx" => Value::from_array(pos_t)
            .map_err(|e| anyhow::anyhow!("构造 pos_idx 失败: {}", e))?,
    ];
    for i in 0..N_LAYERS {
        // 数组拷贝进 Value（KV 每步一次拷贝；正确性优先，见模块文档差异说明）
        let k = state[2 * i].clone();
        let v = state[2 * i + 1].clone();
        feeds.push((
            format!("past_kv_{i}_k").into(),
            Value::from_array(k)
                .map_err(|e| anyhow::anyhow!("构造 past_kv_{i}_k 失败: {}", e))?
                .into(),
        ));
        feeds.push((
            format!("past_kv_{i}_v").into(),
            Value::from_array(v)
                .map_err(|e| anyhow::anyhow!("构造 past_kv_{i}_v 失败: {}", e))?
                .into(),
        ));
    }
    let outputs = session.run(feeds)?;
    let logits = out_f32(&outputs, 0)?;
    *state = extract_state(&outputs)?;
    Ok(logits.iter().copied().collect())
}

/// 提取 `outputs[1..]` 为 KV state（48 个 f32 张量）
fn extract_state(outputs: &ort::session::SessionOutputs<'_>) -> Result<Vec<ArrayD<f32>>> {
    let mut state = Vec::with_capacity(outputs.len().saturating_sub(1));
    for i in 1..outputs.len() {
        state.push(out_f32(outputs, i)?);
    }
    if state.len() != N_KV_TENSORS {
        bail!("KV state 数量错误: 期望 {}，实际 {}", N_KV_TENSORS, state.len());
    }
    Ok(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn 目录布局辅助函数与engine_py一致() {
        let dir = std::path::Path::new("/tmp/models");
        assert_eq!(
            dir.join("gpt_prefill").join("gpt_prefill.onnx"),
            dir.join("gpt_prefill").join("gpt_prefill.onnx")
        );
        // 平铺引擎命名：semantic_codec_decode.onnx（非 codec_decode.onnx）
        assert!(dir.join("semantic_codec_decode.onnx").ends_with("semantic_codec_decode.onnx"));
        assert_eq!(N_KV_TENSORS, 48);
        assert_eq!(N_LAYERS, 24);
    }
}

/// 定位本地模型目录（仅测试用）：优先 registry 约定目录，其次 HF 缓存快照（PoC 落点）。
#[cfg(test)]
pub(crate) fn find_model_dir() -> Option<std::path::PathBuf> {
    let name = "gpt_prefill";
    let primary = crate::shared::workspace_paths::WorkspacePaths::models_dir()
        .join("tts")
        .join("indextts25");
    if primary.join(name).join(format!("{name}.onnx")).is_file() {
        return Some(primary);
    }
    let snapshots = std::env::var_os("USERPROFILE").map(|p| {
        std::path::PathBuf::from(p)
            .join(".cache/huggingface/hub/models--yunfengwang--IndexTTS-2.5-onnx/snapshots")
    })?;
    let dir = snapshots.as_path();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path().join(name).join(format!("{name}.onnx"));
            if p.is_file() {
                return Some(entry.path());
            }
        }
    }
    None
}
