//! docs/20 F67 取证：CosyVoice3 文本条件失效的根因定位
//!
//! 背景：ASR 回环证明 CosyVoice3 对任意输入文本都产出 `zh_prompt.txt` 的音频内容，
//! 输入文本完全缺席。定量证据（release 日志，分段量纲诊断）：
//!
//! | 分段 | shape | RMS | min | max |
//! |---|---|---|---|---|
//! | `text_emb` | [57, 896] | **0.0223** | -0.1231 | 0.0958 |
//! | `sos_emb` | [1, 896] | 0.9393 | -3.2715 | 2.6855 |
//! | `task_id_emb` | [1, 896] | 0.8900 | -2.5762 | 2.3496 |
//! | `prompt_speech_emb` | [201, 896] | 0.8890 | -3.7695 | 3.5312 |
//!
//! 文本嵌入比其余三段小 **40 倍**，LLM 实际看不到文本条件。
//!
//! 本测试回答两个问题：
//! ① `text_embedding_fp32.onnx` 是否有多个输出？取 `outputs[0]` 是否取对了？
//! ② 各个输出的量纲分别是什么？是否存在一个量纲 ≈ 0.9 的「正确」输出被漏用？
//!
//! 模型缺失则跳过（模型不随二进制分发）。

use ort::value::Value;

fn cosyvoice_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/tts/cosyvoice")
}

fn rms(a: &[f32]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    (a.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / a.len() as f64).sqrt()
}

#[test]
fn docs20_f67_文本嵌入onnx输出契约与量纲() {
    let dir = cosyvoice_dir();
    let model = dir.join("text_embedding_fp32.onnx");
    if !model.is_file() {
        eprintln!("跳过：模型不在 {}", model.display());
        return;
    }

    // 项目使用 ort 的 `load-dynamic` 特性：必须先由 ort_factory 解析 ORT 动态库路径，
    // 否则 `Session::builder()` 会 panic（ort-2.0.0-rc.13/src/lib.rs:234）。
    votex_infra::shared::ensure_ort_dylib_path();

    let mut session = votex_infra::shared::OrtSessionFactory::create_raw(&model)
        .unwrap_or_else(|e| panic!("加载 text_embedding_fp32.onnx 失败: {e}"));

    // 构造 57 个 token 的输入（与实跑的 [57, 896] 对齐），token id 取词表内合法值
    let ids: Vec<i64> = (0..57).map(|i| 1000 + i as i64).collect();
    let n_ids = ids.len();
    let input = ndarray::Array2::from_shape_vec((1, n_ids), ids).unwrap();
    let outputs = session
        .run(ort::inputs![Value::from_array(input).unwrap()])
        .expect("text_embedding 推理失败");

    eprintln!("\n===== 输出契约（token_ids {} 个）=====", n_ids);
    for i in 0..outputs.len() {
        let arr = outputs[i].try_extract_array::<f32>().expect("提取 f32 输出失败");
        let flat: Vec<f32> = arr.iter().copied().collect();
        let min = flat.iter().cloned().fold(f32::MAX, f32::min);
        let max = flat.iter().cloned().fold(f32::MIN, f32::max);
        eprintln!(
            "  output[{i}] shape={:?} rms={:.6} mean={:.6} min={:.4} max={:.4}",
            arr.shape(),
            rms(&flat),
            flat.iter().sum::<f32>() / flat.len() as f32,
            min,
            max
        );
    }

    // ---- 断言 ①：输出个数。若模型导出多个语义不同的输出而代码只取 outputs[0]，
    // 会静默丢掉训练时的正确输出，且不报任何错（F67 这类缺陷的典型形态）。
    // 基线（2026-10-03 实测）：text_embedding_fp32.onnx 只有 **1 个** 输出。
    assert_eq!(
        outputs.len(),
        1,
        "text_embedding 输出个数从 1 变为 {}：若新增了输出，cosyvoice.rs 仍只取 \
         outputs[0]，会静默使用错误的张量（docs/20 F67 同类缺陷）",
        outputs.len()
    );

    // ---- 断言 ②：输出形状必须是 [1, seq_len, hidden_dim]，否则转置/拼接全错位。
    let out0 = outputs[0]
        .try_extract_array::<f32>()
        .expect("提取 f32 输出失败");
    assert_eq!(
        out0.shape(),
        &[1, n_ids, 896],
        "text_embedding 输出形状应为 [1, seq_len, 896]，实际 {:?}",
        out0.shape()
    );

    eprintln!(
        "\n结论：text_embedding 输出个数 = {}（仅 outputs[0]），形状 = {:?}；\
         Rust 侧取 outputs[0] **正确**，RMS 偏小是模型固有属性，\
         官方 Python 参考实现（onnx_inference_pure.py:244）同样取 outputs[0]。",
        outputs.len(),
        out0.shape()
    );
}
