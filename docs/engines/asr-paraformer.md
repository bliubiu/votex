# Paraformer（ASR 本地，token 时间戳 ✅）

> engine id：`paraformer` | 类别：asr | 本地（sherpa-onnx 绑定）| registry：`models/registry/paraformer.yaml`

## 用途

阿里达摩院 FunASR 非自回归端到端模型，中文/中英混合**速度快**；**token 级时间戳实测 ✅**，是词级时间戳对齐（视频配音）的第一优先引擎。

## 依赖与模型

- 推理：sherpa-onnx 绑定；特征提取（fbank）由运行时完成。
- **registry 固定指向 sherpa-onnx 官方转换包** `csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09`：`model.int8.onnx` + `tokens.txt`（2026-10-06 换源）。

## 能力

- 语种：中 / 英 / 中英混合 | 时间戳：✅ token 级（CHANGELOG [2026-10-06] 实测）| 流式：❌

## 已知怪癖（重要教训）

- **FunASR 原始自导出 `model_quant.onnx` 缺 sherpa-onnx 所需 `vocab_size` 元数据，直接加载会导致 C++ 端 abort 整个进程**。处置：加载前显式校验并拒绝（带中文提示改用官方转换包），registry 已换源；FunASR 拒绝护栏测试在目录已有 sherpa 模型时跳过（条件适用）。
- 换源前旧包与转写对不齐的教训沉淀在 `paraformer_integration_test.rs`。

## 测试锚点

- `crates/votex-infra/tests/paraformer_integration_test.rs`
- `crates/votex-infra/tests/asr_cross_engine_0wav_test.rs`
