# FireRedASR（ASR 本地，CTC/AED 双路径 + 方言）

> engine id：`firered-asr` | 类别：asr | 本地（sherpa-onnx 绑定）| registry：`models/registry/firered-asr.yaml`、`firered-asr-ctc.yaml`、`firered-asr-aed.yaml`

## 用途

中文识别 + **20+ 方言**；引擎内含两条推理路径（v2 起），时间戳实测 ✅，方言字幕/配音链路可用。

## 依赖与模型

- **CTC 路径**（v2 CTC）：`models/asr/firered-asr-ctc/`，单图 `model.int8.onnx` + `tokens.txt`。
- **AED 路径**：`models/asr/firered-asr-aed/`，双图 `encoder.int8.onnx` + `decoder.int8.onnx` + `tokens.txt`。
- 目录查找顺序：`models/asr/firered-asr-ctc/` → 回落 `models/firered-asr-ctc/`（兼容旧布局）。

## 能力

- 语种：中 / 英 + 20+ 方言 | 时间戳：✅（CTC 与 AED 实测均通过，CHANGELOG [2026-10-06]）| 流式：❌
- 方言识别与 Qwen3-ASR 互补，按模型可用性选择。

## 已知怪癖

- 双路径并存：引擎按目录内容自动选择，**CTC 与 AED 模型目录不可混装**（单图 vs 双图布局不同）。
- AED 双图加载内存占用更高。

## 测试锚点

- `crates/votex-infra/tests/firered_asr_integration_test.rs`
- `crates/votex-infra/tests/asr_cross_engine_0wav_test.rs`
