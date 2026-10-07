# SenseVoice（ASR 本地，中文 + 情感标签）

> engine id：`sensevoice` | 类别：asr | 本地（sherpa-onnx 绑定）| registry：`models/registry/sensevoice.yaml`

## 用途

中文识别主力之一：自带标点恢复与情感标签输出，字幕链路（无 LLM 后处理）首选。

## 依赖与模型

- 推理：sherpa-onnx 绑定；模型目录见 `models/registry/sensevoice.yaml` 指针。

## 能力

- 语种：中文为主 | 标点：✅（引擎内置）| 情感标签：✅（识别结果附情感标记）
- 时间戳：❌（需要时间戳走 Paraformer）| 流式：❌

## 已知怪癖

- 情感标签会出现在识别文本中，下游字幕/质检链路需按需剥离。
- 中英混合场景弱于 Whisper/Qwen3-ASR，纯中文任务精度/速度均衡。

## 测试锚点

- `crates/votex-infra/tests/asr_cross_engine_0wav_test.rs`（0.wav 金标准）
- `crates/votex-infra/tests/asr_e2e_test.rs`
