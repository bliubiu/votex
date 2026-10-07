# WeNet（ASR 本地，Conformer U2++）

> engine id：`wenet` | 类别：asr | 本地（sherpa-onnx 绑定）| registry：`models/registry/wenet.yaml`

## 用途

WeNet Conformer U2++ 中文普通话识别；流式架构（U2++）但当前按整段接入。

## 依赖与模型

- 推理：sherpa-onnx 绑定；模型目录 `models/asr/wenet/`：`model.onnx` + `tokens.txt`。

## 能力

- 语种：中文普通话 | 时间戳：⚠️ **实测仅首尾正确**（CHANGELOG [2026-10-06]，中间偏移）| 流式：❌（模型支持 U2++ 流式，votex 未接入）

## 已知怪癖

- **需要时间戳的任务不要选它**——用 Paraformer（token 级 ✅）。
- 首尾字符时间戳正确，整段字幕可用的场景可接受。

## 测试锚点

- `crates/votex-infra/tests/wenet_integration_test.rs`（时间戳结论的来源测试）
- `crates/votex-infra/tests/asr_cross_engine_0wav_test.rs`
