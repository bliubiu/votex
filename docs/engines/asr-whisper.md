# Whisper（ASR 本地，多语言）

> engine id：`whisper` | 类别：asr | 本地（sherpa-onnx 绑定）| registry：`models/registry/whisper-base.yaml`、`whisper-small.yaml`

## 用途

多语言识别兜底引擎，中英混合友好；docs/22 评估过的 Orukeet 多语言路线之外的稳态选择。

## 依赖与模型

- 推理：**sherpa-onnx 绑定**（crate `sherpa-onnx = "1.13"`，内部 ONNX Runtime）。
- 模型目录：`models/asr/whisper-base/`（encoder + decoder + tokens），可选 small 档位（registry 两条目）。

## 能力

- 语种：中 / 英 / 中英混合 | 时间戳：部分 | 流式：❌
- 与其他本地 ASR 同属 0.wav 金标准锚定范围。

## 已知怪癖

- base/small 档位精度/耗时差异明显，长音频建议 small；档位选择经模型选择器，不自动回退。
- 采样率要求 16 kHz，重采样由 infra 音频链统一处理。

## 测试锚点

- `crates/votex-infra/tests/asr_cross_engine_0wav_test.rs`（0.wav 金标准）
- `crates/votex-infra/tests/en_asr_test.rs`（英文/中英混合）
- `crates/votex-infra/tests/asr_e2e_test.rs`
