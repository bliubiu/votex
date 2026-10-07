# Qwen3-TTS（TTS 本地，情感 instruct）

> engine id：`qwen3-tts` | 类别：tts | 本地 ONNX | registry：`models/registry/qwen3-tts-0.6b.yaml`、`qwen3-tts-1.7b.yaml`

## 用途

高质量合成与**情感表达**唯一入口：13 个情感枚举经 instruct 子句接线（2026-10-06 落地），有声书情感段落首选。

## 依赖与模型

- 推理：ort；tokenizer + speaker embeddings + ONNX 会话。
- 双档位：
  - **0.6B**：预设 speaker embedding 音色；
  - **1.7B**：VoiceDesign instruct 音色（文本描述驱动，接近 VoiceStudio 的 Voice Design 能力）。
- `qwen3_model_selector` 按 GPU/内存/模型存在性自动推荐档位，含推荐说明文本。

## 能力

- 情感：✅ 13 枚举（instruct 子句注入）| 克隆：❌（1.7B 为描述式生成，非参考音频克隆）
- 语种：中 / 英 | 流式：❌

## 已知怪癖

- **首次加载需 10~60 秒**（大 ONNX 会话 + embeddings），GUI/CLI 首次合成需等待提示。
- 引擎不带 style 的情感枚举会 warn-once 降级（其余引擎无情感通路）。
- 语速/音调/音量参数经 `TtsParams` 通用通路。

## 测试锚点

- `crates/votex-infra/tests/tts_e2e_multi_engine_test.rs`
- `crates/votex-infra/tests/tts_english_test.rs`、`tts_style_test.rs`（风格/情感）
- `crates/votex-infra/tests/tts_asr_闭环_test.rs`
