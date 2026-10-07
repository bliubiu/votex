# Kokoro-82M-zh（TTS 本地）

> engine id：`kokoro` | 类别：tts | 本地 ONNX | registry：`models/registry/kokoro-82m.yaml`、`kokoro-82m-v1.1-zh.yaml`

## 用途

轻量中/英文合成，CPU 即可流畅运行；默认引擎（CLI `tts --engine` 默认值），适合快速试听与批量合成。

## 依赖与模型

- 推理：ort（ONNX Runtime）直连，无 sherpa 绑定。
- 模型：`models/tts/` 下 Kokoro-82M v1.1 中文版（registry 含英文原版与 v1.1-zh 两个条目）。
- 音色：**内置音色池扫描**——GUI TTS 页音色下拉动态扫描音色池（替代早期硬编码 8 音色）。

## 能力

- 克隆：❌ | 情感：❌ | 方言：❌
- 语种：中 / 英 | 输出采样率 24 kHz（运行期以 `model list --json` 为准）

## 已知怪癖

- 音质与表现力弱于 Qwen3-TTS/IndexTTS-2.5，定位为"轻量默认"。
- 无参考音频概念，音色枚举来自音色池，不支持用户自定义。

## 测试锚点

- `crates/votex-infra/tests/kokoro_integration_test.rs`
- `crates/votex-infra/tests/kokoro_provider_test.rs`
- `crates/votex-infra/tests/tts_asr_闭环_test.rs`（跨引擎金标准：合成→识别闭环）
