# Qwen3-ASR（ASR 本地，多语种大模型）

> engine id：`qwen3-asr` | 类别：asr | 本地（sherpa-onnx 绑定）| registry：`models/registry/qwen3-asr.yaml`

## 用途

大模型识别引擎：29 种语言 + 多种中国方言，长音频效果好；方言 ASR 的主选。

## 依赖与模型

- 推理：sherpa-onnx 绑定；模型为**三图布局**：`conv_frontend` + `encoder` + `decoder`，另需 tokenizer 目录。
- registry：`qwen3-asr.yaml`（官方包为 .tar.bz2，infra 依赖 bz2 解码器解包）。

## 能力

- 语种：29 种 + 中国多方言 | 时间戳：部分 | 流式：❌
- 与 firered 同为"方言识别"路线（TTS 粤语由 IndexTTS-2.5 承担）。

## 已知怪癖

- 三图缺一即加载失败，模型完整性校验先行（报错指向重新下载）。
- 模型体积较大，首加载慢；不设 `slow-models` 门控（默认引擎之一）。

## 测试锚点

- `crates/votex-infra/tests/asr_cross_engine_0wav_test.rs`
- `crates/votex-infra/tests/asr_e2e_test.rs`
