# CosyVoice 3（TTS 本地，零样本克隆）

> engine id：`cosyvoice3` | 类别：tts | 本地 ONNX | registry：`models/registry/cosyvoice.yaml`

## 用途

零样本克隆第二引擎；与 IndexTTS-2.5 互补（中文表现好，需转写驱动）。

## 依赖与模型

- 推理：ort；模型目录见 `models/registry/cosyvoice.yaml` 指针。
- 音色来源：**统一音色库** `models/voices/refs/`，且必须有配套转写文本。
- `voice add` 入库时填写转写（`--transcript`），入库即两引擎通用。

## 能力

- 克隆：✅ 零样本（**需配套参考转写**）
- 情感：❌ | 方言：❌ | 语种：中文为主

## 已知怪癖

- 克隆音色解析时若音色不在 `models/voices/refs/` 或缺转写文本，直接报错并提示 `voice add --transcript` 修复——**不静默降级**。
- 音色未命中报错文案指向 `voice add`（2026-10-06 起）。
- tokenizer 与参考文本一致性由 `cosyvoice_tokenizer_vs_ref_test.rs` 守护。

## 测试锚点

- `crates/votex-infra/tests/cosyvoice_tokenizer_vs_ref_test.rs`
- `crates/votex-infra/tests/cosyvoice_text_emb_probe_test.rs`
- `crates/votex-infra/tests/tts_asr_闭环_test.rs`
