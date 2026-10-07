# NLLB-200（翻译本地，Meta）

> engine id：`nllb-200` | 类别：translation | 本地 ONNX | registry：`models/registry/nllb-200-distilled-600m.yaml`

## 用途

Meta NLLB-200 distilled 600M：**200+ 语种**离线覆盖，小语种翻译唯一本地选项。

## 依赖与模型

- 推理：ort；encoder-decoder 架构；语种代码用 FLORES-200 映射（`zh_CN`/`eng_Latn` 等）。

## 能力

- 语向：200+ 语言任意互译 | 中文质量中规中矩，胜在覆盖面。

## 已知怪癖

- 语种代码必须是 FLORES-200 全称，简写会解码成乱码——映射层统一处理。
- 会话池复用同 HY-MT1.5（加载慢，必须缓存）。

## 测试锚点

- `crates/votex-infra/tests/nllb_translation_test.rs`
