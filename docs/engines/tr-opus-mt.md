# OPUS-MT（翻译本地，中英双向）

> engine id：`opus-mt` | 类别：translation | 本地 ONNX | registry：`models/registry/opus-mt-zh-en.yaml`、`opus-mt-en-zh.yaml`

## 用途

Helsinki-NLP OPUS-MT（Marian 架构）：**中↔英专用双模型**，短句快且准，字幕/标签翻译默认本地选择。

## 依赖与模型

- 推理：ort；Marian encoder-decoder。
- **方向即模型**：zh→en 与 en→zh 是两个独立模型目录，按语向自动选择。

## 能力

- 语向：仅中↔英 | 模型小、CPU 友好。

## 已知怪癖

- 双模型按语向加载：未安装对应方向模型时报错指向下载对应 registry 条目，不静默回退。
- 也是 CTranslate2 加速后端支持的模型之一（可转 CT2 格式提速）。

## 测试锚点

- `crates/votex-infra/tests/opus_mt_translation_test.rs`
