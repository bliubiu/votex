# HY-MT1.5（翻译本地，腾讯）

> engine id：`hy-mt-1.5` | 类别：translation | 本地 ONNX | registry：`models/registry/hy-mt-1.5.yaml`

## 用途

腾讯 HY-MT1.5 Decoder-only 架构离线翻译，中文语向质量好，多语种离线翻译首选。

## 依赖与模型

- 推理：ort；Decoder-only 生成式（自回归解码 + KV 缓存）。
- 模型目录由 registry 指针给出；就绪标志文件由 model_pool 校验。

## 能力

- 语向：多语（中英为核心）| 流式：❌

## 已知怪癖

- 生成式解码较慢，**必须经 `TranslationModelPool` 复用会话**（一次加载、反复使用），否则每次翻译重载分钟级。
- 就绪判定依赖 registry 的 ready 标志文件，缺文件视为未下载。

## 测试锚点

- `crates/votex-infra/tests/hy_mt_translation_test.rs`
