# Qwen-MT（翻译在线，阿里 DashScope）

> engine id：`qwen-mt` | 类别：translation | **在线 API** | registry：无

## 用途

阿里 Qwen-MT 翻译专线（DashScope API）：多语种、术语一致性好的在线高质量选项。

## 依赖与接入

- DashScope HTTP API；Key 经配置/环境变量注入，不落代码不进日志。

## 能力

- 语向：多语 | 术语表/领域提示可经参数下发。

## 已知怪癖

- 在线引擎：不参与本地金标准锚定；选它即主动出网（docs/00 姿态）。
- 与本地引擎共用 `TranslateUseCase` 入口，按 engine id 路由。

## 测试锚点

- `crates/votex-infra/tests/qwen_mt_translation_test.rs`（请求构造/响应解析，mock 网络）
