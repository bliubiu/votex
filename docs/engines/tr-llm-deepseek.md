# LLM 翻译（在线，DeepSeek）

> engine id：`deepseek` | 类别：llm / translation | **在线 API** | registry：无

## 用途

DeepSeek Chat API（`deepseek-chat`）做中↔英翻译与文案改写（视频生成链路的 LLM 文案也走此通道）。

## 依赖与接入

- 端点：`https://api.deepseek.com/v1/chat/completions`；Key 经配置注入，不落代码不进日志。

## 能力

- 语向：中↔英双向 | 提示词工程可带风格/术语约束。

## 已知怪癖

- 在线引擎：可用性 = 配置完整性；选它即主动出网。
- 输出经清洗（去思维链/围栏）后才进下游字幕，防止 markdown 污染 SRT。

## 测试锚点

- 无金标准集成测试（在线）；请求构造/响应清洗逻辑由单元测试覆盖（app 层）。
