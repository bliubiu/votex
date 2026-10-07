# 词典翻译（本地兜底）

> engine id：词典翻译（`dict_translate`，内部模块）| 类别：translation | 本地（内置数据）| registry：无

## 用途

内置中英词典，服务**短视频标签、文件名等简短文本**的离线快速翻译；LLM/模型翻译不可用时的最后兜底。

## 依赖与模型

- 无外部依赖；词条随二进制内置。

## 能力

- **仅支持 zh→en 单向**：反向（en→zh）与自动语向检测返回 `UnsupportedDirection`——显式拒绝，不猜方向。
- 逐词匹配，不做语序重组：只适合词/短语级输入。

## 已知怪癖

- 长句输入会被拒绝/降质处理，长文本必须走模型翻译引擎。
- 与视频生成链路（文件名/标签生成）强绑定，勿在字幕翻译中选用。

## 测试锚点

- 方向拒绝与逐词匹配逻辑由 `crates/votex-infra/src/translation/dict_translate.rs` 内单元测试覆盖（`UnsupportedDirection`）。
