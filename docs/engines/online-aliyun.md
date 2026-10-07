# 阿里云（TTS + ASR 在线）

> engine id：`aliyun-tts` / `aliyun-asr` | 类别：tts / asr | **在线 API** | registry：无（API 服务）

## 用途

面向国内网络环境的在线备选链路：阿里云 TTS 与 NLS 智能语音交互 ASR。

## 依赖与接入

- TTS：阿里云语音合成 REST；ASR：**阿里云 NLS REST API**。
- 凭据经环境变量/配置文件注入（不落代码、不进日志）。

## 能力

- TTS：音色为阿里云音色名 | 情感：❌（votex 未接 style）| 克隆：❌
- ASR：中文为主 | 时间戳：—（按服务返回）

## 已知怪癖

- 同 Azure：在线引擎不参与本地金标准锚定；可用性 = 配置完整性。
- 与"文本不出网"默认姿态的关系同 docs/00——选在线即主动出网。

## 测试锚点

- 无金标准集成测试（在线服务）；配置解析由 `config_file_guard_test.rs` 间接覆盖。
