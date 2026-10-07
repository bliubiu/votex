# Azure Speech（TTS + ASR 在线）

> engine id：`azure-tts` / `azure-asr` | 类别：tts / asr | **在线 API** | registry：无（API 服务，不占模型额度）

## 用途

质量上限高的在线备选链路：TTS（SSML style 情感）与 ASR（多语言），供已购 Azure 服务的用户。

## 依赖与接入

- REST API 直连（无 ONNX）；Key 与区域经统一配置文件（`application.yml`），**不落代码**。
- 需显式选择在线引擎——与"默认无任何网络、文本不出网"的安全姿态（docs/00）并存：**选在线即用户主动同意内容出网**。

## 能力

- TTS：情感 ✅（style 参数）| 音色：Azure 音色名 | 克隆：❌（votex 未接 Azure 自定义音色）
- ASR：多语言 | 时间戳：—（按服务返回）

## 已知怪癖

- 在线引擎不参与本地 0.wav 金标准锚定与 `model list` 本地可用性探测（可用性 = 配置完整性检查）。
- 网络错误须走统一错误链（中文提示 + 不透传 Key）。

## 测试锚点

- 无金标准集成测试（在线服务）；配置解析由 `config_file_guard_test.rs` 间接覆盖。
