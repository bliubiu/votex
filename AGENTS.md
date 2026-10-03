## votex（声阅）

离线桌面语音工具（TTS 配音朗读 + STT 语音转文字）、Rust 开发、面向中文内容创作者（有声小说 / 视频配音）、CLI+GUI 双模、单二进制分发。

- TTS(Text-To-Speech): 语音合成、配音、朗读
- ASR/STT(Automatic Speech Recognition / Speech-To-Text): 语音识别、音频转写、字幕生成
- OCR(): 图文识别，书籍文档识别
- 翻译能力：中英/英中翻译
- 



### 功能

- 文本生成语音（朗读/配音），支持中文普通话 + 方言（粤语、闽南语、吴语、客家话等），支持音频播放、实时试听
- 语音转文本（转写/字幕），支持简体中文 + 中英混合识别
- 文档识别: 支持的文件格式（PDF / 图片 / 扫描件）、多语言识别
- 多格式输出：WAV、MP3、TXT、SRT、LRC
- 音频降噪、字幕可视化编辑
- 批量任务：批量 TTS 合成、批量 ASR 转写、大文档识别、长文本或语音翻译
- 长文本分段算法、优化 CPU 推理调度、降低内存占用
- 流水线模板与预设：用户可自定义流水线步骤组合，一键生成有声书/播客
- 在线 API 接入：支持通过 API Key 接入在线 TTS/ASR/LLM 服务（Azure Speech、阿里云、DeepSeek 等）
- 自动化短视频生成：LLM 文案 + TTS 配音 + ASR 字幕 + 素材合成
- 统一抽象模型接口：便于快速新增适配
- 输入文本编码: UTF-8、GBK（自动检测转换）
- 输出：统一使用 UTF-8

- TTS 支持的语速 / 音调 / 音量调节范围、情感合成
- ASR 支持的音频格式（如 MP3/WAV/M4A/FLAC）、采样率要求、长音频
- 字幕生成的样式自定义（字体 / 颜色 / 位置）、多段字幕的批量编辑规则



### 技术栈

- 基于**DDD 架构（领域驱动设计）**和**TDD（测试驱动开发）**设计思想
- Rust 
- clap（CLI 参数解析）
- egui + eframe（GUI 框架）
- TTS 引擎：Qwen3-TTS 、CosyVoice 3.0、 Kokoro‑82M-zh(ONNX) 、 IndexTTS2(ONNX) 
- STT/ASR 引擎：SenseVoice、Qwen3-ASR、Paraformer、FireRedASR、WeNet (Conformer) 、FireRedASR v2、
- OCR 引擎： PaddleOCR（PP-OCRv6 Medium 最新版本可用）
- 翻译引擎： HY-MT1.5、NLLB-200、M2M-100、CTranslate2、Qwen-MT、OPUS-MT
- rust-ffmpeg（音频编解码）
- 推理引擎：ort（ONNX Runtime Rust 绑定）
- 程序文件：`votex.exe`
- 窗口标题 / 软件名：**声阅**
- 统一配置文件：`application.yml`
- 底层统一使用ONNX Runtime 推理
- 统一使用sqlite(wal)数据库实现持久化存储




### 日志系统

支持多日志级别（DEBUG、INFO、ERROR），并提供文件输出和日志轮转功能。

- 日志级别：支持 DEBUG、INFO、ERROR 三个级别，可通过配置文件指定日志级别，默认 INFO 级别
- 日志格式：`[YYYY-MM-DD HH:MM:SS.SSS] [级别] [线程ID] [模块:行号] - 日志内容`
- 日志输出：支持文件输出和终端输出，文件输出路径默认 `logs/`，日志文件按日期轮转（每日一个文件），保留 32 天日志，自动清理过期日志
- 日志内容：统一使用中文，清晰记录操作行为、执行结果、错误信息，便于问题排查和审计
- 日志内容不能记录敏感信息，必须严格遵循脱敏策略，禁止记录：密码、完整身份证号、手机号
- 文件命名格式：`votex-YYYYMMDD.log`，例如 `votex-20231225.log`



### 文档和注释

- 项目文档统一存储 `docs/` 目录，按顺序标号命名中文文档
- 注释、日志内容统一使用中文
- 统一错误处理



### 安全策略

[安全设计](docs\00-安全设计.md)



### 约束

- 禁止使用软链接
- CLI 与 GUI 功能完全对等（同一套领域层，两套表现层）
- 支持离线运行，所有模型本地存储，首次启动引导下载模型
- 单二进制分发，不依赖外部运行时
- 模型文件不随二进制分发，独立管理
- 用户文本内容不上传网络，本地处理全流程
- 禁止自动删除模型文件，必须二次确认操作
- 禁用 npm 包管理，可用pnpm，bun
- 禁用Maven，可用gradle
- 使用uv管理python 软件包，禁止使用pip/venv/pipenv/poetry
- 必须编译配置优化，从源头减小体积



### 测试文本文件

- `tmp\novel.txt` 、`tmp\e2e_ch1.txt ` 文本文件可用于验证项目TTS/ASR功能

- 



## 参考
- [real-time translation](https://github.com/niedev/RTranslator)
- github.com/abus-aikorea/voice-pro
- DietrichGebert/ponytail
- calesthio/OpenMontage
- https://github.com/KittenML/KittenTTS
- https://github.com/jianchang512/pyvideotrans # 翻译项目
- MoneyPrinterTurbo 
- VideoLingo 
- KrillinAI 




# AI 行为准则（Karpathy Standard）

## 核心原则

1. **先澄清，不假设** — 需求模糊时先问清楚，不猜测意图
2. **简洁优于聪明** — 简单、可读、最小化代码，不过度设计
3. **最小改动** — 只做必要修改，不改无关代码、不重构整个文件
4. **目标驱动** — 聚焦当前任务，不加"多余"功能

## 语言规则

1. 所有回复、解释、对话使用中文
2. 全程保持纯中文交互
3. 注释和文档使用中文