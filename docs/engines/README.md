# 引擎文档三件套

> 状态：活文档 | 建立日期：2026-10-07 | 来源：`docs/27-VoiceStudio对标与借鉴分析.md` 第四节 B
> 三件套 = **每引擎一页文档** + **能力矩阵一张表**（本页）+ **新引擎准入清单**
> 运行期权威数据：`votex model list --json`（CapabilityReport）。本目录是人类可读快照，两者失同步时以代码为准，并应修正本文档。

## 一、能力矩阵

### 1.1 TTS（4 本地 + 2 在线）

| engine id | 引擎 | 本地 | 零样本克隆 | 情感 | 方言 | 音色来源 | registry |
|---|---|---|---|---|---|---|---|
| `kokoro` | Kokoro-82M-zh | ✅ | ❌ | ❌ | ❌ | 内置音色池（扫描） | kokoro-82m-v1.1-zh.yaml |
| `indextts25` | IndexTTS-2.5 | ✅ | ✅（参考音频 ≤15s） | ❌ | ✅ 粤语原生 | `prompts/<id>.wav` → 统一音色库 | indextts-2.5-onnx.yaml |
| `cosyvoice3` | CosyVoice 3 | ✅ | ✅（**需配套转写**） | ❌ | ❌ | 统一音色库（必有转写） | cosyvoice.yaml |
| `qwen3-tts` | Qwen3-TTS | ✅ | ❌（1.7B instruct 音色） | ✅ 13 枚举 | ❌ | 0.6B 预设 / 1.7B instruct | qwen3-tts-0.6b/1.7b.yaml |
| `azure-tts` | Azure Speech | ❌ 在线 | ❌ | ✅（style） | ❌ | Azure 音色名 | —（API 服务） |
| `aliyun-tts` | 阿里云 | ❌ 在线 | ❌ | ❌ | ❌ | 阿里云音色名 | —（API 服务） |

> 采样率等运行期值以 `model list --json` 上报为准（provider `capability()` 覆盖）。

### 1.2 ASR（6 本地 + 2 在线 + 1 独立模块）

本地 ASR 统一经 **sherpa-onnx 绑定**（crate `sherpa-onnx = "1.13"`，内部 ONNX Runtime）加载；`OnnxRuntime` 动态库本身是共享运行时依赖。

| engine id | 引擎 | 本地 | 语种 | 时间戳（实测） | registry |
|---|---|---|---|---|---|
| `whisper` | Whisper | ✅ | 中 / 英 / 中英混合 | 部分 | whisper-base.yaml / whisper-small.yaml |
| `sensevoice` | SenseVoice | ✅ | 中文（含标点、情感标签） | ❌ | sensevoice.yaml |
| `paraformer` | Paraformer（FunASR） | ✅ | 中 / 英 / 中英混合 | ✅ token 级 | paraformer.yaml（sherpa 官方转换包） |
| `qwen3-asr` | Qwen3-ASR | ✅ | 29 语种 + 多方言 | 部分 | qwen3-asr.yaml |
| `firered-asr` | FireRedASR | ✅ | 中 / 英 + 20+ 方言 | ✅（CTC/AED 双路径） | firered-asr-ctc.yaml / firered-asr-aed.yaml |
| `wenet` | WeNet Conformer U2++ | ✅ | 中文普通话 | ⚠️ 仅首尾正确 | wenet.yaml |
| `azure-asr` | Azure Speech | ❌ 在线 | 多语言 | — | —（API 服务） |
| `aliyun-asr` | 阿里云 NLS | ❌ 在线 | 中文 | — | —（API 服务） |
| `speaker-diarization` | 说话人分离 | ✅ | —（VAD+分割+聚类） | ✅ 片段时间轴 | —（Pyannote 分割 + x-vector/ECAPA） |

> 时间戳实测结论（CHANGELOG [2026-10-06]）：Paraformer ✅、FireRed CTC/AED ✅、WeNet ⚠️ 仅首尾正确。**词级对齐类需求以 Paraformer 为第一优先。**

### 1.3 OCR（2 本地）

| 引擎 | 档位 | 语种 | registry |
|---|---|---|---|
| PaddleOCR | PP-OCRv6 tiny/small/**medium（默认）**、v5 mobile/server | 中/英 | paddleocr*.yaml（6 个） |
| EasyOCR | CRAFT + CRNN | 80+ 语言 | easyocr.yaml |

> RapidOCR **不是**可用引擎：与官方 v5 mobile 权重逐字节相同，仅作为 `ocr_rapidocr_eval_test.rs`（`--features slow-models`）的评测对照组。

### 1.4 翻译（5 本地模型 + 1 加速后端 + 2 在线/兜底）

| engine id | 引擎 | 本地 | 语向 | registry |
|---|---|---|---|---|
| `hy-mt-1.5` | HY-MT1.5（腾讯） | ✅ | 多语 | hy-mt-1.5.yaml |
| `nllb-200` | NLLB-200 distilled 600M（Meta） | ✅ | 200+ 语言 | nllb-200-distilled-600m.yaml |
| `m2m-100` | M2M-100 418M（Meta） | ✅ | 100 语言 | m2m-100-418m.yaml |
| `opus-mt` | OPUS-MT（Helsinki） | ✅ | 中↔英 双模型 | opus-mt-zh-en.yaml / opus-mt-en-zh.yaml |
| `qwen-mt` | Qwen-MT（阿里） | ❌ 在线 | 多语 | —（DashScope API） |
| `ctranslate2` | CTranslate2 | ✅（系统库） | —（**加速后端**，非独立引擎） | 复用上述模型 CT2 转换版 |
| DeepSeek | LLM 翻译 | ❌ 在线 | 中↔英 | —（API） |
| 词典翻译 | 内置词典 | ✅ | 仅 zh→en | —（内置数据） |

> 离线模型经 `TranslationModelPool` 按 engine id 缓存会话——**一次加载、反复使用**，避免每次翻译重建 ONNX Session 的分钟级开销。

## 二、新引擎准入清单（四项不全不合并）

> 原则（借鉴 VoiceStudio）：**引擎广度只有在每个引擎都能在目标平台工作的前提下才是资产，否则是负债。**

- [ ] **1. 构建与依赖**：不破坏 Windows 全量构建；新增运行时依赖（动态库、解码器等）需评估单二进制体积影响并在本页 1.x 表登记。
- [ ] **2. 集成测试锚定**：ASR 进 `asr_cross_engine_0wav_test.rs`（0.wav 金标准）；TTS 进 `tts_asr_闭环_test.rs`（合成→识别闭环）；OCR/翻译各进对应 `*_test.rs`，重模型用 `slow-models` 特性门控。
- [ ] **3. 注册一致**：`models/registry/*.yaml` 注册（`model_registry_consistency_test.rs` / `model_registry_guard_test.rs` 守护）；`EngineKind` 枚举 + `capability.rs` 能力画像（`能力画像_覆盖全部引擎` 测试会暴露遗漏）；CLI `--engine` 可解析。
- [ ] **4. 文档页**：本目录新建 `<类别>-<engine>.md`，含用途/依赖/模型来源/能力/已知怪癖/测试锚点六节。

## 三、引擎页索引

| 类别 | 页 |
|---|---|
| TTS 本地 | [tts-kokoro](tts-kokoro.md) · [tts-indextts25](tts-indextts25.md) · [tts-cosyvoice3](tts-cosyvoice3.md) · [tts-qwen3](tts-qwen3.md) |
| TTS/ASR 在线 | [online-azure-speech](online-azure-speech.md) · [online-aliyun](online-aliyun.md) |
| ASR 本地 | [asr-whisper](asr-whisper.md) · [asr-sensevoice](asr-sensevoice.md) · [asr-paraformer](asr-paraformer.md) · [asr-qwen3-asr](asr-qwen3-asr.md) · [asr-firered-asr](asr-firered-asr.md) · [asr-wenet](asr-wenet.md) · [asr-speaker-diarization](asr-speaker-diarization.md) |
| OCR | [ocr-paddleocr](ocr-paddleocr.md) · [ocr-easyocr](ocr-easyocr.md) |
| 翻译 | [tr-hy-mt-1.5](tr-hy-mt-1.5.md) · [tr-nllb-200](tr-nllb-200.md) · [tr-m2m-100](tr-m2m-100.md) · [tr-opus-mt](tr-opus-mt.md) · [tr-qwen-mt](tr-qwen-mt.md) · [tr-ctranslate2](tr-ctranslate2.md) · [tr-llm-deepseek](tr-llm-deepseek.md) · [tr-dict](tr-dict.md) |

## 四、已知怪癖速查（教训沉淀）

| 怪癖 | 引擎 | 处置 |
|---|---|---|
| FunASR 自导出 ONNX 缺 `vocab_size` → sherpa-onnx C++ 端 **abort 整个进程** | Paraformer | 加载前显式拒绝；registry 固定用 sherpa-onnx 官方转换包 |
| WeNet 时间戳仅首尾正确 | WeNet | 时间戳需求不选它 |
| v6 字典不外置（字符表在 ONNX 元数据）；Tiny/Mobile 对字形近似字有稳定单字误差 | PaddleOCR | 默认 v6 Medium；字符表元数据优先、外部字典回退 |
| 参考音频超 15s：无转写自动 best_window 截取，**有转写显式拒绝** | IndexTTS-2.5 | `ref_audio.rs` 契约 |
| 克隆音色必须有配套转写，否则报错指引 `voice add --transcript` | CosyVoice 3 | 音色库入库时强制 |
| Qwen3-TTS 首次加载 10~60s | qwen3-tts | `qwen3_model_selector` 按 GPU/内存推荐 0.6B/1.7B |
| 离线翻译模型加载慢 | 全部翻译引擎 | `TranslationModelPool` 会话池 |
| CTranslate2 是 FFI 骨架，需系统安装 C++ 库 | ctranslate2 | 可选加速，缺库时走 ort 路径 |
