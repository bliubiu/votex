# IndexTTS-2.5 粤语方言支持：引入 Python 处理的可行性评估

> 评估日期：2026-10-05
> 结论先行：**不建议为方言能力引入常驻 Python 运行时**；推荐「Python 一次性验证（uvx 临时环境）→ Rust ONNX 原生移植」两步走。Python 在本方案中是**蓝本与验收工具**，不是分发依赖。

---

## 1. 背景与目标

AGENTS.md 将方言支持（粤语、闽南语、吴语、客家话）列为 TTS 功能目标。当前 `indextts2` 适配器基于 IndexTTS-**2** 社区 ONNX 导出（`ThreadAbort/IndexTTS-Rust`，4.5G），无粤语能力。IndexTTS-**2.5**（2026-08 发布）的 tokenizer 换为 `multilingual_zh_ja_yue_char_del.tiktoken`——**yue（粤语）在词表中**，社区转换版本（libwaifu）明确声明 "English, Chinese, Japanese and Cantonese all understood"。

因此问题变成：**为了拿到 2.5 的粤语能力，接入方式选什么？**

## 2. 事实核查（2026-10 实测）

### 2.1 官方栈（github.com/index-tts/index-tts）

| 维度 | 事实 |
| :--- | :--- |
| 运行环境 | Python 3.10–3.11 + **PyTorch** + transformers，`uv sync --all-extras` |
| 硬件 | **NVIDIA GPU 约 6GB 显存**（BF16），官方以 GPU 为目标平台 |
| 权重 | 主模型 2–3GB + 辅助模型（w2v-bert-2.0、MaskGCT codec、CAMPPlus、BigVGAN，首次运行自动下载） |
| 语言 | 官方宣称 zh/en/ja/es/ar 五语；**粤语不是官方一等语种**，属词表+数据覆盖，音质需实测 |

结论：官方栈与 votex「纯 CPU / ort / 离线 / 单二进制」四项约束全部冲突，直接排除。

### 2.2 社区 ONNX 导出（yunfengwang/IndexTTS-2.5-onnx，PyPI `index-tts-2.5-onnx` 0.1.0）

- **fp32 bit-exact**：对官方 PyTorch CPU 参考实现，每段余弦=1.0000，greedy acoustic token 逐个复现（fx0 73/73、fx1 83/83）；声码器 mel-SNR ≈ 85 dB。
- **torch-free**：依赖仅 `onnxruntime / tiktoken / kaldi-native-fbank / librosa / soundfile / wetext / pyyaml / huggingface-hub`，**无 PyTorch**。
- CPU 默认，跨平台（Linux/Windows/macOS，x86/ARM），`uvx` 一键运行。
- **不做 int8 量化是实测结论**：动态 QInt8 会翻转 GPT greedy argmax（仅 26% token 命中），权重-only int8 更差（cos 0.985）且在 ORT CPU 上更慢——**必须保持 fp32**，全量约 5.1G。
- 文件布局即流水线：`campplus.onnx` / `emo_vec.onnx` / `semantic_model/`（w2v-bert）/ `semantic_codec_decode.onnx` / `gpt_prefill/`+`gpt_step/`（自回归拆分）/ `length_regulator.onnx` / `cfm_estimator/`（flow-matching DiT）/ `bigvgan.onnx`，含情绪向量输入（`emo_vec.onnx`）——**情绪控制能力保留**。
- 同仓库另有 MLX（Apple Silicon 超实时）与 MNN（快速 CPU）版本，说明该流水线在非 torch 后端可跑通。

### 2.3 纯 Rust 现成实现（ling0322/libwaifu，MIT）

- 支持 `indextts`（2.5，fp32 全链路，声明支持 zh/en/ja/**yue**）与 Fun-CosyVoice3 0.5B。
- **推理后端是自研 C++ 内核 flint**（CPU AVX2/AVX-512、CUDA CUTLASS、Vulkan、MLX），Rust 仅是外壳——**不是 ort 栈**，直接引入会带入 CMake/C++ 构建链，破坏 votex「纯 Rust + ort」架构。
- 功能取舍：无 beam search（单序列采样）、无独立情绪控制。
- 定位：**只作参考实现**（可对照其 `docs/indextts.md` 的七模型流水线文档），不作依赖。

### 2.4 粤语能力的本质

粤语能力来自**权重 + tiktoken 词表（含 yue）**，不来自 Python 代码。ONNX 导出保权保词表 → **任何正确的接入方式（Rust/Python/C++）都能拿到同等的粤语能力**。这直接否定了「要方言就必须 Python」的前提。

另需注意：AGENTS.md 方言目标含闽南语、吴语、客家话，**2.5 只覆盖 yue**，方言路线图不能押注单一引擎（Qwen3-TTS / CosyVoice 3 的方言覆盖需另行评估）。

## 3. 三条路线对比

| 维度 | A. Python sidecar（官方栈） | B. Python sidecar（`index-tts-2.5-onnx` + uvx） | C. Rust ONNX 原生移植（推荐） |
| :--- | :--- | :--- | :--- |
| 粤语能力 | 有（同权重） | 有（同权重，fp32 bit-exact） | 有（同权重，同导出） |
| 运行时依赖 | torch + CUDA，Python 3.10 | Python 运行时（uv 管理），~150–250MB 包 | **零新增**（ort 已在栈内） |
| 架构合规 | 违反单二进制/离线/CPU 三项 | 违反单二进制/不依赖外部运行时两项；AGENTS.md 允许 uv 属灰色 | **完全合规**（DDD provider trait、EngineState、registry 全部现成） |
| 进程模型 | 子进程 + IPC | 子进程 + IPC，长驻则常驻内存 | 同进程，`&self` trait、InferenceGate、取消令牌直接复用 |
| 工程量 | 大（GPU 部署 + 打包） | 小（PoC 一天内可跑） | **中**：主要是流水线编排移植，数值由 bit-exact 参考兜底 |
| 长期维护 | 受制于官方栈 | 双语言栈，安全面扩大 | 单栈，与现有 5 引擎适配器同构 |
| 风险 | 打包体积爆炸、离线分发困难 | uvx 首跑联网、子进程生命周期管理 | 移植期 fbank/wetext 需 Rust 等价物（见 §4） |

## 4. 推荐方案：两步走

### 第一步（本周可完成）：Python 一次性验证 harness —— 只做工具，不做分发

用 **uvx**（AGENTS.md 允许 uv）临时运行 `index-tts-2.5-onnx`，产出三样东西：

1. **粤语音质主观样本**：用 votex 场景的真实文本（有声小说章节片段）合成，人工评估口音自然度——验证「词表有 yue」是否等于「粤语可用」。
2. **金标 fixture**：按 docs/20 金标对齐方法论，录制粤语文本的 greedy acoustic tokens / mel 中间产物，供第二步 Rust 移植做逐级对齐验收（沿用 CosyVoice F67 模式）。
3. **性能基线**：CPU RTF 实测（fp32 GPT 自回归步进的开销）。

```bash
# 临时环境，不进项目依赖
uvx index-tts-2.5-onnx synth --ref voice.wav \
    --text "你好，今天天气几好。" --out out_yue.wav
```

**决策门**：粤语样本主观评估不达标 → 止损，2.5 接入缓行，方言路线转向 Qwen3-TTS/CosyVoice3 评估。

### 第二步（验证通过后）：Rust `IndexTts25Provider` 原生移植

- **蓝本**：PyPI 参考实现逐行对照（其全部数值已被证明 bit-exact），gitcode 镜像 IndexTeam/IndexTTS-2.5 官方实现兜底歧义。
- **复用现有基建**：`TtsProvider` trait（P2-17 后全 `&self`）、`EngineState<T>`、`InferenceGate`、`Arc<AtomicBool>` 取消、`model_registry` 清单与 SHA256 校验、`find_onnx_file` 候选表（新增 `IndexTts25` 引擎分组）。
- **需要落地的 Rust 等价物**（移植的主要工作量）：
  - `tiktoken` 词表加载：自定义 tiktoken 文件格式可手写 BPE loader 或用 `tiktoken-rs` crate（fancy-regex 依赖需审查体积）；
  - `kaldi-native-fbank` → w2v-bert 特征提取的 Rust 实现（现有 whisper mel 代码可改造，但需对照对齐）；
  - `wetext`（文本归一化）→ 排查 votex 现有 TTS 文本正则覆盖面，缺项补充；
  - `gpt_prefill`/`gpt_step` 自回归循环 → 与现有 indextts2 AR 循环同构，`emo_vec.onnx` 情绪向量接入对应 AGENTS.md 的情感合成目标。
- **registry 清单**：新增 `indextts-2.5-onnx.yaml`（源指向 yunfengwang HF 仓库 + hf-mirror），fp32 约 5.1G，与现有 IndexTTS-2 清单并存、用户按需下载。
- **金标验收**：对齐第一步录制的 fixture，token 级完全一致后合入。

### 明确不做

- **不引入官方 Python 栈**（torch + GPU，违反三项架构约束）。
- **不引入 libwaifu/flint**（C++ 构建链破坏纯 Rust 栈）。
- **不删除现有 IndexTTS-2 适配器**（4.5G 已在用户本地，2.5 验收通过前是唯一可用引擎）。

## 5. 风险登记

| 风险 | 影响 | 缓解 |
| :--- | :--- | :--- |
| 粤语实测音质不达标（yue 非官方一等语种） | 方言目标落空 | 第一步决策门止损；方言主路线不押注单一引擎 |
| fp32-only，无量化空间 | 磁盘 5.1G + CPU RTF 偏高 | 第一步实测 RTF；复用现有 InferenceGate 并发控制；必要时后续评估社区 fp16 导出 |
| fbank/wetext Rust 移植偏差 | 移植结果与参考不一致 | 金标逐级对齐（tokens → mel → wav SNR），沿用 F67 方法论 |
| 上游社区仓库变动（yunfengwang 为个人维护） | 下载源失效 | registry 支持双源（HF + hf-mirror）；SHA256 锁定版本 |
| bilibili 模型许可（>1亿 MAU 或年营收超 10 亿需单独授权；医疗/军事/关键基础设施禁止部署） | 合规 | votex 桌面工具场景不触及红线，登记于 docs/00-安全设计.md 即可 |

## 6. 结论

「为了粤语引入 Python」这个命题**不成立**——粤语能力在权重与词表里，不在 Python 里。正确的问法是「为了粤语，选哪条移植路径」：

1. **Python 的正确角色**：用 uvx 跑 `index-tts-2.5-onnx` 做**一次性**音质验证与金标录制，不进分发链。
2. **正路是 Rust ONNX 原生移植**：架构零破坏、工程量受控（有 bit-exact 蓝本）、与现有 5 引擎适配器同构。
3. **先过决策门再动工**：粤语样本实测不达标就止损——2.5 对方言路线图的价值需要先被证明，而不是被假设。

## 7. 粤语 PoC 实测（2026-10-05）

管线已在本机端到端跑通（8 逻辑核 CPU、纯 ONNX Runtime、torch-free），产出两段粤语合成音频：

### 7.1 实测数据

| 项 | 第 1 次（默认参数） | 第 2 次（`--lang yue --threads 8`） |
| :--- | :--- | :--- |
| 文本 | 粤语书面语 47 字 | 同左 |
| 参考音色 | `tmp/cosyvoice_official_zhprompt.wav`（普通话 prompt，跨语种克隆） | 同左 |
| 输出音频 | 9.46 s | 10.46 s |
| 模型加载 | 33.1 s | 38.6 s |
| 说话人克隆 | 40.3 s | 19.5 s |
| 合成合计 | 403.3 s | 533.9 s |
| **RTF（CPU）** | **42.6** | **51.0** |
| 阶段耗时 | gpt 35.6 / cfm **286.5** / bigvgan 80.9 | gpt 69.7 / cfm **343.8** / bigvgan 119.9 |

关键结论：

1. **`--lang yue` 是官方支持的语种参数**（zh/en/ja/yue 四选一），粤语路径在 ONNX 导出中完整保留。
2. **CFM（flow-matching DiT）是绝对热点**（占合成 71%~64%），其次 bigvgan 声码器（20%~22%）。
3. **8 线程反而更慢**（8 逻辑核为超线程，DiT 卷积算子在线程争抢下吞吐下降）——默认参数更快，RTF ≈ 42。
4. **CPU RTF ≈ 42~51 意味着 1 秒音频要合成 40~50 秒**——比现有 IndexTTS-2 适配器慢一个量级。这是**决策门的核心负面信号**：离线有声书场景（长文本批量合成）在纯 CPU 上不实用；除非 (a) 用户有 GPU，(b) 上游出 fp16/int8 可行导出，或 (c) `--n-timesteps` 降步数换取质量。
5. 缓解杠杆（未实测）：`--n-timesteps` 降步数、长文本分段后并行、每音色缓存 speaker embedding（clone 阶段 19~40s 可省）。

### 7.2 待人工验收

`tmp/indextts25_yue.wav` 与 `tmp/indextts25_yue8.wav` 需**人工试听**判定粤语口音是否自然（词表含 yue ≠ 音质达标）。若判定通过，进入 §4 第二步（Rust 移植）；若不达标，方言主路线转向 Qwen3-TTS / CosyVoice3 评估。

### 7.3 Windows 环境坑（已解决，供复现）

huggingface_hub 在无开发者模式的 Windows 上创建符号链接失败，**快照文件全部为 0 字节但报下载成功**，ORT 加载报 `ModelProto does not have a graph`。修复方式：按仓库 API 的 `lfs.oid`/`oid` 映射 blob 名，发现 5 个文件（bigvgan 450M、semantic_model.data 1.6G 等，共 2.2G）blob 实为 0 字节，用 curl `-C -` 断点续传补齐后复制到快照目录。votex 若用 registry 驱动下载（项目现有 `ModelRegistryLoader` + SHA256 校验），可完全绕开 hf_hub 的这个坑——这也是「Rust 原生移植」路线的附带优势。
