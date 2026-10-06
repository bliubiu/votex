# 22-Orukeet 多语言 ASR 模型评估分析

> 评估对象：Oruk AI（联合 Stanford / Cambridge / OpenWhispr / Hoid）2026-09 发布的开源多语言语音识别模型 **Orukeet**（arXiv: 2609.10054）。
> 评估视角：① 模型技术定位与基准水位；② 对 votex（CPU 离线、ONNX Runtime、4 核桌面机、中文场景为主）的接入价值与成本。
> 信息来源：Hugging Face 模型卡（oruk/orukeet）、GitHub 仓库 Oruk-AI/orukeet、sherpa-onnx 官方示例、arXiv 技术报告（2026-10-03 检索）。

## 1. 总体定位

Orukeet 是 **NVIDIA Parakeet TDT 0.6B v3 的多语言精调版**：25 种欧洲语言的离线（非流式）ASR，定位为本地转写的 drop-in 升级，已被 OpenWhispr 桌面应用列为推荐本地模型。

```
Parakeet TDT 0.6B v3 (FastConformer 24层 + TDT transducer, 627M 参数)
        │  结构手术：24,576 个 9-tap 时间卷积核中
        │  选出拟合误差最低的 12,288 个 → 替换为拟合 Gabor 核并冻结
        ▼
剩余 626.9M 参数在多语言/多口音数据上继续训练 → Orukeet r3
```

**核心创新（Gabor 核替换）**：将编码器一半的时间深度卷积核替换为解析拟合的 Gabor 函数 `g(t)=A·exp[-(t-μ)²/2σ²]·cos(2πf(t-μ)+φ)`，拟合后**冻结**（110,592 个 tap 固定不动），其余参数正常训练。Gabor 核在导出时物化为普通卷积权重——**推理侧无需任何自定义算子**，这是它能直接进 sherpa-onnx 的关键。

## 2. 基准水位（vs 原版 Parakeet，同解码设置配对比较）

| 测试集 | Parakeet WER | Orukeet WER |
|---|---|---|
| LibriSpeech test-clean | 1.53% | **1.46%** |
| LibriSpeech test-other | 3.14% | **2.86%** |
| FLEURS English | 4.28% | **3.82%** |
| FLEURS 25 语言 pooled（20,146 条） | 11.01% | **9.85%**（相对降 10.6%） |
| 口音/域 pooled（47 splits） | 16.72% | **15.25%** |

74 个测试 split 中 61 个优于 Parakeet，25 种语言中 23 种占优。

**必须附带的水分提示**：r3 最终适配**直接在 LibriSpeech test-other 上跑了 3 遍**（168 次 AdamW 更新），且 checkpoint 选择也用 test-other——LibriSpeech 数字存在训练集污染，作为公开基准引用时应打折看待。

## 3. 关键工程事实

| 项 | 内容 |
|---|---|
| 语言 | **25 种全为欧洲语言**（英法德西俄葡意等），**不含中文** |
| 流式/离线 | 离线整段转写（streaming: false）；token 级时间戳；源模型支持语言检测 |
| 不提供 | 说话人分离（diarization）、标定置信度、翻译 |
| 格式 | NeMo 源 2.51GB / 原生 Q8 GGUF 714MB / F16 1.30GB / **sherpa-onnx INT8 ONNX（压缩包 487MB，解压 672MB）** / Transformers FP32 / Core ML |
| sherpa-onnx 接入 | 官方提供标准 Parakeet TDT v3 布局（encoder/decoder/joiner.int8.onnx + tokens.txt），`OfflineRecognizer.from_transducer(model_type="nemo_transducer", feature_dim=128, num_threads=4)` 即可加载；需 sherpa-onnx ≥ 1.13.4；k2-fsa 仓库已收录官方示例 `python-api-examples/orukeet.md` |
| 速度参考 | INT8 在 M5 Max 上 160 段中位延迟 390ms（原版 Parakeet 428ms）；4 核老平台 CPU 需实测，量级上 0.6B transducer INT8 离线转写可行 |
| 许可证 | 代码 MIT；**模型权重（含 ONNX 导出）CC BY-SA 4.0**（须保留 NVIDIA Parakeet 归属声明） |

## 4. 对 votex 的接入评估（核心结论）

结合 votex 约束：CPU 离线、4 核（i7-4710MQ）、底层统一 sherpa-onnx/ONNX、单二进制、中文听书/字幕为主场景。

### 4.1 决定性短板：无中文

Orukeet 的 25 种语言**全部是欧洲语言**，中文场景直接不可用。votex 现有 ASR 引擎（SenseVoice / Paraformer / WeNet / FireRed / Whisper）对中文的覆盖与质量远超 Orukeet 能提供的任何价值——**中文主场景没有引入动机**。

### 4.2 其余维度

| 维度 | 评估 |
|---|---|
| 运行时匹配 | ★★★★★ 官方 sherpa-onnx INT8 导出，`nemo_transducer` 离线 transducer 路径与 votex 现有引擎管线同构，集成成本接近"加一个模型注册项" |
| 算力 | ★★★★☆ 0.6B INT8 约 672MB 内存，4 核 CPU 离线整段转写可行（无 R2T2 那种逐 chunk 连续解码压力） |
| 能力匹配 | ★★☆☆☆ 非流式 + 无中文 + 无 diarization；对 votex 唯一增量是"欧洲语言字幕/转写"备选 |
| 许可 | 权重 CC BY-SA 4.0（非 copyleft 代码污染，仅要求权重分发带归属），可接受 |

### 4.3 结论

**不建议当前接入。** 中文主场景零收益；但与 R2T2 不同，Orukeet 属于"低成本可选"档：sherpa-onnx 官方支持使其成为 votex 未来若做**欧洲语言字幕**特性时的首选候选（比 Whisper-small 同量级精度更高、CPU 友好），登记为跟踪项即可。

### 4.4 可借鉴的设计（零成本收益）

1. **"结构先验 + 冻结"改造范式**：Gabor 核拟合-替换-冻结的做法对 votex 无直接应用，但"导出时物化为普通算子、推理零改动"的导出纪律值得在自定义 ONNX 导出（如 CosyVoice flow/HiFT）中对齐——**导出图永远不引入自定义算子**
2. **配对基准纪律**：所有对比"同音频、同解码设置、逐 split 配对发布"，且官方主动披露 test-other 污染事实——votex 三引擎对比报告（tmp/tts_engine_comparison）可借鉴其"分数与 caveats 绑定发布"的写法
3. **manifest + SHA-256 校验下载**：`manifest.json → 校验 → 下载 → 复核 size/sha256` 的模型分发流程与 votex 的 model_registry / download_repo 机制同构，可直接对照加固

## 5. 参考链接

- 模型卡：https://huggingface.co/oruk/orukeet （不可变 revision 555136b…）
- sherpa-onnx 导出说明：https://huggingface.co/oruk/orukeet/blob/main/onnx/README.md
- 官方代码：https://github.com/Oruk-AI/orukeet
- sherpa-onnx 官方示例：https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/orukeet.md
- 技术报告：arXiv 2609.10054《Orukeet: Multilingual ASR with Frozen Gabor Kernels》
