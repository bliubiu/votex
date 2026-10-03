# 更新日志（CHANGELOG）

> 项目：声阅（votex）—— Rust 离线 TTS / ASR / 翻译 / OCR 多引擎工作台
> 格式参考 Keep a Changelog；按日期倒序记录功能、修复与验证结论。

## [2026-10-03] CosyVoice 复述修复 + 三引擎质量对标

### TTS / CosyVoice 3.0
- **F72 回声剥除（新）**：新增 `strip_prompt_echo()`——子段 LLM 生成后，扫描开头 8~80 token 与 `prompt_speech_tokens` 的 LCS 占比（阈值 0.6），命中即剥除再送 flow。复述段语音 token 与 prompt 高度有序重合（同一文本同一音色的重新生成），剥除后复述音频物理消失（1 token ≈ 40ms）。
  - e2e 四轮对照（150 字 / 5 子段）：无剥除 90.4% → **前缀剥除 96.5% / 98.2%**（历史最高，超过 Qwen3 的 92.3%）
  - F72b 实验：头尾交替剥除 73.7%（尾部误伤正文结尾），已回退，实验记录保留于代码注释
  - 残留（影响小）：极短子段（~10 字）交错回声抓不到；长子段尾部偶发幻觉漂移
- **CosyVoice 长文本管线修复**（10-02 引入、10-03 验证）：`split_text_chunks()` 按句读切 ≤60 字子段（对齐官方 frontend）、`max_new_tokens` 收紧至 `text_tokens×10` 抑制复读（740→370 token）、子段进度 `eprintln` 输出；长文本 OOM 与复读循环消除
- 采样恢复**官方纯 top-k**（撤销自加的温度/重复/prompt 三项惩罚，F67 取证结论）；移除目标文本 `“”` 引号包裹
- **F67 复核**：v1.5 审计判定「完全忽略输入文本」，但 10-03 三轮 e2e 实测内容覆盖率 90.4~98.2%，当前构建下内容合成正常（F67 结论待按当前构建重新取证）

### 引擎选型结论（三引擎同文本对比）
| 引擎 | 476 字 ASR 往返覆盖率 | 速度 | 听书管线定位 |
|---|---|---|---|
| Qwen3-TTS 0.6B | 92.3% | ~30 分钟/章 | ✅ **主引擎推荐** |
| Kokoro-82M-zh | 80.1% | <1 分钟 | 快速草稿/预览 |
| CosyVoice 3.0 | **96.5~98.2%**（修复后） | 20-25x 实时 | 音色克隆实验特性（速度/复述残留待改进） |

### 运行时 / 性能
- ort 升级 `2.0.0-rc.13` + `load-dynamic` + `models/runtime/onnxruntime.dll`（ORT 1.30）；图优化级别 `Level3`→`All`（`ORT_ENABLE_ALL`，此前 Level3 实际映射 `ORT_ENABLE_LAYOUT` 少一档 Attention/LayerNorm 融合）
- CosyVoice LLM decode 单步 425ms → **138ms**（对齐 Python 官方 122ms，9 倍差距破案）

### E2E 验证资产（docs/20 v1.4/v1.5）
- `pipeline` 命令修复 F62（引擎映射收敛 domain 层单一权威 `parse_tts_engine`）/ F63（subtitle 管线不可用）/ F64（m4b 章节表 JSON）/ F66（日志噪声）
- `check_pinyin_test.rs` 阻断编译修复（F68）、日志换行/初始化幂等（F58/F59）
- 审计与取证：F67~F71 登记，`cargo check --workspace --all-targets` 恢复通过

### 翻译 / OCR 全量验证（集成测试，真实模型推理）
- **翻译 4 个离线引擎全绿**：HY-MT-1.5 7 passed（128.8s）、M2M-100 7 passed（50.4s）、NLLB-200 6 passed（45.8s）、Opus-MT 8 passed（22.9s，含双向往返）——中↔英句子/段落/往返全覆盖
- Qwen-MT 为云端 API（`DASHSCOPE_API_KEY`），无 Key 时 5 用例静默跳过（预期行为）
- **OCR 三档对比**（真值 `你好世界测试文字`）：v6-tiny conf 0.96 精确命中；v5-mobile conf 0.99 精确命中（置信度最高）；v4 `你→尔` 1 字错误（87.5%，已知模型精度限制）。CLI 中英混排图 3 行全对，任务持久化正常

### Pipeline 管线 E2E 全量复验
- **三条管线通过**（kokoro/zf_001，600 字符章节样本）：`audiobook`（wav + chapters.json + segments/，ffprobe 100.025s）、`audiobook-sub`（wav + 章节表 + SRT，4 条字幕时间轴与音频精确对齐）、`subtitle`（音频→SRT，内容与源音频一致）
- 新登记缺陷：**F73**（管线文本输入未走 EncodingDetector，GBK/非法 UTF-8 报英文错误，与 TTS 单命令行为不一致）、**F74**（chapters.json 伪章节——正文行被当章边界且标题丢字，根因在管线分段环节待取证）

### Pipeline 缺陷修复 F73 / F74（10-03 下午）
- **F73 ✅**：`pipeline_use_case.rs` 文本入口接入 `EncodingDetector::read_text_file`（UTF-8/GBK 自动检测）；GBK 编码样本直接跑通全管线
- **F74 ✅**：管线分段阶段改走 `build_synthesis_plan`（与 tts 直连同链路）；章节表只认 `is_new_chapter` 真实边界，无标记兜底「全文」单章。chapters.json 从「2 章（含伪章节）」变为仅真实章节「第0001章 女儿？」
- **连带修复**：纯标题输入（标题独立成段后 body 为空）导致合成计划 0 段报「没有可输出的音频段」——`build_synthesis_plan` 补「纯标题输入标题自身成段」（常规 + 多角色双路径），新增单测 `计划_纯标题输入不丢段`
- 回归：`votex-app` lib **94 passed**；audiobook / audiobook-sub / GBK 输入 audiobook 三轮 E2E exit 0

### 测试素材基准（TTS / ASR E2E）
以下文本文件为 TTS / ASR 功能与 E2E 验证的**标准输入素材**（真值来源，随仓库 tmp/ 分发）：
- `tmp/e2e_ch1.txt`（21,747 字节，含「第0001章」标题）——章节感知切分、m4b 单章闭环、多引擎 e2e 对比（`tts_e2e_multi_engine_test.rs` 即取其节选）
- `tmp/novel.txt`（48,135 字节）——长文本 TTS / 章节切分
- 配套：`tmp/e2e_sample.txt`（≈3455 字 13 段）、`tmp/e2e_roles.json`（多角色映射 schema）
- 评估方法：TTS 合成 → SenseVoice ASR 往返 → 与原文 LCS 字符覆盖率（可懂度代理指标）

## [2026-10-02] 多引擎 e2e 建设与构建陷阱排雷

- 新建 `tts_e2e_multi_engine_test.rs`：Qwen3 / CosyVoice 同文本 e2e（合成→WAV→ASR 回验→覆盖率断言）+ decode 性能基准
- CosyVoice e2e 四处修复：`max_new_tokens` 动态上限、中文 prompt（Kokoro zf_001 合成 8.03s `zh_prompt.wav`）、flow conds 填入真实 slaney log-mel、ByteLevel `add_prefix_space=false`（与官方 GPT2Tokenizer 34 token 逐位一致）
- **构建陷阱排雷（F55）**：`votex` bin 在根 package，`cargo build -p votex-cli` **不产出 exe**；cargo Edit/Build mtime 竞态致指纹"假最新"——二进制验证方法：搜新日志字符串
- Qwen3 0.6B 整章试产：完成 7/19 段后按用户决策停止（4 核 CPU 每段 ~30 分钟），断点续转验证可用
- `ensure_ort_dylib_path()` 插入 6 个直调 `Session::builder()` 的测试文件

## [2026-10-01] 翻译质量工程 + 长文本资源治理 S1~S5 + 竞品补全 T1~T6

### 翻译（P0~P2 全部落地）
- 修复模型目录名不一致（`hy-mt-1.5` / `m2m-100` / opus-mt 双向 registry 重写）
- 新增 `TranslationModelPool`（Arc 会话池，GUI/CLI 共享，不再每次翻译重建 GB 级 Session）
- trait 扩展：`load/unload/supported_pairs/translate_batch/translate_with_glossary/translate_stream`
- 翻译质量工程：术语表占位符方案（`\u{E000}idx\u{E001}` 前替换后回填）、长文本按句分段、md5 LRU 翻译缓存、简繁转换、`translation_pipeline` 全编排（进度/流式回调）
- SQLite `translation_tasks` 表；CLI `translate` 升级 + `glossary-init` + `batch translate`
- **关键发现**：DirectML（EP=auto）本机段错误 → `application.yml` 默认 `execution_provider: cpu`
- 8 处调用改走 `OrtSessionFactory::create_raw*`，`inference.*` 配置对翻译生效

### 长文本 TTS 资源治理（S1~S5 收官）
- `ResourceMonitor`（sysinfo 三级内存压力）+ `InferenceGate`（RAII 信号量 + 提交预检）
- `tts_use_case` 重写：流式输出（段完成即写盘，6GB 峰值→单段级）、断点续转 session（progress.json 哈希校验）、取消令牌
- `StreamingWav` 追加写 + RIFF 回填；`wav_to_format` 真编码（m4a=AAC/flac/ogg，修复假格式）
- S5：章节感知切分、内联停顿标签 `[[pause:300]]`、**m4b 章节输出**（FFMETADATA + AAC 192k）、多音字词组消歧（~70 条 + 「一/不」变调）、数字→中文读法
- `batch_tts_use_case` 重写：共享单模型实例、工作池、目录/GBK 输入、条目级进度

### 竞品补全（对标微信读书/番茄畅听，T1~T6）
- T1 语义断句（引号感知）；T2 多角色（`split_dialogue` + `RoleVoiceMap`，`--role-map`）
- T3 文档解析 EPUB/DOCX/PDF（quick-xml 实体引用坑修复）；T4 音色库（`voice add/list/remove`）
- T5 后处理（loudnorm EBU R128 / atempo）；T6 播放器（倍速/章节跳转/高亮/SQLite 续听）

### 文档产出
- `docs/17-翻译模块分析与LunaTranslator对比.md`、`docs/18-有声书长文本优化与资源治理设计.md`、`docs/19-竞品有声书TTS对比与差距分析.md`、`docs/20-功能E2E验证测试计划.md`（v1.0→v1.6，100 条用例 + F1~F72 缺陷台账）

## [2026-10-01 之前] 基础能力

- 五 crate 架构（domain/app/infra/cli/gui，kube-rs 风格）
- TTS 四引擎同 trait：Kokoro-82M-zh（G2P/拼音/多音字）、Qwen3-TTS、CosyVoice 3.0、IndexTTS2
- ASR：SenseVoice / Whisper / FireRed（三层候选回退探测）
- 翻译：Opus-MT（en↔zh）/ M2M-100 / NLLB-200 / HY-MT-1.5 / Qwen-MT
- OCR：PaddleOCR v4 / v5-mobile / v5-server / v6-tiny / v6-small / v6-medium（det→cls→rec）+ EasyOCR
- G2P：拼音、多音字、英文 phoneme、方言映射（Zhuyin/IPA）
