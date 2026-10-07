# 更新日志（CHANGELOG）

> 项目：声阅（votex）—— Rust 离线 TTS / ASR / 翻译 / OCR 多引擎工作台
> 格式参考 Keep a Changelog；按日期倒序记录功能、修复与验证结论。

## [2026-10-07] 实时听写 + 说话人分离 + 词级时间戳对齐（CLI/GUI 对等）

### 实时/流式听写（新功能）
- **领域层**（`domain/asr/streaming.rs` 新模块）：`StreamingAsrProvider` /
  `StreamingAsrSession` / `StreamingUpdate`（partial + 端点定稿 + 累计时长），
  会话对象 `Send` 可整体移交工作线程；新增 `EngineKind::StreamingZipformer`
  全链路注册（能力自描述 / 显示名 / 解析）
- **引擎**（`infra/asr/streaming.rs` 新模块）：流式 Zipformer transducer
  （sherpa-onnx `OnlineRecognizer`，中英混合 int8），端点检测自动分句
  （句尾静音 2.4s / 连续语音后 1.4s）；共享识别器经 `Arc<Mutex>` 串行化
- **麦克风采集**（`infra/audio/capture.rs` 新模块）：cpal 默认输入设备 →
  单声道下混 → 有状态线性插值重采样到 16kHz（`LinearResampler`，分段调用
  与整体重采样一致的契约测试）；`cpal::Stream` 非 Send——采集句柄全程
  留在 worker 线程内启停
- **用例**（`app/use_case/dictation_use_case.rs` 新模块）：模型加载 → 麦克风
  → 流式解码 → `DictationEvent { Started/Partial/Final/Stopped/Error }`
  事件回调；停止时冲刷残余音频、全文落盘 UTF-8 txt
- **CLI**：`votex dictate [-o 转写.txt]`——`○` 未定稿单行覆盖刷新、`●`
  定稿换行，Ctrl+C 置停止令牌后落盘
- **GUI**：新「实时听写」页（语音处理分组）——模型/保存路径选择、开始停止、
  中间结果灰字实时刷新、定稿逐段追加可编辑转写区；事件走页面专属 channel
  （TaskEvent 无流式增量语义），每帧非阻塞拉取

### 说话人分离（死代码修复 + 用例整合落地）
- `speaker_diarization.rs` 此前 trait 签名与领域层脱节、sherpa 结构体写法
  错误，**整个工作区编译不过**；按 sherpa-onnx 1.13 真实 API 重写：
  `OfflineSpeakerSegmentationModelConfig` 为结构体（pyannote + num_threads）、
  `set_config` 动态覆盖聚类参数（固定人数 / 阈值自动二选一）、模型目录
  自动发现（兼容 tar.bz2 解包子目录）；`recognize` 明确报错引导走
  `diarize_with`
- `AsrUseCase::recognize_ex` 扩展入口：识别后对全音频分离，每条字幕按
  **时间重叠最大化**标注 `SubtitleEntry.speaker`（领域字段，serde default
  向后兼容）；SRT/TXT 输出 `[说话人N]` 前缀（LRC 不标），导出
  `<output>.diarization.json` 说话人时间轴
- CLI `votex asr --diarize [--num-speakers N]`；GUI 识别页勾选项 + 人数设置

### 词级时间戳对齐（伪时间戳 → 真实对齐）
- `votex_infra::asr::recognize_output_from` 统一组装各引擎结果：Whisper /
  Paraformer / WeNet / FireRedASR（CTC 类）等 sherpa 引擎提取**真实 token
  级时间戳**（词终点 = 下一词起点，清理 `▁` 占位与 `<|...|>` 特殊标记）；
  SenseVoice 等无时间戳引擎回退整段伪时间戳（`start == end == 0` 契约）
- `group_words_into_entries` 组句：句尾标点 / 24 字符 / 6 秒 /
  600ms 静音间隔断句，CJK 连写、英文词间补空格；切片偏移无条件累加
  （docs/23 A2 语义保留）；导出 `<output>.words.json`（绝对时间轴词列表）
- CLI `votex asr --words`；GUI 识别页勾选项

### 配套
- 模型清单：`streaming-zipformer-zh-en.yaml`（HF + hf-mirror 双源，int8 必需
  + float 可选）、`speaker-diarization.yaml`（Pyannote 分割 tar.bz2 逐成员
  提取 + 3D-Speaker ERes2Net 嵌入单文件）；落地目录与 provider 查找一致
- 文档：`docs/29-实时听写与说话人分离与词级时间戳落地.md`
- 验证：全工作区编译零错误；domain 211 / infra 483 / app 149 / cli 58 /
  gui 40 单测通过；重采样 44.1k→16k 长度/分段一致性/立体声下混专项测试

## [2026-10-07] CTranslate2 翻译加速后端（`ct2` feature 门控双形态）

- **路线修正**：CTranslate2 无官方 C API（仅 C++/Python），原骨架的
  `libloading` 运行时直调假设不可行；改用 [`ct2rs`](https://crates.io/crates/ct2rs)
  绑定（vendor 模式源码编译 C++ 库，需 CMake + MSVC + ninja）。
- **`ct2` feature 门控双形态**（`translation/ctranslate2/` 目录模块）：
  - `imp_enabled`（`--features ct2`）：int8 量化真实推理，`Translator` +
    双 SentencePiece 分词（自动 `</s>` 追加，等价 opus_mt 手写流程）；
    方向路由对齐 opus-mt（目录名 `en-zh` 判向、`Auto` 跟随、方向不符
    `UnsupportedDirection`）；`supported_pairs` 动态声明、`max_input_chars=256`
  - `imp_fallback`（默认构建）：模型探测 + 「未启用 ct2 feature」构建引导
    提示；默认构建零影响（不链接 C++ 库）
  - 双形态契约测试（未加载 → `ModelNotLoaded` / 空文本 → `EmptyText`）；
    引擎列表与工厂分支按 feature cfg 接线（`ct2` 别名归一化已有）
- **模型转换脚本 `scripts/convert_ct2.py`**（uv + PEP 723 内联依赖，禁 pip）：
  官方转换器 int8 转换 + 补齐分词文件（官方转换器不复制 `source.spm`/
  `target.spm`，ct2rs 加载必需）+ 产物枚举校验；HF 镜像兜底。
  实测 opus-mt-zh-en：int8 model.bin 75.1 MB（原始约 1/4）
- **registry 条目 `ct2-opus-mt-zh-en.yaml`**：分词文件指向 Helsinki-NLP
  真实 URL；model.bin / shared_vocabulary.json 为本地转换产物（sources 为
  源模型溯源链接，下载入口即转换脚本），SHA256 已回填
- **集成测试 `ctranslate2_translation_test.rs`**（`--features ct2` 编译，
  金标用例 `slow-models` 门控）：方向路由 / 批量一致性 / 未加载契约
- **文档**：`docs/engines/tr-ctranslate2.md` 重写（准入清单 + 行为契约 +
  构建与转换指引）；capability 备注改「翻译加速后端（需 --features ct2）」

## [2026-10-07] docs/27 P1 落地：克隆产品化闭环 + 视频配音流水线

### P1-a 克隆产品化闭环（参考音频工程 + 音色库 GUI）
- **参考音频工程**（`infra/audio/ref_audio.rs` 新模块）：
  - `best_window`：按短时能量选「语音最多」的窗口——替代引擎侧静默截头；
    `voice add` 新增 `--max-ref-seconds`（默认 15，0 = 不截取），超上限时
    无转写自动截取、**有转写显式拒绝**（借鉴 VoiceStudio `clone_ref_too_long`
    契约：截取窗口与转写无法对应，宁可拒绝不静默出错）
  - `silence_ratio`：全静音参考入库直接拒绝（此前会静默产出失真音色）
  - 元数据新增 `trimmed_from_ms` / `ref_strategy` 字段（serde default 向后兼容）
- **IndexTTS-2.5 音色库打通**：参考音频查找顺序 `prompts/<id>.wav` →
  统一音色库 `models/voices/refs/<id>.wav`；`list_voices` 合并两处并去重——
  `voice add` 入库音色免手动复制即可用，音色未命中报错改指 `voice add`
- **修复**：`voice_library::list()` 只认 `*.meta.json`（此前扫 `*.json`
  会把目录里其他 JSON 误读成音色元数据）
- **GUI 音色库页**（新页面「音色库」）：入库（选文件/命名/转写/降噪/上限）、
  列表（时长/采样率/降噪/转写状态）、试听（复用播放器）、删除二次确认；
  TTS 页音色下拉动态化——Kokoro 扫音色池、克隆引擎合并音色库列表
  （无转写音色标注「仅 IndexTTS-2.5 可用」），替代此前硬编码 8 个音色

### P1-b 视频配音流水线（含时长拟合、二次 ASR 质检）
- **ffmpeg 原语**（`infra/video/dub_ops.rs` 新模块）：提取音轨（归一 24kHz
  单声道）、WAV→WAV 变速（atempo 多级串联 0.25~4.0）、音轨替换/混入视频
  （视频流 `-c:v copy` 不重编码；混音 amix normalize=0 保留各自增益）
- **`VideoDubUseCase`**（`votex-app/use_case/video_dub_use_case.rs` 新用例）：
  提取音轨 → ASR 转写 → 中文分句 + 按字数比例分配原时间轴 →（可选）
  `translate_batch` 逐段翻译 → 逐段 TTS（段边界取消检查）→ **时长拟合**
  （起点对齐原时间轴；超长段按 `max_tempo` 上限变速；前段溢出顺延不追赶）
  → 流式拼装配音轨（`StreamingWav` 段间补静音）→ **用实测段时长重建字幕**
  （非字数估算）→（可选）二次 ASR 质检（字符二元组相似度打分 + 质检报告）
  → 音轨替换/混入视频。设计取舍：配音是段粒度操作，不硬塞现有
  StageFn 文件粒度注册表，独立用例 + 阶段进度经 ProgressEvent 上报
- **CLI `votex dub`**：全参数入口（--translate / --max-tempo / --verify /
  --keep-background 等）
- **GUI「视频工具」页**：模式切换「AI 短视频生成 / 视频配音」，配音模式
  全参数表单 + 取消令牌（复用视频页事件通道）
- **测试**：ref_audio 5 例、voice_library 入库截取/拒绝/过滤 4 例、
  dub 用例分句/时间轴/拟合/相似度 8 例；infra 458 + app 135 全绿
- **已知边界**：音频参数不一致的段会显式报错（同引擎恒定不受影响）；
  视频减速 50/50 分摊（译制配音场景）留待后续；amix normalize=0 需
  ffmpeg ≥ 4.4，老版本请用默认替换模式

## [2026-10-07] 失败诊断脱敏面板（GUI 错误链统一收口）

- 落地 docs/27 第四节 B 的「失败诊断脱敏面板」：GUI 此前在结果面板原样
  透传 anyhow 错误链（含模型绝对路径，存在泄露风险），现改为渲染期统一
  收口——失败类消息不再直出原文。
- **领域层 `votex_domain::shared/diagnosis.rs`**（零 IO，纯字符串处理，
  9 个中文命名单测）：
  - `diagnose()` 模式匹配错误链 → 中文一句话结论 + 「为什么？」原因解释 +
    引擎文档页映射（模型缺失/校验失败/vocab_size 不兼容/音色缺转写/
    参考音频契约/鉴权失败/网络/内存不足/ffmpeg/磁盘/权限/超时/取消/兜底）
  - `sanitize()` 抹去 Windows 盘符 / UNC / Unix 绝对路径（→ `<路径>`）
    与密钥碎片（`sk-***`、`Bearer ***`、`key=***`）；相对路径与 URL 保留
- **GUI `page_template::result_message` 收口**：成功与校验提示保持原渲染；
  失败类消息渲染诊断面板（红色结论 + 「为什么？」折叠 + 「详情（已脱敏）」
  等宽折叠 + 「打开引擎文档」按钮跳转 `docs/engines/<页>`），全部 8 个
  页面自动生效，state 结构零改动
- **顺带修复既有编译错误**：`SubtitleEntry` 新增 `speaker` 字段后 4 处
  初始化未同步（domain/app 测试与 infra 字幕策略）；speaker_diarization
  对齐 sherpa-onnx 1.13 真实 API（`OfflineSpeakerSegmentationModelConfig`
  为结构体而非枚举）——`cargo test -p votex-domain` 211 通过，GUI/infra
  编译零错误零警告

## [2026-10-07] 引擎文档三件套（docs/engines/）

- 落地 docs/27 第四节 B 的「引擎文档规范化」：新增 `docs/engines/` 目录——
  README（能力矩阵一张表 + 新引擎四项准入清单 + 已知怪癖速查）+ 23 个引擎页
  （TTS 本地 4、ASR 本地 6 + 说话人分离、OCR 2、翻译 8、在线 TTS/ASR 2），
  每页六节：用途 / 依赖 / 模型来源 / 能力 / 已知怪癖 / 测试锚点。
- 事实澄清：本地 ASR（Whisper/Paraformer/Qwen3-ASR/FireRed/WeNet/分离）经
  **sherpa-onnx 绑定（crate 1.13）**加载，模型必须用 sherpa-onnx 官方转换包；
  CTranslate2 为 FFI **加速骨架**而非独立引擎（缺系统库须回落 ort 路径）；
  RapidOCR 与官方 v5 mobile 权重逐字节相同，仅作评测对照组。
- 准入清单四项：Windows 全量构建、金标准测试锚定（ASR 0.wav / TTS 闭环）、
  registry + EngineCapability 注册一致、写引擎页——不全不合并。
- 运行期权威数据仍以 `votex model list --json` 为准；EasyOCR 独立集成测试
  与引擎决策日志为准入清单遗留项。

## [2026-10-07] VoiceStudio 对标分析（docs/27，合并版活文档）

- 新增 `docs/27-VoiceStudio对标与借鉴分析.md`：对标开源全本地 ElevenLabs 替代品
  VoiceStudio（Electron + Python，30+ 引擎），盘点功能与引擎矩阵差距。
  两次对标调研已**合并为唯一活文档**（原 28 号对标分析并入后删除），
  并对照代码库修正事实：OCR 实际为 PaddleOCR/EasyOCR（RapidOCR 仅评估测试）、
  ASR 本地 6 引擎含 Whisper（FireRedASR v2 为同引擎 CTC 变体）、
  本地 ASR 经 sherpa-onnx 绑定（crate 1.13）加载、模型固定用官方转换包
  （FunASR 自导出缺 vocab_size 会 abort 进程）、
  WeNet 时间戳实测仅首尾正确；P0（serve/能力自描述）与 P1（克隆闭环/视频配音）
  状态列标注落地情况。
- 核心结论：不照搬其双运行时架构与碎片化引擎接入；可借鉴点按优先级排序——
  ① 每引擎一页文档 + 新引擎准入门槛（沉淀 FunASR vocab_size 等教训），
  ② 词级时间戳驱动的配音对齐（配音主链路已落地，此为对齐精度优化方向），
  ③ inline 情感标记（[笑]/[停顿]）管线级支持，④ 有声书章级缓存键，
  ⑤ `votex serve` 本地 API + 失败诊断脱敏面板，⑥ 悬浮窗听写（观察项）。
- 差异化定位：不拼引擎数量（646 语言广度），拼中文听感与单二进制离线开箱即用。

## [2026-10-06] TTS 能力缺口补全：情感接线 / CosyVoice 克隆音色发现 / 音色未命中报错

### 情感参数接线（此前 `TtsParams.emotion` 被所有引擎忽略）
- **Qwen3-TTS 1.7B（VoiceDesign）**：非 Neutral 的 emotion 生成英文指令语
  追加到 instruct（`emotion_instruct_clause`，13 种情感各有措辞；
  intensity ≥1.5 映射为 strongly 强化）——四引擎中唯一真实情感通道
- **0.6B / Kokoro / CosyVoice / IndexTTS-2.5**：无情感控制通道（预设
  embedding / 风格向量 / 参考音频条件 / 无情感 token），`EmotionWarnOnce`
  每引擎进程内告警一次（长文本分段不刷屏），其余情况静默放行
- 单测：指令语措辞与强度映射

### CosyVoice 克隆音色发现与解析（此前 `_voice` 被完全忽略）
- **list_voices**：default 之外返回克隆音色库（`models/voices/refs/`）
  全部音色，标注是否含转写——GUI/CLI 音色下拉从此可见克隆音色
  （对等性缺口：克隆音色库此前仅 CLI 管理、GUI 不可见）
- **synthesize**：voice ≠ default 时从音色库解析参考音频 + 转写文本；
  音色缺失/参考音频丢失/缺转写分别明确报错（CosyVoice 零样本克隆的
  prompt 必须配套转写才能条件 LLM）
- **voice_library**：`CloneVoiceMeta` 新增 `transcript` 字段（serde default
  向后兼容），`add()` 接受转写，新增 `find_meta()`
- **CLI**：`voice add --transcript "..."`；`voice list` 显示转写状态；
  使用提示更新为 indextts25/cosyvoice3 双引擎

### 音色未命中报错（审查 A3）
- `find_voice` 未命中时不再静默回退第一个音色（"音色名写错 → 用别的
  声音合成"），改为报错并列出可用音色（最多 10 个 + 总数）；
  多角色映射未命中的既有回退路径（warn + 旁白音色）保持不变
- 单测：命中/未命中报错信息含请求音色名与可用列表

### 验证
- `cargo test --workspace`：全绿（数字见文末提交记录）；clippy 零新增

## [2026-10-06] 小说角色扫描 CLI（docs/26 模块一 S3）

### 新增 `votex role` 子命令
- **`role scan`**：从整本小说提取角色候选表，消除「读完全书手工统计说话人 +
  手写 role_map JSON」的成本。支持 txt/md/epub/docx/pdf，**自动探测 GBK/UTF-8**
  - `--top N` 候选上限（默认 20，按台词数降序）
  - `--assign-voices` 自动分配音色，产出可直接喂给 `tts --role-map` 的映射表
  - `--out` 保存 JSON、`--json` 输出机器可读格式、`--models-dir` 指定模型目录
- **`role voices`**：按性别分组列出可用音色池
- 终端输出为中文表格（角色/台词数/性别/已分配音色/**首段台词**），末行直接给出
  后续合成命令；`--json` 供 GUI/脚本消费

### 新增能力
- **`kokoro::list_voice_pool(tts_models_dir)`**：枚举音色池**无需加载模型**，
  仅扫 `voices/*.bin`。此前 `list_voices()` 要求 provider 已 `load_model()`，
  角色扫描这类纯文本操作不该被迫加载 80M 模型
- **`role_scan_use_case`**：扫描逻辑全部下沉到 app 层（CLI 分层约束：
  业务逻辑不得写在命令里），GUI 角色页可直接复用
- `voice_display_name` 把音色前缀知识收口为唯一实现——原 `list_voices` 内联的
  6 分支 if 链已改为复用它，与 `VoiceGender`/`VoiceLocale` 的前缀推断合一

### 修复：CLI 首轮真实书稿验证暴露的3 个问题
- **噪声候选 `张重点头`(72 次)**：「张重**点头**没再问」——动作词不含助词，
  原`MODIFIER_TAIL_CUT`拦不住。动作词表并入 `NAME_TAIL_NOISE_WORDS`
  （词频取自真实语料：点头 1301 / 摇头 628 / 摆手 229 / 皱眉 71）
- **12 个角色全部同一音色 `zm_009`**：真实书稿说话人几乎不带称谓线索
  （12 个主要角色中**0 个**能定性），全落 `dialogue_default`。
  新增 `neutral_rotation_pool`——**男女交错 + 同性别跳号(步长 8) 等距取样**，
  保证不同角色听得出来不同（不做`i*STEP % len`绕回，那会产生重复项）
- **中文小说被分到英文音色 `af_maple`**：轮转池只按 gender 过滤，
  而英文音色字典序在 `zf_` 之前。改为**性别 + 语种**双条件过滤

### 能力边界（已写入 docs/26）
- 真实书稿的说话人**几乎不带称谓线索**，性别启发在小说文体上基本无效；
  `suggested_gender` 仅作提示，自动分配的正确目标不是「性别正确」
  而是「**声线可分辨**」。若书稿大量使用「爸爸」「爷爷」这类称谓（口语对话体），
  性别启发才有效——CLI 输出以「待确认?男声」如实告知用户
- CLI 报告 976ms（4.9MB GBK/ 1083 章）

### 测试
- `role_scan_use_case` 6 passed；`tts::role` 34 passed（新增3 个：轮转池/ 动作词剥离）
- CLI clap 参数测试 3 个 + 子命令清单纳入 `cli_定义完整性`
- 回归：domain 194 / app 123 / cli 47 / gui 40 全绿

### 文档
- `docs/02-CLI接口规范.md` 新增 §2.1「多角色配音：角色扫描命令」
- `docs/26` 补S3 实施记录

## [2026-10-06] 小说角色自动提取与音色自动分配（docs/26 模块一 S1+S2）

### 新增能力
- **`extract_role_candidates(text, top_n)`**：纯规则全书扫描角色候选（频次降序），
  输出 `RoleCandidate { name, mentions, gender, suggested_gender, first_chapter, sample_line }`。
  `sample_line` 为该角色**首段真实台词**，供 GUI 试听（用通用样句听不出音色是否贴合角色）。
  4.9MB GBK 全本（1083 章）实测 **0.74s**，全部候选均带可试听台词
- **`build_role_map(candidates, voices, assignments)`**：按「显式指定 → 性别+语种
  匹配 → 兜底」三级策略自动生成 `RoleVoiceMap`，可直接喂给既有
  `build_synthesis_plan`，自动生成的 role_map 覆盖全 53658 段的音色路由
- **音色元数据** `VoiceGender` / `VoiceLocale` / `VoiceMeta`：`zf_`/`zm_`/`xf_`（中文）、
  `af_`/`am_`/`bf_`/`bm_`（英文）前缀推断收口为**单一实现**，替代原先散落在
  Kokoro `list_voices` 与 GUI 硬编码下拉中的三处重复判断。`voice_id` 字段与既有
  音色串完全一致，**向后兼容**
- 方案与实施记录：`docs/26-小说制作链路落地方案.md`

### 修复：说话人识别噪声（真实全本暴露的既有缺陷）
首轮 E2E Top5 为 `张重(682)/笑着(190)/张重笑着(133)/许雨涵(96)/笑呵呵地(79)`
——**4/5 是噪声**。已修：
- `MODIFIER_TAIL_CUT`（着/地/的/了）截断 `X说着` 类形态的修饰语
- `VERB_TAIL_NOISE` 剥离截断后残留的动词字（`张重笑` → `张重`）
- `ADVERB_ONLY_WORDS` 整串黑名单（`悄悄`/`慢慢` 等长度合法的纯副词绕得过长度保护）
- `NAME_TAIL_NOISE_WORDS` 双字副词先剥离（`李经理赶忙` → `李经理`）
- `NAME_TAIL_NOISE` 单字粘连副词（`张重又说` → `张重`，此前会拆成两条候选
  并导致 `RoleVoiceMap` 匹配失效）
- `looks_like_chapter_marker` 过滤章节编号（`第三章`/`第3章`/`第十二回`），
  复用 `chapter::CHAPTER_UNITS`（提升为 `pub(crate)`）避免两处定义漂移
- **修复后 Top5**：`张重(982)/许雨涵(124)/芃芃(88)/胡慧芳(87)/庄语(75)`，全为真实角色

### 修复：音色分配的语种错配与丢失
- **中文小说旁白被分到英文音色**：按性别筛选时 `af_` 字典序在 `zf_` 之前，
  实测 `narrator=af_maple`。新增 `VoiceLocale` 并改为「性别+语种」二维选音
- **台词段全丢音色**：真实 Kokoro 池 114 个音色中**无中性、无童声**，
  原「取首个 Neutral」兜底返回 None，实测 **29006 段无音色**。改为逐级回退
  `中性 → 男声 → 任意`，宁可中文男声也不留空

### 能力边界（已写入文档）
- **性别只能靠称谓确认**。叠字名（`芃芃`）等无称谓线索者 `gender` 恒为 `Neutral`；
  `suggested_gender` 是频次弱先验推测，**不参与自动分配**——猜错性别会让全书
  配音越听越违和，交由用户试听确认
- **真实音色池无童声**（`xf_` 数量为 0，`xf_child` 仅存在于测试构造）。
  童声角色须由用户在 GUI 指定克隆音色，或走 IndexTTS-2.5 零样本克隆

### 测试
- `tts::role` 31 passed（新增 20）；`tts::` 99 passed
- `e2e_novel_parse.rs` 新增 4 个真实全本用例（8 passed），含性能（<2s）、
  Top3 人名信噪比、中文音色断言、role_map 反序列化往返
- 回归：domain 191 / app 117 全绿（无既有断言被破坏）

## [2026-10-06] Paraformer 换源 sherpa 官方包 + ASR 跨引擎交叉验证

### Paraformer（换源 + 正向识别打通）
- **registry 换源**：`models/registry/paraformer.yaml` 从 FunASR 原始导出
  （model_quant.onnx + tokens.json，缺 sherpa 所需 `vocab_size` 元数据、
  加载会崩溃进程）换为 **sherpa-onnx 官方转换包**
  `csukuangfj/sherpa-onnx-paraformer-zh-2024-03-09`（model.int8.onnx +
  tokens.txt，zh+en+yue，单文件直连 + sha256 校验，源：hf-mirror/huggingface）。
  本地已下载落位（217MB，sha256 已录入清单）。FunASR 旧资产
  （model_quant.onnx/tokens.json/config.yaml）留在磁盘但移出清单，
  待用户确认后处置（AGENTS 禁止自动删模型）
- **正向识别测试通过**：0.wav 识别「昨天是 monday today is 礼拜二 the day
  after tomorrow 是星期三」——语义正确，加载+识别全套 3.3 秒；
  FunASR 拒绝护栏测试改为条件适用（目录已有 sherpa 模型时跳过）

### ASR 跨引擎 0.wav 交叉验证（新测试 asr_cross_engine_0wav_test）
- **关键结论：0.wav 是中英混说音频**（"昨天是 Monday, today is 礼拜二,
  the day after tomorrow 是星期三"），并非 firered 测试注释假定的纯中文
  "昨天是星期一……"——5 个引擎独立输出英文星期词可证。此前 FireRed
  2 例"失败"实为断言假定纯中文所致，模型本身识别正确
- **六引擎横评**：Qwen3-ASR ✅（混说结构最完整）、SenseVoice ✅、
  Paraformer（sherpa 包）✅、FireRed CTC/AED ✅、WeNet ⚠️（首尾正确，
  中段同音字噪声）、Whisper-base ❌（经典重复幻觉「天天×52」，诊断用例）
- **发现并修复 wenet 集成测试假绿**：其 test_wavs 指向不存在的
  `models/asr/wenet/test_wavs`，全部测试一直在静默跳过（0 秒假通过）。
  现指向共享测试集真跑：3/3 通过（8k.wav 长句识别质量良好），
  0.wav 断言统一为星期语义口径

### 验证
- `cargo test --workspace`：**855 通过 / 0 失败 / 103 忽略**
- registry 守护/一致性测试 10/10 通过
- slow-models 实测：paraformer 3/3、wenet 3/3（首次真跑）、
  firered 7/7、跨引擎 5/5

## [2026-10-06] OCR：PP-OCRv6 Medium 字典修复 + cls 输入自适应 + 三模型对比评测

### 修复
- **v6-medium 字典缺失导致无法加载**：v6 medium/small 的 rec ONNX 未内嵌
  字符表元数据（输出 18710 类 = 18708 字 + blank + space），引擎回退找
  `ppocr_keys_v1.txt` 时目录内缺失。现落地官方 `ppocrv6_dict.txt`
  （sha256 b5f2bfe2…，gitee/github 双源字节一致），并声明进
  `paddleocr-v6-medium.yaml` / `paddleocr-v6-small.yaml` 下载清单——
  此前用 v4 的 6623 字字典会输出乱码（实测验证过错误字典的失败形态）
- **cls 方向分类输入尺寸自适应**：`classify_direction` 原写死 192×48，
  v5-server 的 textline cls（80×160）直接报错。现从模型声明的静态输入
  形状读取（动态维度回退 48×192），mobile/server 两类 cls 模型均可加载
- OCR 四变体测试（v4/v5-mobile/v6-tiny/v6-medium）恢复并通过
  （slow-models 门控）；v6-medium 对标准图实现逐字精确识别（8/8）

### 评测（`ocr_rapidocr_eval_test.rs`，slow-models 门控）
- 统一管线下三方对比（9 张合成图：3 字体/多字号/噪声/JPEG，LCS 字符准确率）：
  | 模型 | 体积 | 准确率 | 置信度 | 耗时 |
  |---|---|---|---|---|
  | PP-OCRv6-medium | 168.3MB | 0.902 | 0.982 | ~2.1s |
  | RapidOCR-v5-mobile | 21.0MB | 0.940 | 0.949 | ~1.3s |
  | PP-OCRv5-server | 165.3MB | **0.972** | 0.978 | ~4.7s |
- 结论：v5-server 准确率最高；RapidOCR-v5-mobile 效率之王（1/8 体积且
  准确率反超 v6-medium）；v6-medium「你→尔」跨字体稳定误识别，置信度
  虚高。经实验排除 BGR/RGB 通道序因素，属 rec 模型在特定输入尺度下的
  固有字形混淆（det 裁剪尺度为主要变量）。经确认**默认引擎维持
  PP-OCRv6-medium**（AGENTS.md 口径不变），报告见 `tmp/ocr_eval_report.md`

## [2026-10-06] 质量审查修复：正确性缺陷批次一（P1/P2 共 14 项）

> 依据深度审查报告（4 个专项审查代理逐文件阅读 + 人工核实）实施。
> 验证基线：`cargo test --workspace` **855 通过 / 0 失败 / 97 忽略**；
> clippy 警告 233 → 231（修复本身零新增）。剩余架构治理项（app 层 53 处
> infra 直引、DTO 死代码、CLI/GUI 对等性、clippy 清零）为下一批次。

### ASR（正确性）
- **非 WAV 输入静默返回 5 秒静音 → 明确报错/转码**：`AsrUseCase::load_audio`
  此前对 mp3/m4a/flac 等格式 warn 后返回静音并照常识别，产出与输入无关的垃圾
  字幕（批量 ASR 默认扩展名即含 mp3，必然踩中）。现：WAV 直读；压缩格式经
  ffmpeg 解码为 WAV 后识别（infra 新增 `FfmpegEncoder::decode_to_wav`，
  `-vn -map a:0?` 忽略容器视频轨，临时文件用后即删）；未知格式返回
  `AsrError::UnsupportedFormat`（VA004，此前从未被使用）
- **空识别切片不推进时间轴**：静音切片不累计 `offset_ms`，其后字幕整体提前
  一个切片时长——推进改为无条件执行
- **Paraformer 伪特征修复 + 进程崩溃护栏**：原"特征提取"只把帧能量乘线性
  系数填满 80 维（非 mel，识别输出为无效数据但注释标注"已完成"）。改走
  sherpa-onnx 绑定（真实 fbank+LFR+CMVN，与 firered/qwen3 同范式）；实测发现
  本地 FunASR 导出（model_quant.onnx）缺 sherpa 要求的 `vocab_size` 元数据，
  sherpa C++ 端会 **abort 整个进程**——加载前明确拒绝并指引改用
  sherpa-onnx 官方转换包（registry 源待换，见遗留）
- **Qwen3-ASR 复查**：走 sherpa 分图包（conv 前端在模型内），无伪特征问题，
  审查误报排除

### TTS（正确性）
- **Qwen3-TTS 同变体重复重载**：`ensure_loaded` 此前只要 `model_override`
  非空就无条件 unload+load（GUI 每次都传），每次合成多等 10~60 秒。现对比
  `loaded_is_small()` 与目标变体，同变体直接复用
- **取消被当成成功**：闸门拒绝/取消时 `break` 跳出循环走成功路径，
  `finish_all` 把半成品音频当正式产物落盘。改为返回错误（临时文件由
  abort 清理，断点续转段已落盘可恢复）
- **数字转中文补零**：`12000034` 读成「一千二百万三十四」——组间补零实现
  （前导零组/跳过的全零组补一个「零」），并补上文档声称却从未实现的
  「4 位数字+年 → 逐位读」（1998年 → 一九九八年）；新增组间零/年份回归测试

### 字幕（正确性，三连缺陷）
- **词级时间戳错位**：`from_word_timestamps` 用「句子数」当「词索引」取下一句
  起始时间，偏差随句子数累积——改为独立词游标
- **时间轴塌缩**：单句 500ms 下限在「句数×500ms > 总时长」时把几十条字幕
  堆叠在结尾同一时刻——改为纯比例分配（总和恒 ≤ 总时长）
- **LRC 丢小时位**：1h01m23s 写成 `[01:23.45]`——分钟位折入小时（`[61:23.45]`）
- 三处均补时间断言单测（旧测试只断言文本，缺陷因此长期潜伏）

### GUI（欺骗性 UI 清理）
- **翻译/流水线/视频页假取消修复**：三页「取消/停止」只置 `is_running=false`，
  后台任务继续跑且完成事件把「已取消」覆盖回「完成」。现在：三页 state 增加
  `cancel_token`（复用 TTS 页正确模式）；`spawn_translate/spawn_pipeline/
  spawn_video` 传递令牌；用例链路打通取消——翻译管线逐段检查（新增
  `TranslationError::Cancelled` 路径）、流水线阶段边界+逐段检查
  （`PipelineContext` 携带令牌）、视频生成各阶段边界检查（TTS/ASR 透传）
- **视频生成**：临时文件从写死 cwd（并发互踩、失败残留）改为系统临时目录+
  唯一名、成败都清理；第三份引擎名解析（静默把 whisper/sensevoice 当 TTS
  引擎、未知值回退 Kokoro）删除，改走 domain `parse_tts_engine` 并报错

### 任务/模型/持久化（状态与信任）
- **task_manager 状态机竞态**：执行期间取消的任务会被执行线程无条件覆盖回
  Completed/Failed——执行前跳过已取消任务、收尾时重读库状态保持 Cancelled；
  `list_asr_tasks` 恒返回空列表 → 仓储 trait 新增 `find_all()`（SQLite 两实现
  + 内存实现，按创建时间倒序）；`create_asr_task` 丢弃用户输出路径 →
  `AsrTask` 新增 `output_path` 字段（JSON blob 存储，零迁移）并生效
- **模型「释放」假释放**：`unload_model` 只改状态位不释放会话。现 TTS/ASR/OCR
  用例新增按引擎真实卸载（`TtsUseCase::unload` / `AsrUseCase::unload` /
  `OcrUseCase::unload_engine`），`ModelUseCase::unload_model` 路由到进程级
  单例真实释放内存
- **GUI 配置路径传播**：设置页保存写死相对路径 `application.yml`（双击启动时
  cwd 任意会写错位置）——`GuiHandles`/`AppState` 传播 bootstrap 的
  `config_path`，设置页按其读写
- **OCR 分页更新加事务**：主表 UPDATE 与页面 INSERT 各自自动提交，中途失败
  留下不一致（与 save() 的事务承诺矛盾）——`unchecked_transaction` 包裹；
  页面结果序列化失败由静默写 NULL 改为报错留痕

### 安全/配置
- **配置加密闭环补全**：`decrypt_value` 全仓库零调用、读取路径拿 `ENC(...)`
  密文当明文 key 用——`ConfigCrypto::decrypt_config_content` 新增（与加密
  同字段清单对称），接入 `ConfigLoader::from_file/load_optional`；解密失败
  （密钥丢失/文件跨机复制）保留密文原值并告警；加密↔解密往返 + 失败保留
  原值均有单测

### 引擎身份（路由正确性）
- **云端 Provider 谎报 engine_kind**：Azure/阿里云 ASR 返回 `SenseVoice`、
  TTS 返回 `Kokoro`（4 处）——按 kind 路由/列音色会混淆云端与本地引擎。
  改返回 `AzureAsr/AliyunAsr/AzureTts/AliyunTts`（变体早已存在且
  parse_tts_engine 已接线），音色归属随字段自动修正

### 测试基建
- **集成测试 slow-models 门控**（docs/23 §9.6 遗留待办）：19 个会加载真实
  模型的 tests/ 集成测试补 `cfg_attr(not(feature = "slow-models"), ignore)`，
  此前 `cargo test` 全目标默认全跑（FireRedASR 一个就 9 分钟且 2 例失败）
- **环境依赖测试缺失即跳过**：indextts25 金标对拍（tmp/indextts25_refs/）、
  g2p/phoneme/tokenizer（tmp/novel.txt、本地模型文件）在依赖物缺失时打印
  提示并跳过，不再假失败
- 新增：Paraformer 拒绝护栏测试、配置加密闭环测试、数字补零/年份测试、
  字幕时间断言测试

### 遗留（下一批次）
- Paraformer registry 源需换 sherpa-onnx 官方转换包
  （sherpa-onnx-paraformer-zh-*），换源后补正向识别测试
- FireRedASR 集成测试 2 例失败（0.wav 识别不出内容）——模型质量问题，
  已被门控隔离，待排查
- 架构治理：app 层 53 处 `votex_infra::` 直引收敛、DTO 死代码处置、
  引擎名解析归一、CLI video 改走用例、GUI task_runner 参数透传补齐、
  clippy 231 条清零

## [2026-10-06] 移除 IndexTTS2，IndexTTS-2.5 全面替代

### ASR / Qwen3-ASR 链路补全（sherpa-onnx）
- **模型资产补全**：落位 sherpa-onnx 官方预转换包 `sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25`
  （conv_frontend 43MB + encoder.int8 174MB + decoder.int8 721MB + tokenizer/，共 ~943MB）至
  `models/asr/qwen3-asr/`；此前目录内仅有 safetensors 原始权重，无任何 ONNX，链路不可用
- **Provider 重写**：`qwen3_asr.rs` 从无效的 ort 单模型 stub 重写为 sherpa-onnx 绑定
  （`OfflineQwen3ASRModelConfig` 三图 + tokenizer 目录，与 firered_asr 同范式）；
  AsrProvider trait 与 `EngineKind::Qwen3Asr` 不变，app/GUI 层零改动
- **下载链路**：`archive.rs` 新增 `tarbz2` 压缩包类型（bzip2 + tar，成员提取含路径穿越防护），
  registry `qwen3-asr.yaml` 重写为 sherpa 包逐成员声明（sha256 + size + archive），支持一键重下
- **e2e 验证通过**：IndexTTS-2.5 合成粤语短句 → Qwen3-ASR 识别「今日天气极好，我一起去饮茶了。」，
  与 SenseVoice 结果一致；能力边界：29 语言 + 22 中国方言（含粤语广东/香港口音、吴语、闽南语）

### 修复
- `model_registry_consistency_test` 两处既有失败：白名单缺 `IndexTTS25/OnnxRuntime`；
  目录推导未处理 `target_dir` 字段（onnxruntime 误判到 models/other/）

### TTS / 引擎清理（破坏性变更）
- **移除 IndexTTS2（2.0）引擎**：模型无语言/方言 token（无 `<|yue|>`、无 `lang_id` 输入），
  不支持粤语，且此前 `supported_dialects` 声明的粤语/闽南语 Native 支持为错误元数据
  （对应音色 yue_male/nan_male 不存在）。删除 `indextts2.rs`（~935 行）、registry 清单
  `indextts2.yaml` 及慢速集成测试；`models/tts/indextts2/`（4.5G）待用户确认后删除
- **迁移别名**：历史串 `indextts2` / `indextts` / `IndexTTS2`（配置、CLI、持久化任务）统一
  归一到 `IndexTTS25`，旧数据无缝续用
- **同步面**：EngineKind 变体、TtsUseCase/ModelUseCase/VideoGenerate、EngineLoader 候选表、
  GUI 四页面引擎下拉 + 引导页下载勾选、voice_library 音色库根目录（改由 WorkspacePaths 直推，
  不再借道 indextts2 路径）、CLI 帮助文本、docs/01/02/03/05/09、AGENTS.md 技术栈
- **粤语能力收敛**：离线 TTS 粤语仅 IndexTTS-2.5 原生支持（`<|yue|>` + lang_id，金标已验证）；
  GUI 语言下拉随引擎联动（昨日已改），闽南语选项移除

## [2026-10-05] IndexTTS-2.5 Rust 移植完成（粤语原生）

### TTS / IndexTTS-2.5（新增引擎）
- **Rust 端到端移植**（蓝本 docs/25，8 任务 #41~#48 全部完成）：基于 yunfengwang
  fp32 分图 ONNX（11.7GB，8 session：gpt_prefill/gpt_step/cfm_estimator/semantic_model/bigvgan…），
  无 Python 依赖（uvx 参考实现仅作金标对拍）
- **分级金标验收全过**：前端 token 逐 id 一致（含 100 语言修正修复 `<|yue|>` 未注册的上游
  off-by-one）；GPT greedy 真实模型逐 token 一致（硬契约）；CFM ≈1e-5；BigVGAN **逐位 0 误差**；
  端到端粤语合成 3.55s wav 落盘（`tmp/indextts25_e2e_yue.wav`）
- **wetext 归一化真实移植（#49）**：`normalizer.rs` 三层结构（rustfst FST 引擎 /
  wetext nbest=1 快路径 / 参考实现包装层），FST 资产逐字节取自 wetext 0.1.8 wheel
  内嵌（13.6MB，zh+en TN）；**金标 50 例逐例精确相等**（含 en 无数字文本差异面，
  修正 wetext-rs 上游的门控偏差）；`use_normalization=true` 时 zh/zhen/en 生效
- **引擎接入**：`EngineKind::IndexTTS25`（CLI/GUI 串 `indextts25`）——TtsUseCase / 模型管理 /
  EngineLoader / GUI 四页面 / task_runner 全链路；音色 = `prompts/<id>.wav` 零样本克隆；
  方言：普通话 + 粤语 Native（闽南语/吴语 token 越界，不支持）
- **长文本粤语 e2e（#51）**：tmp/e2e_ch1.txt（三章小说）多段合成，`VOTEX_E2E_MAX_SEGMENTS`
  控制段数上限，落盘 `tmp/indextts25_e2e_yue_ch1.wav`
- 已知限制：首次加载 ~33s（fp32 11.7GB），speaker 缓存后复用

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
