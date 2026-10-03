# 17 - 翻译模块分析与 LunaTranslator 对比

> 编写日期：2026-10-01
> 分析对象：`votex`（声阅）翻译模块
> 对比对象：[HIllya51/LunaTranslator](https://github.com/HIllya51/LunaTranslator)（视觉小说翻译器）

---

## 第一部分：votex 翻译实现分析

### 1.1 分层架构

votex 采用 DDD 四层架构，翻译模块在各层的落地情况如下：

| 层 | Crate | 翻译相关内容 | 完整度 |
|---|---|---|---|
| 领域层 | `votex-domain` | `translation/provider.rs`（trait）、`translation/value_object.rs`（方向枚举）、`error.rs`（VTR001~004） | ⚠️ 半成品 |
| 应用层 | `votex-app` | `use_case/translation_use_case.rs`、`dto/translation_dto.rs`、`services/keyword_translator.rs` | ⚠️ 单薄 |
| 基础设施层 | `votex-infra` | `translation/` 下 8 个引擎实现 | ✅ 较完整 |
| 表现层 | `votex-cli` / `votex-gui` | `commands/translate.rs`、`pages/translation_page.rs` | ✅ 可用 |

**关键结构缺陷**：对比 TTS / ASR / OCR 三兄弟都有完整的「实体 + 领域服务 + 仓储接口」三件套，**翻译领域层只有「Provider trait + 方向值对象」两层**：

```
crates/votex-domain/src/translation/
├── mod.rs
├── provider.rs        # TranslationProvider trait（仅 2 个方法）
└── value_object.rs    # TranslationDirection
                       # ← 缺 entity.rs（翻译实体）
                       # ← 缺 service.rs（领域服务）
                       # ← 缺 repository.rs（仓储接口）
```

### 1.2 抽象层设计

**Provider trait**（`crates/votex-domain/src/translation/provider.rs`，全文 11 行）：

```rust
pub trait TranslationProvider: Send + Sync {
    /// 提供者名称
    fn name(&self) -> &str;

    /// 翻译文本
    fn translate(&self, text: &str, direction: TranslationDirection)
        -> Result<String, TranslationError>;
}
```

只有 `name` + `translate` 两个方法，**缺少**以下扩展点（这是后续所有能力缺失的根源）：

- ❌ `load()` / `unload()` — 生命周期管理，模型常驻必须
- ❌ `is_loaded()` / `supports()` — 能力探测，GUI 需要据此灰显选项
- ❌ `translate_batch()` — 批量翻译
- ❌ `translate_stream()` — 流式输出
- ❌ `with_glossary()` — 术语表/词典注入
- ❌ `supported_pairs()` — 支持的语言对列表

**方向值对象**：

```rust
pub enum TranslationDirection {
    ZhToEn,
    EnToZh,
    Auto,
    ByLanguagePair { source: String, target: String },
}
```

**错误码**（`error.rs:65-90`）：`ApiError`(VTR001) / `ModelNotLoaded`(VTR002) / `UnsupportedDirection`(VTR003) / `EmptyText`(VTR004)。

### 1.3 引擎实现矩阵

`crates/votex-infra/src/translation/` 共 9 个文件，8 个引擎：

| 文件 | 引擎 | 类型 | 架构 | KV Cache | 状态 |
|---|---|---|---|---|---|
| `opus_mt.rs` (19.2K) | Opus-MT (MarianMT) | 离线 ONNX | Encoder-Decoder | ✅ **真增量** | ✅ 可用 |
| `nllb.rs` (16.4K) | NLLB-200-600M | 离线 ONNX | Encoder-Decoder | ❌ 每步全量 | ✅ 可用 |
| `m2m100.rs` (17.8K) | M2M-100-418M | 离线 ONNX | Encoder-Decoder | ⚠️ 已加载未使用 | ✅ 可用 |
| `hy_mt.rs` (13.4K) | HY-MT1.5-1.8B | 离线 ONNX | Decoder-only | ❌ 每步全量 | ✅ 可用 |
| `qwen_mt.rs` (6.0K) | Qwen-MT | 在线 API | DashScope | — | ✅ 可用 |
| `llm_translate.rs` (4.1K) | DeepSeek | 在线 API | — | — | ✅ 可用 |
| `dict_translate.rs` (5.1K) | 内置词典 | 离线硬编码 | HashMap ~80 条 | — | ⚠️ 单向 zh→en |
| `ctranslate2.rs` (5.8K) | CTranslate2 | FFI 骨架 | libloading | — | ❌ **未实现** |

**性能瓶颈**：HY-MT / NLLB / M2M-100 均为「每步全量重算」的自回归解码，复杂度 O(n²)。其中 M2M-100 已经加载了 `decoder_with_past_state` 却在 `translate_internal` 中完全没用（代码里 `step == 0` 之后仍走 `decoder_state`）——这是明显的遗漏。Opus-MT 是唯一真正跑通 `decoder_with_past` 增量解码的引擎（`decode_first_step` 出 24 个 present，`decode_next_step` 吃 26 输入）。

**CTranslate2 是空壳**：`translate_ct2()` 恒返回 `ApiError("未实现")`，但 GUI 引擎下拉框仍把它列为可选项，用户选中必然报错。

### 1.4 数据流

```
GUI 点击「翻译」(translation_page.rs)
  → task_runner::spawn_translate(tx, text, engine, direction)      [std::thread::spawn]
      → TranslationUseCase::new(engine)
          → infr::translation::create_translation_provider(engine)   [硬编码路径探测]
              → provider.load_from_dir(&dir)
                  → OrtSessionFactory::create(&onnx_path)
      → use_case.translate(text, direction)
          → 方向字符串解析 → TranslationProvider::translate()
              → 空文本校验 → 方向校验 → 分词 → 编码 → 自回归解码 → 过滤特殊 token → 解码 → trim
      → TaskEvent::Success { data: Some(translated) }
  → app.rs 写入 state.translation.target_text
```

各引擎管线对比：

| 阶段 | Opus-MT | NLLB | M2M-100 | HY-MT1.5 | Qwen-MT/LLM | Dict |
|---|---|---|---|---|---|---|
| 语言检测 | 无 | 无 | 无 | 无 | 无 | 无 |
| 分词 | source.spm + vocab.json 重映射 | SP-BPE（id_offset+1） | SP-BPE | tokenizers::Tokenizer | 无需 | 查表 |
| 前缀注入 | 无 | 源语言 token + EOS | `[lang]+ids+[EOS]` | ChatML prompt | system prompt | — |
| 解码上限 | **64 步** | 200 | 512 | 512 | — | — |
| 采样 | greedy argmax | greedy + 首步强制 lang | greedy + step0 强制 | argmax | API | — |

### 1.5 模型管理

**磁盘现状**（`models/translation/`）：

```
hy-mt-1.5/                 model_q4.onnx(449KB) + model_q4.onnx_data(1.41GB) + tokenizer.json
m2m-100/                   encoder_model.int8.onnx(284MB) + decoder_model.int8.onnx(860MB) + ...
nllb-200-distilled-600m/   encoder_model_int8.onnx(415MB) + decoder_model_int8.onnx(1.52GB)
opus-mt-zh-en/             encoder/decoder/decoder_with_past + source.spm + target.spm + vocab.json
opus-mt-en-zh/             ↑ 同上
```

**已验证的路径不一致问题**（3 处真实 Bug）：

| # | 注册表 id | 下载落盘目录 | 代码查找目录 | 结论 |
|---|---|---|---|---|
| 1 | `hy-mt-1.5-1.8b` | `translation/hy-mt-1.5-1.8b` | `translation/hy-mt-1.5` | ❌ 下载后**永远加载不到** |
| 2 | `m2m-100-418m` | `translation/m2m-100-418m` | `translation/m2m-100-418m` | ⚠️ 一致，但磁盘存量是 `m2m-100` → **现有模型加载不到** |
| 3 | （无 `opus-mt-en-zh.yaml`） | — | `translation/opus-mt-en-zh` | ❌ 无法下载，只能手工放置 |

代码出处（`crates/votex-infra/src/translation/mod.rs`）：

```rust
pub fn create_translation_provider(name: &str) -> Result<Box<dyn TranslationProvider>, String> {
    let models_dir = Path::new("models").join("translation");   // ← 硬编码相对路径
    match name {
        "m2m-100" | "m2m100" => { let dir = models_dir.join("m2m-100-418m"); ... }
        "hy-mt-1.5" | ...    => { let dir = models_dir.join("hy-mt-1.5");   ... }
        ...
    }
}
```

两个附加问题：
- **硬编码相对路径**：`Path::new("models")` 忽略 `AppState.models_dir`，工作目录一变就失效（TTS/ASR/OCR 都支持传入）。
- **无会话缓存**：每次 `create_translation_provider` 都重建 ONNX Session，GUI 每点一次翻译就重新加载 1.4GB 的 HY-MT（Level3 图优化 + 全核 intra_threads），体验上无法接受。

**ONNX 配置未生效**（已验证）：翻译 8 处调用全部走 `OrtSessionFactory::create()`，而该方法不读取全局配置：

```rust
pub fn create(model_path: &Path) -> Result<Session> {
    let n_threads = std::thread::available_parallelism()...;   // 强制全核
    Self::create_with_threads(model_path, n_threads)            // 无 EP、无 inference 配置
}
// 只有 create_raw* 系列才应用 GLOBAL_EP 与 GLOBAL_CONFIG
```

结果：`application.yml` 的 `inference.num_threads: 3`、`execution_provider`、`memory_limit_mb: 2048`、`kv_cache`、`enable_memory_pattern` 对翻译推理**完全无效**。

### 1.6 能力矩阵

| 能力 | 状态 | 说明 |
|---|---|---|
| 术语表 / Glossary（用户可配） | ❌ **不存在** | 全仓 grep 无翻译相关术语表 |
| 内置离线词典 | ⚠️ 硬编码不可配 | ~80 条，仅 zh→en |
| 批量翻译 | ❌ **不存在** | 有 `batch_tts_use_case` / `batch_asr_use_case`，无翻译版 |
| 长文本分段 | ❌ **不存在** | TTS 有分段器，翻译整段送入，被 `max_length` 硬截断 |
| 翻译缓存 | ❌ | 无 |
| 上下文记忆 | ❌ | 每次翻译独立 |
| 流式输出 | ❌ | 同步阻塞返回 |
| 语言检测 | ❌ | `Auto` 只是别名到 zh→en |
| 翻译历史/持久化 | ❌ | SQLite 只有 `tts_tasks`/`asr_tasks`/`ocr_tasks`/`models`/`downloads`，**无 `translation_tasks`** |
| 简繁转换 | ❌ | 无 |
| 后处理管线 | ❌ | 无标点规范化、无术语回填、无结果修复 |

AGENTS.md 把「长文本或语音翻译」「长文本分段算法」列为功能目标，但翻译侧尚未落地。

### 1.7 测试情况

| 类型 | 文件 | 用例数 | 备注 |
|---|---|---|---|
| 集成测试 | `tests/opus_mt_translation_test.rs` | 8 | ✅ 有 golden 断言（`"你好世界" → "- Good world."`） |
| 集成测试 | `tests/hy_mt_translation_test.rs` | 7 | 模型缺失软跳过 |
| 集成测试 | `tests/m2m100_translation_test.rs` | 7 | 模型缺失软跳过 |
| 集成测试 | `tests/nllb_translation_test.rs` | 6 | 从 registry yaml 读 TokenizerConfig |
| 集成测试 | `tests/qwen_mt_translation_test.rs` | 5 | 无 Key 时静默跳过 |
| 单测 | `hy_mt.rs` / `nllb.rs` / `m2m100.rs` / `opus_mt.rs` | 8 | prompt 构造、argmax、语言映射 |

覆盖缺口：`TranslationUseCase`、CLI `translate.rs`、GUI 翻译页、工厂函数 `create_translation_provider`、`TranslationDirection::from_str` **全部无测试**。NLLB / Opus-MT 测试缺模型守卫，缺失会直接 panic fail。

### 1.8 配置

**`application.yml` 零翻译配置项。** `AppConfig` 结构体字段为 `models/tts/asr/output/log/ui/task/inference/pipeline`，无 `translation` 段。TTS 有 `resolve_tts_engine()`、ASR 有 `resolve_asr_engine()`，**翻译没有 `resolve_translation_engine()`**，默认引擎只能靠 CLI `--engine` 默认值 `"dict"` 与 GUI `TranslationState::default()` 硬编码。

另外：通用 `ProviderRegistry`（`domain/provider.rs` + `infra/provider.rs`）已实现，但**只注册了 TTS/ASR resolver，没有任何翻译 Provider 被注册进去**。这是架构上的「应然」与「实然」脱节。

### 1.9 已验证问题清单（按严重度）

| 级别 | 问题 | 位置 |
|---|---|---|
| 🔴 P0 | HY-MT 注册表 id 与查找目录不一致，下载后加载不到 | `models/registry/hy-mt-1.5.yaml` vs `translation/mod.rs` |
| 🔴 P0 | 每次翻译重建 ONNX Session，无模型池，GUI 重复加载 GB 级模型 | `create_translation_provider` |
| 🟠 P1 | `OrtSessionFactory::create()` 绕过全局推理配置，`inference.*` 对翻译无效 | 8 处调用点 |
| 🟠 P1 | 无术语表机制（对比 LunaTranslator 的核心能力） | 架构缺失 |
| 🟠 P1 | HY-MT/NLLB/M2M-100 无 KV cache，长句 O(n²)；M2M 已加载却未使用 | `m2m100.rs` 等 |
| 🟠 P1 | M2M-100 磁盘目录 `m2m-100` 与代码 `m2m-100-418m` 不符 | 磁盘 vs 代码 |
| 🟡 P2 | CTranslate2 未实现但 GUI 可选 | `ctranslate2.rs` |
| 🟡 P2 | `opus-mt-en-zh` 无注册表，无法下载 | `models/registry/` |
| 🟡 P2 | `TranslationUseCase` 用 `"->"` 解析方向，`TranslationDirection::from_str` 用 `"-"`，不一致 | app vs domain |
| 🟡 P2 | `TranslationDto` 定义后从未使用 | `dto/translation_dto.rs` |
| 🟡 P2 | 硬编码 `Path::new("models")`，忽略 `models_dir` | `translation/mod.rs` |
| 🟡 P2 | 无长文本分段、无批量、无持久化、无配置段 | 全局 |

---

## 第二部分：LunaTranslator 项目分析

### 2.1 项目定位

| 项 | 值 |
|---|---|
| 定位 | **视觉小说 / Galgame 实时翻译器**（不是通用翻译工具） |
| Star / Fork | 13,490 / 1,166 |
| 主语言 | C++ 4.29MB + Python 2.51MB（另有 HTML/C/C#/CMake/JS） |
| 许可 | **GPLv3** |
| 创建 / 最近推送 | 2022-09-28 / 2026-09-30（4,998 commits，高度活跃） |
| 平台 | **仅 Windows**（依赖 HOOK 注入、窗口 API） |
| GUI | PyQt5（+ qtawesome / QSS 主题） |
| 分发 | PyInstaller 打包 + Python 运行时，体积数百 MB |

**核心使用场景**：把日文 Galgame 的文本实时抓出来、翻译成中文、再显示/内嵌回游戏。

### 2.2 目录结构

```
src/
├── LunaTranslator/                 # Python 主体
│   ├── LunaTranslator.py (68KB)    # 主控制器（巨型）
│   ├── NativeUtils.py (39KB)       # Windows 原生调用封装
│   ├── translator/       ← 44 个翻译引擎
│   ├── transoptimi/      ← 翻译优化管线 ⭐ 核心差异
│   ├── cishu/            ← 词典（mdict/mojidict/jisho/weblio/jpdb/youdao）
│   ├── ocrengines/       ← 19 个 OCR 引擎
│   ├── tts/              ← 15 个 TTS 引擎
│   ├── textio/           ← 文本输入源(HOOK/OCR/剪贴板) 与输出(内嵌/悬浮/Anki/Yomitan)
│   ├── gui/  myutils/  network/  defaultconfig/
├── NativeImpl/                     # C++ 钩子注入实现（C++ 占比最大的部分）
├── files/  scripts/  docs/
```

### 2.3 翻译抽象层：`basetrans`

`src/LunaTranslator/translator/basetranslator.py`（13.4KB）是所有 44 个翻译器的基类，这是**最值得 votex 借鉴的设计**：

```python
class basetrans(commonbase):
    # —— 子类必须实现 ——
    def langmap(self): ...          # 标准语言码 → API 语言码映射，未声明 cht 则"简体+转繁"
    def init(self): ...             # 初始化（建 session / 读 key）
    def translate(self, content: "str|GptTextWithDict"): ...   # 可为 generator 实现流式

    # —— 框架提供的能力 ——
    def result_cache_key(self, src, tgt, sentence): ...
    @property transtype      # free / dev / api / offline / pre —— 决定调度策略
    @property using_gpt_dict # 决定传入 GptTextWithDict 还是 str
    @property use_trans_cache
    @property is_gpt_like
    @property onlymanual     # 仅手动翻译时使用
```

框架内置能力：

1. **优先级队列 + 中断机制**
   ```python
   self.queue = PriorityQueue()          # 有 callback 的请求 priority=1，否则 0
   # 若队列来了新请求，当前请求通过 Interrupted 异常被放弃
   checktutukufunction = lambda: ((waitforresultcallback is not None) or self.queue.empty()) and self.using
   ```
   `_fythread` 常驻工作线程消费队列；实时 HOOK 场景下新文本不断涌入，旧请求必须能被丢弃——这是实时翻译的刚需。

2. **请求间隔 + 多 Key 轮换**：`intervaledtranslate()` 按 `globalconfig["requestinterval"]` 限速，`multiapikeywrapper` 轮换 API Key。

3. **翻译缓存**：`md5(src, tgt, sentence)` 为 key，`use_trans_cache` 开关；`pre`/`other` 类型不缓存；LLM 手动翻译不缓存。

4. **流式输出**：`translate()` 可返回 generator，框架逐块 `callback(collectiterres, 1)`，结束时 `callback(res, 2)`；`\0` 表示重置累加器。

5. **简繁转换**：`maybezhconvwrapper` / `checklangzhconv` + `zhconv.py`，API 不支持繁体时自动「译简体 → 转繁体」。

6. **错误自愈**：异常时 `self.needreinit = True`，下次 `maybeneedreinit()` 重建 session。

7. **离线优先不中断**：`if self.transtype == "offline": func()` —— 离线推理不套 timeoutfunction，因为「中断了服务端还在跑」。

### 2.4 翻译优化管线 `transoptimi/`（最大差异点）

这是 LunaTranslator 翻译质量的护城河，votex 完全缺失：

```
原文 → process_before(每个优化器) → 翻译引擎 → process_after(每个优化器) → 译文
        ↑ 返回 (处理后文本, context)            ↑ 拿回 context 做逆变换
```

**① `noundict.py` — 专有名词翻译（术语表）**

机制非常巧妙：翻译前把术语替换成**占位符**，同时把术语表注入 LLM prompt，翻译后再把占位符换回译文。

```python
def process_before(self, japanese):
    for gpt in self.usewhich():                    # 全局 / 游戏私有 / 合并 三作用域
        src = re.escape(gpt["src"])
        if gpt.get("whole-word"): src = r"\b" + src + r"\b"
        flags = 0 if gpt.get("case-sensitive") else re.IGNORECASE
        if not re.search(src, japanese, flags): continue
        gpt_dict.append(gpt); used.append((src, gpt["dst"]))
    japanese1, mp1 = self.process_before1(japanese, used)   # → "ZX{}Z" 占位符
    return japanese1, {"gpt_dict": gpt_dict, "gpt_dict_origin": japanese, "zhanweifu": mp1}

def process_after(self, res, context):
    for key in context["zhanweifu"]:
        res = case_insensitive_replace(res, key, mp1[key])   # 占位符 → 指定译文
    return res
```

每条术语含 `src / dst / info`（原文/译文/注释），支持整词匹配、大小写敏感，且**可按游戏（gameuid）单独配置**。

**② `vndbnamemap.py` — 翻译前替换**：从 VNDB 导入人名表，正则/普通替换，同样支持全局 + 游戏私有。

**③ `myprocess.py` — 用户自定义前后处理脚本**：通过 `checkmd5reloadmodule` 实现**热重载**，用户改脚本立即生效，无需重启。

**④ 其他**：`transerrorfix.py`（翻译结果修复，如修常见 LLM 输出毛病）、`skiponlypunctuations.py`（跳过纯标点，省 API 调用）、`arabic_reshaper.py`（阿拉伯文连写整形，94KB）。

### 2.5 GPT 词典与上下文记忆

```python
class GptTextWithDict:
    rawtext      # 原文（未替换）
    parsedtext   # 已做术语占位替换的文本
    dictionary   # GptDict: [{src, dst, info}, ...]
```

对 LLM 类引擎，`basetrans` 会把术语表拼进 prompt：

```
{DictWithPrompt[When translating, please ensure to translate the specified nouns
 into the translations I have designated: ]}
{sentence}
```

`sakura_base.py` 更是内置了 **6 套 prompt 模板**并按模型名自动识别：

| prompt_version | 适用 |
|---|---|
| `SakuraLLM v0.9` / `v0.10` / `v1.0` / `v1.5` | Sakura 系列离线模型 |
| `GalTransl` | GalTransl 微调模型 |
| `Hy-MT2` | 腾讯混元翻译 |
| `auto` | 通用系统提示词 |

上下文记忆：`self.context` / `self.contextReal` 保存历史 N 轮对话（`append_context_num` + `use_context`），把「历史翻译」作为 few-shot 塞进 prompt——这是保证长篇译文术语/人称一致性的关键。

### 2.6 离线与生态

- **离线翻译 Sakura**：不是直接跑 ONNX，而是调用本地 `llama-server.exe` 的 OpenAI 兼容 HTTP API，并用 `NativeUtils.GetProcessListenPort("llama-server.exe")` **自动探测本地端口**。
- **翻译引擎 44 个**： DeepL / Google / Bing / 百度 / 有道 / 腾讯 / 阿里 / 火山 / 华为 / 彩云 / 小牛 / Yandex / Papago / OpenAI 兼容（gptcommon.py 18KB）/ Sakura / EzTrans / 金山 / 人工翻译……
- **词典 11 个**：`mdict.py`（27KB，支持 .mdx 格式）、mojidict、jisho、weblio、jpdb、youdao、japandict
- **OCR 19 个**、**TTS 15 个**
- **语言学习**：日语分词 + 假名注音 + AnkiConnect + Yomitan 插件

---

## 第三部分：逐项对比

| 维度 | votex（声阅） | LunaTranslator | 判定 |
|---|---|---|---|
| **定位** | 离线语音工具（TTS/ASR/OCR/翻译一体的内容创作流水线） | 视觉小说实时翻译器 | 赛道不同 |
| **语言/架构** | Rust，DDD 四层，5 crates | Python + C++，单体 + 插件式目录 | votex 更规范 |
| **平台** | 跨 Windows/Linux/macOS | **仅 Windows** | ✅ votex |
| **分发** | 单二进制，零运行时依赖 | PyInstaller + Python 运行时，数百 MB | ✅ votex |
| **离线能力** | 4 个本地 ONNX 模型，零网络 | 需 `llama-server.exe` 子进程 | ✅ votex 更纯粹 |
| **许可证** | （项目未明示） | GPLv3 | ⚠️ 注意：GPLv3 有传染性，**不可直接借鉴其代码** |
| **翻译引擎数** | 8（4 离线 + 2 API + 词典 + 空壳） | 44（含 40+ 在线 API） | ⚠️ 生态广度差距大 |
| **Provider 抽象** | 2 方法 trait，无生命周期/能力探测 | `basetrans` 含类型/队列/缓存/流式/简繁/自愈 | ❌ Luna 更成熟 |
| **术语表** | ❌ 无 | ✅ 占位符 + prompt 注入 + 三作用域 + 按游戏配置 | ❌ **核心差距** |
| **上下文记忆** | ❌ 无 | ✅ 历史 N 轮 few-shot | ❌ 差距 |
| **翻译前/后处理** | ❌ 无 | ✅ `transoptimi/` 6 个优化器 + 用户脚本热重载 | ❌ 差距 |
| **翻译缓存** | ❌ 无 | ✅ md5(src,tgt,text) | ❌ 差距 |
| **请求调度** | 单线程 spawn，无队列 | ✅ 优先级队列 + 中断 + 限速 + 多 Key 轮换 | ❌ 差距 |
| **流式输出** | ❌ 同步阻塞 | ✅ generator + 逐块 callback | ❌ 差距 |
| **简繁转换** | ❌ 无 | ✅ zhconv 自动降级 | ❌ 差距 |
| **批量翻译** | ❌ 无 | ✅（文本源可批量） | ❌ 差距 |
| **长文本分段** | ❌ 无（硬截断） | ⚠️ 按句/换行拆分 | ❌ 差距 |
| **KV Cache** | ✅ Opus-MT 真增量；其余 3 个缺失 | N/A（走 llama.cpp） | ⚠️ votex 有基础但没铺开 |
| **OCR/TTS/ASR 联动** | ✅ 统一流水线模板 | ✅ 各自独立但都有 | 持平（votex 更一体化） |
| **安全合规** | ✅ 配置加密 AES-256-GCM、日志脱敏、路径遍历防护 | ⚠️ 未见体系化设计 | ✅ votex |
| **日志体系** | ✅ 分级 + 轮转 + 32 天保留 + 中文 | ⚠️ 简易 | ✅ votex |
| **配置管理** | ✅ `application.yml` + CLI > ENV > YAML 优先级 | ⚠️ JSON 全局配置 + GUI | ✅ votex |
| **测试** | ✅ 33 个翻译集成/单测，Opus-MT 有 golden | ❌ 未见系统测试 | ✅ votex |
| **GUI** | egui（Rust 原生，跨平台） | PyQt5（成熟、控件丰富） | 各有优劣 |

---

## 第四部分：对 votex 的改进建议（按优先级）

> ⚠️ **法律前提**：LunaTranslator 为 GPLv3。本节仅借鉴**设计思想**（占位符术语替换、前后处理管线、优先级队列），**不得复制其源码**。votex 若需闭源分发，必须保持代码独立实现。

### P0 — 先修 Bug（1~2 天）

1. 统一模型目录名：把 `hy-mt-1.5.yaml` 的 `id` 改为 `hy-mt-1.5`（与磁盘和代码一致），或反过来改代码查找逻辑；`m2m-100` 同理。
2. 引入 **模型会话缓存 / ModelPool**：`Arc<Mutex<HashMap<EngineKind, Arc<Session>>>>`，避免每次翻译重建 Session。这是 GUI 可用性的生死线。
3. 翻译引擎改用 `OrtSessionFactory::create_raw()` 系列，让 `application.yml` 的 `inference.*` 真正生效。

### P1 — 补齐架构（1~2 周）

4. **扩展 `TranslationProvider` trait**（对标 `basetrans`）：
   ```rust
   pub trait TranslationProvider: Send + Sync {
       fn name(&self) -> &str;
       fn translate(&self, text: &str, dir: TranslationDirection) -> Result<String, TranslationError>;

       // 新增
       fn supported_pairs(&self) -> Vec<(String, String)> { vec![] }
       fn is_loaded(&self) -> bool { true }
       fn load(&mut self, dir: &Path) -> Result<(), TranslationError> { Ok(()) }
       fn unload(&mut self) {}
       fn translate_batch(&self, texts: &[&str], dir: TranslationDirection)
           -> Result<Vec<String>, TranslationError>;      // 默认逐个调用
       fn with_glossary(&self, _g: &Glossary) -> Result<(), TranslationError> { Ok(()) }
       fn translate_stream(&self, _text: &str, _dir: TranslationDirection,
                           _cb: &mut dyn FnMut(&str)) -> Result<String, TranslationError>;
   }
   ```
5. **实现术语表（最高价值）**：新增 `domain/translation/glossary.rs`
   ```rust
   pub struct GlossaryEntry { pub src: String, pub dst: String,
                              pub info: Option<String>, pub whole_word: bool,
                              pub case_sensitive: bool }
   pub struct Glossary { entries: Vec<GlossaryEntry> }
   impl Glossary {
       /// 翻译前：术语 → 占位符，返回 (处理后文本, 回填映射, 注入 LLM 的词典)
       pub fn apply_before(&self, text: &str) -> (String, HashMap<String, String>, Vec<GlossaryEntry>);
       /// 翻译后：占位符 → 指定译文
       pub fn apply_after(&self, text: &str, map: &HashMap<String, String>) -> String;
       /// 生成注入 LLM prompt 的词典文本（对齐 Sakura 的 src->dst #info 格式）
       pub fn to_prompt_text(&self) -> String;
   }
   ```
   占位符建议用 `\u{E000}{idx}\u{E001}`（Unicode 私用区，不会被分词器切开、不会与原文冲突）。
6. **长文本分段**：复用 TTS 已有的分段器，按句/标点切分后逐段翻译再拼接，突破 `max_length` 硬截断。
7. **翻译缓存**：`HashMap<u64 /*md5(src+tgt+text+engine)*/, String>` + LRU，落 SQLite。
8. **启用 M2M-100 的 `decoder_with_past`**，并把 KV cache 推广到 NLLB / HY-MT。

### P2 — 完善能力（后续迭代）

9. 新增 `translation:` 配置段 + `resolve_translation_engine()`，把 8 个 Provider 注册进已有的 `ProviderRegistry`。
10. 新增 `translation_tasks` 表 + `TranslationRepository`，支持翻译历史与批量任务持久化。
11. 批量翻译：新增 `batch_translation_use_case.rs` + CLI `votex batch translate`。
12. 流式输出：trait 增加 `translate_stream`，GUI 逐块刷新。
13. 简繁转换：引入 `zhconv` 等价实现（crates.io 有 `zhconv` crate）。
14. CTranslate2：要么实现，要么从 GUI 引擎列表移除（避免用户踩坑）。
15. 补 `opus-mt-en-zh.yaml` 注册表。
16. 补测试：工厂函数、`TranslationUseCase`、术语表前后处理、方向解析一致性。

### 不建议照搬的部分

- ❌ **44 个在线 API 引擎**：与 votex「离线优先、用户文本不上传」的承诺冲突。建议只保留 1~2 个（现有 DeepSeek / Qwen-MT 足够），把精力放在离线能力。
- ❌ **HOOK / 内嵌翻译**：赛道完全不同，votex 面向内容创作而非游戏实时翻译。
- ❌ **GPLv3 代码复用**：任何形式的复制都会污染 votex 的许可。

---

## 结论

**votex 翻译模块的「引擎层」已具雏形**（8 个引擎、4 个可跑的离线 ONNX 模型、Opus-MT 有真 KV cache、有 golden 测试），但**「能力层」几乎是空白**：没有术语表、没有上下文、没有缓存、没有分段、没有批量、没有持久化。 LunaTranslator 恰恰相反——它自己不训练/不跑翻译模型，而是把**「翻译质量工程」**（术语占位替换、前后处理管线、上下文记忆、调度与缓存）做到了极致。

**一句话**：votex 缺的不是翻译引擎，是**围绕翻译的工程化能力**。建议下一阶段把 80% 精力放在 P0/P1 的「术语表 + 会话缓存 + trait 扩展 + 分段」上，而不是继续堆引擎数量。
