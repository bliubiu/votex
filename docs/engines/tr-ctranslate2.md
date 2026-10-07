# CTranslate2（翻译加速后端）

> engine id：`ctranslate2`（别名 `ct2`）| 类别：translation | 本地（ct2rs 绑定）| registry：`ct2-opus-mt-zh-en`
>
> 状态：**可用（`--features ct2` 门控）** | 模型：本地转换产物，默认构建不携带

## 定位

CTranslate2 是 C++ 高性能 Transformer 推理引擎（int8 量化 + 优化算子 + KV-cache 管理），
对现有 Marian/OPUS-MT 模型实现 **2~4 倍加速 + 约 50% 内存降低**。它是**加速后端**而非独立引擎：
译码语义与原生 opus-mt 对齐，吞吐更高。

## 接入方式（与 libloading 路线的差异）

CTranslate2 **无官方 C API**（仅 C++ / Python），运行时 `libloading` 直调路线已证伪。
最终采用 [`ct2rs`](https://crates.io/crates/ct2rs) 绑定（vendor 模式从源码编译 C++ 库），
按 `ct2` feature 门控为双形态（`crates/votex-infra/src/translation/ctranslate2/`）：

| 形态 | 条件 | 行为 |
| --- | --- | --- |
| `imp_enabled` | `--features ct2` | 真实 int8 推理（`Translator` + 双 SentencePiece 分词） |
| `imp_fallback` | 默认构建 | 模型探测 + 构建引导提示（不链接 C++ 库，默认构建零影响） |

## 构建

```bash
# Windows 需静态 CRT；首次 vendor 编译 CTranslate2 C++ 需 CMake + MSVC + ninja（约 10~20 分钟）
$env:RUSTFLAGS="-C target-feature=+crt-static"; cargo build --features ct2
```

## 模型获取（本地转换）

```bash
uv run scripts/convert_ct2.py        # Helsinki-NLP/opus-mt-zh-en → int8，自动补齐分词文件
```

产物目录（与工厂探测候选一致）：

```
models/translation/ct2-opus-mt-zh-en/
  model.bin                  # int8 权重（就绪标记，~75 MB，原模型约 1/4）
  shared_vocabulary.json     # 词表（转换器产物）
  source.spm / target.spm    # SentencePiece 分词（ct2rs 读取；官方转换器不复制，脚本补齐）
  vocab.json / config.json   # 源模型附带（对照用）
```

## 行为契约

- **方向路由**（对齐 opus-mt）：目录名含 `en-zh` → 英→中模型，否则中→英；
  `Auto` 跟随已加载模型方向；方向不符返回 `UnsupportedDirection`（VTR003）。
- **能力声明**：`supported_pairs()` 动态报告已加载语言对；`max_input_chars = 256`；
  离线引擎（不享受在线缓存语义），术语表走**占位符替换**。
- **批量**：默认逐条调用；ct2rs `translate_batch` 支持真批处理（后续可覆写提效）。
- **降级路径**：未启用 `ct2` feature 时，引擎列表不展示本引擎；即便模型就绪，翻译请求
  会返回「未启用 ct2 feature」的构建引导提示（接入失败诊断脱敏面板）。

## 准入清单

| 项 | 状态 |
| --- | --- |
| 金标准测试 | ✅ `crates/votex-infra/tests/ctranslate2_translation_test.rs`（`--features ct2,slow-models`） |
| 与 opus-mt 金标对拍 | 首跑标定（int8 + beam 搜索可能与 ONNX 贪心结果有细微差异，以测试内断言为准） |
| 失败诊断接入 | ✅ 兜底 + 构建引导提示 |
| 默认构建零影响 | ✅（feature 门控 + fallback 形态） |

## 测试锚点

```bash
# 契约测试（无需模型，仅 feature 开）
cargo test -p votex-infra --features ct2 --test ctranslate2_translation_test
# 金标准（需本地转换模型）
cargo test -p votex-infra --features ct2,slow-models --test ctranslate2_translation_test -- --nocapture
```
