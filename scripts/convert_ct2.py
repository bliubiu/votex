#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ctranslate2>=4.5,<5",
#     "transformers[sentencepiece]>=4.44,<5",
#     "torch>=2.3",
#     "huggingface-hub>=0.24",
# ]
# ///
"""将 HuggingFace 翻译模型转换为 CTranslate2 格式（votex 离线加速后端）。

votex 通过 ct2rs 绑定加载 CTranslate2 模型（`cargo build --features ct2`）。
本脚本使用官方转换器将 HF 权重转换为 int8 量化产物，并补齐分词文件
（官方转换器**不会**自动复制 SentencePiece 分词器，ct2rs 读取 `source.spm`
/ `target.spm` 时缺失会直接报错）。

用法（uv 管理，禁止 pip）：

    uv run scripts/convert_ct2.py                                    # 默认 opus-mt-zh-en
    uv run scripts/convert_ct2.py --model Helsinki-NLP/opus-mt-zh-en \
                                  --output models/translation/ct2-opus-mt-zh-en

产物目录约定（与 votex-infra 工厂探测一致）：

    models/translation/ct2-opus-mt-zh-en/
      model.bin                    # int8 权重（就绪标记）
      shared_vocabulary.bin        # 词表（转换器生成）
      source.spm / target.spm      # 分词器（本脚本补齐）
      vocab.json                   # 原始 Marian 词表（调试/对照用）

网络说明：默认模型从 HuggingFace 下载；若直连失败，脚本自动切换
`HF_ENDPOINT=https://hf-mirror.com` 镜像兜底。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# 必须在 import hf 相关库之前设置镜像兜底（huggingface_hub 在导入时读取环境变量）
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

DEFAULT_MODEL = "Helsinki-NLP/opus-mt-zh-en"
DEFAULT_OUTPUT = Path("models/translation/ct2-opus-mt-zh-en")

# 转换后必须存在的核心文件
REQUIRED = ["model.bin"]
# 分词文件：官方转换器不复制，需要从 HF 仓库补齐（ct2rs sentencepiece 依赖）
TOKENIZER_FILES = ["source.spm", "target.spm", "vocab.json"]

# 词表文件命名因模型架构而异，转换器只生成其中一种
VOCAB_CANDIDATES = [
    "shared_vocabulary.bin",
    "shared_vocabulary.json",
    "source_vocabulary.bin",
    "vocabulary.bin",
]


def log(msg: str) -> None:
    print(f"[convert_ct2] {msg}", flush=True)


def hf_download(repo_id: str, filename: str, cache_dir: Path) -> Path:
    """从 HF 仓库下载单个文件（走镜像兜底），返回本地缓存路径。"""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(cache_dir),
        )
    )


def main() -> int:
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="HF 模型 → CTranslate2 int8 转换")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF 模型名（默认 %(default)s）")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="输出目录（默认 %(default)s）",
    )
    parser.add_argument(
        "--quantization",
        default="int8",
        choices=["int8", "int8_float16", "float16", "float32"],
        help="量化方式（默认 %(default)s）",
    )
    args = parser.parse_args()

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)

    # ===== 1. 官方转换器：加载 HF 模型并转换（含 int8 量化） =====
    log(f"转换 {args.model} → {output}（{args.quantization}）")
    log("首次运行会通过 uv 安装 torch 等依赖并下载模型，耗时较长，请耐心等待")
    from ctranslate2.converters import TransformersConverter

    converter = TransformersConverter(args.model, load_as_float16=True)
    converter.convert(str(output), quantization=args.quantization, force=True)
    log("转换完成")

    # ===== 2. 补齐分词文件（官方转换器不会复制） =====
    cache_dir = Path(os.environ.get("VOTEX_HF_CACHE", ".cache/hf"))
    for name in TOKENIZER_FILES:
        target = output / name
        if target.exists():
            log(f"分词文件已存在，跳过下载: {name}")
            continue
        try:
            src = hf_download(args.model, name, cache_dir)
        except Exception as e:  # noqa: BLE001
            log(f"下载 {name} 失败: {e}")
            continue
        shutil.copyfile(src, target)
        log(f"已复制分词文件: {name}")

    # ===== 3. 产物枚举校验 =====
    log("产物清单：")
    missing = []
    total = 0
    for p in sorted(output.iterdir()):
        if p.is_file():
            size = p.stat().st_size
            total += size
            log(f"  {p.name:<32} {size / 1024 / 1024:>10.1f} MB")
    for name in REQUIRED:
        if not (output / name).exists():
            missing.append(name)
    if not any((output / v).exists() for v in VOCAB_CANDIDATES):
        missing.append("词表文件（shared_vocabulary.bin 等）")
    for name in ("source.spm", "target.spm"):
        if not (output / name).exists():
            missing.append(name)

    if missing:
        log(f"校验失败，缺少关键文件: {missing}")
        log("提示： Marian 模型分词文件为 source.spm/target.spm；若仓库无 vocab.json 可忽略该文件缺失")
        return 1

    log(f"校验通过，共 {total / 1024 / 1024:.1f} MB → {output}")
    log("votex 侧使用：cargo build --features ct2 后选择 ctranslate2 引擎即可自动探测该目录")
    return 0


if __name__ == "__main__":
    sys.exit(main())
