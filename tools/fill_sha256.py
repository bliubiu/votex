"""为已下载的模型文件回填真实 SHA256。

原则：
- 只对**本地真实存在**的文件计算哈希，绝不臆造。
- 本地不存在的条目保持原样（sha256 缺省），由下载器在下载时校验。
- 回填时保留原有注释与字段顺序，只在 `name:` 后插入 `sha256:`。

用法：
    python tools/fill_sha256.py --dry-run   # 只报告
    python tools/fill_sha256.py --apply     # 实际写入
"""
import argparse
import hashlib
import io
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY = os.path.join(ROOT, "models", "registry")

KIND_DIR = {
    "tts": "tts",
    "asr": "asr",
    "ocr": "ocr",
    "translation": "translation",
    "runtime": "runtime",
}


def parse_registry(text):
    """极简 YAML 解析：只取顶层标量与 files 列表。"""
    entry = {"files": []}
    cur = None
    in_files = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        st = line.strip()

        if indent == 0:
            in_files = False
            if ":" in st:
                k, v = st.split(":", 1)
                entry[k.strip()] = v.strip()
            continue

        if indent == 2 and st == "files:":
            in_files = True
            continue

        if in_files and indent == 2 and st.startswith("- name:"):
            cur = {"name": st.split(":", 1)[1].strip(), "has_sha": False, "size": None}
            entry["files"].append(cur)
            continue

        if in_files and cur is not None and indent == 4:
            if st.startswith("sha256:"):
                cur["has_sha"] = True
            elif st.startswith("size:"):
                cur["size"] = st.split(":", 1)[1].strip()
    return entry


def storage_dir(entry):
    """与 Rust 侧 storage_dir() 保持一致。"""
    if entry.get("target_dir"):
        return entry["target_dir"].strip("/")
    leaf = entry.get("sub_dir") or entry.get("id", "")
    kind = KIND_DIR.get(entry.get("kind", "").lower(), entry.get("kind", ""))
    return f"{kind}/{leaf}"


def sha256_of(path, chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fill(text, entry, models_root, apply_changes, stats):
    """返回填好 sha256 的文本。

    幂等性要求：同一个条目内**最多只能有一行** `sha256:`。
    早期版本只在 `- name:` 的紧邻下一行判断是否已存在哈希，
    而部分清单（如 onnxruntime.yaml）把 `sha256:` 放在
    `platforms:` / `required:` 之后，于是被误判为"缺哈希"而重复插入，
    生成 YAML 重复键 —— serde_yaml 会让**整个文件**解析失败并被静默跳过。
    因此这里改为：先收集条目区间，剔除所有已存在的 sha256 行，
    再按需在 `- name:` 后插入唯一一行。
    """
    sdir = storage_dir(entry)
    base = os.path.join(models_root, sdir)
    lines = text.splitlines()

    # ---- 第一遍：定位每个文件条目的行区间 [start, end) ----
    starts = []
    for i, line in enumerate(lines):
        st = line.strip()
        if not st or st.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 2 and st.startswith("- name:"):
            starts.append(i)
    spans = {}
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        spans[s] = e  # 末条目延伸到文件尾（含注释块，删除时只认 sha256 行，安全）

    # ---- 第二遍：剔除条目内已有的所有 sha256 行，记住原值 ----
    drop = set()
    existing = {}
    for s, e in spans.items():
        name = lines[s].strip().split(":", 1)[1].strip()
        val = None
        for j in range(s + 1, e):
            st = lines[j].strip()
            if not st or st.startswith("#"):
                continue
            indent = len(lines[j]) - len(lines[j].lstrip())
            if indent == 4 and st.startswith("sha256:"):
                drop.add(j)
                if val is None:
                    val = st.split(":", 1)[1].strip()
        existing[s] = (name, val)

    # ---- 第三遍：重建输出，需要时补唯一一行 ----
    out_lines = []
    for i, line in enumerate(lines):
        if i in drop:
            continue
        out_lines.append(line)
        s = i
        if s in spans:
            name, val = existing[s]
            if val:
                stats["already"] += 1
            else:
                fpath = os.path.join(base, name.replace("/", os.sep))
                if os.path.isfile(fpath):
                    out_lines.append(f"    sha256: {sha256_of(fpath)}")
                    stats["filled"] += 1
                else:
                    stats["missing_local"] += 1

    return "\n".join(out_lines) + ("\n" if text.endswith("\n") else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="实际写入（默认只报告）")
    ap.add_argument("--only", help="只处理指定 yaml（不含扩展名）")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(REGISTRY) if f.endswith(".yaml"))
    if args.only:
        files = [f for f in files if os.path.splitext(f)[0] == args.only]

    models_root = os.path.join(ROOT, "models")
    grand = {"filled": 0, "missing_local": 0, "already": 0}
    t0 = time.time()

    for fname in files:
        path = os.path.join(REGISTRY, fname)
        text = io.open(path, encoding="utf-8").read()
        entry = parse_registry(text)
        stats = {"filled": 0, "missing_local": 0, "already": 0}
        new_text = fill(text, entry, models_root, args.apply, stats)

        for k in grand:
            grand[k] += stats[k]

        if stats["filled"] or stats["missing_local"]:
            print(
                f"{fname:34s} 回填 {stats['filled']:3d}  本地缺失 {stats['missing_local']:3d}"
                f"  已有 {stats['already']:3d}"
            )
        if args.apply and new_text != text:
            io.open(path, "w", encoding="utf-8", newline="\n").write(new_text)

    print(
        f"\n合计：回填 {grand['filled']}  本地缺失 {grand['missing_local']}  "
        f"已有 {grand['already']}   耗时 {time.time() - t0:.1f}s"
    )
    if not args.apply:
        print("（这是演练，加 --apply 实际写入）")


if __name__ == "__main__":
    main()
