# PaddleOCR（OCR 本地，默认）

> engine id：`paddleocr` | 类别：ocr | 本地 ONNX | registry：`paddleocr.yaml`、`paddleocr-v5-mobile.yaml`、`paddleocr-v5-server.yaml`、`paddleocr-v6-tiny/small/medium.yaml`

## 用途

中文图文识别主力：det（检测）→ cls（方向）→ rec（识别）三段 ONNX 管线；书籍/扫描件/PDF 识别默认引擎。

## 依赖与模型

- 推理：ort 直连；每档位含 det/cls/rec 三个 ONNX。
- 档位：PP-OCRv5 mobile / v5 server / **v6 tiny / v6 small / v6 medium（默认）**。
- 模型目录由各 registry yaml 指针给出（`models/ocr/...`）。

## 能力

- 语种：中 / 英 | 档位-速度/精度梯度完整，GUI/CLI 可选。

## 已知怪癖（重要）

- **v6 系列不使用独立字典文件**：字符表从 ONNX 元数据提取，提取失败回退外部字典（`paddleocr.rs` 字典加载逻辑）——「OCR v6 字典修复」即指此链路。
- **Tiny / Mobile 档对字形近似字存在稳定单字误差**（同字同错），因此默认档位为 v6 Medium。
- RapidOCR 与官方 v5 mobile 权重**逐字节相同**，仅作为评测对照组，不是可用引擎。

## 测试锚点

- `crates/votex-infra/tests/ocr_paddleocr_v6_test.rs`（功能）
- `crates/votex-infra/tests/ocr_rapidocr_eval_test.rs`（档位评测：PP-OCRv6-medium vs PP-OCRv5-mobile(RapidOCR) vs PP-OCRv5-server，需 `--features slow-models`）
