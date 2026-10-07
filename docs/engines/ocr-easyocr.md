# EasyOCR（OCR 本地，多语言备选）

> engine id：`easyocr` | 类别：ocr | 本地 ONNX | registry：`models/registry/easyocr.yaml`

## 用途

80+ 语言的识别备选：非中英文档（如日韩、拉丁语系混排）时优于 PaddleOCR 的中文专项模型。

## 依赖与模型

- 推理：ort；**CRAFT**（文本检测）+ **CRNN**（序列识别）双模型 + 字符字典。

## 能力

- 语种：80+ 语言 | 中文精度不如 PP-OCRv6，定位为多语言兜底。

## 已知怪癖

- 字典与模型档位必须配套（同源下载），混用会输出乱码。
- CRNN 对弯曲/透视文本弱于 CRAFT+注意力识别器，复杂版式优先 PaddleOCR。

## 测试锚点

- 暂无独立集成测试（`slow-models` 候选）；准入清单第 2 项待补。
