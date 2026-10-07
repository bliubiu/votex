# IndexTTS-2.5（TTS 本地，零样本克隆 + 粤语原生）

> engine id：`indextts25`（别名 `indextts2`）| 类别：tts | 本地 ONNX | registry：`models/registry/indextts-2.5-onnx.yaml`

## 用途

零样本音色克隆主力引擎；粤语原生支持（docs/24/25 完成移植蓝本），多角色有声书与方言合成的首选。

## 依赖与模型

- 推理：ort；模型为 **yunfengwang fp32 分图布局**（已取代早期 IndexTTS2 接入）。
- 模型目录：`models/tts/indextts25/`；克隆参考音频：
  1. 先查 `models/tts/indextts25/prompts/<id>.wav`（引擎私有库）；
  2. 再回落统一音色库 `models/voices/refs/<id>.wav`（`voice add` 入库即用，免手动复制）。
- `list_voices` 合并两处并去重。

## 能力

- 克隆：✅ 零样本，参考音频上限 **15 秒**（`max_ref_seconds` 契约）
- 情感：❌（走其自带情感/语调控制路径，votex 侧无枚举映射）
- 方言：✅ 粤语原生 | 语种：中文

## 已知怪癖

- 参考音频超 15s：**无转写**自动按短时能量 `best_window` 截取；**有转写显式拒绝**（截取窗口与转写无法对应，宁可拒绝不静默出错）——`infra/audio/ref_audio.rs`。
- 全静音参考入库直接拒绝（`silence_ratio`），避免静默产出失真音色。
- 无转写音色在 GUI 标注「仅 IndexTTS-2.5 可用」（CosyVoice 不接受无转写音色）。

## 测试锚点

- `crates/votex-infra/tests/indextts25_yue_e2e_test.rs`（粤语端到端）
- `crates/votex-infra/tests/tts_e2e_voice_role_test.rs`、`tts_voice_multirole_e2e_test.rs`（角色/多角色）
- `crates/votex-infra/tests/tts_asr_闭环_test.rs`
