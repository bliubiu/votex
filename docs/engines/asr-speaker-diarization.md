# 说话人分离（独立模块，非 ASR 引擎）

> engine id：`speaker-diarization` | 类别：asr（特殊）| 本地（sherpa-onnx 绑定）| registry：无独立模型条目（组合管线）

## 用途

**VAD → Pyannote 分割 → x-vector/ECAPA 嵌入 → 聚类**，输出带说话人标签的时间轴片段；服务于多角色有声书（角色↔音色映射）与视频配音（按说话人分段）。

## 依赖与模型

- sherpa-onnx diarization 组件（Pyannote 分割模型 + 嵌入模型），离线运行。

## 能力

- 输出：`[(start, end, speaker_id)]` 时间轴片段
- **不支持普通 `recognize()`**——调用返回 `UnsupportedEngine`；它是基础设施模块，不是可独立转写的引擎。

## 已知怪癖

- `engine_kind()` 归为 `SpeakerDiarization` 但 `AsrProvider::recognize` 不可用，CLI/GUI 不得把它列进普通 ASR 引擎下拉。
- 与 ASR 引擎组合使用：分离出片段 → 逐段送 Paraformer/Whisper 转写。

## 测试锚点

- `crates/votex-infra/tests/asr_e2e_test.rs`（组合链路）
- `crates/votex-infra/tests/en_asr_test.rs`
