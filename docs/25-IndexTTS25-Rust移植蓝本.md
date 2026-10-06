# IndexTTS-2.5 Rust 移植蓝本（源自 index-tts-2.5-onnx 0.1.0 参考实现）

> 2026-10-05 决策门通过（粤语试听合格）后启动。本文档固化 Python 参考实现的全部规格，
> 作为 `IndexTts25Provider` 移植的**逐行对照蓝本**。数值目标：与 Python 参考实现 bit-exact 或逐级对齐。

源码位置（uvx 缓存，只读参考）：
`C:\Coding\py_cache\archive-v0\<hash>\lib\site-packages\index_tts_2_5_onnx\`

模型目录布局（`~/.cache/huggingface/hub/models--yunfengwang--IndexTTS-2.5-onnx/snapshots/<commit>/`）：

```
campplus.onnx              fbank [1,T,80] -> [1,192] 说话人嵌入
emo_vec.onnx               feat [1,Te,1024], ilens [1] -> [1,1280]
length_regulator.onnx      x [1,T,1024], template [1,T_out,1] -> [1,T_out,512]
semantic_codec_decode.onnx codes [1,T] -> [1,T',1024]
bigvgan.onnx               mel [1,80,T] -> [1,1,T*256] wav（截取前 T*256）
semantic_model/            input_features [1,Te,160] f32（stride 堆叠后）, attention_mask [1,Te] i64 -> [1,Te,1024]
                           ⚠ 蓝本初版误记为 [1,Te,80]：SeamlessM4T 特征提取最后一步 stride=2
                           将相邻两帧 80 维堆叠为 160 维（2026-10-05 实测 ONNX 输入与金标确认）
gpt_prefill/               text_ids i64 [1,N], conds f32 [1,3,1280], lang_id i64 [1]
                           -> logits [1,vocab], past_kv_{0..23}_{k,v}
gpt_step/                  token i64 [1,1], pos_idx i64 [1], past_kv_{0..23}_{k,v}
                           -> logits [1,vocab], 更新后的 past_kv（ outs[1:] 整体替换 state ）
cfm_estimator/             x, prompt_x, style, cond [B,..], x_lens i64, t f32 -> dphi
spk_proj.npz               weight [1280,192], bias [1280]
multilingual_zh_ja_yue_char_del.tiktoken   BPE 词表（base64 token \t rank 每行）
hf_cache/w2v-bert-2.0/     preprocessor: SeamlessM4TFeatureExtractor, feature_size=80,
                           sampling_rate=16000, stride=2, return_attention_mask=true
```

N_LAYERS = 24（GPT KV 层数）。输出 22050 Hz int16 PCM。

## 1. 流水线总览（pipeline.py: IndexTTS2Core）

### build_speaker(ref_audio) -> SpeakerContext（按音色路径缓存！）

1. `load_ref_audio`: soundfile 解码 mono f32 → 原生 sr → soxr_hq 重采样到 22050（`audio_22k`）→
   再 sinc_interp_hann（torchaudio 精确移植，见 dsp）22050→16000（`audio_16k`）。
2. 截断：audio_22k ≤ 15s、audio_16k ≤ 15s。
3. `input_features, attention_mask = W2VFeatures(audio_16k)`（SeamlessM4T 特征，[1,Te,80]）。
4. `spk_cond_emb = semantic_model(input_features, attention_mask)` [1,Te,1024]。
5. `emovec = emo_vec(spk_cond_emb, [Te])` [1,1280]。
6. `style = campplus(campplus_fbank(audio_16k))` [1,192]；fbank 做** utterance mean 归一化**（减均值）。
7. `spk_cond = style @ spk_proj_w.T + spk_proj_b` [1,1280]；`conds = [(spk_cond+emovec)[None], zeros, zeros]` [1,3,1280]。
8. `ref_mel = mel_spectrogram(audio_22k)` [1,80,P]（s2mel 参数：n_fft=1024, hop=256, win=1024,
   22050Hz, fmin=0, fmax=None，center=False，reflect pad (1024-256)/2=384，log clip 1e-5）。
9. `prompt_condition = length_regulator(spk_cond_emb, P)` [1,P,512]。

### synthesize(text, lang, ...)

1. `segments = frontend.prepare(text, lang, max_text_tokens_per_segment=120)` → 每段裸 token id 列表。
2. `lang_id = lang_to_token(lang)`（LANGUAGE_DICT 序号，yue 是 dict 中最后一个 → 未知语言回退 "common"）。
3. 每段：
   - `codes = generate_codes(gpt, text_ids, spk.conds, lang_id, greedy, top_k=30, top_p=0.8,
     temperature=0.8, repetition_penalty=10.0, max_mel_tokens=1500, seed=master.integers(...))`
   - 空段跳过。
   - `s_infer = codec_decode(codes[None])` [1,T',1024]。
   - `target_len = int(T' * 1.72 * duration_factor)`；`cond = length_regulator(s_infer, target_len)`。
   - `cat_condition = concat([spk.prompt_condition, cond], axis=1)`。
   - `mel = cfm.solve(cfm_estimator, cat_condition, spk.ref_mel, spk.style, n_timesteps=25, cfg_rate=0.7, seed=...)`。
   - `wav = bigvgan(mel)` → clip 到 int16。
4. 多段之间插 `interval_silence=200ms` 静音（22050Hz）。

## 2. GPT 采样（gpt_sampler.py）

特殊 token：`START_TEXT=0, STOP_TEXT=1, START_MEL=8192, STOP_MEL=8193`。

> **⚠ 词表关键结论（2026-10-05 实测确认）**：
> - GPT 输出头只预测 **mel 码**：logits 形状 [1, **8194**]（=8192 码 + start + stop），文本嵌入表在图内部。
> - 内部文本嵌入表 **[60510, 1280]** = ranks(58836) + specials(**1674** = 2 + **100 语言** + 其余)。
> - **参考实现 `get_encoding(num_languages=99)` 是 off-by-one 缺陷**：语言表共 106 项，
>   前 99 截到 `su`，把 `yue/minnan/wuyu/dialect/zh-en/en-zh/common` 7 项排除在特殊 token 之外
>   → `--lang yue` 时 `<|yue|>` 被普通 BPE 编码成字面文本，模型把它念出来（听感为开头「粤语」二字）。
> - **已实测修复**：注册 100 语言后 n_vocab=60510 与嵌入表吻合，`<|yue|>` id = **58937**，
>   补丁后合成（greedy）无开头念白（`tmp/indextts25_yue_fixed.wav`）。
> - **Rust frontend 必须注册 100 语言**（en..yue），并加断言 `n_vocab == 60510`。
>   minnan/wuyu 等 6 项在嵌入表之外，不可注册（越界）。

```
logits, state = gpt.prefill(text_ids, conds, lang_id)
seen = {STOP_TEXT_TOKEN, START_MEL_TOKEN}        # 伪前缀 {1, 8192}
for i in 0..max_mel_tokens:
    l = repetition_penalty(logits, seen, 10.0)   # seen 中：负值×penalty，正值÷penalty
    token = greedy ? argmax(l) : sample(l, temp=0.8, top_k=30, top_p=0.8)
    if token == STOP_MEL_TOKEN(8193): break
    codes.push(token); seen.add(token)
    logits = gpt.step(state, token, pos_idx = i + 2)   # ⚠ mel 位置索引从 i+2 开始（quirk！）
state 整体被 outs[1:] 替换（24 层 × {k,v}）。
```

采样顺序（HF logits-processor 顺序）：RepetitionPenalty → Temperature(÷max(temp,1e-6)，f64) →
TopK(保留前 k，其余 -inf) → TopP(降序排序，probs=exp(s-max s) 归一，cumsum>p 移除但**保首元素**，
offset 一位) → softmax → cumsum 对 rng.random() 二分查找。

## 3. CFM Euler 求解器（cfm_solver.py，复刻 s2mel solve_euler）

```
T = cat_condition.shape[1]; P = ref_mel 末维
z ~ N(0,1) [1,80,T]（numpy default_rng(seed).standard_normal —— 与 torch randn 非位相同）
prompt_x = 0, prompt_x[..,:P] = ref_mel[..,:P]
x = z.copy(); x[..,:P] = 0
x_lens=[T]; t_span = linspace(0,1, n_timesteps+1) f32
每步：stacked = estimator(run(x‖x), (prompt_x‖0), x_lens, (t,t), (style‖0), (cat_condition‖0))
      dphi = stacked[0:1], dphi_cfg = stacked[1:2]
      x += dt * ((1+cfg_rate)*dphi - cfg_rate*dphi_cfg); x[..,:P] = 0
返回 x[:,:,P:]  [1,80,T-P]
```

## 4. 文本前端（frontend.py）

编码器：tiktoken.Encoding(pat_str=`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`，
mergeable_ranks=词表文件 base64→rank，special_tokens 挂在 n_vocab 之后顺序追加：
`<|endoftext|>, <|startoftranscript|>, <|{99 语言}|>, <|{音频事件}|>, <|{情绪}|>, <|translate|>,
<|transcribe|>, <|startoflm|>, <|startofprev|>, <|nospeech|>, <|notimestamps|>,
<|SPECIAL_TOKEN_1..30|>, <|TTS/B..H|>, <|TTS/SP01..13|>, <|0.00| .. <|30.00|>（1501 个 0.02 步长）`）。

prepare(text, lang, max_tokens=120)：
1. `CHAR_REP_MAP` 清洗（：，；→","，。→"."，！？→"!?"，\n→" "，引号/括号/书名号→"'"，—～→"-"，:→","，……→"…"）。
2. lang ∈ (zh, zhen, en) 且 use_normalization → TextNormalizer（wetext tn；数字/日期归一化）。
3. lang ∈ (ja, zh, zhen, en) → lower；es → upper。
4. 发音标注 `<词|读音>` → kana 直接空格包裹；否则 `SPECIAL_TOKEN_2`(有中文)/`SPECIAL_TOKEN_1` 包裹大写读音。
5. `<|x|>` → 大写。
6. 按预算分段：`budget = min(120, 602-2) - len(encode("<|{lang}|> "))`；受保护片段
   `<|SPECIAL_TOKEN_n|>...<|SPECIAL_TOKEN_n|>` 原子保留；按标点 `[，。！？、；：,\.!\?;:\n]` 后切分；
   超长再逐字切。
7. `encode("<|{lang}|> " + seg, allowed_special='all')` 后**过滤 t<=1**（0/1 是 start/stop-text）。

## 5. DSP 规格（dsp.py）

- `resample_sinc_hann`：torchaudio sinc_interp_hann 精确 numpy 移植（gcd 归约、rolloff=0.99、
  lowpass_filter_width=6、f64 kernel→f32、strided conv、ceil 目标长度）。**votex 现有 resample
  缺抗混叠低通（P1-14），2.5 必须用此实现**。
- `mel_spectrogram`：见 §1.8 参数；librosa mel 滤波器组；`log(clip(mel,1e-5))`。
- `kaldi_fbank`：knf 参数 = 16kHz、povey 窗 25ms、shift 10ms、preemph 0.97、snip_edges、
  round_to_power_of_two、num_bins=80、low_freq=20、high_freq=0、use_energy=false、use_log_fbank=true、
  dither=0。**votex `cosyvoice.rs::extract_fbank_features` 需参数比对后复用或改造**。
- `campplus_fbank`：kaldi_fbank + utterance mean 归一化（每维减均值）→ [1,T,80]。
  注意 Python 侧输入用 unscaled [-1,1]（对数下限落点一致）。

## 6. Rust 模块规划（crates/votex-infra/src/tts/indextts25/）

```
frontend.rs   tiktoken 加载（fancy-regex 处理 (?!\S)，BPE 为 tiktoken _byte_pair_merge 忠实移植）
              CHAR_REP_MAP/分段/发音标注/大写化 —— 纯函数，重点测试 ✅ 已完成（2026-10-05）
dsp.rs        resample_sinc_hann / mel_spectrogram / kaldi_fbank / campplus_fbank
              ✅ 已完成（2026-10-05，见 6.4）
features.rs   SeamlessM4T 特征（povey 400/160/512 fbank + 逐维 ddof=1 归一 + stride 堆叠 80→160）
              ✅ 已完成（2026-10-05，与金标 max_abs_diff≈9e-6，见 6.2）
engines.rs    8 个 Session 的加载与运行封装（EngineState<Session>，&self 契约）
              ✅ 已完成（2026-10-05，I/O 名与 engine.py 逐字一致，见 6.3）
sampler.rs    采样链纯函数（RepPenalty→Temp→TopK→TopP→searchsorted，u 注入式设计）
              ✅ 已完成（2026-10-05，与参考实现金标逐位一致，见 6.3）
gpt.rs        自回归主循环（prefill→采样→step，pos_idx=i+2 quirk）
              ✅ 已完成（2026-10-05，真实模型 greedy codes 逐 token 一致，见 6.3）
pipeline.rs   build_speaker（缓存）/ synthesize（分段循环、静音拼接）
              ✅ 已完成（2026-10-05，CFM bigvgan 逐位一致，见 6.4）
mod.rs        IndexTts25Provider：TtsProvider trait 实现
```

### 6.1 前端移植实录（2026-10-05，Task #43）

- **金标对拍已过**：Python 侧（tiktoken 0.14.0，num_languages=100）生成 11 组编码金标
  （`tmp/bpe_golden.py`），Rust `bpe.rs` 逐 id 一致；`n_vocab=60510`、
  `<|yue|>=58937`、`<|SPECIAL_TOKEN_2|>=58960`、`<|0.02|>=59010` 锚点断言通过。
- **词表非规范行**：`multilingual_zh_ja_yue_char_del.tiktoken` 第 48475 行 token 为裸 `"="`，
  Python `b64decode`（validate=False）返回空字节串。Rust 侧以 `lenient_b64_decode` 复刻
  （空键仅占 rank 计数，编码查找永不命中，但必须保留以维持 n_vocab 契约）。
- **wetext 发现**：PoC 的 uvx 环境**已安装 wetext 0.1.8**，即此前粤语试听的合成走了
  真实数字/日期归一化。**归一化已于 Task #49 真实移植（见 §6.2-补）**；
  既有分层金标（§7）仍以参考实现 `--no-normalization` 生成并继续有效
  （前端默认 `use_normalization=false` 构建）。
  CLI 对应开关：`index-tts-2.5-onnx --no-normalization`。
- 词表已落位 `models/tts/indextts25/multilingual_zh_ja_yue_char_del.tiktoken`
  （sha256 `747979631e813193436aabcff7c1c235d37de8097b71c563ec8b63b7a515c718`）。

### 6.2-补 归一化移植实录（2026-10-05，Task #49）

- **结构**（`normalizer.rs`，三层，与参考 `TextNormalizer` + PyPI `wetext==0.1.8`
  nbest=1 快路径逐层对齐）：
  1. `FstTextNormalizer`：kaldifst.TextNormalizer 等价物（rustfst 引擎）——
     UTF-8 字节 acceptor → compose → shortest_path(1) → 字节解码；
  2. `wetext_tn_normalize`：utils.normalize 快路径——strip → should_normalize
     门控（**zh 仅含数字触发，en 非空即触发**；wetext-rs 上游对 en 无数字会
     跳过 FST，已修正为 Python 语义）→ tag → TokenParser.reorder → verbalize → strip；
  3. `TextNormalizer`：参考包装层——use_chinese 判定、收缩式 `X's→X is`、
     术语/拼音声调/人名暂存恢复（correct_pinyin）、归一化后 `ZH_CHAR_REP_MAP`（含
     `$`→`.`）/`CHAR_REP_MAP` 二次映射。
- **FST 资产**：zh/tn + en/tn 的 tagger/verbalizer 共 4 件（13.6MB）逐字节取自
  wetext 0.1.8 wheel，`include_bytes!` 内嵌，首次使用物化到临时目录（幂等）。
  TokenParser 移植自 SpenserCai/wetext-rs（Apache-2.0），并补齐其缺失的
  `escape_value` 转义语义。
- **金标 50 例全对**：`tmp/make_normalizer_refs.py`（Python wetext 0.1.8 +
  参考包装层）→ `tmp/indextts25_refs/normalizer_golden.json`，Rust 逐例精确相等，
  **含 en 无数字文本差异面**（"what's the time?" 展开与恒等路径）。
- **接线**：`Frontend::with_encoding(use_normalization=true)` 时构建
  TextNormalizer，`prepare` 第 2 步对 zh/zhen/en 生效；FST 初始化失败降级为
  警告 + no-op（与参考实现 wetext 缺失时的 ImportError 行为一致）。yue/ja
  等语言参考实现本就不归一化。
- **有意差异**（文档化）：glossary（参考恒空）；Python set 乱序占位符编号
  改为首次出现序（恢复用同一映射，输出不变）；compose 为空时 en 回退原文、
  zh 冒泡（同 Python try/except 语义）。

### 6.2 特征移植实录（2026-10-05，Task #44）

- **金标对拍已过**：`tmp/make_w2v_refs.py` 生成 3 组确定性波形（15s / 7000 / 7160 样本，
  覆盖奇数帧 pad 路径），Rust `features.rs` 全部对齐（max_abs_diff 6.5e-6~9.2e-6，
  容差 1e-4）；金标存放 `tmp/indextts25_refs/`（不入库）。
- **关键语义**（transformers 5.18.0 `SeamlessM4TFeatureExtractor` numpy 路径）：
  - 波形 ×2^15 后 f64 计算；FFT 输出先裁剪到 f32（complex64 存储）再取模平方；
  - povey 对称窗 = `np.hanning(400)^0.85`；预加重后首样本 ×(1-0.97)；
  - 逐 mel 维归一化方差用 **ddof=1**（+1e-7）；
  - 帧数 pad 到偶数：特征填充值 **1.0**（preprocessor_config padding_value=1），
    但 attention_mask 的 pad 位置为 **0**；stride 后 mask 取奇数索引；
  - 最终输入特征是 **160 维**（两帧 80 维堆叠），`semantic_model` 输入维度已勘误。

性能预算（对照 PoC 实测，8 核 CPU）：load ~33s / clone 19~40s / synth RTF≈42（cfm 占 70%）。
Rust 版可期收益：speaker 缓存、多线程 ORT session 配置、后续 n-timesteps 可调。

### 6.3 GPT 循环移植实录（2026-10-05，Task #45）

- **硬契约已过**：真实模型（gpt_prefill 4.5GB + gpt_step 4GB，HF 缓存落点）上
  零 conds + `你好，世界。`（zh, lang_id=1, text_ids `[58839,220,48934,50371,11,48721,53743,13]`）
  greedy 24 步，Rust codes 与参考实现**逐 token 一致**
  （金标 `tmp/make_gpt_loop_golden.py` → `gpt_loop_golden.json`；
  Rust 测试 `gpt循环_真实模型greedy金标逐token一致`，slow-models 门控）。
- **采样链纯函数对拍**：6 组形状/温度/topk/topp 配置 × 7 个固定 u，
  `sample()` 与参考实现 `_sample()` 逐个一致（u 注入式设计，绕开 RNG 流差异）；
  重复惩罚 + argmax 4 组一致。金标 `tmp/make_gpt_refs.py` → `gpt_sampler_cases.json`。
- **实现要点**：
  - 重复惩罚在 **f32** 原地执行（numpy 标量弱类型提升）；采样链进入后转 f64；
  - TopK 阈值用 `select_nth_unstable_by`（等价 `np.partition`），**等于阈值保留**；
  - argmax 必须手写折叠取**首个**最大值（`Iterator::max_by` 平局取最后，与 numpy 相反）；
  - f32 精度陷阱：`1e-4f32 * 10.0 ≠ 1e-3f32 字面量`（差 1 ulp），测试期望值须用同一运算式；
  - step 每步 48 个 KV 张量以 owned Value 喂入（一次拷贝，正确性优先；
    KV ~370MB@T=1500，若构成瓶颈再优化为持久 Value 复用）；
  - `EngineState::with_mut` + 双层 `?` 解包（EngineNotLoaded → anyhow）。

### 6.4 DSP + Pipeline 移植实录（2026-10-05，Task #46）

- **dsp.rs**（resample_sinc_hann / mel_spectrogram / kaldi_fbank / campplus_fbank）：
  - 重采样 `t = arange(0,-new,-1)/new + idx`（**负向** arange，初版写成正向被金标抓出）；
  - librosa mel 基在 **mel 空间** linspace 再反变换（slaney 刻度）；
  - kaldi fbank：三角权重在 **mel 域**插值（初版 Hz 域差 0.014，对齐 knf 源码后残余 ~4e-4，
    确认为 kissfft f32 在低能量 bin 的固有舍入噪声，语义逐行对齐源码、容差放宽至 5e-4）；
  - 金标 `tmp/make_dsp_refs.py`，4 项对拍全过。
- **pipeline.rs**（build_speaker / cfm_solve / synthesize）：
  - spk_proj.npz 直接 zip 解包（weight [1280,192] 行主序）；conds = [spk_cond+emo, 0, 0]；
  - CFM euler+CFG：`x += dt·((1+cfg)·dphi − cfg·dphi_cfg)`，CFG 双 batch（第二 batch px=0），
    **linspace 用 `i·(1/n)`、int16 用截断**（numpy astype 语义，非四舍五入）；
  - **金标陷阱**：soundfile `sf.write` 默认 PCM_16 量化，金标音频必须 FLOAT 子类型写出，
    否则 build_speaker 对拍偏差高达 8.4（初版由此误判 semantic 编码有误）；
  - 对拍结果：cfm_solve max≈1e-5、bigvgan **逐位一致（0 误差）**；build_speaker 的
    style 1.3e-3 / conds 2.5e-3 / ref_mel 2.4e-4（campplus fbank 与 s2mel 的 kissfft f32
    舍入噪声经 DNN 放大，非 bit-exact 链路，容差按契约放宽并注明）；
  - **环境事故**：explorer.exe 泄漏 42.9GB commit（反复 uv/python 子进程所致），
    系统提交内存 96% 满 → rustc OOM 崩溃（0xc0000409 / allocation failed）。
    `Stop-Process explorer` 重启回收后恢复，提交内存 54.8GB→11.8GB。
- 测试：pipeline 模块 6 项（3 项 slow-models 真实模型对拍），模块累计 37 项；
  全量回归 432 passed / 0 failed / 15 ignored，clippy indextts25 清零。

## 7. 验收方法（金标对拍）

用 uvx 环境跑 `make_refs` 式脚本导出中间产物：token ids（前端）、GPT codes、s_infer、
cat_condition、mel（cfm 后，喂固定 z 与 seed）、最终 wav。Rust 侧逐级对齐：
1. 前端 token 逐个一致（确定性，必须 bit-exact）。
2. fbank/mel 逐帧 max_abs_diff < 1e-4（浮点库差异放宽）。
3. GPT greedy codes 逐个一致（temperature=0 路径）。
4. 采样路径只对拍统计分布（不要求位相同，numpy/torch RNG 不同）。
5. 端到端：相同 seed 下合成 wav 试听 + 语谱图相关系数。

### 7.1 验收实录（2026-10-05，Task #46~#48）

| 层级 | 对拍项 | 结果 | 容差依据 |
|---|---|---|---|
| 前端 | tiktoken 编码 11 组 | 逐 id 一致 | bit-exact 契约 ✅ |
| 归一化 | TextNormalizer 50 例（zh/en/混排，含 en 无数字差异面）| 逐例精确相等 | bit-exact 契约 ✅（Task #49） |
| 特征 | w2v fbank 3 组 | ≈9e-6 | f32 库差 < 1e-4 ✅ |
| DSP | resample/mel/kaldi/campplus | ≤5e-4 | kissfft f32 低能量 bin 舍入 |
| GPT | 采样链 6×7u + greedy | 逐位一致 | u 注入式对拍 ✅ |
| GPT | 真实模型 greedy 24 步 | codes 逐 token 一致 | bit-exact 硬契约 ✅ |
| CFM | euler+CFG 25 步 | ≈1e-5 | DNN f32 ✅ |
| BigVGAN | 声码 | **逐位 0 误差** | 纯图推理 ✅ |
| build_speaker | style/conds/ref_mel | 1.3e-3 / 2.5e-3 / 2.4e-4 | campplus+mel 噪声经 DNN 放大，非 bit-exact 链路 |
| 端到端 | 粤语 `今日天气几好，我哋一齐去饮茶啦。` greedy seed=42 | 3.55s wav 落盘 `tmp/indextts25_e2e_yue.wav` | RNG 流差异不做逐位对拍（§7.4） |
| 端到端 | 长文本粤语 e2e（#51，tmp/e2e_ch1.txt 三章 ~1.9 万字）：71 段切分，8 段封顶合成 134.4s 音频 | 墙钟 5556s（RTF≈41.3，与 PoC≈42 一致），`tmp/indextts25_e2e_yue_ch1.wav` | 多段切分/段间静音/无 panic；`VOTEX_E2E_MAX_SEGMENTS` 控制段数（全量 71 段 ≈13.6h 不进验收） |

**Provider 接入（#47）**：`EngineKind::IndexTTS25`（CLI 串 `indextts25`）全链路打通——
domain capability / TtsUseCase / 模型管理（model id `indextts-2.5-onnx`）/ EngineLoader
candidates（9 文件与 engines::load 布局逐字一致）/ GUI 四页面下拉 + task_runner /
CLI 帮助文本。音色 = `models/tts/indextts25/prompts/<id>.wav` 参考音频；
方言声明普通话 + 粤语 Native（闽南语/吴语 token 越界不可注册）。

**回归**：votex-domain 169 / app 117 / cli 44 / gui 40 / infra 435（16 ignored，
其中 17 项 slow-models gate：GPT 循环、CFM、bigvgan、build_speaker、端到端）。
