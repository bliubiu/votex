## CLI 接口规范

votex CLI 与 GUI 功能完全对等，共享同一套领域层逻辑。所有命令通过 `votex.exe` 子命令调用。

无子命令时默认启动 GUI 界面。



### 1. 通用约定

- 所有路径支持绝对路径和相对路径
- `--help` 或 `-h` 查看命令帮助
- `--version` 或 `-V` 查看版本号
- 布尔型参数：`--flag` 开启，`--no-flag` 关闭
- 枚举型参数：`--value <选项>` 指定值
- 所有参数可选，未指定时使用 `application.yml` 中的默认值
- 错误输出到 stderr，正常输出到 stdout



### 2. TTS 合成命令

```
votex tts --input <文件> --output <文件> [选项]
```

#### 参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `--input` / `-i` | 路径(必填) | 输入 TXT 文本文件路径 | `-i ./novel.txt` |
| `--output` / `-o` | 路径(必填) | 输出音频文件路径（由扩展名推断格式；无扩展名时使用配置文件默认格式） | `-o ./output.wav` |
| `--engine` | 枚举 | TTS 引擎：`kokoro` / `indextts25`，默认 `kokoro` | `--engine indextts25` |
| `--voice` | 字符串 | 音色名称，依赖引擎 | `--voice "xiaobei"` |
| `--speed` | 浮点 | 语速 0.5 ~ 2.0，默认 1.0 | `--speed 1.2` |
| `--pitch` | 整数 | 音调 -20 ~ +20，默认 0 | `--pitch 5` |
| `--volume` | 整数 | 音量 0 ~ 100，默认 80 | `--volume 90` |
| `--segment-size` | 整数 | 分段字数 300/500/800，默认 500 | `--segment-size 800` |
| `--segment-silence` | 整数 | 段落间停顿时长(ms) 0 ~ 2000，默认 300 | `--segment-silence 500` |
| `--crossfade` | 整数 | 交叉淡入淡出时长(ms) 0 ~ 200，默认 50 | `--crossfade 100` |
| `--no-crossfade` | 布尔 | 关闭无缝拼接 | `--no-crossfade` |
| `--num-to-chinese` | 布尔 | 数字转中文（默认开启） | `--no-num-to-chinese` |
| `--bitrate` | 整数 | MP3 比特率(kbps)：128/192/256/320 | `--bitrate 256` |
| `--sample-depth` | 枚举 | WAV 位深：`16` / `24`，默认 16 | `--sample-depth 24` |
| `--denoise` | 布尔 | 输出降噪（双引擎均可用，默认关闭） | `--denoise` |
| `--play` | 布尔 | 合成后自动播放 | `--play` |
| `--open-dir` | 布尔 | 合成后打开输出目录 | `--open-dir` |

#### 示例

```bash
# 基础合成
votex tts -i novel.txt -o output.mp3

# 自定义音色和语速
votex tts -i story.txt -o chapter1.wav --voice "xiaobei" --speed 1.3 --bitrate 256

# 使用 IndexTTS2 引擎 + 粤语音色
votex tts -i cantonese.txt -o output.mp3 --engine indextts25 --voice default

# 长文本分段 + 自定义停顿
votex tts -i long_novel.txt -o audiobook.mp3 --segment-size 800 --segment-silence 500
```



### 3. ASR 识别命令

```
votex asr --input <文件> --output <文件> [选项]
```

#### 参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `--input` / `-i` | 路径(必填) | 输入音频/视频文件路径 | `-i ./recording.mp3` |
| `--output` / `-o` | 路径(必填) | 输出文件路径（自动识别扩展名） | `-o ./subtitle.srt` |
| `--model` | 枚举 | Whisper 模型：`base` / `small`，默认 `base` | `--model small` |
| `--language` | 枚举 | 识别语种：`zh` / `zh-en`，默认 `zh` | `--language zh-en` |
| `--no-punctuation` | 布尔 | 关闭自动添加标点 | `--no-punctuation` |
| `--no-auto-slice` | 布尔 | 关闭长音频自动切片 | `--no-auto-slice` |
| `--slice-length` | 整数 | 切片时长(秒)：15/30/60，默认 30 | `--slice-length 60` |
| `--denoise` | 布尔 | 输入音频降噪预处理（默认关闭） | `--denoise` |
| `--denoise-level` | 枚举 | 降噪强度：`low` / `medium` / `high`，默认 `medium` | `--denoise-level high` |
| `--format` | 枚举 | 导出格式：`txt` / `srt` / `lrc`（默认由输出扩展名推断） | `--format srt` |

#### 示例

```bash
# 基础识别，输出 SRT 字幕
votex asr -i meeting.mp3 -o subtitle.srt

# 中英混合 + small 模型
votex asr -i interview.mp4 -o transcript.txt --model small --language zh-en

# 带降噪识别
votex asr -i noisy_recording.wav -o result.srt --denoise --denoise-level high
```



### 4.2 OCR 识别命令

```
votex ocr --input <文件> [选项]
```

#### 参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `--input` / `-i` | 路径(必填) | 输入图片文件路径 | `-i ./scan.png` |
| `--output` / `-o` | 路径(可选) | 输出文件路径（不指定则输出到终端） | `-o ./result.txt` |
| `--format` | 枚举 | 输出格式：`txt` / `json`，默认 `txt` | `--format json` |
| `--no-cls` | 布尔 | 禁用方向分类（默认启用） | `--no-cls` |

#### 示例

```bash
# 识别图片文字，输出到终端
votex ocr -i scan.png

# 识别并保存到文件
votex ocr -i document.jpg -o result.txt

# JSON 格式输出（含坐标和置信度）
votex ocr -i receipt.png -o result.json --format json

# 禁用方向分类（加速识别）
votex ocr -i photo.png --no-cls
```



### 4. 模型管理命令

```
votex model <子命令> [选项]
```

#### 子命令

| 子命令 | 说明 |
|--------|------|
| `list` | 列出所有已下载和可下载的模型 |
| `download <模型名>` | 下载指定模型 |
| `import <路径>` | 从本地文件导入模型 |
| `remove <模型名>` | 删除已下载的模型 |
| `info <模型名>` | 查看模型详细信息 |

#### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `--type` | 枚举 | 模型类型过滤：`tts` / `asr` / `all`，默认 `all` |
| `--engine` | 枚举 | TTS 引擎过滤：`kokoro` / `indextts25`（仅 `--type tts` 时有效） |

#### 示例

```bash
# 列出所有可用模型
votex model list

# 列出 TTS 模型
votex model list --type tts

# 下载 Kokoro 模型
votex model download kokoro-82m

# 下载 Whisper base 模型
votex model download whisper-base

# 下载 IndexTTS2 模型
votex model download indextts25

# 手动导入本地模型
votex model import ./whisper-base-q5_1.gguf

# 查看模型信息
votex model info kokoro-82m

# 删除模型
votex model remove whisper-base
```

#### 模型名称对照表

| 模型名 | 类型 | 引擎 | 说明 |
|--------|------|------|------|
| `kokoro-82m` | TTS | Kokoro | 普通话 8 种音色 |
| `indextts25` | TTS | IndexTTS-2.5 | 普通话 + 粤语（原生） |
| `whisper-base` | ASR | Whisper | 基础模型，140MB |
| `whisper-small` | ASR | Whisper | 高精度模型，460MB |



### 5. 批量任务命令

#### 5.1 批量 TTS

```
votex batch-tts --input-dir <目录> --output-dir <目录> [选项]
```

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `--input-dir` / `-i` | 路径(必填) | 包含 TXT 文件的输入目录 | `-i ./novels/` |
| `--output-dir` / `-o` | 路径(必填) | 输出音频文件目录 | `-o ./output/` |
| `--engine` | 枚举 | TTS 引擎：`kokoro` / `indextts25` | `--engine indextts25` |
| `--voice` | 字符串 | 音色名称 | `--voice "xiaobei"` |
| `--speed` | 浮点 | 语速 0.5 ~ 2.0 | `--speed 1.2` |
| `--pitch` | 整数 | 音调 -20 ~ +20 | `--pitch 5` |
| `--volume` | 整数 | 音量 0 ~ 100 | `--volume 90` |
| `--segment-size` | 整数 | 分段字数 300/500/800 | `--segment-size 800` |
| `--format` | 枚举 | 导出格式：`wav` / `mp3`，默认 `mp3` | `--format wav` |
| `--bitrate` | 整数 | MP3 比特率(kbps)：128/192/256/320 | `--bitrate 256` |
| `--recursive` | 布尔 | 递归扫描子目录（默认关闭） | `--recursive` |

#### 示例

```bash
# 批量合成目录下所有 TXT
votex batch-tts -i ./novels/ -o ./audiobooks/ --voice "xiaobei"

# 递归扫描子目录
votex batch-tts -i ./novels/ -o ./audiobooks/ --recursive --engine indextts25
```

#### 5.2 批量 ASR

```
votex batch-asr --input-dir <目录> --output-dir <目录> [选项]
```

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `--input-dir` / `-i` | 路径(必填) | 包含音频/视频文件的输入目录 | `-i ./recordings/` |
| `--output-dir` / `-o` | 路径(必填) | 输出文件目录 | `-o ./subtitles/` |
| `--model` | 枚举 | Whisper 模型：`base` / `small` | `--model small` |
| `--language` | 枚举 | 识别语种：`zh` / `zh-en` | `--language zh-en` |
| `--format` | 枚举 | 导出格式：`txt` / `srt` / `lrc` | `--format srt` |
| `--denoise` | 布尔 | 输入音频降噪预处理 | `--denoise` |
| `--recursive` | 布尔 | 递归扫描子目录（默认关闭） | `--recursive` |

#### 示例

```bash
# 批量生成 SRT 字幕
votex batch-asr -i ./recordings/ -o ./subtitles/ --format srt

# 带降噪 + small 模型
votex batch-asr -i ./recordings/ -o ./subtitles/ --model small --denoise
```

### 6. 任务管理命令

```
votex task <子命令> [选项]
```

| 子命令 | 说明 |
|--------|------|
| `list` | 列出所有任务（含排队中、执行中、已完成、已失败） |
| `resume` | 恢复上次未完成的任务（程序重启后使用） |
| `cancel <任务ID>` | 取消指定任务 |
| `clear` | 清空已完成和已失败的任务记录 |

#### 示例

```bash
# 查看任务列表
votex task list

# 恢复未完成任务
votex task resume

# 取消指定任务
votex task cancel task-001

# 清空历史任务
votex task clear
```

### 7. 配置管理命令

```
votex config <子命令> [选项]
```

#### 子命令

| 子命令 | 说明 |
|--------|------|
| `show` | 显示当前全部配置 |
| `get <键>` | 获取指定配置项的值 |
| `set <键> <值>` | 设置指定配置项的值 |
| `reset` | 恢复默认配置 |
| `path` | 显示配置文件路径 |

#### 示例

```bash
# 查看全部配置
votex config show

# 查看默认 TTS 引擎
votex config get tts.default_engine

# 设置默认语速
votex config set tts.default_speed 1.3

# 设置默认导出格式
votex config set output.default_format mp3

# 查看配置文件路径
votex config path
```



### 8. 退出码

| 退出码 | 含义 |
|--------|------|
| 0 | 成功 |
| 1 | 参数错误（输入文件不存在、参数值非法等） |
| 2 | 模型未就绪（模型未下载或加载失败） |
| 3 | 运行时错误（推理失败、磁盘空间不足等） |
| 4 | 用户取消（Ctrl+C） |

### 9. 进度输出

CLI 模式下，进度信息输出到 stderr，格式为：

```
[进度] 当前阶段: N/M (百分比%)
```

示例：
```
[进度] TTS 合成中: 3/12 (25%)
[进度] ASR 识别中: 5/8 (62%)
[进度] 模型下载中: 45MB/140MB (32%)
```