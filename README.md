# Video Editor — AI 剧集解说 / 混剪工具

把一整集电视剧自动剪成**带中文解说或影评弹幕的短视频**，同时输出竖屏 9:16 和横屏 16:9 成片。

```
原片 → 语音转写+说话人分离 → 角色命名 → LLM 挑剧情节拍 → 解说配音 / 影评评论 → 裁剪拼接 → 混音 → 字幕渲染
```

- **语言无关**：ASR 用 WhisperX，LLM 直接读原文台词，韩剧、日剧、美剧、国产剧都能处理。
- **三种剪辑模式**：画面主导解说、旁白主导解说、原声混剪+影评评论。
- **防"张冠李戴"**：说话人分离 → LLM 读台词给说话人命名 → 可选人脸库 + 视觉模型看画面 → 整集自检。
- **Web 操作台**：选集、调参、编辑剧情背景、看日志、做封面、人脸标注，全在浏览器里完成。
- **智能缓存**：每一步都有产物指纹，参数变了自动重跑，没变就复用。

---

## 目录

- [剪辑模式](#剪辑模式)
- [环境要求](#环境要求)
- [安装](#安装)
- [配置](#配置)
- [使用指南](#使用指南)
- [处理流程](#处理流程)
- [缓存与重跑](#缓存与重跑)
- [输出结构](#输出结构)
- [常见问题](#常见问题)
- [项目结构](#项目结构)
- [开发与测试](#开发与测试)

---

## 剪辑模式

由 `narration.explain.mode` 决定（Web 操作台运行页也能直接选）：

| 模式 | 画面 | 声音 | 适合 |
|---|---|---|---|
| **`picture_led`**（默认） | 原片连续完整场景（每段约 12–40 秒） | 解说配音叠在画面上，只在解说说话时压低原声 | 保留原片节奏、临场感强 |
| `narration_led` | 画面长度由解说配音时长反推（每拍 5–15 秒蒙太奇） | 原声全程压低铺底 | 信息密度高、纯口播节奏 |
| `remix` | 挑高潮片段原样播放，最高钩子片段可前置作预告 | 保留原声台词，不配音 | 混剪 + 屏幕上的影评/弹幕短评 |

`picture_led` 还有三种选材方案（`narration.explain.cut_style`）：`bridge`（高潮 + 解说桥接）、
`skeleton`（主线骨架 + 高潮）、`pure`（纯高潮爽剪）。

---

## 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Linux（在 Ubuntu 20.04/22.04 上验证）；macOS 可跑但没有 CUDA 加速 |
| Python | 3.10 – 3.12 |
| GPU | 推荐 NVIDIA GPU ≥ 8 GB 显存（ASR/说话人分离/本地视觉模型）。没有 GPU 也能跑，只是慢很多 |
| ffmpeg | 必需，≥ 5.0，需要在 `PATH` 里 |
| LLM | 任意 **OpenAI 兼容** 的 Chat API（DeepSeek、通义千问、Kimi、OpenAI 等） |
| HuggingFace 账号 | 说话人分离用到 pyannote 的 gated 模型，需要 token |

---

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/VasiliShi/ai-drama-editor.git
cd ai-drama-editor
```

### 2. 安装系统依赖

```bash
# Ubuntu / Debian
sudo apt update
sudo apt install -y ffmpeg fonts-wqy-microhei

# macOS
brew install ffmpeg
```

字幕需要中文字体。程序按以下顺序查找，找到第一个即用：

1. `fonts/wqy-microhei.ttc`（项目目录，可自己放入任意中文字体并在 `subtitle.font_path` 指定）
2. `/usr/share/fonts/truetype/wqy/wqy-microhei.ttc`（上面 apt 装的就是这个）
3. `/usr/share/fonts/{truetype,opentype}/noto/NotoSansCJK-Regular.ttc`

### 3. 创建 Python 环境

```bash
conda create -n video_editor python=3.12 -y
conda activate video_editor
# 或者：python -m venv .venv && source .venv/bin/activate
```

### 4. 安装 PyTorch

**先装 torch，再装其他依赖**，否则 pip 可能拉一个不匹配 CUDA 的版本。项目在
`torch 2.7.0 + CUDA 12.8` 上验证，按自己的 CUDA 版本到 [pytorch.org](https://pytorch.org/get-started/locally/) 选择对应命令，例如：

```bash
# CUDA 12.8
pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# 仅 CPU
pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu
```

> ⚠️ `torchaudio` 必须和 `torch` 版本、CUDA 后缀完全一致。如果环境里已经有 torch（比如云 GPU
> 镜像自带），只补装同版本的 torchaudio：`pip install torchaudio==<同版本> --index-url ... --no-deps`。

### 5. 安装项目依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 里把 `whisperx==3.3.1 / pyannote.audio==3.3.2 / ctranslate2==4.5.0 /
faster-whisper==1.1.1` 钉死了——whisperx 4.x 要求 torch 2.8 + pyannote 4，与本项目不兼容，**不要升级**。
装完检查 torch 没被替换：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

可选功能按需安装：

```bash
pip install -r requirements-optional.txt
```

| 可选依赖 | 用于 |
|---|---|
| insightface、onnxruntime-gpu、opencv-python、scikit-learn | Web 操作台「人脸标注」页（无 GPU 把 `onnxruntime-gpu` 换成 `onnxruntime`） |
| transformers、accelerate、qwen-vl-utils | 画面备注用本地视觉模型（`scene_notes.backend: local`） |

### 6. 配置 HuggingFace token（说话人分离）

1. 注册 [huggingface.co](https://huggingface.co)，在 *Settings → Access Tokens* 创建一个 Read token。
2. 打开下面两个页面，分别点 **Agree and access repository** 接受协议：
   - <https://huggingface.co/pyannote/speaker-diarization-3.1>
   - <https://huggingface.co/pyannote/segmentation-3.0>
3. 把 token 告诉程序，任选一种：

   ```bash
   huggingface-cli login                 # 写入 ~/.cache/huggingface/token
   # 或
   export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
   ```

> 国内网络访问 HuggingFace 可设置 `export HF_ENDPOINT=https://hf-mirror.com`，或者走代理
> （SOCKS 代理需要 `socksio`，`requirements.txt` 里的 `httpx[socks]` 已包含）。
> 不需要说话人信息时可以在配置里关掉 `asr.diarize`，就不需要 token。

### 7. 创建配置文件

```bash
cp config.yaml.example config.yaml
```

至少要改 `llm` 段（见下一节）。`config.yaml` 已在 `.gitignore` 里，不会被提交。

### 8. 验证安装

```bash
python -m pytest -q                       # 单元测试，不需要 GPU / 网络
python src/main.py --help                 # CLI 能正常启动
python -c "import sys; sys.path.insert(0,'src'); from gpu import get_video_encoder; print(get_video_encoder(23))"
                                          # 查看实际使用的视频编码器（h264_nvenc / libx264 …）
```

---

## 配置

完整字段及注释见 [`config.yaml.example`](config.yaml.example)。上手只需要关心下面几项：

```yaml
llm:                                    # 必填：任意 OpenAI 兼容接口
  base_url: "https://api.deepseek.com/v1"
  api_key: "${DEEPSEEK_API_KEY}"        # 可直接写密钥，也可用 ${环境变量}
  model_name: "deepseek-chat"

video:
  input_dir: "input"
  skip_intro_seconds: 90                # 片头时长，自动排除
  skip_outro_seconds: 180               # 片尾时长，自动排除
  series_background_file: "data/series_background.md"   # 剧情背景（见下文，强烈建议）

asr:
  model: "medium"                       # tiny | base | small | medium | large-v3，剧情类建议 medium 以上
  language: "ko"                        # 片源语言：zh | en | ja | ko …
  diarize: true                         # 说话人分离（需要 HF token）
  compute_type: "float16"               # GPU 用 float16；显存紧张用 int8；CPU 用 float32
  min_speakers: 4                       # 已知人数范围填上能明显提升分离准确度
  max_speakers: 18

narration:
  tts:
    engine: "edge"                      # edge（微软免费，无需密钥）| volcano（火山引擎，付费）
  explain:
    mode: "picture_led"                 # picture_led | narration_led | remix
    target_minutes: 7                   # 目标成片时长
  continuity:
    series_name: "剧名"                 # 填了会启用跨集连续性/「上集回顾」

author:
  name: "你的署名"                      # 竖排水印
```

其余常用开关：

| 字段 | 说明 |
|---|---|
| `naming.enabled` | LLM 读台词把 `SPEAKER_xx` 换成角色真名（需要剧情背景里有人物介绍） |
| `scene_notes.enabled` | 让视觉模型看每个节拍的画面，给解说/评论补「画面事实」。`backend: local` 用本机 GPU（默认 Qwen3-VL-2B，约 4.5 GB 显存），`backend: api` 用 OpenAI 兼容多模态接口 |
| `commentary.*` | remix 模式的评论字数、密度、三类评论占比、字色 |
| `output_formats` | 输出画幅和分辨率，默认竖屏 1080×1920 + 横屏 1920×1080 |
| `description / copyright_notice / ending / bgm` | 顶部标题、二创声明、片尾文字、背景音乐 |

### 剧情背景文件（强烈建议）

LLM 只能看到台词。给它一份剧情背景，分析、命名、解说、评论的准确度都会明显提升。
参考 [`docs/series_background.example.md`](docs/series_background.example.md)，复制到
`data/series_background.md` 改写即可（也可以在 Web 操作台「📖 剧情背景」页直接编辑）。

两条格式约定：

1. 人物介绍段的标题（`##` / `###`）里要包含 **「人物」** 二字——这是说话人命名和评论人名白名单的唯一来源；
2. 分集剧情用 `### 第N集：标题`——处理第 N 集时只抽取这一段。

---

## 使用指南

### 准备视频

把视频放进 `input/`，文件名里带集数即可，例如 `1.mp4`、`第02集.mkv`、`S01E03.mp4`。
支持 `mp4 / mkv / mov / avi / flv / wmv`。

### 方式一：Web 操作台（推荐）

```bash
python src/webui/app.py          # 前台运行
./start_webui.sh                 # 或后台常驻（幂等，日志在 logs/webui_stdout.log）
```

浏览器打开 `http://<服务器IP>:6006`。在云服务器上需要开放 6006 端口或用 SSH 端口转发：
`ssh -L 6006:localhost:6006 user@server`。

| 标签页 | 作用 |
|---|---|
| ▶ 运行 / 调试 | 多选集数 → 选模式、选材方案、目标时长 →（可选）强制重做某一步 → 运行。管线在独立子进程里跑，日志实时显示，「停止」会连同 ffmpeg 一起终止。多集自动排队 |
| 📊 集状态 | 每集各阶段（转写/节拍/解说·评论/trimmed/成片）是否已生成 |
| 📖 剧情背景 | 编辑剧情背景 markdown，保存自动备份；可预览某集实际抽到的背景 |
| ⚙️ 设置 | 常用配置的表单 + 整篇 YAML 高级编辑。「仅本次生效」不写文件，「保存为默认」写回 `config.yaml`（保留注释，自动备份 `.bak`） |
| 📜 历史日志 | 查看历史运行日志（密钥已脱敏） |
| 🖼️ 封面图 | 从成片截帧 + 叠加标题，导出封面 |
| 🙂 人脸标注 | 建立人脸库，让画面备注确定性地认出角色（见下文） |

日常用法：运行页**什么都不勾直接跑**，智能缓存会自动决定哪些步骤需要重做。

### 方式二：命令行

```bash
python src/main.py                         # 处理 input/ 下所有视频
python src/main.py --video 2               # 只处理第 2 集（精确匹配 2.mp4）
python src/main.py --video 2 --mode remix  # 临时切换模式
python src/main.py --video 2 --cut-style pure
python src/main.py --config my.yaml        # 使用其他配置文件
python src/main.py --help                  # 全部参数
```

### 处理一部新剧（完整流程）

1. **放视频**：新剧视频放进 `input/`；旧剧的 `output/` 子目录挪走或删掉。
2. **写背景**：按上面的格式写剧情背景（人物介绍必须有，分集剧情有更好）。
3. **改配置**：片源语言 `asr.language`、说话人数范围 `asr.min/max_speakers`、剧名、署名、片头片尾秒数。
4. **（可选）先只转写**：运行页勾「仅转写」批量跑一遍，抽查 `output/<集>/<集>_transcript.json` 质量。
5. **（可选）人脸标注**：只有开启 `scene_notes.enabled` 才需要。「🙂 人脸标注」页：
   ① 生成人物候选 → ② 每格填角色真名（路人留空，同一人拆成两格就填同一个名字）→ ③ 保存。
   会生成 `data/face_gallery.json` 和 `data/cast/<角色名>.jpg`。⚠️ 标记表示该簇可能混了人，重点核对。
6. **运行**：全选集数开跑。角色表首次运行时从背景文件自动抽取（缓存在 `data/cast_roster.json`，背景改了会自动重抽），之后每集自动命名。
7. **取成片**：`output/<集>/` 下的 `*_portrait.mp4` / `*_landscape.mp4`；`bash collect_results.sh` 可把所有成片移到 `output/results/`。

### 说话人命名

`SPEAKER_xx` 是 pyannote 每集单独分配的编号，跨集没有意义。命名步骤让 LLM 读本集台词，
根据自称、称谓、职务（「张队」「王总」「朕」）把编号对应到角色表里的名字：

- 只接受角色表里的名字，表外的一律丢弃；
- 判断不了的保持 `SPEAKER_xx`——匿名比安错名字安全；
- 每集独立，不需要人工标注。

改了角色表想重新命名：

```bash
python src/speaker_naming.py --rename            # 全部集
python src/speaker_naming.py --rename --episode 3   # 参数是 output/ 下的目录名
```

---

## 处理流程

| # | 步骤 | 模块 | 产物 |
|---|---|---|---|
| 1 | 语音识别 | `src/asr.py` | WhisperX 转写 → 词级对齐 → pyannote 说话人分离；跨人长段按词级说话人拆开。`*_transcript.json`、`*_full.srt` |
| 1.5 | 说话人命名 | `src/speaker_naming.py` | 转写里的 `SPEAKER_xx` 换成真名（原编号保留在 `speaker_raw`） |
| 2 | 剧情分析 | `src/analyzer.py` | LLM 选剧情节拍 + 标题，带覆盖度护栏。`*_key_segments.json` |
| 2.5 | 画面备注（可选） | `src/scene_notes.py` | 每拍抽帧给视觉模型，产出「谁在场/在做什么/场景/氛围」；人脸库已建时由 InsightFace 认人。`*_scene_notes.json` |
| 3a | 解说生成 | `src/narration_generator.py` | 解说模式：LLM 写整集解说 + 自检 → TTS 配音 → 按配音时长反推画面时长 |
| 3b | 影评评论 | `src/commentary_generator.py` | remix 模式：逐拍生成解读/影评/共鸣三类短评 → 整集自检 |
| 4 | 裁剪拼接 | `src/cutter.py` | 按节拍裁片段无缝拼接。`trimmed.mp4` |
| 5 | 混音 | `src/audio_mixer.py` | 解说叠加 + 原声压低（picture_led 只在解说时压低，梯形包络防爆音）+ 可选 BGM |
| 6 | 字幕渲染 | `src/subtitle_adder.py` | 缩放加黑边、解说字幕/评论、标题、署名、声明。最终成片 |

**音画对齐**：裁剪、混音、字幕三个模块都使用同一个每拍时长 `end_time - start_time` 和累积起点，
所以第 i 段画面、第 i 条解说、第 i 块字幕逐帧对齐。

**GPU 编码**：`src/gpu.py` 实际试编码 1 秒来探测编码器，按 NVENC → QSV → VideoToolbox → AMF → libx264 依次回退。

---

## 缓存与重跑

每一步都会检查产物是否存在、指纹是否有效（节拍、相关配置变了就失效），有效就复用（日志显示 `CACHED`），
失效就自动重跑。**只有「产物有效但你想换个结果」时才需要强制重做**，并且只勾最上游的那一步，下游会自动级联。

| 想要的效果 | 强制重做 |
|---|---|
| 换了视频文件 / Whisper 模型 / 语言 / 说话人数范围 | 语音识别 `--force-asr` |
| 对选的片段不满意；改了目标时长或模式；改了角色表 | 剧情分析 `--force-analysis` |
| 只想换一版解说词 / 配音 | 解说生成 `--force-narration`（自动连带重新裁剪+混音） |
| remix 模式只想换一批评论 | 影评评论 `--force-commentary` |
| 改了音量 / 原声压低参数 | 混音 `--force-audio-mix`（自动连带重新裁剪） |
| 全部重来 | `--force` |

每一步也都有 `--skip-<step>`（如 `--skip-asr`），在产物缺失时也不跑该步。完整列表见 `python src/main.py --help`。

---

## 输出结构

```text
output/<视频名>/
├── <视频名>_transcript.json      # 转写（带 speaker）
├── <视频名>_full.srt             # 完整字幕
├── <视频名>_key_segments.json    # 节拍、标题、解说词/评论
├── <视频名>_scene_notes.json     # 画面备注（开启时）
├── quality_report.json           # 节拍覆盖等质量指标
├── descriptions.txt              # 发布标题
├── trimmed.mp4                   # 裁剪+混音后的中间文件
├── narration/                    # 解说配音 wav
├── <视频名>_<标题>_portrait.mp4   # 竖屏成片
└── <视频名>_<标题>_landscape.mp4  # 横屏成片

data/
├── series_background.md          # 剧情背景（自己写）
├── cast_roster.json              # 角色表缓存（自动生成）
├── face_gallery.json             # 人脸库（人脸标注生成）
└── cast/                         # 角色参考图（人脸标注生成）
logs/webui/                       # Web 操作台运行日志
```

---

## 常见问题

**Web 操作台打开是 503 / 一直转圈**
Gradio 启动自检会被 `http_proxy` / `all_proxy` 拦截。`app.py` 已经在导入 gradio 前清掉代理、
运行管线时再把代理传给子进程，正常不需要处理。仍有问题时用 `curl --noproxy '*' http://127.0.0.1:6006` 确认服务本身是否正常。

**`unexpected keyword argument 'use_auth_token'` / `weights_only` / 找不到 `libcudnn`**
这三个是 whisperx 3.3 + pyannote 3.3 在新版 huggingface_hub / torch 下的已知兼容问题，`src/asr.py` 里已打补丁自动处理。
如果仍出现，检查是否误升级了 `whisperx` / `pyannote.audio` / `ctranslate2`（版本要和 `requirements.txt` 一致）。

**说话人分离报 401 / 403**
没有接受两个 pyannote 模型的协议，或 token 没生效。见 [安装第 6 步](#6-配置-huggingface-token说话人分离)。

**一集被分出二十多个说话人**
整集级别的 pyannote 会过度切分。填上 `asr.min_speakers` / `asr.max_speakers` 能明显改善；主要角色通常是稳定的。

**`No capable devices found`，编码很慢**
NVENC 不可用（容器没开放 `video` capability 或驱动不匹配），已自动回退 CPU 编码。可用
`ffmpeg -y -f lavfi -i color=c=black:s=320x240:d=1 -c:v h264_nvenc -f null -` 单独测试。

**显存不够**
`asr.compute_type: int8`、调小 `asr.batch_size`；画面备注调小 `scene_notes.batch_size` 或改用 `backend: api`；人脸标注设 `faces.device: cpu`。

**字幕是方块**
没找到中文字体，见 [安装第 2 步](#2-安装系统依赖)。

**edge-tts 偶尔卡住**
微软端点偶发挂死，程序会按 `narration.tts.edge.timeout` 超时重试。频繁失败时可换 `engine: volcano`。

---

## 项目结构

```text
├── config.yaml.example        # 配置模板（复制为 config.yaml）
├── config_schema.yaml         # Web 操作台设置表单的字段定义
├── requirements.txt           # 核心依赖
├── requirements-optional.txt  # 人脸标注 / 本地视觉模型依赖
├── start_webui.sh             # Web 操作台后台常驻启动
├── collect_results.sh         # 把各集成片收集到 output/results/
├── docs/
│   └── series_background.example.md
├── src/
│   ├── main.py                # CLI 入口
│   ├── pipeline.py            # 步骤注册、缓存指纹、流程编排
│   ├── config.py              # 配置加载（支持 ${ENV} 注入）
│   ├── asr.py                 # WhisperX 转写 + 说话人分离
│   ├── speaker_naming.py      # LLM 说话人命名
│   ├── analyzer.py            # 剧情节拍分析
│   ├── scene_notes.py         # 视觉模型画面备注
│   ├── faces.py               # InsightFace 人脸聚类 / 识别
│   ├── narration_generator.py # 解说词 + TTS
│   ├── commentary_generator.py# remix 影评评论
│   ├── cutter.py              # 裁剪拼接
│   ├── audio_mixer.py         # 混音
│   ├── subtitle_adder.py      # 字幕 / 标题 / 水印渲染
│   ├── prompts.py             # 所有 LLM prompt 模板
│   ├── llm.py                 # OpenAI 兼容客户端
│   ├── media.py / gpu.py      # ffmpeg 封装 / 编码器探测
│   ├── episode_utils.py       # 集数解析、背景抽取
│   └── webui/                 # Gradio Web 操作台
└── tests/                     # pytest 单元测试
```

单独调试某个模块：

```bash
python src/cutter.py         config.yaml input/2.mp4        output/2/2_key_segments.json
python src/audio_mixer.py    config.yaml output/2/trimmed.mp4 output/2/2_key_segments.json
python src/subtitle_adder.py config.yaml output/2/trimmed.mp4 output/2/2_key_segments.json
python src/faces.py --config config.yaml       # 命令行生成人脸候选
```

---

## 开发与测试

```bash
pip install pytest
python -m pytest -q
```

测试不需要 GPU、网络和模型文件（LLM 调用已 mock），只需要 ffmpeg 生成合成素材。

---

## 声明

本工具仅用于学习研究和合法的二次创作。处理受版权保护的影视内容前，请确认你拥有相应授权，并遵守所在地法律及发布平台的规则。

## 许可证

[MIT](LICENSE)
