"""电影解说工作台入口：python src/webui/app.py

代理处理：管线子进程内 LLM/edge-tts 需走代理，但 Gradio 自检 localhost:6006
不能走代理（否则 503）。必须在 import gradio 之前清掉代理；子进程运行时由
runner 把代理装回子进程 env。
"""
import os
import signal
import subprocess
import sys
import time
import copy
import urllib.parse
from pathlib import Path

# 让 from config/pipeline import 生效（脚本目录 src/webui 自动在 path，再补 src）
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ── 代理剥离（必须在 import gradio 之前）──
_PROXY_KEYS = ("http_proxy", "https_proxy", "all_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")
_SAVED_PROXY = {k: os.environ.get(k) for k in _PROXY_KEYS}
for _k in _PROXY_KEYS:
    os.environ.pop(_k, None)

import yaml
import gradio as gr

from pipeline import STEPS
from runner import JobRunner, list_history_logs, read_history_log
import status as st
import schema as sc
import background as bg

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO_ROOT / "config.yaml"

# 步骤中文标签 ↔ key（force/skip 勾选框用）
STEP_LABELS = [(zh, key) for key, _l, zh, can_f, can_s in STEPS if can_f or can_s]
LABEL_TO_KEY = {zh: key for zh, key in STEP_LABELS}


def _raw_config() -> dict:
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except OSError:
        return {}
def _start_face_job():
    """启动人脸候选子进程（不等待）。stdout 直写日志文件——不能接 PIPE 后不读，
    输出超过管道缓冲(64KB)会和子进程互等死锁。返回 (proc, log_path, timeout_s)。"""
    faces_cfg = _raw_config().get("faces", {}) or {}
    try:
        timeout_seconds = max(30, int(faces_cfg.get("timeout_seconds", 900) or 900))
    except (TypeError, ValueError):
        timeout_seconds = 900
    log_dir = REPO_ROOT / "logs" / "webui"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_faces.log"
    cmd = [sys.executable, "-u", str(_SRC / "faces.py"), "--config", str(CONFIG_FILE)]
    lf = open(log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=lf,
                                stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        lf.close()   # 子进程已持有自己的 fd 副本
    return proc, log_path, timeout_seconds


def _kill_job(proc):
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(10):
        if proc.poll() is not None:
            return
        time.sleep(0.5)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except ProcessLookupError:
        pass


def _collect_secrets(cfg: dict):
    vals = []
    llm = cfg.get("llm", {})
    vals += [llm.get("api_key")]
    vol = cfg.get("narration", {}).get("tts", {}).get("volcano", {})
    vals += [vol.get("access_token"), vol.get("api_key"), vol.get("app_id")]
    return [v for v in vals if v]


RUNNER = JobRunner(_SAVED_PROXY, secrets=_collect_secrets(_raw_config()))


def _deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)

    def _m(d, s):
        for k, v in s.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                _m(d[k], v)
            else:
                d[k] = v
    _m(out, extra)
    return out


# ── Tab: 运行 / 调试 ──────────────────────────────────────────────

def _build_run_tab():
    D = _raw_config().get("narration", {}).get("explain", {})
    with gr.Row():
        with gr.Column(scale=1):
            videos = gr.CheckboxGroup(label="选择视频集（可多选，批量顺序处理）",
                                      choices=st.input_videos(), interactive=True)
            with gr.Row():
                refresh_vid = gr.Button("🔄 刷新", size="sm")
                sel_all = gr.Button("全选", size="sm")
                sel_none = gr.Button("清空", size="sm")
            mode = gr.Radio(["picture_led", "narration_led", "remix"],
                            value=D.get("mode", "picture_led"), label="剪辑模式")
            cut_style = gr.Dropdown(["bridge", "skeleton", "pure"],
                                    value=D.get("cut_style", "bridge"),
                                    label="picture_led 选材方案")
            target_minutes = gr.Slider(2, 15, value=D.get("target_minutes", 7), step=0.5,
                                       label="目标成片时长 (分钟)")
            gr.Markdown("其余全部参数在「⚙️ 设置」页调整（可仅本次生效或保存为默认）。")
            asr_only = gr.Checkbox(
                value=False, label="仅转写（只跑语音识别就停）",
                info="批量准备转写：出带 SPEAKER 的转写即停，不花 LLM/TTS。供人脸标注建库、或先核对转写质量")
            force_steps = gr.CheckboxGroup(
                [zh for zh, _ in STEP_LABELS],
                value=[],
                label="强制重做",
                info="只勾「想改变结果的最上游一步」，下游自动跟着重做。全不勾=智能缓存（推荐日常用法：产物有效就复用、参数变了自动重跑）")
            with gr.Accordion("❓ 强制重做怎么选（速查）", open=False):
                gr.Markdown(
                    "| 勾选 | 这一步做什么 | 什么时候勾 |\n"
                    "|---|---|---|\n"
                    "| 语音识别 | 整集音频→带说话人的转写文本（最慢，GPU 数分钟） | 换了视频文件；改了 Whisper 模型/语言/说话人数范围 |\n"
                    "| 剧情分析 | LLM 读转写挑选精彩节拍（决定剪哪些片段） | **最常用**：对选片不满意想重选；改了目标时长/模式；说话人命名后想让分析用上真名 |\n"
                    "| 画面备注 | VLM 观看选中节拍的画面出备注（需在设置开启） | 基本不用手勾：节拍/模型变了自动重看；换了角色参考图想重新指认时勾 |\n"
                    "| 解说生成 | LLM 写解说词 + TTS 配音 | 只对解说词/配音不满意、选片可以不动（自动连带重新裁剪混音） |\n"
                    "| 影评评论 | remix 模式的三色弹幕影评 | 仅 remix 模式想换一批评论 |\n"
                    "| 裁剪拼接 | ffmpeg 按节拍剪出 trimmed.mp4 | 基本不用手勾：节拍变了会自动重剪 |\n"
                    "| 混音 | 解说配音压混进画面原声 | 改了音量/原声压低等混音参数（自动连带重新裁剪） |\n"
                    "| 字幕渲染 | 缩放+字幕+标题+署名→成片 | 基本不用手勾：字幕/标题/署名参数变了会自动重渲 |\n"
                    "\n上游重做后下游产物指纹自动失效、无需逐个勾选。"
                    "例：想整体重来一集 → 只勾「剧情分析」即可。")
            run_btn = gr.Button("▶ 运行管线", variant="primary", size="lg")
            stop_btn = gr.Button("■ 停止全部（真终止子进程+清空排队）", variant="stop", size="sm")
        with gr.Column(scale=1):
            queue_box = gr.Textbox(label="任务队列", lines=4, interactive=False)
            log_box = gr.Textbox(label="运行日志", lines=22, max_lines=500, autoscroll=True)
            refresh_out = gr.Button("🔄 刷新输出", size="sm")
            out_list = gr.Dropdown(label="选择成片预览", choices=[])
            prev_video = gr.Video(label="预览", height=380)
            out_zip = gr.File(label="📦 下载选中成片", height=70)

    def do_run(vids, md, cs, tmin, force_labels, only_asr):
        if not vids:
            yield "请先勾选至少一集视频", RUNNER.queue_view(), gr.update(), gr.update()
            return
        run_ov = {"narration": {"explain": {
            "mode": md, "cut_style": cs, "target_minutes": tmin}}}
        overrides = _deep_merge(sc.SESSION_OVERRIDES, run_ov)
        force_keys = [LABEL_TO_KEY[l] for l in force_labels]
        # 仅转写：ASR 之后的步骤全部 skip（pipeline 里 skip 优先于 force）
        skip_keys = ([k for k, _l, _z, _f, can_s in STEPS if can_s and k != "asr"]
                     if only_asr else [])
        # 未勾的步骤走智能缓存（不再有独立「跳过」）；每选中一集提交一个任务，单 worker 顺序执行
        jids = [RUNNER.submit(Path(v).stem, overrides, force_keys, skip_keys) for v in vids]

        def _active():
            return [j for j in jids if RUNNER.is_active(j)]

        while _active():
            # 展示当前正在跑的那一集的日志（没有就展示第一个）
            running = next((j for j in jids if RUNNER.get(j).status == "running"), None)
            cur = RUNNER.get(running or jids[0])
            yield cur.log, RUNNER.queue_view(), gr.update(), gr.update()
            time.sleep(0.4)
        # 收尾：拼各集结果概要 + 刷新成片列表
        tail = "\n".join(f"{RUNNER.get(j).video_name}: {RUNNER.get(j).status}" for j in jids)
        last = RUNNER.get(jids[-1])
        outs = st.all_output_mp4s()
        yield (last.log + "\n" + "═" * 40 + "\n批量完成：\n" + tail,
               RUNNER.queue_view(),
               gr.update(choices=outs, value=outs[0] if outs else None), gr.update())

    def _preview(sel):
        p = REPO_ROOT / "output" / sel if sel else None
        return (str(p), str(p)) if p and p.exists() else (None, None)

    refresh_vid.click(lambda: gr.update(choices=st.input_videos()), None, videos)
    sel_all.click(lambda: gr.update(value=st.input_videos()), None, videos)
    sel_none.click(lambda: gr.update(value=[]), None, videos)
    refresh_out.click(lambda: gr.update(choices=st.all_output_mp4s()), None, out_list)
    out_list.change(_preview, out_list, [prev_video, out_zip])
    ev = run_btn.click(do_run,
                       [videos, mode, cut_style, target_minutes, force_steps, asr_only],
                       [log_box, queue_box, out_list, out_zip])
    stop_btn.click(lambda: (RUNNER.stop_all(), RUNNER.queue_view())[1], None, queue_box,
                   cancels=[ev])


# ── Tab: 集状态 ───────────────────────────────────────────────────

def _build_status_tab():
    gr.Markdown("### 每集产物状态（各阶段是否已生成）")
    df = gr.Dataframe(headers=st.STATUS_HEADERS, value=st.episode_rows(),
                      interactive=False, wrap=True)
    gr.Button("🔄 刷新", size="sm").click(lambda: gr.update(value=st.episode_rows()), None, df)


# ── Tab: 剧情背景 ─────────────────────────────────────────────────

def _build_background_tab():
    gr.Markdown(
        "### 剧情背景介绍（人名/人物关系/分集剧情的权威来源）\n"
        "整篇 markdown 编辑：**人物介绍**段（标题含「人物」）每集都会喂给 LLM 校正人名；"
        "**分集剧情**段用 `### 第N集：…` 标题，按集精准抽取。改完保存，重跑分析/解说即生效。")
    content0, info0 = bg.load_background()
    path_info = gr.Markdown(info0)
    editor = gr.Code(value=content0, language="markdown", label="背景 markdown", lines=28)
    msg = gr.Textbox(label="结果", lines=2, interactive=False)
    with gr.Row():
        reload_btn = gr.Button("重新载入", size="sm")
        save_btn = gr.Button("💾 保存", variant="primary")
    with gr.Accordion("🔍 按集预览（analyzer/narration 实际抽到的该集背景）", open=False):
        with gr.Row():
            ep_num = gr.Number(value=1, label="集数", precision=0, scale=1)
            prev_btn = gr.Button("预览抽取结果", scale=1)
        ep_preview = gr.Textbox(label="该集实际背景", lines=12, interactive=False)

    def _reload():
        c, i = bg.load_background()
        return c, i
    reload_btn.click(_reload, None, [editor, path_info])
    save_btn.click(bg.save_background, editor, msg)
    prev_btn.click(bg.preview_episode, [editor, ep_num], ep_preview)


# ── Tab: 设置 ─────────────────────────────────────────────────────

def _build_settings_tab():
    gr.Markdown("### 全部配置（表单改常用项，「高级编辑」改整篇 YAML）\n"
                "密钥字段留空=不改；「仅本次生效」不写文件，「保存为默认」写入 config.yaml(自动备份)。")
    with gr.Tabs():
        with gr.TabItem("表单"):
            comps, paths, types = sc.build_form(_raw_config())
            msg = gr.Textbox(label="结果", lines=3, interactive=False)
            with gr.Row():
                once_btn = gr.Button("仅本次生效", variant="secondary")
                save_btn = gr.Button("保存为默认（写 config.yaml）", variant="primary")
            once_btn.click(lambda *v: sc.apply_once(paths, types, v), comps, msg)
            save_btn.click(lambda *v: sc.save_to_yaml(paths, types, v), comps, msg)
        with gr.TabItem("高级编辑 (YAML)"):
            code = gr.Code(value=sc.load_yaml_text(), language="yaml", label="config.yaml")
            ymsg = gr.Textbox(label="结果", lines=2, interactive=False)
            with gr.Row():
                reload_btn = gr.Button("重新载入", size="sm")
                ysave_btn = gr.Button("保存整篇", variant="primary")
            reload_btn.click(lambda: sc.load_yaml_text(), None, code)
            ysave_btn.click(sc.save_yaml_text, code, ymsg)


# ── Tab: 历史日志 ─────────────────────────────────────────────────

def _build_logs_tab():
    gr.Markdown("### 历史运行日志（logs/webui/，密钥已脱敏）")
    with gr.Row():
        sel = gr.Dropdown(label="选择日志", choices=list_history_logs(), scale=3)
        refresh = gr.Button("🔄", size="sm", scale=1)
    content = gr.Textbox(label="内容", lines=26, max_lines=600)
    refresh.click(lambda: gr.update(choices=list_history_logs()), None, sel)
    sel.change(read_history_log, sel, content)


# ── Tab: 封面图 ───────────────────────────────────────────────────

def _build_cover_tab():
    from media import run_ffmpeg
    import tempfile

    gr.Markdown("### 从成片截帧 + 叠加标题生成封面")
    cover_video = gr.Dropdown(label="选择成片", choices=[])
    gr.Button("刷新列表", size="sm").click(
        lambda: gr.update(choices=st.all_output_mp4s()), None, cover_video)
    with gr.Row():
        with gr.Column():
            capture_time = gr.Slider(0, 600, value=30, step=1, label="截图时间点 (秒)")
            capture_btn = gr.Button("📸 截图", variant="primary")
        with gr.Column():
            title_text = gr.Textbox(value="精彩片段", label="标题文字")
            title_color = gr.Dropdown(["white", "yellow", "red", "gold", "orange"],
                                      value="orange", label="文字颜色")
            title_size = gr.Slider(24, 80, value=48, step=2, label="字号")
            overlay_btn = gr.Button("➕ 叠加文字", variant="secondary")
    preview = gr.Image(label="封面预览", height=400)

    def cap(video, sec):
        if not video:
            return None
        src = REPO_ROOT / "output" / video
        out = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        r = run_ffmpeg(["-ss", str(sec), "-i", str(src), "-vframes", "1", "-q:v", "2", out.name])
        return out.name if r.returncode == 0 else None

    def overlay(img, t, c, s):
        if not img:
            return None
        out = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        r = run_ffmpeg(["-i", img, "-vf",
                        f"drawtext=text='{t}':fontsize={s}:fontcolor={c}:"
                        f"x=(w-text_w)/2:y=h*0.85:fontfile=fonts/wqy-microhei.ttc",
                        "-q:v", "2", out.name])
        return out.name if r.returncode == 0 else None

    capture_btn.click(cap, [cover_video, capture_time], preview)
    overlay_btn.click(overlay, [preview, title_text, title_color, title_size], preview)


# ── Tab: 人脸标注 ─────────────────────────────────────────────────
# 注意:候选行必须用 @gr.render 动态生成。此前用 20 个隐藏 Group + 批量
# gr.update(visible=True) 的方案会触发 gradio 前端(Svelte)无限反应循环,
# 浏览器渲染线程 1 秒内硬死锁(6.5.1/6.10/6.14 均复现,已用无头浏览器矩阵定位:
# 隐藏→可见的批量翻转是唯一触发点,值更新/pending 标记/常驻可见均安全)。


def _crops_html(crops, frames, height=110):
    tags = []
    for c, f in zip(crops, frames):
        tags.append(
            f'<a href="{_file_url(f)}" target="_blank" title="点击看整帧上下文">'
            f'<img src="{_file_url(c)}" loading="lazy" decoding="async" '
            f'style="height:{height}px;margin:0;border-radius:6px"></a>')
    if not tags:
        return "<i>无</i>"
    # flex 流式排列;不用默认 prose 样式(会把 img 变成块级独占一行)
    return ('<div style="display:flex;flex-wrap:wrap;gap:4px;align-items:flex-start">'
            + "".join(tags) + "</div>")


def _build_faces_tab():
    from config import load_config
    from faces import FaceEngine

    gr.Markdown(
        "### 人脸标注：看脸打名字，机器自动检测/裁剪/归类\n"
        "**①** 生成人物候选（各集说话时刻抽帧→检脸→聚类）→ **②** 每格是同一个人，"
        "填真名（路人留空）→ **③** 保存人脸库（自动写 `data/cast/` 参考图）。\n"
        "> 建库后：画面备注(VLM)的在场人物直接出真名，不再靠模型隔空对脸猜。\n"
        "> 同一人拆成两格就填同一个名字（自动合并）；点缩略图可看整帧上下文。\n"
        "> 🔒=已入库人物自动认出（预填名字，改名会覆盖库里的）；⚠️=该簇纯度低可能混人，重点核对。")
    fc_status = gr.Textbox(label="状态", lines=2, interactive=False)
    with gr.Row():
        fgen_btn = gr.Button("① 生成人物候选", variant="primary")
        freload_btn = gr.Button("↻ 重载已有候选", variant="secondary")
    fcands = gr.State([])

    @gr.render(inputs=[fcands])
    def _face_rows(cands):
        if not cands:
            gr.Markdown("_暂无候选。点「① 生成人物候选」，已生成过则点「↻ 重载已有候选」。_")
            return
        ids, boxes = [], []
        n_cols = 5
        for start in range(0, len(cands), n_cols):
            with gr.Row(equal_height=False):
                row_cands = cands[start:start + n_cols]
                for c in row_cands:
                    purity = float(c.get("purity", 1.0) or 0)
                    warn = " ⚠️" if purity < 0.45 else ""
                    known = c.get("name", "")
                    tag = f" · 🔒{known}" if known else ""
                    with gr.Column(scale=1, min_width=180):
                        with gr.Group():
                            n_prof = int(c.get("n_profile", 0) or 0)
                            prof = f" · 侧脸 {n_prof}" if n_prof else ""
                            gr.Markdown(f"**人物 #{c['face_id']}**{tag} · 出现 {c['count']} 次"
                                        f"{prof} · 第 {'、'.join(c.get('episodes', []))} 集{warn}")
                            gr.HTML(_crops_html(c.get("crops", []),
                                                c.get("frames", []), height=72))
                            # 锚点簇(已在人脸库里的人)预填名字,确认即可
                            nm = gr.Textbox(label="真名（留空=忽略）", value=known,
                                            interactive=True,
                                            key=f"face-nm-{c['face_id']}")
                        ids.append(c["face_id"]); boxes.append(nm)
                # 末行补空列,保持 5 等分列宽不被拉伸
                for _ in range(n_cols - len(row_cands)):
                    gr.Column(scale=1, min_width=180)
        fsave_btn = gr.Button("③ 保存人脸库(+cast参考图)", variant="primary")

        def fc_save(*nms):
            name_map = {fid: str(nm).strip() for fid, nm in zip(ids, nms)
                        if str(nm or "").strip()}
            if not name_map:
                return "没有填任何名字。"
            eng = FaceEngine(load_config(str(CONFIG_FILE)))
            eng.build_gallery(name_map)
            return (f"已保存 {len(set(name_map.values()))} 人到 {eng.gallery_file}，"
                    f"参考图写入 {eng.cast_dir}/。画面备注(VLM)会用这些真名指认在场人物。")

        fsave_btn.click(fc_save, boxes, fc_status)

    def _load_face_cands():
        return FaceEngine(load_config(str(CONFIG_FILE))).load_manifest().get("clusters", [])

    def fc_reload():
        try:
            cands = _load_face_cands()
        except Exception as e:
            return f"读取候选清单失败：{e}", []
        if not cands:
            return "暂无候选，请点「生成人物候选」。", []
        return (f"↻ 已从清单重载。共 {len(cands)} 个人物候选。"
                "每格填真名（路人留空），然后保存。"), cands

    freload_btn.click(fc_reload, None, [fc_status, fcands])

    def fc_generate():
        """生成器：立即反馈 + 每 3 秒心跳（保住 SSE 流不被代理闷死）。
        心跳只更新状态框，State 用 gr.skip() 跳过，避免反复触发重绘。"""
        try:
            proc, log_path, timeout_s = _start_face_job()
        except OSError as e:
            yield f"无法启动人脸子进程：{e}", gr.skip()
            return
        yield (f"⏳ 人脸子进程已启动（日志 logs/webui/{log_path.name}），"
               "页面其他功能不受影响…"), gr.skip()
        t0 = time.time()
        while proc.poll() is None:
            waited = int(time.time() - t0)
            if waited > timeout_s:
                _kill_job(proc)
                yield f"❌ 超过 {timeout_s}s 已终止；日志 logs/webui/{log_path.name}", gr.skip()
                return
            yield (f"⏳ 生成中 {waited}s…（faces.device=cpu 时 1-3 分钟；日志 "
                   f"logs/webui/{log_path.name}）"), gr.skip()
            time.sleep(3)
        if proc.returncode != 0:
            yield f"❌ 子进程失败(exit {proc.returncode})；日志 logs/webui/{log_path.name}", gr.skip()
            return
        try:
            cands = _load_face_cands()
        except Exception as e:
            yield f"✅ 生成完成，但读取候选清单失败：{e}", gr.skip()
            return
        yield (f"✅ 生成完成。共 {len(cands)} 个人物候选。"
               "每格填真名（路人留空），然后保存。"), cands

    fgen_btn.click(fc_generate, None, [fc_status, fcands])


def _file_url(p):
    return "gradio_api/file=" + urllib.parse.quote(str(Path(p).resolve()))


# ── 组装 ──────────────────────────────────────────────────────────

def build_app():
    with gr.Blocks(title="电影解说工作台") as app:
        gr.Markdown("# 🎬 电影解说工作台")
        with gr.Tabs():
            with gr.TabItem("▶ 运行 / 调试"):
                _build_run_tab()
            with gr.TabItem("📊 集状态"):
                _build_status_tab()
            with gr.TabItem("📖 剧情背景"):
                _build_background_tab()
            with gr.TabItem("⚙️ 设置"):
                _build_settings_tab()
            with gr.TabItem("📜 历史日志"):
                _build_logs_tab()
            with gr.TabItem("🖼️ 封面图"):
                _build_cover_tab()
            with gr.TabItem("🙂 人脸标注"):
                # 注意:不给 tab 挂 select 回调——切页导航必须零成本,重载走显式按钮
                _build_faces_tab()
    return app


def main():
    (REPO_ROOT / "input").mkdir(exist_ok=True)
    (REPO_ROOT / "output").mkdir(exist_ok=True)
    app = build_app()
    app.queue(default_concurrency_limit=4).launch(
        server_name="0.0.0.0", server_port=6006, share=False,
        theme=gr.themes.Soft(), css="footer{display:none!important}",
        allowed_paths=[str((REPO_ROOT / "output").resolve()),
                       str((REPO_ROOT / "input").resolve())])


if __name__ == "__main__":
    main()
