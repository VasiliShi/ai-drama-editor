"""管线步骤注册表与执行器。

- STEPS 是唯一的步骤清单：CLI 的 force/skip 参数、webui 的步骤开关、
  运行日志的 [i/8 Label] 前缀都从这里派生。
- 缓存语义：
  * asr/analysis/narration/commentary 沿用「产物存在且非空」判断；
  * cutting/render 用「节拍指纹」判断（存 output/N/.pipeline_state.json），
    上游节拍变了会自动重切/重渲——不再需要手动预清理旧产物；
    老产物没有指纹时，cutting 按旧语义视为命中、render 视为过期重渲。
  * audiomix 只在本次 cutting 真正重切后执行（trimmed 是"已混音"状态的
    就地覆盖产物，重复混会叠两遍解说）；--force-audio-mix 会连带强制重切。
- 步骤日志固定整行输出（START / OK 时长 / CACHED / SKIPPED / DISABLED / FAILED），
  webui 靠它做实时状态解析，不要改回半行 print。
"""
import hashlib
import json
import re
import time
from pathlib import Path


# key, 日志标签, 中文名, 可 force, 可 skip
STEPS = [
    ("asr",       "ASR",        "语音识别", True,  True),
    ("naming",    "Naming",     "说话人命名", False, False),
    ("analysis",  "Analysis",   "剧情分析", True,  True),
    ("scene_notes", "SceneNotes", "画面备注", True, True),
    ("narration", "Narration",  "解说生成", True,  True),
    ("commentary", "Commentary", "影评评论", True,  True),
    ("cutting",   "Cutting",    "裁剪拼接", True,  True),
    ("audio_mix", "AudioMix",   "混音",     True,  True),
    ("render",    "Render",     "字幕渲染", True,  True),
]

SKIP_FLAGS = [f"skip_{k}" for k, _l, _z, _f, can_skip in STEPS if can_skip]


def add_step_arguments(parser):
    """把步骤的 --force-x / --skip-x 参数注册到 argparse（CLI 与 webui 共用步骤定义）。"""
    parser.add_argument("--force", action="store_true", help="强制重做所有步骤")
    for key, _label, zh, can_force, can_skip in STEPS:
        flag = key.replace("_", "-")
        if can_force:
            parser.add_argument(f"--force-{flag}", action="store_true",
                                dest=f"force_{key}", help=f"强制重做{zh}")
        if can_skip:
            parser.add_argument(f"--skip-{flag}", action="store_true",
                                dest=f"skip_{key}", help=f"跳过{zh}")


def timer(start=None):
    if start is None:
        return time.time()
    elapsed = time.time() - start
    if elapsed < 60:
        return f"{elapsed:.1f}s"
    return f"{elapsed:.0f}s ({elapsed/60:.1f}min)"


def _clean_publish_text(value, max_len=20):
    text = str(value or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", "", text)
    text = text.strip(" -_，。,.!！?？")
    return text[:max_len]


def select_publish_title(data, fallback=""):
    segments = data.get("segments", []) if isinstance(data, dict) else []
    candidates = [
        data.get("cover_title", "") if isinstance(data, dict) else "",
        data.get("hook_title", "") if isinstance(data, dict) else "",
        data.get("video_description", "") if isinstance(data, dict) else "",
        fallback,
        segments[0].get("summary", "") if segments else "",
        "精彩片段",
    ]
    for candidate in candidates:
        title = _clean_publish_text(candidate)
        if title:
            return title
    return "精彩片段"


def write_descriptions(key_segments_path, output_dir, video_desc=""):
    with open(key_segments_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    desc = select_publish_title(data, video_desc)
    desc_path = Path(output_dir) / "descriptions.txt"
    with open(desc_path, "w", encoding="utf-8") as f:
        f.write(desc + "\n")
    return str(desc_path)


# ── 指纹式缓存状态 ────────────────────────────────────────────────

def _state_path(output_dir) -> Path:
    return Path(output_dir) / ".pipeline_state.json"


def _load_state(output_dir) -> dict:
    p = _state_path(output_dir)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_state(output_dir, state: dict):
    _state_path(output_dir).write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def beats_fingerprint(key_data: dict, video_path: str) -> str:
    """裁剪输入指纹：只含影响画面切割的字段（节拍区间），评论/标题变化不触发重切。"""
    beats = [[round(float(s.get("start_time", 0)), 3),
              round(float(s.get("end_time", 0)), 3),
              round(float(s.get("src_end", s.get("end_time", 0))), 3)]
             for s in (key_data.get("segments") or [])]
    return _sha1(json.dumps({"video": Path(video_path).name, "beats": beats}))


def render_fingerprint(key_segments_path, trimmed_path, fmt: dict, config) -> str:
    """渲染输入指纹：key_segments 全文 + trimmed 文件签名 + 输出格式 + 相关展示配置。"""
    try:
        ks = Path(key_segments_path).read_bytes()
    except OSError:
        ks = b""
    try:
        st = Path(trimmed_path).stat()
        trimmed_sig = f"{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        trimmed_sig = "missing"
    display_cfg = {
        "subtitle": config.subtitle, "description": config.description,
        "author": config.author, "copyright": config.copyright_notice,
        "ending": config.ending, "commentary": config.commentary,
        "caption": config.narration.get("caption", {}),
        "transition": config.transition,
    }
    return _sha1(hashlib.sha1(ks).hexdigest() + trimmed_sig
                 + json.dumps(fmt, sort_keys=True)
                 + json.dumps(display_cfg, sort_keys=True, ensure_ascii=False, default=str))


# ── 执行器 ────────────────────────────────────────────────────────

class _StepRunner:
    def __init__(self, opts, total: int):
        self.opts = opts
        self.total = total

    def _flag(self, name: str) -> bool:
        return bool(getattr(self.opts, name, False))

    def line(self, idx: int, label: str, status: str):
        print(f"  [{idx}/{self.total} {label}] {status}", flush=True)

    def run(self, idx: int, key: str, label: str, run_fn,
            cached_fn=None, enabled: bool = True, optional: bool = False) -> bool:
        """返回该步骤本次是否真正执行了。"""
        if not enabled:
            self.line(idx, label, "DISABLED")
            return False
        if self._flag(f"skip_{key}"):
            self.line(idx, label, "SKIPPED")
            return False
        force = self._flag("force") or self._flag(f"force_{key}")
        if cached_fn is not None and not force and cached_fn():
            self.line(idx, label, "CACHED")
            return False
        self.line(idx, label, "START")
        t0 = time.time()
        try:
            run_fn()
            self.line(idx, label, f"OK {timer(t0)}")
            return True
        except Exception as e:
            if optional:
                self.line(idx, label, f"WARN 跳过 ({timer(t0)}): {e}")
                return False
            self.line(idx, label, f"FAILED ({timer(t0)})")
            print(f"  Error: {e}", flush=True)
            raise


def run_pipeline(video_path: str, config, output_base: str, opts) -> bool:
    """单个视频的完整管线。步骤定义见模块头注释。"""
    from asr import transcribe_video
    from analyzer import analyze_transcript
    from narration_generator import generate_narration
    from commentary_generator import generate_commentaries
    from audio_mixer import mix_audio
    from cutter import cut_video
    from subtitle_adder import add_subtitles

    video_name = Path(video_path).stem
    output_dir = Path(output_base) / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Processing: {video_name}", flush=True)

    runner = _StepRunner(opts, total=len(STEPS))
    state = _load_state(output_dir)

    transcript_path = output_dir / f"{video_name}_transcript.json"
    srt_path = output_dir / f"{video_name}_full.srt"
    key_segments_path = output_dir / f"{video_name}_key_segments.json"

    explain_mode = str(config.narration.get("explain", {}).get("mode", "")).strip().lower()
    is_remix = (explain_mode == "remix")
    nar_enabled = config.narration.get("enabled", False) and not is_remix

    # 1 ASR
    runner.run(1, "asr", "ASR",
               lambda: transcribe_video(video_path, config, str(output_dir)),
               cached_fn=lambda: transcript_path.exists() and srt_path.exists())

    # 2 Naming（可选：SPEAKER_xx → 真名；开启时每次都应用，失败仅告警）
    def do_naming():
        from speaker_naming import name_speakers
        name_speakers(str(transcript_path), config, video_name)

    runner.run(2, "naming", "Naming", do_naming,
               enabled=config.naming.get("enabled", False) and transcript_path.exists(),
               optional=True)

    # 3 Analysis
    def has_segments():
        if not key_segments_path.exists():
            return False
        with open(key_segments_path, encoding="utf-8") as f:
            return len(json.load(f).get("segments", [])) > 0

    runner.run(3, "analysis", "Analysis",
               lambda: analyze_transcript(str(transcript_path), config, str(output_dir)),
               cached_fn=has_segments)

    def load_key_data() -> dict:
        if key_segments_path.exists():
            with open(key_segments_path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    print(f"    {len(load_key_data().get('segments', []))} beats", flush=True)

    # 4 SceneNotes（可选：VLM 画面备注，解说据此「照着画面写」）
    scene_notes_path = output_dir / f"{video_name}_scene_notes.json"

    def sn_fingerprint() -> str:
        # 只认节拍起点（narration 会延长 end_time，不应触发重看画面）+ 模型与抽帧参数
        sn_cfg = config.scene_notes
        starts = [round(float(s.get("start_time", 0)), 3)
                  for s in load_key_data().get("segments", [])]
        return _sha1(json.dumps({
            "video": Path(video_path).name, "starts": starts,
            "backend": sn_cfg.get("backend"),
            "model": sn_cfg.get("local_model") if sn_cfg.get("backend") != "api"
                     else (sn_cfg.get("api", {}) or {}).get("model_name"),
            "frames": sn_cfg.get("frames_per_beat", 3)}))

    def do_scene_notes():
        from scene_notes import generate_scene_notes
        generate_scene_notes(video_path, str(key_segments_path), config,
                             str(scene_notes_path))
        state["scene_notes"] = {"fp": sn_fingerprint()}
        _save_state(output_dir, state)

    runner.run(4, "scene_notes", "SceneNotes", do_scene_notes,
               cached_fn=lambda: (scene_notes_path.exists()
                                  and state.get("scene_notes", {}).get("fp") == sn_fingerprint()),
               enabled=config.scene_notes.get("enabled", False) and has_segments(),
               optional=True)

    # 5 Narration（非 remix）
    def has_narration():
        for s in load_key_data().get("segments", []):
            n = s.get("narration")
            if n and n.get("audio") and Path(n["audio"]).exists():
                return True
        return False

    runner.run(5, "narration", "Narration",
               lambda: generate_narration(str(transcript_path), str(key_segments_path),
                                          config, str(output_dir)),
               cached_fn=has_narration, enabled=nar_enabled)

    # 6 Commentary（remix：影评/弹幕评论）
    def has_commentaries():
        return any(s.get("commentaries") for s in load_key_data().get("segments", []))

    runner.run(6, "commentary", "Commentary",
               lambda: generate_commentaries(str(transcript_path), str(key_segments_path),
                                             config, str(output_dir)),
               cached_fn=has_commentaries, enabled=is_remix)

    key_data = load_key_data()
    all_segments = key_data.get("segments", [])
    if not all_segments:
        # 「仅转写」等跳过分析的场景：没有节拍就没有可裁剪/渲染的内容，干净收尾
        print("    无节拍数据（分析被跳过或未产出），到此为止", flush=True)
        return True
    video_description = key_data.get("video_description", "")
    publish_title = select_publish_title(key_data) if key_data else ""

    output_formats = config.output_formats
    max_parts = config.max_parts
    all_outputs = []

    def render_one(video_in, seg_path, out_dir, out_base, fmt_dict, part_suffix="", use_cache=True):
        suffix = fmt_dict.get("suffix", "_portrait")
        vdesc = publish_title or video_description or ""
        safe_desc = re.sub(r'[^\w\-]', '_', vdesc).strip('_')[:20]
        desc_part = f"_{safe_desc}" if safe_desc else ""
        out_path = str(Path(out_dir) / f"{out_base}{desc_part}{suffix}{part_suffix}.mp4")

        fp = render_fingerprint(seg_path, video_in, fmt_dict, config)
        cache_key = f"render{part_suffix}:{suffix}"

        def cached():
            return (use_cache and Path(out_path).exists()
                    and state.get(cache_key, {}).get("fp") == fp
                    and state.get(cache_key, {}).get("out") == out_path)

        def do_render():
            add_subtitles(video_in, seg_path, config,
                          output_path=out_path,
                          aspect_ratio=fmt_dict.get("aspect_ratio", "9:16"),
                          target_resolution=fmt_dict.get("target_resolution", "1080x1920"))
            state[cache_key] = {"fp": fp, "out": out_path}
            _save_state(output_dir, state)

        runner.run(9, "render", f"Render {fmt_dict.get('aspect_ratio')}{part_suffix}",
                   do_render, cached_fn=cached)
        all_outputs.append(out_path)

    # --force-narration 必须连带重切混：新 TTS 音频要混进 trimmed，而混音只在
    # 裁剪实际执行时跑（防双混）；若节拍时间没变裁剪会 CACHED，新配音就丢了。
    if getattr(opts, "force_narration", False):
        opts.force_cutting = True
    # --force-audio-mix 必须连带重切：trimmed 是已混音的就地覆盖产物，
    # 在旧 trimmed 上再混一遍会叠双份解说。
    if getattr(opts, "force_audio_mix", False):
        opts.force_cutting = True

    def cut_and_mix(seg_path, out_dir, seg_list, part_suffix=""):
        """cutting + audio_mix + descriptions（part 模式与整片共用）。"""
        trimmed_out = Path(out_dir) / "trimmed.mp4"
        fp = beats_fingerprint({"segments": seg_list}, video_path)
        cache_key = f"cutting{part_suffix}"

        def cutting_cached():
            if not trimmed_out.exists():
                return False
            rec = state.get(cache_key)
            if rec is None:
                return True  # 老产物无指纹：按旧语义视为命中
            return rec.get("fp") == fp

        def do_cut():
            cut_video(video_path, seg_path, config, str(out_dir))
            state[cache_key] = {"fp": fp}
            _save_state(output_dir, state)

        cut_ran = runner.run(7, "cutting", f"Cutting{part_suffix}", do_cut,
                             cached_fn=cutting_cached)

        def do_mix():
            mix_audio(str(trimmed_out), seg_path, config, str(out_dir))

        runner.run(8, "audio_mix", f"AudioMix{part_suffix}", do_mix,
                   enabled=nar_enabled and cut_ran, optional=True)
        if nar_enabled and not cut_ran:
            runner.line(8, f"AudioMix{part_suffix}", "CACHED (随裁剪产物)")

        write_descriptions(seg_path, str(out_dir), publish_title)
        return str(trimmed_out)

    if max_parts > 1 and len(all_segments) > 1:
        total_dur = sum(s["end_time"] - s["start_time"] for s in all_segments)
        target = total_dur / max_parts
        parts, cur, cur_dur = [], [], 0.0
        for seg in all_segments:
            cur.append(seg)
            cur_dur += seg["end_time"] - seg["start_time"]
            if len(parts) < max_parts - 1 and cur_dur >= target * 0.8:
                parts.append(cur)
                cur, cur_dur = [], 0.0
        if cur:
            parts.append(cur)

        for p, part_segs in enumerate(parts, 1):
            part_dir = output_dir / f"part{p}"
            part_dir.mkdir(parents=True, exist_ok=True)
            part_key = part_dir / f"{video_name}_key_segments_part{p}.json"
            with open(part_key, "w", encoding="utf-8") as f:
                json.dump({
                    "hook_title": key_data.get("hook_title", ""),
                    "cover_title": key_data.get("cover_title", ""),
                    "video_description": video_description,
                    "segments": part_segs,
                }, f, ensure_ascii=False, indent=2)

            p_dur = sum(s["end_time"] - s["start_time"] for s in part_segs)
            print(f"\n  --- Part {p}/{len(parts)} ({len(part_segs)} beats, {p_dur:.0f}s) ---",
                  flush=True)
            trimmed = cut_and_mix(str(part_key), str(part_dir), part_segs, f"_part{p}")
            for fmt in output_formats:
                render_one(trimmed, str(part_key), part_dir, video_name, fmt, f"_part{p}")
    else:
        trimmed = cut_and_mix(str(key_segments_path), str(output_dir), all_segments)
        for fmt in output_formats:
            render_one(trimmed, str(key_segments_path), output_dir, video_name, fmt)

    print(f"\n  Outputs ({len(all_outputs)} files)")
    for p in all_outputs:
        print(f"    {p}")
    return True
