"""画面备注（scene notes）：每个节拍抽帧 → VLM 描述「谁在场/在做什么/场景/氛围」。

为什么存在：解说 LLM 只有 ASR 文本，对画面一无所知，容易张冠李戴、描述画面上
不存在的事。本步骤给每拍生成画面备注 JSON，解说 prompt 逐拍附上，让 LLM「照着
画面写」而不是脑补。

省钱设计：只看剧情分析选中的节拍（不通看全片）；帧缩到 max_edge；结果按节拍
指纹缓存（重跑解说零 VLM 成本）。backend=local 用本机 GPU 跑 Qwen2.5-VL（免费），
backend=api 走 OpenAI 兼容多模态端点（如 Qwen3.5 系列）。

角色参考图（可选）：data/cast/ 下放 `角色名.jpg` 的脸部截图，VLM 会用真名指认。
"""
import base64
import gc
import json
from pathlib import Path

from media import run_ffmpeg

NOTE_PROMPT = (
    "以上是电视剧同一片段按时间顺序的 {n_frames} 张截图{cast_hint}。\n"
    "只根据画面可见内容（不要编造、不要推测画外剧情），输出一个 JSON 对象：\n"
    '{{"on_screen": [{on_screen_rule}],\n'
    ' "action": "画面中正在发生什么，1-2 句",\n'
    ' "scene": "地点与时间(如:办公室,夜)",\n'
    ' "mood": "情绪氛围，2-4 个字"}}\n'
    "只输出 JSON，不要其他文字。"
)
# 不封顶会退化成同一句外貌描述重复十几次，撑爆 max_new_tokens 把后面的字段截没
ON_SCREEN_RULE = ('"出镜人物的外貌描述(如:短发中年男性/白衣年轻女性)；'
                  '同一人物只列一次，最多 4 项"')
ON_SCREEN_RULE_KNOWN = ('"画面里除上述已确认人物之外、其他出镜者的外貌描述；'
                        '同一人物只列一次，最多 3 项；没有别人就给空数组 []"')
CAST_HINT = "；再往前的 {n_cast} 张是主要角色参考照片（依次为：{names}），截图里认出他们时 on_screen 直接用其名字"


class SceneNotesGenerator:
    def __init__(self, config, logger=None):
        cfg = config.scene_notes
        self.config = config
        self.backend = str(cfg.get("backend", "local")).strip().lower()
        self.local_model = cfg.get("local_model", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.api_cfg = cfg.get("api", {}) or {}
        self.frames_per_beat = max(1, int(cfg.get("frames_per_beat", 3) or 3))
        self.max_edge = int(cfg.get("max_edge", 512) or 512)
        self.cast_dir = Path(cfg.get("cast_dir", "data/cast") or "data/cast")
        self.max_new_tokens = int(cfg.get("max_new_tokens", 160) or 160)
        # 视觉 token 直接决定 prefill 耗时：max_pixels/(28*28) 就是每图 token 数
        self.max_pixels = int(cfg.get("max_pixels", 401408) or 401408)  # 512*28*28
        self.batch_size = max(1, int(cfg.get("batch_size", 4) or 4))
        self.max_cast_refs = int(cfg.get("max_cast_refs", 6) or 6)
        self.extract_workers = max(1, int(cfg.get("extract_workers", 4) or 4))
        self.logger = logger

    # ── 抽帧 ─────────────────────────────────────────────────────
    def _extract_frames(self, video_path: str, start: float, end: float,
                        frames_dir: Path, beat_no: int) -> list:
        """节拍内均匀取 N 帧（快 seek，缩到 max_edge），返回 jpg 路径列表。"""
        dur = max(end - start, 1.0)
        paths = []
        for j in range(self.frames_per_beat):
            t = start + dur * (j + 0.5) / self.frames_per_beat
            out = frames_dir / f"beat{beat_no:03d}_{j + 1}.jpg"
            try:
                run_ffmpeg(["-ss", f"{t:.2f}", "-i", video_path, "-frames:v", "1",
                            "-vf", f"scale='min({self.max_edge},iw)':-2",
                            "-q:v", "4", str(out)], check=True)
            except Exception as e:  # 单帧失败(如时间点异常)不拖垮整拍
                if self.logger:
                    self.logger.warning(f"beat{beat_no} 抽帧 {t:.1f}s 失败: {e}")
                continue
            if out.exists() and out.stat().st_size > 0:
                paths.append(out)
        return paths

    def _load_cast(self) -> list:
        """[(角色名, 图路径)]，文件名即角色名。"""
        if not self.cast_dir.is_dir():
            return []
        refs = []
        for p in sorted(self.cast_dir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.stat().st_size > 0:
                refs.append((p.stem, p))
        return refs

    def _prompt(self, n_frames: int, cast: list, known: list = None) -> str:
        hint = (CAST_HINT.format(n_cast=len(cast), names="、".join(n for n, _ in cast))
                if cast else "")
        p = NOTE_PROMPT.format(
            n_frames=n_frames, cast_hint=hint,
            on_screen_rule=ON_SCREEN_RULE_KNOWN if known else ON_SCREEN_RULE)
        if known:
            # 人脸库确定性识别的结果，比 VLM 隔空对脸可靠——作为已知事实注入。
            # 这些人已经写进 identified 字段了，不必让 VLM 再复述一遍浪费 token
            p = (f"已通过人脸识别确认画面中出现：{'、'.join(known)}（这是确定的，"
                 f"不用重复列出，也不要改口）。描述动作时直接用这些名字。\n" + p)
        return p

    def _content(self, frames: list, cast: list, known: list) -> list:
        """一拍的多模态 content：cast 参考图 + 本拍截图 + 指令。"""
        return ([{"type": "image", "image": f"file://{p}"} for _n, p in cast]
                + [{"type": "image", "image": f"file://{p}"} for p in frames]
                + [{"type": "text", "text": self._prompt(len(frames), cast, known)}])

    # ── 本地 backend（Qwen-VL，用完即释放）───────────────────────
    def _run_local(self, beats: list, cast: list) -> list:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from qwen_vl_utils import process_vision_info

        if self.logger:
            self.logger.info(f"Loading local VLM: {self.local_model} "
                             f"(batch={self.batch_size}, max_pixels={self.max_pixels})")
        model = AutoModelForImageTextToText.from_pretrained(
            self.local_model, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa")
        processor = AutoProcessor.from_pretrained(
            self.local_model, min_pixels=64 * 28 * 28, max_pixels=self.max_pixels)
        processor.tokenizer.padding_side = "left"   # 批量生成必须左填充

        notes = []
        try:
            # 逐拍串行时 GPU 大半时间在等单条序列，批量后 prefill/decode 都能打满
            for i in range(0, len(beats), self.batch_size):
                chunk = beats[i:i + self.batch_size]
                texts, images = [], []
                for _bi, frames, known in chunk:
                    messages = [{"role": "user",
                                 "content": self._content(frames, cast, known)}]
                    texts.append(processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True))
                    imgs, _videos = process_vision_info(messages)
                    images.extend(imgs or [])
                inputs = processor(text=texts, images=images, padding=True,
                                   return_tensors="pt").to(model.device)
                with torch.inference_mode():
                    out = model.generate(**inputs, max_new_tokens=self.max_new_tokens,
                                         do_sample=False)
                replies = processor.batch_decode(
                    out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                for (bi, _f, _k), reply in zip(chunk, replies):
                    notes.append((bi, self._parse_note(reply)))
                if self.logger:
                    self.logger.info(f"VLM {min(i + len(chunk), len(beats))}/{len(beats)} beats")
        finally:
            del model, processor
            gc.collect()
            torch.cuda.empty_cache()
        return notes

    # ── API backend（OpenAI 兼容多模态端点）──────────────────────
    def _run_api(self, beats: list, cast: list) -> list:
        from llm import make_httpx_client

        base_url = str(self.api_cfg.get("base_url", "")).rstrip("/")
        api_key = self.api_cfg.get("api_key", "")
        model_name = self.api_cfg.get("model_name", "")
        if not (base_url and model_name):
            raise RuntimeError("scene_notes.api 需要 base_url + model_name")

        def img_part(p: Path) -> dict:
            b64 = base64.b64encode(p.read_bytes()).decode()
            return {"type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

        notes = []
        with make_httpx_client(timeout=180) as client:
            for bi, frames, known in beats:
                content = [img_part(p) for _n, p in cast]
                content += [img_part(p) for p in frames]
                content.append({"type": "text",
                                "text": self._prompt(len(frames), cast, known)})
                resp = client.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": model_name, "temperature": 0,
                          "messages": [{"role": "user", "content": content}]})
                resp.raise_for_status()
                reply = resp.json()["choices"][0]["message"]["content"] or ""
                notes.append((bi, self._parse_note(reply)))
        return notes

    @staticmethod
    def _parse_note(reply: str) -> dict:
        from json_repair import repair_json
        try:
            obj = json.loads(repair_json(reply))
        except (ValueError, TypeError):
            obj = {}
        if not isinstance(obj, dict):
            obj = {}
        # 小模型爱把同一句外貌描述重复多遍，去重并封顶，免得下游 prompt 被灌满噪声
        seen, on_screen = set(), []
        raw = obj.get("on_screen") or []
        for x in (raw if isinstance(raw, list) else [raw]):
            s = str(x).strip()
            if s and s not in seen:
                seen.add(s)
                on_screen.append(s)
        return {"on_screen": on_screen[:4],
                "action": str(obj.get("action") or "").strip(),
                "scene": str(obj.get("scene") or "").strip(),
                "mood": str(obj.get("mood") or "").strip()}

    # ── 主流程 ───────────────────────────────────────────────────
    def generate(self, video_path: str, key_segments_path: str, output_path: str) -> str:
        with open(key_segments_path, encoding="utf-8") as f:
            segments = json.load(f).get("segments", [])
        if not segments:
            raise RuntimeError("key_segments 为空，先跑剧情分析")

        frames_dir = Path(output_path).parent / "scene_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # 人脸库已建时不再发 cast 参考图：认人由 InsightFace 确定性完成，名字以文字
        # 事实注入即可。cast 图会被拼进每一拍的请求，18 张就是每拍多上万视觉 token，
        # 是本步骤最大的一笔开销，而它要解决的问题人脸库已经解决了。
        from faces import get_engine
        try:
            engine = get_engine(self.config)
            has_gallery = engine.load_gallery()
        except Exception as e:
            engine, has_gallery = None, False
            if self.logger:
                self.logger.warning(f"人脸库不可用，回退 cast 参考图: {e}")
        cast = [] if has_gallery else self._load_cast()[:self.max_cast_refs]
        if self.logger:
            self.logger.info(
                f"Scene notes: {len(segments)} beats × {self.frames_per_beat} 帧, "
                f"backend={self.backend}, "
                + ("人脸库认人（不发 cast 参考图）" if has_gallery
                   else f"角色参考 {len(cast)} 张"))

        # 抽帧走线程池：每帧一个 ffmpeg 进程，30 拍 ×3 帧串行纯属干等 IO
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.extract_workers) as pool:
            futures = [pool.submit(self._extract_frames, video_path,
                                   float(seg.get("start_time", 0)),
                                   float(seg.get("end_time", seg.get("start_time", 0))),
                                   frames_dir, i + 1)
                       for i, seg in enumerate(segments)]
            frames_by_idx = [f.result() for f in futures]

        # 确定性认人：复用同一个 FaceEngine，否则每拍都要重新加载 InsightFace 会话
        beats, n_known = [], 0
        for i, frames in enumerate(frames_by_idx):
            if not frames:
                continue
            known = []
            if engine is not None and has_gallery:
                for p in frames:
                    for n in engine.identify_frame(p):
                        if n not in known:
                            known.append(n)
            n_known += bool(known)
            beats.append((i, frames, known))
        if self.logger and has_gallery:
            self.logger.info(f"人脸识别: {n_known}/{len(beats)} 拍确认了在场人物")

        if not beats:                       # 全拍抽帧失败时别白加载几 GB 的 VLM
            raise RuntimeError("所有节拍都没抽到帧，检查视频路径与节拍时间")
        raw = self._run_api(beats, cast) if self.backend == "api" \
            else self._run_local(beats, cast)

        known_by_idx = {bi: known for bi, _f, known in beats}
        notes = []
        by_idx = dict(raw)
        for i, seg in enumerate(segments):
            note = by_idx.get(i, {})
            if known_by_idx.get(i):
                known = known_by_idx[i]
                note["identified"] = known
                # 已确认的人不留在 on_screen 里重复一遍(小模型总爱复述),
                # 下游只需「人脸确认的是谁」+「画面里还有谁没认出来」两份信息
                note["on_screen"] = [x for x in note.get("on_screen", [])
                                     if not any(k in x for k in known)]
            notes.append({"beat": i + 1,
                          "start_time": seg.get("start_time"),
                          "end_time": seg.get("end_time"),
                          **note})
        payload = {"video": Path(video_path).stem,
                   "backend": self.backend,
                   "model": (self.api_cfg.get("model_name") if self.backend == "api"
                             else self.local_model),
                   "notes": notes}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        if self.logger:
            ok = sum(1 for n in notes if n.get("action"))
            self.logger.info(f"Scene notes done: {ok}/{len(notes)} beats → {output_path}")
        return output_path


def generate_scene_notes(video_path: str, key_segments_path: str, config,
                         output_path: str) -> str:
    import logging
    gen = SceneNotesGenerator(config, logging.getLogger("scene_notes"))
    return gen.generate(video_path, key_segments_path, output_path)
