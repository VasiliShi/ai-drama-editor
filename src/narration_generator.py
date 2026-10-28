import json
import os
import shutil
import uuid
import base64
from pathlib import Path

from episode_utils import extract_episode_background
from llm import LLMClient, make_httpx_client
from media import get_duration
from prompts import NARRATION_EXPLAIN


STYLE_DESC = {
    "casual": "口语自然，像朋友聊天一样聊剧情",
    "dramatic": "富有感染力，语调起伏，像说书人",
    "humorous": "轻松幽默，穿插调侃和吐槽",
}


def clamp_beat_end(nat_end: float, desired_end: float, remaining_extend):
    """picture_led 时长硬护栏：节拍想从 nat_end 延长到 desired_end 时，
    只允许消耗 remaining_extend（全片剩余可延长秒数）以内的量。

    remaining_extend=None 表示不设限（非 picture_led 模式）。
    返回 (end, remaining_extend)。护栏生效时解说音频可能略长于画面、
    尾部叠到下一拍，但保证成片总时长不破 max_minutes 上限。
    """
    if remaining_extend is None:
        return desired_end, None
    extension = max(0.0, desired_end - nat_end)
    allowed = min(extension, max(0.0, remaining_extend))
    return nat_end + allowed, remaining_extend - allowed


class NarrationGenerator:
    """电影解说配音：一次性生成贯穿全集、连贯的解说词（起承转合+衔接+上集回顾+结尾钩子），
    逐节拍 TTS，并用解说时长反推每个画面的长度（画面服从文案）。"""

    def __init__(self, base_url: str, api_key: str, model_name: str,
                 volcano_config: dict = None,
                 gen_config: dict = None,
                 continuity_config: dict = None,
                 explain_config: dict = None,
                 series_background: str = "",
                 engine: str = "volcano",
                 edge_config: dict = None,
                 tts_concurrency: int = 4):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.engine = str(engine or "volcano").strip().lower()
        self.volcano_config = volcano_config or {}
        self.edge_config = edge_config or {}
        self.gen_config = gen_config or {}
        self.continuity_config = continuity_config or {}
        self.explain_config = explain_config or {}
        self.series_background = series_background or ""
        self.tts_concurrency = max(1, int(tts_concurrency or 4))
        self.logger = None
        self.llm = LLMClient(base_url, api_key, model_name)
        self._client = None  # 火山 TTS 复用的直连 httpx client(httpx 线程安全,可并发)

    def set_logger(self, logger):
        self.logger = logger
        self.llm.logger = logger

    @staticmethod
    def _safe_float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _get_client(self):
        if self._client is None:
            self._client = make_httpx_client(300.0)
        return self._client

    # ───────────────────────── 集数与连续性 ─────────────────────────

    def _get_episode_number(self, filepath: str, existing_episodes: list) -> int:
        from episode_utils import parse_episode_number
        ep = parse_episode_number(Path(filepath).stem, fallback=0)
        if ep > 0:
            return ep
        if existing_episodes:
            return max(e.get("episode_num", 0) for e in existing_episodes) + 1
        return 1

    def _build_series_context(self, episodes: list, current_episode_num: int,
                              current_filename: str = "") -> str:
        """加厚版「上集回顾」：
        - 更早的集：用一句话概要（来自连续性记录）维持脉络；
        - 紧邻的上一集：用背景文档里的**完整分集剧情**（权威、详细），再附上次解说实际收尾的悬念。
        """
        from episode_utils import extract_episode_plot

        previous = [
            e for e in episodes
            if e.get("filename", "") != current_filename
            and e.get("episode_num", 0) < current_episode_num
        ]
        previous.sort(key=lambda e: e.get("episode_num", 0))

        prev_num = current_episode_num - 1
        prev_plot = extract_episode_plot(self.series_background, prev_num) if prev_num >= 1 else ""

        if not previous and not prev_plot:
            return "本剧第一集，无需回顾。请直接引入本集内容。"

        parts = []
        # 更早的集：一句话脉络
        for ep in previous:
            if ep.get("episode_num") == prev_num:
                continue
            summary = ep.get("plot_summary", "")
            if summary:
                parts.append(f"第{ep.get('episode_num', '?')}集：{summary}")
        # 上一集：完整剧情（加厚的核心）
        if prev_plot:
            parts.append(f"【上一集（第{prev_num}集）完整剧情】\n{prev_plot}")
        elif previous:
            last = previous[-1]
            parts.append(f"第{last.get('episode_num', '?')}集：{last.get('plot_summary', '')}")
        # 上一集解说实际收尾的悬念（衔接用）
        for ep in previous:
            if ep.get("episode_num") == prev_num and ep.get("ending_hook"):
                parts.append(f"上一集解说收尾的悬念：{ep['ending_hook']}")
        return "\n".join(parts)

    def _save_continuity(self, segments, episodes, episode_num, video_description,
                         key_segments_path, series_name, continuity_path):
        if not continuity_path:
            return
        first_hook = ""
        for s in reversed(segments):
            n = s.get("narration")
            if n and n.get("text"):
                first_hook = n["text"][-30:]
                break
        new_entry = {
            "filename": Path(key_segments_path).name,
            "episode_num": episode_num,
            "plot_summary": video_description,
            "key_characters": [],
            "ending_hook": first_hook,
        }
        existing = [e for e in episodes if e.get("filename") != Path(key_segments_path).name]
        existing.append(new_entry)
        with open(continuity_path, "w", encoding="utf-8") as f:
            json.dump({"series_name": series_name, "episodes": existing}, f, ensure_ascii=False, indent=2)
        if self.logger:
            self.logger.info(f"Continuity saved to: {continuity_path}")

    # ───────────────────────── LLM / TTS ─────────────────────────

    def _call_llm(self, prompt: str) -> str:
        # 重试耗尽/内容过滤会抛异常，让 narration 步骤显式失败。
        # （历史版本静默返回空串，整集解说会悄悄退化成 summary 复读——质量事故隐患。）
        return self.llm.chat(prompt)

    def _get_audio_duration(self, audio_path: str) -> float:
        return get_duration(audio_path)

    def _generate_tts(self, text: str, output_path: str) -> bool:
        """按 engine 分发到具体 TTS 后端。"""
        if self.engine == "edge":
            return self._generate_tts_edge(text, output_path)
        return self._generate_tts_volcano(text, output_path)

    _PROXY_KEYS = ('http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY',
                   'all_proxy', 'ALL_PROXY')

    @classmethod
    def _strip_proxy(cls) -> dict:
        """摘掉代理环境变量（edge-tts 的 aiohttp 走代理连不通 Bing）。返回原值供恢复。
        注意：os.environ 是进程级共享，摘/装必须在并发批次外层做一次，不能在线程里做。"""
        old = {}
        for key in cls._PROXY_KEYS:
            if key in os.environ:
                old[key] = os.environ.pop(key)
        return old

    @staticmethod
    def _restore_proxy(old: dict):
        os.environ.update(old or {})

    def _edge_synth(self, text: str, output_path: str) -> bool:
        """微软 Edge TTS 单段合成（不碰环境变量，线程安全，可并发）。
        输出为 mp3，下游 ffmpeg 按内容识别不受扩展名影响。"""
        import asyncio
        import time
        cfg = self.edge_config
        voice = str(cfg.get("voice", "zh-CN-YunyangNeural"))
        rate = str(cfg.get("rate", "+0%"))
        pitch = str(cfg.get("pitch", "+0Hz"))
        volume = str(cfg.get("volume", "+0%"))
        # 单次合成超时(秒)：微软端点偶发 websocket 挂住又不报错，必须有超时兜底，
        # 否则会无限等待（重试逻辑也永远触发不到）。一段解说最长十几秒音频，60s 足够宽松。
        per_try_timeout = float(cfg.get("timeout", 60))

        async def _run():
            import edge_tts
            communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume)
            await asyncio.wait_for(communicate.save(output_path), timeout=per_try_timeout)

        # 微软免费端点会偶发 NoAudioReceived / 空文件，重试几次基本都能成功
        last_err = None
        for attempt in range(5):
            try:
                Path(output_path).unlink(missing_ok=True)
                asyncio.run(_run())   # 每线程各自建事件循环，互不干扰
                if Path(output_path).exists() and Path(output_path).stat().st_size > 1000:
                    return True
                last_err = "empty audio"
            except asyncio.TimeoutError:
                last_err = f"timeout >{per_try_timeout:.0f}s"
            except Exception as e:
                last_err = e
            if self.logger:
                self.logger.warning(f"Edge TTS attempt {attempt+1}/5 failed ({last_err})")
            if attempt < 4:
                time.sleep(1.0 + attempt)
        if self.logger:
            self.logger.warning(f"Edge TTS failed after 5 tries: {last_err}")
        return False

    def _generate_tts_edge(self, text: str, output_path: str) -> bool:
        """单段入口（兼容保留）：摘代理→合成→装回。批量请走 _tts_batch。"""
        old_env = self._strip_proxy()
        try:
            return self._edge_synth(text, output_path)
        finally:
            self._restore_proxy(old_env)

    def _tts_batch(self, items: list) -> dict:
        """并发合成 {key: ok}。items: [(key, text, out_path)]。
        网络 IO 是解说步骤的最大耗时（串行一集干等 2-3 分钟），并发直接砍掉大头；
        edge 的代理摘装提级到批次外层做一次（进程级 env 不能在线程里各摘各的）。"""
        if not items:
            return {}
        from concurrent.futures import ThreadPoolExecutor, as_completed
        synth = self._edge_synth if self.engine == "edge" else self._generate_tts_volcano
        old_env = self._strip_proxy() if self.engine == "edge" else {}
        results = {}
        try:
            with ThreadPoolExecutor(max_workers=self.tts_concurrency) as ex:
                futs = {ex.submit(synth, text, path): key for key, text, path in items}
                for fut in as_completed(futs):
                    try:
                        results[futs[fut]] = bool(fut.result())
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"TTS beat {futs[fut]} crashed: {e}")
                        results[futs[fut]] = False
        finally:
            self._restore_proxy(old_env)
        return results

    def _generate_tts_volcano(self, text: str, output_path: str) -> bool:
        """火山引擎大模型语音合成（HTTP 单向流式，字符版）。
        响应为 NDJSON：每行一个 JSON，code=0 为音频块、20000000 为结束，
        将各块 data 字段 base64 解码后拼接即整段音频。"""
        cfg = self.volcano_config
        endpoint = cfg.get("endpoint", "https://openspeech.bytedance.com/api/v3/tts/unidirectional")
        headers = {
            "X-Api-Resource-Id": str(cfg.get("resource_id", "seed-tts-2.0")),
            "X-Api-Request-Id": str(uuid.uuid4()),
            "Content-Type": "application/json",
        }
        # 鉴权可切换：api_key=新版单密钥(X-Api-Key)；app_id=旧版(AppID+AccessToken，当前账号在用)
        auth_mode = str(cfg.get("auth_mode", "app_id")).strip().lower()
        if auth_mode == "api_key":
            headers["X-Api-Key"] = str(cfg.get("api_key", ""))
        else:
            headers["X-Api-App-Id"] = str(cfg.get("app_id", ""))
            headers["X-Api-Access-Key"] = str(cfg.get("access_token", ""))
        audio_params = {"format": "mp3", "sample_rate": int(cfg.get("sample_rate", 24000))}
        if cfg.get("speech_rate") is not None:
            audio_params["speech_rate"] = int(cfg.get("speech_rate"))
        if cfg.get("loudness_rate") is not None:
            audio_params["loudness_rate"] = int(cfg.get("loudness_rate"))
        body = {
            "user": {"uid": "video_editor"},
            "req_params": {"text": text, "speaker": str(cfg.get("speaker", "")), "audio_params": audio_params},
        }
        try:
            resp = self._get_client().post(endpoint, headers=headers, json=body)
            resp.raise_for_status()
            audio = b""
            last_err = None
            for line in resp.text.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("code") not in (0, 20000000):
                    last_err = obj.get("message") or obj.get("code")
                if obj.get("data"):
                    audio += base64.b64decode(obj["data"])
            if not audio:
                if self.logger:
                    self.logger.warning(f"Volcano TTS returned no audio: {last_err}")
                return False
            with open(output_path, "wb") as f:
                f.write(audio)
            return Path(output_path).exists() and Path(output_path).stat().st_size > 1000
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Volcano TTS failed: {e}")
            return False

    def _format_transcript(self, all_segments: list, seg_start: float, seg_end: float) -> str:
        lines = []
        for ts in all_segments:
            if ts["end"] > seg_start and ts["start"] < seg_end:
                start_rel = max(ts["start"] - seg_start, 0)
                end_rel = min(ts["end"] - seg_start, seg_end - seg_start)
                spk = ts.get("speaker")
                tag = f"{spk}: " if spk else ""
                lines.append(f"[{start_rel:.0f}s-{end_rel:.0f}s] {tag}{ts['text']}")
        return "\n".join(lines) if lines else "(无对白)"

    @staticmethod
    def _load_scene_notes(key_segments_path: str) -> dict:
        from episode_utils import load_scene_notes
        return load_scene_notes(key_segments_path)

    def _self_check(self, segments: list, narration_map: dict,
                    all_transcript_segs: list, visual_notes: dict) -> dict:
        """解说一致性自检：把每拍解说+对白+画面备注交回 LLM 校对，违规拍改写。
        失败不致命——校对挂了就用原稿。"""
        lines = []
        for i, seg in enumerate(segments):
            text = (narration_map.get(i + 1) or "").strip()
            if not text:
                continue
            dialog = self._format_transcript(
                all_transcript_segs, seg["start_time"], seg["end_time"])[:300]
            note = visual_notes.get(i, "")
            note_line = f"\n  画面备注：{note}" if note else ""
            lines.append(f"[节拍{i+1}] 解说：{text}{note_line}\n  对白：{dialog}")
        if not lines:
            return narration_map
        prompt = (
            "你是电视剧解说的校对编辑。下面每个节拍给出【解说词】和依据材料（原片对白、"
            "画面备注=视觉模型看画面的记录）。请逐拍检查解说词是否有以下问题：\n"
            "(a) 张冠李戴：人物动作/台词归因与对白或画面备注矛盾；\n"
            "(b) 无中生有：描述了对白和画面备注都不支持的具体画面动作；\n"
            "(c) 剧透：提前说出只有后面节拍才发生的反转/结局。\n"
            "只改写有问题的节拍：保持原字数±20%、语气风格一致、仍要承接上下文；"
            "没问题的不要动。\n\n" + "\n\n".join(lines) +
            '\n\n输出 JSON（没有问题就输出 {"fixes": []}）：\n'
            '{"fixes": [{"i": 节拍号, "narration": "改写后的解说词"}]}')
        try:
            result = self.llm.chat_json(prompt)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Self-check skipped (LLM failed): {e}")
            return narration_map
        fixes = (result or {}).get("fixes") if isinstance(result, dict) else None
        applied = 0
        for f in fixes or []:
            try:
                idx = int(f.get("i"))
                new_text = str(f.get("narration", "")).strip()
            except (TypeError, ValueError, AttributeError):
                continue
            old = narration_map.get(idx, "")
            # 长度护栏：改写过短/过长视为异常，弃用
            if old and new_text and 0.5 * len(old) <= len(new_text) <= 2.0 * len(old):
                narration_map[idx] = new_text
                applied += 1
        if self.logger:
            self.logger.info(f"Self-check: {applied} beats rewritten")
        return narration_map

    # ───────────────────────── 主流程 ─────────────────────────

    def generate(self, transcript_path: str, key_segments_path: str, output_dir: str) -> str:
        from json_repair import repair_json

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
        with open(key_segments_path, "r", encoding="utf-8") as f:
            key_data = json.load(f)

        all_transcript_segs = transcript_data.get("segments", [])
        segments = key_data.get("segments", [])
        video_description = key_data.get("video_description", "剧情解说")
        if not segments:
            if self.logger:
                self.logger.warning("No segments to generate narration for")
            return key_segments_path

        # 连续性上下文（上集回顾）
        series_name = self.continuity_config.get("series_name", "")
        continuity_enabled = self.continuity_config.get("enabled", True) and bool(series_name)
        episodes = []
        continuity_path = None
        if continuity_enabled:
            data_dir = Path(output_dir).parent / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            continuity_path = data_dir / f"{series_name}_narration_context.json"
            if continuity_path.exists():
                try:
                    episodes = json.load(open(continuity_path, encoding="utf-8")).get("episodes", [])
                except json.JSONDecodeError:
                    episodes = []

        episode_num = self._get_episode_number(key_segments_path, episodes)
        series_context = self._build_series_context(
            episodes, episode_num, Path(key_segments_path).name) if continuity_enabled else ""
        episode_background = extract_episode_background(self.series_background, episode_num)

        # 字数预算
        mode = str(self.explain_config.get("mode", "narration_led")).strip().lower()
        target_minutes = self._safe_float(self.explain_config.get("target_minutes", 7), 7)
        cps = self._safe_float(self.explain_config.get("chars_per_second", 4.5), 4.5)
        clip_pad = self._safe_float(self.explain_config.get("clip_pad", 0.4), 0.4)
        total_budget = int(target_minutes * 60 * cps)
        per_beat = max(20, total_budget // max(1, len(segments)))

        central_question = key_data.get("central_question", "")
        style = self.gen_config.get("style", "dramatic")
        style_desc = STYLE_DESC.get(style, STYLE_DESC["dramatic"])

        # picture_led：画面是分析选定的自然场景，解说要能在该时长内读完 → 按每段时长给字数预算
        seg_budgets = []
        for seg in segments:
            nat = max(2.0, self._safe_float(seg.get("end_time"), 0) - self._safe_float(seg.get("start_time"), 0))
            seg_budgets.append(max(12, int(nat * cps)))

        # 画面备注（可选，scene_notes 步骤产物）：让 LLM「照着画面写」而非脑补
        visual_notes = self._load_scene_notes(key_segments_path)

        # 组装所有节拍（含原片对白），一次性喂给 LLM 以保证整集解说连贯
        beat_lines = []
        for i, seg in enumerate(segments):
            dialog = self._format_transcript(all_transcript_segs, seg["start_time"], seg["end_time"])
            if mode == "picture_led":
                head = f"[节拍{i+1}]（画面 {seg_budgets[i]//int(cps) if cps else 0}秒，解说≤{seg_budgets[i]}字）{seg.get('summary', '')}"
            else:
                head = f"[节拍{i+1}] {seg.get('summary', '')}"
            note = visual_notes.get(i)
            visual_line = f"\n  画面备注：{note}" if note else ""
            beat_lines.append(f"{head}{visual_line}\n  原片对白：{dialog}")
        beats_block = "\n\n".join(beat_lines)

        has_speaker = any(ts.get("speaker") for ts in all_transcript_segs)
        speaker_hint = (
            "\n> 对白中每行前的 SPEAKER_xx 是按声音区分的说话人标签：同一标签=同一个人在说话"
            "（不是真名）。请据此判断每句台词是谁说的、对话是谁对谁，正确归因人物动作与关系，"
            "**不要把某个角色的台词或行为安到别人头上**。\n"
            if has_speaker else ""
        )
        # 有画面备注时，附使用规则（借道 speaker_hint 占位符，不动模板契约）
        if visual_notes:
            speaker_hint += (
                "\n> 部分节拍附有「画面备注」，来自视觉模型直接观看该拍画面：描述画面上谁在做什么时"
                "**以备注为准**，不要编造备注和对白之外的画面细节；人物归因同时参考备注(谁在场)与"
                "对白(谁在说)，两者交叉印证。\n"
            )

        recap_rule = (
            "第1段是「冷开场」：开头一两句直接甩出本集最炸的悬念/冲突（围绕「核心问题」），"
            "一句话点破主角此刻的处境，先把观众勾住；如需衔接上集，只用半句话带过，"
            "**绝不要用「上集说到…」式的长篇回顾开场**；"
            if series_context and "第一集" not in series_context else
            "第1段是「冷开场」：开头一两句直接甩出本集最炸的悬念/冲突勾住观众，不要先铺垫背景；"
        )

        if mode == "picture_led":
            char_budget_rule = ("每段解说尽量在该节拍画面时长内读完（参考每个节拍标注的「解说≤N字」），力求精炼；"
                                "需要桥接、补叙被跳过剧情的段落可以适当多写，**但最多写到该节拍字数预算的 1.5 倍，"
                                "这是硬上限**——画面虽会跟着解说延长，但全片总时长有上限，超预算的解说会被截断。")
            cut_style = str(self.explain_config.get("cut_style", "bridge")).strip().lower()
            if cut_style == "skeleton":
                bridge_rule = ("\n10. 画面保留了主线关键节点，请顺着把这一集的来龙去脉讲清楚、让没看过的观众也能看懂；"
                               "在高潮节拍着重渲染冲突与情绪，次要过场一句话带过即可。")
            elif cut_style == "pure":
                bridge_rule = ("\n10. 画面是跳着选的最炸名场面，**不必补全剧情、不必强行连贯**："
                               "重在用激情吐槽、造梗、情绪化的短句把爽感和情绪拉满，每段短促有力、有记忆点。")
            else:  # bridge
                bridge_rule = ("\n10. **桥接被跳过的剧情（让跳剪不违和的关键）**：本片画面只保留了高潮段落，平淡部分"
                               "已被剪掉，相邻两个节拍之间剧情常有跳跃。每当某节拍与上一节拍之间剧情有明显跳跃"
                               "（中间发生的事画面没演），就在这段解说开头先用一两句交代中间发生的关键经过"
                               "（依据背景与对白推断），再讲当前画面，让观众衔接得上、不出戏。")
        else:
            char_budget_rule = f"每段解说控制在 {per_beat} 字以内，全集总字数约 {total_budget} 字。"
            bridge_rule = ""
        bg_block = f"\n## 本集人物与剧情背景（人名以此为准）\n{episode_background}\n" if episode_background else ""
        # 分析步骤产出的整集梗概：桥接被剪掉的剧情靠它，不再瞎猜
        synopsis = str(key_data.get("episode_synopsis") or "").strip()
        if synopsis:
            bg_block += ("\n## 本集完整剧情梗概（内部资料：桥接被剪掉的剧情以此为准；"
                         "只可用于交代已发生的过渡剧情，禁止提前说出后面节拍才揭晓的反转）\n"
                         f"{synopsis}\n")

        prompt = NARRATION_EXPLAIN.substitute(
            video_description=video_description, central_question=central_question,
            series_name=(series_name or "本剧"), episode_num=episode_num,
            bg_block=bg_block, series_context=(series_context or "（无）"),
            speaker_hint=speaker_hint, beats_block=beats_block, style_desc=style_desc,
            recap_rule=recap_rule, char_budget_rule=char_budget_rule,
            bridge_rule=bridge_rule, n_segments=len(segments))

        if self.logger:
            self.logger.info(f"Explain narration: ep{episode_num}, {len(segments)} beats, "
                             f"budget ~{total_budget} chars (~{per_beat}/beat), target {target_minutes:.0f} min")

        raw = self._call_llm(prompt)
        narration_map = {}
        if raw:
            try:
                arr = json.loads(repair_json(raw))
                if isinstance(arr, list):
                    for item in arr:
                        if isinstance(item, dict) and "i" in item:
                            try:
                                narration_map[int(item["i"])] = str(item.get("narration", "")).strip()
                            except (TypeError, ValueError):
                                continue
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Explain script JSON parse failed: {e}")

        # 一致性自检（可关）：TTS 之前逐拍校对归因/无中生有/剧透，违规拍改写
        if narration_map and self.explain_config.get("self_check", True):
            narration_map = self._self_check(
                segments, narration_map, all_transcript_segs, visual_notes)

        nar_dir = Path(output_dir) / "narration"
        nar_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(key_segments_path).stem.replace("_key_segments", "")
        last_end = all_transcript_segs[-1]["end"] if all_transcript_segs else 0.0

        # picture_led 硬护栏：分析层已保证 Σ自然时长 ≤ max_minutes，这里限制
        # 解说导致的额外延长总量，确保最终 Σ(end-start) 不破上限（ep7/9/11 曾实测突破）。
        extend_budget = None
        if mode == "picture_led":
            max_minutes = self._safe_float(
                self.explain_config.get("max_minutes", round(target_minutes * 1.3, 1)),
                target_minutes * 1.3)
            natural_total = sum(
                max(0.0, self._safe_float(s.get("end_time"), 0) - self._safe_float(s.get("start_time"), 0))
                for s in segments)
            extend_budget = max(0.0, max_minutes * 60.0 - natural_total)
            if self.logger:
                self.logger.info(f"Duration guard: natural {natural_total/60:.1f} min, "
                                 f"extend budget {extend_budget:.0f}s (cap {max_minutes:.0f} min)")

        # 两阶段：先并发合成全部配音（网络 IO 瓶颈），再顺序算时长/延长（顺序状态）
        beat_texts = {}
        for i, seg in enumerate(segments):
            beat_texts[i] = (narration_map.get(i + 1) or seg.get("summary", "")).strip()
        tts_items = [(i, t, str(nar_dir / f"{base_name}_beat{i+1:03d}.wav"))
                     for i, t in beat_texts.items() if t]
        if self.logger:
            self.logger.info(f"TTS batch: {len(tts_items)} beats × concurrency {self.tts_concurrency} "
                             f"({self.engine})")
        tts_ok = self._tts_batch(tts_items)

        ok_count = 0
        for i, seg in enumerate(segments):
            text = beat_texts[i]
            if not text:
                seg["narration"] = None
                continue
            nar_file = nar_dir / f"{base_name}_beat{i+1:03d}.wav"
            if not tts_ok.get(i):
                if self.logger:
                    self.logger.warning(f"    TTS failed for beat {i+1}")
                seg["narration"] = None
                continue
            dur = self._get_audio_duration(str(nar_file))
            seg["narration"] = {"text": text, "audio": str(nar_file), "duration": dur}
            plot_end = self._safe_float(seg.get("end_time"), seg["start_time"])
            if mode == "picture_led":
                # 画面主导：画面=分析选定的自然场景时长，原样连续播放；解说叠在上面。
                # 仅当解说比场景还长时才延长，且总延长量受 extend_budget 硬护栏约束。
                nat_end = plot_end
                need_end = seg["start_time"] + dur + clip_pad
                desired = max(nat_end, need_end)
                new_end, extend_budget = clamp_beat_end(nat_end, desired, extend_budget)
                if new_end < desired - 0.05 and self.logger:
                    self.logger.warning(f"    Beat {i+1}: 延长被时长护栏截断 "
                                        f"({desired - nat_end:.1f}s → {new_end - nat_end:.1f}s)")
            else:
                # 画面服从文案：画面跟随解说长度，B1 在 [start, plot_end] 内采样铺满。
                new_end = seg["start_time"] + dur + clip_pad
            if last_end > 0:
                new_end = min(new_end, last_end)
            if new_end <= seg["start_time"]:
                new_end = seg["start_time"] + 3.0
            seg["end_time"] = round(new_end, 3)
            # picture_led: src_end=end_time → 剪辑器走连续单片段（不采样）；
            # narration_led: 保留 plot_end 作为 B1 采样窗口右界。
            seg["src_end"] = round(new_end if mode == "picture_led" else max(plot_end, new_end), 3)
            ok_count += 1

        key_data["segments"] = segments
        out_file = Path(output_dir) / f"{base_name}_key_segments.json"
        if out_file.exists():
            shutil.copyfile(out_file, f"{out_file}.bak")  # 原地改写前留底，方便回溯分析层原始节拍
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(key_data, f, ensure_ascii=False, indent=2)

        if self.logger:
            total = sum(s["end_time"] - s["start_time"] for s in segments)
            self.logger.info(f"Explain narration done: {ok_count}/{len(segments)} beats with TTS, "
                             f"final length ~{total/60:.1f} min")

        self._save_continuity(segments, episodes, episode_num, video_description,
                              key_segments_path, series_name, continuity_path)
        return str(out_file)


def generate_narration(transcript_path: str, key_segments_path: str, config, output_dir: str) -> str:
    nar_config = config.narration
    tts = nar_config.get("tts", {})
    ng = NarrationGenerator(
        base_url=config.llm.get("base_url"),
        api_key=config.llm.get("api_key"),
        model_name=config.llm.get("model_name"),
        volcano_config=tts.get("volcano", {}),
        gen_config=nar_config.get("generation", {}),
        continuity_config=nar_config.get("continuity", {}),
        explain_config=nar_config.get("explain", {}),
        series_background=config.video.get("series_background", ""),
        engine=tts.get("engine", "volcano"),
        edge_config=tts.get("edge", {}),
        tts_concurrency=tts.get("concurrency", 4),
    )
    ng.set_logger(config.get_logger("Narration"))
    return ng.generate(transcript_path, key_segments_path, output_dir)


if __name__ == "__main__":
    from config import load_config
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    transcript_path = sys.argv[2] if len(sys.argv) > 2 else "output/test_transcript.json"
    key_segments_path = sys.argv[3] if len(sys.argv) > 3 else "output/test_key_segments.json"
    config = load_config(config_path)
    generate_narration(transcript_path, key_segments_path, config, "output")
