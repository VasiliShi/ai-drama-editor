import json
import re
from pathlib import Path

from episode_utils import parse_episode_number, extract_episode_background
from llm import LLMClient
from prompts import EXPLAIN_ANALYSIS


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class Analyzer:
    """电影解说分析器：把一整集字幕拆成覆盖全剧情、按时间排序的「解说分镜节拍」。
    画面服从解说——这里只挑节拍和锚定源画面区间，最终时长由后续解说决定。"""

    def __init__(self, base_url: str, api_key: str, model_name: str, strategy: dict,
                 skip_intro_seconds: float = 0, skip_outro_seconds: float = 0,
                 series_background: str = "", explain_config: dict = None):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.strategy = strategy or {}
        self.skip_intro_seconds = skip_intro_seconds
        self.skip_outro_seconds = skip_outro_seconds
        self.series_background = series_background or ""
        self.explain_config = explain_config or {}
        self.target_minutes = self._score_value(self.explain_config.get("target_minutes", 7), 7)
        self.max_beats = int(self._score_value(self.explain_config.get("max_beats", 30), 30))
        self.mode = str(self.explain_config.get("mode", "narration_led")).strip().lower()
        # picture_led 选材方案：bridge(高潮精选+解说桥接) | skeleton(主线骨架+高潮加重) | pure(纯高潮爽剪)
        self.cut_style = str(self.explain_config.get("cut_style", "bridge")).strip().lower()
        # remix(混剪评论)模式：最高钩子前置；hook_preview=true 时开头只插 3-6 秒短预告，完整片段留在原时间线
        self.hook_preview_enabled = bool(self.explain_config.get("hook_preview", True))
        preview_sec = float(self.explain_config.get("hook_preview_seconds", 5) or 5)
        self.hook_preview_seconds = min(6.0, max(3.0, preview_sec))
        # picture_led 成片时长=Σ节拍时长，必须卡范围（否则 LLM 选太多/太长，成片动辄十几分钟）。
        # 默认按 target 上下浮动，可在 config 用 min_minutes/max_minutes 显式指定。
        self.min_minutes = float(self.explain_config.get("min_minutes", round(self.target_minutes * 0.8, 1)))
        self.max_minutes = float(self.explain_config.get("max_minutes", round(self.target_minutes * 1.3, 1)))
        self.logger = None
        self.llm = LLMClient(base_url, api_key, model_name)

    def set_logger(self, logger):
        self.logger = logger
        self.llm.logger = logger

    @staticmethod
    def _score_value(value, default: float = 5.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _tension_from_score(self, score: float) -> str:
        if score >= 9:
            return "peak"
        if score >= 8:
            return "high"
        if score >= 6:
            return "medium"
        return "low"

    def _clean_title(self, value, fallback: str = "剧情解说", max_len: int = 20,
                     keep_punct: bool = False) -> str:
        text = str(value or "").replace("\n", " ").strip()
        text = re.sub(r"\s+", "", text)
        strip_chars = " -_，。,." if keep_punct else " -_，。,.!！?？"
        text = text.strip(strip_chars)
        if not text:
            text = fallback
        return text[:max_len]

    def _clean_metadata_text(self, value, fallback: str = "", max_len: int = 30) -> str:
        text = str(value or "").replace("\n", " ").strip()
        text = re.sub(r"\s+", "", text)
        text = text.strip(" -_，。,.!")
        if not text:
            text = fallback
        return text[:max_len]

    def _merge_segments(self, segments: list, min_gap: float = 1.0) -> list:
        if not segments:
            return []
        merged = [dict(segments[0])]
        for seg in segments[1:]:
            last = merged[-1]
            gap = seg["start"] - last["end"]
            # 只有同一说话人（或都无 speaker）才合并，避免把不同人的台词粘成一句
            same_speaker = seg.get("speaker") == last.get("speaker")
            if same_speaker and gap < min_gap and len(last["text"] + seg["text"]) < 200:
                last["text"] += seg["text"]
                last["end"] = seg["end"]
            else:
                merged.append(dict(seg))
        return merged

    def _picture_led_rules(self, max_beats: int):
        """按 cut_style 返回 (feature_line, beat_rule, coverage_rule)，三套选材策略。
        remix 模式无视 cut_style，用自己的原声混剪规则。"""
        lo, hi = self.min_minutes, self.max_minutes
        dur_clause = (f"**硬性时长：所有节拍时长（end_time-start_time）加起来必须落在 {lo:.0f}-{hi:.0f} 分钟"
                      f"之间，绝不能超过 {hi:.0f} 分钟**——成片总长就等于这个总和。")
        if self.mode == "remix":
            feature_line = ("这类视频是**原声混剪**：挑出高潮片段原样连续播放、保留原声台词，"
                            "**没有旁白解说来补剧情**，画面和台词要自己把故事讲明白。"
                            "你的任务是挑出本集最抓人的冲突、对峙、反转、揭露、情绪爆发等高潮片段，"
                            "串成观众不看原剧也能跟上的一条线。")
            beat_rule = (f"2. 节拍数量控制在 8-{max_beats} 个之间，**每个节拍必须是一段完整的对话/场景单元**"
                         f"（自然时长约 15-45 秒，从一句台词的开头切入、在一句台词说完后切出，"
                         f"不要把对话拦腰截断）。没有旁白补叙，所以**相邻节拍的剧情跳跃要小，"
                         f"观众凭画面和台词就能自行衔接**。{dur_clause}")
            coverage_rule = ("1. 节拍按 start_time 从小到大排列，聚焦高潮和关键转折，可以跳过平淡过场；"
                             "但因为没有旁白补叙，被跳过的剧情必须是观众能自行脑补的，不能留逻辑断层。")
            return feature_line, beat_rule, coverage_rule
        style = self.cut_style
        if style == "skeleton":
            feature_line = ("这类视频的特点：**画面保留主线关键节点+高潮段落、原样连续播放，"
                            "让没看过的人也能看懂这一集，平淡的过场由解说带过**。"
                            "你的任务是搭一条清晰的主线骨架（开端→发展→高潮→结局的关键场景都要有），"
                            "并把高潮戏给足时长，次要细节才跳过。")
            beat_rule = (f"2. 节拍数量控制在 12-{max_beats} 个之间，每个节拍是一段关键剧情或高潮戏的连续真实区间"
                         f"（自然时长约 12-50 秒、原样连续播放）。**主线关键节点（交代来龙去脉的场景）要保留，"
                         f"高潮戏给足时长，只跳过真正次要的日常细节**。{dur_clause}")
            coverage_rule = ("1. 节拍按 start_time 从小到大排列，**搭出能看懂的主线骨架**："
                             "本集的开端、关键转折、高潮、结局都要有代表性场景，既不要只堆高潮也不要流水账，"
                             "次要细节可以跳过、由解说一句带过。")
        elif style == "pure":
            feature_line = ("这类视频的特点：**只保留全集最炸、最有冲击力的名场面，原样连续播放，纯情绪爽感**。"
                            "你的任务是挑出本集最劲爆的冲突、撕逼、反转、名台词、名场面，"
                            "不必管前因后果和剧情连贯，怎么炸怎么来。")
            beat_rule = (f"2. 节拍数量控制在 5-{min(12, max_beats)} 个之间，每个节拍是一段最有冲击力的高潮戏连续区间"
                         f"（自然时长约 10-45 秒、原样连续播放）。**只挑最炸的，大胆舍弃一切铺垫和过渡**。"
                         f"{dur_clause}")
            coverage_rule = ("1. 节拍按 start_time 从小到大排列，**只挑最有冲击力的名场面**，"
                             "允许剧情大幅跳跃、不必连贯，目标是每一秒都有爽感。")
        else:  # bridge（默认，方案A）
            feature_line = ("这类视频的特点：**画面只保留本集最有张力的高潮段落、原样连续播放，"
                            "平淡的铺垫和过场不进画面、改由解说一句话带过**。"
                            "你的任务是挑出本集最抓人的冲突、对峙、反转、揭露、情绪爆发、名场面等高潮片段，"
                            "串成一条紧凑又吸引人的线——**不必覆盖全集每一处剧情**。")
            beat_rule = (f"2. 节拍数量控制在 8-{max_beats} 个之间，**每个节拍是一段高潮戏的连续真实区间**"
                         f"（自然时长约 12-40 秒、原样连续播放，单个节拍不要超过 50 秒）。"
                         f"**只挑高潮，主动跳过没有戏剧张力的日常铺垫、交代、过场**"
                         f"——这些被跳过的剧情后续由解说补叙，所以放心跳，宁缺毋滥。{dur_clause}")
            coverage_rule = ("1. 节拍按 start_time 从小到大排列，**聚焦高潮和关键转折，可以跳过平淡的铺垫、"
                             "交代、过场**；但相邻两个高潮之间被跳过的剧情，必须是能靠一两句解说衔接上的——"
                             "不要跳到观众看不懂，被跳的剧情解说会补，但不能留逻辑断层。")
        return feature_line, beat_rule, coverage_rule

    def build_explain_prompt(self, transcript_json: dict, episode_background: str,
                             video_duration: float = 0, feedback: str = "") -> str:
        raw_segments = transcript_json.get("segments", [])
        merged_segments = self._merge_segments(raw_segments)

        skip_intro = self.skip_intro_seconds
        skip_outro = self.skip_outro_seconds
        if skip_intro > 0 or skip_outro > 0:
            filtered = []
            for seg in merged_segments:
                if skip_intro > 0 and seg["start"] < skip_intro:
                    continue
                if skip_outro > 0 and video_duration > 0 and seg["end"] > video_duration - skip_outro:
                    continue
                filtered.append(seg)
            merged_segments = filtered

        lines = []
        has_speaker = False
        for seg in merged_segments:
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            spk = seg.get("speaker")
            tag = ""
            if spk:
                has_speaker = True
                tag = f"{spk}: "
            lines.append(f"[{start} --> {end}] {tag}{seg['text']}")
        content = "\n".join(lines)

        speaker_note = ""
        if has_speaker:
            speaker_note = (
                "\n字幕里每行前面的 SPEAKER_xx 是按声音自动区分的说话人标签："
                "同一个标签代表同一个人在说话（标签本身不是真名，真名请结合背景与剧情判断）。"
                "请据此判断对话是谁对谁说、谁在做什么，避免把某个角色的台词或行为安到别人头上。\n"
            )

        background_section = ""
        if episode_background:
            background_section = (
                f"\n下面是本集相关的剧集背景（人物关系、本集剧情以此为准，"
                f"用于修正 ASR 人名识别错误）：\n\n{episode_background}\n"
            )

        max_beats = self.max_beats
        target_min = self.target_minutes
        example = ('{"hook_title": "她以为赢了，其实早被算计", "cover_title": "全员演技，反转到最后", '
                   '"video_description": "豪门联姻背后的复仇局", "central_question": "谁才是幕后布局者？", '
                   '"segments": [{"start_time": 12.0, "end_time": 22.0, "summary": "开场:许蜜语签下联姻协议，以为攀上高枝"}, '
                   '{"start_time": 95.0, "end_time": 108.0, "summary": "纪封冷淡相待，暗示这桩婚姻另有目的"}]}')

        remix_rule = ""
        if self.mode == "remix":
            example = ('{"hook_title": "她以为赢了，其实早被算计", "cover_title": "全员演技，反转到最后", '
                       '"video_description": "豪门联姻背后的复仇局", "central_question": "谁才是幕后布局者？", '
                       '"segments": [{"start_time": 95.0, "end_time": 128.0, "summary": "纪封当众撕毁协议，摊牌婚姻另有目的", '
                       '"hook_score": 9}, '
                       '{"start_time": 402.0, "end_time": 431.0, "summary": "许蜜语发现房间里的监控，意识到被算计", '
                       '"hook_score": 7}]}')
            remix_rule = ("\n8. 每个节拍额外输出 hook_score 字段：1-10 分，"
                          "这个节拍单独放在视频开头做钩子的吸引力（冲突烈度/悬念感/情绪冲击）。")

        if self.mode in ("picture_led", "remix"):
            feature_line, beat_rule, coverage_rule = self._picture_led_rules(max_beats)
        else:
            feature_line = ("这类视频的特点：**解说旁白主导叙事，画面只是配合解说的素材**。"
                            "你的任务不是挑高光，而是按时间顺序把整集剧情**完整讲清楚**。")
            beat_rule = (f"2. 节拍数量控制在 18-{max_beats} 个之间。最终成片约 {target_min} 分钟，"
                         f"画面长度由后续解说决定，所以**每个节拍只需给一段有代表性的源画面区间**"
                         f"（5-15 秒即可），无需凑时长。")
            coverage_rule = ("1. 节拍要按 start_time 从小到大、覆盖全集主线剧情"
                             "（开场铺垫→发展→冲突→转折→结尾悬念），不要只选高潮、不要跳过关键交代。")

        feedback_section = ""
        if feedback:
            feedback_section = f"\n**上一次的结果不合格，请按以下反馈修正后重新输出：{feedback}**\n"

        return EXPLAIN_ANALYSIS.substitute(
            background_section=background_section, speaker_note=speaker_note,
            feature_line=feature_line, feedback_section=feedback_section,
            coverage_rule=coverage_rule, beat_rule=beat_rule, remix_rule=remix_rule,
            example=example, content=content)

    def _validate_explain_segments(self, segments: list, video_duration: float) -> list:
        cleaned = []
        for seg in segments:
            try:
                st = float(seg.get("start_time"))
                en = float(seg.get("end_time"))
            except (TypeError, ValueError):
                continue
            if st < 0:
                st = 0.0
            if video_duration > 0 and st >= video_duration - 1.0:
                continue  # LLM 幻觉出片尾之后的节拍：整拍丢弃，不制造片外碎片
            if video_duration > 0 and en > video_duration:
                en = video_duration
            if en <= st:
                en = st + 5.0
            seg["start_time"] = st
            seg["end_time"] = en
            cleaned.append(seg)
        cleaned.sort(key=lambda s: s["start_time"])
        if len(cleaned) > self.max_beats:
            cleaned = cleaned[:self.max_beats]
        return cleaned

    def _build_hook_preview_segment(self, hook_seg: dict) -> dict:
        """从最高钩子节拍截取开头 3-6 秒做成片头短预告；完整节拍仍留在原时间线。"""
        start = float(hook_seg.get("start_time", 0.0))
        end = float(hook_seg.get("end_time", start))
        preview_end = min(end, start + self.hook_preview_seconds)
        if preview_end <= start:
            return {}
        preview = dict(hook_seg)
        preview["start_time"] = start
        preview["end_time"] = preview_end
        preview["is_hook_preview"] = True
        preview["narration"] = None
        return preview

    def _apply_remix_ordering(self, segments: list) -> list:
        """remix 混剪排序：最高 hook_score 节拍(或其短预告)前置作开头钩子，其余按时间顺序。"""
        if len(segments) <= 1:
            return segments

        def hook_key(s):
            try:
                score = float(s.get("hook_score", 0) or 0)
            except (TypeError, ValueError):
                score = 0.0
            return (score, s.get("end_time", 0) - s.get("start_time", 0))

        chronological = sorted(segments, key=lambda s: s["start_time"])
        hook_seg = max(chronological, key=hook_key)
        if chronological[0] is hook_seg:
            return chronological  # 钩子本来就在开头，无需重排

        if self.hook_preview_enabled:
            preview = self._build_hook_preview_segment(hook_seg)
            if preview:
                ordered = [preview] + chronological
                if self.logger:
                    self.logger.info(f"Remix: hook preview {preview['start_time']:.0f}s "
                                     f"(+{preview['end_time']-preview['start_time']:.1f}s) 前置")
                return ordered
        # 无预告：整段钩子前置，第二段回到时间线开头
        remaining = [s for s in chronological if s is not hook_seg]
        if self.logger:
            self.logger.info(f"Remix: hook beat {hook_seg['start_time']:.0f}s 整段前置")
        return [hook_seg] + remaining

    def _compress_beats_to(self, segments: list, target_sec: float) -> list:
        """等比压缩每个节拍时长，使总时长 ≈ target_sec。保留全部节拍与 start_time，
        只收缩 end_time（从场景开头播得短一点）；每拍保底 3 秒避免压成闪帧。"""
        total = sum(s["end_time"] - s["start_time"] for s in segments)
        if total <= target_sec or total <= 0:
            return segments
        factor = target_sec / total
        out = []
        for s in segments:
            dur = max((s["end_time"] - s["start_time"]) * factor, 3.0)
            ns = dict(s)
            ns["end_time"] = round(s["start_time"] + dur, 3)
            out.append(ns)
        return out

    def _call_llm_json(self, prompt: str):
        return self.llm.chat_json(prompt)

    def _write_quality_report(self, result_json: dict, report_path: Path):
        segments = result_json.get("segments", [])
        total_duration = sum(s["end_time"] - s["start_time"] for s in segments)
        report = {
            "mode": "explain",
            "hook_title": result_json.get("hook_title", ""),
            "cover_title": result_json.get("cover_title", ""),
            "video_description": result_json.get("video_description", ""),
            "central_question": result_json.get("central_question", ""),
            "beat_count": len(segments),
            "source_coverage_seconds": round(total_duration, 2),
            "segments": [
                {"summary": s.get("summary", ""),
                 "start_time": s.get("start_time"),
                 "end_time": s.get("end_time")}
                for s in segments
            ],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    def analyze(self, transcript_path: str, output_dir: str = "output") -> str:
        if self.logger:
            self.logger.info(f"Loading transcript from: {transcript_path}")

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_json = json.load(f)

        all_segs = transcript_json.get("segments", [])
        video_dur = all_segs[-1]["end"] if all_segs else 3600

        episode_num = parse_episode_number(Path(transcript_path).stem)
        episode_background = extract_episode_background(self.series_background, episode_num)
        if self.logger:
            self.logger.info(f"Explain mode: episode {episode_num}, "
                             f"episode background {len(episode_background)} chars")

        # 两道护栏，不达标就带反馈重试，保留综合评分最高的一次：
        #  ① 覆盖度：LLM 偶发把节拍挤在开头一小段，音画严重对不上。
        #  ② 时长（仅 picture_led）：成片时长=Σ节拍时长，必须落在 [min,max] 分钟，否则成片动辄十几分钟。
        valid_lo = max(self.skip_intro_seconds, 0.0)
        valid_hi = video_dur - self.skip_outro_seconds if self.skip_outro_seconds > 0 else video_dur
        valid_range = max(valid_hi - valid_lo, 1.0)
        # remix 与 picture_led 一样：成片时长 = Σ节拍时长（remix 无解说撑长，上限更是硬的）
        is_pic = self.mode in ("picture_led", "remix")
        min_sec, max_sec = self.min_minutes * 60.0, self.max_minutes * 60.0

        feedback = ""
        best = None  # (result_json, segments, score)
        for attempt in range(3):
            prompt = self.build_explain_prompt(transcript_json, episode_background,
                                               video_duration=video_dur, feedback=feedback)
            if self.logger and attempt == 0:
                self.logger.info(f"Explain prompt size: {len(prompt)} chars, calling {self.model_name}")
            result_json = self._call_llm_json(prompt)
            if not isinstance(result_json, dict):
                result_json = {}
            segments = self._validate_explain_segments(result_json.get("segments", []), video_dur)
            if not segments:
                continue
            starts = [s["start_time"] for s in segments]
            span = max(s["end_time"] for s in segments) - min(starts)
            below_intro = sum(1 for x in starts if x < valid_lo - 1.0) / len(starts)
            coverage = span / valid_range
            total_dur = sum(s["end_time"] - s["start_time"] for s in segments)
            cluster_ok = coverage >= 0.25 and below_intro <= 0.3
            dur_ok = (not is_pic) or (min_sec <= total_dur <= max_sec)
            # 评分：先满足覆盖度(2)，再满足时长(1)，最后偏好跨度大的(<=0.1)
            score = (2.0 if cluster_ok else 0.0) + (1.0 if dur_ok else 0.0) + min(span / valid_range, 1.0) * 0.1
            if best is None or score > best[2]:
                best = (result_json, segments, score)
            if cluster_ok and dur_ok:
                break
            fb = []
            if not cluster_ok:
                fb.append(f"节拍挤在一小段时间里（只覆盖全片 {coverage:.0%}），"
                          f"请让节拍按时间顺序覆盖从开场到结尾的完整剧情")
            if is_pic and total_dur > max_sec:
                fb.append(f"所有节拍时长加起来 {total_dur/60:.1f} 分钟，超过 {self.max_minutes:.0f} 分钟上限，"
                          f"请砍掉次要场景、缩短过长片段，把总时长压到 "
                          f"{self.min_minutes:.0f}-{self.max_minutes:.0f} 分钟（务必保留开场和结尾）")
            elif is_pic and total_dur < min_sec:
                fb.append(f"所有节拍时长加起来只有 {total_dur/60:.1f} 分钟，不足 {self.min_minutes:.0f} 分钟，"
                          f"请多覆盖主线场景，把总时长补到 {self.min_minutes:.0f}-{self.max_minutes:.0f} 分钟")
            feedback = "；".join(fb)
            if self.logger:
                self.logger.warning(f"Explain beats rejected (coverage {coverage:.0%}, "
                                    f"total {total_dur/60:.1f}min, cluster_ok={cluster_ok} dur_ok={dur_ok}); "
                                    f"retry {attempt + 1}/3")

        if not best:
            raise RuntimeError("Explain analysis produced no usable plot beats")
        result_json, segments, _ = best

        # 兜底：picture_led 若 3 次重试后总时长仍超上限，等比压缩每拍时长到 max
        # （保留全部节拍含结尾，只是每段播得短一点，不丢剧情、不丢结局）。
        if is_pic:
            total_dur = sum(s["end_time"] - s["start_time"] for s in segments)
            if total_dur > max_sec:
                segments = self._compress_beats_to(segments, max_sec)
                if self.logger:
                    new_tot = sum(s["end_time"] - s["start_time"] for s in segments)
                    self.logger.warning(f"Beats still {total_dur/60:.1f}min after retries; "
                                        f"compressed to {new_tot/60:.1f}min (cap {self.max_minutes:.0f}min)")

        # remix：时长定稿后再做混剪排序（钩子前置会额外加 3-6 秒预告，不参与时长护栏）
        if self.mode == "remix":
            segments = self._apply_remix_ordering(segments)
            result_json["remix_enabled"] = True

        first_summary = segments[0].get("summary", "") if segments else ""
        result_json["video_description"] = self._clean_title(
            result_json.get("video_description") or first_summary, fallback="剧情解说", max_len=20)
        result_json["hook_title"] = self._clean_title(
            result_json.get("hook_title"), fallback=result_json["video_description"],
            max_len=22, keep_punct=True)
        result_json["cover_title"] = self._clean_title(
            result_json.get("cover_title"), fallback=result_json["hook_title"],
            max_len=20, keep_punct=True)
        result_json["central_question"] = self._clean_metadata_text(
            result_json.get("central_question") or result_json["hook_title"],
            fallback=result_json["hook_title"], max_len=30)
        result_json["narration_mode"] = "explain"
        result_json["explain_mode"] = self.mode
        result_json["episode_num"] = episode_num
        # 整集梗概：解说桥接被剪掉剧情的内部资料（prompt 第7条产出）
        result_json["episode_synopsis"] = str(
            result_json.get("episode_synopsis") or "").strip()[:800]

        for idx, seg in enumerate(segments):
            seg["summary"] = str(seg.get("summary") or "精彩片段").strip()[:40]
            seg["source_order"] = idx
            score = self._score_value(seg.get("engagement_score"), 7)
            seg["engagement_score"] = score
            seg["tension_level"] = self._tension_from_score(score)
        result_json["segments"] = segments

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        base_name = Path(transcript_path).stem.replace("_transcript", "")
        output_file = output_path / f"{base_name}_key_segments.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result_json, f, ensure_ascii=False, indent=2)

        self._write_quality_report(result_json, output_path / "quality_report.json")

        if self.logger:
            total = sum(s["end_time"] - s["start_time"] for s in segments)
            self.logger.info(f"Explain beats: {len(segments)} beats, source coverage {total/60:.1f} min "
                             f"(final length set by narration)")
        return str(output_file)


def analyze_transcript(transcript_path: str, config, output_dir: str = "output") -> str:
    strategy = config.strategy or {}
    video_cfg = config.video
    nar_cfg = config.narration or {}
    analyzer = Analyzer(
        base_url=config.llm.get("base_url"),
        api_key=config.llm.get("api_key"),
        model_name=config.llm.get("model_name"),
        strategy=strategy,
        skip_intro_seconds=video_cfg.get("skip_intro_seconds", 0),
        skip_outro_seconds=video_cfg.get("skip_outro_seconds", 0),
        series_background=video_cfg.get("series_background", ""),
        explain_config=nar_cfg.get("explain", {}),
    )
    analyzer.set_logger(config.get_logger("Analyzer"))
    return analyzer.analyze(transcript_path, output_dir)


if __name__ == "__main__":
    from config import load_config
    import sys

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    transcript_path = sys.argv[2] if len(sys.argv) > 2 else "output/test_transcript.json"
    config = load_config(config_path)
    print(f"Key segments: {analyze_transcript(transcript_path, config)}")
