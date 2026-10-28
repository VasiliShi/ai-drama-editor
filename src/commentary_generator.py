import json
import shutil
from pathlib import Path

from episode_utils import parse_episode_number, extract_episode_background, load_scene_notes
from llm import LLMClient, ContentFilterError
from prompts import COMMENTARY as PROMPT_TEMPLATE
from prompts import COMMENTARY_SELF_CHECK


class CommentaryGenerator:
    """remix(混剪评论)模式:为每个节拍生成影评/弹幕式评论,叠加在原声画面上。

    准确性设施(与解说同一套思路;对白直接用 ASR 原文,LLM 多语言能力足够,不做翻译层——
    保持语言无关,换其他语种剧集零改动):
    - scene_notes 画面事实(VLM+人脸在场)注入 prompt,解读必须有画面依据;
    - episode_synopsis 全集梗概防剧情走向误判;
    - 人名白名单(人脸库真名)硬约束;
    - 整集生成后 LLM 自检一遍,张冠李戴/无中生有的评论改写或删除。"""

    PROMPT_TEMPLATE = PROMPT_TEMPLATE

    # 每个节拍调一次 LLM，完整背景会成倍放大 token 成本；截取前段(人物阵容/主线)即可满足人名校正。
    MAX_BACKGROUND_CHARS = 1500

    def __init__(self, base_url: str, api_key: str, model_name: str,
                 series_background: str = "", commentary_config: dict = None,
                 cast_names: list = None):
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self.series_background = series_background or ""
        self.config = commentary_config or {}
        self.cast_names = list(cast_names or [])
        self.logger = None
        self.llm = LLMClient(base_url, api_key, model_name)

    def set_logger(self, logger):
        self.logger = logger
        self.llm.logger = logger

    def _build_background_section(self, episode_num: int) -> str:
        bg = extract_episode_background(self.series_background, episode_num) or self.series_background
        bg = (bg or "").strip()
        if not bg:
            return ""
        if len(bg) > self.MAX_BACKGROUND_CHARS:
            bg = bg[:self.MAX_BACKGROUND_CHARS].rstrip() + "……"
        return f"\n## 剧集背景（人名/人物关系以此为准，用于修正 ASR 听错的人名）\n{bg}\n"

    # ── 整集评论自检（仿解说 self_check）────────────────────────────

    def _self_check(self, segments: list, beat_facts: dict, max_length: int) -> None:
        blocks = []
        for i, seg in enumerate(segments):
            comments = seg.get("commentaries") or []
            if not comments:
                continue
            facts = beat_facts.get(i, "")
            com_lines = "\n".join(f"  ({j+1}) [{c.get('kind','')}] {c.get('text','')}"
                                  for j, c in enumerate(comments))
            blocks.append(f"[节拍{i+1}]\n{facts}\n  评论:\n{com_lines}")
        if not blocks:
            return
        prompt = COMMENTARY_SELF_CHECK.format(beats="\n\n".join(blocks))
        try:
            result = self.llm.chat_json(prompt)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"评论自检跳过(LLM 失败): {e}")
            return
        fixes = (result or {}).get("fixes") if isinstance(result, dict) else None
        parsed = []
        for f in fixes or []:
            try:
                parsed.append((int(f.get("beat")) - 1, int(f.get("idx")) - 1,
                               str(f.get("text", "")).strip()))
            except (TypeError, ValueError, AttributeError):
                continue
        rewritten = deleted = 0
        # 先改写(不动索引)，再按索引从大到小删除(避免 pop 后索引漂移)
        for bi, ci, new_text in parsed:
            if not new_text or not (0 <= bi < len(segments)):
                continue
            comments = segments[bi].get("commentaries") or []
            if 0 <= ci < len(comments) and len(new_text) <= max_length + 6:
                comments[ci]["text"] = new_text[:max_length]
                rewritten += 1
        for bi, ci, new_text in sorted((p for p in parsed if not p[2]),
                                       key=lambda p: (p[0], -p[1])):
            if not (0 <= bi < len(segments)):
                continue
            comments = segments[bi].get("commentaries") or []
            if 0 <= ci < len(comments):
                comments.pop(ci)
                deleted += 1
        if self.logger:
            self.logger.info(f"评论自检: 改写 {rewritten} 条, 删除 {deleted} 条")

    def _fallback_commentary(self, seg_summary: str, dur: float) -> list:
        text = (seg_summary or "").strip()
        if not text:
            return []
        return [{"timing": max(0.0, min(1.0, dur - 2)) if dur > 0 else 0.0,
                 "text": text[:18], "kind": "explain"}]

    def generate(self, transcript_path: str, key_segments_path: str, output_dir: str) -> str:
        gen = self.config.get("generation", {})
        style_note = str(gen.get("style_note", "")).strip()
        max_length = int(gen.get("max_length", 18))
        auto_interval = float(gen.get("auto_interval", 12))
        try:
            explain_ratio = max(0.0, min(1.0, float(gen.get("explain_ratio", 0.4))))
        except (TypeError, ValueError):
            explain_ratio = 0.4
        try:
            review_ratio = max(0.0, min(1.0, float(gen.get("review_ratio", 0.3))))
        except (TypeError, ValueError):
            review_ratio = 0.3

        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
        with open(key_segments_path, "r", encoding="utf-8") as f:
            key_data = json.load(f)

        all_transcript_segs = transcript_data.get("segments", [])
        segments = key_data.get("segments", [])
        if not segments:
            if self.logger:
                self.logger.warning("No segments to generate commentary for")
            return key_segments_path

        base_name = Path(key_segments_path).stem.replace("_key_segments", "")
        episode_num = key_data.get("episode_num") or parse_episode_number(base_name)
        background_section = self._build_background_section(episode_num)
        video_desc = key_data.get("video_description", "精彩剧集")

        # 准确性材料:全集梗概 + 画面事实(scene_notes 步骤产物) + 人名白名单
        synopsis = str(key_data.get("episode_synopsis", "") or "").strip()
        synopsis_section = (f"\n## 本集完整剧情梗概（内部资料，用于理解剧情走向）\n{synopsis[:800]}\n"
                            if synopsis else "")
        if self.cast_names:
            synopsis_section += ("\n## 人名白名单（人脸库已确认的角色名，评论只能用这些名字或背景介绍里的名字）\n"
                                 + "、".join(self.cast_names) + "\n")
        visual_notes = load_scene_notes(key_segments_path)

        beat_facts = {}      # 自检用:每拍的对白+画面事实
        previous_texts = []  # 跨片段累积，用于去重与递进
        for si, seg in enumerate(segments):
            seg_start = seg["start_time"]
            seg_end = seg["end_time"]
            dur = seg_end - seg_start
            seg_summary = str(seg.get("summary") or "").strip()

            # hook_preview 是完整钩子片段的重复预告，不单独配评论（画面很短，留给开场标题）
            if seg.get("is_hook_preview"):
                seg["commentaries"] = []
                continue

            relevant_texts = []
            for ts in all_transcript_segs:
                if ts["end"] > seg_start and ts["start"] < seg_end:
                    start_rel = max(ts["start"] - seg_start, 0)
                    end_rel = min(ts["end"] - seg_start, dur)
                    spk = ts.get("speaker") or ""
                    spk_tag = f" {spk}:" if spk else ""
                    relevant_texts.append(f"[{start_rel:.0f}s-{end_rel:.0f}s]{spk_tag} {ts['text']}")

            if not relevant_texts:
                if self.logger:
                    self.logger.warning(f"  Segment {si+1}: no matching transcript found")
                seg["commentaries"] = self._fallback_commentary(seg_summary, dur)
                continue

            content = "\n".join(relevant_texts)
            note = visual_notes.get(si, "")
            scene_section = (f"\n## 画面事实（视觉模型看画面的记录+人脸识别在场名单，解读必须与此一致）\n{note}\n"
                             if note else "")
            beat_facts[si] = (f"  画面事实: {note or '(无)'}\n  对白: "
                              + " / ".join(relevant_texts)[:400])
            count = min(max(int(dur / auto_interval), 2), 8)
            explain_count = max(1, round(count * explain_ratio))
            review_count = max(1, round(count * review_ratio)) if count >= 3 else 0
            emotion_count = max(0, count - explain_count - review_count)

            previous_section = ""
            if previous_texts:
                recent = previous_texts[-15:]
                previous_section = ("\n## 已生成评论（不要重复，解读要在此基础上推进）\n"
                                    + "\n".join(f"- {t}" for t in recent) + "\n")

            prompt = self.PROMPT_TEMPLATE.format(
                background_section=background_section,
                synopsis_section=synopsis_section,
                scene_section=scene_section,
                video_description=video_desc,
                segment_summary=seg_summary,
                previous_section=previous_section,
                content=content,
                count=count,
                explain_count=explain_count,
                review_count=review_count,
                emotion_count=emotion_count,
                max_length=max_length,
            )
            if style_note:
                prompt += f"\n\n补充风格要求：{style_note}"

            if self.logger:
                self.logger.info(f"  Segment {si+1}/{len(segments)}: count={count} "
                                 f"(解读{explain_count}/影评{review_count}/共鸣{emotion_count})")

            try:
                items = self.llm.chat_json(prompt)
            except ContentFilterError as e:
                # 单拍被过滤不拖垮整集：降级为 summary 兜底评论
                if self.logger:
                    self.logger.warning(f"    content_filter，该拍降级兜底: {e}")
                items = None
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"    LLM failed after retries: {e}")
                items = None
            if not isinstance(items, list):
                items = []

            cleaned = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                text = str(item.get("text", "")).strip()[:max_length]
                if not text:
                    continue
                kind = str(item.get("kind", "")).strip().lower()
                if kind not in ("explain", "review", "emotion"):
                    kind = "explain"
                try:
                    timing = float(item.get("timing", 0))
                except (TypeError, ValueError):
                    timing = 0.0
                cleaned.append({"text": text, "kind": kind,
                                "timing": max(0.0, min(timing, max(dur - 2, 0.0)))})
            if not cleaned:
                cleaned = self._fallback_commentary(seg_summary, dur)

            seg["commentaries"] = cleaned
            previous_texts.extend(c["text"] for c in cleaned if c.get("text"))
            if self.logger:
                self.logger.info(f"    -> {len(cleaned)} commentaries")

        # 整集一致性自检：张冠李戴/无中生有的评论改写或删除
        if self.config.get("self_check", True):
            self._self_check(segments, beat_facts, max_length)

        out_file = Path(output_dir) / f"{base_name}_key_segments.json"
        if out_file.exists():
            shutil.copyfile(out_file, f"{out_file}.bak")  # 原地改写前留底
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(key_data, f, ensure_ascii=False, indent=2)

        if self.logger:
            total = sum(len(s.get("commentaries", [])) for s in segments)
            self.logger.info(f"Commentary done: {total} comments across {len(segments)} beats")
        return str(out_file)


def generate_commentaries(transcript_path: str, key_segments_path: str, config, output_dir: str) -> str:
    # 人名白名单：取自背景资料的角色表(音译统一,避免同一人多种译名)。
    # 原先取自人脸库 keys——那份名单既不全，又带着建库时的错误命名。
    try:
        from speaker_naming import load_roster
        cast_names = load_roster(config, config.get_logger("Commentary"))
    except Exception:
        cast_names = []
    cg = CommentaryGenerator(
        base_url=config.llm.get("base_url"),
        api_key=config.llm.get("api_key"),
        model_name=config.llm.get("model_name"),
        series_background=config.video.get("series_background", ""),
        commentary_config=config.commentary,
        cast_names=cast_names,
    )
    cg.set_logger(config.get_logger("Commentary"))
    return cg.generate(transcript_path, key_segments_path, output_dir)
