import re
import shutil
from pathlib import Path

from media import get_video_encoder, get_resolution, run_ffmpeg


class SubtitleAdder:
    """电影解说渲染：缩放到目标比例 + 底部解说字幕条 + 标题/作者/版权/片尾文字。"""

    def __init__(self, caption_font_size: int = 42, caption_font_color: str = "white",
                 caption_y_position: float = 0.80,
                 font_path: str = None,
                 aspect_ratio: str = "9:16",
                 target_resolution: str = "1080x1920",
                 author_name: str = None,
                 author_font_size: int = 28,
                 author_font_color: str = "white",
                 desc_font_size: int = 48,
                 desc_font_color: str = "white",
                 desc_y_offset: float = 0.4,
                 ending_config: dict = None,
                 copyright_config: dict = None,
                 commentary_config: dict = None):
        self.caption_font_size = caption_font_size
        self.caption_font_color = caption_font_color
        self.caption_y_position = self._coerce_float(caption_y_position, 0.80)
        self.font_path = str(Path(font_path or self._find_chinese_font()).resolve())
        self.aspect_ratio = aspect_ratio
        self.target_resolution = target_resolution
        self.tgt_w, self.tgt_h = map(int, target_resolution.split("x"))
        self.author_name = author_name
        self.author_font_size = author_font_size
        self.author_font_color = author_font_color
        self.desc_font_size = self._coerce_int(desc_font_size, 48)
        self.desc_font_color = desc_font_color
        self.desc_y_offset = desc_y_offset
        self.ending_config = ending_config or {}
        self.copyright_config = copyright_config or {}
        self.commentary_config = commentary_config or {}
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger

    @staticmethod
    def _coerce_int(value, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_float(value, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _find_chinese_font(self) -> str:
        candidates = [
            str(Path("fonts/wqy-microhei.ttc").resolve()),
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ]
        for path in candidates:
            if Path(path).exists():
                return str(Path(path).resolve())
        return "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

    def _get_video_resolution(self, video_path: str) -> tuple:
        return get_resolution(video_path) or (1920, 1080)

    def _build_scale_pad_filter(self, vw: int, vh: int) -> str:
        scale_w = self.tgt_w
        scale_h = int(scale_w * vh / vw)
        if scale_h % 2 == 1:
            scale_h += 1
        if scale_h > self.tgt_h:
            scale_h = self.tgt_h
            scale_w = int(scale_h * vw / vh)
            if scale_w % 2 == 1:
                scale_w += 1
        pad_x = (self.tgt_w - scale_w) // 2
        pad_y = (self.tgt_h - scale_h) // 2
        return f"scale={scale_w}:{scale_h},pad={self.tgt_w}:{self.tgt_h}:{pad_x}:{pad_y}:black"

    def _calc_pad(self, vw: int, vh: int):
        scale_w = self.tgt_w
        scale_h = int(scale_w * vh / vw)
        if scale_h % 2 == 1:
            scale_h += 1
        if scale_h > self.tgt_h:
            scale_h = self.tgt_h
            scale_w = int(scale_h * vw / vh)
            if scale_w % 2 == 1:
                scale_w += 1
        pad_y = (self.tgt_h - scale_h) // 2
        return pad_y, scale_h

    def _safe_text(self, text) -> str:
        safe = str(text or "").replace("\n", " ").strip()
        safe = re.sub(r"\s+", " ", safe)
        return re.sub(r"[\\'\[\]:%]", "", safe)

    def _drawtext_style(self, borderw: int = 2, shadow: bool = True) -> str:
        parts = [f"borderw={borderw}", "bordercolor=black@0.75"]
        if shadow:
            parts.extend(["shadowcolor=black@0.45", "shadowx=2", "shadowy=2"])
        return ":".join(parts)

    def _select_display_title(self, key_data: dict) -> str:
        for key in ("hook_title", "video_description", "cover_title"):
            text = self._safe_text(key_data.get(key, ""))
            if text:
                return text[:24]
        return ""

    def _split_caption(self, text: str, max_len: int = 18) -> list:
        """把一段解说词切成适合底部字幕的短行：优先按标点断，再按长度兜底。"""
        text = self._safe_text(text)
        pieces = re.split(r'(?<=[，。！？、,!?])', text)
        lines = []
        cur = ""
        for p in pieces:
            p = p.strip()
            if not p:
                continue
            if len(cur) + len(p) <= max_len:
                cur += p
            else:
                if cur:
                    lines.append(cur)
                while len(p) > max_len:
                    lines.append(p[:max_len])
                    p = p[max_len:]
                cur = p
        if cur:
            lines.append(cur)
        return [ln.strip("，。！？、,!? ") for ln in lines if ln.strip("，。！？、,!? ")]

    def _narration_subtitle_filters(self, segments: list, trimmed_starts: list,
                                    durations: list, font_path: str) -> list:
        """每个节拍的旁白拆成短行，沿该节拍画面均匀分布，做成底部字幕条。"""
        filters = []
        y_pos = int(self.tgt_h * self.caption_y_position)
        for i, seg in enumerate(segments):
            nar = seg.get("narration") or {}
            text = nar.get("text", "") if isinstance(nar, dict) else ""
            if not text or i >= len(trimmed_starts):
                continue
            clip_dur = durations[i]
            span = min(self._coerce_float(nar.get("duration", clip_dur), clip_dur), clip_dur)
            if span <= 0:
                continue
            lines = self._split_caption(text)
            if not lines:
                continue
            per_line = span / len(lines)
            for k, line in enumerate(lines):
                start = trimmed_starts[i] + k * per_line
                end = start + per_line
                filters.append(
                    f"drawtext=text='{self._safe_text(line)}':"
                    f"fontsize={self.caption_font_size}:"
                    f"fontcolor={self.caption_font_color}:"
                    f"{self._drawtext_style(borderw=3)}:"
                    f"x=(w-text_w)/2:y={y_pos}:"
                    f"enable='between(t,{start:.2f},{end:.2f})':"
                    f"fontfile='{font_path}'"
                )
        return filters

    def _commentary_filters(self, segments: list, trimmed_starts: list,
                            durations: list, font_path: str, vw: int, vh: int) -> list:
        """remix 混剪：把每个节拍的影评/弹幕评论按 timing 叠加到画面上。
        文本包「」；位置在左半屏(以画面中心为原点的第二、三象限)贴近水平中线，
        随机抖动(固定种子,重渲染可复现)；三档高度轮换避免相邻评论互相遮挡。"""
        import random
        cfg = self.commentary_config
        font_size = self._coerce_int(cfg.get("font_size", 32), 32)
        display_dur = self._coerce_float(cfg.get("duration", 4), 4.0)
        kind_colors = {
            "explain": cfg.get("explain_color", "white"),
            "review": cfg.get("review_color", "gold"),
            "emotion": cfg.get("emotion_color", "orange"),
        }

        filters = []
        slot = 0
        for i, seg in enumerate(segments):
            comments = seg.get("commentaries") or []
            if not comments or i >= len(trimmed_starts):
                continue
            clip_dur = durations[i]
            for ci, com in enumerate(sorted(
                    comments, key=lambda c: self._coerce_float(c.get("timing", 0), 0))):
                text = self._safe_text(com.get("text", ""))
                if not text:
                    continue
                timing = self._coerce_float(com.get("timing", 0), 0.0)
                timing = max(0.0, min(timing, max(clip_dur - 1.0, 0.0)))
                start = trimmed_starts[i] + timing
                end = min(start + display_dur, trimmed_starts[i] + clip_dur)
                if end <= start:
                    continue
                color = kind_colors.get(str(com.get("kind", "")).lower(), "white")
                rng = random.Random(i * 97 + ci)   # 固定种子:同一数据重渲染位置不变
                band = (-1, 0, 1)[slot % 3]        # 中线上下三档轮换,防相邻评论重叠
                y_frac = 0.5 + band * 0.085 + rng.uniform(-0.02, 0.02)
                x_frac = rng.uniform(0.06, 0.92)   # 左半屏内随机横向落点
                slot += 1
                filters.append(
                    f"drawtext=text='「{text}」':"
                    f"fontsize={font_size}:fontcolor={color}:"
                    f"{self._drawtext_style(borderw=2)}:"
                    f"x='max(w*0.02,(w/2-text_w)*{x_frac:.2f})':y=h*{y_frac:.3f}:"
                    f"enable='between(t,{start:.2f},{end:.2f})':"
                    f"fontfile='{font_path}'"
                )
        return filters

    def add_subtitles(self, video_path: str, key_segments_path: str, output_path: str = None) -> str:
        import json
        with open(key_segments_path, "r", encoding="utf-8") as f:
            key_data = json.load(f)

        segments = key_data.get("segments", [])
        display_title = self._select_display_title(key_data)

        if not segments:
            if self.logger:
                self.logger.warning("No segments, copying video without subtitles")
            if output_path:
                shutil.copyfile(video_path, output_path)
            return output_path or video_path

        durations = [seg["end_time"] - seg["start_time"] for seg in segments]
        trimmed_starts = []
        cum = 0.0
        for i in range(len(segments)):
            trimmed_starts.append(cum)
            cum += durations[i]

        vw, vh = self._get_video_resolution(video_path)
        scale_pad = self._build_scale_pad_filter(vw, vh)
        font_path = self.font_path

        if self.logger:
            self.logger.info(f"Video {vw}x{vh} -> {self.target_resolution}, {len(segments)} beats")

        drawtext_filters = self._narration_subtitle_filters(segments, trimmed_starts, durations, font_path)
        # remix 混剪：影评/弹幕评论 + 转场提示（非 remix 时 segments 里没有这些字段，天然为空）
        drawtext_filters.extend(self._commentary_filters(segments, trimmed_starts, durations, font_path, vw, vh))

        if self.copyright_config.get("enabled", False):
            cp_text = self.copyright_config.get("text", "二次创作 · 仅供交流")
            cp_font_size = self.copyright_config.get("font_size", 36)
            cp_font_color = self.copyright_config.get("font_color", "white")
            cp_display = self.copyright_config.get("display_duration", 4)
            drawtext_filters.append(
                f"drawtext=text='{self._safe_text(cp_text)}':"
                f"fontsize={cp_font_size}:fontcolor={cp_font_color}@0.7:"
                f"{self._drawtext_style(borderw=1)}:"
                f"x=(w-text_w)/2:y=(h*0.85):"
                f"enable='between(t,0,{cp_display})':fontfile='{font_path}'"
            )

        if self.author_name:
            x_pos = self.tgt_w - self.author_font_size - 80
            y_center = self.tgt_h // 2
            char_spacing = self.author_font_size + 4
            y_start = y_center - len(self.author_name) * char_spacing // 2
            for i, ch in enumerate(self.author_name):
                drawtext_filters.append(
                    f"drawtext=text='{self._safe_text(ch)}':"
                    f"fontsize={self.author_font_size}:fontcolor={self.author_font_color}:"
                    f"{self._drawtext_style(borderw=1)}:"
                    f"x={x_pos}:y={y_start + i * char_spacing}:fontfile='{font_path}'"
                )

        if display_title and self.desc_font_size > 0:
            margin = 40
            avail_width = self.tgt_w - margin * 2
            chars_per_line = max(1, int(avail_width / self.desc_font_size))
            wrapped_lines = [display_title[i:i + chars_per_line]
                             for i in range(0, len(display_title), chars_per_line)]
            line_height = int(self.desc_font_size * 1.3)
            block_height = len(wrapped_lines) * line_height
            pad_y, _ = self._calc_pad(vw, vh)
            if pad_y > 0:
                desc_y0 = int(pad_y * self.desc_y_offset) - block_height // 2
            else:
                desc_y0 = self.desc_font_size
            desc_y0 = max(desc_y0, self.desc_font_size // 2)
            for li, line_text in enumerate(wrapped_lines):
                drawtext_filters.insert(0,
                    f"drawtext=text='{self._safe_text(line_text)}':"
                    f"fontsize={self.desc_font_size}:fontcolor={self.desc_font_color}:"
                    f"{self._drawtext_style(borderw=2)}:"
                    f"x=(w-text_w)/2:y={desc_y0 + li * line_height}:fontfile='{font_path}'"
                )

        if self.ending_config.get("enabled", False):
            ending_text = self.ending_config.get("text", "感谢观看")
            ending_font_size = self.ending_config.get("font_size", 48)
            ending_font_color = self.ending_config.get("font_color", "white")
            ending_display = self.ending_config.get("display_duration", 3)
            total_video_dur = trimmed_starts[-1] + durations[-1] if trimmed_starts else 0
            if total_video_dur > ending_display:
                ending_start = total_video_dur - ending_display
                drawtext_filters.append(
                    f"drawtext=text='{self._safe_text(ending_text)}':"
                    f"fontsize={ending_font_size}:fontcolor={ending_font_color}:"
                    f"{self._drawtext_style(borderw=2)}:"
                    f"x=(w-text_w)/2:y=(h-text_h)/2:"
                    f"enable='between(t,{ending_start},{total_video_dur})':fontfile='{font_path}'"
                )

        vf_string = ",".join([scale_pad] + drawtext_filters)

        if output_path is None:
            output_path = str(Path(video_path).parent / f"{Path(video_path).stem}_final.mp4")

        v_enc, v_args = get_video_encoder(quality=23)
        if self.logger:
            self.logger.info(f"Rendering {self.target_resolution} with {len(drawtext_filters)} drawtext filters")

        result = run_ffmpeg([
            "-i", video_path, "-vf", vf_string,
            "-c:v", v_enc, *v_args, "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            output_path
        ])
        if result.returncode != 0:
            if self.logger:
                self.logger.error(f"FFmpeg error: {result.stderr[:5000]}")
            raise RuntimeError("Failed to add subtitles")
        if self.logger:
            self.logger.info(f"Output saved to: {output_path}")
        return output_path


def add_subtitles(video_path: str, key_segments_path: str, config,
                  output_path: str = None, aspect_ratio: str = None,
                  target_resolution: str = None) -> str:
    cap = config.narration.get("caption", {})
    adder = SubtitleAdder(
        caption_font_size=cap.get("font_size", 42),
        caption_font_color=cap.get("font_color", "white"),
        caption_y_position=cap.get("y_position", 0.80),
        font_path=config.subtitle.get("font_path"),
        aspect_ratio=aspect_ratio or "9:16",
        target_resolution=target_resolution or "1080x1920",
        author_name=config.author.get("name") if config.author.get("enabled", True) else None,
        author_font_size=config.author.get("font_size", 28),
        author_font_color=config.author.get("font_color", "white"),
        desc_font_size=config.description.get("font_size", 48) if config.description.get("enabled", True) else 0,
        desc_font_color=config.description.get("font_color", "white"),
        desc_y_offset=config.description.get("y_offset", 0.4),
        ending_config=config.ending,
        copyright_config=config.copyright_notice,
        commentary_config=config.commentary,
    )
    adder.set_logger(config.get_logger("SubtitleAdder"))
    return adder.add_subtitles(video_path, key_segments_path, output_path)


if __name__ == "__main__":
    from config import load_config
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    video_path = sys.argv[2] if len(sys.argv) > 2 else "output/trimmed.mp4"
    key_segments_path = sys.argv[3] if len(sys.argv) > 3 else "output/test_key_segments.json"
    config = load_config(config_path)
    fmt = config.output_formats[0] if config.output_formats else {"aspect_ratio": "9:16", "target_resolution": "1080x1920"}
    print(add_subtitles(video_path, key_segments_path, config,
                        aspect_ratio=fmt.get("aspect_ratio"), target_resolution=fmt.get("target_resolution")))
