import json
from pathlib import Path
from media import get_video_encoder, get_duration, run_ffmpeg


class VideoCutter:
    """电影解说切条：按节拍区间裁出连续 B-roll 片段，直接拼接（无过渡字卡）。"""

    def __init__(self, ending_config: dict = None, audio_fade_duration: float = 0.2):
        self.ending_config = ending_config or {}
        self.audio_fade_duration = audio_fade_duration
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger

    def _plan_subclips(self, start: float, window: float, clip_dur: float):
        """在剧情窗口内规划子片段，使总时长恰好等于 clip_dur（解说音频长度）。
        窗口明显长于解说时长时，拆成 2-3 段分别取自场景前/中/后，让画面覆盖整段戏。
        返回 (sub_starts, sub_len)；单片段时列表长度为 1。"""
        if clip_dur <= 0 or window <= clip_dur * 1.2:
            return [start], clip_dur
        k = max(2, min(3, round(window / 9.0)))
        sub_len = clip_dur / k
        if window - sub_len <= 0:
            return [start], clip_dur
        step = (window - sub_len) / (k - 1)
        return [start + j * step for j in range(k)], sub_len

    def _cut_beat(self, video_path, clip_file, sub_starts, sub_len, clip_dur,
                  v_enc, v_args, af_filter) -> bool:
        """切出单个节拍片段。单片段直接裁剪；多片段用 concat 滤镜一遍编码拼好，
        节拍级淡入淡出施加在拼接后的整段音频上。总时长 == clip_dur，保证音画对齐。"""
        if len(sub_starts) == 1:
            args = ["-ss", f"{sub_starts[0]:.3f}", "-i", video_path,
                    "-t", f"{clip_dur:.3f}", "-c:v", v_enc, *v_args, "-c:a", "aac", "-ac", "2",
                    "-avoid_negative_ts", "make_zero"]
            if af_filter:
                args.extend(["-af", af_filter])
            args.append(str(clip_file))
        else:
            inputs = []
            for ss in sub_starts:
                inputs += ["-ss", f"{ss:.3f}", "-t", f"{sub_len:.3f}", "-i", video_path]
            n = len(sub_starts)
            streams = "".join(f"[{j}:v][{j}:a]" for j in range(n))
            fc = f"{streams}concat=n={n}:v=1:a=1[cv][ca]"
            amap = "[ca]"
            if af_filter:
                fc += f";[ca]{af_filter}[ao]"
                amap = "[ao]"
            args = [*inputs, "-filter_complex", fc,
                    "-map", "[cv]", "-map", amap,
                    "-c:v", v_enc, *v_args, "-c:a", "aac", "-ac", "2", str(clip_file)]
        result = run_ffmpeg(args)
        if result.returncode != 0 and self.logger:
            self.logger.error(f"FFmpeg error: {result.stderr[:500]}")
        return result.returncode == 0

    def cut_segments(self, video_path: str, key_segments_path: str, output_dir: str = "output") -> str:
        with open(key_segments_path, "r", encoding="utf-8") as f:
            key_data = json.load(f)

        segments = key_data.get("segments", [])
        video_duration = get_duration(video_path)

        valid_segments = []
        for seg in segments:
            if video_duration > 0 and seg["end_time"] > video_duration:
                seg["end_time"] = video_duration
            if seg["start_time"] < 0:
                seg["start_time"] = 0
            if seg["start_time"] >= seg["end_time"]:
                if self.logger:
                    self.logger.warning(f"Skipping invalid segment: {seg['start_time']} >= {seg['end_time']}")
                continue
            valid_segments.append(seg)
        segments = valid_segments

        if not segments:
            raise RuntimeError("No valid segments to cut")

        if self.logger:
            self.logger.info(f"Cutting {len(segments)} beats (continuous B-roll)")

        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        final_output = output_path / "trimmed.mp4"

        v_enc, v_args = get_video_encoder(quality=23)
        fade_dur = self.audio_fade_duration

        clip_files = []
        for i, seg in enumerate(segments):
            start = seg["start_time"]
            clip_dur = seg["end_time"] - start
            src_end = seg.get("src_end", seg["end_time"])
            if video_duration > 0:
                src_end = min(src_end, video_duration)
            clip_file = output_path / f"clip_{i:03d}.mp4"

            af_parts = []
            if i > 0:
                af_parts.append(f"afade=t=in:d={fade_dur}")
            if i < len(segments) - 1:
                fade_out_start = max(0, clip_dur - fade_dur)
                af_parts.append(f"afade=t=out:st={fade_out_start:.3f}:d={fade_dur}")
            af_filter = ",".join(af_parts) if af_parts else None

            # 在剧情窗口 [start, src_end] 内采样铺满 clip_dur，让画面贴合整段解说（B1）
            sub_starts, sub_len = self._plan_subclips(start, max(0.0, src_end - start), clip_dur)

            if self.logger:
                tag = f"{start:.1f}s +{clip_dur:.1f}s"
                if len(sub_starts) > 1:
                    tag += f" ×{len(sub_starts)} (窗口{src_end - start:.0f}s)"
                self.logger.info(f"Cutting beat {i+1}/{len(segments)}: {tag}")

            if not self._cut_beat(video_path, clip_file, sub_starts, sub_len, clip_dur,
                                  v_enc, v_args, af_filter):
                raise RuntimeError(f"Failed to cut beat {i+1}")
            clip_files.append(str(clip_file))

        if len(clip_files) == 1:
            Path(clip_files[0]).rename(final_output)
        else:
            concat_file = output_path / "concat.txt"
            with open(concat_file, "w", encoding="utf-8") as f:
                for cf in clip_files:
                    f.write(f"file '{Path(cf).resolve().as_posix()}'\n")
            result = run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_file),
                                 "-c", "copy", str(final_output)])
            concat_file.unlink(missing_ok=True)
            for cf in clip_files:
                Path(cf).unlink(missing_ok=True)
            if result.returncode != 0:
                if self.logger:
                    self.logger.error(f"Concat failed: {result.stderr[:500]}")
                raise RuntimeError("Failed to concatenate beats")

        if self.ending_config.get("enabled", False):
            fade_dur = self.ending_config.get("fade_duration", 1.5)
            total_dur = get_duration(str(final_output))
            if total_dur > fade_dur:
                fade_start = total_dur - fade_dur
                faded_output = output_path / "trimmed_faded.mp4"
                v_enc, v_args = get_video_encoder(quality=23)
                result = run_ffmpeg([
                    "-i", str(final_output),
                    "-vf", f"fade=type=out:duration={fade_dur}:start_time={fade_start}",
                    "-af", f"afade=type=out:duration={fade_dur}:start_time={fade_start}",
                    "-c:v", v_enc, *v_args, "-c:a", "aac", "-ac", "2", "-b:a", "128k",
                    str(faded_output)
                ])
                if result.returncode == 0:
                    Path(final_output).unlink(missing_ok=True)
                    faded_output.rename(final_output)
                    if self.logger:
                        self.logger.info(f"Ending fade-out applied: {fade_dur}s at {fade_start:.1f}s")
                elif self.logger:
                    self.logger.warning(f"Ending fade-out failed: {result.stderr[:300]}")

        if self.logger:
            self.logger.info(f"Trimmed video saved to: {final_output}")
        return str(final_output)


def cut_video(video_path: str, key_segments_path: str, config, output_dir: str = "output") -> str:
    cutter = VideoCutter(ending_config=config.ending, audio_fade_duration=0.2)
    cutter.set_logger(config.get_logger("Cutter"))
    return cutter.cut_segments(video_path, key_segments_path, output_dir)


if __name__ == "__main__":
    from config import load_config
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    video_path = sys.argv[2] if len(sys.argv) > 2 else "input/test.mp4"
    key_segments_path = sys.argv[3] if len(sys.argv) > 3 else "output/test_key_segments.json"
    config = load_config(config_path)
    cut_video(video_path, key_segments_path, config)
