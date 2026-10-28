import json
from pathlib import Path

from media import get_duration, run_ffmpeg


class AudioMixer:
    """电影解说混音：原声全程压低保留环境声，连续解说叠在上面主导，可选 BGM 垫底。"""

    def __init__(self, narration_volume: float = 1.5,
                 original_volume: float = 0.08,
                 transition_duration: float = 0.0,
                 bgm_config: dict = None,
                 duck_narration_only: bool = False,
                 original_volume_full: float = 1.0,
                 duck_ramp: float = 0.3):
        self.narration_volume = narration_volume
        self.original_volume = original_volume
        self.transition_duration = transition_duration
        self.bgm_config = bgm_config or {}
        # duck_narration_only: 只在解说覆盖的时间窗内压低原声，其余时段恢复到
        # original_volume_full（picture_led 用：画面长、解说短，避免无解说时原声也被压死）
        self.duck_narration_only = duck_narration_only
        self.original_volume_full = original_volume_full
        self.duck_ramp = duck_ramp
        self.logger = None

    def set_logger(self, logger):
        self.logger = logger

    def _duck_expr(self, nar_inputs: list) -> str:
        """构造原声音量随时间变化的表达式：解说窗内=original_volume，窗外=original_volume_full。
        每个解说窗 [s,e] 用梯形包络（前后各 ramp 秒线性过渡）避免音量突变爆音。
        gain(t) = full - (full-duck) * max_i(trap_i(t))"""
        full = self.original_volume_full
        duck = self.original_volume
        r = max(self.duck_ramp, 0.01)
        traps = []
        for _audio, start, dur in nar_inputs:
            s = float(start)
            e = float(start) + float(dur or 0)
            # 上升沿 (s-r→s)、平台 (s→e)、下降沿 (e→e+r)
            traps.append(f"clip(min((t-{s - r:.3f})/{r:.3f},({e + r:.3f}-t)/{r:.3f}),0,1)")
        if not traps:
            return f"{full}"
        expr = traps[0]
        for t in traps[1:]:
            expr = f"max({expr},{t})"
        return f"{full}-{full - duck:.3f}*({expr})"

    def mix(self, video_path: str, key_segments_path: str, output_dir: str) -> str:
        with open(key_segments_path, "r", encoding="utf-8") as f:
            key_data = json.load(f)

        segments = key_data.get("segments", [])
        if not segments:
            if self.logger:
                self.logger.warning("No segments for audio mixing")
            return video_path

        td = self.transition_duration
        durations = [seg["end_time"] - seg["start_time"] for seg in segments]
        trimmed_starts = []
        cum = 0.0
        for i in range(len(segments)):
            trimmed_starts.append(cum)
            cum += durations[i] + td

        nar_inputs = []
        for i, seg in enumerate(segments):
            nar = seg.get("narration")
            if nar and nar.get("audio") and Path(nar["audio"]).exists():
                nar_inputs.append((nar["audio"], trimmed_starts[i], nar.get("duration", 0)))
        has_narration = bool(nar_inputs)

        bgm_enabled = self.bgm_config.get("enabled", False) and bool(self.bgm_config.get("file", ""))
        if not has_narration and not bgm_enabled:
            if self.logger:
                self.logger.info("No narration or BGM to mix")
            return video_path

        video_dur = get_duration(video_path)
        if video_dur <= 0:
            video_dur = sum(durations) + (len(segments) - 1) * td

        filter_parts = []
        mix_inputs = []

        if has_narration:
            if self.duck_narration_only:
                # 仅解说时间窗内压低，其余时段恢复正常音量（梯形过渡防爆音）
                filter_parts.append(
                    f"[0:a]volume=eval=frame:volume='{self._duck_expr(nar_inputs)}'[a_adj]"
                )
            else:
                # 原声全程压低保留环境声
                filter_parts.append(f"[0:a]volume={self.original_volume}[a_adj]")
            mix_inputs.append("[a_adj]")
            for idx, (audio, start, dur) in enumerate(nar_inputs):
                delay_ms = int(start * 1000)
                label = f"n{idx}"
                filter_parts.append(
                    f"[{idx+1}:a]adelay={delay_ms}|{delay_ms},volume={self.narration_volume}[{label}]"
                )
                mix_inputs.append(f"[{label}]")
        else:
            filter_parts.append("[0:a]volume=1.0[a_orig]")
            mix_inputs.append("[a_orig]")

        bgm_file = self.bgm_config.get("file", "")
        if bgm_enabled and bgm_file and Path(bgm_file).exists():
            bgm_volume = self.bgm_config.get("volume", 0.15)
            fade_in = self.bgm_config.get("fade_in", 2.0)
            fade_out = self.bgm_config.get("fade_out", 3.0)
            fade_out_start = max(0, video_dur - fade_out)
            bgm_input_idx = len(nar_inputs) + 1 if has_narration else 1
            filter_parts.append(
                f"[{bgm_input_idx}:a]aloop=loop=-1:size=2e+09,atrim=0:{video_dur:.3f},"
                f"afade=t=in:d={fade_in},afade=t=out:st={fade_out_start:.3f}:d={fade_out},"
                f"volume={bgm_volume}[bgm_adj]"
            )
            mix_inputs.append("[bgm_adj]")

        filter_parts.append(
            f"{''.join(mix_inputs)}amix=inputs={len(mix_inputs)}:duration=first:normalize=0[out]"
        )
        filter_complex = ";".join(filter_parts)

        cmd_args = ["-i", str(video_path)]
        if has_narration:
            for audio, start, dur in nar_inputs:
                cmd_args.extend(["-i", audio])
        if bgm_enabled:
            cmd_args.extend(["-i", bgm_file])

        output = Path(output_dir) / "trimmed.mp4"
        mixed_output = Path(output_dir) / "trimmed_narrated.mp4"

        if self.logger:
            parts_desc = []
            if has_narration:
                parts_desc.append(f"{len(nar_inputs)} narration tracks (original@{self.original_volume})")
            if bgm_enabled:
                parts_desc.append(f"BGM: {Path(bgm_file).name}")
            self.logger.info(f"Mixing: {', '.join(parts_desc)}")

        result = run_ffmpeg([
            *cmd_args,
            "-filter_complex", filter_complex,
            "-map", "0:v", "-map", "[out]",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            str(mixed_output)
        ])
        if result.returncode != 0:
            if self.logger:
                self.logger.error(f"Audio mix failed: {result.stderr[:500]}")
            return str(video_path)

        if output.exists():
            output.unlink()
        mixed_output.rename(output)
        if self.logger:
            self.logger.info(f"Audio mix complete: {output}")
        return str(output)


def mix_audio(video_path: str, key_segments_path: str, config, output_dir: str) -> str:
    nar_config = config.narration
    mixing = nar_config.get("mixing", {})
    mode = str(nar_config.get("explain", {}).get("mode", "narration_led")).strip().lower()
    mixer = AudioMixer(
        narration_volume=mixing.get("narration_volume", 1.5),
        original_volume=mixing.get("explain_original_volume", mixing.get("original_volume", 0.08)),
        transition_duration=0.0,
        bgm_config=config.bgm,
        # picture_led：画面长、解说短 → 只在解说窗内压低，窗外恢复正常音量
        duck_narration_only=(mode == "picture_led"),
        original_volume_full=mixing.get("original_volume_recover", 1.0),
        duck_ramp=mixing.get("duck_ramp", 0.3),
    )
    mixer.set_logger(config.get_logger("AudioMixer"))
    return mixer.mix(video_path, key_segments_path, output_dir)


if __name__ == "__main__":
    from config import load_config
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    video_path = sys.argv[2] if len(sys.argv) > 2 else "output/trimmed.mp4"
    key_segments_path = sys.argv[3] if len(sys.argv) > 3 else "output/test_key_segments.json"
    config = load_config(config_path)
    mix_audio(video_path, key_segments_path, config, "output")
