import time
import json
import re
import argparse
from pathlib import Path

import sys

from config import load_config
from pipeline import run_pipeline, add_step_arguments, timer, SKIP_FLAGS


def process_video(video_path: str, config, output_base: str, opts):
    """向后兼容的薄封装：委托给 pipeline.run_pipeline。"""
    return run_pipeline(video_path, config, output_base, opts)


def main():
    parser = argparse.ArgumentParser(description="AI Video Clipping Tool")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径")
    parser.add_argument("--video", help="只处理指定视频（文件名，不含路径）")
    # --force / --force-<step> / --skip-<step> 由步骤注册表统一生成（含新增 render/audio_mix）
    add_step_arguments(parser)
    parser.add_argument("--cut-style", default=None,
                        help="picture_led 选材方案: bridge(高潮+桥接) | skeleton(主线骨架) | pure(纯高潮)")
    parser.add_argument("--mode", default=None,
                        help="解说模式: picture_led(画面主导+旁白) | narration_led(旁白主导) | remix(原声混剪+影评评论)")
    parser.add_argument("--overrides", default=None,
                        help="JSON 文件路径，深合并进 config（webui/脚本传运行期配置覆盖用）")
    opts = parser.parse_args()

    start_time = time.time()

    print("=" * 60)
    print("Video Editor - AI-powered Video Clipping Tool")
    print("=" * 60)

    config = load_config(opts.config)
    if opts.overrides:
        with open(opts.overrides, "r", encoding="utf-8") as f:
            config.deep_update(json.load(f))
        print(f"  Overrides applied from: {opts.overrides}")
    if opts.cut_style:
        config.override("narration.explain.cut_style", opts.cut_style)
    if opts.mode:
        config.override("narration.explain.mode", opts.mode)
    logger = config.get_logger("Main")

    input_dir = Path(config.video.get("input_dir", "input"))
    output_base = "output"

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        print(f"[Error] Input directory not found: {input_dir}")
        return False

    video_exts = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")

    # 自然排序：让 1,2,…,9,10,11 按数值顺序加载（普通字典序会变成 1,10,11,2…），
    # 整部剧按集数正确顺序处理，连续性上下文才能正确累积。
    def _natural_key(p):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', p.stem)]

    all_videos = sorted(
        [f for f in input_dir.iterdir() if f.suffix.lower() in video_exts],
        key=_natural_key,
    )

    if opts.video:
        # 精确匹配优先：--video 1 只命中 1.mp4，而不是 1/10/11/12.mp4
        target = Path(opts.video).stem
        video_files = [v for v in all_videos if v.stem == target]
        if not video_files:
            video_files = [v for v in all_videos if opts.video in v.name]
        if not video_files:
            print(f"[Error] No video matching '{opts.video}' found")
            return False
    else:
        video_files = all_videos

    if not video_files:
        print(f"[Error] No video files found in {input_dir}")
        return False

    print(f"\n{len(video_files)} video(s) to process")
    if opts.force:
        print("  --force: all steps will be re-run")
    else:
        skips = [s.replace("skip_", "") for s in SKIP_FLAGS if getattr(opts, s, False)]
        if skips:
            print(f"  Skipped steps: {', '.join(skips)}")

    success_count = 0
    for vf in video_files:
        try:
            process_video(str(vf), config, output_base, opts)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to process {vf.name}: {e}")
            print(f"[Error] {vf.name} failed: {e}")

    print(f"\n{'=' * 60}")
    print(f"Done: {success_count}/{len(video_files)} videos, {timer(start_time)}")
    print(f"Output: {output_base}/")
    for vf in video_files:
        print(f"  - {output_base}/{vf.stem}/")
    print(f"{'=' * 60}")

    return success_count > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)