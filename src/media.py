"""统一 ffmpeg/ffprobe 封装：全项目唯一的 subprocess 出口。

filter 字符串拼装仍归各模块（cutter/audio_mixer/subtitle_adder），
这里只收敛命令执行、探测类查询和编码器选择的重复实现。
"""
import subprocess

from gpu import get_video_encoder  # noqa: F401  统一从 media 导入


class FFmpegError(RuntimeError):
    pass


def run_ffmpeg(args, check: bool = False) -> subprocess.CompletedProcess:
    """跑 ffmpeg（自动加 -hide_banner -nostdin -y），返回 CompletedProcess。

    check=True 时非零退出抛 FFmpegError（截取 stderr 尾部）；
    check=False 时由调用方自行按 returncode 处理（warn/降级/raise 语义各不同）。
    """
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-y", *(str(a) for a in args)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise FFmpegError(f"ffmpeg failed: {result.stderr[-800:]}")
    return result


def run_ffprobe(args) -> subprocess.CompletedProcess:
    cmd = ["ffprobe", "-v", "error", *(str(a) for a in args)]
    return subprocess.run(cmd, capture_output=True, text=True)


def get_duration(path: str) -> float:
    """媒体文件时长（秒）；探测失败返回 0.0。"""
    result = run_ffprobe(["-show_entries", "format=duration",
                          "-of", "default=noprint_wrappers=1:nokey=1", path])
    if result.returncode == 0:
        try:
            return float(result.stdout.strip())
        except ValueError:
            pass
    return 0.0


def get_resolution(path: str):
    """视频分辨率 (width, height)；探测失败返回 None。"""
    result = run_ffprobe(["-select_streams", "v:0",
                          "-show_entries", "stream=width,height",
                          "-of", "csv=s=x:p=0", path])
    if result.returncode == 0:
        parts = result.stdout.strip().split("x")
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                pass
    return None
