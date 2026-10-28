import subprocess
import platform


_ENCODER_CACHE = None


def _encoder_works(encoder: str, extra_args: list) -> bool:
    """Verify a hardware encoder actually produces output."""
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=320x240:d=1",
             "-c:v", encoder, *extra_args, "-f", "null", "-"],
            capture_output=True, text=True, timeout=10
        )
        err = r.stderr or ""
        return "Error" not in err and r.returncode == 0 and "0 frames" not in err
    except Exception:
        return False


def get_video_encoder(quality: int = 23):
    """Detect best available video encoder.

    Args:
        quality: CRF/CQ equivalent quality value (lower=better, default 23)

    Returns:
        tuple: (encoder_name, encoder_args_list)
        Example: ("h264_nvenc", ["-preset", "p1", "-cq", "23", "-b:v", "0"])
    """
    global _ENCODER_CACHE
    if _ENCODER_CACHE is not None:
        enc, args_template = _ENCODER_CACHE
        return enc, [a.format(q=quality) for a in args_template]

    enc, args_template = _detect()
    _ENCODER_CACHE = (enc, args_template)
    return enc, [a.format(q=quality) for a in args_template]


def _detect():
    """Detect GPU encoder. Returns (encoder_name, arg_templates_with_{q})."""

    # 1. NVIDIA NVENC
    try:
        r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            r2 = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10)
            if "h264_nvenc" in r2.stdout and _encoder_works("h264_nvenc", ["-preset", "p1"]):
                return ("h264_nvenc", ["-preset", "p1", "-cq", "{q}", "-b:v", "0"])
    except Exception:
        pass

    # 2. Intel QSV
    try:
        r = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10)
        if "h264_qsv" in r.stdout and _encoder_works("h264_qsv", []):
            return ("h264_qsv", ["-global_quality", "{q}"])
    except Exception:
        pass

    # 3. Apple VideoToolbox
    if platform.system() == "Darwin":
        try:
            r = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10)
            if "h264_videotoolbar" in r.stdout and _encoder_works("h264_videotoolbar", []):
                return ("h264_videotoolbar", ["-quality", "85"])
        except Exception:
            pass

    # 4. AMD AMF
    try:
        r = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=10)
        if "h264_amf" in r.stdout and _encoder_works("h264_amf", ["-quality", "speed"]):
            return ("h264_amf", ["-quality", "speed", "-qp", "{q}"])
    except Exception:
        pass

    # Fallback to software
    return ("libx264", ["-preset", "fast", "-crf", "{q}"])
