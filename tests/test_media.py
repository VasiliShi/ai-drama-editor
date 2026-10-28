import shutil

import pytest

from media import run_ffmpeg, get_duration, get_resolution


pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                reason="ffmpeg not available")


def _make_silence(path, seconds=1.0):
    run_ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono",
                "-t", str(seconds), str(path)], check=True)


def _make_video(path, seconds=1.0, size="320x240"):
    run_ffmpeg(["-f", "lavfi", "-i", f"color=c=black:s={size}:d={seconds}",
                str(path)], check=True)


def test_get_duration_on_silence(tmp_path):
    wav = tmp_path / "s.wav"
    _make_silence(wav, 1.5)
    assert abs(get_duration(str(wav)) - 1.5) < 0.15


def test_get_duration_missing_returns_zero(tmp_path):
    assert get_duration(str(tmp_path / "nope.wav")) == 0.0


def test_get_resolution(tmp_path):
    mp4 = tmp_path / "v.mp4"
    _make_video(mp4, 0.5, "320x240")
    assert get_resolution(str(mp4)) == (320, 240)


def test_get_resolution_missing_returns_none(tmp_path):
    assert get_resolution(str(tmp_path / "nope.mp4")) is None
