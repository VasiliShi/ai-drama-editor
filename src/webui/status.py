"""每集产物状态扫描：让用户在页面上看到每一集处于哪个阶段。"""
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EP_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv")


def input_videos():
    d = REPO_ROOT / "input"
    if not d.exists():
        return []
    return sorted([f.name for f in d.iterdir() if f.suffix.lower() in EP_EXTS],
                  key=lambda n: (len(n), n))


def _mtime(p: Path) -> str:
    try:
        return time.strftime("%m-%d %H:%M", time.localtime(p.stat().st_mtime))
    except OSError:
        return ""


def _has_narration_audio(key_path: Path) -> bool:
    try:
        d = json.loads(key_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    for s in d.get("segments", []):
        n = s.get("narration")
        if n and n.get("audio") and (REPO_ROOT / n["audio"]).exists():
            return True
    return False


def _has_commentaries(key_path: Path) -> bool:
    try:
        d = json.loads(key_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return any(s.get("commentaries") for s in d.get("segments", []))


def episode_rows():
    """返回每集各阶段状态，用于 gr.Dataframe。"""
    rows = []
    for name in input_videos():
        stem = Path(name).stem
        out = REPO_ROOT / "output" / stem
        transcript = out / f"{stem}_transcript.json"
        key = out / f"{stem}_key_segments.json"
        trimmed = out / "trimmed.mp4"
        finals = [f for f in out.glob("*.mp4") if f.name != "trimmed.mp4"] if out.exists() else []

        def mark(p: Path):
            return f"✅ {_mtime(p)}" if p.exists() else "—"

        nar_com = "—"
        if key.exists():
            if _has_narration_audio(key):
                nar_com = f"✅解说 {_mtime(key)}"
            elif _has_commentaries(key):
                nar_com = f"✅评论 {_mtime(key)}"
        rows.append([
            stem,
            mark(transcript),
            mark(key),
            nar_com,
            mark(trimmed),
            f"✅ ×{len(finals)}" if finals else "—",
        ])
    return rows


STATUS_HEADERS = ["集", "转写", "节拍", "解说/评论", "trimmed", "成片"]


def all_output_mp4s():
    base = REPO_ROOT / "output"
    if not base.exists():
        return []
    outs = []
    for d in sorted(base.iterdir()):
        if d.is_dir():
            outs.extend(str(f.relative_to(base)) for f in sorted(d.rglob("*.mp4")))
    return outs
