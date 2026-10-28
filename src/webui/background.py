"""剧情背景文件的读写：让用户在页面上编辑 series_background_file，不必手改文件。

背景文件是整部剧的 markdown（人物介绍段 + `### 第N集` 分集剧情段），
analyzer / narration 通过 episode_utils 按标题解析、按集抽取。编辑保持整篇 markdown。
"""
import shutil
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPO_ROOT / "config.yaml"


def _raw_config() -> dict:
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except OSError:
        return {}


def bg_path() -> Path:
    """当前配置指向的背景文件绝对路径；未配置时按剧名给个默认路径。"""
    cfg = _raw_config()
    video = cfg.get("video", {})
    rel = video.get("series_background_file")
    if not rel:
        series = cfg.get("narration", {}).get("continuity", {}).get("series_name", "series")
        rel = f"data/{series}_background.md"
    p = Path(rel)
    return p if p.is_absolute() else (REPO_ROOT / rel)


def load_background():
    """返回 (content, 提示语)。文件不存在返回空内容 + 说明。"""
    p = bg_path()
    if p.exists():
        return p.read_text(encoding="utf-8"), f"📄 {p}（{p.stat().st_size} 字节）"
    return "", f"⚠ 文件不存在，将新建：{p}"


def save_background(text: str) -> str:
    p = bg_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        shutil.copyfile(p, str(p) + ".bak")  # 覆盖前留底
    p.write_text(text or "", encoding="utf-8")
    return f"✅ 已保存到 {p}（{len(text or '')} 字，备份 .bak）。重跑分析/解说即生效。"


def preview_episode(text: str, episode_num) -> str:
    """预览某集实际会喂给 LLM 的背景（人物介绍 + 该集剧情），所见即所得。"""
    try:
        ep = int(episode_num)
    except (TypeError, ValueError):
        return "（请输入有效集数）"
    from episode_utils import extract_episode_background
    out = extract_episode_background(text or "", ep)
    if not out.strip():
        return f"（第 {ep} 集：未匹配到分集段，实际会回退为背景开头截断）"
    return out
