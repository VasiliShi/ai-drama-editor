"""集数解析与「按集抽取背景介绍」工具，供 analyzer / narration 共用。

剧集背景文档（config.video.series_background）是整部剧的资料，里面含：
- 人物介绍段（## …人物…）：用于修正 ASR 人名识别错误，每集都需要；
- 分集剧情段（### 第N集：…）：当前正在剪的这一集的剧情，按集数精准抽取。
"""

import re


def parse_episode_number(name: str, fallback: int = 1) -> int:
    """从文件名/路径名解析集数。兼容 1/2/3.mp4、EP01、第1集、1集 等命名。"""
    stem = str(name)
    for suffix in ("_transcript", "_key_segments"):
        stem = stem.replace(suffix, "")
    stem = stem.strip()
    if stem.isdigit():
        return int(stem)
    for pat in (r'[Ee][Pp]?\s*(\d+)', r'第\s*(\d+)\s*[集话期]', r'(\d+)\s*集', r'^\s*(\d+)'):
        m = re.search(pat, stem)
        if m:
            return int(m.group(1))
    return fallback


def _section_after_heading(text: str, heading_match) -> str:
    """从某标题行开始，截到下一个「同级或更高级」标题之前（含其下更低级子标题）。"""
    heading = heading_match.group(0)
    level = len(heading) - len(heading.lstrip("#"))  # 标题级别 = 前导 # 数
    rest = text[heading_match.end():]
    nxt = re.search(r'(?m)^#{1,%d}\s' % level, rest)
    end = heading_match.end() + (nxt.start() if nxt else len(rest))
    return text[heading_match.start():end].strip()


def extract_episode_plot(series_background: str, episode_num: int,
                         max_chars: int = 600) -> str:
    """只抽取「第N集」的完整剧情段（不含人物介绍），用于加厚「上集回顾」。"""
    bg = (series_background or "").strip()
    if not bg:
        return ""
    m = re.search(r'(?m)^#{2,3}\s*第\s*%d\s*集[^\n]*$' % int(episode_num), bg)
    if not m:
        return ""
    return _section_after_heading(bg, m)[:max_chars]


def extract_episode_background(series_background: str, episode_num: int,
                              max_chars: int = 2600) -> str:
    """抽取「人物介绍 + 第N集剧情」。找不到分集段则回退为背景开头截断。"""
    bg = (series_background or "").strip()
    if not bg:
        return ""

    char_section = ""
    m = re.search(r'(?m)^#{2,3}[^\n]*人物[^\n]*$', bg)
    if m:
        char_section = _section_after_heading(bg, m)

    ep_section = ""
    m2 = re.search(r'(?m)^#{2,3}\s*第\s*%d\s*集[^\n]*$' % int(episode_num), bg)
    if m2:
        ep_section = _section_after_heading(bg, m2)

    parts = [p for p in (char_section, ep_section) if p]
    if not parts:
        return bg[:max_chars]
    result = "\n\n".join(parts)
    return result[:max_chars]


def load_scene_notes(key_segments_path) -> dict:
    """读取 scene_notes 步骤产物，返回 {节拍下标: 单行备注文本}；无文件返回空。
    解说与混剪评论共用：让 LLM「照着画面写」而非脑补。"""
    import json
    from pathlib import Path
    p = Path(str(key_segments_path).replace("_key_segments.json", "_scene_notes.json"))
    if not p.exists():
        return {}
    try:
        data = json.load(open(p, encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    out = {}
    for n in data.get("notes", []):
        parts = []
        if n.get("identified"):
            parts.append("在场(人脸确认):" + "、".join(str(x) for x in n["identified"]))
        if n.get("on_screen"):
            parts.append("在场:" + "、".join(str(x) for x in n["on_screen"]))
        if n.get("action"):
            parts.append("动作:" + n["action"])
        if n.get("scene"):
            parts.append("场景:" + n["scene"])
        if n.get("mood"):
            parts.append("氛围:" + n["mood"])
        if parts:
            out[int(n.get("beat", 0)) - 1] = "；".join(parts)
    return out
