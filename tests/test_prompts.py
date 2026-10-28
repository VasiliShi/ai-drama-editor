"""锁定 prompt 模板的占位契约：改模板漏/多变量能被立刻发现。"""
from string import Template

import pytest

from prompts import EXPLAIN_ANALYSIS, NARRATION_EXPLAIN, COMMENTARY

ANALYSIS_KEYS = ["background_section", "speaker_note", "feature_line", "feedback_section",
                 "coverage_rule", "beat_rule", "remix_rule", "example", "content"]
NARRATION_KEYS = ["video_description", "central_question", "series_name", "episode_num",
                  "bg_block", "series_context", "speaker_hint", "beats_block", "style_desc",
                  "recap_rule", "char_budget_rule", "bridge_rule", "n_segments"]
COMMENTARY_KEYS = ["background_section", "synopsis_section", "scene_section",
                   "video_description", "segment_summary",
                   "previous_section", "content", "count", "explain_count",
                   "review_count", "emotion_count", "max_length"]


def _identifiers(tmpl: Template):
    return {m.group("named") or m.group("braced")
            for m in tmpl.pattern.finditer(tmpl.template)
            if m.group("named") or m.group("braced")}


def test_analysis_template_keys_exact():
    assert _identifiers(EXPLAIN_ANALYSIS) == set(ANALYSIS_KEYS)


def test_narration_template_keys_exact():
    assert _identifiers(NARRATION_EXPLAIN) == set(NARRATION_KEYS)


def test_analysis_substitute_no_leak():
    out = EXPLAIN_ANALYSIS.substitute({k: f"<{k}>" for k in ANALYSIS_KEYS})
    assert "$" not in out and "字幕内容" in out


def test_narration_substitute_no_leak():
    out = NARRATION_EXPLAIN.substitute({k: f"<{k}>" for k in NARRATION_KEYS})
    assert "$" not in out and "追更钩子" in out


def test_commentary_format_preserves_json_braces():
    out = COMMENTARY.format(**{k: (2 if "count" in k or k == "max_length" else f"<{k}>")
                               for k in COMMENTARY_KEYS})
    # JSON 示例的花括号必须保留，占位符必须全部替换
    assert '"text"' in out and "{background_section}" not in out


def test_commentary_missing_key_raises():
    with pytest.raises(KeyError):
        COMMENTARY.format(background_section="x")  # 缺其它键应报错，暴露契约漂移
