from pipeline import beats_fingerprint, select_publish_title, _clean_publish_text


def _kd(beats):
    return {"segments": [{"start_time": s, "end_time": e} for s, e in beats]}


def test_fingerprint_stable_for_same_beats():
    kd = _kd([(1.0, 5.0), (10.0, 20.0)])
    assert beats_fingerprint(kd, "2.mp4") == beats_fingerprint(kd, "2.mp4")


def test_fingerprint_changes_when_beat_moves():
    a = beats_fingerprint(_kd([(1.0, 5.0)]), "2.mp4")
    b = beats_fingerprint(_kd([(1.0, 6.0)]), "2.mp4")
    assert a != b


def test_fingerprint_changes_with_video():
    kd = _kd([(1.0, 5.0)])
    assert beats_fingerprint(kd, "1.mp4") != beats_fingerprint(kd, "2.mp4")


def test_fingerprint_ignores_non_beat_fields():
    # 评论/摘要变化不应触发重切
    a = {"segments": [{"start_time": 1, "end_time": 5, "summary": "x"}]}
    b = {"segments": [{"start_time": 1, "end_time": 5, "summary": "y",
                       "commentaries": [{"text": "hi"}]}]}
    assert beats_fingerprint(a, "2.mp4") == beats_fingerprint(b, "2.mp4")


def test_fingerprint_tracks_src_end():
    a = {"segments": [{"start_time": 1, "end_time": 5, "src_end": 8}]}
    b = {"segments": [{"start_time": 1, "end_time": 5, "src_end": 9}]}
    assert beats_fingerprint(a, "2.mp4") != beats_fingerprint(b, "2.mp4")


def test_publish_title_prefers_cover_title():
    d = {"cover_title": "封面标题", "hook_title": "钩子", "video_description": "描述"}
    assert select_publish_title(d) == "封面标题"


def test_publish_title_falls_through_empties():
    d = {"cover_title": "", "hook_title": "  ", "video_description": "有效描述"}
    assert select_publish_title(d) == "有效描述"


def test_publish_title_default():
    assert select_publish_title({}) == "精彩片段"


def test_clean_publish_text_strips_edges_and_whitespace():
    # 去首尾空白/标点、折叠内部空白；内部标点保留
    assert _clean_publish_text("  你好 世界！ ") == "你好世界"
    assert _clean_publish_text("，标题。") == "标题"
    assert _clean_publish_text("abcdefghijklmnopqrstuvwxyz", max_len=5) == "abcde"
