import json
import logging

import speaker_naming as sn


class _Config:
    def __init__(self, naming=None, background="", language="zh"):
        self.naming = naming or {}
        self.video = {"series_background": background}
        self.asr = {"language": language}
        self.llm = {"base_url": "http://x", "api_key": "k", "model_name": "m"}

    @staticmethod
    def get_logger(_name):
        return logging.getLogger("test.naming")


def _transcript(tmp_path, segments):
    p = tmp_path / "7_transcript.json"
    p.write_text(json.dumps({"segments": segments}, ensure_ascii=False), encoding="utf-8")
    return p


def _stub_llm(monkeypatch, reply):
    class _C:
        def chat_json(self, _prompt):
            return reply
    monkeypatch.setattr(sn, "_client", lambda *a, **k: _C())


def test_lang_name_is_derived_not_hardcoded():
    assert sn.lang_name("zh") == "中文"
    assert sn.lang_name("ko") == "韩语"
    assert sn.lang_name(None) == "原声"
    assert sn.lang_name("xx") == "xx"      # 未知代码原样透传，别硬塞一个错语种


def test_build_dialogue_keeps_chronological_order():
    """身份线索藏在互动里(A 叫 B「大人」、B 回「臣…」)，按说话人分组会拆散这些配对。"""
    segs = [{"start": 5, "speaker": "SPEAKER_01", "text": "臣不敢"},
            {"start": 1, "speaker": "SPEAKER_00", "text": "海大人"},
            {"start": 9, "speaker": "SPEAKER_00", "text": "退下吧"}]

    out = sn.build_dialogue(segs).splitlines()

    assert [l.split(": ")[1] for l in out] == ["臣不敢", "海大人", "退下吧"]
    assert out[0].startswith("[5s] SPEAKER_01")


def test_build_dialogue_samples_blocks_across_the_episode_when_over_budget():
    """超预算时取连续片段铺满整集：既覆盖全集，又不切断相邻对白的呼应关系。"""
    segs = [{"start": i, "speaker": f"SPEAKER_{i % 4:02d}", "text": "x" * 40}
            for i in range(400)]

    out = sn.build_dialogue(segs, max_chars=1200, n_blocks=4)
    kept = [int(l.split("s]")[0][1:]) for l in out.splitlines()]

    assert len(out) <= 1200
    # 覆盖到全集尾部，不是只取开头
    assert max(kept) > 250 and min(kept) < 50
    # 每个说话人都露面（等距抽行会和 4 人轮换共振，只抽到一个人）
    assert {f"SPEAKER_{i:02d}" for i in range(4)} <= set(
        l.split(": ")[0].split()[-1] for l in out.splitlines())
    # 块内保持连续，相邻呼应没被切断
    assert any(b - a == 1 for a, b in zip(kept, kept[1:]))


def test_name_speakers_applies_mapping_and_keeps_raw_label(tmp_path, monkeypatch):
    cfg = _Config(background="角色：海瑞、嘉靖帝")
    monkeypatch.setattr(sn, "load_roster", lambda *a, **k: ["海瑞", "嘉靖帝"])
    _stub_llm(monkeypatch, [
        {"speaker": "SPEAKER_00", "name": "嘉靖帝", "evidence": "自称朕"},
        {"speaker": "SPEAKER_01", "name": "海瑞", "evidence": "被称海大人"}])
    p = _transcript(tmp_path, [
        {"start": 0, "end": 3, "speaker": "SPEAKER_00", "text": "朕不回宫了"},
        {"start": 3, "end": 6, "speaker": "SPEAKER_01", "text": "臣领旨"}])

    assert sn.name_speakers(str(p), cfg) is True

    segs = json.loads(p.read_text(encoding="utf-8"))["segments"]
    assert [s["speaker"] for s in segs] == ["嘉靖帝", "海瑞"]
    assert [s["speaker_raw"] for s in segs] == ["SPEAKER_00", "SPEAKER_01"]


def test_name_speakers_rejects_names_outside_the_roster(tmp_path, monkeypatch):
    """LLM 发明的名字一律丢弃——宁可保持匿名，也不要一个查不到的角色名。"""
    cfg = _Config(background="角色：海瑞")
    monkeypatch.setattr(sn, "load_roster", lambda *a, **k: ["海瑞"])
    _stub_llm(monkeypatch, [{"speaker": "SPEAKER_00", "name": "张三丰"}])
    p = _transcript(tmp_path, [
        {"start": 0, "end": 3, "speaker": "SPEAKER_00", "text": "在下张三丰"}])

    assert sn.name_speakers(str(p), cfg) is False
    assert json.loads(p.read_text(encoding="utf-8"))["segments"][0]["speaker"] == "SPEAKER_00"


def test_name_speakers_leaves_unsure_speakers_anonymous(tmp_path, monkeypatch):
    """空名字=没把握。保留匿名远好过安错名字，错名会让下游把台词算到别人头上。"""
    cfg = _Config(background="角色：海瑞")
    monkeypatch.setattr(sn, "load_roster", lambda *a, **k: ["海瑞"])
    _stub_llm(monkeypatch, [{"speaker": "SPEAKER_00", "name": "海瑞"},
                            {"speaker": "SPEAKER_01", "name": ""}])
    p = _transcript(tmp_path, [
        {"start": 0, "end": 3, "speaker": "SPEAKER_00", "text": "臣海瑞"},
        {"start": 3, "end": 6, "speaker": "SPEAKER_01", "text": "嗯"}])

    sn.name_speakers(str(p), cfg)

    segs = json.loads(p.read_text(encoding="utf-8"))["segments"]
    assert segs[0]["speaker"] == "海瑞"
    assert segs[1]["speaker"] == "SPEAKER_01" and "speaker_raw" not in segs[1]


def test_name_speakers_skips_when_no_roster(tmp_path, monkeypatch):
    cfg = _Config(background="")
    monkeypatch.setattr(sn, "load_roster", lambda *a, **k: [])
    p = _transcript(tmp_path, [
        {"start": 0, "end": 3, "speaker": "SPEAKER_00", "text": "x"}])

    assert sn.name_speakers(str(p), cfg) is False


def test_load_roster_caches_and_reuses_until_background_changes(tmp_path, monkeypatch):
    calls = []

    class _C:
        def chat_json(self, _prompt):
            calls.append(1)
            return ["海瑞", "严嵩", "海瑞"]     # 含重复，应去重
    monkeypatch.setattr(sn, "_client", lambda *a, **k: _C())
    cache = tmp_path / "roster.json"
    cfg = _Config({"roster_file": str(cache)}, background="角色表 A")

    assert sn.load_roster(cfg) == ["海瑞", "严嵩"]
    assert sn.load_roster(cfg) == ["海瑞", "严嵩"]
    assert len(calls) == 1                      # 第二次命中缓存

    cfg.video["series_background"] = "角色表 B"  # 背景变了 → 重抽
    sn.load_roster(cfg)
    assert len(calls) == 2
