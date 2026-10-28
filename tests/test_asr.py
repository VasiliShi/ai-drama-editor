from asr import ASR


def _asr(**kw):
    return ASR(diarize=True, **kw)


def test_split_by_speaker_breaks_a_multi_speaker_segment():
    """Whisper 的段按停顿切，一段常跨好几个人；whisperx 只记多数派 speaker。

    不拆的话，声纹拿到的是一段混合音——这是声纹认不准的源头。
    """
    seg = {"start": 0.0, "end": 6.0, "speaker": "A", "text": "甲说乙说",
           "words": [
               {"word": "今年", "start": 0.0, "end": 0.6, "speaker": "A"},
               {"word": "腊月", "start": 0.6, "end": 1.2, "speaker": "A"},
               {"word": "回大人", "start": 3.0, "end": 3.8, "speaker": "B"},
               {"word": "臣不知", "start": 3.8, "end": 4.5, "speaker": "B"},
           ]}

    parts = _asr()._split_by_speaker(seg)

    assert [p["speaker"] for p in parts] == ["A", "B"]
    assert (parts[0]["start"], parts[0]["end"]) == (0.0, 1.2)
    assert (parts[1]["start"], parts[1]["end"]) == (3.0, 4.5)
    assert parts[0]["text"] == "今年腊月" and parts[1]["text"] == "回大人臣不知"


def test_split_by_speaker_breaks_on_long_pause_within_one_speaker():
    seg = {"start": 0.0, "end": 20.0, "speaker": "A", "text": "",
           "words": [
               {"word": "前半句", "start": 0.0, "end": 1.0, "speaker": "A"},
               {"word": "后半句", "start": 15.0, "end": 16.0, "speaker": "A"},
           ]}

    parts = _asr(speaker_gap=1.0)._split_by_speaker(seg)

    assert len(parts) == 2      # 同一人但隔了 14 秒，不该串成一段


def test_split_by_speaker_folds_unaligned_words_into_previous_run():
    seg = {"start": 0.0, "end": 3.0, "speaker": "A", "text": "",
           "words": [
               {"word": "对齐上", "start": 0.0, "end": 1.0, "speaker": "A"},
               {"word": "没对上", "start": None, "end": None, "speaker": "A"},
           ]}

    parts = _asr()._split_by_speaker(seg)

    assert len(parts) == 1
    assert parts[0]["text"] == "对齐上没对上"


def test_split_by_speaker_returns_empty_without_word_alignment():
    """对齐失败时没有 words，交给调用方按整段落盘。"""
    assert _asr()._split_by_speaker({"start": 0.0, "end": 3.0, "text": "x"}) == []
    assert _asr(split_by_speaker=False)._split_by_speaker(
        {"words": [{"word": "a", "start": 0.0, "end": 1.0, "speaker": "A"}]}) == []
