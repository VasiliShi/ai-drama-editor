
import httpx
import pytest

import llm
from llm import LLMClient, ContentFilterError


def _client_with(responses):
    """responses: list of (status, json_body)。按调用顺序逐个返回。"""
    calls = {"n": 0}

    def handler(request):
        i = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        status, body = responses[i]
        return httpx.Response(status, json=body)

    c = LLMClient("http://x/api/v1", "k", "m")
    c._client = httpx.Client(transport=httpx.MockTransport(handler))
    return c, calls


def _msg(content, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": content}}]}


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(llm.time, "sleep", lambda *_: None)


def test_chat_success():
    c, calls = _client_with([(200, _msg("hello"))])
    assert c.chat("hi") == "hello"
    assert calls["n"] == 1


def test_chat_retries_then_succeeds():
    c, calls = _client_with([(500, {}), (500, {}), (200, _msg("ok"))])
    assert c.chat("hi", retries=3) == "ok"
    assert calls["n"] == 3


def test_chat_exhausts_retries_raises():
    c, calls = _client_with([(500, {})])
    with pytest.raises(Exception):
        c.chat("hi", retries=2)
    assert calls["n"] == 2


def test_content_filter_raises_without_retry():
    c, calls = _client_with([(200, _msg("拒绝", finish="content_filter"))])
    with pytest.raises(ContentFilterError):
        c.chat("hi", retries=3)
    assert calls["n"] == 1  # 不重试


def test_empty_content_retried():
    c, calls = _client_with([(200, _msg("")), (200, _msg("x"))])
    assert c.chat("hi", retries=3) == "x"
    assert calls["n"] == 2


def test_chat_json_parses_and_repairs():
    # 带尾随逗号的坏 JSON，json_repair 应能修复
    c, _ = _client_with([(200, _msg('{"a": 1, "b": [2,3,],}'))])
    assert c.chat_json("hi") == {"a": 1, "b": [2, 3]}


def test_temperature_is_omitted_by_default_and_sent_when_set():
    """创作类调用沿用服务端默认温度；事实抽取类必须能显式钉成 0。"""
    from llm import LLMClient
    import inspect

    src = inspect.getsource(LLMClient.chat)
    assert '"temperature"' in src

    assert LLMClient("u", "k", "m").temperature is None
    assert LLMClient("u", "k", "m", temperature=0).temperature == 0


def test_speaker_naming_pins_temperature_to_zero():
    """命名/角色表抽取必须确定性——默认温度下实测会把相近角色对调。"""
    import speaker_naming as sn

    class _Cfg:
        llm = {"base_url": "u", "api_key": "k", "model_name": "m"}

    assert sn._client(_Cfg(), None).temperature == 0
