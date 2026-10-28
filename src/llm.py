"""统一 LLM 客户端：代理剥离、重试退避、content_filter 显式报错、JSON 解析。

全项目唯一的 LLM 出口。四个调用方（analyzer / narration_generator /
commentary_generator / speaker_naming）共用，不要再各自手搓 httpx POST。
"""
import json
import os
import time

import httpx
from json_repair import repair_json


class ContentFilterError(RuntimeError):
    """服务端内容过滤拦截（finish_reason=content_filter）。

    返回 200 但正文是拒绝语，重试无效；建议换 LLM 端点/模型或跳过该集。
    """


def make_httpx_client(timeout: float = 300.0) -> httpx.Client:
    """构造直连的 httpx.Client：构造期临时摘掉代理环境变量。

    HF 下载等走 SOCKS 代理，但 LLM / 火山 TTS 端点要直连。
    """
    old_env = {}
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                "all_proxy", "ALL_PROXY"):
        if key in os.environ:
            old_env[key] = os.environ.pop(key)
    try:
        return httpx.Client(timeout=timeout)
    finally:
        os.environ.update(old_env)


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model_name: str,
                 timeout: float = 300.0, logger=None, temperature: float = None):
        """temperature=None 表示不下发该字段（沿用服务端默认，适合解说词/评论这类创作）。
        事实抽取类调用（说话人命名、角色表）必须显式传 0：默认温度下同一份证据
        两次运行可能给出不同答案——实测出现过把「沈一石/杨金水」对调的情况。"""
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key
        self.model_name = model_name
        self.timeout = timeout
        self.logger = logger
        self.temperature = temperature
        self._client = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = make_httpx_client(self.timeout)
        return self._client

    def _warn(self, msg: str):
        if self.logger:
            self.logger.warning(msg)

    def chat(self, prompt: str, retries: int = 3) -> str:
        """单轮对话，返回非空正文。

        - HTTP/网络错误、空正文：指数退避重试，耗尽后抛最后一次异常。
        - content_filter：立即抛 ContentFilterError，不重试（重试无效）。
        """
        payload = {"model": self.model_name,
                   "messages": [{"role": "user", "content": prompt}]}
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        last_error = None
        for attempt in range(retries):
            try:
                resp = self._get_client().post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}",
                             "Content-Type": "application/json"},
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                choice = (data.get("choices") or [{}])[0]
                if choice.get("finish_reason") == "content_filter":
                    raise ContentFilterError(
                        "LLM 内容过滤拦截（finish_reason=content_filter），"
                        "该内容被服务端拒绝，重试无效；建议换 LLM 端点/模型或跳过该集")
                text = (choice.get("message") or {}).get("content", "")
                text = (text or "").strip()
                if not text:
                    raise RuntimeError("LLM 返回空正文")
                return text
            except ContentFilterError:
                raise
            except Exception as e:
                last_error = e
                self._warn(f"LLM attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        raise last_error

    def chat_json(self, prompt: str, retries: int = 3):
        """chat + JSON 修复解析；解析失败同样计入重试。"""
        last_error = None
        for attempt in range(retries):
            try:
                text = self.chat(prompt, retries=1)
                return json.loads(repair_json(text))
            except ContentFilterError:
                raise
            except Exception as e:
                last_error = e
                self._warn(f"LLM JSON attempt {attempt + 1}/{retries} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        raise last_error
