"""Tests for llm_handler.LLMHandler.call_model.

These tests mock httpx so no real network request is ever made to the
configured base_url. call_model is async; rather than depend on the
pytest-asyncio plugin, each test drives the coroutine via asyncio.run().
"""
import asyncio
import json

import httpx
import pytest

from llm_handler import LLMHandler


class FakeResponse:
    """Minimal stand-in for httpx.Response used by the mocked client."""

    def __init__(self, content, status_code=200):
        self._content = content
        self.status_code = status_code
        self.text = content if isinstance(content, str) else json.dumps(content)

    def json(self):
        return {
            "choices": [{"message": {"content": self._content}}]
        }

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://testserver/v1/chat/completions")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                "error", request=request, response=response
            )


class CapturingClient:
    """Fake httpx.AsyncClient that records the kwargs passed to post()."""

    def __init__(self, response=None, raise_exc=None):
        self._response = response
        self._raise_exc = raise_exc
        self.captured_kwargs = None
        self.captured_url = None

    async def post(self, url, **kwargs):
        self.captured_url = url
        self.captured_kwargs = kwargs
        if self._raise_exc is not None:
            raise self._raise_exc
        return self._response


def _call(client, **overrides):
    handler = LLMHandler()
    params = dict(
        client=client,
        base_url="http://testserver/v1",
        model_name="test-model",
        system="You are a stance classifier.",
        messages=[{"role": "user", "content": "Some comment."}],
        semaphore=asyncio.Semaphore(1),
    )
    params.update(overrides)
    return asyncio.run(handler.call_model(**params))


def test_happy_path_returns_parsed_dict_and_response_format_is_dict():
    content = '{"stance": "support", "explanation": "x"}'
    client = CapturingClient(response=FakeResponse(content))

    result = _call(client)

    assert result == {"stance": "support", "explanation": "x"}

    # Regression guard for both bugs:
    body = client.captured_kwargs["json"]
    response_format = body["response_format"]
    assert isinstance(response_format, dict), (
        "response_format must be a dict, not a JSON-encoded string"
    )
    assert response_format["type"] == "json_schema"


def test_thinking_mode_strips_think_block():
    content = '<think>let me reason about this</think>{"stance": "oppose", "explanation": "y"}'
    client = CapturingClient(response=FakeResponse(content))

    result = _call(client, enable_thinking=True)

    assert result == {"stance": "oppose", "explanation": "y"}
    # extra_body should be present when thinking is enabled.
    assert "extra_body" in client.captured_kwargs["json"]


def test_markdown_fenced_json_is_stripped():
    content = '```json\n{"stance": "neutral", "explanation": "z"}\n```'
    client = CapturingClient(response=FakeResponse(content))

    result = _call(client)

    assert result == {"stance": "neutral", "explanation": "z"}


def test_http_error_returns_none():
    client = CapturingClient(response=FakeResponse("server error", status_code=400))

    result = _call(client)

    assert result is None


def test_json_decode_error_returns_none():
    client = CapturingClient(response=FakeResponse("this is not json at all"))

    result = _call(client)

    assert result is None
