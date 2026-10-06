"""Tests for llm_handler.LLMHandler.call_model.

These tests mock httpx so no real network request is ever made to the
configured base_url. call_model is async; rather than depend on the
pytest-asyncio plugin, each test drives the coroutine via asyncio.run().
"""
import asyncio
import json

import httpx
import pytest

from llm_handler import LLMHandler, normalize_stance_label


class FakeResponse:
    """Minimal stand-in for httpx.Response used by the mocked client."""

    def __init__(self, content, status_code=200, headers=None):
        self._content = content
        self.status_code = status_code
        self.text = content if isinstance(content, str) else json.dumps(content)
        self.headers = headers or {}

    def json(self):
        return {
            "choices": [{"message": {"content": self._content}}]
        }

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://testserver/v1/chat/completions")
            response = httpx.Response(
                self.status_code, request=request, headers=self.headers
            )
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


class SequenceClient:
    """Fake httpx.AsyncClient returning a different outcome on each successive
    post() call, for testing retry-then-succeed / retry-then-exhaust paths.

    Each item in `outcomes` is either an Exception instance (raised) or a
    response-like object (returned).
    """

    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.call_count = 0

    async def post(self, url, **kwargs):
        self.call_count += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome




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


@pytest.mark.parametrize("content", [
    '{"secret":"PRIVATE","secret":"PRIVATE"}',
    '{"targets":{"a":{"attribution":"unaddressed"},"a":{"attribution":"no_position"}}}',
    '{"targets":{"a":{"explanation":"PRIVATE","explanation":"PRIVATE"}}}',
])
def test_duplicate_json_rejection_is_opt_in_and_sanitized(content, capsys):
    from llm_handler import DuplicateJSONKeyError
    assert _call(CapturingClient(FakeResponse(content))) == json.loads(content)
    with pytest.raises(DuplicateJSONKeyError, match="^duplicate_json_key$"):
        _call(CapturingClient(FakeResponse(content)), reject_duplicate_json_keys=True)
    assert "PRIVATE" not in capsys.readouterr().out


@pytest.mark.parametrize("content,expected", [
    ('[{"attribution":"unaddressed"}]', [{"attribution": "unaddressed"}]),
    ('{"stance":"AGAINST"}', {"stance": "AGAINST"}),
    ('PRIVATE invalid json', None),
])
def test_strict_parsing_does_not_extract_list_objects_or_rewrite_labels(content, expected, capsys):
    assert _call(CapturingClient(FakeResponse(content)), reject_duplicate_json_keys=True) == expected
    assert "PRIVATE" not in capsys.readouterr().out


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


@pytest.mark.parametrize("wrapper", ["```json\n%s\n```", "```\n%s\n```", "PRIVATE %s", "%s PRIVATE",
                                    "<think>PRIVATE</think>%s", "\u00a0%s\u00a0"])
def test_strict_json_rejects_wrappers_without_changing_default(wrapper, monkeypatch, capsys):
    monkeypatch.setattr("llm_handler.RATE_LIMIT_DELAY", 0)
    content = wrapper % '{"stance":"neutral"}'
    strict_client = CapturingClient(FakeResponse(content))
    legacy_client = CapturingClient(FakeResponse(content))
    assert _call(strict_client, strict_json=True, enable_thinking=True) is None
    assert "PRIVATE" not in capsys.readouterr().out
    assert _call(legacy_client, enable_thinking=True) == {"stance": "neutral"}
    assert strict_client.captured_kwargs["json"] == legacy_client.captured_kwargs["json"]


def test_duplicate_only_option_preserves_fence_cleanup(monkeypatch):
    monkeypatch.setattr("llm_handler.RATE_LIMIT_DELAY", 0)
    assert _call(CapturingClient(FakeResponse('```json\n{"stance":"AGAINST"}\n```')),
                 reject_duplicate_json_keys=True) == {"stance": "AGAINST"}


def test_strict_json_preserves_valid_json_and_rejects_duplicates_and_malformed(monkeypatch, capsys):
    from llm_handler import DuplicateJSONKeyError
    monkeypatch.setattr("llm_handler.RATE_LIMIT_DELAY", 0)
    assert _call(CapturingClient(FakeResponse(' \r\n{"stance":"AGAINST"}\t')),
                 strict_json=True) == {"stance": "AGAINST"}
    with pytest.raises(DuplicateJSONKeyError):
        _call(CapturingClient(FakeResponse('{"x":"PRIVATE","x":2}')), strict_json=True)
    assert _call(CapturingClient(FakeResponse('{"x":"PRIVATE",}')), strict_json=True) is None
    assert "PRIVATE" not in capsys.readouterr().out


def test_http_error_returns_none():
    client = CapturingClient(response=FakeResponse("server error", status_code=400))

    result = _call(client)

    assert result is None


def test_json_decode_error_returns_none():
    client = CapturingClient(response=FakeResponse("this is not json at all"))

    result = _call(client)

    assert result is None


# ── normalize_stance_label unit tests ──────────────────────────────


@pytest.mark.parametrize(
    "raw, expected",
    [
        # All 3 canonical mappings.
        ("IN FAVOR", "in favor"),
        ("AGAINST", "ablehnung"),
        ("NEUTRAL", "neutral"),
        # Case variations.
        ("neutral", "neutral"),
        ("Neutral", "neutral"),
        ("NeUtRaL", "neutral"),
        ("in favor", "in favor"),
        ("In Favor", "in favor"),
        ("against", "ablehnung"),
        # Whitespace robustness.
        (" NEUTRAL ", "neutral"),
        ("  in favor", "in favor"),
        ("AGAINST\n", "ablehnung"),
    ],
)
def test_normalize_stance_label_recognized(raw, expected):
    assert normalize_stance_label(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ["MAYBE", "support", "oppose"],
)
def test_normalize_stance_label_passthrough_unchanged(raw):
    # Unrecognized values are returned unchanged, original casing preserved.
    assert normalize_stance_label(raw) == raw


@pytest.mark.parametrize(
    "raw",
    [None, "", 123],
)
def test_normalize_stance_label_safety(raw):
    # Non-string and empty inputs must pass through without raising.
    assert normalize_stance_label(raw) == raw


# ── call_model normalization regression tests ──────────────────────


def test_call_model_normalizes_in_favor():
    content = '{"stance": "IN FAVOR", "explanation": "x"}'
    client = CapturingClient(response=FakeResponse(content))

    result = _call(client)

    assert result == {"stance": "in favor", "explanation": "x"}


def test_call_model_normalizes_neutral_with_whitespace_and_keeps_explanation():
    content = '{"stance": " neutral ", "explanation": "y"}'
    client = CapturingClient(response=FakeResponse(content))

    result = _call(client)

    assert result["stance"] == "neutral"
    assert result["explanation"] == "y"


def test_call_model_normalizes_against_to_ablehnung():
    content = '{"stance": "AGAINST", "explanation": "z"}'
    client = CapturingClient(response=FakeResponse(content))

    result = _call(client)

    assert result["stance"] == "ablehnung"


def test_call_model_unrecognized_stance_passes_through():
    content = '{"stance": "GARBLED", "explanation": "w"}'
    client = CapturingClient(response=FakeResponse(content))

    result = _call(client)

    assert result["stance"] == "GARBLED"


# ── retry/backoff behavior ──────────────────────────────────────────

_FAST_RETRY_KWARGS = dict(retry_base_delay=0.001, retry_max_delay=0.005)


def test_429_retries_then_succeeds():
    content = '{"stance": "in favor", "explanation": "x"}'
    client = SequenceClient(
        [FakeResponse("rate limited", status_code=429), FakeResponse(content)]
    )

    result = _call(client, max_retries=3, **_FAST_RETRY_KWARGS)

    assert result == {"stance": "in favor", "explanation": "x"}
    assert client.call_count == 2


def test_5xx_retries_then_succeeds():
    content = '{"stance": "neutral", "explanation": "x"}'
    client = SequenceClient(
        [FakeResponse("server error", status_code=503), FakeResponse(content)]
    )

    result = _call(client, max_retries=3, **_FAST_RETRY_KWARGS)

    assert result == {"stance": "neutral", "explanation": "x"}
    assert client.call_count == 2


def test_retries_are_bounded_then_returns_none():
    client = SequenceClient(
        [
            FakeResponse("e1", status_code=503),
            FakeResponse("e2", status_code=503),
            FakeResponse("e3", status_code=503),
            FakeResponse("e4", status_code=503),  # never reached: max_retries=2 -> 3 total attempts
        ]
    )

    result = _call(client, max_retries=2, **_FAST_RETRY_KWARGS)

    assert result is None
    assert client.call_count == 3  # 1 initial + 2 retries


def test_zero_max_retries_matches_pre_retry_behavior():
    client = SequenceClient([FakeResponse("server error", status_code=503)])

    result = _call(client, max_retries=0, **_FAST_RETRY_KWARGS)

    assert result is None
    assert client.call_count == 1


@pytest.mark.parametrize("status_code", [401, 403])
def test_auth_errors_fail_fast_without_retrying(status_code):
    client = SequenceClient(
        [
            FakeResponse("unauthorized", status_code=status_code),
            FakeResponse("would succeed if retried"),  # must never be reached
        ]
    )

    result = _call(client, max_retries=3, **_FAST_RETRY_KWARGS)

    assert result is None
    assert client.call_count == 1


def test_non_retryable_4xx_still_fails_fast():
    client = SequenceClient(
        [
            FakeResponse("not found", status_code=404),
            FakeResponse("would succeed if retried"),
        ]
    )

    result = _call(client, max_retries=3, **_FAST_RETRY_KWARGS)

    assert result is None
    assert client.call_count == 1


def test_retry_after_header_is_honored(monkeypatch):
    sleeps = []

    async def fake_sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr("llm_handler.asyncio.sleep", fake_sleep)

    content = '{"stance": "in favor", "explanation": "x"}'
    client = SequenceClient(
        [
            FakeResponse("rate limited", status_code=429, headers={"Retry-After": "7"}),
            FakeResponse(content),
        ]
    )

    result = _call(client, max_retries=3, retry_base_delay=999, retry_max_delay=999)

    assert result == {"stance": "in favor", "explanation": "x"}
    # The Retry-After value (7s) was used, not the huge configured backoff.
    assert 7 in sleeps


def test_unparseable_retry_after_header_falls_back_to_computed_backoff():
    content = '{"stance": "in favor", "explanation": "x"}'
    client = SequenceClient(
        [
            FakeResponse("rate limited", status_code=429, headers={"Retry-After": "not-a-number"}),
            FakeResponse(content),
        ]
    )

    result = _call(client, max_retries=3, **_FAST_RETRY_KWARGS)

    assert result == {"stance": "in favor", "explanation": "x"}
    assert client.call_count == 2


def test_unexpected_exception_returns_none():
    class BoomClient:
        async def post(self, *a, **k):
            raise ValueError("something unrelated to HTTP broke")

    result = _call(BoomClient())

    assert result is None


def test_request_error_retries_then_succeeds():
    content = '{"stance": "in favor", "explanation": "x"}'
    client = SequenceClient(
        [httpx.ConnectError("connection reset"), FakeResponse(content)]
    )

    result = _call(client, max_retries=3, **_FAST_RETRY_KWARGS)

    assert result == {"stance": "in favor", "explanation": "x"}
    assert client.call_count == 2


def test_request_error_bounded_then_returns_none():
    client = SequenceClient(
        [
            httpx.ConnectError("e1"),
            httpx.ConnectError("e2"),
        ]
    )

    result = _call(client, max_retries=1, **_FAST_RETRY_KWARGS)

    assert result is None
    assert client.call_count == 2


def test_rate_limit_delay_fires_once_per_call_not_per_retry_attempt(monkeypatch):
    """RATE_LIMIT_DELAY spaces out distinct call_model() invocations, not
    individual retry attempts within one invocation."""
    sleep_calls = []

    async def fake_sleep(seconds):
        sleep_calls.append(seconds)

    monkeypatch.setattr("llm_handler.asyncio.sleep", fake_sleep)

    content = '{"stance": "in favor", "explanation": "x"}'
    client = SequenceClient(
        [FakeResponse("rate limited", status_code=429), FakeResponse(content)]
    )

    _call(client, max_retries=3, retry_base_delay=1, retry_max_delay=1)

    import llm_handler

    rate_limit_sleeps = [s for s in sleep_calls if s == llm_handler.RATE_LIMIT_DELAY]
    assert len(rate_limit_sleeps) == 1
