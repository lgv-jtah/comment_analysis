"""Tests for key_message_extractor.KeyMessageExtraction.

These tests never make a real network request. Following the project's
convention (see test_create_dataset.py / test_llm_handler.py), coroutines are
driven via asyncio.run(...) rather than pytest-asyncio, and the LLM call is
monkeypatched to an async stub instead of relying on an external mocking
library.
"""
import asyncio

from key_message_extractor import KeyMessageExtraction
from llm_handler import KeyMessageList
from stance_detector import StanceDetector


def _make_extractor():
    """Build an extractor using the real DEFAULT_PROMPTS_DIR prompt files."""
    return KeyMessageExtraction(base_url="http://testserver/v1", model_name="test-model")


def test_extract_happy_path(monkeypatch):
    """extract() returns the list of key-message dicts from the LLM response."""
    extractor = _make_extractor()
    canned_result = {
        "key_messages": [
            {
                "key_message": "Mehr Bäume pflanzen",
                "key_message_type": "Wunsch",
                "related_sentences": ["Wir brauchen mehr Bäume."],
                "explanation": "Der Bürger schlägt vor...",
            }
        ]
    }

    async def fake_call_model(**kwargs):
        return canned_result

    monkeypatch.setattr(extractor._llm_handler, "call_model", fake_call_model)

    client = None
    semaphore = asyncio.Semaphore(1)
    result = asyncio.run(
        extractor.extract(client, "Wir brauchen mehr Bäume.", "Stadtpark Planung", semaphore)
    )
    assert result == canned_result["key_messages"]


def test_extract_returns_none_when_llm_fails(monkeypatch):
    """extract() propagates None when the underlying LLM call returns None."""
    extractor = _make_extractor()

    async def fake_call_model(**kwargs):
        return None

    monkeypatch.setattr(extractor._llm_handler, "call_model", fake_call_model)

    client = None
    semaphore = asyncio.Semaphore(1)
    result = asyncio.run(
        extractor.extract(client, "Irgendein Beitrag.", "Irgendein Verfahren", semaphore)
    )
    assert result is None


def test_extract_substitutes_template_variables(monkeypatch):
    """The contribution and proceeding text are interpolated into the user message."""
    extractor = _make_extractor()
    captured = {}

    async def fake_call_model(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(extractor._llm_handler, "call_model", fake_call_model)

    contribution = "Wir brauchen mehr Bäume."
    proceeding = "Stadtpark Planung"
    semaphore = asyncio.Semaphore(1)
    asyncio.run(extractor.extract(None, contribution, proceeding, semaphore))

    content = captured["messages"][0]["content"]
    assert contribution in content
    assert proceeding in content
    assert captured["response_schema"] is KeyMessageList


def test_prompt_loaded_once_at_init(monkeypatch):
    """Prompt config is loaded once at construction, not on each extract() call."""
    calls = {"count": 0}
    original = StanceDetector.load_prompt_config

    def counting_wrapper(self, config_path, prompt_key):
        calls["count"] += 1
        return original(self, config_path, prompt_key)

    monkeypatch.setattr(StanceDetector, "load_prompt_config", counting_wrapper)

    extractor = _make_extractor()
    assert calls["count"] == 1

    async def fake_call_model(**kwargs):
        return {}

    monkeypatch.setattr(extractor._llm_handler, "call_model", fake_call_model)

    semaphore = asyncio.Semaphore(1)
    asyncio.run(extractor.extract(None, "Beitrag", "Verfahren", semaphore))
    asyncio.run(extractor.extract(None, "Beitrag", "Verfahren", semaphore))

    assert calls["count"] == 1


def test_system_prompt_content():
    """The loaded system prompt contains the expected domain keyword."""
    extractor = _make_extractor()
    assert "Stadtplanung" in extractor._prompt_config["system"]


def test_response_schema_is_keymessage(monkeypatch):
    """extract() passes the KeyMessageList schema through to the LLM call."""
    extractor = _make_extractor()
    captured = {}

    async def fake_call_model(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(extractor._llm_handler, "call_model", fake_call_model)

    semaphore = asyncio.Semaphore(1)
    asyncio.run(extractor.extract(None, "Beitrag", "Verfahren", semaphore))
    assert captured["response_schema"] is KeyMessageList
