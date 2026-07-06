"""Tests for key_message_judge.KeyMessageJudge.

These tests never make a real network request. Following the project's
convention (see test_key_message_extractor.py), coroutines are driven via
asyncio.run(...) rather than pytest-asyncio, and the LLM call is monkeypatched
to an async stub instead of relying on an external mocking library.
"""
import asyncio

from key_message_judge import KeyMessageJudge
from llm_handler import KeyMessageJudgment
from stance_detector import StanceDetector


def _make_judge():
    """Build a judge using the real DEFAULT_PROMPTS_DIR prompt files."""
    return KeyMessageJudge(base_url="http://testserver/v1", model_name="test-model")


_CANNED_JUDGMENT = {
    "completeness": {
        "score": 4,
        "justification": "Fast alle Punkte erfasst.",
        "missing_topics": [],
    },
    "semantic_precision": {
        "score": 5,
        "justification": "Klar formuliert.",
        "issues": [],
    },
    "downstream_viability": {
        "score": 3,
        "justification": "Eine Aussage ist persönlich.",
        "issues": [
            {
                "key_message_index": 0,
                "problem": "Persönlicher Umstand.",
                "suggested_improvement": "Als öffentliches Anliegen umformulieren.",
            }
        ],
    },
    "overall_score": 4,
    "overall_assessment": "Insgesamt gute Hauptaussagen.",
}


_KEY_MESSAGES = [
    {
        "key_message": "Mein Fahrrad ist ungeeignet",
        "key_message_type": "Problem",
        "related_sentences": ["Mein Fahrrad ist ungeeignet."],
        "explanation": "…",
    },
    {
        "key_message": "Mehr Bäume pflanzen",
        "key_message_type": "Wunsch",
        "related_sentences": ["Wir brauchen mehr Bäume."],
        "explanation": "…",
    },
]


def test_judge_happy_path(monkeypatch):
    """judge() returns the judgment dict from the LLM response unchanged."""
    judge = _make_judge()

    async def fake_call_model(**kwargs):
        return _CANNED_JUDGMENT

    monkeypatch.setattr(judge._llm_handler, "call_model", fake_call_model)

    semaphore = asyncio.Semaphore(1)
    result = asyncio.run(
        judge.judge(None, "Ein Beitrag.", "Radverkehr", _KEY_MESSAGES, semaphore)
    )
    assert result == _CANNED_JUDGMENT


def test_judge_returns_none_when_llm_fails(monkeypatch):
    """judge() propagates None when the underlying LLM call returns None."""
    judge = _make_judge()

    async def fake_call_model(**kwargs):
        return None

    monkeypatch.setattr(judge._llm_handler, "call_model", fake_call_model)

    semaphore = asyncio.Semaphore(1)
    result = asyncio.run(
        judge.judge(None, "Ein Beitrag.", "Radverkehr", _KEY_MESSAGES, semaphore)
    )
    assert result is None


def test_judge_substitutes_template_variables(monkeypatch):
    """Contribution, category, and formatted key messages land in the user message."""
    judge = _make_judge()
    captured = {}

    async def fake_call_model(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(judge._llm_handler, "call_model", fake_call_model)

    contribution = "Wir brauchen mehr Bäume."
    category = "Grünflächen"
    semaphore = asyncio.Semaphore(1)
    asyncio.run(judge.judge(None, contribution, category, _KEY_MESSAGES, semaphore))

    content = captured["messages"][0]["content"]
    assert contribution in content
    assert category in content
    assert "Mehr Bäume pflanzen" in content
    assert "Mein Fahrrad ist ungeeignet" in content
    assert captured["response_schema"] is KeyMessageJudgment


def test_format_key_messages_numbers_and_types():
    """format_key_messages 0-indexes items and includes each type and text."""
    formatted = KeyMessageJudge.format_key_messages(_KEY_MESSAGES)
    lines = formatted.split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("[0]")
    assert "(Problem)" in lines[0]
    assert "Mein Fahrrad ist ungeeignet" in lines[0]
    assert lines[1].startswith("[1]")
    assert "(Wunsch)" in lines[1]
    assert "Mehr Bäume pflanzen" in lines[1]


def test_format_key_messages_handles_missing_fields():
    """format_key_messages renders empty strings for missing fields, no crash."""
    formatted = KeyMessageJudge.format_key_messages([{}])
    assert formatted == "[0] () "


def test_prompt_loaded_once_at_init(monkeypatch):
    """Prompt config is loaded once at construction, not on each judge() call."""
    calls = {"count": 0}
    original = StanceDetector.load_prompt_config

    def counting_wrapper(self, config_path, prompt_key):
        calls["count"] += 1
        return original(self, config_path, prompt_key)

    monkeypatch.setattr(StanceDetector, "load_prompt_config", counting_wrapper)

    judge = _make_judge()
    assert calls["count"] == 1

    async def fake_call_model(**kwargs):
        return {}

    monkeypatch.setattr(judge._llm_handler, "call_model", fake_call_model)

    semaphore = asyncio.Semaphore(1)
    asyncio.run(judge.judge(None, "Beitrag", "Kat", _KEY_MESSAGES, semaphore))
    asyncio.run(judge.judge(None, "Beitrag", "Kat", _KEY_MESSAGES, semaphore))

    assert calls["count"] == 1


def test_system_prompt_content():
    """The loaded system prompt contains the expected German domain keyword."""
    judge = _make_judge()
    assert "Hauptaussagen" in judge._prompt_config["system"]


def test_api_key_defaults_to_empty_string():
    """A judge built without api_key stores an empty string (no secret)."""
    judge = _make_judge()
    assert judge._api_key == ""


def test_api_key_stored_and_forwarded(monkeypatch):
    """api_key is stored on the instance and threaded into the LLM call."""
    judge = KeyMessageJudge(
        base_url="http://testserver/v1", model_name="test-model", api_key="tok"
    )
    assert judge._api_key == "tok"

    captured = {}

    async def fake_call_model(**kwargs):
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(judge._llm_handler, "call_model", fake_call_model)

    semaphore = asyncio.Semaphore(1)
    asyncio.run(judge.judge(None, "Beitrag", "Kat", _KEY_MESSAGES, semaphore))
    assert captured["api_key"] == "tok"


def test_judgment_schema_builds():
    """KeyMessageJudgment.model_json_schema() builds without raising."""
    schema = KeyMessageJudgment.model_json_schema()
    assert isinstance(schema, dict)
    assert "properties" in schema
