import json

import pytest

from stance_detector import StanceDetector


@pytest.fixture
def detector():
    return StanceDetector()


def _write_gold_labels(tmp_path, items):
    path = tmp_path / "gold_labels_corrected.json"
    path.write_text(json.dumps(items), encoding="utf-8")
    return path


def test_load_data_returns_loaded_list(detector, tmp_path, monkeypatch):
    fixture = [
        {"id": 1, "gold_stance": "neutral"},
        {"id": 2, "gold_stance": "in favor"},
    ]
    _write_gold_labels(tmp_path, fixture)
    monkeypatch.chdir(tmp_path)

    result = detector.load_data()

    assert result == fixture


def test_load_data_counts_distribution(detector, tmp_path, monkeypatch, capsys):
    fixture = [
        {"gold_stance": "in favor"},
        {"gold_stance": "neutral"},
        {"gold_stance": "ablehnung"},
        {"gold_stance": "ambiguous"},
    ]
    _write_gold_labels(tmp_path, fixture)
    monkeypatch.chdir(tmp_path)

    detector.load_data()

    out = capsys.readouterr().out
    assert "in favor: 1" in out
    assert "neutral: 1" in out
    assert "ablehnung: 1" in out
    assert "ambiguous: 1" in out


def test_load_data_reports_missing(detector, tmp_path, monkeypatch, capsys):
    fixture = [
        {"gold_stance": "neutral"},
        {"gold_stance": None},
        {"id": 3},
    ]
    _write_gold_labels(tmp_path, fixture)
    monkeypatch.chdir(tmp_path)

    detector.load_data()

    out = capsys.readouterr().out
    assert "missing/None: 2" in out


def test_load_data_no_missing(detector, tmp_path, monkeypatch, capsys):
    fixture = [
        {"gold_stance": "neutral"},
        {"gold_stance": "in favor"},
    ]
    _write_gold_labels(tmp_path, fixture)
    monkeypatch.chdir(tmp_path)

    detector.load_data()

    out = capsys.readouterr().out
    assert "missing/None: 0" in out


# --- General-reaction hint (format_gr_hint) tests ---

_HINT_PREFIX = "Hinweis: Dieser Kommentar wurde als allgemeine positive Reaktion"


def test_format_gr_hint_two_decimals():
    hint = StanceDetector.format_gr_hint(0.87)
    assert "Score: 0.87" in hint
    assert _HINT_PREFIX in hint


def test_format_gr_hint_rounds():
    hint = StanceDetector.format_gr_hint(0.7417)
    assert "Score: 0.74" in hint


# --- build_messages hint-injection tests ---

_TEMPLATE_CONFIG = {
    "system": "system prompt",
    "user_template": "Kommentar: $comment_text",
    "format": "template",
}
_FORMAT_MAP_CONFIG = {
    "system": "system prompt",
    "user_template": "Kommentar: {comment_text}",
}
_ITEM = {"comment_text": "Das ist ein Kommentar"}


def test_build_messages_without_hint_unchanged(detector):
    _, messages = detector.build_messages(_TEMPLATE_CONFIG, _ITEM)
    content = messages[0]["content"]
    assert content == "Kommentar: Das ist ein Kommentar"
    assert "Hinweis:" not in content


def test_build_messages_appends_hint(detector):
    hint = detector.format_gr_hint(0.85)
    _, messages = detector.build_messages(_TEMPLATE_CONFIG, _ITEM, gr_hint=hint)
    content = messages[0]["content"]
    assert "Kommentar: Das ist ein Kommentar" in content
    assert _HINT_PREFIX in content
    assert "Score: 0.85" in content


def test_build_messages_hint_with_format_map_branch(detector):
    hint = detector.format_gr_hint(0.85)
    _, messages = detector.build_messages(_FORMAT_MAP_CONFIG, _ITEM, gr_hint=hint)
    content = messages[0]["content"]
    assert "Kommentar: Das ist ein Kommentar" in content
    assert _HINT_PREFIX in content
    assert "Score: 0.85" in content
