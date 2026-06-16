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
