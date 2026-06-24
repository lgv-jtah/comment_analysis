"""Tests for get_contribution_with_comments and its menu wiring in main.

These tests never touch a real database or network. ``main.requests`` and
``main.CommentFetcher`` are monkeypatched, and a small real parquet is written
to pytest's ``tmp_path`` with ``main.CONTRIBUTIONS_PATH`` redirected at it.
"""
import builtins

import pandas as pd
import pytest
import requests
from unittest.mock import MagicMock

import main


def _write_parquet(tmp_path, rows):
    """Write rows to a parquet under tmp_path and point CONTRIBUTIONS_PATH at it."""
    df = pd.DataFrame(rows)
    path = tmp_path / "contributions.parquet"
    df.to_parquet(path)
    return path


def _single_row():
    return [{
        "project_id": "p1",
        "id": 10,
        "title": "Test",
        "contributionContent": "Hello",
        "category": "cat1",
    }]


def _mock_requests(features):
    mock_response = MagicMock()
    mock_response.json.return_value = {"features": features}
    mock_requests = MagicMock()
    mock_requests.get.return_value = mock_response
    mock_requests.RequestException = requests.RequestException
    return mock_requests


def _fake_fetcher_class(called_with, comments):
    class FakeCommentFetcher:
        def get_comments(self, contribution_id):
            called_with["contribution_id"] = contribution_id
            return comments

        def close(self):
            called_with["closed"] = True

    return FakeCommentFetcher


def test_id_provided_returns_contribution_and_comments(tmp_path, monkeypatch):
    path = _write_parquet(tmp_path, _single_row())
    monkeypatch.setattr(main, "CONTRIBUTIONS_PATH", path)

    features = [{"properties": {
        "id": 10, "title": "Test", "contributionContent": "Hello", "category": "cat1",
    }}]
    monkeypatch.setattr(main, "requests", _mock_requests(features))

    fake_comments = [{"cid": 1, "field_comment_value": "Nice"}]
    monkeypatch.setattr(main, "CommentFetcher", _fake_fetcher_class({}, fake_comments))

    result = main.get_contribution_with_comments(10)

    assert result is not None
    assert result["contribution"]["id"] == 10
    assert result["contribution"]["title"] == "Test"
    assert result["comments"] == fake_comments


def test_id_provided_calls_get_comments_with_native_int(tmp_path, monkeypatch):
    path = _write_parquet(tmp_path, _single_row())
    monkeypatch.setattr(main, "CONTRIBUTIONS_PATH", path)

    features = [{"properties": {
        "id": 10, "title": "Test", "contributionContent": "Hello", "category": "cat1",
    }}]
    monkeypatch.setattr(main, "requests", _mock_requests(features))

    called_with = {}
    monkeypatch.setattr(main, "CommentFetcher", _fake_fetcher_class(called_with, []))

    main.get_contribution_with_comments(10)

    assert type(called_with["contribution_id"]) is int
    assert called_with["contribution_id"] == 10


def test_random_pick_uses_no_api(tmp_path, monkeypatch):
    path = _write_parquet(tmp_path, _single_row())
    monkeypatch.setattr(main, "CONTRIBUTIONS_PATH", path)

    mock_requests = _mock_requests([])
    monkeypatch.setattr(main, "requests", mock_requests)

    fake_comments = [{"cid": 1, "field_comment_value": "Nice"}]
    called_with = {}
    monkeypatch.setattr(main, "CommentFetcher", _fake_fetcher_class(called_with, fake_comments))

    result = main.get_contribution_with_comments(None)

    assert result is not None
    assert result["contribution"]["id"] == 10
    assert result["comments"] == fake_comments
    mock_requests.get.assert_not_called()
    assert type(called_with["contribution_id"]) is int


def test_missing_parquet_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "CONTRIBUTIONS_PATH", tmp_path / "does_not_exist.parquet")
    assert main.get_contribution_with_comments(10) is None


def test_id_not_in_parquet_returns_none(tmp_path, monkeypatch):
    path = _write_parquet(tmp_path, _single_row())
    monkeypatch.setattr(main, "CONTRIBUTIONS_PATH", path)

    mock_requests = _mock_requests([])
    monkeypatch.setattr(main, "requests", mock_requests)

    assert main.get_contribution_with_comments(999) is None
    mock_requests.get.assert_not_called()


def test_api_request_failure_returns_none(tmp_path, monkeypatch):
    path = _write_parquet(tmp_path, _single_row())
    monkeypatch.setattr(main, "CONTRIBUTIONS_PATH", path)

    mock_requests = _mock_requests([])
    mock_requests.get.side_effect = requests.RequestException("boom")
    monkeypatch.setattr(main, "requests", mock_requests)

    assert main.get_contribution_with_comments(10) is None


def test_api_returns_no_matching_feature_returns_none(tmp_path, monkeypatch):
    path = _write_parquet(tmp_path, _single_row())
    monkeypatch.setattr(main, "CONTRIBUTIONS_PATH", path)

    features = [{"properties": {"id": 42, "title": "Other"}}]
    monkeypatch.setattr(main, "requests", _mock_requests(features))

    assert main.get_contribution_with_comments(10) is None


def test_no_comments_returns_dict_with_empty_list(tmp_path, monkeypatch):
    path = _write_parquet(tmp_path, _single_row())
    monkeypatch.setattr(main, "CONTRIBUTIONS_PATH", path)

    features = [{"properties": {
        "id": 10, "title": "Test", "contributionContent": "Hello", "category": "cat1",
    }}]
    monkeypatch.setattr(main, "requests", _mock_requests(features))
    monkeypatch.setattr(main, "CommentFetcher", _fake_fetcher_class({}, []))

    result = main.get_contribution_with_comments(10)

    assert result is not None
    assert result["comments"] == []


def test_run_get_contribution_with_comments_blank_input(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda _="": "")
    called = {}
    monkeypatch.setattr(
        main, "get_contribution_with_comments",
        lambda cid: called.setdefault("cid", cid),
    )

    main.run_get_contribution_with_comments()

    assert "cid" in called
    assert called["cid"] is None


def test_run_get_contribution_with_comments_id_input(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda _="": "  77  ")
    called = {}
    monkeypatch.setattr(
        main, "get_contribution_with_comments",
        lambda cid: called.setdefault("cid", cid),
    )

    main.run_get_contribution_with_comments()

    assert called["cid"] == 77
    assert type(called["cid"]) is int


def test_run_get_contribution_with_comments_invalid_input(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda _="": "abc")
    called = {}
    monkeypatch.setattr(
        main, "get_contribution_with_comments",
        lambda cid: called.setdefault("cid", cid),
    )

    main.run_get_contribution_with_comments()

    assert "cid" not in called


def test_run_menu_returns_6(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda _="": "6")
    assert main.run_menu() == "6"


def test_main_dispatches_to_get_contribution(monkeypatch):
    monkeypatch.setattr(main, "run_menu", lambda: "6")
    called = {}
    monkeypatch.setattr(
        main, "run_get_contribution_with_comments",
        lambda: called.setdefault("ran", True),
    )

    main.main()

    assert called.get("ran") is True
