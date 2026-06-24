from __future__ import annotations

"""Unit tests for comment_fetcher.CommentFetcher (no real database)."""

from unittest.mock import MagicMock, patch

import pytest

import comment_fetcher
from comment_fetcher import DEFAULT_DB_URL, CommentFetcher


def _make_fetcher(mock_create_engine: MagicMock) -> tuple[CommentFetcher, MagicMock, MagicMock]:
    """Build a CommentFetcher with a fully mocked engine/connection chain.

    Returns:
        A tuple of (fetcher, mock_engine, mock_conn) where mock_conn is the
        connection yielded by the ``with engine.connect() as conn`` block.
    """
    mock_engine = MagicMock()
    mock_create_engine.return_value = mock_engine
    mock_conn = MagicMock()
    mock_engine.connect.return_value.__enter__.return_value = mock_conn
    fetcher = CommentFetcher()
    return fetcher, mock_engine, mock_conn


def _set_rows(mock_conn: MagicMock, rows: list[dict]) -> MagicMock:
    """Wire the execute().mappings() chain to return the given rows.

    Returns:
        The mock result returned by conn.execute(...).
    """
    mock_result = MagicMock()
    # mappings() returns a single-use iterator, mirroring the real SQLAlchemy result.
    mock_result.mappings.return_value = iter(rows)
    mock_conn.execute.return_value = mock_result
    return mock_result


@patch("comment_fetcher.create_engine")
def test_init_builds_engine_with_default_url(mock_create_engine: MagicMock) -> None:
    CommentFetcher()
    mock_create_engine.assert_called_once_with(DEFAULT_DB_URL)


@patch("comment_fetcher.create_engine")
def test_init_accepts_custom_url(mock_create_engine: MagicMock) -> None:
    custom = "postgresql://user:pass@host:1234/db"
    CommentFetcher(db_url=custom)
    mock_create_engine.assert_called_once_with(custom)


@patch("comment_fetcher.create_engine")
def test_get_comments_returns_list_of_dicts(mock_create_engine: MagicMock) -> None:
    fetcher, _engine, mock_conn = _make_fetcher(mock_create_engine)
    rows = [
        {"cid": 1, "comment_type": "default", "entity_id": 42, "field_comment_value": "hi"},
        {"cid": 2, "comment_type": "default", "entity_id": 42, "field_comment_value": "bye"},
    ]
    _set_rows(mock_conn, rows)

    result = fetcher.get_comments(42)

    assert result == rows
    assert all(isinstance(item, dict) and type(item) is dict for item in result)


@patch("comment_fetcher.create_engine")
def test_get_comments_empty_result_returns_empty_list(mock_create_engine: MagicMock) -> None:
    fetcher, _engine, mock_conn = _make_fetcher(mock_create_engine)
    _set_rows(mock_conn, [])

    assert fetcher.get_comments(7) == []


@patch("comment_fetcher.create_engine")
def test_get_comments_passes_bound_parameter(mock_create_engine: MagicMock) -> None:
    fetcher, _engine, mock_conn = _make_fetcher(mock_create_engine)
    _set_rows(mock_conn, [])

    fetcher.get_comments(42)

    args, _kwargs = mock_conn.execute.call_args
    assert args[1] == {"contribution_id": 42}


@patch("comment_fetcher.create_engine")
@pytest.mark.parametrize("bad_value", ["5", 5.0, None, True])
def test_get_comments_rejects_non_int(mock_create_engine: MagicMock, bad_value: object) -> None:
    fetcher, _engine, mock_conn = _make_fetcher(mock_create_engine)

    with pytest.raises(TypeError):
        fetcher.get_comments(bad_value)  # type: ignore[arg-type]

    mock_conn.execute.assert_not_called()


@patch("comment_fetcher.create_engine")
def test_close_disposes_engine(mock_create_engine: MagicMock) -> None:
    fetcher, mock_engine, _conn = _make_fetcher(mock_create_engine)

    fetcher.close()

    mock_engine.dispose.assert_called_once_with()


def test_module_exposes_create_engine() -> None:
    assert hasattr(comment_fetcher, "create_engine")
