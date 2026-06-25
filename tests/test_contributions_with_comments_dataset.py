"""Tests for ContributionsWithCommentsDataset and its main.py wiring.

These tests never touch a real database: a ``FakeCommentFetcher`` is injected
into the dataset, and the contributions source is a real Parquet file written
into pytest's ``tmp_path``. The main.py wiring tests monkeypatch
``builtins.input``, ``main.run_menu``, and ``main.CONTRIBUTIONS_PATH``.
"""
import pandas as pd
import pytest

import main
from contributions_with_comments_dataset import (
    CONTRIBUTIONS_PATH,
    ContributionsWithCommentsDataset,
)


def _make_parquet(tmp_path, rows=None):
    """Write ``rows`` to ``tmp_path/contributions.parquet`` and return the path.

    Args:
        tmp_path: pytest temp directory.
        rows: Optional list of row dicts. Defaults to five rows with ids 1-5.

    Returns:
        The path to the written Parquet file.
    """
    if rows is None:
        rows = [
            {
                "project_id": "p1",
                "id": i,
                "title": f"title {i}",
                "contributionContent": f"content {i}",
                "category": "cat",
            }
            for i in range(1, 6)
        ]
    path = tmp_path / "contributions.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


class FakeCommentFetcher:
    """Network-free stand-in for ``comment_fetcher.CommentFetcher``."""

    def __init__(self, comments_map=None):
        """Create a fetcher with a fixed id -> comments mapping.

        Args:
            comments_map: Mapping of contribution id to its comment list.
        """
        self.comments_map = comments_map or {}
        self.closed = False

    def get_comments(self, cid):
        """Return the configured comments, asserting a native int was passed."""
        assert type(cid) is int, f"expected native int, got {type(cid).__name__}"
        return self.comments_map.get(cid, [])

    def close(self):
        """Record that the fetcher was closed."""
        self.closed = True


def test_build_happy_path(tmp_path):
    """All ids have comments: every sampled row is kept with a comments column."""
    source = _make_parquet(tmp_path)
    comments_map = {i: [{"field_comment_value": f"c{i}"}] for i in range(1, 6)}
    fetcher = FakeCommentFetcher(comments_map)
    dataset = ContributionsWithCommentsDataset(
        n=5, source_path=source, fetcher=fetcher, random_state=0
    )

    out = dataset.build()

    assert out.exists()
    result = pd.read_parquet(out)
    assert len(result) == 5
    assert "comments" in result.columns
    first_id = int(result.iloc[0]["id"])
    assert list(result.iloc[0]["comments"]) == comments_map[first_id]


def test_contributions_without_comments_are_skipped(tmp_path):
    """Rows whose fetcher returns an empty list are dropped from the output."""
    source = _make_parquet(tmp_path)
    comments_map = {1: [{"v": 1}], 3: [{"v": 3}], 5: [{"v": 5}]}
    fetcher = FakeCommentFetcher(comments_map)
    dataset = ContributionsWithCommentsDataset(
        n=5, source_path=source, fetcher=fetcher, random_state=0
    )

    out = dataset.build()
    result = pd.read_parquet(out)

    assert len(result) == 3
    assert sorted(int(i) for i in result["id"]) == [1, 3, 5]


def test_all_without_comments_writes_empty_parquet(tmp_path):
    """When no row has comments an empty Parquet file is still written."""
    source = _make_parquet(tmp_path)
    fetcher = FakeCommentFetcher({})
    dataset = ContributionsWithCommentsDataset(
        n=5, source_path=source, fetcher=fetcher, random_state=0
    )

    out = dataset.build()

    assert out.exists()
    assert len(pd.read_parquet(out)) == 0


def test_get_comments_called_with_native_int(tmp_path):
    """get_comments receives a native int (asserted inside the fake fetcher)."""
    source = _make_parquet(tmp_path)
    fetcher = FakeCommentFetcher({i: [{"v": i}] for i in range(1, 6)})
    dataset = ContributionsWithCommentsDataset(
        n=5, source_path=source, fetcher=fetcher, random_state=0
    )

    # Build succeeds only if every get_comments call passed `type(cid) is int`.
    dataset.build()


def test_fetcher_close_called(tmp_path):
    """build closes the fetcher after a successful run."""
    source = _make_parquet(tmp_path)
    fetcher = FakeCommentFetcher({1: [{"v": 1}]})
    dataset = ContributionsWithCommentsDataset(
        n=5, source_path=source, fetcher=fetcher, random_state=0
    )

    dataset.build()

    assert fetcher.closed is True


def test_fetcher_close_called_on_error(tmp_path):
    """build propagates a fetcher error but still closes the fetcher."""
    source = _make_parquet(tmp_path)

    class RaisingFetcher(FakeCommentFetcher):
        def get_comments(self, cid):
            raise RuntimeError("boom")

    fetcher = RaisingFetcher()
    dataset = ContributionsWithCommentsDataset(
        n=5, source_path=source, fetcher=fetcher, random_state=0
    )

    with pytest.raises(RuntimeError):
        dataset.build()

    assert fetcher.closed is True


def test_output_path_naming(tmp_path):
    """output_path encodes n in its filename."""
    source = _make_parquet(tmp_path)
    dataset = ContributionsWithCommentsDataset(n=7, source_path=source)
    assert dataset.output_path.name == "contributions_with_comments_7.parquet"


def test_invalid_n_raises_valueerror(tmp_path):
    """Non-positive n is rejected at construction time."""
    source = _make_parquet(tmp_path)
    with pytest.raises(ValueError):
        ContributionsWithCommentsDataset(n=0, source_path=source)
    with pytest.raises(ValueError):
        ContributionsWithCommentsDataset(n=-1, source_path=source)


def test_source_file_missing_raises_filenotfound(tmp_path):
    """A missing source file raises FileNotFoundError on build."""
    missing = tmp_path / "nope.parquet"
    fetcher = FakeCommentFetcher({})
    dataset = ContributionsWithCommentsDataset(
        n=5, source_path=missing, fetcher=fetcher
    )
    with pytest.raises(FileNotFoundError):
        dataset.build()


def test_missing_id_column_raises_valueerror(tmp_path):
    """A source without an id column raises ValueError on build."""
    source = _make_parquet(tmp_path, rows=[{"project_id": "p1", "title": "t"}])
    fetcher = FakeCommentFetcher({})
    dataset = ContributionsWithCommentsDataset(
        n=5, source_path=source, fetcher=fetcher
    )
    with pytest.raises(ValueError):
        dataset.build()


def test_n_exceeds_available_clamps(tmp_path):
    """n larger than the row count clamps to all available rows."""
    source = _make_parquet(tmp_path)
    fetcher = FakeCommentFetcher({i: [{"v": i}] for i in range(1, 6)})
    dataset = ContributionsWithCommentsDataset(
        n=100, source_path=source, fetcher=fetcher, random_state=0
    )

    out = dataset.build()

    assert len(pd.read_parquet(out)) == 5


def test_build_lazily_constructs_fetcher(tmp_path, monkeypatch):
    """When no fetcher is injected, build constructs CommentFetcher lazily."""
    import comment_fetcher

    source = _make_parquet(tmp_path)
    lazy = FakeCommentFetcher({i: [{"v": i}] for i in range(1, 6)})

    # Replace the class the lazy `from comment_fetcher import CommentFetcher`
    # resolves so build never touches a real database.
    monkeypatch.setattr(comment_fetcher, "CommentFetcher", lambda: lazy)

    dataset = ContributionsWithCommentsDataset(
        n=5, source_path=source, random_state=0
    )
    out = dataset.build()

    assert out.exists()
    assert lazy.closed is True


def test_default_n_and_source_path():
    """The defaults are n=100 and the module-level CONTRIBUTIONS_PATH."""
    dataset = ContributionsWithCommentsDataset()
    assert dataset.n == 100
    assert dataset.source_path == CONTRIBUTIONS_PATH


# --- main.py wiring ---


def test_run_menu_returns_7(monkeypatch):
    """run_menu returns "7" when the user selects option 7."""
    monkeypatch.setattr("builtins.input", lambda *a: "7")
    assert main.run_menu() == "7"


def test_main_dispatches_to_create_dataset(monkeypatch):
    """main routes choice "7" to create_contributions_with_comments_dataset."""
    monkeypatch.setattr(main, "run_menu", lambda: "7")
    called = {"build": False}
    monkeypatch.setattr(
        main,
        "create_contributions_with_comments_dataset",
        lambda: called.__setitem__("build", True),
    )
    main.main()
    assert called["build"] is True


def test_create_dataset_missing_file(monkeypatch, tmp_path):
    """The flow aborts (no dataset constructed) when the source file is missing."""
    missing = tmp_path / "contributions.parquet"
    monkeypatch.setattr(main, "CONTRIBUTIONS_PATH", missing)

    constructed = {"called": False}

    def fake_ctor(*args, **kwargs):
        constructed["called"] = True

    monkeypatch.setattr(main, "ContributionsWithCommentsDataset", fake_ctor)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no input expected"))

    main.create_contributions_with_comments_dataset()

    assert constructed["called"] is False


def test_create_dataset_happy_path(monkeypatch, tmp_path):
    """The flow constructs the dataset with n and source_path, then builds it."""
    source = _make_parquet(tmp_path)
    monkeypatch.setattr(main, "CONTRIBUTIONS_PATH", source)

    captured = {}

    class FakeDataset:
        def __init__(self, *args, **kwargs):
            captured["kwargs"] = kwargs

        @property
        def output_path(self):
            return tmp_path / "contributions_with_comments_3.parquet"

        def build(self):
            return tmp_path / "contributions_with_comments_3.parquet"

    monkeypatch.setattr(main, "ContributionsWithCommentsDataset", FakeDataset)
    monkeypatch.setattr("builtins.input", lambda *a: "3")

    main.create_contributions_with_comments_dataset()

    assert captured["kwargs"]["n"] == 3
    assert captured["kwargs"]["source_path"] == source
