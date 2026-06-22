"""Tests for the interactive menu and contribution-fetch flow in main.

These tests never touch the network: ``main.CreateDataset`` is monkeypatched to
a local ``FakeDataset`` and ``builtins.input`` is monkeypatched to simulate user
choices. Tests that write files redirect ``main.DATA_DIR`` and
``main.CONTRIBUTIONS_PATH`` into pytest's ``tmp_path``.
"""
import pandas as pd
import pytest

import main


class FakeDataset:
    """Network-free stand-in for ``create_dataset.CreateDataset``.

    Class-level attributes let each test configure the returned projects /
    contributions and inspect what was called without instance plumbing.
    """

    projects_df = pd.DataFrame({"id": [1, 2]})
    contributions_df = pd.DataFrame(
        {"project_id": ["1", "1", "2"], "id": [10, 11, 12]}
    )
    summary_text = "1    2\n2    1"

    create_called = False
    fetch_called_with = None

    def __init__(self, *args, **kwargs):
        pass

    def create(self):
        type(self).create_called = True
        return self.projects_df

    def fetch_all_contributions(self, project_ids):
        type(self).fetch_called_with = project_ids
        return self.contributions_df

    def contributions_summary(self, df, top_n=10):
        return self.summary_text


@pytest.fixture
def fresh_fake():
    """Reset FakeDataset class-level state and yield the class."""
    FakeDataset.projects_df = pd.DataFrame({"id": [1, 2]})
    FakeDataset.contributions_df = pd.DataFrame(
        {"project_id": ["1", "1", "2"], "id": [10, 11, 12]}
    )
    FakeDataset.summary_text = "1    2\n2    1"
    FakeDataset.create_called = False
    FakeDataset.fetch_called_with = None
    return FakeDataset


@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    """Redirect DATA_DIR / CONTRIBUTIONS_PATH into a temp directory."""
    data_dir = tmp_path / "data"
    contrib_path = data_dir / "contributions.parquet"
    monkeypatch.setattr(main, "DATA_DIR", data_dir)
    monkeypatch.setattr(main, "CONTRIBUTIONS_PATH", contrib_path)
    return data_dir, contrib_path


# --- run_menu ---


def test_run_menu_returns_1(monkeypatch):
    """run_menu returns "1" when the user selects option 1."""
    monkeypatch.setattr("builtins.input", lambda *a: "1")
    assert main.run_menu() == "1"


def test_run_menu_returns_2(monkeypatch):
    """run_menu returns "2" when the user selects option 2."""
    monkeypatch.setattr("builtins.input", lambda *a: "2")
    assert main.run_menu() == "2"


def test_run_menu_returns_q(monkeypatch):
    """run_menu returns "q" when the user chooses to quit."""
    monkeypatch.setattr("builtins.input", lambda *a: "q")
    assert main.run_menu() == "q"


def test_run_menu_reprompts_on_invalid(monkeypatch):
    """run_menu re-prompts on invalid input until a valid choice is entered."""
    calls = {"n": 0}
    answers = iter(["x", "1"])

    def fake_input(*a):
        calls["n"] += 1
        return next(answers)

    monkeypatch.setattr("builtins.input", fake_input)
    assert main.run_menu() == "1"
    assert calls["n"] == 2


# --- _prompt_overwrite ---


def test_prompt_overwrite_yes(monkeypatch, tmp_path):
    """_prompt_overwrite returns True when the user answers "y"."""
    monkeypatch.setattr("builtins.input", lambda *a: "y")
    assert main._prompt_overwrite(tmp_path / "f.parquet") is True


def test_prompt_overwrite_no(monkeypatch, tmp_path):
    """_prompt_overwrite returns False when the user answers "n"."""
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    assert main._prompt_overwrite(tmp_path / "f.parquet") is False


def test_prompt_overwrite_blank(monkeypatch, tmp_path):
    """_prompt_overwrite treats a blank answer as a decline (False)."""
    monkeypatch.setattr("builtins.input", lambda *a: "")
    assert main._prompt_overwrite(tmp_path / "f.parquet") is False


# --- fetch_contributions ---


def test_fetch_contributions_writes_parquet(
    monkeypatch, tmp_paths, fresh_fake
):
    """fetch_contributions writes the fetched contributions to a parquet file."""
    _, contrib_path = tmp_paths
    monkeypatch.setattr(main, "CreateDataset", fresh_fake)
    monkeypatch.setattr("builtins.input", lambda *a: "")

    main.fetch_contributions()

    assert contrib_path.exists()
    written = pd.read_parquet(contrib_path)
    assert written.shape == (3, 2)
    assert list(written.columns) == ["project_id", "id"]


def test_fetch_contributions_skips_on_decline(
    monkeypatch, tmp_paths, fresh_fake
):
    """fetch_contributions leaves the existing file untouched when overwrite is declined."""
    data_dir, contrib_path = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    original = pd.DataFrame({"keep": [1]})
    original.to_parquet(contrib_path)

    monkeypatch.setattr(main, "CreateDataset", fresh_fake)
    monkeypatch.setattr("builtins.input", lambda *a: "n")

    main.fetch_contributions()

    assert fresh_fake.create_called is False
    assert pd.read_parquet(contrib_path).equals(original)


def test_fetch_contributions_overwrites_on_confirm(
    monkeypatch, tmp_paths, fresh_fake
):
    """fetch_contributions replaces the existing file when overwrite is confirmed."""
    data_dir, contrib_path = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"keep": [1]}).to_parquet(contrib_path)

    monkeypatch.setattr(main, "CreateDataset", fresh_fake)
    monkeypatch.setattr("builtins.input", lambda *a: "y")

    main.fetch_contributions()

    assert fresh_fake.create_called is True
    written = pd.read_parquet(contrib_path)
    assert list(written.columns) == ["project_id", "id"]
    assert "keep" not in written.columns


def test_fetch_contributions_aborts_on_empty_projects(
    monkeypatch, tmp_paths, fresh_fake
):
    """fetch_contributions aborts without fetching or writing when no projects are returned."""
    _, contrib_path = tmp_paths
    fresh_fake.projects_df = pd.DataFrame()
    monkeypatch.setattr(main, "CreateDataset", fresh_fake)
    monkeypatch.setattr("builtins.input", lambda *a: "")

    main.fetch_contributions()

    assert not contrib_path.exists()
    assert fresh_fake.fetch_called_with is None


def test_fetch_contributions_passes_stringified_ids(
    monkeypatch, tmp_paths, fresh_fake
):
    """fetch_contributions drops NaN project ids and passes the rest as strings."""
    # Use pandas' nullable Int64 so the NaN row does not upcast the column to
    # float (which would stringify "1" as "1.0"); this mirrors real DIPAS ids,
    # which are integers.
    fresh_fake.projects_df = pd.DataFrame(
        {"id": pd.array([1, None, 3], dtype="Int64")}
    )
    monkeypatch.setattr(main, "CreateDataset", fresh_fake)
    monkeypatch.setattr("builtins.input", lambda *a: "")

    main.fetch_contributions()

    assert fresh_fake.fetch_called_with == ["1", "3"]


# --- main dispatcher ---


def test_main_dispatches_to_fetch(monkeypatch):
    """main routes choice "1" to fetch_contributions."""
    monkeypatch.setattr(main, "run_menu", lambda: "1")
    called = {"fetch": False}
    monkeypatch.setattr(
        main, "fetch_contributions", lambda: called.__setitem__("fetch", True)
    )
    main.main()
    assert called["fetch"] is True


def test_main_dispatches_to_stance_detection(monkeypatch):
    """main routes choice "2" to run_stance_detection via asyncio.run."""
    monkeypatch.setattr(main, "run_menu", lambda: "2")
    called = {"run": False}

    async def fake_run():
        called["run"] = True

    monkeypatch.setattr(main, "run_stance_detection", fake_run)
    main.main()
    assert called["run"] is True


def test_main_quits(monkeypatch):
    """main takes no action on choice "q"."""
    monkeypatch.setattr(main, "run_menu", lambda: "q")
    # Should not raise and should not invoke either action.
    monkeypatch.setattr(
        main, "fetch_contributions", lambda: pytest.fail("should not fetch")
    )
    main.main()
