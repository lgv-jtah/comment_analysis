"""Tests for the WholeProcess pipeline and its main.py wiring.

These tests never touch the network or load the real BERT model: the reaction
classifier, key-message extractor, and LLM call are all replaced with fakes.
Following the project's convention (see test_key_message_extractor.py /
test_main_menu.py), coroutines are driven via ``asyncio.run(...)`` rather than
pytest-asyncio, and the interactive flows are exercised by monkeypatching
``builtins.input`` and ``main.DATA_DIR``.
"""
import asyncio

import pandas as pd
import pytest

import main
import whole_process
from whole_process import (
    FAST_PATH_EXPLANATION,
    OUTPUT_COLUMNS,
    WholeProcess,
    should_ensemble_path,
    should_fast_path,
)


# --- Fakes ---------------------------------------------------------------


class FakeClassifier:
    """Network-free stand-in for ReactionClassifier.

    Records whether load_classifier was called and returns a configurable
    classification dict (default routes to the LLM path).
    """

    def __init__(self) -> None:
        self.load_called_with = None
        self.result = {"label": "nein", "score": 0.5}

    def load_classifier(self, path: str) -> None:
        self.load_called_with = path

    def classify_comment(self, text: str) -> dict:
        return self.result


class FakeExtractor:
    """Async stand-in for KeyMessageExtraction returning a fixed km list."""

    def __init__(self, km_list) -> None:
        self.km_list = km_list

    async def extract(self, **kwargs):
        return self.km_list


def _make_processor(monkeypatch, km_list, classifier=None, call_model=None):
    """Build a WholeProcess with all heavy collaborators replaced by fakes."""
    processor = WholeProcess(base_url="http://testserver/v1", model_name="test-model")
    processor._extractor = FakeExtractor(km_list)
    processor._classifier = classifier if classifier is not None else FakeClassifier()

    if call_model is None:

        async def call_model(**kwargs):
            return {"stance": "in favor", "explanation": "test reason"}

    monkeypatch.setattr(processor._llm_handler, "call_model", call_model)
    return processor


def _make_parquet(tmp_path, rows=None):
    """Write a rows dict to a parquet file and return its path."""
    if rows is None:
        rows = {
            "contribution_id": [101],
            "contributionContent": ["Eine Idee."],
            "comments": [["Gut", "Schlecht"]],
        }
    path = tmp_path / "input.parquet"
    pd.DataFrame(rows).to_parquet(path)
    return path


# --- routing functions ---------------------------------------------------


def test_should_fast_path_true():
    assert should_fast_path("ja", 0.9, "Gut") is True


def test_should_fast_path_score_at_threshold():
    assert should_fast_path("ja", 0.6, "Gut") is False


def test_should_fast_path_long_text():
    assert should_fast_path("ja", 0.9, "word " * 6) is False


def test_should_fast_path_wrong_label():
    assert should_fast_path("nein", 0.9, "Gut") is False


def test_should_ensemble_path_true():
    assert should_ensemble_path("ja", 0.9, "word " * 6) is True


def test_should_ensemble_path_short():
    assert should_ensemble_path("ja", 0.9, "Gut") is False


# --- WholeProcess.run ----------------------------------------------------


def test_run_flattens_rows(monkeypatch):
    """Two key messages x two comments yields four flattened rows."""
    km_list = [
        {"key_message": "km1", "key_message_type": "Wunsch"},
        {"key_message": "km2", "key_message_type": "Kritik"},
    ]
    processor = _make_processor(monkeypatch, km_list)
    df = pd.DataFrame(
        {
            "contribution_id": [7],
            "contributionContent": ["Beitrag"],
            "comments": [["c1", "c2"]],
        }
    )

    result = asyncio.run(processor.run(df))

    assert list(result.columns) == OUTPUT_COLUMNS
    assert len(result) == 4
    assert set(result["contribution_id"]) == {7}
    assert set(result["key_message"]) == {"km1", "km2"}
    assert set(result["comment_text"]) == {"c1", "c2"}


def test_run_fast_path_no_llm_call(monkeypatch):
    """A short confident-positive comment is fast-pathed without an LLM call."""
    classifier = FakeClassifier()
    classifier.result = {"label": "ja", "score": 0.9}
    called = {"hit": False}

    async def call_model(**kwargs):
        called["hit"] = True
        return {"stance": "x", "explanation": "y"}

    km_list = [{"key_message": "km1", "key_message_type": "t"}]
    processor = _make_processor(
        monkeypatch, km_list, classifier=classifier, call_model=call_model
    )
    df = pd.DataFrame(
        {
            "contribution_id": [1],
            "contributionContent": ["Beitrag"],
            "comments": [["Gut"]],
        }
    )

    result = asyncio.run(processor.run(df))

    assert called["hit"] is False
    assert result.iloc[0]["stance"] == "neutral"
    assert result.iloc[0]["explanation"] == FAST_PATH_EXPLANATION


def test_run_ensemble_hint_present(monkeypatch):
    """A long confident-positive comment routes through the ensemble path with hint."""
    classifier = FakeClassifier()
    classifier.result = {"label": "ja", "score": 0.9}
    captured = {}

    async def call_model(**kwargs):
        captured.update(kwargs)
        return {"stance": "in favor", "explanation": "reason"}

    km_list = [{"key_message": "km1", "key_message_type": "t"}]
    processor = _make_processor(
        monkeypatch, km_list, classifier=classifier, call_model=call_model
    )
    long_comment = "word " * 6
    df = pd.DataFrame(
        {
            "contribution_id": [1],
            "contributionContent": ["Beitrag"],
            "comments": [[long_comment]],
        }
    )

    asyncio.run(processor.run(df))

    content = captured["messages"][0]["content"]
    assert "allgemeine positive Reaktion" in content


def test_run_llm_path_default(monkeypatch):
    """The default nein/0.5 classifier routes to the LLM, using its stance."""
    km_list = [{"key_message": "km1", "key_message_type": "t"}]
    captured = {"hit": False}

    async def call_model(**kwargs):
        captured["hit"] = True
        return {"stance": "in favor", "explanation": "test reason"}

    processor = _make_processor(monkeypatch, km_list, call_model=call_model)
    df = pd.DataFrame(
        {
            "contribution_id": [1],
            "contributionContent": ["Beitrag"],
            "comments": [["a longer comment here"]],
        }
    )

    result = asyncio.run(processor.run(df))

    assert captured["hit"] is True
    assert result.iloc[0]["stance"] == "in favor"
    assert result.iloc[0]["explanation"] == "test reason"


def test_run_skips_when_extract_returns_none(monkeypatch):
    """A contribution whose extraction returns None produces no rows."""
    processor = _make_processor(monkeypatch, None)
    df = pd.DataFrame(
        {
            "contribution_id": [1],
            "contributionContent": ["Beitrag"],
            "comments": [["c1"]],
        }
    )

    result = asyncio.run(processor.run(df))

    assert len(result) == 0
    assert list(result.columns) == OUTPUT_COLUMNS


def test_run_skips_when_extract_returns_empty(monkeypatch):
    """A contribution whose extraction returns [] produces no rows."""
    processor = _make_processor(monkeypatch, [])
    df = pd.DataFrame(
        {
            "contribution_id": [1],
            "contributionContent": ["Beitrag"],
            "comments": [["c1"]],
        }
    )

    result = asyncio.run(processor.run(df))

    assert len(result) == 0


def test_run_stance_none_writes_row(monkeypatch):
    """When the LLM call returns None the row is still written with stance None."""
    km_list = [{"key_message": "km1", "key_message_type": "t"}]

    async def call_model(**kwargs):
        return None

    processor = _make_processor(monkeypatch, km_list, call_model=call_model)
    df = pd.DataFrame(
        {
            "contribution_id": [1],
            "contributionContent": ["Beitrag"],
            "comments": [["a longer comment here"]],
        }
    )

    result = asyncio.run(processor.run(df))

    assert len(result) == 1
    assert result.iloc[0]["stance"] is None
    assert result.iloc[0]["explanation"] is None


def test_run_empty_dataframe(monkeypatch):
    """An empty input DataFrame yields an empty result with the right columns."""
    processor = _make_processor(monkeypatch, [{"key_message": "km1"}])
    df = pd.DataFrame(
        {"contribution_id": [], "contributionContent": [], "comments": []}
    )

    result = asyncio.run(processor.run(df))

    assert len(result) == 0
    assert list(result.columns) == OUTPUT_COLUMNS


def test_run_handles_nan_content(monkeypatch):
    """A NaN contributionContent reaches the extractor as an empty string."""
    captured = {}

    class CapturingExtractor:
        async def extract(self, **kwargs):
            captured.update(kwargs)
            return []

    processor = _make_processor(monkeypatch, [])
    processor._extractor = CapturingExtractor()
    df = pd.DataFrame(
        {
            "contribution_id": [1],
            "contributionContent": [float("nan")],
            "comments": [["c1"]],
        }
    )

    asyncio.run(processor.run(df))

    assert captured["contribution"] == ""


def test_init_does_not_load_classifier(monkeypatch):
    """Constructing WholeProcess must not load the classifier model."""
    load_calls = {"n": 0}

    class TrackingClassifier:
        def load_classifier(self, path):
            load_calls["n"] += 1

        def classify_comment(self, text):
            return {"label": "nein", "score": 0.5}

    monkeypatch.setattr(whole_process, "ReactionClassifier", TrackingClassifier)

    WholeProcess(base_url="http://testserver/v1", model_name="test-model")

    assert load_calls["n"] == 0


# --- main.py wiring ------------------------------------------------------


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Redirect main.DATA_DIR into a temp directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "DATA_DIR", data_dir)
    return data_dir


class FakeWholeProcess:
    """Stand-in for main.WholeProcess used by the run_whole_process tests."""

    last_instance = None
    run_result = None

    def __init__(self, *args, **kwargs):
        self.init_kwargs = kwargs
        FakeWholeProcess.last_instance = self

    async def run(self, df, proceeding_description=""):
        self.run_df = df
        if FakeWholeProcess.run_result is not None:
            return FakeWholeProcess.run_result
        return pd.DataFrame(
            [
                {
                    "contribution_id": 1,
                    "key_message": "km1",
                    "key_message_type": "t",
                    "comment_text": "c1",
                    "stance": "in favor",
                    "explanation": "reason",
                }
            ],
            columns=OUTPUT_COLUMNS,
        )


def test_run_menu_returns_9(monkeypatch):
    """run_menu returns "9" when the user selects option 9."""
    monkeypatch.setattr("builtins.input", lambda *a: "9")
    assert main.run_menu() == "9"


def test_main_dispatches_to_whole_process(monkeypatch):
    """main routes choice "9" to run_whole_process via asyncio.run."""
    monkeypatch.setattr(main, "run_menu", lambda: "9")
    called = {"run": False}

    async def fake_run():
        called["run"] = True

    monkeypatch.setattr(main, "run_whole_process", fake_run)
    main.main()
    assert called["run"] is True


def test_run_whole_process_happy_path(monkeypatch, tmp_data_dir):
    """run_whole_process writes the flattened output parquet."""
    FakeWholeProcess.run_result = None
    source = tmp_data_dir / "input.parquet"
    pd.DataFrame(
        {
            "contribution_id": [1],
            "contributionContent": ["Beitrag"],
            "comments": [["c1"]],
        }
    ).to_parquet(source)

    monkeypatch.setattr(main, "WholeProcess", FakeWholeProcess)
    monkeypatch.setattr("builtins.input", lambda *a: "1")

    asyncio.run(main.run_whole_process())

    output = tmp_data_dir / "whole_process_input.parquet"
    assert output.exists()
    written = pd.read_parquet(output)
    assert list(written.columns) == OUTPUT_COLUMNS
    assert written.iloc[0]["key_message"] == "km1"


def test_run_whole_process_missing_comments(monkeypatch, tmp_data_dir):
    """A source lacking the comments column produces no output."""
    source = tmp_data_dir / "input.parquet"
    pd.DataFrame({"contributionContent": ["Beitrag"]}).to_parquet(source)

    monkeypatch.setattr(
        main, "WholeProcess", lambda *a, **k: pytest.fail("should not construct")
    )
    monkeypatch.setattr("builtins.input", lambda *a: "1")

    asyncio.run(main.run_whole_process())

    assert list(tmp_data_dir.glob("whole_process_*.parquet")) == []


def test_run_whole_process_missing_content(monkeypatch, tmp_data_dir):
    """A source lacking the contributionContent column produces no output."""
    source = tmp_data_dir / "input.parquet"
    pd.DataFrame({"comments": [["c1"]]}).to_parquet(source)

    monkeypatch.setattr(
        main, "WholeProcess", lambda *a, **k: pytest.fail("should not construct")
    )
    monkeypatch.setattr("builtins.input", lambda *a: "1")

    asyncio.run(main.run_whole_process())

    assert list(tmp_data_dir.glob("whole_process_*.parquet")) == []


def test_run_whole_process_skips_on_overwrite_decline(monkeypatch, tmp_data_dir):
    """An existing output is left untouched when the user declines to overwrite."""
    source = tmp_data_dir / "input.parquet"
    pd.DataFrame(
        {
            "contribution_id": [1],
            "contributionContent": ["Beitrag"],
            "comments": [["c1"]],
        }
    ).to_parquet(source)

    output = tmp_data_dir / "whole_process_input.parquet"
    original = pd.DataFrame({"keep": [1]})
    original.to_parquet(output)

    monkeypatch.setattr(
        main, "WholeProcess", lambda *a, **k: pytest.fail("should not construct")
    )
    # File selection "1", blank count (all), then "n" to decline the overwrite.
    answers = iter(["1", "", "n"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    asyncio.run(main.run_whole_process())

    assert pd.read_parquet(output).equals(original)


def _make_multi_row_source(tmp_data_dir, n_rows):
    """Write a parquet with ``n_rows`` contributions and return its path."""
    source = tmp_data_dir / "input.parquet"
    pd.DataFrame(
        {
            "contribution_id": list(range(1, n_rows + 1)),
            "contributionContent": [f"Beitrag {i}" for i in range(1, n_rows + 1)],
            "comments": [["c1", "c2"] for _ in range(n_rows)],
        }
    ).to_parquet(source)
    return source


def test_run_whole_process_processes_subset(monkeypatch, tmp_data_dir):
    """A count of "3" against 5 rows runs the pipeline on a 3-row sample."""
    FakeWholeProcess.run_result = None
    _make_multi_row_source(tmp_data_dir, 5)

    monkeypatch.setattr(main, "WholeProcess", FakeWholeProcess)
    inputs = iter(["1", "3"])  # file=1, count=3
    monkeypatch.setattr("builtins.input", lambda *_: next(inputs))

    asyncio.run(main.run_whole_process())

    assert len(FakeWholeProcess.last_instance.run_df) == 3


def test_run_whole_process_processes_all_on_blank(monkeypatch, tmp_data_dir):
    """A blank count against 5 rows runs the pipeline on all 5 rows."""
    FakeWholeProcess.run_result = None
    _make_multi_row_source(tmp_data_dir, 5)

    monkeypatch.setattr(main, "WholeProcess", FakeWholeProcess)
    inputs = iter(["1", ""])  # file=1, count=blank (all)
    monkeypatch.setattr("builtins.input", lambda *_: next(inputs))

    asyncio.run(main.run_whole_process())

    assert len(FakeWholeProcess.last_instance.run_df) == 5


def test_run_whole_process_invalid_count_aborts(monkeypatch, tmp_data_dir):
    """A non-numeric count aborts before any output is written."""
    _make_multi_row_source(tmp_data_dir, 5)

    monkeypatch.setattr(
        main, "WholeProcess", lambda *a, **k: pytest.fail("should not construct")
    )
    inputs = iter(["1", "abc"])  # file=1, count=abc
    monkeypatch.setattr("builtins.input", lambda *_: next(inputs))

    asyncio.run(main.run_whole_process())

    assert list(tmp_data_dir.glob("whole_process_*.parquet")) == []


def test_run_whole_process_zero_count_aborts(monkeypatch, tmp_data_dir):
    """A zero count aborts before any output is written."""
    _make_multi_row_source(tmp_data_dir, 5)

    monkeypatch.setattr(
        main, "WholeProcess", lambda *a, **k: pytest.fail("should not construct")
    )
    inputs = iter(["1", "0"])  # file=1, count=0
    monkeypatch.setattr("builtins.input", lambda *_: next(inputs))

    asyncio.run(main.run_whole_process())

    assert list(tmp_data_dir.glob("whole_process_*.parquet")) == []
