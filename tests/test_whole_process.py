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
    find_sentence_spans,
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


# --- find_sentence_spans -------------------------------------------------


def test_find_sentence_spans_single_occurrence():
    """A sentence occurring once yields a single span that slices back to it."""
    text = "Hallo Welt. Das ist ein Test."
    sentences = ["Das ist ein Test."]
    result = find_sentence_spans(text, sentences)
    assert len(result) == 1
    assert len(result[0]) == 1
    start, end = result[0][0]
    assert text[start:end] == sentences[0]


def test_find_sentence_spans_multiple_occurrences():
    """A sentence repeated in the text yields one span per occurrence."""
    text = "ab cd ab ef ab"
    sentences = ["ab"]
    result = find_sentence_spans(text, sentences)
    assert len(result) == 1
    assert len(result[0]) == 3
    for start, end in result[0]:
        assert text[start:end] == "ab"


def test_find_sentence_spans_not_found_returns_none():
    """A sentence absent from the text maps to None."""
    text = "Hallo Welt."
    result = find_sentence_spans(text, ["nicht vorhanden"])
    assert result[0] is None


def test_find_sentence_spans_empty_list():
    """An empty sentence list yields an empty result list."""
    assert find_sentence_spans("irgendein Text", []) == []


def test_find_sentence_spans_empty_sentence_string():
    """An empty sentence maps to None (guards the ''.find('') infinite loop)."""
    assert find_sentence_spans("irgendein Text", [""]) == [None]


def test_find_sentence_spans_mixed():
    """Found, missing, and empty sentences resolve independently in one call."""
    text = "foo bar foo baz"
    sentences = ["foo", "missing", "baz", ""]
    result = find_sentence_spans(text, sentences)
    assert len(result) == len(sentences)
    # "foo" appears twice.
    assert len(result[0]) == 2
    for start, end in result[0]:
        assert text[start:end] == "foo"
    # "missing" is absent.
    assert result[1] is None
    # "baz" appears once.
    assert len(result[2]) == 1
    start, end = result[2][0]
    assert text[start:end] == "baz"
    # Empty string guard.
    assert result[3] is None


# --- WholeProcess.run ----------------------------------------------------


def test_run_flattens_rows(monkeypatch):
    """Two key messages x two comments yields four flattened rows."""
    km_list = [
        {
            "key_message": "km1",
            "key_message_type": "Wunsch",
            "related_sentences": ["s1a", "s1b"],
            "explanation": "km expl 1",
        },
        {
            "key_message": "km2",
            "key_message_type": "Kritik",
            "related_sentences": ["s2a"],
            "explanation": "km expl 2",
        },
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
    assert set(result["contributionContent"]) == {"Beitrag"}
    assert set(result["key_message_explanation"]) == {"km expl 1", "km expl 2"}
    assert "related_sentences" in result.columns
    # Per-row related_sentences track the originating key message.
    expected_rs = {"km1": ["s1a", "s1b"], "km2": ["s2a"]}
    for _, row in result.iterrows():
        assert list(row["related_sentences"]) == expected_rs[row["key_message"]]


def test_run_includes_related_sentences(monkeypatch):
    """Every (km x comment) row carries the km's related_sentences as a list."""
    related = ["alpha", "beta", "gamma"]
    km_list = [
        {
            "key_message": "km1",
            "key_message_type": "t",
            "related_sentences": related,
            "explanation": "km expl",
        }
    ]
    processor = _make_processor(monkeypatch, km_list)
    df = pd.DataFrame(
        {
            "contribution_id": [1],
            "contributionContent": ["Beitrag"],
            "comments": [["c1", "c2", "c3"]],
        }
    )

    result = asyncio.run(processor.run(df))

    assert len(result) == 3
    for value in result["related_sentences"]:
        assert isinstance(value, list)
        assert value == related


def test_run_related_sentences_defaults_when_missing(monkeypatch):
    """A km without a related_sentences key yields an empty list in the row."""
    km_list = [{"key_message": "km1", "key_message_type": "t", "explanation": "km expl"}]
    processor = _make_processor(monkeypatch, km_list)
    df = pd.DataFrame(
        {
            "contribution_id": [1],
            "contributionContent": ["Beitrag"],
            "comments": [["a longer comment here"]],
        }
    )

    result = asyncio.run(processor.run(df))

    assert result.iloc[0]["related_sentences"] == []


def test_run_related_sentences_survive_parquet_roundtrip(monkeypatch, tmp_path):
    """related_sentences persists as a list-valued column through parquet I/O."""
    related = ["s1", "s2"]
    km_list = [
        {
            "key_message": "km1",
            "key_message_type": "t",
            "related_sentences": related,
            "explanation": "km expl",
        }
    ]
    processor = _make_processor(monkeypatch, km_list)
    df = pd.DataFrame(
        {
            "contribution_id": [1],
            "contributionContent": ["Beitrag"],
            "comments": [["c1"]],
        }
    )

    result = asyncio.run(processor.run(df))

    out_path = tmp_path / "out.parquet"
    result.to_parquet(out_path)
    read_back = pd.read_parquet(out_path)

    assert "related_sentences" in read_back.columns
    assert list(read_back.iloc[0]["related_sentences"]) == related


def test_run_includes_sentence_spans(monkeypatch):
    """Spans are computed from contributionContent and shared across rows."""
    contribution = "Wir brauchen mehr Bäume. Der Park ist schön."
    km_list = [
        {
            "key_message": "km1",
            "key_message_type": "t",
            "related_sentences": ["Wir brauchen mehr Bäume.", "Der Park ist schön."],
            "explanation": "km expl",
        }
    ]
    processor = _make_processor(monkeypatch, km_list)
    df = pd.DataFrame(
        {
            "contribution_id": [1],
            "contributionContent": [contribution],
            "comments": [["c1", "c2"]],
        }
    )

    result = asyncio.run(processor.run(df))

    assert "related_sentence_spans" in result.columns
    assert len(result) == 2
    expected = [
        [(0, len("Wir brauchen mehr Bäume."))],
        [(25, 25 + len("Der Park ist schön."))],
    ]
    for _, row in result.iterrows():
        spans = row["related_sentence_spans"]
        assert spans == expected
        # Slicing invariant holds for every recorded span.
        for sentence, sentence_spans in zip(
            row["related_sentences"], spans, strict=True
        ):
            for start, end in sentence_spans:
                assert contribution[start:end] == sentence
    # Spans are identical across comment rows for the same key message.
    assert result.iloc[0]["related_sentence_spans"] == result.iloc[1][
        "related_sentence_spans"
    ]


def test_run_sentence_spans_none_when_not_found(monkeypatch):
    """A related sentence absent from the contribution yields a None span."""
    km_list = [
        {
            "key_message": "km1",
            "key_message_type": "t",
            "related_sentences": ["nicht im Beitrag enthalten"],
            "explanation": "km expl",
        }
    ]
    processor = _make_processor(monkeypatch, km_list)
    df = pd.DataFrame(
        {
            "contribution_id": [1],
            "contributionContent": ["Ein völlig anderer Beitrag."],
            "comments": [["c1"]],
        }
    )

    result = asyncio.run(processor.run(df))

    assert result.iloc[0]["related_sentence_spans"] == [None]


def test_run_sentence_spans_survive_parquet_roundtrip(monkeypatch, tmp_path):
    """related_sentence_spans persists through parquet I/O."""
    contribution = "Satz eins. Satz zwei."
    km_list = [
        {
            "key_message": "km1",
            "key_message_type": "t",
            "related_sentences": ["Satz eins.", "fehlt"],
            "explanation": "km expl",
        }
    ]
    processor = _make_processor(monkeypatch, km_list)
    df = pd.DataFrame(
        {
            "contribution_id": [1],
            "contributionContent": [contribution],
            "comments": [["c1"]],
        }
    )

    result = asyncio.run(processor.run(df))

    out_path = tmp_path / "out.parquet"
    result.to_parquet(out_path)
    read_back = pd.read_parquet(out_path)

    assert "related_sentence_spans" in read_back.columns

    def _normalize(spans):
        # pyarrow may round-trip tuples as numpy arrays / lists; preserve None.
        if spans is None:
            return None
        return [
            None if s is None else [tuple(int(x) for x in pair) for pair in s]
            for s in spans
        ]

    expected = _normalize(result.iloc[0]["related_sentence_spans"])
    assert _normalize(read_back.iloc[0]["related_sentence_spans"]) == expected


def test_run_fast_path_no_llm_call(monkeypatch):
    """A short confident-positive comment is fast-pathed without an LLM call."""
    classifier = FakeClassifier()
    classifier.result = {"label": "ja", "score": 0.9}
    called = {"hit": False}

    async def call_model(**kwargs):
        called["hit"] = True
        return {"stance": "x", "explanation": "y"}

    km_list = [{"key_message": "km1", "key_message_type": "t", "explanation": "km expl"}]
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
    assert result.iloc[0]["contributionContent"] == "Beitrag"
    assert result.iloc[0]["key_message_explanation"] == "km expl"


def test_run_ensemble_hint_present(monkeypatch):
    """A long confident-positive comment routes through the ensemble path with hint."""
    classifier = FakeClassifier()
    classifier.result = {"label": "ja", "score": 0.9}
    captured = {}

    async def call_model(**kwargs):
        captured.update(kwargs)
        return {"stance": "in favor", "explanation": "reason"}

    km_list = [{"key_message": "km1", "key_message_type": "t", "explanation": "km expl"}]
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
    km_list = [{"key_message": "km1", "key_message_type": "t", "explanation": "km expl"}]
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
    assert result.iloc[0]["contributionContent"] == "Beitrag"
    assert result.iloc[0]["key_message_explanation"] == "km expl"


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
    km_list = [{"key_message": "km1", "key_message_type": "t", "explanation": "km expl"}]

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


def test_api_key_defaults_to_empty_string():
    """A WholeProcess built without api_key stores an empty string (no secret)."""
    processor = WholeProcess(base_url="http://testserver/v1", model_name="test-model")
    assert processor._api_key == ""


def test_api_key_stored_and_forwarded(monkeypatch):
    """api_key is stored and threaded into the stance-resolution LLM call."""
    captured = {}

    async def call_model(**kwargs):
        captured.update(kwargs)
        return {"stance": "in favor", "explanation": "reason"}

    processor = WholeProcess(
        base_url="http://testserver/v1", model_name="test-model", api_key="tok"
    )
    assert processor._api_key == "tok"
    # The extractor must receive the same key.
    assert processor._extractor._api_key == "tok"

    processor._extractor = FakeExtractor([{"key_message": "km1", "key_message_type": "t"}])
    processor._classifier = FakeClassifier()
    monkeypatch.setattr(processor._llm_handler, "call_model", call_model)

    df = pd.DataFrame(
        {
            "contribution_id": [1],
            "contributionContent": ["Beitrag"],
            "comments": [["a longer comment here"]],
        }
    )
    asyncio.run(processor.run(df))
    assert captured["api_key"] == "tok"


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
