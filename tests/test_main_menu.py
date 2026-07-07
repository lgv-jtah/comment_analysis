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
        {"project_id": ["1", "1", "2"], "contribution_id": [10, 11, 12]}
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
        {"project_id": ["1", "1", "2"], "contribution_id": [10, 11, 12]}
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


# --- _sanitize_model_name ---


def test_sanitize_model_name_slashes_and_dots():
    """Slashes collapse to single hyphens, keeping alphanumerics intact."""
    assert main._sanitize_model_name("Qwen/Qwen3-32B") == "Qwen-Qwen3-32B"


def test_sanitize_model_name_collapses_symbol_runs():
    """Runs of slashes, dots, and colons collapse to a single hyphen each."""
    assert main._sanitize_model_name("a/./:b") == "a-b"
    assert main._sanitize_model_name("model.v1:2") == "model-v1-2"


def test_sanitize_model_name_strips_leading_trailing():
    """Leading/trailing separators are stripped, not left as hyphens."""
    assert main._sanitize_model_name("/Qwen/") == "Qwen"


def test_sanitize_model_name_all_symbols_maps_to_model():
    """An all-symbol or empty input maps to the literal 'model'."""
    assert main._sanitize_model_name("///...") == "model"
    assert main._sanitize_model_name("") == "model"


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
    assert list(written.columns) == ["project_id", "contribution_id"]


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
    assert list(written.columns) == ["project_id", "contribution_id"]
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


# --- run_stance_detection ---


class FakeStanceDetector:
    """Network-free stand-in for ``stance_detector.StanceDetector``."""

    data = []

    def __init__(self, *args, **kwargs):
        pass

    def load_data(self):
        return type(self).data

    def load_prompt_config(self, path, key):
        return {"key": key}

    def format_gr_hint(self, score):
        return f"hint:{score}"

    def build_messages(self, *, prompt_config, item, gr_hint=None):
        return "system", [{"role": "user", "content": item.get("comment_text", "")}]


class FakeLLMHandler:
    """Network-free stand-in for ``llm_handler.LLMHandler``.

    ``call_results`` is consumed one entry per call so a test can return a real
    result for one branch and ``None`` for another to exercise both print paths.
    """

    call_results = []
    call_index = 0

    def __init__(self, *args, **kwargs):
        pass

    async def call_model(self, **kwargs):
        cls = type(self)
        result = cls.call_results[cls.call_index]
        cls.call_index += 1
        return result


class FakeEvaluator:
    """Network-free stand-in for ``evaluator.StanceEvaluator``."""

    def __init__(self, *args, **kwargs):
        pass

    def evaluate(self, predictions):
        return {"predictions": predictions}

    def format_report(self, report):
        return "report-text"


def test_run_stance_detection_runs_all_three_paths(monkeypatch, tmp_path):
    """run_stance_detection routes rows through fast-path, ensemble, and LLM branches."""
    import asyncio

    # Three rows: one fast-path (short, confident "ja"), one ensemble (long,
    # confident "ja"), one plain LLM (label not "ja"). The ensemble call returns
    # a real result; the LLM call returns None to cover the FAILED print branch.
    FakeStanceDetector.data = [
        {
            "row_index": 1,
            "comment_text": "ja gut",
            "general_reaction_label": "ja",
            "general_reaction_score": 0.99,
            "gold_stance": "neutral",
        },
        {
            "row_index": 2,
            "comment_text": "this is a much longer comment that exceeds the word limit easily",
            "general_reaction_label": "ja",
            "general_reaction_score": 0.99,
            "gold_stance": "pro",
        },
        {
            "row_index": 3,
            "comment_text": "neutral text here",
            "general_reaction_label": "nein",
            "general_reaction_score": 0.1,
            "gold_stance": "contra",
        },
    ]
    monkeypatch.setattr(main, "StanceDetector", FakeStanceDetector)
    monkeypatch.setattr(main, "LLMHandler", FakeLLMHandler)
    monkeypatch.setattr(main, "StanceEvaluator", FakeEvaluator)
    monkeypatch.setattr(
        main, "write_markdown_report", lambda report, metadata: tmp_path / "report.md"
    )
    monkeypatch.setattr(
        main,
        "write_result_json",
        lambda data, predictions, **kwargs: tmp_path / "result.json",
    )

    # First run: ensemble succeeds, LLM fails — covers the ensemble success and
    # LLM FAILED print branches.
    FakeLLMHandler.call_results = [{"stance": "pro"}, None]
    FakeLLMHandler.call_index = 0
    asyncio.run(main.run_stance_detection())
    assert FakeLLMHandler.call_index == 2

    # Second run: ensemble fails, LLM succeeds — covers the ensemble FAILED and
    # LLM success print branches.
    FakeLLMHandler.call_results = [None, {"stance": "pro"}]
    FakeLLMHandler.call_index = 0
    asyncio.run(main.run_stance_detection())
    assert FakeLLMHandler.call_index == 2


# --- sample contributions ---


class FakeSampler:
    """Network-free stand-in for ``sample_contributions.SampleContributions``.

    Class-level attributes let each test configure the return value / raised
    exception and inspect how the sampler was constructed and called.
    """

    init_args = None
    sample_called = False
    sample_return = "/tmp/sample_5.parquet"
    sample_raises = None

    def __init__(self, *args, **kwargs):
        type(self).init_args = (args, kwargs)

    def sample(self):
        type(self).sample_called = True
        if type(self).sample_raises is not None:
            raise type(self).sample_raises
        return type(self).sample_return


@pytest.fixture
def fresh_sampler():
    """Reset FakeSampler class-level state and yield the class."""
    FakeSampler.init_args = None
    FakeSampler.sample_called = False
    FakeSampler.sample_return = "/tmp/sample_5.parquet"
    FakeSampler.sample_raises = None
    return FakeSampler


def test_run_menu_returns_3(monkeypatch):
    """run_menu returns "3" when the user selects option 3."""
    monkeypatch.setattr("builtins.input", lambda *a: "3")
    assert main.run_menu() == "3"


def test_main_dispatches_to_sample(monkeypatch):
    """main routes choice "3" to sample_contributions."""
    monkeypatch.setattr(main, "run_menu", lambda: "3")
    called = {"sample": False}
    monkeypatch.setattr(
        main, "sample_contributions", lambda: called.__setitem__("sample", True)
    )
    main.main()
    assert called["sample"] is True


def test_prompt_positive_int_valid(monkeypatch):
    """_prompt_positive_int returns the parsed integer for valid input."""
    monkeypatch.setattr("builtins.input", lambda *a: "5")
    assert main._prompt_positive_int("n: ") == 5


def test_prompt_positive_int_reprompts_on_nonnumeric(monkeypatch):
    """_prompt_positive_int re-prompts when the input is not a number."""
    answers = iter(["abc", "5"])
    calls = {"n": 0}

    def fake_input(*a):
        calls["n"] += 1
        return next(answers)

    monkeypatch.setattr("builtins.input", fake_input)
    assert main._prompt_positive_int("n: ") == 5
    assert calls["n"] == 2


def test_prompt_positive_int_reprompts_on_nonpositive(monkeypatch):
    """_prompt_positive_int re-prompts on zero or negative input."""
    answers = iter(["0", "-1", "3"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    assert main._prompt_positive_int("n: ") == 3


def test_prompt_project_ids_comma_separated(monkeypatch):
    """_prompt_project_ids splits a comma-separated list and strips tokens."""
    monkeypatch.setattr("builtins.input", lambda *a: "A, B")
    assert main._prompt_project_ids(["A", "B", "C"]) == ["A", "B"]


def test_prompt_project_ids_all(monkeypatch):
    """_prompt_project_ids returns every available ID for input "all"."""
    monkeypatch.setattr("builtins.input", lambda *a: "all")
    available = ["A", "B", "C"]
    assert main._prompt_project_ids(available) == available


def test_prompt_project_ids_reprompts_on_empty(monkeypatch):
    """_prompt_project_ids re-prompts until a non-empty selection is given."""
    answers = iter(["", "  ", "A"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))
    assert main._prompt_project_ids(["A", "B"]) == ["A"]


def test_sample_contributions_happy_path(
    monkeypatch, tmp_paths, fresh_sampler
):
    """sample_contributions builds the sampler with the chosen IDs and calls sample."""
    data_dir, contrib_path = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"project_id": ["A", "B"], "contribution_id": [1, 2]}).to_parquet(contrib_path)

    monkeypatch.setattr(main, "SampleContributions", fresh_sampler)
    answers = iter(["A, B", "5"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    main.sample_contributions()

    assert fresh_sampler.sample_called is True
    _, kwargs = fresh_sampler.init_args
    assert kwargs["project_ids"] == ["A", "B"]
    assert kwargs["n"] == 5
    assert kwargs["source_path"] == contrib_path


def test_sample_contributions_missing_file(
    monkeypatch, tmp_paths, fresh_sampler
):
    """sample_contributions aborts (no sampler) when the file is missing."""
    monkeypatch.setattr(main, "SampleContributions", fresh_sampler)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no input expected"))

    main.sample_contributions()

    assert fresh_sampler.init_args is None
    assert fresh_sampler.sample_called is False


def test_sample_contributions_missing_project_id_column(
    monkeypatch, tmp_paths, fresh_sampler
):
    """sample_contributions aborts (no sampler) when the project_id column is absent."""
    data_dir, contrib_path = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"other": [1, 2]}).to_parquet(contrib_path)

    monkeypatch.setattr(main, "SampleContributions", fresh_sampler)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no input expected"))

    main.sample_contributions()

    assert fresh_sampler.init_args is None
    assert fresh_sampler.sample_called is False


def test_sample_contributions_catches_valueerror(
    monkeypatch, tmp_paths, fresh_sampler
):
    """sample_contributions swallows a ValueError raised by the sampler."""
    data_dir, contrib_path = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"project_id": ["A"], "contribution_id": [1]}).to_parquet(contrib_path)

    fresh_sampler.sample_raises = ValueError("boom")
    monkeypatch.setattr(main, "SampleContributions", fresh_sampler)
    answers = iter(["A", "5"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    # Should not raise.
    main.sample_contributions()
    assert fresh_sampler.sample_called is True


def test_sample_contributions_catches_filenotfounderror(
    monkeypatch, tmp_paths, fresh_sampler
):
    """sample_contributions swallows a FileNotFoundError raised by the sampler."""
    data_dir, contrib_path = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"project_id": ["A"], "contribution_id": [1]}).to_parquet(contrib_path)

    fresh_sampler.sample_raises = FileNotFoundError("gone")
    monkeypatch.setattr(main, "SampleContributions", fresh_sampler)
    answers = iter(["A", "5"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    # Should not raise.
    main.sample_contributions()
    assert fresh_sampler.sample_called is True


# --- key message extraction ---


class FakeExtractor:
    """Network-free stand-in for ``key_message_extractor.KeyMessageExtraction``.

    Class-level attributes capture the constructor kwargs, record every
    ``extract`` call's contribution, and let each test configure the returned
    value (including ``None`` to simulate a failed extraction).
    """

    init_args = None
    extract_return = [
        {
            "key_message": "km",
            "key_message_type": "Vorschlag",
            "related_sentences": ["s"],
            "explanation": "why",
        }
    ]
    extract_calls = []

    def __init__(self, *args, **kwargs):
        FakeExtractor.init_args = kwargs

    async def extract(
        self,
        *,
        client,
        contribution,
        proceeding_description,
        semaphore,
        enable_thinking=False,
    ):
        FakeExtractor.extract_calls.append(contribution)
        return FakeExtractor.extract_return


@pytest.fixture
def fresh_extractor():
    """Reset FakeExtractor class-level state and yield the class."""
    FakeExtractor.init_args = None
    FakeExtractor.extract_return = [
        {
            "key_message": "km",
            "key_message_type": "Vorschlag",
            "related_sentences": ["s"],
            "explanation": "why",
        }
    ]
    FakeExtractor.extract_calls = []
    return FakeExtractor


def test_run_menu_returns_4(monkeypatch):
    """run_menu returns "4" when the user selects option 4."""
    monkeypatch.setattr("builtins.input", lambda *a: "4")
    assert main.run_menu() == "4"


def test_main_dispatches_to_key_message_extraction(monkeypatch):
    """main routes choice "4" to run_key_message_extraction via asyncio.run."""
    monkeypatch.setattr(main, "run_menu", lambda: "4")
    called = {"run": False}

    async def fake_run():
        called["run"] = True

    monkeypatch.setattr(main, "run_key_message_extraction", fake_run)
    main.main()
    assert called["run"] is True


def test_pick_parquet_file_returns_selection(monkeypatch, tmp_path):
    """_pick_parquet_file returns the Path matching the entered index."""
    first = tmp_path / "a.parquet"
    second = tmp_path / "b.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(first)
    pd.DataFrame({"x": [2]}).to_parquet(second)

    files = sorted(tmp_path.glob("*.parquet"))
    target = files[1]
    index = str(files.index(target) + 1)
    monkeypatch.setattr("builtins.input", lambda *a: index)

    assert main._pick_parquet_file(tmp_path) == target


def test_pick_parquet_file_reprompts_on_out_of_range(monkeypatch, tmp_path):
    """_pick_parquet_file re-prompts on an out-of-range index then returns."""
    only = tmp_path / "a.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(only)

    answers = iter(["5", "1"])
    calls = {"n": 0}

    def fake_input(*a):
        calls["n"] += 1
        return next(answers)

    monkeypatch.setattr("builtins.input", fake_input)
    assert main._pick_parquet_file(tmp_path) == only
    assert calls["n"] == 2


def test_pick_parquet_file_no_files(monkeypatch, tmp_path):
    """_pick_parquet_file returns None without prompting when no files exist."""
    missing = tmp_path / "does_not_exist"
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no input expected"))
    assert main._pick_parquet_file(missing) is None


def test_run_key_message_extraction_aborts_without_files(
    monkeypatch, tmp_paths, fresh_extractor
):
    """run_key_message_extraction aborts (no extractor) when no files exist."""
    import asyncio

    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "KeyMessageExtraction", fresh_extractor)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no input expected"))

    asyncio.run(main.run_key_message_extraction())

    assert fresh_extractor.init_args is None
    assert list(data_dir.glob("key_messages_*.parquet")) == []


def test_run_key_message_extraction_happy_path(
    monkeypatch, tmp_paths, fresh_extractor
):
    """run_key_message_extraction writes a parquet with the key_messages column."""
    import asyncio

    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "input.parquet"
    pd.DataFrame({"contributionContent": ["hello", "world"]}).to_parquet(source)

    monkeypatch.setattr(main, "KeyMessageExtraction", fresh_extractor)
    answers = iter(["1", "some context"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    asyncio.run(main.run_key_message_extraction())

    slug = main._sanitize_model_name(main.EXTRACTION_LLM_MODEL_NAME)
    output = data_dir / f"key_messages_input_{slug}.parquet"
    assert output.exists()
    written = pd.read_parquet(output)
    assert "key_messages" in written.columns
    assert [list(v) for v in written["key_messages"]] == [
        fresh_extractor.extract_return,
        fresh_extractor.extract_return,
    ]
    assert fresh_extractor.extract_calls == ["hello", "world"]
    # Every row records the extraction model name for provenance.
    assert list(written["extraction_model_name"]) == [
        main.EXTRACTION_LLM_MODEL_NAME,
        main.EXTRACTION_LLM_MODEL_NAME,
    ]


def test_run_key_message_extraction_handles_none_result(
    monkeypatch, tmp_paths, fresh_extractor
):
    """run_key_message_extraction stores None when extract returns None."""
    import asyncio

    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "input.parquet"
    pd.DataFrame({"contributionContent": ["hello"]}).to_parquet(source)

    fresh_extractor.extract_return = None
    monkeypatch.setattr(main, "KeyMessageExtraction", fresh_extractor)
    answers = iter(["1", ""])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    asyncio.run(main.run_key_message_extraction())

    slug = main._sanitize_model_name(main.EXTRACTION_LLM_MODEL_NAME)
    output = data_dir / f"key_messages_input_{slug}.parquet"
    written = pd.read_parquet(output)
    assert written["key_messages"].isna().all()


def test_run_key_message_extraction_aborts_on_missing_column(
    monkeypatch, tmp_paths, fresh_extractor
):
    """run_key_message_extraction aborts (no extractor, no output) when the source lacks contributionContent."""
    import asyncio

    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "input.parquet"
    pd.DataFrame({"other_col": [1, 2]}).to_parquet(source)

    monkeypatch.setattr(main, "KeyMessageExtraction", fresh_extractor)
    answers = iter(["1", ""])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    asyncio.run(main.run_key_message_extraction())

    assert fresh_extractor.init_args is None
    assert list(data_dir.glob("key_messages_*.parquet")) == []


def test_pick_parquet_file_reprompts_on_non_integer(monkeypatch, tmp_path):
    """_pick_parquet_file re-prompts on non-integer input then returns the selection."""
    only = tmp_path / "a.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(only)

    calls = {"n": 0}
    answers = iter(["abc", "1"])

    def fake_input(*a):
        calls["n"] += 1
        return next(answers)

    monkeypatch.setattr("builtins.input", fake_input)
    assert main._pick_parquet_file(tmp_path) == only
    assert calls["n"] == 2


def test_run_key_message_extraction_skips_on_overwrite_decline(
    monkeypatch, tmp_paths, fresh_extractor
):
    """run_key_message_extraction leaves output untouched and never extracts on decline."""
    import asyncio

    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "input.parquet"
    pd.DataFrame({"contributionContent": ["hello"]}).to_parquet(source)

    slug = main._sanitize_model_name(main.EXTRACTION_LLM_MODEL_NAME)
    output = data_dir / f"key_messages_input_{slug}.parquet"
    original = pd.DataFrame({"keep": [1]})
    original.to_parquet(output)

    monkeypatch.setattr(main, "KeyMessageExtraction", fresh_extractor)
    # File selection "1", then "n" to decline the overwrite prompt.
    answers = iter(["1", "", "n"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    asyncio.run(main.run_key_message_extraction())

    assert pd.read_parquet(output).equals(original)
    assert fresh_extractor.extract_calls == []


# --- inspect key messages ---


def _make_key_message(text="km"):
    """Return a single valid key-message dict for inspection tests."""
    return {
        "key_message": text,
        "key_message_type": "Vorschlag",
        "related_sentences": ["s1", "s2"],
        "explanation": "why",
    }


def test_run_menu_returns_5(monkeypatch):
    """run_menu returns "5" when the user selects option 5."""
    monkeypatch.setattr("builtins.input", lambda *a: "5")
    assert main.run_menu() == "5"


def test_main_dispatches_to_inspect(monkeypatch):
    """main routes choice "5" to inspect_key_messages."""
    monkeypatch.setattr(main, "run_menu", lambda: "5")
    called = {"inspect": False}
    monkeypatch.setattr(
        main, "inspect_key_messages", lambda: called.__setitem__("inspect", True)
    )
    main.main()
    assert called["inspect"] is True


def test_inspect_aborts_without_files(monkeypatch, tmp_paths):
    """inspect_key_messages returns cleanly (no input) when no files exist."""
    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no input expected"))

    main.inspect_key_messages()


def test_inspect_missing_key_messages_column(monkeypatch, tmp_paths):
    """inspect_key_messages aborts when the source lacks key_messages."""
    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "input.parquet"
    pd.DataFrame({"contributionContent": ["hello"]}).to_parquet(source)

    monkeypatch.setattr("builtins.input", lambda *a: "1")

    main.inspect_key_messages()


def test_inspect_missing_contribution_column(monkeypatch, tmp_paths):
    """inspect_key_messages aborts when the source lacks contributionContent."""
    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "input.parquet"
    df = pd.DataFrame({"key_messages": [[_make_key_message()]]})
    df.to_parquet(source)

    monkeypatch.setattr("builtins.input", lambda *a: "1")

    main.inspect_key_messages()


def test_inspect_happy_path_quit(monkeypatch, tmp_paths):
    """inspect_key_messages renders a row then quits on "q"."""
    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "input.parquet"
    df = pd.DataFrame(
        {
            "contributionContent": ["hello"],
            "key_messages": [[_make_key_message()]],
        }
    )
    df.to_parquet(source)

    answers = iter(["1", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    main.inspect_key_messages()


def test_inspect_next_advances(monkeypatch, tmp_paths):
    """inspect_key_messages advances to the next row on "n"."""
    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "input.parquet"
    df = pd.DataFrame(
        {
            "contributionContent": ["hello", "world"],
            "key_messages": [[_make_key_message("a")], [_make_key_message("b")]],
        }
    )
    df.to_parquet(source)

    answers = iter(["1", "n", "q"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    main.inspect_key_messages()


def test_inspect_handles_failed_row(monkeypatch, tmp_paths):
    """inspect_key_messages renders FAILED for a None key_messages cell."""
    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "input.parquet"
    df = pd.DataFrame(
        {
            "contributionContent": ["hello"],
            "key_messages": pd.Series([None], dtype="object"),
        }
    )
    df.to_parquet(source)

    answers = iter(["1", "n"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    main.inspect_key_messages()


def test_inspect_handles_nan_contribution(monkeypatch, tmp_paths):
    """inspect_key_messages does not crash on a NaN contribution cell."""
    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "input.parquet"
    df = pd.DataFrame(
        {
            "contributionContent": pd.Series([None], dtype="object"),
            "key_messages": [[_make_key_message()]],
        }
    )
    df.to_parquet(source)

    answers = iter(["1", "n"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    main.inspect_key_messages()


# --- inspect parquet ---


def test_run_menu_returns_8(monkeypatch):
    """run_menu returns "8" when the user selects option 8."""
    monkeypatch.setattr("builtins.input", lambda *a: "8")
    assert main.run_menu() == "8"


def test_main_dispatches_to_inspect_parquet(monkeypatch):
    """main routes choice "8" to inspect_parquet."""
    monkeypatch.setattr(main, "run_menu", lambda: "8")
    called = {"inspect": False}
    monkeypatch.setattr(
        main, "inspect_parquet", lambda: called.__setitem__("inspect", True)
    )
    main.main()
    assert called["inspect"] is True


def test_inspect_parquet_aborts_without_files(monkeypatch, tmp_paths):
    """inspect_parquet returns cleanly (no input) when no files exist."""
    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no input expected"))

    main.inspect_parquet()


def test_inspect_parquet_happy_path(monkeypatch, tmp_paths):
    """inspect_parquet renders metadata and a table for a 12-row file."""
    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "sample.parquet"
    pd.DataFrame(
        {"id": list(range(12)), "text": [f"row {i}" for i in range(12)]}
    ).to_parquet(source)

    monkeypatch.setattr("builtins.input", lambda *a: "1")

    main.inspect_parquet()


def test_inspect_parquet_truncates_long_cells(monkeypatch, tmp_paths, capsys):
    """inspect_parquet truncates a long cell rather than printing it verbatim."""
    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "long.parquet"
    long_value = "a" * 200
    pd.DataFrame({"id": [1], "text": [long_value]}).to_parquet(source)

    monkeypatch.setattr("builtins.input", lambda *a: "1")

    main.inspect_parquet()

    out = capsys.readouterr().out
    # The full 200-char string must not appear verbatim; truncation kicked in.
    assert long_value not in out
    assert "…" in out


def test_inspect_parquet_handles_nan_cells(monkeypatch, tmp_paths):
    """inspect_parquet does not crash on a NaN/None cell."""
    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "nan.parquet"
    pd.DataFrame(
        {"id": [1, 2], "text": pd.Series(["hello", None], dtype="object")}
    ).to_parquet(source)

    monkeypatch.setattr("builtins.input", lambda *a: "1")

    main.inspect_parquet()


def test_inspect_parquet_empty_dataframe(monkeypatch, tmp_paths):
    """inspect_parquet renders cleanly for a file with columns but no rows."""
    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "empty.parquet"
    pd.DataFrame(
        {
            "id": pd.Series([], dtype="int64"),
            "text": pd.Series([], dtype="object"),
        }
    ).to_parquet(source)

    monkeypatch.setattr("builtins.input", lambda *a: "1")

    main.inspect_parquet()


# --- export parquet to excel ---


def test_run_menu_returns_10(monkeypatch):
    """run_menu returns "10" when the user selects option 10."""
    monkeypatch.setattr("builtins.input", lambda *a: "10")
    assert main.run_menu() == "10"


def test_main_dispatches_to_export_excel(monkeypatch):
    """main routes choice "10" to export_parquet_to_excel."""
    monkeypatch.setattr(main, "run_menu", lambda: "10")
    called = {"export": False}
    monkeypatch.setattr(
        main, "export_parquet_to_excel", lambda: called.__setitem__("export", True)
    )
    main.main()
    assert called["export"] is True


def test_export_parquet_to_excel_aborts_without_files(monkeypatch, tmp_paths):
    """export_parquet_to_excel returns cleanly (no input) when no files exist."""
    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no input expected"))

    main.export_parquet_to_excel()


def test_export_parquet_to_excel_writes_xlsx(monkeypatch, tmp_paths):
    """export_parquet_to_excel writes an .xlsx file matching the source data."""
    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "sample.parquet"
    pd.DataFrame({"id": [1, 2], "text": ["a", "b"]}).to_parquet(source)

    monkeypatch.setattr("builtins.input", lambda *a: "1")

    main.export_parquet_to_excel()

    output = data_dir / "sample.xlsx"
    assert output.exists()
    written = pd.read_excel(output)
    assert written.to_dict("list") == {"id": [1, 2], "text": ["a", "b"]}


# --- key message judge ---


def _make_km(text="km", km_type="Wunsch"):
    """Return a single valid key-message dict for judge tests."""
    return {
        "key_message": text,
        "key_message_type": km_type,
        "related_sentences": ["s1"],
        "explanation": "why",
    }


class FakeJudge:
    """Network-free stand-in for ``key_message_judge.KeyMessageJudge``.

    Class-level attributes capture the constructor kwargs, record every
    ``judge`` call's key-message list, and let each test configure the returned
    judgment (including ``None`` to simulate a failed judge call).
    """

    init_args = None
    judge_return = {"overall_score": 4, "overall_assessment": "gut"}
    judge_calls = []

    def __init__(self, *args, **kwargs):
        FakeJudge.init_args = kwargs

    async def judge(
        self,
        *,
        client,
        contribution,
        category,
        key_messages,
        semaphore,
        enable_thinking=False,
    ):
        FakeJudge.judge_calls.append(key_messages)
        return FakeJudge.judge_return


@pytest.fixture
def fresh_judge():
    """Reset FakeJudge class-level state and yield the class."""
    FakeJudge.init_args = None
    FakeJudge.judge_return = {"overall_score": 4, "overall_assessment": "gut"}
    FakeJudge.judge_calls = []
    return FakeJudge


def test_run_menu_returns_11(monkeypatch):
    """run_menu returns "11" when the user selects option 11."""
    monkeypatch.setattr("builtins.input", lambda *a: "11")
    assert main.run_menu() == "11"


def test_main_dispatches_to_key_message_judge(monkeypatch):
    """main routes choice "11" to run_key_message_judge via asyncio.run."""
    monkeypatch.setattr(main, "run_menu", lambda: "11")
    called = {"run": False}

    async def fake_run():
        called["run"] = True

    monkeypatch.setattr(main, "run_key_message_judge", fake_run)
    main.main()
    assert called["run"] is True


def test_run_key_message_judge_aborts_without_files(
    monkeypatch, tmp_paths, fresh_judge
):
    """run_key_message_judge aborts (no judge) when no files exist."""
    import asyncio

    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main, "KeyMessageJudge", fresh_judge)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("no input expected"))

    asyncio.run(main.run_key_message_judge())

    assert fresh_judge.init_args is None
    assert list(data_dir.glob("judged_*.parquet")) == []


def test_run_key_message_judge_aborts_on_missing_key_messages(
    monkeypatch, tmp_paths, fresh_judge
):
    """run_key_message_judge aborts when the source lacks key_messages."""
    import asyncio

    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "input.parquet"
    pd.DataFrame({"contributionContent": ["hello"]}).to_parquet(source)

    monkeypatch.setattr(main, "KeyMessageJudge", fresh_judge)
    monkeypatch.setattr("builtins.input", lambda *a: "1")

    asyncio.run(main.run_key_message_judge())

    assert fresh_judge.init_args is None
    assert list(data_dir.glob("judged_*.parquet")) == []


def test_run_key_message_judge_aborts_on_missing_contribution(
    monkeypatch, tmp_paths, fresh_judge
):
    """run_key_message_judge aborts when the source lacks contributionContent."""
    import asyncio

    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "input.parquet"
    pd.DataFrame({"key_messages": [[_make_km()]]}).to_parquet(source)

    monkeypatch.setattr(main, "KeyMessageJudge", fresh_judge)
    monkeypatch.setattr("builtins.input", lambda *a: "1")

    asyncio.run(main.run_key_message_judge())

    assert fresh_judge.init_args is None
    assert list(data_dir.glob("judged_*.parquet")) == []


def test_run_key_message_judge_happy_path(monkeypatch, tmp_paths, fresh_judge):
    """run_key_message_judge writes a parquet with the judge_result column."""
    import asyncio

    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "input.parquet"
    pd.DataFrame(
        {
            "contributionContent": ["hello", "world"],
            "category": ["Radverkehr", "Grünflächen"],
            "key_messages": [[_make_km("a")], [_make_km("b")]],
        }
    ).to_parquet(source)

    monkeypatch.setattr(main, "KeyMessageJudge", fresh_judge)
    monkeypatch.setattr("builtins.input", lambda *a: "1")

    asyncio.run(main.run_key_message_judge())

    assert fresh_judge.init_args is not None
    assert fresh_judge.init_args["base_url"] == main.JUDGE_LLM_BASE_URL
    assert fresh_judge.init_args["model_name"] == main.JUDGE_LLM_MODEL_NAME
    assert fresh_judge.init_args["api_key"] == main.JUDGE_LLM_API_KEY

    output = data_dir / "judged_input.parquet"
    assert output.exists()
    written = pd.read_parquet(output)
    assert "judge_result" in written.columns
    assert list(written["judge_result"]) == [
        fresh_judge.judge_return,
        fresh_judge.judge_return,
    ]
    # Two rows judged; key-message lists forwarded to the judge.
    assert len(fresh_judge.judge_calls) == 2


def test_run_key_message_judge_skips_none_row(monkeypatch, tmp_paths, fresh_judge):
    """A row with key_messages=None is skipped: not judged, output cell is None."""
    import asyncio

    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "input.parquet"
    pd.DataFrame(
        {
            "contributionContent": ["hello", "world"],
            "key_messages": pd.Series([None, [_make_km("b")]], dtype="object"),
        }
    ).to_parquet(source)

    monkeypatch.setattr(main, "KeyMessageJudge", fresh_judge)
    monkeypatch.setattr("builtins.input", lambda *a: "1")

    asyncio.run(main.run_key_message_judge())

    output = data_dir / "judged_input.parquet"
    written = pd.read_parquet(output)
    # First row skipped -> None; second row judged.
    assert written["judge_result"].iloc[0] is None
    assert written["judge_result"].iloc[1] == fresh_judge.judge_return
    # The skipped row's key messages must not reach the judge.
    assert len(fresh_judge.judge_calls) == 1


def test_run_key_message_judge_handles_none_result(
    monkeypatch, tmp_paths, fresh_judge
):
    """A failed judge (None) writes a None cell and the summary avoids div-by-zero."""
    import asyncio

    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "input.parquet"
    pd.DataFrame(
        {
            "contributionContent": ["hello"],
            "key_messages": [[_make_km("a")]],
        }
    ).to_parquet(source)

    fresh_judge.judge_return = None
    monkeypatch.setattr(main, "KeyMessageJudge", fresh_judge)
    monkeypatch.setattr("builtins.input", lambda *a: "1")

    # Must not raise despite zero valid overall scores.
    asyncio.run(main.run_key_message_judge())

    output = data_dir / "judged_input.parquet"
    written = pd.read_parquet(output)
    assert written["judge_result"].isna().all()


def test_run_key_message_judge_skips_on_overwrite_decline(
    monkeypatch, tmp_paths, fresh_judge
):
    """run_key_message_judge leaves output untouched and never judges on decline."""
    import asyncio

    data_dir, _ = tmp_paths
    data_dir.mkdir(parents=True, exist_ok=True)
    source = data_dir / "input.parquet"
    pd.DataFrame(
        {
            "contributionContent": ["hello"],
            "key_messages": [[_make_km("a")]],
        }
    ).to_parquet(source)

    output = data_dir / "judged_input.parquet"
    original = pd.DataFrame({"keep": [1]})
    original.to_parquet(output)

    monkeypatch.setattr(main, "KeyMessageJudge", fresh_judge)
    # File selection "1", then "n" to decline the overwrite prompt.
    answers = iter(["1", "n"])
    monkeypatch.setattr("builtins.input", lambda *a: next(answers))

    asyncio.run(main.run_key_message_judge())

    assert pd.read_parquet(output).equals(original)
    assert fresh_judge.judge_calls == []
