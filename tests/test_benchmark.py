import asyncio

import pandas as pd
import pytest

import benchmark
from dataset_builder import compute_pair_id


# ---------------------------------------------------------------------------
# load_frozen_comments
# ---------------------------------------------------------------------------


def _source_row(**overrides):
    base = dict(
        contribution_id=1,
        contributionContent="Der Beitragstext.",
        key_message="km1",
        key_message_type="Wunsch",
        related_sentences=["s1"],
        comment_text="Ein Kommentar.",
    )
    base.update(overrides)
    return base


def test_load_frozen_comments_groups_by_contribution_and_comment():
    source = pd.DataFrame(
        [
            _source_row(key_message="km1", comment_text="c1"),
            _source_row(key_message="km2", comment_text="c1"),
            _source_row(key_message="km1", comment_text="c2"),
        ]
    )

    comments = benchmark.load_frozen_comments(source)

    assert len(comments) == 2
    by_comment = {c["comment_text"]: c for c in comments}
    assert len(by_comment["c1"]["key_messages"]) == 2
    assert len(by_comment["c2"]["key_messages"]) == 1


def test_load_frozen_comments_dedupes_key_messages_by_text():
    source = pd.DataFrame(
        [
            _source_row(key_message="km1", related_sentences=["s1"]),
            _source_row(key_message="km1", related_sentences=["s1"]),  # duplicate row
        ]
    )

    comments = benchmark.load_frozen_comments(source)

    assert len(comments) == 1
    assert len(comments[0]["key_messages"]) == 1


def test_load_frozen_comments_preserves_related_sentences():
    source = pd.DataFrame([_source_row(related_sentences=["a sentence", "another"])])

    comments = benchmark.load_frozen_comments(source)

    assert comments[0]["key_messages"][0]["related_sentences"] == ["a sentence", "another"]


def test_load_frozen_comments_handles_missing_contribution_content():
    source = pd.DataFrame([_source_row(contributionContent=None)])

    comments = benchmark.load_frozen_comments(source)

    assert comments[0]["contribution_text"] is None


def test_load_frozen_comments_contribution_id_is_stringified():
    source = pd.DataFrame([_source_row(contribution_id=42)])

    comments = benchmark.load_frozen_comments(source)

    assert comments[0]["contribution_id"] == "42"


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------


def _scope_response(**overrides):
    base = {
        "scope_type": "whole_contribution_approval",
        "affected_key_message_ids": [],
        "general_support_extends_to_unaddressed": None,
        "comment_evidence": [],
        "uncertainty_flags": [],
        "explanation": "x",
    }
    base.update(overrides)
    return base


async def _fake_call_model(self, *, response_schema, **kwargs):
    # Patched onto the LLMHandler class (run_benchmark constructs its own
    # instance internally, so there's no object to monkeypatch directly) --
    # must accept `self` as the implicit first positional argument.
    return _scope_response()


def test_run_benchmark_produces_expected_columns(monkeypatch):
    monkeypatch.setattr("policy.LLMHandler.call_model", _fake_call_model)

    source = pd.DataFrame([_source_row()])

    result = asyncio.run(
        benchmark.run_benchmark(
            source=source,
            base_url="http://testserver",
            model_name="test-model",
            verify_ionos=False,
        )
    )

    assert list(result.columns) == list(benchmark.OUTPUT_COLUMNS_V2)
    assert len(result) == 1
    assert result.iloc[0]["final_stance"] == "in_favor"
    assert result.iloc[0]["basis"] == "general_approval"


def test_run_benchmark_pair_ids_match_dataset_builder_compute_pair_id(monkeypatch):
    """The whole point of benchmark.py over WholeProcess.run_v2(): pair_ids
    must match dataset_builder's hash exactly, since key messages are frozen."""
    monkeypatch.setattr("policy.LLMHandler.call_model", _fake_call_model)

    source = pd.DataFrame(
        [_source_row(contribution_id=101, key_message="Mehr Fahrradständer", comment_text="Super Idee!")]
    )

    result = asyncio.run(
        benchmark.run_benchmark(
            source=source, base_url="http://testserver", model_name="test-model", verify_ionos=False
        )
    )

    expected_pair_id = compute_pair_id("101", "Mehr Fahrradständer", "Super Idee!")
    assert result.iloc[0]["pair_id"] == expected_pair_id


def test_run_benchmark_respects_limit(monkeypatch):
    monkeypatch.setattr("policy.LLMHandler.call_model", _fake_call_model)

    source = pd.DataFrame(
        [
            _source_row(contribution_id=1, comment_text="c1"),
            _source_row(contribution_id=2, comment_text="c2"),
            _source_row(contribution_id=3, comment_text="c3"),
        ]
    )

    result = asyncio.run(
        benchmark.run_benchmark(
            source=source, base_url="http://testserver", model_name="test-model", verify_ionos=False, limit=2
        )
    )

    assert result["contribution_id"].nunique() == 2


def test_run_benchmark_verifies_ionos_by_default(monkeypatch):
    import ionos_config

    def fake_get(*a, **k):
        return type(
            "R",
            (),
            {
                "raise_for_status": lambda self: None,
                "json": lambda self: {
                    "data": [{"model_name": "test-model", "litellm_params": {"api_base": None}}]
                },
            },
        )()

    monkeypatch.setattr(ionos_config.httpx, "get", fake_get)

    source = pd.DataFrame([_source_row()])

    with pytest.raises(ionos_config.IonosVerificationError):
        asyncio.run(
            benchmark.run_benchmark(source=source, base_url="http://testserver", model_name="test-model")
        )


def test_run_benchmark_run_id_is_recorded_on_every_row(monkeypatch):
    monkeypatch.setattr("policy.LLMHandler.call_model", _fake_call_model)

    source = pd.DataFrame([_source_row()])

    result = asyncio.run(
        benchmark.run_benchmark(
            source=source,
            base_url="http://testserver",
            model_name="test-model",
            verify_ionos=False,
            run_id="fixed-run-id",
        )
    )

    assert (result["run_id"] == "fixed-run-id").all()
