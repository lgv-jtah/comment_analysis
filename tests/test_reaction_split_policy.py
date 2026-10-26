"""Tests for the opt-in general-reaction split stage (docs/reaction_split_plan.md, phase 2).

LLM calls are stubbed by response schema; no network. The split prompt does
not exist yet (phase 3), so the engine is built from a temporary prompt
config holding the real scope/stance prompts plus a placeholder split prompt.
"""
import asyncio
import json
import shutil

import pandas as pd
import pytest

import whole_process
from context_builder import build_comment_context
from llm_handler import LLMHandler
from policy import PairOutput, StancePolicyEngineV2, validate_reaction_split
from policy_schemas import (
    PerTargetStanceResult,
    ReactionSplitResult,
    ScopeAttributionResult,
)

KM1, KM2 = "1:km1", "1:km2"
COMMENT = "Super Idee! Am Bahnhof fehlen wirklich Fahrradständer."
REST = "Am Bahnhof fehlen wirklich Fahrradständer."


@pytest.fixture
def prompt_config(tmp_path):
    with open("prompts/config.json", encoding="utf-8") as f:
        config = json.load(f)
    entries = {key: config[key] for key in ("scope_attribution", "per_target_stance")}
    for entry in entries.values():
        for name in (entry["system_prompt"], entry["user_prompt"]):
            shutil.copy(f"prompts/{name}", tmp_path / name)
    (tmp_path / "split_system.txt").write_text("System", encoding="utf-8")
    (tmp_path / "split_user.txt").write_text("Kommentar: $comment_text", encoding="utf-8")
    entries["reaction_split"] = {"system_prompt": "split_system.txt", "user_prompt": "split_user.txt"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return str(path)


def _split(**overrides):
    base = {"has_general_reaction": True, "polarity": "positive", "reaction_spans": ["Super Idee!"],
            "hint_explanation": "Pauschales Lob.", "rest_spans": [REST], "split_possible": True,
            "uncertainty_flags": []}
    base.update(overrides)
    return base


def _scope(**overrides):
    base = {"scope_type": "target_specific", "affected_key_message_ids": [KM1],
            "general_support_extends_to_unaddressed": None, "comment_evidence": [REST],
            "uncertainty_flags": [], "explanation": "Fahrradständer."}
    base.update(overrides)
    return base


def _stance(**overrides):
    target = {"key_message_id": KM1, "stance": "in_favor", "neutral_basis": None,
              "comment_evidence": ["fehlen wirklich Fahrradständer"], "source_evidence": [],
              "explanation": "Bestätigt den Bedarf."}
    target.update(overrides)
    return {"targets": [target]}


class Stub:
    """call_model stub answering per response schema and recording the prompts."""

    def __init__(self, split=None, scope=None, stance=None):
        self.replies = {ReactionSplitResult: split, ScopeAttributionResult: scope, PerTargetStanceResult: stance}
        self.calls = []

    async def __call__(self, *, response_schema, messages, **kwargs):
        self.calls.append((response_schema, messages[0]["content"]))
        reply = self.replies[response_schema]
        return reply.pop(0) if isinstance(reply, list) else reply


def _engine(monkeypatch, prompt_config, stub, **kwargs):
    engine = StancePolicyEngineV2(llm_handler=LLMHandler(), base_url="http://localhost:4000", model_name="m",
                                  prompt_config_path=prompt_config, reaction_split_enabled=True, **kwargs)
    monkeypatch.setattr(engine._llm, "call_model", stub)
    return engine


def _context(comment_text=COMMENT):
    return build_comment_context(
        run_id="run-1", contribution_id="1", comment_text=comment_text,
        key_messages=[{"key_message": "Fahrradständer am Bahnhof", "related_sentences": ["s1"]},
                      {"key_message": "Bäume pflanzen", "related_sentences": ["s2"]}],
        contribution_text="Beitrag über Fahrradständer und Bäume.")


def _run(engine, context=None):
    return asyncio.run(engine.resolve_comment(client=None, semaphore=None, context=context or _context()))


# --------------------------------------------------------------------------
# validate_reaction_split
# --------------------------------------------------------------------------


@pytest.mark.parametrize("overrides,comment,errors", [
    ({}, COMMENT, []),
    ({"rest_spans": []}, "Super Idee!", []),
    ({"reaction_spans": ["Super   Idee!"], "rest_spans": ["Am Bahnhof fehlen\nwirklich Fahrradständer."]}, COMMENT, []),
    ({"polarity": None}, COMMENT, ["general_reaction_missing_polarity"]),
    ({"reaction_spans": []}, COMMENT, ["general_reaction_missing_evidence"]),
    ({"has_general_reaction": False, "polarity": None, "reaction_spans": [], "rest_spans": [COMMENT]}, COMMENT, []),
    ({"has_general_reaction": False}, COMMENT,
     ["polarity_without_general_reaction", "reaction_spans_without_general_reaction"]),
    ({"has_general_reaction": False, "polarity": None, "reaction_spans": [], "rest_spans": []}, "!!!",
     ["empty_split"]),
    ({"reaction_spans": ["Tolle Idee!"]}, COMMENT, ["reaction_span_not_substring_of_comment"]),
    ({"rest_spans": ["Am Bahnhof fehlen Fahrradständer."]}, COMMENT, ["rest_span_not_substring_of_comment"]),
    ({"rest_spans": ["Idee! Am Bahnhof fehlen wirklich Fahrradständer."]}, COMMENT, ["split_spans_overlap"]),
    ({"rest_spans": ["Am Bahnhof fehlen"]}, COMMENT, ["split_does_not_cover_comment"]),
    # A model that cannot split cleanly is not held to full coverage.
    ({"rest_spans": ["Am Bahnhof fehlen"], "split_possible": False}, COMMENT, []),
    # Only punctuation and symbols may stay uncovered.
    ({"reaction_spans": ["Super Idee"], "rest_spans": ["Am Bahnhof fehlen wirklich Fahrradständer"]},
     COMMENT + " :-)", []),
    # A short span that also occurs inside a longer one takes another occurrence.
    ({"reaction_spans": ["gut"], "rest_spans": ["gut gemacht"]}, "gut gemacht, gut", []),
])
def test_validate_reaction_split(overrides, comment, errors):
    assert validate_reaction_split(ReactionSplitResult.model_validate(_split(**overrides)), comment) == errors


# --------------------------------------------------------------------------
# Engine routing
# --------------------------------------------------------------------------


def test_split_requires_a_prompt_config_entry():
    with pytest.raises(ValueError, match="reaction_split"):
        StancePolicyEngineV2(llm_handler=LLMHandler(), base_url="http://x", model_name="m",
                             reaction_split_enabled=True)


@pytest.mark.parametrize("polarity,basis", [("positive", "general_approval"), ("negative", "general_rejection")])
def test_pure_general_reaction_needs_no_further_calls(monkeypatch, prompt_config, polarity, basis):
    stub = Stub(split=_split(polarity=polarity, reaction_spans=["Super Idee!"], rest_spans=[]))
    results = _run(_engine(monkeypatch, prompt_config, stub), _context("Super Idee!"))

    assert [schema for schema, _ in stub.calls] == [ReactionSplitResult]
    assert all(r.status == "accepted" and r.final_stance == "neutral" and r.basis == basis for r in results)
    assert all(r.comment_evidence == ["Super Idee!"] and r.explanation == "Pauschales Lob." for r in results)
    assert all((r.general_reaction, r.general_reaction_polarity, r.rest_text) == (True, polarity, "")
               for r in results)


def test_stance_stages_see_only_the_rest_text(monkeypatch, prompt_config):
    stub = Stub(split=_split(), scope=_scope(), stance=_stance())
    results = _run(_engine(monkeypatch, prompt_config, stub))

    scope_prompt = next(content for schema, content in stub.calls if schema is ScopeAttributionResult)
    analysed = scope_prompt.split("## Zu analysierender Kommentar")[1]
    assert REST in analysed and "Super Idee!" not in analysed
    by_id = {r.key_message_id: r for r in results}
    assert (by_id[KM1].final_stance, by_id[KM1].basis) == ("in_favor", "explicit_target")
    assert (by_id[KM2].final_stance, by_id[KM2].basis) == ("neutral", "unaddressed")
    # The general reaction is a comment-level fact on every pair; pair IDs use the full comment.
    assert all((r.general_reaction, r.general_reaction_polarity, r.reaction_evidence, r.rest_text)
               == (True, "positive", ["Super Idee!"], REST) for r in results)


def test_templates_can_show_the_full_comment_as_context(monkeypatch, prompt_config, tmp_path):
    with open(prompt_config, encoding="utf-8") as f:
        scope_user = json.load(f)["scope_attribution"]["user_prompt"]
    (tmp_path / scope_user).write_text("Rest: $comment_text | Voll: $full_comment_text", encoding="utf-8")
    stub = Stub(split=_split(), scope=_scope(), stance=_stance())
    _run(_engine(monkeypatch, prompt_config, stub))
    scope_prompt = next(content for schema, content in stub.calls if schema is ScopeAttributionResult)
    assert scope_prompt == f"Rest: {REST} | Voll: {COMMENT}"


def test_pair_ids_are_computed_from_the_full_comment(monkeypatch, prompt_config):
    from dataset_builder import compute_pair_id

    stub = Stub(split=_split(), scope=_scope(), stance=_stance())
    results = _run(_engine(monkeypatch, prompt_config, stub))
    assert results[0].pair_id == compute_pair_id("1", "Fahrradständer am Bahnhof", COMMENT)


def test_stance_evidence_from_the_reaction_is_rejected(monkeypatch, prompt_config):
    stub = Stub(split=_split(), scope=_scope(), stance=_stance(comment_evidence=["Super Idee!"]))
    by_id = {r.key_message_id: r for r in _run(_engine(monkeypatch, prompt_config, stub))}
    assert by_id[KM1].status == "needs_review"
    assert by_id[KM1].review_reasons == ["invalid_stance_output", "evidence_not_substring_of_comment"]
    assert by_id[KM1].general_reaction_polarity == "positive"


def test_general_reaction_found_in_rest_text_goes_to_review(monkeypatch, prompt_config):
    stub = Stub(split=_split(), scope=_scope(scope_type="whole_contribution_approval", affected_key_message_ids=[]))
    results = _run(_engine(monkeypatch, prompt_config, stub))
    assert all(r.status == "needs_review" and r.review_reasons == ["general_reaction_in_rest_text"] for r in results)


def test_general_support_does_not_extend_in_split_mode(monkeypatch, prompt_config):
    stub = Stub(split=_split(), stance=_stance(),
                scope=_scope(scope_type="mixed_qualified", general_support_extends_to_unaddressed=True))
    by_id = {r.key_message_id: r for r in _run(_engine(monkeypatch, prompt_config, stub))}
    assert (by_id[KM2].final_stance, by_id[KM2].basis) == ("neutral", "unaddressed")


def test_failed_split_call_fails_every_pair(monkeypatch, prompt_config):
    results = _run(_engine(monkeypatch, prompt_config, Stub(split=None)))
    assert all(r.status == "failed" and r.general_reaction is None for r in results)


def test_invalid_split_goes_to_review_without_split_fields(monkeypatch, prompt_config):
    stub = Stub(split=_split(rest_spans=["Am Bahnhof fehlen"]))
    results = _run(_engine(monkeypatch, prompt_config, stub))
    assert all(r.status == "needs_review" and r.rest_text is None for r in results)
    assert results[0].review_reasons == ["invalid_split_output", "split_does_not_cover_comment"]
    assert len(stub.calls) == 1


def test_unsplittable_comment_goes_to_review_with_split_fields(monkeypatch, prompt_config):
    stub = Stub(split=_split(split_possible=False, uncertainty_flags=["reaction_qualified_by_rest"]))
    results = _run(_engine(monkeypatch, prompt_config, stub))
    assert all(r.status == "needs_review" for r in results)
    assert results[0].review_reasons == ["split_not_possible", "reaction_qualified_by_rest"]
    assert results[0].general_reaction is True and len(stub.calls) == 1


def test_verifier_routes_polarity_disagreement_to_review(monkeypatch, prompt_config):
    replies = [_split(rest_spans=[]), _split(rest_spans=[], polarity="negative")]
    stub = Stub(split=replies)
    results = _run(_engine(monkeypatch, prompt_config, stub, verifier_enabled=True), _context("Super Idee!"))
    assert all(r.status == "needs_review" and r.review_reasons == ["verifier_disagreement"] for r in results)


def test_verifier_accepts_agreeing_split(monkeypatch, prompt_config):
    stub = Stub(split=[_split(rest_spans=[]), _split(rest_spans=[])])
    results = _run(_engine(monkeypatch, prompt_config, stub, verifier_enabled=True), _context("Super Idee!"))
    assert all(r.status == "accepted" for r in results)


def test_disabled_split_leaves_split_fields_empty(monkeypatch):
    engine = StancePolicyEngineV2(llm_handler=LLMHandler(), base_url="http://x", model_name="m")
    stub = Stub(scope=_scope(scope_type="whole_contribution_approval", affected_key_message_ids=[],
                             comment_evidence=["Super Idee!"]))
    monkeypatch.setattr(engine._llm, "call_model", stub)
    results = _run(engine, _context("Super Idee!"))
    assert [schema for schema, _ in stub.calls] == [ScopeAttributionResult]
    assert all(r.basis == "general_approval" and r.general_reaction is None and r.rest_text is None
               for r in results)


# --------------------------------------------------------------------------
# Output rows and summary
# --------------------------------------------------------------------------


def test_pair_output_fields_cover_the_engine_columns():
    output = PairOutput(pair_id="p", key_message_id="k", proposed_stance="neutral", status="accepted",
                        final_stance="neutral", basis="general_rejection", general_reaction=True,
                        general_reaction_polarity="negative", reaction_evidence=["Bitte nicht."], rest_text="")
    fields = whole_process.pair_output_fields(output)
    assert set(fields) | {"contribution_id", "contributionContent", "key_message_id", "key_message", "comment_text",
                          "extraction_model_name", "stance_model_name"} == set(whole_process.OUTPUT_COLUMNS_V2)
    assert fields["general_reaction_polarity"] == "negative" and fields["reaction_evidence"] == ["Bitte nicht."]


def _row(km, comment, stance, basis, polarity=None):
    return {"contribution_id": 1, "contributionContent": "B", "key_message_id": f"1:{km}", "key_message": km,
            "comment_text": comment, "pair_id": "p", "status": "accepted", "proposed_stance": stance,
            "final_stance": stance, "basis": basis, "comment_evidence": [], "source_evidence": [],
            "explanation": "", "review_reasons": [], "general_reaction": polarity is not None,
            "general_reaction_polarity": polarity, "reaction_evidence": [], "rest_text": "",
            "policy_version": "v2", "run_id": "r", "extraction_model_name": "x", "stance_model_name": "y"}


def test_summary_counts_general_reactions_with_and_without_content():
    rows = [_row(km, "Bitte nicht.", "neutral", "general_rejection", "negative") for km in ("km1", "km2")]
    rows += [_row("km1", COMMENT, "in_favor", "explicit_target", "positive"),
             _row("km2", COMMENT, "neutral", "unaddressed", "positive")]
    rows += [_row(km, "Mehr Bäume!", s, b) for km, s, b in (("km1", "neutral", "unaddressed"),
                                                           ("km2", "in_favor", "explicit_target"))]

    summary = whole_process.summarize_v2_by_key_message(pd.DataFrame(rows, columns=whole_process.OUTPUT_COLUMNS_V2))

    assert list(summary.columns) == whole_process.SUMMARY_COLUMNS_V2
    by_km = summary.set_index("key_message_id")
    assert by_km.loc["1:km1", ["comments", "in_favor", "neutral", "general_approval", "general_rejection"]].tolist() \
        == [3, 1, 1, 0, 1]
    assert by_km.loc["1:km1", ["contribution_comments", "contribution_general_approval_comments",
                               "contribution_general_rejection_comments"]].tolist() == [3, 1, 1]


def test_summary_accepts_frames_without_split_columns():
    rows = [_row("km1", "Stimmt", "neutral", "general_approval")]
    frame = pd.DataFrame(rows).drop(columns=["general_reaction", "general_reaction_polarity", "reaction_evidence",
                                             "rest_text"])
    summary = whole_process.summarize_v2_by_key_message(frame)
    assert summary.loc[0, ["general_approval", "contribution_general_approval_comments",
                           "contribution_general_rejection_comments"]].tolist() == [1, 1, 0]
