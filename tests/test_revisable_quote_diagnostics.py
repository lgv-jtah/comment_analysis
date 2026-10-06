import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from context_builder import ParentContext, build_comment_context
from revisable_attribution import RevisableAttributionEngine, _stage1_quote_diagnostics


def context(comment="unrelated comment"):
    return build_comment_context(
        run_id="SECRET", contribution_id="SECRET", project_id="SECRET", comment_id="SECRET",
        comment_text=comment, contribution_text="contribution words",
        parent_context=ParentContext(available=True, parent_comment_text="parent words"),
        key_messages=[{"key_message": "target words", "key_message_type": "SECRET",
                       "related_sentences": ["sentence words", "next sentence"]},
                      {"key_message": "second target", "related_sentences": ["third sentence"]}],
    )


@pytest.mark.parametrize("quote,field", [
    ("contribution\n words", "contribution"), ("target words", "key_message"),
    ("sentence\twords", "related_sentence"), ("parent words", "parent"),
])
def test_individual_overlap_fields(quote, field):
    assert _stage1_quote_diagnostics(quote, context()) == [f"stage1_quote_overlap_{field}"]


@pytest.mark.parametrize("quote", ["SECRET", "SECRET:km1", "missing_parent_context",
                                  "words second", "words next", "words third", "words target"])
def test_no_metadata_ids_or_cross_field_joins(quote):
    ctx = context()
    ctx.context_flags = ["SECRET"]
    assert _stage1_quote_diagnostics(quote, ctx) == ["stage1_quote_unexplained_text_mismatch"]


def test_absent_fields():
    ctx = context()
    ctx.contribution_text = None
    ctx.parent_context = ParentContext()
    ctx.key_messages = []
    assert _stage1_quote_diagnostics("missing", ctx) == ["stage1_quote_unexplained_text_mismatch"]


@pytest.mark.parametrize("quote,comment,code", [
    ("HELLO", "hello there", "case_only"),
    ("Stra\u00dfe", "STRA\u1e9eE", "case_only"),
    ("don't", "don\u2019t", "punctuation_only"),
    ("car-park", "car\u2014park", "punctuation_only"),
    ("car park", "car\u2013park", "punctuation_only"),
    ("CAR-PARK", "car\u2014park", "case_and_punctuation"),
    ("CAF\u00c9", "cafe\u0301", "case_only"),
    ("cafe\u0301", "caf\u00e9", "unexplained_text_mismatch"),
    ("STRASSE", "Stra\u00dfe", "unexplained_text_mismatch"),
    ("CAR", "scar", "unexplained_text_mismatch"),
    ("CAR", "cars", "unexplained_text_mismatch"),
    ("CAR", "_car", "case_and_punctuation"),
    ("car park", "carpark", "unexplained_text_mismatch"),
    ("carpark", "car-park", "unexplained_text_mismatch"),
    ("a+b", "a b", "unexplained_text_mismatch"),
    ("a\u20acb", "a b", "unexplained_text_mismatch"),
    ("A+B", "a+b", "case_only"),
    ("", "anything", "unexplained_text_mismatch"),
    (" \n ", "anything", "unexplained_text_mismatch"),
    ("!!!", "anything", "unexplained_text_mismatch"),
    ("\u2019\u2014", "anything", "unexplained_text_mismatch"),
    ("+", "+", "unexplained_text_mismatch"),
])
def test_local_format_diagnostics(quote, comment, code):
    ctx = context(comment)
    before = ctx.model_dump()
    assert _stage1_quote_diagnostics(quote, ctx) == [f"stage1_quote_{code}"]
    assert ctx.model_dump() == before


@pytest.mark.parametrize("mark", ["\u0338", "\u0903", "\u20dd", "\u200d", "\u200c", "\u2060", "\u202e", "\x00"])
@pytest.mark.parametrize("side", ["before", "after"])
@pytest.mark.parametrize("quote,text", [("CAR", "car"), ("car-park", "car park"), ("CAR-PARK", "car park")])
def test_unicode_marks_and_controls_block_format_boundaries(mark, side, quote, text):
    comment = "s" + mark + text if side == "before" else text + mark + "s"
    ctx = context(comment)
    before = ctx.model_dump()
    assert _stage1_quote_diagnostics(quote, ctx) == ["stage1_quote_unexplained_text_mismatch"]
    assert ctx.model_dump() == before


@pytest.mark.parametrize("mark", ["\u0338", "\u200d", "\u200c", "\u2060"])
def test_marks_and_controls_preserved_inside_quotes_and_later_valid_matches(mark):
    assert _stage1_quote_diagnostics("CAR", context("ca" + mark + "r")) == [
        "stage1_quote_unexplained_text_mismatch"]
    assert _stage1_quote_diagnostics("CA" + mark + "R", context("ca" + mark + "r")) == [
        "stage1_quote_case_only"]
    assert _stage1_quote_diagnostics("CAR", context("car" + mark + " car")) == ["stage1_quote_case_only"]
    assert _stage1_quote_diagnostics("CAR", context("car" + mark + " CAR")) == [
        "stage1_quote_unexplained_text_mismatch"]


def test_multiple_overlap_and_format_codes():
    ctx = context("TARGET WORDS")
    ctx.contribution_text = ctx.parent_context.parent_comment_text = "target words"
    ctx.key_messages[0].related_sentences = ["target words"]
    assert _stage1_quote_diagnostics("target words", ctx) == sorted([
        "stage1_quote_case_only", "stage1_quote_overlap_contribution", "stage1_quote_overlap_parent",
        "stage1_quote_overlap_key_message", "stage1_quote_overlap_related_sentence",
    ])
    assert _stage1_quote_diagnostics("CAR-PARK", context("car-park and CAR PARK")) == [
        "stage1_quote_case_only", "stage1_quote_punctuation_only"]


@pytest.mark.parametrize("variant", ["baseline", "stage1_comment_only", "stage2_required_targets"])
def test_rejected_quotes_only_add_sorted_unique_codes_and_never_reach_stage2(variant):
    ctx = context("TARGET WORDS")
    handler = AsyncMock()
    handler.call_model.return_value = {"scope_type": "target_specific", "comment_evidence": [
        "target words", "SECRET", "target words", " "]}
    engine = RevisableAttributionEngine(handler, "unused", "mock", revisable_variant=variant)
    outputs = asyncio.run(engine.resolve_comment(None, None, ctx))
    assert handler.call_model.await_count == 1
    for output in outputs:
        assert output.status == "failed"
        assert output.proposed_stance is output.final_stance is None
        assert output.comment_evidence == output.source_evidence == []
        assert output.review_reasons == sorted([
            "stage1_empty_scope_evidence", "stage1_nonmatching_scope_evidence", "stage1_quote_case_only",
            "stage1_quote_overlap_key_message", "stage1_quote_unexplained_text_mismatch",
        ])
        assert "SECRET" not in json.dumps(output.review_reasons)


@pytest.mark.parametrize("quotes", [["TARGET\nWORDS"], ["!!!"], [" "], []])
def test_diagnostics_not_called_for_valid_blank_or_absent_quotes(monkeypatch, quotes):
    import revisable_attribution
    monkeypatch.setattr(revisable_attribution, "_stage1_quote_diagnostics",
                        lambda *args: pytest.fail("diagnostics outside rejected nonblank branch"))
    ctx = context("TARGET WORDS !!!")
    handler = AsyncMock()
    scope = {"scope_type": "target_specific", "comment_evidence": quotes}
    handler.call_model.side_effect = [scope, {"revised_scope": {"scope_type": "target_specific"},
        "targets": [{"key_message_id": km.key_message_id, "attribution": "unaddressed"}
                    for km in ctx.key_messages]}]
    outputs = asyncio.run(RevisableAttributionEngine(handler, "unused", "mock").resolve_comment(None, None, ctx))
    assert handler.call_model.await_count == (1 if quotes == [" "] else 2)
    assert all(o.status == ("failed" if quotes == [" "] else "accepted") for o in outputs)


def test_stage2_nonmatching_evidence_does_not_use_stage1_diagnostics(monkeypatch):
    import revisable_attribution
    monkeypatch.setattr(revisable_attribution, "_stage1_quote_diagnostics",
                        lambda *args: pytest.fail("stage2 diagnostics changed"))
    ctx = context()
    handler = AsyncMock()
    handler.call_model.side_effect = [{"scope_type": "target_specific"}, {
        "revised_scope": {"scope_type": "target_specific", "comment_evidence": ["SECRET"]},
        "targets": [{"key_message_id": km.key_message_id, "attribution": "unaddressed"}
                    for km in ctx.key_messages]}]
    outputs = asyncio.run(RevisableAttributionEngine(handler, "unused", "mock").resolve_comment(None, None, ctx))
    assert all(o.status == "needs_review" and o.review_reasons == [
        "invalid_scope_evidence", "nonmatching_scope_evidence"] for o in outputs)
