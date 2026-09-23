"""Tests for policy.py.

Following this codebase's existing convention (see test_whole_process.py):
no pytest-asyncio is installed, so coroutines are driven via asyncio.run()
inside plain sync test functions rather than `async def test_...`.

All tests mock LLMHandler.call_model -- no real network calls. Real prompt
files under prompts/ are used (via the default prompt_config_path) so
construction also validates the prompt-loading path end to end.
"""
import asyncio

from context_builder import KeyMessageRef, build_comment_context
from llm_handler import LLMHandler
from policy import OneCallPolicyEngineV2, StancePolicyEngineV2, _format_key_messages_block


def _make_engine(monkeypatch, call_model=None, verifier_enabled=False) -> StancePolicyEngineV2:
    engine = StancePolicyEngineV2(
        llm_handler=LLMHandler(),
        base_url="http://localhost:4000",
        model_name="test-model",
        verifier_enabled=verifier_enabled,
    )
    if call_model is not None:
        monkeypatch.setattr(engine._llm, "call_model", call_model)
    return engine


def _context(**overrides):
    defaults = dict(
        run_id="run-1",
        contribution_id="1",
        comment_text="Super Idee!",
        key_messages=[
            {"key_message": "Mehr Fahrradständer", "related_sentences": ["s1"]},
            {"key_message": "Parkplätze entfernen", "related_sentences": ["s2"]},
        ],
        contribution_text="Ein Beitrag über Fahrradständer und Parkplätze.",
    )
    defaults.update(overrides)
    return build_comment_context(**defaults)


def _scope_response(**overrides):
    base = {
        "scope_type": "whole_contribution_approval",
        "affected_key_message_ids": [],
        "general_support_extends_to_unaddressed": None,
        "comment_evidence": [],
        "uncertainty_flags": [],
        "explanation": "Allgemeine Zustimmung.",
    }
    base.update(overrides)
    return base


def _stance_response(targets):
    return {"targets": targets}


def _run(engine, context):
    return asyncio.run(engine.resolve_comment(client=None, semaphore=None, context=context))


# ---------------------------------------------------------------------------
# Prompt formatting includes related_sentences, not just key-message text --
# the model can't produce a valid source_evidence quote for text it was
# never shown.
# ---------------------------------------------------------------------------


def test_format_key_messages_block_includes_related_sentences():
    kms = [
        KeyMessageRef(
            key_message_id="1:km1",
            text="Mehr Fahrradständer",
            related_sentences=["Es fehlen Abstellmöglichkeiten.", "Ein zweiter Originalsatz."],
        )
    ]

    block = _format_key_messages_block(kms)

    assert "Es fehlen Abstellmöglichkeiten." in block
    assert "Ein zweiter Originalsatz." in block


def test_format_key_messages_block_handles_no_related_sentences():
    kms = [KeyMessageRef(key_message_id="1:km1", text="Mehr Fahrradständer", related_sentences=[])]

    block = _format_key_messages_block(kms)

    assert "1:km1" in block
    assert "Mehr Fahrradständer" in block


# ---------------------------------------------------------------------------
# General approval reaches every key message
# ---------------------------------------------------------------------------


def test_general_approval_reaches_every_key_message(monkeypatch):
    async def call_model(*, response_schema, **kwargs):
        return _scope_response(scope_type="whole_contribution_approval", comment_evidence=["Super Idee!"])

    engine = _make_engine(monkeypatch, call_model)
    context = _context(comment_text="Super Idee!")

    results = _run(engine, context)

    assert len(results) == 2
    assert all(r.status == "accepted" for r in results)
    assert all(r.final_stance == "in_favor" for r in results)
    assert all(r.basis == "general_approval" for r in results)


# ---------------------------------------------------------------------------
# Explicit target-specific text does not trigger blanket support
# ---------------------------------------------------------------------------


def test_targeted_text_does_not_trigger_blanket_support(monkeypatch):
    context = _context(comment_text="Endlich mehr Fahrradständer!")
    km1_id, km2_id = [km.key_message_id for km in context.key_messages]

    async def call_model(*, response_schema, **kwargs):
        if response_schema.__name__ == "ScopeAttributionResult":
            return _scope_response(
                scope_type="target_specific",
                affected_key_message_ids=[km1_id],
                comment_evidence=["Endlich mehr Fahrradständer!"],
            )
        return _stance_response(
            [
                {
                    "key_message_id": km1_id,
                    "stance": "in_favor",
                    "neutral_basis": None,
                    "comment_evidence": ["Endlich mehr Fahrradständer!"],
                    "source_evidence": [],
                    "explanation": "Zustimmung.",
                }
            ]
        )

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, context)
    by_id = {r.key_message_id: r for r in results}

    assert by_id[km1_id].final_stance == "in_favor"
    assert by_id[km1_id].basis == "explicit_target"
    # The unaddressed key message stays neutral, not swept into support.
    assert by_id[km2_id].final_stance == "neutral"
    assert by_id[km2_id].basis == "unaddressed"


# ---------------------------------------------------------------------------
# mixed_qualified: explicit exception overrides inherited support
# ---------------------------------------------------------------------------


def test_mixed_qualified_explicit_exception_overrides_general_support(monkeypatch):
    context = _context(
        comment_text="Ich unterstütze den gesamten Vorschlag, nur die Parkplätze sollten bleiben."
    )
    km1_id, km2_id = [km.key_message_id for km in context.key_messages]  # bikes, parking

    async def call_model(*, response_schema, **kwargs):
        if response_schema.__name__ == "ScopeAttributionResult":
            return _scope_response(
                scope_type="mixed_qualified",
                affected_key_message_ids=[km2_id],
                general_support_extends_to_unaddressed=True,
                comment_evidence=["nur die Parkplätze sollten bleiben"],
            )
        return _stance_response(
            [
                {
                    "key_message_id": km2_id,
                    "stance": "against",
                    "neutral_basis": None,
                    "comment_evidence": ["die Parkplätze sollten bleiben"],
                    "source_evidence": [],
                    "explanation": "Widerspricht der Entfernung.",
                }
            ]
        )

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, context)
    by_id = {r.key_message_id: r for r in results}

    # Explicit exception: parking removal is opposed.
    assert by_id[km2_id].final_stance == "against"
    assert by_id[km2_id].basis == "explicit_target"
    # General support extends to the unaddressed key message.
    assert by_id[km1_id].final_stance == "in_favor"
    assert by_id[km1_id].basis == "general_approval"


def test_mixed_qualified_without_extension_leaves_unaddressed_neutral(monkeypatch):
    context = _context(
        comment_text="Gute Idee mit den Fahrradständern, aber die Parkplätze sollten bleiben."
    )
    km1_id, km2_id = [km.key_message_id for km in context.key_messages]

    async def call_model(*, response_schema, **kwargs):
        if response_schema.__name__ == "ScopeAttributionResult":
            return _scope_response(
                scope_type="mixed_qualified",
                affected_key_message_ids=[km1_id, km2_id],
                general_support_extends_to_unaddressed=False,
            )
        return _stance_response(
            [
                {
                    "key_message_id": km1_id,
                    "stance": "in_favor",
                    "neutral_basis": None,
                    "comment_evidence": [],
                    "source_evidence": [],
                    "explanation": "",
                },
                {
                    "key_message_id": km2_id,
                    "stance": "against",
                    "neutral_basis": None,
                    "comment_evidence": [],
                    "source_evidence": [],
                    "explanation": "",
                },
            ]
        )

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, context)
    by_id = {r.key_message_id: r for r in results}

    assert by_id[km1_id].final_stance == "in_favor"
    assert by_id[km2_id].final_stance == "against"


# ---------------------------------------------------------------------------
# Multiple targets addressable in one comment
# ---------------------------------------------------------------------------


def test_multiple_targets_can_each_get_a_directional_stance(monkeypatch):
    context = _context()
    km1_id, km2_id = [km.key_message_id for km in context.key_messages]

    async def call_model(*, response_schema, **kwargs):
        if response_schema.__name__ == "ScopeAttributionResult":
            return _scope_response(scope_type="target_specific", affected_key_message_ids=[km1_id, km2_id])
        return _stance_response(
            [
                {"key_message_id": km1_id, "stance": "in_favor", "neutral_basis": None, "comment_evidence": [], "source_evidence": [], "explanation": ""},
                {"key_message_id": km2_id, "stance": "against", "neutral_basis": None, "comment_evidence": [], "source_evidence": [], "explanation": ""},
            ]
        )

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, context)
    by_id = {r.key_message_id: r for r in results}

    assert by_id[km1_id].final_stance == "in_favor"
    assert by_id[km2_id].final_stance == "against"


# ---------------------------------------------------------------------------
# Negative-sounding confirmation is not mistaken for opposition -- the
# orchestrator trusts the validated stage-2 decision rather than inferring
# polarity itself from comment tone.
# ---------------------------------------------------------------------------


def test_negative_toned_confirmation_is_not_inverted_by_orchestration(monkeypatch):
    context = _context(comment_text="Es ist ein Drama, dass hier so viel Wohnraum verdichtet wird.")
    km1_id = context.key_messages[0].key_message_id

    async def call_model(*, response_schema, **kwargs):
        if response_schema.__name__ == "ScopeAttributionResult":
            return _scope_response(scope_type="target_specific", affected_key_message_ids=[km1_id])
        return _stance_response(
            [
                {
                    "key_message_id": km1_id,
                    "stance": "in_favor",  # negative tone, but confirms the KM's complaint
                    "neutral_basis": None,
                    "comment_evidence": ["Es ist ein Drama"],
                    "source_evidence": [],
                    "explanation": "Bestätigt das Problem.",
                }
            ]
        )

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, context)

    assert results[0].final_stance == "in_favor"


# ---------------------------------------------------------------------------
# Missing/unresolved reference context routes to review
# ---------------------------------------------------------------------------


def test_unresolved_scope_routes_every_key_message_to_review(monkeypatch):
    async def call_model(*, response_schema, **kwargs):
        return _scope_response(
            scope_type="unresolved", uncertainty_flags=["unresolved_reply_reference"]
        )

    engine = _make_engine(monkeypatch, call_model)
    context = _context(comment_text="Siehe oben.")

    results = _run(engine, context)

    assert all(r.status == "needs_review" for r in results)
    assert all(r.final_stance is None for r in results)
    assert all("unresolved_reply_reference" in r.review_reasons for r in results)


# ---------------------------------------------------------------------------
# Malformed output, unknown IDs, invalid evidence, and failed calls never
# become a silently accepted neutral result.
# ---------------------------------------------------------------------------


def test_failed_scope_call_marks_every_pair_failed_not_neutral(monkeypatch):
    async def call_model(*, response_schema, **kwargs):
        return None

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, _context())

    assert all(r.status == "failed" for r in results)
    assert all(r.final_stance is None for r in results)


def test_scope_with_unknown_key_message_id_routes_to_review(monkeypatch):
    async def call_model(*, response_schema, **kwargs):
        return _scope_response(
            scope_type="target_specific", affected_key_message_ids=["not-a-real-id"]
        )

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, _context())

    assert all(r.status == "needs_review" for r in results)
    assert any("unknown_key_message_id" in reason for r in results for reason in r.review_reasons)


def test_scope_evidence_not_substring_of_comment_routes_to_review(monkeypatch):
    async def call_model(*, response_schema, **kwargs):
        return _scope_response(
            scope_type="whole_contribution_approval",
            comment_evidence=["dieser Text steht nicht im Kommentar"],
        )

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, _context())

    assert all(r.status == "needs_review" for r in results)
    assert any("evidence_not_substring_of_comment" in r.review_reasons for r in results)


def test_scope_evidence_with_normalized_whitespace_is_accepted(monkeypatch):
    """A multi-line comment quoted with newlines folded to spaces is the
    same text, not a fabricated quote -- see docs/phase5_screening_results.md's
    evidence-grounding investigation."""
    context = _context(comment_text="Mit freundlichen Grüßen\nHans Biel\n\nRadfahrer")

    async def call_model(*, response_schema, **kwargs):
        return _scope_response(
            scope_type="whole_contribution_approval",
            comment_evidence=["Mit freundlichen Grüßen Hans Biel Radfahrer"],
        )

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, context)

    assert all(r.status == "accepted" for r in results)


def test_scope_evidence_with_real_word_change_still_rejected(monkeypatch):
    """Whitespace-only tolerance must not mask an actual word substitution
    (e.g. the model "correcting" the source's spelling/casing)."""
    context = _context(comment_text="Das stimmt, hier auf die STrasse")

    async def call_model(*, response_schema, **kwargs):
        return _scope_response(
            scope_type="whole_contribution_approval",
            comment_evidence=["Das stimmt, hier auf die Straße"],
        )

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, context)

    assert all(r.status == "needs_review" for r in results)
    assert any("evidence_not_substring_of_comment" in r.review_reasons for r in results)


def test_malformed_scope_output_fails_validation_and_routes_to_review(monkeypatch):
    async def call_model(*, response_schema, **kwargs):
        # Missing the required "scope_type" field entirely -> Pydantic
        # ValidationError inside _call_scope_stage, caught, returns None.
        return {"affected_key_message_ids": []}

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, _context())

    assert all(r.status == "failed" for r in results)


def test_stance_target_with_unknown_key_message_id_routes_to_review(monkeypatch):
    context = _context()
    km1_id = context.key_messages[0].key_message_id

    async def call_model(*, response_schema, **kwargs):
        if response_schema.__name__ == "ScopeAttributionResult":
            return _scope_response(scope_type="target_specific", affected_key_message_ids=[km1_id])
        return _stance_response(
            [
                {
                    "key_message_id": "not-in-affected-set",
                    "stance": "in_favor",
                    "neutral_basis": None,
                    "comment_evidence": [],
                    "source_evidence": [],
                    "explanation": "",
                }
            ]
        )

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, context)
    by_id = {r.key_message_id: r for r in results}

    assert by_id[km1_id].status == "needs_review"
    assert any("unknown_key_message_id" in reason for reason in by_id[km1_id].review_reasons)


def test_stance_comment_evidence_not_substring_routes_to_review(monkeypatch):
    context = _context()
    km1_id = context.key_messages[0].key_message_id

    async def call_model(*, response_schema, **kwargs):
        if response_schema.__name__ == "ScopeAttributionResult":
            return _scope_response(scope_type="target_specific", affected_key_message_ids=[km1_id])
        return _stance_response(
            [
                {
                    "key_message_id": km1_id,
                    "stance": "in_favor",
                    "neutral_basis": None,
                    "comment_evidence": ["nicht im Kommentar vorhanden"],
                    "source_evidence": [],
                    "explanation": "",
                }
            ]
        )

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, context)
    by_id = {r.key_message_id: r for r in results}

    assert by_id[km1_id].status == "needs_review"
    assert "evidence_not_substring_of_comment" in by_id[km1_id].review_reasons


def test_stance_comment_evidence_with_normalized_whitespace_is_accepted(monkeypatch):
    context = _context(comment_text="Ich finde\ndas eine gute Idee")
    km1_id = context.key_messages[0].key_message_id

    async def call_model(*, response_schema, **kwargs):
        if response_schema.__name__ == "ScopeAttributionResult":
            return _scope_response(scope_type="target_specific", affected_key_message_ids=[km1_id])
        return _stance_response(
            [
                {
                    "key_message_id": km1_id,
                    "stance": "in_favor",
                    "neutral_basis": None,
                    "comment_evidence": ["Ich finde das eine gute Idee"],
                    "source_evidence": [],
                    "explanation": "",
                }
            ]
        )

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, context)
    by_id = {r.key_message_id: r for r in results}

    assert by_id[km1_id].status == "accepted"


def test_stance_call_failure_marks_addressed_targets_failed(monkeypatch):
    context = _context()
    km1_id = context.key_messages[0].key_message_id

    async def call_model(*, response_schema, **kwargs):
        if response_schema.__name__ == "ScopeAttributionResult":
            return _scope_response(scope_type="target_specific", affected_key_message_ids=[km1_id])
        return None

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, context)
    by_id = {r.key_message_id: r for r in results}

    assert by_id[km1_id].status == "failed"


def test_neutral_stance_without_basis_routes_to_review(monkeypatch):
    context = _context()
    km1_id = context.key_messages[0].key_message_id

    async def call_model(*, response_schema, **kwargs):
        if response_schema.__name__ == "ScopeAttributionResult":
            return _scope_response(scope_type="target_specific", affected_key_message_ids=[km1_id])
        return _stance_response(
            [
                {
                    "key_message_id": km1_id,
                    "stance": "neutral",
                    "neutral_basis": None,  # invalid: neutral requires a basis
                    "comment_evidence": [],
                    "source_evidence": [],
                    "explanation": "",
                }
            ]
        )

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, context)
    by_id = {r.key_message_id: r for r in results}

    assert by_id[km1_id].status == "needs_review"
    assert "neutral_missing_basis" in by_id[km1_id].review_reasons


def test_source_evidence_not_in_related_sentences_routes_to_review(monkeypatch):
    context = _context()
    km1_id = context.key_messages[0].key_message_id

    async def call_model(*, response_schema, **kwargs):
        if response_schema.__name__ == "ScopeAttributionResult":
            return _scope_response(scope_type="target_specific", affected_key_message_ids=[km1_id])
        return _stance_response(
            [
                {
                    "key_message_id": km1_id,
                    "stance": "in_favor",
                    "neutral_basis": None,
                    "comment_evidence": [],
                    "source_evidence": ["invented sentence not in related_sentences"],
                    "explanation": "",
                }
            ]
        )

    engine = _make_engine(monkeypatch, call_model)
    results = _run(engine, context)
    by_id = {r.key_message_id: r for r in results}

    assert by_id[km1_id].status == "needs_review"
    assert "source_evidence_not_in_related_sentences" in by_id[km1_id].review_reasons


def test_no_key_messages_returns_empty_result(monkeypatch):
    engine = _make_engine(monkeypatch, call_model=None)
    context = build_comment_context(
        run_id="run-1", contribution_id="1", comment_text="text", key_messages=[]
    )

    results = _run(engine, context)

    assert results == []


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


def test_verifier_disagreement_routes_to_review(monkeypatch):
    context = _context(comment_text="Super Idee!")
    call_count = {"n": 0}

    async def call_model(*, response_schema, **kwargs):
        call_count["n"] += 1
        # First pass: whole_contribution_approval. Second (verifier) pass: disagrees.
        if call_count["n"] == 1:
            return _scope_response(scope_type="whole_contribution_approval")
        return _scope_response(scope_type="target_specific", affected_key_message_ids=[])

    engine = _make_engine(monkeypatch, call_model, verifier_enabled=True)
    results = _run(engine, context)

    # Primary said in_favor via general_approval; verifier's target_specific
    # pass (no affected ids) leaves both KMs neutral -- disagreement on both.
    assert all(r.status == "needs_review" for r in results)
    assert all("verifier_disagreement" in r.review_reasons for r in results)


def test_verifier_agreement_keeps_accepted_result(monkeypatch):
    context = _context(comment_text="Super Idee!")

    async def call_model(*, response_schema, **kwargs):
        return _scope_response(scope_type="whole_contribution_approval")

    engine = _make_engine(monkeypatch, call_model, verifier_enabled=True)
    results = _run(engine, context)

    assert all(r.status == "accepted" for r in results)
    assert all(r.final_stance == "in_favor" for r in results)


# ---------------------------------------------------------------------------
# OneCallPolicyEngineV2
# ---------------------------------------------------------------------------


def _make_one_call_engine(monkeypatch, call_model=None) -> OneCallPolicyEngineV2:
    engine = OneCallPolicyEngineV2(
        llm_handler=LLMHandler(),
        base_url="http://localhost:4000",
        model_name="test-model",
    )
    if call_model is not None:
        monkeypatch.setattr(engine._llm, "call_model", call_model)
    return engine


def _one_call_response(targets):
    return {"targets": targets}


def test_one_call_general_approval_reaches_every_key_message(monkeypatch):
    context = _context(comment_text="Super Idee!")
    km1_id, km2_id = [km.key_message_id for km in context.key_messages]

    async def call_model(*, response_schema, **kwargs):
        return _one_call_response(
            [
                {
                    "key_message_id": km1_id,
                    "stance": "in_favor",
                    "basis": "general_approval",
                    "comment_evidence": ["Super Idee!"],
                    "explanation": "",
                },
                {
                    "key_message_id": km2_id,
                    "stance": "in_favor",
                    "basis": "general_approval",
                    "comment_evidence": ["Super Idee!"],
                    "explanation": "",
                },
            ]
        )

    engine = _make_one_call_engine(monkeypatch, call_model)
    results = _run(engine, context)

    assert all(r.status == "accepted" for r in results)
    assert all(r.final_stance == "in_favor" for r in results)
    assert all(r.basis == "general_approval" for r in results)


def test_one_call_mixed_targets(monkeypatch):
    context = _context()
    km1_id, km2_id = [km.key_message_id for km in context.key_messages]

    async def call_model(*, response_schema, **kwargs):
        return _one_call_response(
            [
                {"key_message_id": km1_id, "stance": "in_favor", "basis": "explicit_target", "comment_evidence": [], "explanation": ""},
                {"key_message_id": km2_id, "stance": "neutral", "basis": "unaddressed", "comment_evidence": [], "explanation": ""},
            ]
        )

    engine = _make_one_call_engine(monkeypatch, call_model)
    results = _run(engine, context)
    by_id = {r.key_message_id: r for r in results}

    assert by_id[km1_id].final_stance == "in_favor"
    assert by_id[km2_id].final_stance == "neutral"
    assert by_id[km2_id].basis == "unaddressed"


def test_one_call_missing_target_routes_all_to_review(monkeypatch):
    context = _context()
    km1_id = context.key_messages[0].key_message_id

    async def call_model(*, response_schema, **kwargs):
        # Only returns a target for km1, silently drops km2.
        return _one_call_response(
            [{"key_message_id": km1_id, "stance": "in_favor", "basis": "explicit_target", "comment_evidence": [], "explanation": ""}]
        )

    engine = _make_one_call_engine(monkeypatch, call_model)
    results = _run(engine, context)

    assert all(r.status == "needs_review" for r in results)
    assert any("missing_targets" in reason for r in results for reason in r.review_reasons)


def test_one_call_unknown_key_message_id_routes_to_review(monkeypatch):
    context = _context()

    async def call_model(*, response_schema, **kwargs):
        return _one_call_response(
            [{"key_message_id": "not-a-real-id", "stance": "in_favor", "basis": "explicit_target", "comment_evidence": [], "explanation": ""}]
        )

    engine = _make_one_call_engine(monkeypatch, call_model)
    results = _run(engine, context)

    assert all(r.status == "needs_review" for r in results)
    assert any("unknown_key_message_id" in reason for r in results for reason in r.review_reasons)


def test_one_call_evidence_not_substring_routes_to_review(monkeypatch):
    context = _context()
    km1_id, km2_id = [km.key_message_id for km in context.key_messages]

    async def call_model(*, response_schema, **kwargs):
        return _one_call_response(
            [
                {"key_message_id": km1_id, "stance": "in_favor", "basis": "explicit_target", "comment_evidence": ["nicht im kommentar"], "explanation": ""},
                {"key_message_id": km2_id, "stance": "neutral", "basis": "unaddressed", "comment_evidence": [], "explanation": ""},
            ]
        )

    engine = _make_one_call_engine(monkeypatch, call_model)
    results = _run(engine, context)

    assert all(r.status == "needs_review" for r in results)
    assert any("evidence_not_substring_of_comment" in r.review_reasons for r in results)


def test_one_call_evidence_with_normalized_whitespace_is_accepted(monkeypatch):
    context = _context(comment_text="Ich finde\ndas eine gute Idee")
    km1_id, km2_id = [km.key_message_id for km in context.key_messages]

    async def call_model(*, response_schema, **kwargs):
        return _one_call_response(
            [
                {"key_message_id": km1_id, "stance": "in_favor", "basis": "explicit_target", "comment_evidence": ["Ich finde das eine gute Idee"], "explanation": ""},
                {"key_message_id": km2_id, "stance": "neutral", "basis": "unaddressed", "comment_evidence": [], "explanation": ""},
            ]
        )

    engine = _make_one_call_engine(monkeypatch, call_model)
    results = _run(engine, context)

    assert all(r.status == "accepted" for r in results)


def test_one_call_failed_call_marks_everything_failed(monkeypatch):
    async def call_model(*, response_schema, **kwargs):
        return None

    engine = _make_one_call_engine(monkeypatch, call_model)
    results = _run(engine, _context())

    assert all(r.status == "failed" for r in results)


def test_one_call_malformed_output_routes_to_review(monkeypatch):
    async def call_model(*, response_schema, **kwargs):
        return {"not_targets_at_all": True}

    engine = _make_one_call_engine(monkeypatch, call_model)
    results = _run(engine, _context())

    assert all(r.status == "needs_review" for r in results)
    assert all(any("missing_targets" in reason for reason in r.review_reasons) for r in results)


def test_one_call_no_key_messages_returns_empty(monkeypatch):
    engine = _make_one_call_engine(monkeypatch, call_model=None)
    context = build_comment_context(run_id="run-1", contribution_id="1", comment_text="text", key_messages=[])

    results = _run(engine, context)

    assert results == []
