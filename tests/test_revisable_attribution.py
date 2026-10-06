import asyncio
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from context_builder import ParentContext, build_comment_context
from revisable_attribution import RevisableAttributionEngine


def context():
    return build_comment_context(
        run_id="test", contribution_id="1", contribution_text="Original full contribution",
        comment_text="Alles gut, aber Parken nein. Baeume ausgenommen.",
        parent_context=ParentContext(available=True, parent_kind="comment", parent_comment_text="Elterntext"),
        key_messages=[{"key_message": name, "related_sentences": [f"Original {name} sentence"]}
                      for name in ["Rad", "Parken", "Baeume"]],
    )


def scope(kind="whole_contribution_approval", **kwargs):
    return dict(scope_type=kind, affected_key_message_ids=[], comment_evidence=["Alles gut"], **kwargs)


def revised():
    return dict(revised_scope=scope("mixed_qualified", general_support_extends_to_unaddressed=True), targets=[
        dict(key_message_id="1:km1", attribution="unaddressed"),
        dict(key_message_id="1:km2", attribution="explicit_target", stance="against",
             comment_evidence=["Parken nein"], source_evidence=["Parken"]),
        dict(key_message_id="1:km3", attribution="explicit_exclusion", stance="neutral",
             comment_evidence=["Baeume ausgenommen"]),
    ])


def run(first=None, second=None):
    handler = AsyncMock()
    handler.call_model.side_effect = [first, second]
    engine = RevisableAttributionEngine(handler, "http://unused", "mock")
    outputs = asyncio.run(engine.resolve_comment(None, None, context()))
    return outputs, handler


@pytest.mark.parametrize("kind", ["whole_contribution_approval", "unresolved", "target_specific", "mixed_qualified"])
def test_every_first_stage_scope_sees_all_original_targets_and_can_be_revised(kind):
    outputs, handler = run(scope(kind), revised())
    assert [o.final_stance for o in outputs] == ["neutral", "against", "neutral"]
    assert [o.basis for o in outputs] == ["general_approval", "explicit_target", "explicit_exclusion"]
    assert handler.call_model.await_count == 2
    for call in handler.call_model.call_args_list:
        prompt = call.kwargs["messages"][0]["content"]
        assert all(km.key_message_id in prompt for km in context().key_messages)
        assert "Original full contribution" in prompt
        assert "Original Parken sentence" in prompt
        assert "Elterntext" in prompt
    assert kind in handler.call_model.call_args_list[1].kwargs["messages"][0]["content"]


def test_targeted_support_can_replace_blanket_approval_and_cover_multiple_targets():
    response = revised()
    response["revised_scope"] = scope("target_specific")
    response["targets"][2] = dict(key_message_id="1:km3", attribution="explicit_target", stance="in_favor",
                                  comment_evidence=["Baeume"])
    outputs, _ = run(scope(), response)
    assert [o.final_stance for o in outputs] == ["neutral", "against", "in_favor"]


@pytest.mark.parametrize("attribution", ["unaddressed", "no_position"])
def test_general_approval_inherits_only_without_explicit_exception(attribution):
    response = revised()
    response["revised_scope"] = scope()
    response["targets"] = [dict(key_message_id=km.key_message_id, attribution=attribution)
                           for km in context().key_messages]
    outputs, _ = run(scope("unresolved"), response)
    assert all(o.final_stance == "neutral" and o.basis == "general_approval" for o in outputs)
    assert all(o.comment_evidence == ["Alles gut"] for o in outputs)
    response["revised_scope"] = scope("target_specific")
    outputs, _ = run(scope(), response)
    assert all(o.final_stance == "neutral" and o.basis == attribution for o in outputs)


def blanket_support(revised_scope, evidence=("Alles gut",), stance="in_favor", explanation="Zustimmung."):
    return dict(revised_scope=revised_scope, targets=[
        dict(key_message_id=km.key_message_id, attribution="explicit_target", stance=stance,
             comment_evidence=list(evidence), explanation=explanation)
        for km in context().key_messages
    ])


@pytest.mark.parametrize("revised_scope", [
    scope(), scope("mixed_qualified", general_support_extends_to_unaddressed=True),
])
@pytest.mark.parametrize("evidence", [("Alles gut",), ("gut",), ("Alles  gut",)])
def test_explicit_support_quoting_only_general_evidence_is_general_approval(revised_scope, evidence):
    outputs, _ = run(scope(), blanket_support(revised_scope, evidence))
    assert all(o.status == "accepted" and o.final_stance == "neutral" for o in outputs)
    assert all(o.basis == "general_approval" and o.comment_evidence == ["Alles gut"] for o in outputs)
    # The target's "explicit support" explanation would contradict neutral.
    assert all(o.explanation == "Allgemeine Zustimmung zum gesamten Beitrag, kein eigener Stance zu dieser Kernaussage." for o in outputs)


@pytest.mark.parametrize("revised_scope", [
    scope("target_specific"), scope("mixed_qualified", general_support_extends_to_unaddressed=False),
    scope("mixed_qualified"),
])
def test_explicit_support_without_general_support_stays_explicit(revised_scope):
    outputs, _ = run(scope(), blanket_support(revised_scope))
    assert all(o.final_stance == "in_favor" and o.basis == "explicit_target" for o in outputs)


def test_target_specific_evidence_or_opposition_stays_explicit():
    outputs, _ = run(scope(), blanket_support(scope(), ("Alles gut", "Parken nein")))
    assert all(o.basis == "explicit_target" and o.comment_evidence == ["Alles gut", "Parken nein"]
               for o in outputs)
    outputs, _ = run(scope(), blanket_support(scope(), stance="against"))
    assert all(o.final_stance == "against" and o.basis == "explicit_target" for o in outputs)


@pytest.mark.parametrize("global_explanation", ["Revidierte allgemeine Zustimmung.", ""])
def test_relabelled_support_explanation_falls_back_and_review_keeps_provenance(global_explanation):
    response = blanket_support(scope(explanation=global_explanation))
    response["targets"][0]["uncertainty_flags"] = ["suspected_sarcasm"]
    outputs, _ = run(scope(), response)
    assert outputs[0].status == "needs_review" and outputs[0].final_stance is None
    assert outputs[0].proposed_stance == "neutral"
    assert all(o.basis == "general_approval" for o in outputs)
    assert all(o.explanation == (global_explanation or
               "Allgemeine Zustimmung zum gesamten Beitrag, kein eigener Stance zu dieser Kernaussage.")
               for o in outputs)


@pytest.mark.parametrize("failure_stage", [1, 2])
def test_failed_call_fails_safely_without_neutral_fallback(failure_stage):
    outputs, handler = run(None if failure_stage == 1 else scope(), None)
    assert all(o.status == "failed" and o.proposed_stance is None for o in outputs)
    assert handler.call_model.await_count == failure_stage
    assert all(o.review_reasons == ["llm_call_failed", f"stage{failure_stage}_llm_call_failed"] for o in outputs)


def test_invalid_stage_one_schema_fails_without_accepted_fallback():
    outputs, handler = run({"scope_type": "made_up"}, revised())
    assert all(o.status == "failed" for o in outputs)
    assert handler.call_model.await_count == 1


@pytest.mark.parametrize("overrides", [
    {"comment_evidence": []}, {"comment_evidence": [""]}, {"comment_evidence": ["invented"]},
    {"affected_key_message_ids": ["unknown"]}, {"affected_key_message_ids": ["1:km1", "1:km1"]},
])
def test_invalid_provisional_output_fails_dependency_safely(overrides):
    first = scope()
    first.update(overrides)
    outputs, handler = run(first, revised())
    assert all(o.status == "failed" for o in outputs)
    assert handler.call_model.await_count == 1


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "empty_quote", "invented_quote",
                                      "wrong_source", "empty_source", "missing_explicit_evidence",
                                      "neutral_explicit_target", "invalid_label", "global_no_evidence",
                                      "unknown_scope_id", "global_empty_quote"])
def test_invalid_outputs_cannot_be_accepted(mutation):
    response = revised()
    target = response["targets"][1]
    if mutation == "missing":
        response["targets"].pop()
    elif mutation == "duplicate":
        response["targets"].append(copy.deepcopy(target))
    elif mutation == "unknown":
        target["key_message_id"] = "unknown"
    elif mutation == "empty_quote":
        target["comment_evidence"] = ["  "]
    elif mutation == "invented_quote":
        target["comment_evidence"] = ["not in comment"]
    elif mutation == "wrong_source":
        target["source_evidence"] = ["Original Rad sentence"]
    elif mutation == "empty_source":
        target["source_evidence"] = [""]
    elif mutation == "missing_explicit_evidence":
        target["comment_evidence"] = []
    elif mutation == "neutral_explicit_target":
        target["stance"] = "neutral"
    elif mutation == "invalid_label":
        target["stance"] = "positive"
    elif mutation == "global_no_evidence":
        response["revised_scope"]["comment_evidence"] = []
    elif mutation == "global_empty_quote":
        response["revised_scope"]["comment_evidence"] = [""]
    else:
        response["revised_scope"]["affected_key_message_ids"] = ["unknown"]
    outputs, _ = run(scope(), response)
    assert all(o.status == "needs_review" and o.final_stance is None for o in outputs)


def test_unresolved_preserves_grounded_review_proposal_not_final_label():
    response = revised()
    response["targets"][1]["attribution"] = "unresolved"
    outputs, _ = run(scope(), response)
    assert outputs[1].status == "needs_review"
    assert outputs[1].proposed_stance == "against"
    assert outputs[1].final_stance is None


def test_scope_uncertainty_routes_to_review_and_no_rejection_propagation():
    response = revised()
    response["revised_scope"] = scope("unresolved", uncertainty_flags=["unresolved_reply_reference"])
    outputs, _ = run(scope(), response)
    assert all(o.status == "needs_review" for o in outputs)
    assert outputs[0].proposed_stance == "neutral"
    assert outputs[1].proposed_stance == "against"
    assert "unresolved_reply_reference" in outputs[0].review_reasons


def test_verifier_is_rejected():
    with pytest.raises(ValueError, match="verifier"):
        RevisableAttributionEngine(AsyncMock(), "unused", "mock", verifier_enabled=True)


@pytest.mark.parametrize("review", [False, True])
@pytest.mark.parametrize("global_explanation", ["Revidierte allgemeine Zustimmung.", ""])
def test_inherited_support_explains_global_policy_for_accepted_and_review(review, global_explanation):
    response = revised()
    response["revised_scope"]["explanation"] = global_explanation
    response["targets"][0]["explanation"] = "Nicht erwaehnt, daher neutral."
    if review:
        response["targets"][0]["uncertainty_flags"] = ["suspected_sarcasm"]
    outputs, _ = run(scope(), response)
    output = outputs[0]
    assert output.proposed_stance == "neutral"
    assert output.status == ("needs_review" if review else "accepted")
    assert output.explanation == (global_explanation or
        "Allgemeine Zustimmung zum gesamten Beitrag, kein eigener Stance zu dieser Kernaussage.")
    assert output.basis == "general_approval"
    assert outputs[1].explanation == response["targets"][1].get("explanation", "")


SECRET = "MALICIOUS_SECRET_SENTINEL"


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("field,value,reason", [
    ("comment_evidence", [], "general_support_missing_evidence"),
    ("comment_evidence", [" \n "], "empty_scope_evidence"),
    ("comment_evidence", [SECRET], "nonmatching_scope_evidence"),
    ("affected_key_message_ids", [SECRET], "unknown_scope_target"),
    ("affected_key_message_ids", ["1:km1", "1:km1"], "duplicate_scope_target"),
])
def test_context_diagnostics_are_precise_and_fail_closed(stage, field, value, reason):
    first, second = scope(), revised()
    (first if stage == 1 else second["revised_scope"])[field] = value
    outputs, handler = run(first, second)
    for output in outputs:
        assert output.status == ("failed" if stage == 1 else "needs_review")
        assert output.proposed_stance is output.final_stance is None
        assert (f"stage1_{reason}" if stage == 1 else reason) in output.review_reasons
        assert SECRET not in json.dumps(output.review_reasons)
    assert handler.call_model.await_count == stage


@pytest.mark.parametrize("mutation,reason", [
    ("missing", "missing_target_ids"), ("duplicate", "duplicate_target_ids"),
    ("unknown", "unknown_target_ids"),
    ("empty_comment", "empty_comment_evidence"),
    ("nonmatching_comment", "nonmatching_comment_evidence"),
    ("empty_source", "empty_target_source_evidence"),
    ("nonmatching_source", "nonmatching_target_source_evidence"),
])
def test_revised_diagnostic_details_do_not_expose_ids_or_quotes(mutation, reason):
    response = revised()
    target = response["targets"][1]
    if mutation == "missing":
        response["targets"].pop()
    elif mutation == "duplicate":
        response["targets"].append(copy.deepcopy(target))
    elif mutation == "unknown":
        target["key_message_id"] = SECRET
    else:
        field = "comment_evidence" if mutation.endswith("comment") else "source_evidence"
        target[field] = [" " if mutation.startswith("empty") else SECRET]
    outputs, _ = run(scope(), response)
    for output in outputs:
        assert output.status == "needs_review"
        assert output.proposed_stance is output.final_stance is None
        assert reason in output.review_reasons
        assert SECRET not in json.dumps(output.review_reasons)
        if mutation in ("missing", "duplicate", "unknown"):
            assert "invalid_exact_target_coverage" in output.review_reasons
        if mutation == "unknown":
            assert "missing_target_ids" in output.review_reasons
            assert "unknown_target_source_evidence" in output.review_reasons


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("mutation,detail", [
    ("extra", "extra_forbidden:unknown_field"),
    ("enum", "enum:scope_type"),
    ("type", "string_type:comment_evidence.0"),
    ("missing", "missing:scope_type"),
])
def test_schema_diagnostics_allowlist_types_and_locations(stage, mutation, detail):
    first, second = scope(), revised()
    value = first if stage == 1 else second["revised_scope"]
    if mutation == "extra":
        value[SECRET] = SECRET
    elif mutation == "enum":
        value["scope_type"] = SECRET
    elif mutation == "type":
        value["comment_evidence"] = [{SECRET: SECRET}]
    else:
        del value["scope_type"]
    outputs, _ = run(first, second)
    for output in outputs:
        assert output.status == ("failed" if stage == 1 else "needs_review")
        assert output.proposed_stance is output.final_stance is None
        assert f"stage{stage}_schema:{detail.split(':')[0]}:" + (
            "" if stage == 1 else "revised_scope."
        ) + detail.split(':')[1] in output.review_reasons
        assert ("invalid_hypothesis_schema" if stage == 1 else "invalid_revised_schema") in output.review_reasons
        assert SECRET not in json.dumps(output.review_reasons)


def test_schema_model_validator_error_does_not_include_message_or_context():
    response = revised()
    response["targets"][1]["comment_evidence"] = []
    response["targets"][1]["explanation"] = SECRET
    outputs, _ = run(scope(), response)
    assert outputs[0].review_reasons == ["invalid_revised_schema", "stage2_schema:value_error:targets.1"]


def test_unknown_schema_error_type_and_location_are_redacted():
    from pydantic import ValidationError
    from pydantic_core import PydanticCustomError
    from revisable_attribution import _schema_error_reasons

    error = ValidationError.from_exception_data("ignored", [{
        "type": PydanticCustomError(SECRET, SECRET, {"secret": SECRET}),
        "loc": (SECRET, 2), "input": SECRET,
    }])
    assert _schema_error_reasons(error, "stage1") == ["stage1_schema:other:unknown_field.2"]


def test_uncertainty_reason_sanitization_preserves_proposal_and_review():
    response = revised()
    response["revised_scope"]["uncertainty_flags"] = [SECRET]
    response["targets"][1]["uncertainty_flags"] = [SECRET]
    outputs, _ = run(scope(), response)
    assert [o.proposed_stance for o in outputs] == ["neutral", "against", "neutral"]
    assert [o.basis for o in outputs] == ["general_approval", "explicit_target", "explicit_exclusion"]
    assert all(o.status == "needs_review" and o.final_stance is None for o in outputs)
    assert outputs[1].review_reasons == ["scope_uncertainty_flag:0", "target_uncertainty_flag:0"]
    assert SECRET not in json.dumps([o.review_reasons for o in outputs])


def test_failed_default_and_explicit_reasons():
    from policy import _failed

    ctx = context()
    assert _failed(ctx, ctx.key_messages[0]).review_reasons == ["llm_call_failed"]
    assert _failed(ctx, ctx.key_messages[0], []).review_reasons == []
    assert _failed(ctx, ctx.key_messages[0], ["stage1_unknown_scope_target"]).review_reasons == [
        "stage1_unknown_scope_target"]


@pytest.mark.parametrize("stage", [1, 2])
def test_benchmark_parquet_preserves_diagnostic_reasons(tmp_path, monkeypatch, stage):
    import benchmark
    import pandas as pd

    ctx = context()
    source = pd.DataFrame([dict(contribution_id=ctx.contribution_id, comment_text=ctx.comment_text,
                                contributionContent=ctx.contribution_text, key_message=km.text,
                                related_sentences=km.related_sentences) for km in ctx.key_messages])
    first, second = scope(), revised()
    (first if stage == 1 else second["revised_scope"])["comment_evidence"] = [SECRET]
    monkeypatch.setattr(benchmark.LLMHandler, "call_model", AsyncMock(side_effect=[first, second]))
    result = asyncio.run(benchmark.run_benchmark(source=source, base_url="unused", model_name="mock",
                         mode="revisable_attribution", verify_ionos=False))
    path = tmp_path / "reasons.parquet"
    result.to_parquet(path)
    restored = pd.read_parquet(path)
    assert [list(v) for v in restored.review_reasons] == result.review_reasons.tolist()
    assert set(restored.status) == {"failed" if stage == 1 else "needs_review"}
    assert restored.proposed_stance.isna().all() and restored.final_stance.isna().all()
    assert SECRET not in json.dumps(result.review_reasons.tolist())


def test_response_schema_and_prompt_hashes_unchanged():
    from revisable_attribution import GlobalHypothesis, RevisedAttributionResult, PROMPT_CONFIG

    for schema, digest in [
        (GlobalHypothesis, "77f7f3a469247620de07c6ec896e6f8c958a2365014538fec1b2455501d8e4b7"),
        (RevisedAttributionResult, "5e9445eb3a65df5fb23f68fbddfa58017c6bd5ae7e20d7e2f2b06143b0ba904e"),
    ]:
        assert hashlib.sha256(json.dumps(schema.model_json_schema(), sort_keys=True).encode()).hexdigest() == digest
    for filename, digest in [
        ("revisable_attribution_system.txt", "70058ec56f9f0c93a62e1f5fb2fc8f83984dc4cb483733db988e7dcae39a62fa"),
        ("global_hypothesis_user.txt", "e4f1e3dc74e3dfc6f91f9deffce575b00cb7ebf06d977d6fdc98d484d9269da3"),
        ("revised_attribution_user.txt", "fe51530b3d1eb9d6b612c27ad607d863d85f87141e417d30791c239ddb42d4c4"),
    ]:
        assert hashlib.sha256((Path(PROMPT_CONFIG).parent / filename).read_bytes()).hexdigest() == digest


def keyed_revised():
    response = revised()
    response["targets"] = {target["key_message_id"]: {k: v for k, v in target.items() if k != "key_message_id"}
                           for target in reversed(response["targets"])}
    return response


def test_required_factory_aliases_fields_and_original_order():
    from revisable_attribution import required_targets_schema, canonical_revised_result, TargetAssessment

    ids = ["a:b", "model_dump", "with space", "target_0"]
    schema = required_targets_schema(ids)
    wire = {"revised_scope": scope(), "targets": {i: {"attribution": "unaddressed"} for i in reversed(ids)}}
    result = canonical_revised_result(schema.model_validate(wire), ids)
    assert [target.key_message_id for target in result.targets] == ids
    definition = schema.model_json_schema()["$defs"]["RequiredTargets"]
    assert definition["required"] == ids
    assert list(definition["properties"]) == ids
    assert definition["additionalProperties"] is False
    value_fields = schema.model_fields["targets"].annotation.model_fields["target_0"].annotation.model_fields
    assert set(value_fields) == set(TargetAssessment.model_fields) - {"key_message_id"}
    for name, field in value_fields.items():
        assert field.asdict() == TargetAssessment.model_fields[name].asdict()


@pytest.mark.parametrize("mutation", ["missing", "extra", "null", "list", "internal", "extra_value", "id_value"])
def test_required_factory_rejects_invalid_wire(mutation):
    from pydantic import ValidationError
    from revisable_attribution import required_targets_schema, _schema_error_reasons

    ids = [km.key_message_id for km in context().key_messages]
    response = keyed_revised()
    if mutation == "missing":
        del response["targets"][ids[1]]
    elif mutation == "extra":
        response["targets"][SECRET] = {"attribution": "unaddressed"}
    elif mutation in ("null", "list"):
        response["targets"] = None if mutation == "null" else revised()["targets"]
    elif mutation == "internal":
        response["targets"]["target_0"] = response["targets"].pop(ids[0])
    else:
        response["targets"][ids[0]]["key_message_id" if mutation == "id_value" else SECRET] = SECRET
    with pytest.raises(ValidationError) as caught:
        required_targets_schema(ids).model_validate(response)
    reasons = _schema_error_reasons(caught.value, "stage2", ids)
    assert SECRET not in json.dumps(reasons)
    if mutation == "missing":
        assert reasons == ["stage2_schema:missing:targets.1"]


@pytest.mark.parametrize("ids", [[], ["a", "a"], [""], [None]])
def test_required_factory_rejects_bad_context_ids(ids):
    from revisable_attribution import required_targets_schema
    with pytest.raises(ValueError):
        required_targets_schema(ids)


@pytest.mark.parametrize("variant", ["baseline", "stage1_comment_only", "stage2_required_targets"])
def test_variants_are_isolated_and_preserve_policy(variant):
    from revisable_attribution import GlobalHypothesis, RevisedAttributionResult

    baseline_outputs, baseline = run(scope(), revised())
    handler = AsyncMock()
    handler.call_model.side_effect = [scope(), keyed_revised() if variant == "stage2_required_targets" else revised()]
    engine = RevisableAttributionEngine(handler, "http://unused", "mock", revisable_variant=variant)
    outputs = asyncio.run(engine.resolve_comment(None, None, context()))
    assert outputs == baseline_outputs
    first, second = [call.kwargs for call in handler.call_model.call_args_list]
    base_first, base_second = [call.kwargs for call in baseline.call_model.call_args_list]
    assert first["response_schema"] is GlobalHypothesis
    assert "reject_duplicate_json_keys" not in first
    if variant == "stage1_comment_only":
        assert first["messages"] != base_first["messages"]
        assert first["system"] == base_first["system"]
        assert second == base_second
        prompt = first["messages"][0]["content"]
        assert "obersten Feld comment_text" in prompt and "Nicht paraphrasieren" in prompt
        assert "Original full contribution" in prompt and "Elterntext" in prompt
    elif variant == "stage2_required_targets":
        assert first == base_first
        assert second["reject_duplicate_json_keys"] is True
        assert second["system"] == base_second["system"]
        assert second["response_schema"] is not RevisedAttributionResult
    else:
        assert first == base_first and second == base_second


@pytest.mark.parametrize("mutation,status,reason", [
    ("missing", "needs_review", "stage2_schema:missing:targets.0"),
    ("semantic", "needs_review", "stage2_schema:value_error:targets.1"),
    ("evidence", "needs_review", "nonmatching_comment_evidence"),
    ("null", "failed", "stage2_llm_call_failed"),
    ("duplicate", "failed", "stage2_duplicate_json_key"),
])
def test_required_variant_failures_keep_null_labels(mutation, status, reason):
    from llm_handler import DuplicateJSONKeyError
    response = keyed_revised()
    if mutation == "missing":
        del response["targets"]["1:km1"]
    elif mutation == "semantic":
        response["targets"]["1:km2"]["stance"] = "neutral"
    elif mutation == "evidence":
        response["targets"]["1:km2"]["comment_evidence"] = [SECRET]
    elif mutation == "null":
        response = None
    else:
        response = DuplicateJSONKeyError()
    handler = AsyncMock()
    handler.call_model.side_effect = [scope(), response]
    engine = RevisableAttributionEngine(handler, "unused", "mock", revisable_variant="stage2_required_targets")
    outputs = asyncio.run(engine.resolve_comment(None, None, context()))
    assert all(o.status == status and o.proposed_stance is o.final_stance is None for o in outputs)
    assert reason in outputs[0].review_reasons
    assert SECRET not in json.dumps(outputs[0].review_reasons)
