import asyncio
import copy
import json
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

import benchmark
import cli
import revisable_attribution as ra
from context_builder import ParentContext, build_comment_context
from llm_handler import DuplicateJSONKeyError, schema_response_format


VARIANT = "source_bound_evidence_ids"


def context(comment="Ja! Aber nicht dort."):
    return build_comment_context(
        run_id="private-run", contribution_id="1", comment_text=comment,
        contribution_text="Full contribution", parent_context=ParentContext(
            available=True, parent_comment_text="Parent context"),
        key_messages=[{"key_message": text, "related_sentences": [" Same source.\r\n", "", " \t"]}
                      for text in ("First target", "Second target")],
    )


def responses(ctx):
    table = ra.build_evidence_table(ctx)
    comment = next(e["id"] for e in table if e["id"].startswith("c:"))
    scope = {"scope_type": "whole_contribution_approval", "comment_evidence_ids": [comment]}
    revised = {"revised_scope": copy.deepcopy(scope), "targets": [
        {"key_message_id": km.key_message_id, "attribution": "unaddressed"} for km in ctx.key_messages]}
    return scope, revised


def run(ctx, values, variant=VARIANT):
    handler = AsyncMock()
    handler.call_model.side_effect = values
    output = asyncio.run(ra.RevisableAttributionEngine(handler, "unused", "mock", revisable_variant=variant)
                         .resolve_comment(None, None, ctx))
    return output, handler


@pytest.mark.parametrize("text", [
    "", " \r\n\t ", "Hallo. Weiter! Schluss?", '  \u201eJa!\u201c\r\n(Nein?)  Ende. ',
    "Dr. M. Beispiel kommt. Weiter.", "Das ist z. B. gut. Weiter.", "Preis 3.14 bzw. ca. 2.5 Euro.",
    "a\u0308\U0001f642! \u00df?\r\n\u4e2d\u6587\u3002", "word\u2028next\u2029last\x85end", "...!? ) \n\n !!!",
    "https://example.org is here. Next.", "a.b. Word.\r\nlast", "Done.) Next.\rEnd",
])
def test_segmentation_exact_nonoverlapping_complete_unicode(text):
    entries = [e for e in ra.build_evidence_table(context(text)) if e["id"].startswith("c:")]
    covered = []
    for entry in entries:
        assert entry["text"] == text[entry["start"]:entry["end"]]
        assert entry["text"] == entry["text"].strip() and entry["text"]
        covered.extend(range(entry["start"], entry["end"]))
    assert covered == sorted(set(covered))
    assert {i for i, c in enumerate(text) if not c.isspace()} <= set(covered)
    assert list(ra.comment_evidence_spans(text)) == [(e["start"], e["end"]) for e in entries]


def test_conservative_abbreviations_closures_and_no_alternate_whole_comment():
    text = 'Dr. A. Meier kommt. \u201eJa!\u201c\r\nEnde.'
    spans = [text[a:b] for a, b in ra.comment_evidence_spans(text)]
    assert spans == ['Dr. A. Meier kommt.', '\u201eJa!\u201c', 'Ende.']
    assert "Meier" not in spans and text not in spans


def test_run_identity_namespaces_whole_sources_target_binding_and_staleness():
    ctx = context("Same source.")
    table = ra.build_evidence_table(ctx)
    assert table == ra.build_evidence_table(ctx.model_copy(update={"run_id": "another"}))
    assert benchmark.revisable_context_metadata(ctx, VARIANT) == benchmark.revisable_context_metadata(
        ctx.model_copy(update={"run_id": "another"}), VARIANT)
    assert len({e["id"] for e in table}) == len(table) == 3
    for entry in table:
        assert len(entry["id"].split(":")[1]) == 64
        assert set(entry) == {"id", "source_field_path", "target_binding", "source_index", "start", "end", "text"}
        if entry["target_binding"]:
            assert entry["text"] == " Same source.\r\n" and entry["source_index"] == 0
            assert entry["start"] == 0 and entry["end"] == len(entry["text"])
        else:
            assert entry["source_field_path"] == "comment_text"
    for changes in ({"comment_text": "different"}, {"contribution_text": "different"},
                    {"parent_context": ParentContext()}, {"context_flags": ["different"]}):
        assert not {e["id"] for e in table} & {e["id"] for e in ra.build_evidence_table(ctx.model_copy(update=changes))}
    changed = ctx.model_copy(deep=True)
    changed.key_messages[0].text = "Changed target"
    assert not {e["id"] for e in table} & {e["id"] for e in ra.build_evidence_table(changed)}


def test_collision_and_duplicate_target_checks(monkeypatch):
    ctx = context()
    ctx.key_messages[1].key_message_id = ctx.key_messages[0].key_message_id
    with pytest.raises(ValueError, match="Duplicate target"):
        ra.build_evidence_table(ctx)
    monkeypatch.setattr(ra, "comment_evidence_spans", lambda text: [(0, 1), (0, 1)])
    with pytest.raises(ValueError, match="collision"):
        ra.build_evidence_table(context())


def test_schema_enumerations_empty_lists_and_generic_targets():
    ctx = context()
    first, second = ra.evidence_id_schemas(ctx)
    table = ra.build_evidence_table(ctx)
    schema = second.model_json_schema()
    assert schema["properties"]["targets"]["type"] == "array"
    assert "RequiredTargets" not in json.dumps(schema)
    for wire in (first, second):
        encoded = json.dumps(wire.model_json_schema())
        assert '"comment_evidence"' not in encoded and '"source_evidence"' not in encoded
    assert set(first.model_json_schema()["properties"]["comment_evidence_ids"]["items"]["enum"]) == {
        e["id"] for e in table if e["id"].startswith("c:")}
    target_schema = schema["$defs"]["EvidenceIDTarget"]
    assert set(target_schema["properties"]["source_evidence_ids"]["items"]["enum"]) == {
        e["id"] for e in table if e["id"].startswith("s:")}
    empty = context(" \r\n")
    for km in empty.key_messages:
        km.related_sentences = ["", " \n"]
    first, second = ra.evidence_id_schemas(empty)
    assert ra.build_evidence_table(empty) == []
    for prop in (first.model_json_schema()["properties"]["comment_evidence_ids"],
                 second.model_json_schema()["$defs"]["EvidenceIDTarget"]["properties"]["source_evidence_ids"]):
        assert prop["maxItems"] == 0 and prop["items"]["type"] == "null"
    first.model_validate({"scope_type": "unresolved", "comment_evidence_ids": []})
    for value in (["sentinel"], [None]):
        with pytest.raises(ValidationError):
            first.model_validate({"scope_type": "unresolved", "comment_evidence_ids": value})


@pytest.mark.parametrize("location", ["stage1", "scope", "comment", "source"])
@pytest.mark.parametrize("mutation", ["unknown", "stale", "wrongfield", "offset", "duplicate", "legacy"])
def test_all_evidence_fields_fail_closed(location, mutation):
    ctx = context()
    first, second = responses(ctx)
    table = ra.build_evidence_table(ctx)
    comment_id = first["comment_evidence_ids"][0]
    source_id = next(e["id"] for e in table if e["id"].startswith("s:"))
    record = first if location == "stage1" else second["revised_scope"] if location == "scope" else second["targets"][0]
    field = "source_evidence_ids" if location == "source" else "comment_evidence_ids"
    valid = source_id if location == "source" else comment_id
    record[field] = [valid]
    if mutation == "unknown":
        record[field] = ["PRIVATE_UNKNOWN"]
    elif mutation == "stale":
        record[field] = [ra.build_evidence_table(context("Changed!"))[0]["id"]]
    elif mutation == "wrongfield":
        record[field] = [comment_id if location == "source" else source_id]
    elif mutation == "offset":
        record[field] = [valid.rsplit(":", 1)[0] + ":999999"]
    elif mutation == "duplicate":
        record[field] *= 2
    else:
        record[field.removesuffix("_ids")] = ["Ja!"]
    outputs, handler = run(ctx, [first, second])
    assert handler.call_model.await_count == (1 if location == "stage1" else 2)
    assert {o.status for o in outputs} == {"failed" if location == "stage1" else "needs_review"}
    assert all(o.final_stance is None and o.proposed_stance is None and not o.comment_evidence for o in outputs)
    assert "PRIVATE_UNKNOWN" not in json.dumps([o.review_reasons for o in outputs])


def test_wrong_target_identical_text_has_no_content_fallback():
    ctx = context()
    first, second = responses(ctx)
    second["targets"][0]["source_evidence_ids"] = [next(
        e["id"] for e in ra.build_evidence_table(ctx) if e["target_binding"] == ctx.key_messages[1].key_message_id)]
    wire = ra.evidence_id_schemas(ctx)[1].model_validate(second)
    with pytest.raises(ValueError, match="binding"):
        ra.resolve_evidence_ids(wire, ctx)
    outputs, _ = run(ctx, [first, second])
    assert all(o.review_reasons == ["stage2_invalid_evidence_binding"] for o in outputs)


def test_resolver_revalidates_current_context_even_for_prevalidated_wire():
    ctx = context()
    first, _ = responses(ctx)
    wire = ra.evidence_id_schemas(ctx)[0].model_validate(first)
    with pytest.raises(ValueError, match="binding"):
        ra.resolve_evidence_ids(wire, ctx.model_copy(update={"contribution_text": "changed"}))
    forged = wire.model_copy(update={"comment_evidence_ids": [first["comment_evidence_ids"][0] + "0"]})
    with pytest.raises(ValueError, match="binding"):
        ra.resolve_evidence_ids(forged, ctx)


@pytest.mark.parametrize("mutation", ["support_without_evidence", "unknown_scope", "duplicate_scope"])
def test_invalid_stage1_canonical_policy_never_reaches_stage2(mutation):
    ctx = context()
    first, second = responses(ctx)
    if mutation == "support_without_evidence":
        first["comment_evidence_ids"] = []
    elif mutation == "unknown_scope":
        first["affected_key_message_ids"] = ["unknown"]
    else:
        first["affected_key_message_ids"] = [ctx.key_messages[0].key_message_id] * 2
    outputs, handler = run(ctx, [first, second])
    assert handler.call_model.await_count == 1
    assert all(o.status == "failed" and o.final_stance is None for o in outputs)


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("error", [None, DuplicateJSONKeyError()])
def test_null_and_duplicate_json_stage_dispositions(stage, error):
    ctx = context()
    first, _ = responses(ctx)
    outputs, handler = run(ctx, [error] if stage == 1 else [first, error])
    assert handler.call_model.await_count == stage
    assert all(o.status == ("needs_review" if stage == 2 and error is not None else "failed") for o in outputs)
    assert all(c.kwargs["reject_duplicate_json_keys"] is True for c in handler.call_model.call_args_list)
    assert all(c.kwargs["strict_json"] is True for c in handler.call_model.call_args_list)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "no_evidence", "uncertain", "proposal"])
def test_coverage_canonical_semantics_and_policy_parity(mutation):
    ctx = context()
    first, second = responses(ctx)
    if mutation == "missing":
        second["targets"].pop()
    elif mutation == "duplicate":
        second["targets"].append(copy.deepcopy(second["targets"][0]))
    elif mutation == "unknown":
        second["targets"][0]["key_message_id"] = "other"
    elif mutation == "no_evidence":
        second["targets"][0].update(attribution="explicit_target", stance="against")
    elif mutation == "uncertain":
        second["revised_scope"].update(scope_type="unresolved", uncertainty_flags=["suspected_sarcasm"])
    else:
        second["targets"][0].update(attribution="unresolved", stance="against",
                                     comment_evidence_ids=first["comment_evidence_ids"])
    outputs, _ = run(ctx, [first, second])
    assert all(o.status == "needs_review" for o in outputs) if mutation != "proposal" else outputs[0].status == "needs_review"
    if mutation == "no_evidence":
        assert all(o.proposed_stance is None for o in outputs)
        return
    schemas = ra.evidence_id_schemas(ctx)
    canonical = [ra.resolve_evidence_ids(s.model_validate(v), ctx) for s, v in zip(schemas, (first, second))]
    baseline, _ = run(ctx, canonical, "baseline")
    assert outputs == baseline
    if mutation == "proposal":
        assert outputs[0].proposed_stance == "against" and outputs[0].final_stance is None


def test_stage2_receives_validated_wire_and_exact_canonical_quotes():
    ctx = context()
    first, second = responses(ctx)
    second["targets"][0].update(attribution="explicit_target", stance="against",
                                 comment_evidence_ids=first["comment_evidence_ids"],
                                 source_evidence_ids=[next(e["id"] for e in ra.build_evidence_table(ctx)
                                                          if e["target_binding"] == ctx.key_messages[0].key_message_id)])
    outputs, handler = run(ctx, [first, second])
    assert [o.final_stance for o in outputs] == ["against", "neutral"]
    assert [o.basis for o in outputs] == ["explicit_target", "general_approval"]
    assert outputs[0].comment_evidence == ["Ja!"] and outputs[0].source_evidence == [" Same source.\r\n"]
    calls = handler.call_model.call_args_list
    prompt = calls[1].kwargs["messages"][0]["content"]
    assert ra.evidence_id_schemas(ctx)[0].model_validate(first).model_dump_json() in prompt
    assert '"comment_evidence":' not in prompt
    for call in calls:
        prompt = call.kwargs["messages"][0]["content"]
        assert all(text in prompt for text in ("Full contribution", "Parent context", "First target", "Second target"))
        assert "private-run" not in prompt


@pytest.mark.parametrize("scope_type,extends,attribution,stance", [
    ("whole_contribution_approval", None, "unaddressed", None),
    ("mixed_qualified", True, "explicit_exclusion", "neutral"),
    ("mixed_qualified", False, "no_position", None),
    ("target_specific", None, "explicit_target", "against"),
    ("target_specific", None, "explicit_target", "in_favor"),
    ("unresolved", None, "unresolved", None),
    ("unresolved", None, "unresolved", "neutral"),
])
def test_unchanged_policy_on_resolved_ids(scope_type, extends, attribution, stance):
    ctx = context()
    first, second = responses(ctx)
    second["revised_scope"].update(scope_type=scope_type, general_support_extends_to_unaddressed=extends)
    second["targets"][0].update(attribution=attribution, stance=stance,
                                 comment_evidence_ids=first["comment_evidence_ids"])
    schemas = ra.evidence_id_schemas(ctx)
    canonical = [ra.resolve_evidence_ids(s.model_validate(v), ctx) for s, v in zip(schemas, (first, second))]
    outputs, _ = run(ctx, [first, second])
    baseline, _ = run(ctx, canonical, "baseline")
    assert outputs == baseline
    if scope_type == "unresolved" and stance is None:
        assert outputs[0].proposed_stance is None and outputs[0].final_stance is None


def test_manifest_actual_formats_canonical_and_cli_offline(tmp_path, monkeypatch):
    from tests.test_benchmark_readiness import source
    monkeypatch.setattr(cli, "load_dotenv", lambda: pytest.fail("credentials"))
    monkeypatch.setattr(benchmark, "verify_ionos_only", lambda *a: pytest.fail("network"))
    path = tmp_path / "source.parquet"
    source().to_parquet(path)
    cli.main(["benchmark", "--source", str(path), "--output", str(tmp_path / "dry.parquet"),
              "--mode", "revisable_attribution", "--revisable-variant", VARIANT, "--dry-run"])
    manifest = json.loads((tmp_path / "dry.manifest.json").read_text())
    diagnostic = manifest["revisable_diagnostic"]
    assert diagnostic["canonical_schemas"] == {"stage1": ra.GlobalHypothesis.model_json_schema(),
                                              "stage2": ra.RevisedAttributionResult.model_json_schema()}
    assert diagnostic["strict_duplicate_json_parser"] is True
    assert diagnostic["strict_json"] is True
    for entry, metadata in zip(benchmark.load_frozen_comments(source()), diagnostic["contexts"]):
        ctx = build_comment_context(run_id="new", **entry)
        assert metadata["context_sha256"] == benchmark.revisable_context_metadata(ctx, "baseline")["context_sha256"]
        for stage, schema in zip(("stage1", "stage2"), ra.evidence_id_schemas(ctx)):
            assert metadata[stage + "_response_format"]["response_format"] == schema_response_format(schema)
            assert metadata[stage + "_response_format"]["schema"] == schema.model_json_schema()
        assert metadata["evidence_table"]["sha256"] == benchmark.json_sha256(ra.build_evidence_table(ctx))
        assert metadata["evidence_table"]["strict_json"] is True
    assert all(text not in json.dumps(diagnostic) for text in ("Full source", "Comment one", "SECRET"))
    for mode, extra in (("two_stage", []), ("one_call", []), ("revisable_attribution", ["--verifier"])):
        with pytest.raises(ValueError):
            cli.main(["benchmark", "--source", "missing", "--output", "unused", "--mode", mode,
                      "--revisable-variant", VARIANT, *extra])


@pytest.mark.parametrize("stage", [1, 2])
def test_real_parser_rejects_duplicate_json_at_both_stages(monkeypatch, stage):
    import httpx
    import llm_handler
    monkeypatch.setattr(llm_handler, "RATE_LIMIT_DELAY", 0)
    ctx = context()
    first, second = responses(ctx)
    payloads = [json.dumps(first), json.dumps(second)]
    payloads[stage - 1] = payloads[stage - 1][:-1] + ',"scope_type":"unresolved","scope_type":"target_specific"}'
    calls = []

    class Client:
        async def post(self, *a, **kw):
            calls.append(kw)
            return httpx.Response(200, json={"choices": [{"message": {"content": payloads[len(calls) - 1]}}]},
                                  request=httpx.Request("POST", "https://unused"))

    output = asyncio.run(ra.RevisableAttributionEngine(llm_handler.LLMHandler(), "https://unused", "mock",
        revisable_variant=VARIANT).resolve_comment(Client(), asyncio.Semaphore(1), ctx))
    assert len(calls) == stage
    assert all(o.review_reasons == [f"stage{stage}_duplicate_json_key"] for o in output)


@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("wrapper", ["```json\n%s\n```", "```\n%s\n```", "PRIVATE %s", "%s PRIVATE", "%s trailing",
                                    "<think>PRIVATE</think>%s", "%s,", "%s %s"])
def test_real_parser_rejects_non_json_at_both_stages(monkeypatch, capsys, stage, wrapper):
    import httpx
    import llm_handler
    monkeypatch.setattr(llm_handler, "RATE_LIMIT_DELAY", 0)
    ctx = context()
    payloads = [json.dumps(value) for value in responses(ctx)]
    payloads[stage - 1] = wrapper.replace("%s", payloads[stage - 1])
    calls = []

    class Client:
        async def post(self, *a, **kw):
            calls.append(kw)
            return httpx.Response(200, json={"choices": [{"message": {"content": payloads[len(calls) - 1]}}]},
                                  request=httpx.Request("POST", "https://unused"))

    output = asyncio.run(ra.RevisableAttributionEngine(llm_handler.LLMHandler(), "https://unused", "mock",
        revisable_variant=VARIANT).resolve_comment(Client(), asyncio.Semaphore(1), ctx))
    assert len(calls) == stage
    assert all(o.status == "failed" and o.final_stance is None and o.proposed_stance is None for o in output)
    assert all(o.review_reasons == ["llm_call_failed", f"stage{stage}_llm_call_failed"] for o in output)
    assert "PRIVATE" not in capsys.readouterr().out
