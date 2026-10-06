import asyncio
import hashlib
import json

import pandas as pd
import pytest

import benchmark
import cli
from dataset_builder import compute_pair_id
from ionos_config import IonosVerificationError, ModelRouteInfo


def source():
    return pd.DataFrame([dict(contribution_id="1", contributionContent="Full source",
                             comment_text=comment, key_message=target, related_sentences=[target],
                             parent_context=dict(available=True, parent_kind="comment", parent_comment_text="Parent"),
                             reference_label="SECRET_GOLD_NOT_FOR_PROMPTS", comment="ANNOTATION_NOTE")
                         for comment in ("Comment one", "Comment two") for target in ("Target a", "Target b")])


def test_dry_run_is_offline_reproducible_full_target_cohort_and_no_overwrite(tmp_path, monkeypatch):
    source_path = tmp_path / "source.parquet"
    source().to_parquet(source_path)
    monkeypatch.setattr(cli, "load_dotenv", lambda: pytest.fail("dry-run must not load secrets"))
    monkeypatch.setattr(cli, "run_benchmark", lambda **kw: pytest.fail("dry-run must not infer"))
    monkeypatch.setattr(benchmark, "verify_ionos_only", lambda *a, **kw: pytest.fail("no provider checks"))
    base = ["benchmark", "--source", str(source_path), "--reference", str(tmp_path / "absent.parquet"),
            "--folds", str(tmp_path / "absent.json"), "--mode", "revisable_attribution",
            "--dry-run", "--limit", "1", "--model-name", "configured-model"]
    for name in ("first", "second"):
        cli.main(base + ["--output", str(tmp_path / f"{name}.parquet")])
    manifest = json.loads((tmp_path / "first.manifest.json").read_text())
    assert manifest == json.loads((tmp_path / "second.manifest.json").read_text())
    expected = pd.read_parquet(tmp_path / "first.expected.parquet")
    assert len(expected) == 2
    assert set(expected.pair_id) == {compute_pair_id("1", target, "Comment one") for target in ("Target a", "Target b")}
    assert manifest["expected_pairs"] == expected.to_dict("records")
    assert manifest["input_hashes"]["source"] == hashlib.sha256(source_path.read_bytes()).hexdigest()
    assert not (tmp_path / "first.parquet").exists()
    assert len(manifest["stages"]) == 2
    assert manifest["decoding"]["temperature"] == .1
    with pytest.raises(FileExistsError):
        cli.main(base + ["--output", str(tmp_path / "first.parquet")])


def test_benchmark_prepares_before_inference_and_scores_explicit_subset(tmp_path, monkeypatch, capsys):
    source_path = tmp_path / "source.parquet"
    source().to_parquet(source_path)
    expected = benchmark.expected_cohort(source(), limit=1)
    ref = expected.assign(resolution_status="resolved", reference_label="neutral")
    other = pd.DataFrame([dict(pair_id=f"outside-{i}", contribution_id="outside",
                               resolution_status="resolved", reference_label="neutral") for i in range(98)])
    ref_path = tmp_path / "ref.parquet"
    pd.concat([ref, other]).to_parquet(ref_path)
    output = tmp_path / "run.parquet"
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)

    async def fake_run(**kwargs):
        assert output.with_suffix(".manifest.json").exists()
        assert output.with_suffix(".expected.parquet").exists()
        checkpoint = kwargs["checkpoint"]
        records = [{**identity, "status": "accepted", "proposed_stance": "neutral", "final_stance": "neutral",
                    "basis": "unaddressed", "comment_evidence": [], "source_evidence": [],
                    "explanation": "", "review_reasons": []} for identity in checkpoint.identities[0]]
        records[1].update(status="failed", proposed_stance=None, final_stance=None, basis=None,
                          review_reasons=["llm_call_failed"])
        checkpoint.commit_comment(records)
        return checkpoint.frame()

    monkeypatch.setattr(cli, "run_benchmark", fake_run)
    cli.main(["benchmark", "--source", str(source_path), "--reference", str(ref_path),
              "--folds", "", "--output", str(output), "--limit", "1", "--model-name", "mock"])
    assert "forced_choice_accuracy=0.500 (1/2)" in capsys.readouterr().out
    metrics_path = tmp_path / "metrics.json"
    cli.main(["evaluate", "--predictions", str(output), "--reference", str(ref_path), "--folds", "",
              "--expected-cohort", str(output.with_suffix(".expected.parquet")), "--role", "primary",
              "--tiers", "A", "--output", str(metrics_path)])
    metrics = json.loads(metrics_path.read_text())["metrics"]
    assert metrics["eligible_pairs"] == 2
    assert metrics["forced_choice_accuracy"] == .5


def test_experimental_runner_routes_and_never_leaks_annotations(monkeypatch):
    calls = []

    async def fake_call(self, **kwargs):
        calls.append(kwargs)
        scope = dict(scope_type="whole_contribution_approval", comment_evidence=["Comment"])
        if len(calls) == 1:
            return scope
        return dict(revised_scope=scope, targets=[dict(key_message_id=f"1:km{i}", attribution="unaddressed")
                                                  for i in (1, 2)])

    monkeypatch.setattr(benchmark.LLMHandler, "call_model", fake_call)
    result = asyncio.run(benchmark.run_benchmark(source=source(), base_url="unused", model_name="mock",
                                                 verify_ionos=False, mode="revisable_attribution", limit=1))
    assert len(result) == 2
    assert set(result.final_stance) == {"in_favor"}
    for call in calls:
        prompt = call["messages"][0]["content"]
        assert "SECRET_GOLD" not in prompt and "ANNOTATION_NOTE" not in prompt
        assert "Parent" in prompt and "Full source" in prompt


def test_experimental_verifier_rejected_before_provider_check(monkeypatch):
    monkeypatch.setattr(benchmark, "verify_ionos_only", lambda *a, **kw: pytest.fail("network"))
    with pytest.raises(ValueError, match="verifier"):
        asyncio.run(benchmark.run_benchmark(source=source(), base_url="unused", model_name="mock",
                                             mode="revisable_attribution", verifier_enabled=True))


def test_parser_defaults_unchanged():
    args = cli.build_parser().parse_args(["benchmark", "--source", "input", "--output", "output"])
    assert args.mode == "two_stage"
    assert not args.verifier
    assert not args.dry_run
    assert args.revisable_variant == "baseline"


@pytest.mark.parametrize("variant", ["stage1_comment_only", "stage2_required_targets"])
@pytest.mark.parametrize("mode,verifier", [("two_stage", False), ("one_call", False), ("revisable_attribution", True)])
def test_invalid_variant_combinations_fail_before_io(tmp_path, monkeypatch, variant, mode, verifier):
    monkeypatch.setattr(benchmark, "verify_ionos_only", lambda *a: pytest.fail("network"))
    monkeypatch.setattr(cli, "load_dotenv", lambda: pytest.fail("credentials"))
    with pytest.raises(ValueError):
        asyncio.run(benchmark.run_benchmark(source=source(), base_url="unused", model_name="mock",
                    mode=mode, verifier_enabled=verifier, revisable_variant=variant))
    with pytest.raises(ValueError):
        cli.main(["benchmark", "--source", "does-not-exist", "--output", str(tmp_path / "out.parquet"),
                  "--mode", mode, "--revisable-variant", variant] + (["--verifier"] if verifier else []))
    with pytest.raises(ValueError):
        benchmark.prepare_benchmark(source=source(), source_path=None, reference_path=None, folds_path=None,
            output_path=tmp_path / "out.parquet", model_name="mock", mode=mode,
            verifier_enabled=verifier, revisable_variant=variant)


@pytest.mark.parametrize("variant", ["baseline", "stage1_comment_only", "stage2_required_targets"])
def test_variant_manifest_matches_actual_requests_and_source_allowlist(tmp_path, monkeypatch, variant):
    from context_builder import build_comment_context
    from llm_handler import schema_response_format
    from revisable_attribution import GlobalHypothesis
    import pydantic

    frame = source()
    frame["explanation"] = "EXTRACTION_SECRET"
    path = tmp_path / "source.parquet"
    frame.to_parquet(path)
    _, manifest = benchmark.prepare_benchmark(source=frame, source_path=path, reference_path=None, folds_path=None,
        output_path=tmp_path / "out.parquet", model_name="mock", mode="revisable_attribution",
        dry_run=True, revisable_variant=variant)
    diagnostic = manifest["revisable_diagnostic"]
    assert diagnostic["variant"] == variant
    assert diagnostic["dependencies"]["pydantic"] == pydantic.__version__
    assert diagnostic["contexts_sha256"] == benchmark.json_sha256(diagnostic["contexts"])
    calls = []

    async def fake_call(self, **kwargs):
        index = len(calls) // 2
        calls.append(kwargs)
        response_format = schema_response_format(kwargs["response_schema"])
        scope = {"scope_type": "target_specific"}
        if kwargs["response_schema"] is GlobalHypothesis:
            assert benchmark.json_sha256(response_format) == diagnostic["stage1_response_format_sha256"]
            return scope
        contract = diagnostic["contexts"][index]["stage2_response_format"]
        assert benchmark.json_sha256(response_format) == contract["sha256"]
        assert benchmark.json_sha256(response_format["json_schema"]["schema"]) == contract["schema_sha256"]
        targets = {i: {"attribution": "unaddressed"} for i in diagnostic["contexts"][index]["target_ids"]}
        return {"revised_scope": scope, "targets": targets if variant == "stage2_required_targets" else
                [{"key_message_id": i, **value} for i, value in targets.items()]}

    monkeypatch.setattr(benchmark.LLMHandler, "call_model", fake_call)
    result = asyncio.run(benchmark.run_benchmark(source=frame, base_url="unused", model_name="mock",
                         mode="revisable_attribution", verify_ionos=False, revisable_variant=variant))
    assert len(calls) == 4 and set(result.status) == {"accepted"}
    for call in calls:
        prompt = call["messages"][0]["content"]
        assert "Full source" in prompt and "Parent" in prompt
        assert not any(text in prompt for text in ("SECRET", "ANNOTATION_NOTE"))
    assert not any(text in json.dumps(diagnostic) for text in ("Comment one", "Full source", "Parent", "SECRET"))
    entry = benchmark.load_frozen_comments(frame)[0]
    assert benchmark.revisable_context_metadata(build_comment_context(run_id="different", **entry), variant) == diagnostic["contexts"][0]


@pytest.mark.parametrize("column,value", [("related_sentences", ["other"]),
                                         ("contributionContent", "other full context")])
def test_conflicting_frozen_context_is_not_arbitrarily_selected(column, value):
    rows = pd.concat([source().iloc[:1], source().iloc[:1]], ignore_index=True)
    rows.at[1, column] = value
    with pytest.raises(ValueError, match="Conflicting frozen"):
        benchmark.expected_cohort(rows)


def test_repeated_live_preparations_share_experiment_but_not_run_identity(tmp_path):
    path = tmp_path / "source.parquet"
    source().to_parquet(path)
    manifests = [benchmark.prepare_benchmark(
        source=source(), source_path=path, reference_path=None, folds_path=None,
        output_path=tmp_path / f"run-{i}.parquet", model_name="mock",
    )[1] for i in range(2)]
    assert manifests[0]["experiment_id"] == manifests[1]["experiment_id"]
    assert manifests[0]["run_id"] != manifests[1]["run_id"]


def test_endpoint_and_verification_independently_change_manifest_identity(tmp_path):
    path = tmp_path / "source.parquet"
    source().to_parquet(path)
    manifests = []
    for i, (endpoint, verify) in enumerate([
        ("https://proxy-a.invalid/v1", True),
        ("https://proxy-b.invalid/v1", True),
        ("https://proxy-a.invalid/v1", False),
        ("https://user:password@proxy-a.invalid:443/v1/?token=private#private", True),
    ]):
        manifests.append(benchmark.prepare_benchmark(
            source=source(), source_path=path, reference_path=None, folds_path=None,
            output_path=tmp_path / f"prepared-{i}.parquet", model_name="same-alias",
            base_url=endpoint, verify_ionos=verify, dry_run=True,
        )[1])
    assert len({m["experiment_id"] for m in manifests[:3]}) == 3
    assert manifests[0] == manifests[3]
    assert manifests[0]["endpoint"] == "https://proxy-a.invalid/v1"
    assert manifests[2]["verification_requested"] is False
    assert "provider_verified" not in manifests[0]  # Preparation cannot claim execution results.
    for m in manifests:
        assert "password" not in json.dumps(m) and "private" not in json.dumps(m)


@pytest.mark.parametrize("verify", [True, False])
def test_cli_persists_sanitized_actual_route_before_inference(tmp_path, monkeypatch, verify):
    path = tmp_path / "source.parquet"
    source().iloc[:1].to_parquet(path)
    output = tmp_path / "run.parquet"
    record_path = output.with_suffix(".checkpoint.json")
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    route_calls = []

    def fake_verify(*args):
        route_calls.append(args)
        return ModelRouteInfo("mock", "openai/model-id", "https://openai.inference.de-txl.ionos.com/v1", True)

    async def fake_call(self, **kwargs):
        record = json.loads(record_path.read_text())["payload"]["sessions"][-1]
        assert record["provider_verified"] is verify
        return dict(scope_type="whole_contribution_approval", comment_evidence=["Comment"])

    monkeypatch.setattr(benchmark, "verify_ionos_only", fake_verify)
    monkeypatch.setattr(benchmark.LLMHandler, "call_model", fake_call)
    cli.main(["benchmark", "--source", str(path), "--output", str(output),
              "--reference", str(tmp_path / "absent.parquet"), "--folds", "", "--model-name", "mock",
              "--base-url", "https://proxy.invalid/v1",
              "--api-key", "synthetic-api-key"] + ([] if verify else ["--no-verify-ionos"]))
    record = json.loads(record_path.read_text())["payload"]["sessions"][-1]
    manifest = json.loads(output.with_suffix(".manifest.json").read_text())
    assert record["run_id"] == manifest["run_id"]
    assert record["endpoint"] == manifest["endpoint"] == "https://proxy.invalid/v1"
    assert record["verification_status"] == ("verified" if verify else "not_requested")
    assert record["verification_requested"] is verify
    assert len(route_calls) == int(verify)
    assert record["route"] == ({"upstream_model": "openai/model-id",
                                 "api_base": "https://openai.inference.de-txl.ionos.com/v1",
                                 "is_ionos": True} if verify else None)
    serialized = record_path.read_text() + output.with_suffix(".manifest.json").read_text()
    assert all(token not in serialized for token in ("password", "hidden", "fragment", "synthetic-api-key"))


def test_verification_failure_records_only_safe_status_and_never_infers(tmp_path, monkeypatch):
    record_path = tmp_path / "failed.execution.json"

    def fake_verify(*args):
        raise IonosVerificationError("private error with token=hidden and password")

    monkeypatch.setattr(benchmark, "verify_ionos_only", fake_verify)
    monkeypatch.setattr(benchmark.LLMHandler, "call_model", lambda **kw: pytest.fail("no inference"))
    with pytest.raises(IonosVerificationError):
        asyncio.run(benchmark.run_benchmark(source=source(), base_url="https://proxy.invalid/v1",
                    model_name="mock", execution_record_path=record_path))
    record = json.loads(record_path.read_text())
    assert record["verification_status"] == "failed"
    assert record["provider_verified"] is False and record["route"] is None
    assert "hidden" not in record_path.read_text() and "password" not in record_path.read_text()
    with pytest.raises(FileExistsError):
        asyncio.run(benchmark.run_benchmark(source=source(), base_url="https://proxy.invalid/v1",
                    model_name="mock", execution_record_path=record_path))
