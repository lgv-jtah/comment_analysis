"""Runs the policy v2 engine over an already-extracted source and scores it.

Deliberately does NOT re-run key-message extraction.
STANCE_PIPELINE_IMPLEMENTATION_BRIEF.md section 7: "Keep extraction fixed
during stance experiments." This session's own empirical check confirmed
why: re-extracting produces different key-message text (an LLM call is not
byte-reproducible), which changes dataset_builder.compute_pair_id's hash and
makes fresh predictions unmatchable against the historical reference dataset
built from the original extraction -- a live check found 23 of 28 fresh
predictions from WholeProcess.run_v2() (which re-extracts) had no match in
dataset/pairs.parquet at all.

So this module reads key messages (and their related_sentences) straight
from an already-extracted source (the same row shape
whole_process.OUTPUT_COLUMNS describes: one row per (key_message, comment)
pair, carrying contributionContent/key_message/related_sentences/
comment_text), groups it back into one entry per comment, builds one
CommentContext per comment from that frozen data, and calls
StancePolicyEngineV2.resolve_comment() directly -- never
WholeProcess.run_v2(), which always re-extracts.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import subprocess
import time
import uuid
import platform
from importlib.metadata import distributions, version
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
import pandas as pd
import pydantic
import pydantic_core

from context_builder import ParentContext, build_comment_context
from dataset_builder import compute_pair_id
from ionos_config import verify_ionos_only
from llm_handler import LLMHandler, MAX_TOKENS, schema_response_format
from policy import OneCallPolicyEngineV2, StancePolicyEngineV2
from policy_schemas import ScopeAttributionResult, PerTargetStanceResult, StanceV2OneCallResult
from revisable_attribution import (
    GlobalHypothesis, RevisedAttributionResult, RevisableAttributionEngine, PROMPT_CONFIG,
    VARIANT_VERSION, validate_revisable_variant, variant_prompt_config, required_targets_schema,
    evidence_id_schemas, build_evidence_table, EVIDENCE_CONTRACT_VERSION, SEGMENTATION_VERSION, OFFSET_CONVENTION,
)
from whole_process import OUTPUT_COLUMNS_V2, pair_output_fields

Mode = Literal["two_stage", "one_call", "revisable_attribution"]


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def revisable_context_metadata(context, variant):
    ids = [km.key_message_id for km in context.key_messages]
    pairs = [compute_pair_id(context.contribution_id, km.text, context.comment_text)
             for km in context.key_messages]
    schema = required_targets_schema(ids) if variant == "stage2_required_targets" else RevisedAttributionResult
    response_format = schema_response_format(schema)
    metadata = {
        "anchor": min(pairs), "contribution_id": context.contribution_id,
        "pair_ids": pairs, "target_ids": ids,
        "context_sha256": json_sha256(context.model_dump(exclude={"run_id"})),
        "stage2_response_format": {
            "name": schema.__name__, "sha256": json_sha256(response_format),
            "schema_sha256": json_sha256(response_format["json_schema"]["schema"]),
        },
    }
    if variant == "source_bound_evidence_ids":
        table = build_evidence_table(context)
        metadata["evidence_table"] = {
            "sha256": json_sha256(table), "count": len(table),
            "bytes": len(json.dumps(table, ensure_ascii=False).encode()),
            "contract_version": EVIDENCE_CONTRACT_VERSION, "segmentation_version": SEGMENTATION_VERSION,
            "offset_convention": OFFSET_CONVENTION, "strict_duplicate_json_parser": True,
            "strict_json": True,
        }
        for stage, wire in zip(("stage1", "stage2"), evidence_id_schemas(context)):
            response_format = schema_response_format(wire)
            metadata[stage + "_response_format"] = {
                "name": wire.__name__, "sha256": json_sha256(response_format),
                "schema_sha256": json_sha256(response_format["json_schema"]["schema"]),
                "schema": wire.model_json_schema(), "response_format": response_format,
                "bytes": len(json.dumps(response_format, ensure_ascii=False).encode()),
            }
    return metadata


def sanitized_endpoint(base_url: str | None) -> str | None:
    """Retain endpoint identity, excluding authentication and query/fragment data."""
    if not base_url:
        return None
    try:
        url = urlsplit(base_url)
        if url.scheme not in ("http", "https") or not url.hostname:
            return None
        host = f"[{url.hostname}]" if ":" in url.hostname else url.hostname
        port = url.port
        if port is not None and (url.scheme, port) not in (("http", 80), ("https", 443)):
            host += f":{port}"
        return urlunsplit((url.scheme, host, url.path.rstrip("/"), "", ""))
    except ValueError:
        return None


def load_frozen_comments(source: pd.DataFrame) -> list[dict]:
    """Group a whole_process-shaped source into one entry per comment.

    Args:
        source: A DataFrame shaped like whole_process.OUTPUT_COLUMNS: one
            row per (key_message, comment) pair, with contribution_id,
            contributionContent, key_message, key_message_type,
            related_sentences, and comment_text columns.

    Returns:
        One dict per (contribution_id, comment_text), each with
        contribution_id (str), contribution_text (str or None),
        comment_text, and key_messages (a list of {key_message,
        key_message_type, related_sentences} dicts, de-duplicated by
        key_message text, in first-seen order).
    """
    comments = []
    if source[["contribution_id", "comment_text", "key_message"]].isna().any().any():
        raise ValueError("Frozen source requires non-null contribution, comment and target")
    if "project_id" in source and source.groupby("contribution_id").project_id.nunique().gt(1).any():
        raise ValueError("Legacy pair identity cannot represent cross-project contribution collisions")
    for (contribution_id, comment_text), group in source.groupby(
        ["contribution_id", "comment_text"], sort=False
    ):
        first = group.iloc[0]
        for column in ("contributionContent", "parent_context"):
            if column in group and group[column].map(lambda value: json.dumps(value, sort_keys=True)).nunique() > 1:
                raise ValueError(f"Conflicting frozen comment context: {column}")
        contribution_text = first.get("contributionContent")
        key_messages = []
        seen_km_texts: dict[str, dict] = {}
        for _, row in group.iterrows():
            km_text = row["key_message"]
            related = row.get("related_sentences")
            if isinstance(related, str):
                raise ValueError("related_sentences must be a list, not a string")
            if related is None or isinstance(related, float) and pd.isna(related):
                related = []
            km = {
                "key_message": km_text,
                "key_message_type": row.get("key_message_type", ""),
                "related_sentences": list(related),
            }
            if km_text in seen_km_texts:
                if seen_km_texts[km_text] != km:
                    raise ValueError("Conflicting frozen target context")
                continue
            seen_km_texts[km_text] = km
            key_messages.append(km)
        comments.append(
            {
                "contribution_id": str(contribution_id),
                "contribution_text": (
                    str(contribution_text) if pd.notna(contribution_text) else None
                ),
                "comment_text": comment_text,
                "key_messages": key_messages,
                "parent_context": (
                    ParentContext.model_validate(first["parent_context"])
                    if isinstance(first.get("parent_context"), dict) else None
                ),
            }
        )
    return comments


def expected_cohort(source: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    comments = load_frozen_comments(source)
    if limit is not None:
        comments = comments[:limit]
    return pd.DataFrame([
        {"pair_id": compute_pair_id(entry["contribution_id"], km["key_message"], entry["comment_text"]),
         "contribution_id": entry["contribution_id"]}
        for entry in comments for km in entry["key_messages"]
    ], columns=["pair_id", "contribution_id"]).drop_duplicates("pair_id")


def prepare_benchmark(*, source, source_path, reference_path, folds_path, output_path,
                      model_name, mode="two_stage", limit=None, enable_thinking=False,
                      verifier_enabled=False, role="primary", tiers=("A", "B"),
                      dry_run=False, base_url=None, verify_ionos=True, revisable_variant="baseline",
                      persist=True, run_id=None, durable=False, execution_settings=None, budget=None):
    """Persist an offline, immutable run specification before any inference."""
    if mode not in ("two_stage", "one_call", "revisable_attribution"):
        raise ValueError(f"Unknown mode {mode!r}")
    validate_revisable_variant(revisable_variant, mode, verifier_enabled)
    output = Path(output_path)
    cohort_path = output.with_suffix(".expected.parquet")
    manifest_path = output.with_suffix(".manifest.json")
    paths = [output, cohort_path, manifest_path, output.with_suffix(".execution.json")]
    if persist and (len(set(p.resolve() for p in paths)) != 4 or any(p.exists() for p in paths)):
        raise FileExistsError("Use a new output basename; existing run artifacts are immutable")
    expected = expected_cohort(source, limit)
    root = Path(__file__).parent
    config_path = Path(variant_prompt_config(revisable_variant)) if mode == "revisable_attribution" else root / "prompts/config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    stages = {
        "two_stage": [("scope_attribution", ScopeAttributionResult), ("per_target_stance", PerTargetStanceResult)],
        "one_call": [("stance_v2_one_call", StanceV2OneCallResult)],
        "revisable_attribution": [("global_hypothesis", GlobalHypothesis), ("revised_attribution", RevisedAttributionResult)],
    }[mode]
    prompt_metadata = []
    for key, schema in stages:
        entry = config[key]
        prompt_metadata.append({
            "stage": key, "schema_version": entry["schema_version"],
            "schema_sha256": hashlib.sha256(json.dumps(schema.model_json_schema(), sort_keys=True).encode()).hexdigest(),
            "prompt_hashes": {name: hashlib.sha256((config_path.parent / entry[name]).read_bytes()).hexdigest()
                              for name in ("system_prompt", "user_prompt")},
        })
    hashes = {}
    for key, path in (("source", source_path), ("reference", reference_path), ("folds", folds_path)):
        path = Path(path) if path else None
        hashes[key] = hashlib.sha256(path.read_bytes()).hexdigest() if path and path.exists() else None
    code_state = {}
    try:
        code_state["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True).strip()
        # Hash diffs without exposing their contents (which could contain private data).
        code_state["diff_sha256"] = hashlib.sha256(subprocess.check_output(
            ["git", "diff", "HEAD", "--", "*.py", "prompts/"], cwd=root)).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        code_state["commit"] = None
    code_state["implementation_hashes"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ("benchmark.py", "cli.py", "policy.py", "revisable_attribution.py", "policy_schemas.py",
                     "context_builder.py", "llm_handler.py", "evaluator_v2.py", "ionos_config.py")
    }
    manifest = {
        "manifest_version": "benchmark_v1", "policy_version": "v2", "engine": mode,
        "model_name": model_name, "provider_requirement": "ionos_only",
        "endpoint": sanitized_endpoint(base_url), "verification_requested": verify_ionos,
        "preparation_only": dry_run,
        "expected_pairs": expected.to_dict("records"),
        "expected_contribution_ids": sorted(expected.contribution_id.unique().tolist()),
        "selection": {"limit_comments": limit, "role": role, "tiers": list(tiers)},
        "input_hashes": hashes, "stages": prompt_metadata, "code_state": code_state,
        "decoding": {"temperature": 0.1, "max_tokens": MAX_TOKENS,
                     "enable_thinking": enable_thinking, "response_format": "json_schema"},
        "verifier_enabled": verifier_enabled, "concurrency": 1,
        "cache_enabled": False, "resumability_enabled": False,
    }
    if durable:
        from benchmark_checkpoint import checked_endpoint, DEFAULT_EXECUTION_SETTINGS, validate_budget
        checked_endpoint(base_url)
        manifest.update(manifest_version="benchmark_checkpoint_v1", resumability_enabled=True)
        manifest["execution_settings"] = execution_settings or DEFAULT_EXECUTION_SETTINGS
        if budget is not None:
            # Part of the fingerprinted contract, so a resume cannot change the limits.
            manifest["budget"] = validate_budget(budget)
        manifest["decoding"]["reasoning_mode"] = "enabled" if enable_thinking else "backend_default"
        manifest["decoding"]["thinking_override"] = True if enable_thinking else None
        manifest["output_columns"] = OUTPUT_COLUMNS_V2
        manifest["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
        manifest["runtime"] = {name: version(name) for name in
                               ("pandas", "pyarrow", "httpx", "pydantic", "pydantic_core")}
        manifest["runtime"]["python"] = platform.python_version()
        manifest["runtime"]["dependencies"] = {dist.metadata["Name"]: dist.version
            for dist in distributions() if dist.metadata["Name"]}
        manifest["prompt_files"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted((root / "prompts").rglob("*")) if p.is_file()}
        # Hash implementation bytes, not unrelated worktree edits or git bookkeeping.
        manifest["code_state"] = {"implementation_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))
        }, "dependency_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                  for name in ("uv.lock", "pyproject.toml")}}
        code_state = manifest["code_state"]
        manifest["ordered_comments"] = []
        for entry in load_frozen_comments(source)[:limit]:
            context = build_comment_context(run_id="contract", **entry)
            manifest["ordered_comments"].append({
                "context_sha256": json_sha256(context.model_dump(exclude={"run_id"})),
                "pair_ids": [compute_pair_id(context.contribution_id, km.text, context.comment_text)
                             for km in context.key_messages],
                "target_ids": [km.key_message_id for km in context.key_messages],
            })
        all_pairs = [p for c in manifest["ordered_comments"] for p in c["pair_ids"]]
        if len(all_pairs) != len(set(all_pairs)) or all_pairs != expected.pair_id.tolist():
            raise ValueError("Duplicate or inconsistent expected identities")
    if mode == "revisable_attribution":
        contexts = [revisable_context_metadata(build_comment_context(run_id="preparation", **entry), revisable_variant)
                    for entry in load_frozen_comments(source)[:limit]]
        manifest["revisable_diagnostic"] = {
            "variant": revisable_variant, "version": VARIANT_VERSION,
            "dependencies": {"pydantic": pydantic.__version__, "pydantic_core": pydantic_core.__version__},
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "canonical_schema_sha256": json_sha256(RevisedAttributionResult.model_json_schema()),
            "stage1_response_format_sha256": json_sha256(schema_response_format(GlobalHypothesis)),
            "contexts": contexts, "contexts_sha256": json_sha256(contexts),
        }
        if revisable_variant == "source_bound_evidence_ids":
            diagnostic = manifest["revisable_diagnostic"]
            diagnostic["dependency_file_hashes"] = {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ("uv.lock", "pyproject.toml")
            }
            code_state["implementation_hashes"].update({
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in ("stance_detector.py", "dataset_builder.py", "model_capabilities.py")
            })
            # Actual context-bound wire formats live with each context, not here.
            diagnostic.pop("stage1_response_format_sha256")
            diagnostic["canonical_schemas"] = {
                "stage1": GlobalHypothesis.model_json_schema(),
                "stage2": RevisedAttributionResult.model_json_schema(),
            }
            diagnostic["source_contexts_sha256"] = json_sha256([c["context_sha256"] for c in contexts])
            diagnostic["strict_duplicate_json_parser"] = True
            diagnostic["strict_json"] = True
            for stage in manifest["stages"]:
                stage["canonical_schema_sha256"] = stage.pop("schema_sha256")
                stage["wire_response_format_location"] = "revisable_diagnostic.contexts"
    manifest["experiment_id"] = "benchmark_" + hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:20]
    manifest["run_id"] = run_id or (manifest["experiment_id"] if dry_run else f"benchmark_{uuid.uuid4().hex}")
    if not persist:
        return expected, manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    with cohort_path.open("xb") as handle:
        expected.to_parquet(handle, index=False)
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return expected, manifest


async def run_benchmark(
    *,
    source: pd.DataFrame,
    base_url: str,
    model_name: str,
    api_key: str = "",
    enable_thinking: bool = False,
    verifier_enabled: bool = False,
    verify_ionos: bool = True,
    run_id: str | None = None,
    limit: int | None = None,
    mode: Mode = "two_stage",
    execution_record_path: Path | None = None,
    revisable_variant: str = "baseline",
    checkpoint=None,
    llm_handler=None,
    client_factory=None,
    verify_provider=None,
) -> pd.DataFrame:
    """Run the policy v2 engine over a frozen, already-extracted source.

    Args:
        source: whole_process-shaped rows; see load_frozen_comments.
        base_url: LLM endpoint base URL.
        model_name: Model identifier.
        api_key: Bearer token for the endpoint.
        enable_thinking: Forwarded to LLMHandler.call_model.
        verifier_enabled: Forwarded to StancePolicyEngineV2; ignored when
            mode == "one_call" (OneCallPolicyEngineV2 has no verifier stage).
        verify_ionos: When True (default), fail fast via
            ionos_config.verify_ionos_only() before doing any work.
        run_id: Identifier recorded on every output row; auto-generated
            from the current time when None.
        limit: Optional cap on the number of comments processed.
        mode: "two_stage" (StancePolicyEngineV2, the default) or "one_call"
            (OneCallPolicyEngineV2) -- brief section 10's experiment matrix
            compares both against the same frozen source/reference.

    Returns:
        A DataFrame with whole_process.OUTPUT_COLUMNS_V2 columns, one row
        per (comment, key_message) pair.

    Raises:
        ionos_config.IonosVerificationError: If verify_ionos is True and
            model_name isn't IONOS-routed.
        ValueError: If mode is neither "two_stage" nor "one_call".
    """
    validate_revisable_variant(revisable_variant, mode, verifier_enabled)
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative")
    run_id = run_id or (checkpoint.manifest["run_id"] if checkpoint is not None else f"benchmark_{int(time.time())}")
    if checkpoint is not None:
        checkpoint.validate_invocation(source=source, base_url=base_url, model_name=model_name,
            enable_thinking=enable_thinking, verifier_enabled=verifier_enabled, verify_ionos=verify_ionos,
            run_id=run_id, limit=limit, mode=mode, revisable_variant=revisable_variant)
        if checkpoint.complete:
            return checkpoint.frame()
        if llm_handler is None:
            llm_handler = LLMHandler()
            llm_handler.safe_logging = True
    execution = {
        "run_id": run_id, "endpoint": sanitized_endpoint(base_url),
        "verification_requested": verify_ionos, "provider_verified": False,
        "verification_status": "not_requested", "route": None,
    }
    # Reserve the new record before a provider call; never overwrite an old run.
    handle = Path(execution_record_path).open("x", encoding="utf-8") if execution_record_path and checkpoint is None else None
    try:
        if verify_ionos:
            execution["verification_status"] = "failed"
            route = (verify_provider or verify_ionos_only)(model_name, base_url, api_key)
            if not route.is_ionos:
                raise ValueError("Provider route was not verified")
            upstream = route.upstream_model
            if checkpoint is not None:
                from benchmark_checkpoint import checked_endpoint
                checked_endpoint(route.api_base)
            # Only model identifiers, never arbitrary provider response fields or errors.
            safe_model = upstream if isinstance(upstream, str) and re.fullmatch(r"[\w./:-]+", upstream) else None
            execution.update(provider_verified=True, verification_status="verified", route={
                "upstream_model": safe_model, "api_base": sanitized_endpoint(route.api_base),
                "is_ionos": True,
            })
    finally:
        if checkpoint is not None:
            checkpoint.record_session(execution)
        if handle is not None:
            with handle:
                json.dump(execution, handle, indent=2, sort_keys=True)

    if mode == "two_stage":
        engine = StancePolicyEngineV2(
            llm_handler=llm_handler or LLMHandler(),
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            enable_thinking=enable_thinking,
            verifier_enabled=verifier_enabled,
        )
    elif mode == "one_call":
        engine = OneCallPolicyEngineV2(
            llm_handler=llm_handler or LLMHandler(),
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            enable_thinking=enable_thinking,
        )
    elif mode == "revisable_attribution":
        engine = RevisableAttributionEngine(
            llm_handler=llm_handler or LLMHandler(), base_url=base_url, model_name=model_name,
            api_key=api_key, enable_thinking=enable_thinking,
            revisable_variant=revisable_variant,
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'two_stage' or 'one_call'.")

    comments = load_frozen_comments(source)
    if limit is not None:
        comments = comments[:limit]

    semaphore = asyncio.Semaphore(1)
    rows: list[dict] = checkpoint.rows() if checkpoint is not None else []
    start = checkpoint.completed_comments if checkpoint is not None else 0
    if client_factory is None and checkpoint is not None and checkpoint.budget is not None:
        # Every completion attempt through this client is reserved against the budget.
        client_factory = lambda: httpx.AsyncClient(event_hooks=checkpoint.client_hooks())  # noqa: E731
    async with (client_factory or httpx.AsyncClient)() as client:
        for entry in comments[start:]:
            if checkpoint is not None:
                checkpoint.check_budget()
            context = build_comment_context(
                run_id=run_id,
                contribution_id=entry["contribution_id"],
                comment_text=entry["comment_text"],
                key_messages=entry["key_messages"],
                contribution_text=entry["contribution_text"],
                parent_context=entry["parent_context"],
            )
            results = await engine.resolve_comment(client, semaphore, context)
            expected_ids = [compute_pair_id(context.contribution_id, km.text, context.comment_text)
                            for km in context.key_messages]
            if (len(results) != len(context.key_messages) or
                [r.pair_id for r in results] != expected_ids or
                [r.key_message_id for r in results] != [km.key_message_id for km in context.key_messages] or
                any(r.run_id != run_id or r.policy_version != "v2" for r in results)):
                raise ValueError("Result identity or exact ordered target coverage mismatch")
            comment_rows = []
            for pair_output, km_ref in zip(results, context.key_messages, strict=True):
                comment_rows.append(
                    {
                        "contribution_id": entry["contribution_id"],
                        "contributionContent": entry["contribution_text"],
                        "key_message_id": km_ref.key_message_id,
                        "key_message": km_ref.text,
                        "comment_text": entry["comment_text"],
                        **pair_output_fields(pair_output),
                        "extraction_model_name": None,
                        "stance_model_name": model_name,
                    }
                )
            if checkpoint is not None:
                checkpoint.commit_comment(comment_rows)
            rows.extend(comment_rows)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS_V2)
