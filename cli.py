"""Command-line entry points for the stance policy v2 pipeline.

Separate from main.py's interactive menu (which keeps driving the existing
fetch/sample/extract/judge/whole-process flows unchanged). Subcommands:
`audit` (Phase 1), `benchmark`/`evaluate` (Phase 4),
`historical-baseline`/`post-fix-baseline` (saved baselines) and
`general-approval-review` (offline review pack for the open general-approval
question; no inference, never loads .env); `export-review` lands in a later
phase.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

import evaluator_v2
from benchmark import run_benchmark, prepare_benchmark
from benchmark_checkpoint import BenchmarkCheckpoint, BudgetExhausted
from dataset_builder import build_dataset, write_dataset
from fold_builder import load_folds
from general_approval_review import (
    DEFAULT_ANNOTATIONS_DB, DEFAULT_RUNS, DEFAULT_SOURCES, RunSpec, run_general_approval_review,
)
from historical_baseline import build_historical_predictions, build_post_fix_baseline_predictions
from revisable_attribution import REVISABLE_VARIANTS, validate_revisable_variant

console = Console()

DEFAULT_OUTPUT_DIR = Path("dataset")
DEFAULT_REFERENCE_PATH = DEFAULT_OUTPUT_DIR / "pairs.parquet"
DEFAULT_FOLDS_PATH = DEFAULT_OUTPUT_DIR / "folds_v1.json"


def _run_audit(args: argparse.Namespace) -> None:
    """Build the provenance-audited dataset and print a summary."""
    console.print("[bold cyan]Running dataset audit...[/bold cyan]")
    result = build_dataset()

    output_dir = Path(args.output_dir)
    write_dataset(result, output_dir)

    manifest = result.manifest
    cohort_a = manifest["cohort_a"]

    table = Table(title="Cohort A (DIPAS) resolution status")
    table.add_column("Status")
    table.add_column("Count", justify="right")
    for status, count in cohort_a["resolution_status_counts"].items():
        table.add_row(status, str(count))
    console.print(table)

    tier_table = Table(title="Tier / status breakdown")
    tier_table.add_column("Tier or status")
    tier_table.add_column("Count", justify="right")
    for key, count in cohort_a["tier_counts"].items():
        tier_table.add_row(str(key), str(count))
    console.print(tier_table)

    if cohort_a["quarantine_reasons"]:
        q_table = Table(title="Quarantine reasons")
        q_table.add_column("Reason")
        q_table.add_column("Count", justify="right")
        for reason, count in cohort_a["quarantine_reasons"].items():
            q_table.add_row(reason, str(count))
        console.print(q_table)

    if cohort_a["unresolved_reasons"]:
        u_table = Table(title="Unresolved reasons")
        u_table.add_column("Reason")
        u_table.add_column("Count", justify="right")
        for reason, count in cohort_a["unresolved_reasons"].items():
            u_table.add_row(reason, str(count))
        console.print(u_table)

    console.print(
        f"[bold]Cohort A:[/bold] {cohort_a['row_count']} rows "
        f"({cohort_a['primary_rows']} primary / {cohort_a['secondary_rows']} secondary)"
    )
    console.print(
        f"[bold]Cohort B:[/bold] {manifest['cohort_b']['row_count']} rows "
        f"across {len(manifest['cohort_b']['distinct_projects'])} projects "
        f"({manifest['cohort_b']['resolved']} resolved / "
        f"{manifest['cohort_b']['unresolved']} unresolved)"
    )
    console.print(f"[green]Wrote dataset artifacts to {output_dir}/[/green]")


def _print_metrics(title: str, metrics: evaluator_v2.EvaluationMetricsV2) -> None:
    """Render one EvaluationMetricsV2 as a rich panel + per-class table."""
    lines = [
        f"total_pairs={metrics.total_pairs}  eligible_pairs={metrics.eligible_pairs}"
        f"  unmatched_prediction_pairs={metrics.unmatched_prediction_pairs}",
        f"unexpected_prediction_pairs={metrics.unexpected_prediction_pairs}  "
        f"missing_prediction_pairs={metrics.missing_prediction_pairs}  "
        f"conflicting_prediction_pairs={metrics.conflicting_prediction_pairs}",
        f"forced_choice_accuracy={metrics.forced_choice_accuracy:.3f} "
        f"({metrics.forced_choice_correct}/{metrics.eligible_pairs})",
        f"accepted_accuracy={metrics.accepted_accuracy if metrics.accepted_accuracy is None else f'{metrics.accepted_accuracy:.3f}'} "
        f"({metrics.accepted_correct}/{metrics.accepted_pairs} accepted)",
        f"labelled_coverage={metrics.labelled_coverage:.3f}  "
        f"operational_coverage={metrics.operational_coverage:.3f}",
        f"review_rate={metrics.review_rate:.3f} ({metrics.review_pairs})  "
        f"failure_rate={metrics.failure_rate:.3f} ({metrics.failed_pairs})",
    ]
    if metrics.insufficient_sample:
        lines.append(f"[yellow]{metrics.insufficient_sample_reason}[/yellow]")
    console.print(f"[bold]{title}[/bold]")
    for line in lines:
        console.print(f"  {line}")

    if metrics.per_class:
        table = Table(title=f"{title}: per-class")
        table.add_column("label")
        table.add_column("eligible", justify="right")
        table.add_column("accepted", justify="right")
        table.add_column("accepted_accuracy", justify="right")
        table.add_column("coverage", justify="right")
        for label, cm in sorted(metrics.per_class.items()):
            acc = "-" if cm.accepted_accuracy is None else f"{cm.accepted_accuracy:.3f}"
            flag = " ⚠" if cm.insufficient_sample else ""
            table.add_row(label, str(cm.eligible), str(cm.accepted), acc, f"{cm.coverage:.3f}{flag}")
        console.print(table)


def _print_basis_breakdown(by_basis: dict[str, evaluator_v2.BasisMetrics]) -> None:
    """Print accepted accuracy split by `basis` (brief section 3).

    Historical reference labels predate the general-approval-propagation
    policy, so basis="general_approval" predictions routinely disagree
    with them for reasons unrelated to prediction quality -- see
    docs/stance_policy_v2.md's "Reporting" section and
    docs/phase5_screening_results.md. Printed after every metrics report
    so this split is never missed rather than requiring ad-hoc analysis.
    """
    if not by_basis:
        return
    table = Table(title="Accepted accuracy by basis (brief section 3: report separately)")
    table.add_column("basis")
    table.add_column("n", justify="right")
    table.add_column("correct", justify="right")
    table.add_column("accuracy", justify="right")
    for basis, bm in sorted(by_basis.items()):
        table.add_row(basis, str(bm.n), str(bm.correct), f"{bm.accuracy:.3f}")
    console.print(table)


def _basis_breakdown_to_json_dict(by_basis: dict[str, evaluator_v2.BasisMetrics]) -> dict:
    return {
        basis: {"n": bm.n, "correct": bm.correct, "accuracy": bm.accuracy} for basis, bm in by_basis.items()
    }


def _asdict_class_metrics(cm: evaluator_v2.ClassMetrics) -> dict:
    return vars(cm)


def _metrics_to_json_dict(metrics: evaluator_v2.EvaluationMetricsV2) -> dict:
    return {
        k: ({ck: _asdict_class_metrics(cv) for ck, cv in v.items()} if k == "per_class" else v)
        for k, v in vars(metrics).items()
    }


def _run_evaluate(args: argparse.Namespace) -> None:
    """Score a predictions file against the reference dataset."""
    predictions = pd.read_parquet(args.predictions)
    reference = pd.read_parquet(args.reference)
    expected = pd.read_parquet(args.expected_cohort) if args.expected_cohort else None
    selection = dict(expected_pairs=expected, role=args.role, tiers=args.tiers.split("+"))
    fold_output = {}

    folds_path = Path(args.folds) if args.folds else None
    if folds_path and folds_path.exists():
        fold_assignment = load_folds(folds_path)
        by_fold = evaluator_v2.evaluate_by_fold(predictions, reference, fold_assignment, **selection)
        for entry in by_fold["per_fold"]:
            _print_metrics(f"Fold {entry['fold']}", entry["metrics"])
        _print_metrics("Pooled out-of-fold", by_fold["pooled"])
        result_for_output = by_fold["overall"]
        _print_metrics("Overall expected cohort", result_for_output)
        fold_output = {
            "pooled_out_of_fold": _metrics_to_json_dict(by_fold["pooled"]),
            "unassigned_expected_pairs": by_fold["unassigned_expected_pairs"],
        }
    else:
        if args.folds:
            console.print(f"[yellow]Folds file not found at {folds_path}; scoring without fold grouping.[/yellow]")
        result_for_output = evaluator_v2.evaluate(predictions, reference, **selection)
        _print_metrics("Overall", result_for_output)

    label, baseline_acc = evaluator_v2.majority_class_baseline(reference, **selection)
    console.print(f"[dim]Majority-class baseline: {label!r} @ {baseline_acc:.3f}[/dim]")
    by_basis = evaluator_v2.accepted_accuracy_by_basis(predictions, reference, **selection)
    _print_basis_breakdown(by_basis)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "x", encoding="utf-8") as f:
            json.dump(
                {
                    "majority_class_baseline": {"label": label, "accuracy": baseline_acc},
                    "metrics": _metrics_to_json_dict(result_for_output),
                    "accepted_accuracy_by_basis": _basis_breakdown_to_json_dict(by_basis),
                    **fold_output,
                },
                f,
                indent=2,
                default=str,
            )
        console.print(f"[green]Wrote metrics to {output_path}[/green]")


def _run_benchmark(args: argparse.Namespace) -> None:
    """Run the v2 engine over a frozen-extraction source and score it."""
    validate_revisable_variant(args.revisable_variant, args.mode, args.verifier)
    if args.dry_run and (args.resume or args.recover_stale_lock or args.evaluate_partial):
        raise ValueError("Dry preparation cannot resume, recover or evaluate partial results")
    if (args.recover_stale_lock or args.evaluate_partial) and not args.resume:
        raise ValueError("Recovery and partial evaluation require --resume")
    source = pd.read_parquet(args.source)
    if args.limit:
        console.print(f"[cyan]Limiting to the first {args.limit} comments.[/cyan]")

    if not args.dry_run:
        load_dotenv()
    model_name = args.model_name or (None if args.dry_run else os.getenv("STANCE_LLM_MODEL_NAME"))
    base_url = args.base_url or (None if args.dry_run else os.getenv("STANCE_LLM_BASE_URL"))
    preparation = dict(
        source=source, source_path=args.source, reference_path=args.reference,
        folds_path=args.folds, model_name=model_name,
        mode=args.mode, limit=args.limit, enable_thinking=args.enable_thinking,
        verifier_enabled=args.verifier, role=args.role, tiers=args.tiers.split("+"),
        base_url=base_url, verify_ionos=not args.no_verify_ionos,
        revisable_variant=args.revisable_variant,
    )
    limits = {"max_wall_seconds": args.max_wall_seconds, "max_total_tokens": args.max_total_tokens,
              "max_http_attempts": args.max_http_attempts}
    if any(value is not None for value in limits.values()):
        if args.dry_run:
            raise ValueError("Run budgets apply to durable live runs, not dry preparation")
        preparation["budget"] = limits
    if args.dry_run:
        expected, manifest = prepare_benchmark(**preparation, output_path=args.output, dry_run=True)
        console.print(f"Prepared {len(expected)} expected pairs; run_id={manifest['run_id']}")
        console.print("Offline preparation only: no provider checks or inference performed.")
        return
    api_key = args.api_key or os.getenv("STANCE_LLM_API_KEY", "")

    with BenchmarkCheckpoint(output_path=args.output, resume=args.resume,
                             recover_stale_lock=args.recover_stale_lock, **preparation) as checkpoint:
        expected, manifest = checkpoint.expected, checkpoint.manifest
        console.print(f"Prepared {len(expected)} expected pairs; run_id={manifest['run_id']}")
        if args.evaluate_partial:
            predictions = checkpoint.export_partial()
            console.print("Explicit partial evaluation against the full original expected cohort.")
        else:
            try:
                asyncio.run(run_benchmark(
                    source=source,
                    base_url=base_url,
                    model_name=model_name,
                    api_key=api_key,
                    enable_thinking=args.enable_thinking,
                    verifier_enabled=args.verifier,
                    mode=args.mode,
                    revisable_variant=args.revisable_variant,
                    verify_ionos=not args.no_verify_ionos,
                    limit=args.limit,
                    run_id=manifest["run_id"],
                    checkpoint=checkpoint,
                ))
            except BudgetExhausted as stop:
                predictions = checkpoint.export_partial()
                console.print(
                    f"[yellow]Run budget exhausted ({stop.reason}) after {checkpoint.completed_comments} "
                    f"comments; partial results saved and scored against the full expected cohort. "
                    f"The budget is persisted, so resuming will not continue this run.[/yellow]"
                )
            else:
                predictions = checkpoint.publish()
                console.print(f"[green]Wrote {len(predictions)} predictions to {args.output}[/green]")

    reference_path = Path(args.reference)
    if reference_path.exists():
        reference = pd.read_parquet(reference_path)
        selection = dict(expected_pairs=expected, role=args.role, tiers=args.tiers.split("+"))
        folds_path = Path(args.folds) if args.folds else None
        if folds_path and folds_path.exists():
            fold_assignment = load_folds(folds_path)
            by_fold = evaluator_v2.evaluate_by_fold(predictions, reference, fold_assignment, **selection)
            _print_metrics("Pooled out-of-fold", by_fold["pooled"])
            _print_metrics("Overall expected cohort", by_fold["overall"])
        else:
            _print_metrics("Overall", evaluator_v2.evaluate(predictions, reference, **selection))
        _print_basis_breakdown(evaluator_v2.accepted_accuracy_by_basis(predictions, reference, **selection))
    else:
        console.print(f"[yellow]Reference file not found at {reference_path}; skipping scoring.[/yellow]")


def _run_historical_baseline(args: argparse.Namespace) -> None:
    """Score the historical saved baseline (brief section 10, step 1)."""
    reference = pd.read_parquet(args.reference)
    predictions = build_historical_predictions(reference)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path)
    console.print(f"[green]Wrote {len(predictions)} historical-baseline predictions to {output_path}[/green]")

    folds_path = Path(args.folds) if args.folds else None
    if folds_path and folds_path.exists():
        fold_assignment = load_folds(folds_path)
        by_fold = evaluator_v2.evaluate_by_fold(predictions, reference, fold_assignment)
        for entry in by_fold["per_fold"]:
            _print_metrics(f"Fold {entry['fold']}", entry["metrics"])
        _print_metrics("Pooled out-of-fold", by_fold["pooled"])
        _print_metrics("Overall expected cohort", by_fold["overall"])
    else:
        _print_metrics("Overall", evaluator_v2.evaluate(predictions, reference))


def _run_post_fix_baseline(args: argparse.Namespace) -> None:
    """Score the validated Sept 8, 2026 fix's own predictions.

    More recent than the historical-baseline subcommand's
    established_original_label (the pipeline's state *before* that fix) --
    see historical_baseline.build_post_fix_baseline_predictions's docstring.
    """
    reference = pd.read_parquet(args.reference)
    predictions = build_post_fix_baseline_predictions(reference)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path)
    console.print(
        f"[green]Wrote {len(predictions)} post-fix-baseline predictions to {output_path}[/green] "
        f"[dim]({len(reference)} annotation rows total; missing labels are failed predictions)[/dim]"
    )

    folds_path = Path(args.folds) if args.folds else None
    if folds_path and folds_path.exists():
        fold_assignment = load_folds(folds_path)
        by_fold = evaluator_v2.evaluate_by_fold(predictions, reference, fold_assignment)
        for entry in by_fold["per_fold"]:
            _print_metrics(f"Fold {entry['fold']}", entry["metrics"])
        _print_metrics("Pooled out-of-fold", by_fold["pooled"])
        _print_metrics("Overall expected cohort", by_fold["overall"])
    else:
        _print_metrics("Overall", evaluator_v2.evaluate(predictions, reference))


def _run_general_approval_review(args: argparse.Namespace) -> None:
    """Build the offline general-approval review pack; never loads .env, no inference.

    Writes the artifacts via general_approval_review.run_general_approval_review
    and prints the scored general-approval counts per total, run and fold,
    the annotator-note counts, the pack size and the written paths.

    Args:
        args: Parsed `general-approval-review` arguments: run (list of
            [NAME, PREDICTIONS, EXPECTED] or None for DEFAULT_RUNS), reference,
            folds, annotations, source (list or None for DEFAULT_SOURCES),
            role, tiers ("A", "A+B" or "B") and output_dir (or None).

    Raises:
        FileNotFoundError: If an input file or the output directory's parent
            is missing.
        FileExistsError: If the output directory already exists.
        RuntimeError: If an input file changed during the run.
        ValueError: If the inputs are inconsistent.
    """
    runs = [RunSpec(name, Path(pred), Path(exp)) for name, pred, exp in args.run] if args.run else DEFAULT_RUNS
    result = run_general_approval_review(
        runs=runs,
        reference_path=Path(args.reference),
        folds_path=Path(args.folds),
        annotations_path=Path(args.annotations),
        source_paths=[Path(p) for p in args.source] if args.source else DEFAULT_SOURCES,
        role=args.role,
        # "A+B" -> ["A", "B"]
        tiers=args.tiers.split("+"),
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    summary = result.summary
    scored = summary["scored_general_approval"]
    table = Table(title="Accepted general_approval pairs with a resolved reference")
    for column in ("group", "pairs", "correct", "neutral", "in_favor", "against", "comments"):
        table.add_column(column, justify="left" if column == "group" else "right")
    groups = [("total", scored["total"])]
    groups += [(f"run {name}", block) for name, block in scored["by_run"].items()]
    groups += [(f"fold {fold}", block) for fold, block in scored["by_fold"].items()]
    for name, block in groups:
        labels = block["reference_labels"]
        table.add_row(name, str(block["pairs"]), str(block["correct"]), str(labels.get("neutral", 0)),
                      str(labels.get("in_favor", 0)), str(labels.get("against", 0)), str(block["comments"]))
    console.print(table)
    notes, pack = summary["annotator_notes"], summary["pack"]
    console.print(f"Annotator general-approval notes: {notes['general_approval_broad']['pairs']} pairs (broad), "
                  f"{notes['general_approval_strict']['pairs']} (strict)")
    console.print(f"Review pack: {pack['comments']} comments ({pack['comments_outside_runs']} outside runs), "
                  f"{pack['key_message_rows']} key-message rows")
    console.print("[dim]Offline analysis only: no inference, no network, .env not loaded.[/dim]")
    for path in result.files.values():
        console.print(f"[green]Wrote {path}[/green]")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser(
        "audit",
        help="Build the provenance-audited dataset from existing annotation files.",
    )
    audit_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to write manifest/pairs/quarantine files to (default: {DEFAULT_OUTPUT_DIR})",
    )
    audit_parser.set_defaults(func=_run_audit)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Score an existing predictions file against the reference dataset.",
    )
    evaluate_parser.add_argument(
        "--predictions", required=True, help="Path to a predictions parquet (OUTPUT_COLUMNS_V2 shape)."
    )
    evaluate_parser.add_argument(
        "--reference",
        default=str(DEFAULT_REFERENCE_PATH),
        help=f"Path to the reference dataset (default: {DEFAULT_REFERENCE_PATH}).",
    )
    evaluate_parser.add_argument(
        "--folds",
        default=str(DEFAULT_FOLDS_PATH),
        help=f"Path to a persisted fold assignment (default: {DEFAULT_FOLDS_PATH}); "
        "scored without fold grouping if not found.",
    )
    evaluate_parser.add_argument("--output", default=None, help="Optional path to write metrics as JSON.")
    evaluate_parser.add_argument("--expected-cohort", help="Prepared expected-pair parquet; defaults to selected reference.")
    evaluate_parser.add_argument("--role", choices=["primary", "secondary", "all"], default="primary")
    evaluate_parser.add_argument("--tiers", choices=["A", "A+B", "B"], default="A+B")
    evaluate_parser.set_defaults(func=_run_evaluate)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run the policy v2 engine over a frozen-extraction source and score it.",
    )
    benchmark_parser.add_argument(
        "--source",
        required=True,
        help="Path to a whole_process-shaped parquet (already-extracted key messages; "
        "extraction is never re-run -- see benchmark.py's module docstring).",
    )
    benchmark_parser.add_argument("--output", required=True, help="Path to write the predictions parquet.")
    benchmark_parser.add_argument(
        "--reference",
        default=str(DEFAULT_REFERENCE_PATH),
        help=f"Path to the reference dataset for scoring (default: {DEFAULT_REFERENCE_PATH}).",
    )
    benchmark_parser.add_argument(
        "--folds",
        default=str(DEFAULT_FOLDS_PATH),
        help=f"Path to a persisted fold assignment (default: {DEFAULT_FOLDS_PATH}).",
    )
    benchmark_parser.add_argument("--base-url", default=None, help="Overrides STANCE_LLM_BASE_URL.")
    benchmark_parser.add_argument("--model-name", default=None, help="Overrides STANCE_LLM_MODEL_NAME.")
    benchmark_parser.add_argument("--api-key", default=None, help="Overrides STANCE_LLM_API_KEY.")
    benchmark_parser.add_argument("--enable-thinking", action="store_true", help="Request thinking mode.")
    benchmark_parser.add_argument(
        "--verifier", action="store_true", help="Enable StancePolicyEngineV2's independent verifier stage."
    )
    benchmark_parser.add_argument(
        "--no-verify-ionos",
        action="store_true",
        help="Skip the IONOS-routing pre-flight check (for non-IONOS testing only).",
    )
    benchmark_parser.add_argument("--limit", type=int, default=None, help="Cap the number of comments processed.")
    benchmark_parser.add_argument(
        "--mode",
        choices=["two_stage", "one_call", "revisable_attribution"],
        default="two_stage",
        help="Policy engine variant: 'two_stage' (StancePolicyEngineV2, default) or "
        "'one_call' (OneCallPolicyEngineV2) -- brief section 10's experiment matrix.",
    )
    benchmark_parser.set_defaults(func=_run_benchmark)
    benchmark_parser.add_argument("--revisable-variant", choices=REVISABLE_VARIANTS, default="baseline")
    benchmark_parser.add_argument("--dry-run", action="store_true", help="Write expected cohort and manifest offline, without loading credentials.")
    benchmark_parser.add_argument("--resume", action="store_true", help="Resume an existing durable checkpoint, repeating only an interrupted comment.")
    benchmark_parser.add_argument("--recover-stale-lock", action="store_true", help="Acknowledge a stale session after acquiring the process lock; requires --resume.")
    benchmark_parser.add_argument("--evaluate-partial", action="store_true", help="With --resume, export and score saved comments against the full original cohort without inference.")
    benchmark_parser.add_argument("--role", choices=["primary", "secondary", "all"], default="primary")
    benchmark_parser.add_argument("--tiers", choices=["A", "A+B", "B"], default="A+B")
    benchmark_parser.add_argument("--max-wall-seconds", type=int, default=None,
                                  help="Absolute run deadline from the first start; never reset on resume.")
    benchmark_parser.add_argument("--max-total-tokens", type=int, default=None,
                                  help="Soft limit on client-visible total tokens; an in-flight request can overshoot.")
    benchmark_parser.add_argument("--max-http-attempts", type=int, default=None,
                                  help="Hard limit on completion requests, including retries, reserved before sending.")

    historical_parser = subparsers.add_parser(
        "historical-baseline",
        help="Score the historical saved baseline (brief section 10, step 1): established "
        "original labels reshaped into predictions and scored through the same evaluator_v2 path.",
    )
    historical_parser.add_argument("--output", required=True, help="Path to write the predictions parquet.")
    historical_parser.add_argument(
        "--reference",
        default=str(DEFAULT_REFERENCE_PATH),
        help=f"Path to the reference dataset (default: {DEFAULT_REFERENCE_PATH}).",
    )
    historical_parser.add_argument(
        "--folds",
        default=str(DEFAULT_FOLDS_PATH),
        help=f"Path to a persisted fold assignment (default: {DEFAULT_FOLDS_PATH}).",
    )
    historical_parser.set_defaults(func=_run_historical_baseline)

    post_fix_parser = subparsers.add_parser(
        "post-fix-baseline",
        help="Score the validated Sept 8, 2026 fix's own predictions (post_fix_original_label) -- "
        "more recent than historical-baseline's pre-fix established_original_label.",
    )
    post_fix_parser.add_argument("--output", required=True, help="Path to write the predictions parquet.")
    post_fix_parser.add_argument(
        "--reference",
        default=str(DEFAULT_REFERENCE_PATH),
        help=f"Path to the reference dataset (default: {DEFAULT_REFERENCE_PATH}).",
    )
    post_fix_parser.add_argument(
        "--folds",
        default=str(DEFAULT_FOLDS_PATH),
        help=f"Path to a persisted fold assignment (default: {DEFAULT_FOLDS_PATH}).",
    )
    post_fix_parser.set_defaults(func=_run_post_fix_baseline)

    review_parser = subparsers.add_parser(
        "general-approval-review",
        help="Offline review pack for the open general-approval question (diagnostic parquet, safe summary, "
        "German xlsx). No inference, no network, never loads .env.",
    )
    review_parser.add_argument(
        "--run", nargs=3, action="append", metavar=("NAME", "PREDICTIONS", "EXPECTED"),
        help="A saved run (repeatable); defaults to the fold-0 option C prompt-2 run and the held-out folds 1-4 run.",
    )
    review_parser.add_argument("--reference", default=str(DEFAULT_REFERENCE_PATH),
                               help=f"Reference dataset (default: {DEFAULT_REFERENCE_PATH}).")
    review_parser.add_argument("--folds", default=str(DEFAULT_FOLDS_PATH),
                               help=f"Persisted fold assignment (default: {DEFAULT_FOLDS_PATH}).")
    review_parser.add_argument("--annotations", default=str(DEFAULT_ANNOTATIONS_DB),
                               help=f"Annotation database, opened read-only (default: {DEFAULT_ANNOTATIONS_DB}).")
    review_parser.add_argument("--source", action="append",
                               help="Frozen extraction source parquet (repeatable); defaults to the two run sources.")
    review_parser.add_argument("--role", choices=["primary", "secondary", "all"], default="primary")
    review_parser.add_argument("--tiers", choices=["A", "A+B", "B"], default="A+B")
    review_parser.add_argument(
        "--output-dir", default=None,
        help="New directory to create (default: dataset/experiments/general_approval_review_<created_utc>); "
        "must not exist.",
    )
    review_parser.set_defaults(func=_run_general_approval_review)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
