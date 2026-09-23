"""Command-line entry points for the stance policy v2 pipeline.

Separate from main.py's interactive menu (which keeps driving the existing
fetch/sample/extract/judge/whole-process flows unchanged). Subcommands:
`audit` (Phase 1), `benchmark`/`evaluate` (Phase 4); `export-review` lands
in a later phase.
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
from benchmark import run_benchmark
from dataset_builder import build_dataset, write_dataset
from fold_builder import load_folds

load_dotenv()

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


def _asdict_class_metrics(cm: evaluator_v2.ClassMetrics) -> dict:
    return {
        "label": cm.label,
        "eligible": cm.eligible,
        "accepted": cm.accepted,
        "accepted_correct": cm.accepted_correct,
        "accepted_accuracy": cm.accepted_accuracy,
        "coverage": cm.coverage,
        "insufficient_sample": cm.insufficient_sample,
    }


def _metrics_to_json_dict(metrics: evaluator_v2.EvaluationMetricsV2) -> dict:
    return {
        k: ({ck: _asdict_class_metrics(cv) for ck, cv in v.items()} if k == "per_class" else v)
        for k, v in vars(metrics).items()
    }


def _run_evaluate(args: argparse.Namespace) -> None:
    """Score a predictions file against the reference dataset."""
    predictions = pd.read_parquet(args.predictions)
    reference = pd.read_parquet(args.reference)

    label, baseline_acc = evaluator_v2.majority_class_baseline(reference)
    console.print(f"[dim]Majority-class baseline: {label!r} @ {baseline_acc:.3f}[/dim]")

    folds_path = Path(args.folds) if args.folds else None
    if folds_path and folds_path.exists():
        fold_assignment = load_folds(folds_path)
        by_fold = evaluator_v2.evaluate_by_fold(predictions, reference, fold_assignment)
        for entry in by_fold["per_fold"]:
            _print_metrics(f"Fold {entry['fold']}", entry["metrics"])
        _print_metrics("Pooled out-of-fold", by_fold["pooled"])
        result_for_output = by_fold["pooled"]
    else:
        if args.folds:
            console.print(f"[yellow]Folds file not found at {folds_path}; scoring without fold grouping.[/yellow]")
        result_for_output = evaluator_v2.evaluate(predictions, reference)
        _print_metrics("Overall", result_for_output)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "majority_class_baseline": {"label": label, "accuracy": baseline_acc},
                    "metrics": _metrics_to_json_dict(result_for_output),
                },
                f,
                indent=2,
                default=str,
            )
        console.print(f"[green]Wrote metrics to {output_path}[/green]")


def _run_benchmark(args: argparse.Namespace) -> None:
    """Run the v2 engine over a frozen-extraction source and score it."""
    source = pd.read_parquet(args.source)
    if args.limit:
        console.print(f"[cyan]Limiting to the first {args.limit} comments.[/cyan]")

    base_url = args.base_url or os.getenv("STANCE_LLM_BASE_URL")
    model_name = args.model_name or os.getenv("STANCE_LLM_MODEL_NAME")
    api_key = args.api_key or os.getenv("STANCE_LLM_API_KEY", "")

    predictions = asyncio.run(
        run_benchmark(
            source=source,
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            enable_thinking=args.enable_thinking,
            verifier_enabled=args.verifier,
            verify_ionos=not args.no_verify_ionos,
            limit=args.limit,
        )
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_path)
    console.print(f"[green]Wrote {len(predictions)} predictions to {output_path}[/green]")

    reference_path = Path(args.reference)
    if reference_path.exists():
        reference = pd.read_parquet(reference_path)
        folds_path = Path(args.folds) if args.folds else None
        if folds_path and folds_path.exists():
            fold_assignment = load_folds(folds_path)
            by_fold = evaluator_v2.evaluate_by_fold(predictions, reference, fold_assignment)
            _print_metrics("Pooled out-of-fold", by_fold["pooled"])
        else:
            _print_metrics("Overall", evaluator_v2.evaluate(predictions, reference))
    else:
        console.print(f"[yellow]Reference file not found at {reference_path}; skipping scoring.[/yellow]")


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
    benchmark_parser.set_defaults(func=_run_benchmark)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
