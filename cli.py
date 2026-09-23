"""Command-line entry points for the stance policy v2 pipeline.

Separate from main.py's interactive menu (which keeps driving the existing
fetch/sample/extract/judge/whole-process flows unchanged). Subcommands are
added here one per implementation phase: `audit` (Phase 1) is the first;
`benchmark`/`evaluate`/`export-review` land in later phases.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from dataset_builder import build_dataset, write_dataset

console = Console()

DEFAULT_OUTPUT_DIR = Path("dataset")


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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
