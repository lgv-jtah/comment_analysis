"""Builds a provenance-audited stance dataset from existing annotation files.

Recovers reference labels for the (contribution, key_message, comment) pairs
in evaluation_data/, with strict tiered provenance (see
STANCE_PIPELINE_IMPLEMENTATION_BRIEF.md section 5) rather than trusting any
single field blindly. Source files are read-only inputs and are never
modified; every resolved or quarantined row records exactly which rule
produced it, so the dataset is auditable rather than a black box.

Two cohorts are built and reported separately, never merged:
  - Cohort A ("DIPAS"): evaluation_data/evaluations_backup20260922.sq3's
    stance_evaluations table, joined against the six CSVs that share its
    legacy key shape (contribution_id, key_message, comment_text).
  - Cohort B ("gold_labels_corrected"): gold_labels_corrected.json, a
    separate, multi-project benchmark with no relationship to Cohort A's
    contribution IDs. stance_evaluations has no project_id column, so
    Cohort A is never joined against Cohort B -- that's the only available
    mitigation against a cross-project contribution_id collision.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from policy_schemas import StanceLabelV2, normalize_legacy_label
from tier_b_parser import parse_note

DATASET_SCHEMA_VERSION = "dataset_v1"

DEFAULT_EVAL_DB = Path("evaluation_data/evaluations_backup20260922.sq3")
DEFAULT_CSV_SOURCES: dict[str, Path] = {
    "sept8_fix": Path("evaluation_data/stance_refit_20260908fix.csv"),
    "exp_full_context": Path("evaluation_data/experiments/exp_full_context.csv"),
    "exp_relevance_gate": Path("evaluation_data/experiments/exp_relevance_gate.csv"),
    "exp_full_context_relevance_gate": Path(
        "evaluation_data/experiments/exp_full_context_relevance_gate.csv"
    ),
    "exp_relevance_gate_v2": Path("evaluation_data/experiments/exp_relevance_gate_v2.csv"),
    "exp_relevance_gate_v3": Path("evaluation_data/experiments/exp_relevance_gate_v3.csv"),
}
DEFAULT_GOLD_LABELS_PATH = Path("gold_labels_corrected.json")

PRIMARY_EVALUATOR = "Antonie"


def compute_pair_id(contribution_id: object, key_message: str, comment_text: str) -> str:
    """Deterministic ID for a (contribution, key_message, comment) triple.

    Args:
        contribution_id: The contribution's ID (any stringifiable type).
        key_message: Exact key-message text.
        comment_text: Exact comment text.

    Returns:
        A 16-character hex digest stable across runs and sources.
    """
    key = f"{contribution_id}|{key_message}|{comment_text}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _sha256_file(path: Path) -> str:
    """Hash a file's contents for manifest provenance."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_commit() -> str | None:
    """Best-effort git commit hash for the manifest; None outside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:
        return None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_stance_evaluations(db_path: Path) -> pd.DataFrame:
    """Load the stance_evaluations table and attach a pair_id to every row."""
    conn = sqlite3.connect(str(db_path))
    try:
        df = pd.read_sql_query(
            "select id, contribution_id, key_message, comment_text, verdict, "
            "comment, evaluator, corrected_label from stance_evaluations",
            conn,
        )
    finally:
        conn.close()
    df["pair_id"] = [
        compute_pair_id(r.contribution_id, r.key_message, r.comment_text)
        for r in df.itertuples()
    ]
    return df


def load_csv_predictions(sources: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Load each source CSV and attach a pair_id to every row."""
    frames: dict[str, pd.DataFrame] = {}
    for name, path in sources.items():
        df = pd.read_csv(path)
        df["pair_id"] = [
            compute_pair_id(r.contribution_id, r.key_message, r.comment_text)
            for r in df.itertuples()
        ]
        frames[name] = df
    return frames


# --------------------------------------------------------------------------
# Duplicate handling (within a single CSV source)
# --------------------------------------------------------------------------


@dataclass
class DuplicateCollapseResult:
    """Result of collapsing duplicate legacy keys within one source."""

    collapsed: pd.DataFrame
    identical_groups: list[dict] = field(default_factory=list)
    conflicting_keys: list[dict] = field(default_factory=list)


def collapse_duplicates(df: pd.DataFrame, source_name: str) -> DuplicateCollapseResult:
    """Collapse rows sharing a pair_id when every other field agrees.

    A duplicate key whose rows disagree on any field is NOT arbitrarily
    resolved by picking one -- it is dropped from the collapsed frame and
    reported so the caller can quarantine it.

    Args:
        df: A source DataFrame with a pair_id column.
        source_name: Label for this source, recorded in the report entries.

    Returns:
        The collapsed DataFrame (one row per pair_id) plus manifest-ready
        reports of which groups were identical vs. conflicting.
    """
    identical_groups: list[dict] = []
    conflicting_keys: list[dict] = []
    keep_rows = []
    compare_cols = [c for c in df.columns if c != "pair_id"]

    for pair_id, group in df.groupby("pair_id", sort=False):
        if len(group) == 1:
            keep_rows.append(group.iloc[0])
            continue
        if group[compare_cols].nunique(dropna=False).eq(1).all():
            keep_rows.append(group.iloc[0])
            identical_groups.append(
                {"source": source_name, "pair_id": pair_id, "row_count": len(group)}
            )
        else:
            conflicting_keys.append(
                {"source": source_name, "pair_id": pair_id, "row_count": len(group)}
            )

    collapsed = pd.DataFrame(keep_rows).reset_index(drop=True) if keep_rows else df.iloc[0:0].copy()
    return DuplicateCollapseResult(collapsed, identical_groups, conflicting_keys)


# --------------------------------------------------------------------------
# Establishing "the original prediction the verdict evaluated"
# --------------------------------------------------------------------------


@dataclass
class OriginalPredictionResult:
    """Cross-CSV consistency check for each pair's original ("old_stance") prediction."""

    established: dict[str, StanceLabelV2] = field(default_factory=dict)
    disagreements: list[dict] = field(default_factory=list)


def establish_original_predictions(
    csv_frames: dict[str, pd.DataFrame]
) -> OriginalPredictionResult:
    """Cross-check old_stance across all CSV sources for each pair.

    stance_evaluations has no column recording the prediction a "correct" or
    "incorrect" verdict actually evaluated -- it must be reconstructed from
    the CSVs' old_stance column. Where all sources containing a given pair
    agree on old_stance, that's the established original. Where they
    disagree, provenance is unresolved rather than guessed.
    """
    by_pair: dict[str, dict[str, str]] = {}
    for source_name, df in csv_frames.items():
        for row in df.itertuples():
            by_pair.setdefault(row.pair_id, {})[source_name] = row.old_stance

    established: dict[str, StanceLabelV2] = {}
    disagreements: list[dict] = []
    for pair_id, by_source in by_pair.items():
        normalized = {src: normalize_legacy_label(val) for src, val in by_source.items()}
        distinct = {v for v in normalized.values() if v is not None}
        if len(distinct) == 1 and all(v is not None for v in normalized.values()):
            established[pair_id] = next(iter(distinct))
        else:
            disagreements.append({"pair_id": pair_id, "by_source": by_source})
    return OriginalPredictionResult(established, disagreements)


def load_post_fix_predictions(sept8_fix_df: pd.DataFrame) -> dict[str, StanceLabelV2]:
    """Loads the validated Sept 8, 2026 fix's own predictions (new_stance).

    Unlike establish_original_predictions (which cross-checks old_stance
    across all six CSVs, requiring agreement), this reads new_stance from
    stance_refit_20260908fix.csv alone. The other five CSVs
    (evaluation_data/experiments/exp_*.csv) are later, different
    prompt-engineering experiments, not revisions of this fix -- their
    new_stance values are different systems' predictions, so cross-checking
    them against each other for "agreement" would conflate unrelated runs
    rather than validate one.

    Args:
        sept8_fix_df: The already-loaded, already-duplicate-collapsed
            stance_refit_20260908fix.csv frame (csv_frames["sept8_fix"] in
            build_dataset()), with pair_id and new_stance columns.

    Returns:
        pair_id -> StanceLabelV2, for every row whose new_stance normalizes
        to a recognized label. Rows with an unrecognized/missing new_stance
        are simply absent -- this is real coverage, not guessed.
    """
    result: dict[str, StanceLabelV2] = {}
    for row in sept8_fix_df.itertuples():
        label = normalize_legacy_label(row.new_stance)
        if label is not None:
            result[row.pair_id] = label
    return result


# --------------------------------------------------------------------------
# Tier A / Tier B resolution
# --------------------------------------------------------------------------


@dataclass
class ResolvedPair:
    """One row of the final Cohort A pairs table."""

    pair_id: str
    contribution_id: object
    key_message: str
    comment_text: str
    source_row_id: int
    evaluator: str
    cohort_role: str  # "primary" | "secondary"
    verdict: str
    resolution_status: str  # "resolved" | "unresolved" | "quarantined"
    tier: str | None = None  # "A" | "B"
    reference_label: str | None = None  # StanceLabelV2 value, as plain str for parquet
    basis: str | None = None
    unresolved_reason: str | None = None
    quarantine_reason: str | None = None
    supporting_text: str | None = None
    parser_rule: str | None = None
    established_original_label: str | None = None
    post_fix_original_label: str | None = None
    label_source: str = "human_gold"


def _resolve_verdict(
    row, established_label: StanceLabelV2 | None, post_fix_label: StanceLabelV2 | None
) -> ResolvedPair:
    """Resolve one stance_evaluations row on its own (no overlap check yet)."""
    base = dict(
        pair_id=row.pair_id,
        contribution_id=row.contribution_id,
        key_message=row.key_message,
        comment_text=row.comment_text,
        source_row_id=int(row.id),
        evaluator=row.evaluator,
        cohort_role="primary" if row.evaluator == PRIMARY_EVALUATOR else "secondary",
        verdict=row.verdict,
        resolution_status="unresolved",
        established_original_label=established_label.value if established_label else None,
        post_fix_original_label=post_fix_label.value if post_fix_label else None,
    )

    if row.verdict == "correct":
        if established_label is None:
            base["unresolved_reason"] = "ambiguous_original_prediction_provenance"
            return ResolvedPair(**base)
        base.update(
            resolution_status="resolved",
            tier="A",
            reference_label=established_label.value,
            basis="verdict_correct",
        )
        return ResolvedPair(**base)

    if row.verdict == "incorrect":
        corrected = normalize_legacy_label(row.corrected_label)
        if corrected is not None:
            if established_label is not None and corrected == established_label:
                base.update(
                    resolution_status="quarantined",
                    quarantine_reason="contradiction_correction_equals_original",
                )
                return ResolvedPair(**base)
            base.update(
                resolution_status="resolved",
                tier="A",
                reference_label=corrected.value,
                basis="verdict_incorrect_corrected",
            )
            return ResolvedPair(**base)

        parsed = parse_note(row.comment)
        if parsed.status == "resolved":
            base.update(
                resolution_status="resolved",
                tier="B",
                reference_label=parsed.label,
                basis="tier_b_parsed",
                supporting_text=parsed.supporting_text,
                parser_rule=parsed.rule,
            )
            return ResolvedPair(**base)

        base["unresolved_reason"] = parsed.reason or parsed.status
        return ResolvedPair(**base)

    base["unresolved_reason"] = "unknown_verdict_value"
    return ResolvedPair(**base)


def resolve_cohort_a(
    eval_df: pd.DataFrame,
    established: dict[str, StanceLabelV2],
    post_fix: dict[str, StanceLabelV2] | None = None,
) -> list[ResolvedPair]:
    """Resolve every stance_evaluations row into a ResolvedPair.

    Each row is first resolved independently (Tier A / Tier B / unresolved),
    then pairs judged by both evaluators are cross-checked: agreement is
    reported as an overlap (secondary row kept, reported separately, never
    merged into the primary denominator); disagreement quarantines BOTH
    rows for that pair rather than silently preferring one evaluator.

    Args:
        eval_df: stance_evaluations rows with a pair_id column.
        established: pair_id -> the cross-checked original (pre-Sept-8-fix)
            prediction, from establish_original_predictions().
        post_fix: pair_id -> the validated Sept 8, 2026 fix's own
            prediction, from load_post_fix_predictions(). Optional --
            defaults to no post-fix coverage (every row's
            post_fix_original_label stays None) so existing callers keep
            working unchanged.
    """
    post_fix = post_fix or {}
    by_pair: dict[str, list] = {}
    for row in eval_df.itertuples():
        by_pair.setdefault(row.pair_id, []).append(row)

    resolved: list[ResolvedPair] = []
    for pair_id, rows in by_pair.items():
        established_label = established.get(pair_id)
        post_fix_label = post_fix.get(pair_id)
        provisional = [_resolve_verdict(row, established_label, post_fix_label) for row in rows]

        primary = [p for p in provisional if p.cohort_role == "primary"]
        secondary = [p for p in provisional if p.cohort_role == "secondary"]

        labels = {p.reference_label for p in provisional if p.reference_label}
        if len(labels) > 1:
            for p in provisional:
                p.resolution_status = "quarantined"
                p.quarantine_reason = (
                    "evaluator_disagreement" if primary and secondary else "conflicting_annotations"
                )
                p.tier = None
                p.reference_label = None
                p.basis = None

        resolved.extend(provisional)

    return resolved


# --------------------------------------------------------------------------
# Cohort B: gold_labels_corrected.json
# --------------------------------------------------------------------------


def build_cohort_b(gold_labels_path: Path) -> pd.DataFrame:
    """Load the separate gold_labels_corrected.json benchmark.

    Never joined against Cohort A: this is a different, multi-project
    dataset (project_id is available here, unlike stance_evaluations).
    """
    with open(gold_labels_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    rows = []
    for r in records:
        contribution_id = r.get("contribution_id")
        key_message = r.get("key_message", "")
        comment_text = r.get("comment_text", "")
        project_id = r.get("project_id")
        pair_id = hashlib.sha256(
            f"{project_id}|{contribution_id}|{key_message}|{comment_text}".encode("utf-8")
        ).hexdigest()[:16]
        label = normalize_legacy_label(r.get("gold_stance"))
        rows.append(
            {
                "pair_id": pair_id,
                "project_id": project_id,
                "contribution_id": contribution_id,
                "key_message": key_message,
                "comment_text": comment_text,
                "row_index": r.get("row_index"),
                "resolution_status": "resolved" if label is not None else "unresolved",
                "reference_label": label.value if label else None,
                "unresolved_reason": None if label is not None else "unrecognized_gold_stance_value",
                "label_source": "human_gold",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Policy-derived fixtures (brief section 4's synthetic examples)
# --------------------------------------------------------------------------

# Verbatim from STANCE_PIPELINE_IMPLEMENTATION_BRIEF.md section 4's synthetic
# policy examples table. Assumes three key messages: add bicycle stands,
# remove car parking spaces, plant trees. Tagged policy_derived, never
# human_gold, never merged into dataset/pairs.parquet. Selection is fixed
# here (not derived from any candidate model's output).
POLICY_FIXTURES: list[dict] = [
    {
        "fixture_id": "policy_fixture_01",
        "comment_text": "Super Idee!",
        "expected": {
            "bicycle_stands": "in_favor",
            "remove_parking": "in_favor",
            "plant_trees": "in_favor",
        },
        "basis": "general_approval",
        "note": "General approval; all three in_favor.",
    },
    {
        "fixture_id": "policy_fixture_02",
        "comment_text": "Danke, ja!",
        "expected": {
            "bicycle_stands": "in_favor",
            "remove_parking": "in_favor",
            "plant_trees": "in_favor",
        },
        "basis": "general_approval",
        "note": "General approval when directly addressed to the contribution; all three in_favor.",
    },
    {
        "fixture_id": "policy_fixture_03",
        "comment_text": "Endlich mehr Fahrradständer!",
        "expected": {
            "bicycle_stands": "in_favor",
            "remove_parking": "neutral",
            "plant_trees": "neutral",
        },
        "basis": "explicit_target",
        "note": "Bicycle stands in_favor; other messages neutral.",
    },
    {
        "fixture_id": "policy_fixture_04",
        "comment_text": "Ich unterstütze den gesamten Vorschlag, nur die Parkplätze sollten bleiben.",
        "expected": {
            "bicycle_stands": "in_favor",
            "remove_parking": "against",
            "plant_trees": "in_favor",
        },
        "basis": "mixed_qualified",
        "note": "Parking removal against; bicycle stands and trees in_favor.",
    },
    {
        "fixture_id": "policy_fixture_05",
        "comment_text": "Gute Idee mit den Fahrradständern, aber die Parkplätze sollten bleiben.",
        "expected": {
            "bicycle_stands": "in_favor",
            "remove_parking": "against",
            "plant_trees": "neutral",
        },
        "basis": "mixed_qualified",
        "note": "Bicycle stands in_favor; parking removal against; trees neutral (scope does not extend).",
    },
    {
        "fixture_id": "policy_fixture_06",
        "comment_text": "Siehe oben.",
        "expected": None,
        "basis": "needs_review",
        "note": "Without resolvable context: review, do not infer approval.",
    },
    {
        "fixture_id": "policy_fixture_07",
        "comment_text": "Na klar, noch mehr Autos. Großartige Idee …",
        "expected": None,
        "basis": "needs_review",
        "note": "Potential sarcasm; review when unresolved.",
    },
]


def build_policy_fixtures() -> list[dict]:
    """Return the fixed, versioned policy-derived fixture set (brief section 4)."""
    return [{**fx, "label_source": "policy_derived"} for fx in POLICY_FIXTURES]


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def build_manifest(
    *,
    eval_db_path: Path,
    csv_sources: dict[str, Path],
    gold_labels_path: Path,
    resolved_a: list[ResolvedPair],
    duplicate_reports: dict[str, DuplicateCollapseResult],
    original_prediction_result: OriginalPredictionResult,
    cohort_b: pd.DataFrame,
) -> dict:
    """Assemble the manifest.json content documenting how the dataset was built."""
    source_hashes = {"eval_db": {"path": str(eval_db_path), "sha256": _sha256_file(eval_db_path)}}
    for name, path in csv_sources.items():
        source_hashes[name] = {"path": str(path), "sha256": _sha256_file(path)}
    source_hashes["gold_labels"] = {
        "path": str(gold_labels_path),
        "sha256": _sha256_file(gold_labels_path),
    }

    tier_counts: dict[str, int] = {}
    for p in resolved_a:
        key = p.tier if p.resolution_status == "resolved" else p.resolution_status
        tier_counts[key] = tier_counts.get(key, 0) + 1

    quarantine_reasons: dict[str, int] = {}
    for p in resolved_a:
        if p.resolution_status == "quarantined":
            quarantine_reasons[p.quarantine_reason] = quarantine_reasons.get(p.quarantine_reason, 0) + 1

    unresolved_reasons: dict[str, int] = {}
    for p in resolved_a:
        if p.resolution_status == "unresolved":
            unresolved_reasons[p.unresolved_reason] = unresolved_reasons.get(p.unresolved_reason, 0) + 1

    primary_count = sum(1 for p in resolved_a if p.cohort_role == "primary")
    secondary_count = sum(1 for p in resolved_a if p.cohort_role == "secondary")
    resolved_with_post_fix = sum(
        1 for p in resolved_a if p.resolution_status == "resolved" and p.post_fix_original_label is not None
    )

    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": _current_commit(),
        "sources": source_hashes,
        "cohort_a": {
            "row_count": len(resolved_a),
            "primary_evaluator": PRIMARY_EVALUATOR,
            "primary_rows": primary_count,
            "secondary_rows": secondary_count,
            "resolution_status_counts": {
                status: sum(1 for p in resolved_a if p.resolution_status == status)
                for status in ("resolved", "unresolved", "quarantined")
            },
            "tier_counts": tier_counts,
            "quarantine_reasons": quarantine_reasons,
            "unresolved_reasons": unresolved_reasons,
            "duplicate_collapse": {
                source: {
                    "identical_groups": r.identical_groups,
                    "conflicting_keys": r.conflicting_keys,
                }
                for source, r in duplicate_reports.items()
            },
            "original_prediction_disagreements": original_prediction_result.disagreements,
            "post_fix_coverage": {
                "resolved_rows_with_post_fix_label": resolved_with_post_fix,
                "note": "post_fix_original_label comes from stance_refit_20260908fix.csv's "
                "new_stance column alone (the validated Sept 8, 2026 fix), not "
                "cross-checked against the later prompt-experiment CSVs.",
            },
        },
        "cohort_b": {
            "row_count": len(cohort_b),
            "resolved": int((cohort_b["resolution_status"] == "resolved").sum()),
            "unresolved": int((cohort_b["resolution_status"] == "unresolved").sum()),
            "distinct_projects": sorted(cohort_b["project_id"].dropna().unique().tolist()),
        },
        "notes": [
            "Cohort A and Cohort B are never joined against each other: "
            "stance_evaluations has no project_id column, so cross-project "
            "contribution_id collisions cannot be structurally ruled out "
            "within Cohort A either.",
            "policy_fixtures.json entries are label_source=policy_derived, "
            "never human_gold, and never merged into pairs.parquet.",
        ],
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass
class AuditResult:
    """Everything build_dataset() produced, for the CLI to write out and summarize."""

    pairs_a: pd.DataFrame
    pairs_b: pd.DataFrame
    quarantine: pd.DataFrame
    manifest: dict
    policy_fixtures: list[dict]


def build_dataset(
    *,
    eval_db_path: Path = DEFAULT_EVAL_DB,
    csv_sources: dict[str, Path] = DEFAULT_CSV_SOURCES,
    gold_labels_path: Path = DEFAULT_GOLD_LABELS_PATH,
) -> AuditResult:
    """Run the full audit/build pipeline and return its results in memory.

    Raw source files are only ever read, never modified.
    """
    eval_df = load_stance_evaluations(eval_db_path)
    raw_csv_frames = load_csv_predictions(csv_sources)

    duplicate_reports: dict[str, DuplicateCollapseResult] = {}
    csv_frames: dict[str, pd.DataFrame] = {}
    for name, df in raw_csv_frames.items():
        result = collapse_duplicates(df, name)
        duplicate_reports[name] = result
        csv_frames[name] = result.collapsed

    original_prediction_result = establish_original_predictions(csv_frames)
    post_fix_predictions = (
        load_post_fix_predictions(csv_frames["sept8_fix"]) if "sept8_fix" in csv_frames else {}
    )
    resolved_a = resolve_cohort_a(eval_df, original_prediction_result.established, post_fix_predictions)

    pairs_a = pd.DataFrame([asdict(p) for p in resolved_a])
    quarantine = pairs_a[pairs_a["resolution_status"] == "quarantined"].copy()

    cohort_b = build_cohort_b(gold_labels_path)
    policy_fixtures = build_policy_fixtures()

    manifest = build_manifest(
        eval_db_path=eval_db_path,
        csv_sources=csv_sources,
        gold_labels_path=gold_labels_path,
        resolved_a=resolved_a,
        duplicate_reports=duplicate_reports,
        original_prediction_result=original_prediction_result,
        cohort_b=cohort_b,
    )

    return AuditResult(
        pairs_a=pairs_a,
        pairs_b=cohort_b,
        quarantine=quarantine,
        manifest=manifest,
        policy_fixtures=policy_fixtures,
    )


def write_dataset(result: AuditResult, output_dir: Path) -> None:
    """Write the audit result to dataset/ as manifest.json, pairs.parquet, etc."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.pairs_a.to_parquet(output_dir / "pairs.parquet", index=False)
    result.pairs_b.to_parquet(output_dir / "pairs_cohort_b.parquet", index=False)
    result.quarantine.to_csv(output_dir / "quarantine.csv", index=False)
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(result.manifest, f, indent=2, ensure_ascii=False, default=str)
    with open(output_dir / "policy_fixtures.json", "w", encoding="utf-8") as f:
        json.dump(result.policy_fixtures, f, indent=2, ensure_ascii=False)
