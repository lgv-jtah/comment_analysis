"""Policy v2 evaluation harness.

Implements STANCE_PIPELINE_IMPLEMENTATION_BRIEF.md sections 2 and 6:
forced-choice/accepted accuracy, labelled/operational coverage, review
rate, per-class breakdown with an explicit insufficient-sample flag,
majority-class baseline, pooled out-of-fold scoring, and per-contribution
bootstrap confidence intervals.

Two input shapes:
  - ``reference``: dataset/pairs.parquet's contents (pair_id,
    resolution_status, reference_label, contribution_id, ...) -- built by
    dataset_builder.py. Only resolution_status == "resolved" rows are
    "eligible" (have a usable gold label); unresolved/quarantined rows are
    real input pairs (they count toward operational_coverage) but never
    enter accuracy/coverage-of-labelled-pairs math.
  - ``predictions``: whole_process.OUTPUT_COLUMNS_V2-shaped rows
    (pair_id, contribution_id, status, proposed_stance, final_stance, ...)
    -- produced by WholeProcess.run_v2() or any v2 inference run.

See the brief's section 2 metric definitions, reproduced here:
  - Forced-choice accuracy: accuracy of proposed labels on all eligible
    pairs, before review routing; a failed call counts as incorrect.
  - Accepted accuracy: accuracy among eligible pairs accepted automatically.
  - Labelled coverage: accepted eligible pairs / all eligible pairs.
  - Operational coverage: accepted pairs / all input pairs (including
    pairs with no usable reference label).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import pandas as pd

# Below this many eligible examples, a class's accepted_accuracy/coverage
# numbers are flagged as too small to support a conclusion (brief section 6:
# "State when accepted samples per class are too small for a reliable
# conclusion.") -- not hidden, just flagged.
MIN_CLASS_SAMPLE = 5

REFERENCE_COLUMNS = ["pair_id", "resolution_status", "reference_label", "contribution_id"]


@dataclass
class ClassMetrics:
    """Per-class accepted accuracy and coverage, with a reliability flag.

    Attributes:
        label: The canonical reference label (e.g. "in_favor").
        eligible: Rows with this reference label among the eligible cohort.
        accepted: Of those, how many were status == "accepted".
        accepted_correct: Of the accepted ones, how many matched the label.
        accepted_accuracy: accepted_correct / accepted, or None if accepted
            is 0.
        coverage: accepted / eligible for this class.
        insufficient_sample: True when accepted < MIN_CLASS_SAMPLE.
    """

    label: str
    eligible: int
    accepted: int
    accepted_correct: int
    accepted_accuracy: float | None
    coverage: float
    insufficient_sample: bool
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    review_rate: float = 0.0


@dataclass
class EvaluationMetricsV2:
    """Full metric set for one set of predictions against one reference.

    Attributes:
        total_pairs: Unique expected input pairs (including non-eligible ones).
        eligible_pairs: Rows with a resolved reference label.
        forced_choice_correct / forced_choice_accuracy: See module docstring.
        accepted_pairs / accepted_correct / accepted_accuracy: Accuracy
            restricted to eligible pairs the pipeline actually accepted;
            accepted_accuracy is None when accepted_pairs is 0.
        labelled_coverage: accepted_pairs / eligible_pairs.
        operational_coverage: all accepted pairs / total_pairs.
        review_pairs / review_rate: status == "needs_review" counts.
        failed_pairs / failure_rate: status == "failed" counts.
        per_class: label -> ClassMetrics.
        insufficient_sample: True if any class is flagged insufficient.
        insufficient_sample_reason: Human-readable reason, or None.
        unmatched_prediction_pairs: Predictions whose pair_id has no row in
            the reference at all (neither eligible nor known-unresolved) --
            recorded rather than silently dropped, since it usually means a
            predictions file and a reference file are out of sync.
    """

    total_pairs: int
    eligible_pairs: int
    forced_choice_correct: int
    forced_choice_accuracy: float
    accepted_pairs: int
    accepted_correct: int
    accepted_accuracy: float | None
    labelled_coverage: float
    operational_coverage: float
    review_pairs: int
    review_rate: float
    failed_pairs: int
    failure_rate: float
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)
    insufficient_sample: bool = False
    insufficient_sample_reason: str | None = None
    unmatched_prediction_pairs: int = 0
    unexpected_prediction_pairs: int = 0
    missing_prediction_pairs: int = 0
    conflicting_prediction_pairs: int = 0
    identical_prediction_duplicates: int = 0
    quarantined_reference_pairs: int = 0
    macro_f1: float = 0.0


def select_reference(reference: pd.DataFrame, *, role="primary", tiers=("A", "B")) -> pd.DataFrame:
    """Select unique scoring pairs without changing stored annotation judgments."""
    if role not in ("primary", "secondary", "all") or not set(tiers) <= {"A", "B"}:
        raise ValueError("Invalid reference role or tiers")
    ref = reference.copy()
    if ref.pair_id.isna().any():
        raise ValueError("Reference pair IDs must not be null")
    ref["pair_id"] = ref["pair_id"].astype(str)
    rows = []
    for _, group in ref.groupby("pair_id", sort=False):
        if group.contribution_id.astype("string").nunique() > 1:
            raise ValueError("Conflicting reference contribution IDs")
        labels = group.loc[group.resolution_status == "resolved", "reference_label"].dropna()
        conflict = labels.nunique() > 1 or (group.resolution_status == "quarantined").any()
        selected = group
        if "cohort_role" in group and role != "all":
            selected = group[group.cohort_role == role]
        if selected.empty:
            continue
        usable = selected[selected.resolution_status == "resolved"]
        if "tier" in usable:
            usable = usable[usable.tier.isin(tiers)].sort_values("tier", kind="stable")
        row = (usable if not usable.empty else selected).iloc[0].copy()
        if conflict or usable.empty or row.reference_label not in {"in_favor", "against", "neutral"}:
            row["resolution_status"] = "quarantined" if conflict else "unresolved"
            row["reference_label"] = None
        rows.append(row)
    return pd.DataFrame(rows, columns=ref.columns).reset_index(drop=True)


def unique_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Collapse identical outputs, fail conflicting pairs, and reject mixed runs."""
    preds = predictions.copy()
    for column in ("pair_id", "status", "proposed_stance", "final_stance"):
        if column not in preds:
            preds[column] = None
    if preds.pair_id.isna().any():
        raise ValueError("Prediction pair IDs must not be null")
    preds["pair_id"] = preds["pair_id"].astype(str)
    if "run_id" in preds and preds.run_id.fillna("").nunique() > 1:
        raise ValueError("Mixed prediction runs cannot be scored together")
    rows = []
    for _, group in preds.groupby("pair_id", sort=False):
        # JSON handles list/array evidence from both in-memory and parquet inputs.
        signatures = group.apply(lambda row: row.to_json(), axis=1)
        row = group.iloc[0].copy()
        row["_prediction_conflict"] = int(signatures.nunique() > 1)
        row["_identical_duplicates"] = len(group) - 1 if signatures.nunique() == 1 else 0
        if signatures.nunique() > 1:
            row["status"] = "failed"
        if not isinstance(row.status, str) or row.status not in {"accepted", "needs_review", "failed"}:
            row["status"] = "failed"
        if row.status == "accepted" and (
            not isinstance(row.proposed_stance, str)
            or row.proposed_stance not in {"in_favor", "against", "neutral"}
            or not isinstance(row.final_stance, str)
            or row.final_stance not in {"in_favor", "against", "neutral"}
            or row.proposed_stance != row.final_stance
        ):
            row["status"] = "failed"
        if row.status == "failed":
            row["proposed_stance"] = None
        if row.status != "accepted":
            row["final_stance"] = None
        rows.append(row)
    result = pd.DataFrame(rows, columns=list(preds.columns) + [
        c for c in ("_prediction_conflict", "_identical_duplicates") if c not in preds
    ])
    return result


def _merge(predictions, reference, expected_pairs=None, *, role="primary", tiers=("A", "B")):
    ref = select_reference(reference, role=role, tiers=tiers)
    if expected_pairs is None:
        expected = ref[["pair_id", "contribution_id"]].copy()
    else:
        expected = (expected_pairs.copy() if isinstance(expected_pairs, pd.DataFrame)
                    else pd.DataFrame({"pair_id": list(expected_pairs)}))
        if expected.pair_id.isna().any():
            raise ValueError("Expected pair IDs must not be null")
        expected["pair_id"] = expected["pair_id"].astype(str)
        if "contribution_id" not in expected:
            expected = expected.merge(ref[["pair_id", "contribution_id"]], on="pair_id", how="left")
        if expected.groupby("pair_id").contribution_id.nunique().gt(1).any():
            raise ValueError("Conflicting expected contribution IDs")
        expected = expected[["pair_id", "contribution_id"]].drop_duplicates("pair_id")
    expected["contribution_id"] = expected.contribution_id.astype("string")
    identities = expected.merge(ref[["pair_id", "contribution_id"]], on="pair_id", suffixes=("", "_ref"))
    if (identities.contribution_id != identities.contribution_id_ref.astype("string")).any():
        raise ValueError("Expected cohort and reference contribution IDs disagree")
    preds = unique_predictions(predictions)
    merged = expected.merge(ref[["pair_id", "resolution_status", "reference_label"]],
                            on="pair_id", how="left", validate="one_to_one")
    output_columns = [c for c in preds if c not in merged or c == "pair_id"]
    merged = merged.merge(preds[output_columns], on="pair_id", how="left", validate="one_to_one", indicator=True)
    missing = merged._merge == "left_only"
    merged["_missing_prediction"] = missing
    merged.loc[missing, "status"] = "failed"
    merged.loc[missing, ["proposed_stance", "final_stance"]] = None
    merged.attrs.update(
        unmatched_prediction_pairs=int((~preds.pair_id.isin(reference.pair_id.astype(str))).sum()),
        unexpected_prediction_pairs=int((~preds.pair_id.isin(expected.pair_id)).sum()),
        missing_prediction_pairs=int(missing.sum()),
        quarantined_reference_pairs=int((merged.resolution_status == "quarantined").sum()),
    )
    return merged.drop(columns="_merge")


def evaluate(predictions: pd.DataFrame, reference: pd.DataFrame, expected_pairs=None,
             *, role="primary", tiers=("A", "B")) -> EvaluationMetricsV2:
    """Score one set of predictions against the reference dataset.

    Args:
        predictions: OUTPUT_COLUMNS_V2-shaped rows.
        reference: dataset/pairs.parquet's contents.

    Returns:
        The full EvaluationMetricsV2 for this predictions set.
    """
    return _score(_merge(predictions, reference, expected_pairs, role=role, tiers=tiers))


def _score(merged: pd.DataFrame) -> EvaluationMetricsV2:
    diagnostics = merged.attrs.copy()
    diagnostics["missing_prediction_pairs"] = int(merged["_missing_prediction"].sum())
    diagnostics["quarantined_reference_pairs"] = int(merged.resolution_status.eq("quarantined").sum())
    diagnostics["conflicting_prediction_pairs"] = int(merged["_prediction_conflict"].sum())
    diagnostics["identical_prediction_duplicates"] = int(merged["_identical_duplicates"].sum())
    total_pairs = len(merged)

    eligible = merged[merged["resolution_status"] == "resolved"]
    eligible_pairs = len(eligible)

    forced_choice_correct = int((eligible["proposed_stance"] == eligible["reference_label"]).sum())
    forced_choice_accuracy = forced_choice_correct / eligible_pairs if eligible_pairs else 0.0

    accepted_eligible = eligible[eligible["status"] == "accepted"]
    accepted_pairs = len(accepted_eligible)
    accepted_correct = int((accepted_eligible["final_stance"] == accepted_eligible["reference_label"]).sum())
    accepted_accuracy = (accepted_correct / accepted_pairs) if accepted_pairs else None

    labelled_coverage = accepted_pairs / eligible_pairs if eligible_pairs else 0.0
    operational_coverage = int((merged["status"] == "accepted").sum()) / total_pairs if total_pairs else 0.0

    review_pairs = int((merged["status"] == "needs_review").sum())
    review_rate = review_pairs / total_pairs if total_pairs else 0.0
    failed_pairs = int((merged["status"] == "failed").sum())
    failure_rate = failed_pairs / total_pairs if total_pairs else 0.0

    per_class: dict[str, ClassMetrics] = {}
    insufficient_any = False
    labels = set(eligible.reference_label.dropna()) | (
        set(eligible.proposed_stance.dropna()) & {"in_favor", "against", "neutral"}
    )
    for label in sorted(labels):
        label_eligible = eligible[eligible["reference_label"] == label]
        label_accepted = label_eligible[label_eligible["status"] == "accepted"]
        n_eligible = len(label_eligible)
        n_accepted = len(label_accepted)
        n_correct = int((label_accepted["final_stance"] == label).sum())
        insufficient = n_accepted < MIN_CLASS_SAMPLE
        insufficient_any = insufficient_any or insufficient
        true_positive = int((label_eligible.proposed_stance == label).sum())
        predicted = int((eligible.proposed_stance == label).sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / n_eligible if n_eligible else 0.0
        per_class[label] = ClassMetrics(
            label=label,
            eligible=n_eligible,
            accepted=n_accepted,
            accepted_correct=n_correct,
            accepted_accuracy=(n_correct / n_accepted) if n_accepted else None,
            coverage=(n_accepted / n_eligible) if n_eligible else 0.0,
            insufficient_sample=insufficient,
            precision=precision,
            recall=recall,
            f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            review_rate=float(label_eligible.status.eq("needs_review").mean()) if n_eligible else 0.0,
        )

    return EvaluationMetricsV2(
        total_pairs=total_pairs,
        eligible_pairs=eligible_pairs,
        forced_choice_correct=forced_choice_correct,
        forced_choice_accuracy=forced_choice_accuracy,
        accepted_pairs=accepted_pairs,
        accepted_correct=accepted_correct,
        accepted_accuracy=accepted_accuracy,
        labelled_coverage=labelled_coverage,
        operational_coverage=operational_coverage,
        review_pairs=review_pairs,
        review_rate=review_rate,
        failed_pairs=failed_pairs,
        failure_rate=failure_rate,
        per_class=per_class,
        insufficient_sample=insufficient_any,
        insufficient_sample_reason=(
            f"one or more classes have fewer than {MIN_CLASS_SAMPLE} accepted examples"
            if insufficient_any
            else None
        ),
        macro_f1=sum(cm.f1 for cm in per_class.values()) / len(per_class) if per_class else 0.0,
        **diagnostics,
    )


@dataclass
class BasisMetrics:
    """Accepted accuracy for one `basis` value (e.g. "general_approval").

    Brief section 3: "Existing annotations do not fully match the new
    general-approval policy. Report results against historical labels
    separately from policy-compliance checks." Phase 5's fold-0 screening
    investigation (docs/phase5_screening_results.md) found this split to
    be highly diagnostic in practice: accepted_accuracy for
    basis="general_approval" predictions was 0.111 versus 0.718 for every
    other basis combined, on the same run -- because the historical
    reference labels predate the general-approval-propagation policy, not
    because the non-general-approval predictions are unreliable. This
    breakdown surfaces that split automatically instead of requiring
    ad-hoc analysis after every run.

    Attributes:
        basis: The PairOutput.basis value shared by this group (e.g.
            "general_approval", "explicit_target", "unaddressed").
        n: Accepted, eligible pairs with this basis.
        correct: Of those, how many matched the reference label.
        accuracy: correct / n, or 0.0 if n is 0.
    """

    basis: str
    n: int
    correct: int
    accuracy: float


def accepted_accuracy_by_basis(predictions: pd.DataFrame, reference: pd.DataFrame,
                               expected_pairs=None, **selection) -> dict[str, BasisMetrics]:
    """Splits accepted-pair accuracy by `basis`, per brief section 3.

    Args:
        predictions: OUTPUT_COLUMNS_V2-shaped rows, including a `basis`
            column.
        reference: dataset/pairs.parquet's contents.

    Returns:
        basis -> BasisMetrics, for every basis value present among
        accepted, eligible pairs. Empty if there are none, or if
        predictions has no "basis" column (e.g. legacy/pre-basis
        predictions files).
    """
    if "basis" not in predictions.columns:
        return {}

    merged = _merge(predictions, reference, expected_pairs, **selection)
    eligible = merged[merged["resolution_status"] == "resolved"]
    accepted = eligible[eligible["status"] == "accepted"]

    result: dict[str, BasisMetrics] = {}
    for basis, group in accepted.groupby("basis"):
        n = len(group)
        correct = int((group["final_stance"] == group["reference_label"]).sum())
        result[str(basis)] = BasisMetrics(basis=str(basis), n=n, correct=correct, accuracy=correct / n if n else 0.0)
    return result


def majority_class_baseline(reference: pd.DataFrame, expected_pairs=None, **selection) -> tuple[str | None, float]:
    """Accuracy of always predicting the most common reference label.

    Args:
        reference: dataset/pairs.parquet's contents.

    Returns:
        (majority_label, its accuracy over the eligible cohort), or
        (None, 0.0) if there are no eligible rows.
    """
    aligned = _merge(pd.DataFrame(), reference, expected_pairs, **selection)
    resolved = aligned[aligned["resolution_status"] == "resolved"]
    if resolved.empty:
        return None, 0.0
    counts = resolved["reference_label"].value_counts()
    majority_label = counts.idxmax()
    return majority_label, float(counts.max() / len(resolved))


def evaluate_by_fold(
    predictions: pd.DataFrame, reference: pd.DataFrame, fold_assignment, expected_pairs=None, **selection
) -> dict:
    """Score predictions per outer fold (its test contributions only) and pooled.

    Args:
        predictions: OUTPUT_COLUMNS_V2-shaped rows, including
            contribution_id, covering (at least) every fold's test set.
        reference: dataset/pairs.parquet's contents.
        fold_assignment: A fold_builder.FoldAssignment (or anything with a
            ``.folds`` list of {"test_contribution_ids": [...]} dicts).

    Returns:
        {"per_fold": [{"fold": i, "metrics": EvaluationMetricsV2}, ...],
         "pooled": EvaluationMetricsV2 over every fold's test predictions
         concatenated}.
    """
    aligned = _merge(predictions, reference, expected_pairs, **selection)
    per_fold = []
    seen = set()
    blocks = []
    for i, fold in enumerate(fold_assignment.folds):
        test_ids = set(map(str, fold["test_contribution_ids"]))
        if seen & test_ids:
            raise ValueError("Overlapping test contributions across folds")
        seen.update(test_ids)
        block = aligned[aligned.contribution_id.isin(test_ids)].copy()
        block.attrs = {}
        blocks.append(block)
        per_fold.append({"fold": i, "metrics": _score(block)})
    pooled = pd.concat(blocks, ignore_index=True) if blocks else aligned.iloc[:0].copy()
    unassigned = aligned[~aligned.contribution_id.isin(seen)]
    missing_eligible = unassigned[unassigned.resolution_status == "resolved"]
    if not missing_eligible.empty:
        raise ValueError(
            f"Missing fold assignments for {len(missing_eligible)} eligible expected pairs; "
            "provide complete folds or explicitly select the intended expected cohort"
        )
    pooled.attrs = {}
    return {"per_fold": per_fold, "pooled": _score(pooled), "overall": _score(aligned),
            "unassigned_expected_pairs": len(unassigned),
            "expected_pairs": aligned[["pair_id", "contribution_id"]].copy()}


def pooled_out_of_fold(fold_predictions: list[pd.DataFrame], reference: pd.DataFrame,
                       expected_pairs=None, **selection) -> EvaluationMetricsV2:
    """Score the concatenation of every fold's held-out predictions at once.

    Args:
        fold_predictions: One predictions DataFrame per fold's test-set
            predictions (each row's pair_id must come from that fold's test
            contributions only -- this function does not check that).
        reference: dataset/pairs.parquet's contents.

    Returns:
        EvaluationMetricsV2 computed over every fold's test predictions
        pooled into one set, per brief section 6: "Report pooled
        out-of-fold metrics, per-fold variation."
    """
    seen = set()
    for frame in fold_predictions:
        ids = set(frame.pair_id.astype(str))
        if seen & ids:
            raise ValueError("Overlapping prediction pairs across folds")
        seen.update(ids)
    pooled = pd.concat(fold_predictions, ignore_index=True) if fold_predictions else pd.DataFrame()
    return evaluate(pooled, reference, expected_pairs, **selection)


@dataclass
class BootstrapCI:
    """A bootstrap confidence interval for one metric.

    Attributes:
        point_estimate: The metric computed on the actual (unresampled) data.
        lower: Lower bound of the interval.
        upper: Upper bound of the interval.
        confidence: The nominal confidence level (e.g. 0.95).
        n_iterations: How many bootstrap resamples were drawn.
    """

    point_estimate: float
    lower: float
    upper: float
    confidence: float
    n_iterations: int


def bootstrap_ci(
    predictions: pd.DataFrame,
    reference: pd.DataFrame,
    metric_fn=lambda m: (m.accepted_accuracy if m.accepted_accuracy is not None else 0.0),
    n_iterations: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
    expected_pairs=None,
    **selection,
) -> BootstrapCI:
    """Bootstrap a confidence interval by resampling contributions, not rows.

    Per brief section 6: "Bootstrap confidence intervals by contribution,
    not by independent pair" -- a comment's several key-message pairs are
    not independent samples, so resampling at the row level would
    understate uncertainty.

    Args:
        predictions: OUTPUT_COLUMNS_V2-shaped rows, including a
            contribution_id column.
        reference: dataset/pairs.parquet's contents.
        metric_fn: Extracts a single float from an EvaluationMetricsV2;
            defaults to accepted_accuracy (0.0 when undefined, so the
            interval stays well-defined even for resamples with no
            accepted pairs).
        n_iterations: Number of bootstrap resamples.
        seed: Random seed for reproducibility.
        confidence: Nominal two-sided confidence level.

    Returns:
        A BootstrapCI. If there are no contribution IDs to resample, the
        interval collapses to the point estimate with n_iterations=0.
    """
    if n_iterations < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap iterations or confidence")
    aligned = _merge(predictions, reference, expected_pairs, **selection)
    if aligned.contribution_id.isna().any():
        raise ValueError("Bootstrap requires expected contribution IDs")
    contribution_ids = sorted(aligned["contribution_id"].unique().tolist())

    point = metric_fn(_score(aligned))
    if not contribution_ids:
        return BootstrapCI(point_estimate=point, lower=point, upper=point, confidence=confidence, n_iterations=0)

    rng = random.Random(seed)
    by_contribution = {cid: group for cid, group in aligned.groupby("contribution_id")}

    samples: list[float] = []
    for _ in range(n_iterations):
        drawn_ids = [rng.choice(contribution_ids) for _ in contribution_ids]
        resampled = pd.concat([by_contribution[cid] for cid in drawn_ids], ignore_index=True)
        # Score repeated blocks directly: deduplicating here would erase draw weights.
        resampled.attrs = {}
        samples.append(metric_fn(_score(resampled)))

    samples.sort()
    lo_idx = int((1 - confidence) / 2 * len(samples))
    hi_idx = min(int((1 + confidence) / 2 * len(samples)), len(samples) - 1)
    return BootstrapCI(
        point_estimate=point,
        lower=samples[lo_idx],
        upper=samples[hi_idx],
        confidence=confidence,
        n_iterations=n_iterations,
    )
