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
        insufficient_sample: True when eligible < MIN_CLASS_SAMPLE.
    """

    label: str
    eligible: int
    accepted: int
    accepted_correct: int
    accepted_accuracy: float | None
    coverage: float
    insufficient_sample: bool


@dataclass
class EvaluationMetricsV2:
    """Full metric set for one set of predictions against one reference.

    Attributes:
        total_pairs: All rows in predictions (including non-eligible ones).
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


def _merge(predictions: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    ref = reference[REFERENCE_COLUMNS].copy()
    ref["pair_id"] = ref["pair_id"].astype(str)
    preds = predictions.copy()
    preds["pair_id"] = preds["pair_id"].astype(str)
    return preds.merge(ref, on="pair_id", how="left", suffixes=("", "_ref"))


def evaluate(predictions: pd.DataFrame, reference: pd.DataFrame) -> EvaluationMetricsV2:
    """Score one set of predictions against the reference dataset.

    Args:
        predictions: OUTPUT_COLUMNS_V2-shaped rows.
        reference: dataset/pairs.parquet's contents.

    Returns:
        The full EvaluationMetricsV2 for this predictions set.
    """
    merged = _merge(predictions, reference)
    total_pairs = len(merged)
    unmatched = int(merged["resolution_status"].isna().sum())

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
    for label in sorted(eligible["reference_label"].dropna().unique().tolist()):
        label_eligible = eligible[eligible["reference_label"] == label]
        label_accepted = label_eligible[label_eligible["status"] == "accepted"]
        n_eligible = len(label_eligible)
        n_accepted = len(label_accepted)
        n_correct = int((label_accepted["final_stance"] == label).sum())
        insufficient = n_eligible < MIN_CLASS_SAMPLE
        insufficient_any = insufficient_any or insufficient
        per_class[label] = ClassMetrics(
            label=label,
            eligible=n_eligible,
            accepted=n_accepted,
            accepted_correct=n_correct,
            accepted_accuracy=(n_correct / n_accepted) if n_accepted else None,
            coverage=(n_accepted / n_eligible) if n_eligible else 0.0,
            insufficient_sample=insufficient,
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
            f"one or more classes have fewer than {MIN_CLASS_SAMPLE} eligible examples"
            if insufficient_any
            else None
        ),
        unmatched_prediction_pairs=unmatched,
    )


def majority_class_baseline(reference: pd.DataFrame) -> tuple[str | None, float]:
    """Accuracy of always predicting the most common reference label.

    Args:
        reference: dataset/pairs.parquet's contents.

    Returns:
        (majority_label, its accuracy over the eligible cohort), or
        (None, 0.0) if there are no eligible rows.
    """
    resolved = reference[reference["resolution_status"] == "resolved"]
    if resolved.empty:
        return None, 0.0
    counts = resolved["reference_label"].value_counts()
    majority_label = counts.idxmax()
    return majority_label, float(counts.max() / len(resolved))


def evaluate_by_fold(
    predictions: pd.DataFrame, reference: pd.DataFrame, fold_assignment
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
    preds = predictions.copy()
    preds["contribution_id"] = preds["contribution_id"].astype(str)

    per_fold = []
    fold_predictions = []
    for i, fold in enumerate(fold_assignment.folds):
        test_ids = set(fold["test_contribution_ids"])
        fold_preds = preds[preds["contribution_id"].isin(test_ids)]
        fold_predictions.append(fold_preds)
        per_fold.append({"fold": i, "metrics": evaluate(fold_preds, reference)})

    return {"per_fold": per_fold, "pooled": pooled_out_of_fold(fold_predictions, reference)}


def pooled_out_of_fold(fold_predictions: list[pd.DataFrame], reference: pd.DataFrame) -> EvaluationMetricsV2:
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
    if not fold_predictions:
        return evaluate(pd.DataFrame(columns=["pair_id", "status", "proposed_stance", "final_stance"]), reference)
    pooled = pd.concat(fold_predictions, ignore_index=True)
    return evaluate(pooled, reference)


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
    preds = predictions.copy()
    preds["contribution_id"] = preds["contribution_id"].astype(str)
    contribution_ids = sorted(preds["contribution_id"].dropna().unique().tolist())

    point = metric_fn(evaluate(predictions, reference))
    if not contribution_ids:
        return BootstrapCI(point_estimate=point, lower=point, upper=point, confidence=confidence, n_iterations=0)

    rng = random.Random(seed)
    by_contribution = {cid: group for cid, group in preds.groupby("contribution_id")}

    samples: list[float] = []
    for _ in range(n_iterations):
        drawn_ids = [rng.choice(contribution_ids) for _ in contribution_ids]
        resampled = pd.concat([by_contribution[cid] for cid in drawn_ids], ignore_index=True)
        samples.append(metric_fn(evaluate(resampled, reference)))

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
