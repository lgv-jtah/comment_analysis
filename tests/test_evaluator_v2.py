import pandas as pd
import pytest

from evaluator_v2 import (
    bootstrap_ci,
    evaluate,
    evaluate_by_fold,
    majority_class_baseline,
    pooled_out_of_fold,
)


def _ref_row(pair_id, contribution_id, resolution_status, reference_label=None):
    return {
        "pair_id": pair_id,
        "contribution_id": contribution_id,
        "resolution_status": resolution_status,
        "reference_label": reference_label,
    }


def _pred_row(pair_id, contribution_id, status, proposed=None, final=None):
    return {
        "pair_id": pair_id,
        "contribution_id": contribution_id,
        "status": status,
        "proposed_stance": proposed,
        "final_stance": final,
    }


# ---------------------------------------------------------------------------
# Hand-computed scenario.
#
# 5 eligible pairs (p1..p5) + 1 unresolved (p6, no reference_label) + 1
# quarantined (p7). Predictions:
#   p1: accepted, correct (in_favor == in_favor)
#   p2: accepted, correct (neutral == neutral)
#   p3: accepted, WRONG (proposed+final=in_favor, ref=against)
#   p4: needs_review (proposed=in_favor matches ref=in_favor -- counts for
#       forced-choice, not for accepted)
#   p5: failed (proposed=None -- wrong for forced-choice, no accepted contribution)
#   p6: accepted, unresolved reference -- doesn't affect eligible-based metrics,
#       but does count toward operational_coverage/total_pairs
#   p7: accepted, quarantined reference -- same as p6
# ---------------------------------------------------------------------------


def _reference():
    return pd.DataFrame(
        [
            _ref_row("p1", "c1", "resolved", "in_favor"),
            _ref_row("p2", "c1", "resolved", "neutral"),
            _ref_row("p3", "c2", "resolved", "against"),
            _ref_row("p4", "c2", "resolved", "in_favor"),
            _ref_row("p5", "c3", "resolved", "neutral"),
            _ref_row("p6", "c3", "unresolved"),
            _ref_row("p7", "c4", "quarantined"),
        ]
    )


def _predictions():
    return pd.DataFrame(
        [
            _pred_row("p1", "c1", "accepted", "in_favor", "in_favor"),
            _pred_row("p2", "c1", "accepted", "neutral", "neutral"),
            _pred_row("p3", "c2", "accepted", "in_favor", "in_favor"),  # wrong
            _pred_row("p4", "c2", "needs_review", "in_favor", None),
            _pred_row("p5", "c3", "failed", None, None),
            _pred_row("p6", "c3", "accepted", "in_favor", "in_favor"),
            _pred_row("p7", "c4", "accepted", "neutral", "neutral"),
        ]
    )


def test_hand_computed_metrics():
    metrics = evaluate(_predictions(), _reference())

    assert metrics.total_pairs == 7
    assert metrics.eligible_pairs == 5  # p1-p5

    # forced-choice: proposed_stance vs reference_label over the 5 eligible pairs
    # p1: in_favor==in_favor correct; p2: neutral==neutral correct;
    # p3: in_favor!=against wrong; p4: in_favor==in_favor correct (review doesn't exclude);
    # p5: None != neutral wrong.
    assert metrics.forced_choice_correct == 3
    assert metrics.forced_choice_accuracy == pytest.approx(3 / 5)

    # accepted+eligible: p1, p2, p3 (p4 is needs_review, p5 is failed)
    assert metrics.accepted_pairs == 3
    # correct among those: p1, p2 (p3 wrong)
    assert metrics.accepted_correct == 2
    assert metrics.accepted_accuracy == pytest.approx(2 / 3)

    assert metrics.labelled_coverage == pytest.approx(3 / 5)
    # operational: accepted pairs (p1,p2,p3,p6,p7 = 5) / total_pairs (7)
    assert metrics.operational_coverage == pytest.approx(5 / 7)

    assert metrics.review_pairs == 1
    assert metrics.review_rate == pytest.approx(1 / 7)
    assert metrics.failed_pairs == 1
    assert metrics.failure_rate == pytest.approx(1 / 7)
    assert metrics.unmatched_prediction_pairs == 0


def test_per_class_breakdown():
    metrics = evaluate(_predictions(), _reference())

    in_favor = metrics.per_class["in_favor"]
    assert in_favor.eligible == 2  # p1, p4
    assert in_favor.accepted == 1  # p1 only (p4 is needs_review)
    assert in_favor.accepted_correct == 1
    assert in_favor.accepted_accuracy == pytest.approx(1.0)

    against = metrics.per_class["against"]
    assert against.eligible == 1  # p3
    assert against.accepted == 1  # p3 was accepted, just wrong
    assert against.accepted_correct == 0
    assert against.accepted_accuracy == pytest.approx(0.0)

    neutral = metrics.per_class["neutral"]
    assert neutral.eligible == 2  # p2, p5
    assert neutral.accepted == 1  # p2 only


def test_insufficient_sample_flagged_for_small_classes():
    # Every class above has fewer than MIN_CLASS_SAMPLE=5 eligible examples.
    metrics = evaluate(_predictions(), _reference())

    assert metrics.insufficient_sample is True
    assert "fewer than 5" in metrics.insufficient_sample_reason
    assert all(cm.insufficient_sample for cm in metrics.per_class.values())


def test_accepted_accuracy_is_none_when_nothing_accepted():
    reference = pd.DataFrame([_ref_row("p1", "c1", "resolved", "in_favor")])
    predictions = pd.DataFrame([_pred_row("p1", "c1", "failed")])

    metrics = evaluate(predictions, reference)

    assert metrics.accepted_pairs == 0
    assert metrics.accepted_accuracy is None


def test_unmatched_predictions_are_counted_not_silently_dropped():
    reference = pd.DataFrame([_ref_row("p1", "c1", "resolved", "in_favor")])
    predictions = pd.DataFrame(
        [
            _pred_row("p1", "c1", "accepted", "in_favor", "in_favor"),
            _pred_row("p-not-in-reference", "c1", "accepted", "in_favor", "in_favor"),
        ]
    )

    metrics = evaluate(predictions, reference)

    assert metrics.total_pairs == 2
    assert metrics.unmatched_prediction_pairs == 1
    # The unmatched pair has no resolution_status match ("resolved"), so it's
    # excluded from eligible_pairs but still counts toward total_pairs.
    assert metrics.eligible_pairs == 1


def test_empty_predictions_do_not_raise():
    reference = _reference()
    empty = pd.DataFrame(columns=["pair_id", "contribution_id", "status", "proposed_stance", "final_stance"])

    metrics = evaluate(empty, reference)

    assert metrics.total_pairs == 0
    assert metrics.eligible_pairs == 0
    assert metrics.forced_choice_accuracy == 0.0
    assert metrics.accepted_accuracy is None
    assert metrics.per_class == {}


# ---------------------------------------------------------------------------
# majority_class_baseline
# ---------------------------------------------------------------------------


def test_majority_class_baseline():
    label, accuracy = majority_class_baseline(_reference())

    # Among 5 eligible rows: in_favor x2, neutral x2, against x1 -- tied,
    # pandas value_counts().idxmax() picks the first in count order (stable).
    assert label in ("in_favor", "neutral")
    assert accuracy == pytest.approx(2 / 5)


def test_majority_class_baseline_no_eligible_rows():
    reference = pd.DataFrame([_ref_row("p1", "c1", "unresolved")])

    label, accuracy = majority_class_baseline(reference)

    assert label is None
    assert accuracy == 0.0


def test_majority_class_baseline_clear_winner():
    reference = pd.DataFrame(
        [
            _ref_row("p1", "c1", "resolved", "neutral"),
            _ref_row("p2", "c1", "resolved", "neutral"),
            _ref_row("p3", "c2", "resolved", "neutral"),
            _ref_row("p4", "c2", "resolved", "in_favor"),
        ]
    )

    label, accuracy = majority_class_baseline(reference)

    assert label == "neutral"
    assert accuracy == pytest.approx(3 / 4)


# ---------------------------------------------------------------------------
# pooled_out_of_fold
# ---------------------------------------------------------------------------


def test_pooled_out_of_fold_concatenates_before_scoring():
    reference = _reference()
    fold_a = pd.DataFrame([_pred_row("p1", "c1", "accepted", "in_favor", "in_favor")])
    fold_b = pd.DataFrame([_pred_row("p2", "c1", "accepted", "neutral", "neutral")])

    pooled = pooled_out_of_fold([fold_a, fold_b], reference)

    assert pooled.eligible_pairs == 2
    assert pooled.accepted_correct == 2


def test_pooled_out_of_fold_empty_list():
    reference = _reference()

    pooled = pooled_out_of_fold([], reference)

    assert pooled.total_pairs == 0


# ---------------------------------------------------------------------------
# evaluate_by_fold
# ---------------------------------------------------------------------------


class _FakeFoldAssignment:
    def __init__(self, folds):
        self.folds = folds


def test_evaluate_by_fold_splits_by_test_contribution_ids():
    reference = _reference()
    predictions = _predictions()  # contributions c1, c2, c3, c4

    fold_assignment = _FakeFoldAssignment(
        folds=[
            {"test_contribution_ids": ["c1"]},
            {"test_contribution_ids": ["c2", "c3", "c4"]},
        ]
    )

    result = evaluate_by_fold(predictions, reference, fold_assignment)

    assert len(result["per_fold"]) == 2
    fold0 = result["per_fold"][0]["metrics"]
    # c1 has p1, p2 -- both eligible and accepted+correct.
    assert fold0.eligible_pairs == 2
    assert fold0.accepted_correct == 2

    fold1 = result["per_fold"][1]["metrics"]
    # c2, c3, c4 have p3 (wrong), p4 (review), p5 (failed), p6, p7 (unresolved/quarantined).
    assert fold1.eligible_pairs == 3  # p3, p4, p5

    # Pooled == scoring everything together (same as test_hand_computed_metrics).
    pooled = result["pooled"]
    assert pooled.eligible_pairs == 5
    assert pooled.accepted_correct == 2


def test_evaluate_by_fold_empty_folds_list():
    result = evaluate_by_fold(_predictions(), _reference(), _FakeFoldAssignment(folds=[]))

    assert result["per_fold"] == []
    assert result["pooled"].total_pairs == 0


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def test_bootstrap_ci_point_estimate_matches_direct_evaluation():
    predictions = _predictions()
    reference = _reference()

    ci = bootstrap_ci(predictions, reference, n_iterations=50, seed=1)

    direct = evaluate(predictions, reference)
    assert ci.point_estimate == pytest.approx(direct.accepted_accuracy)


def test_bootstrap_ci_bounds_are_ordered_and_within_unit_interval():
    ci = bootstrap_ci(_predictions(), _reference(), n_iterations=100, seed=2)

    assert 0.0 <= ci.lower <= ci.point_estimate <= ci.upper <= 1.0
    assert ci.n_iterations == 100


def test_bootstrap_ci_reproducible_for_same_seed():
    a = bootstrap_ci(_predictions(), _reference(), n_iterations=50, seed=7)
    b = bootstrap_ci(_predictions(), _reference(), n_iterations=50, seed=7)

    assert a == b


def test_bootstrap_ci_no_contribution_ids_collapses_to_point_estimate():
    empty = pd.DataFrame(columns=["pair_id", "contribution_id", "status", "proposed_stance", "final_stance"])

    ci = bootstrap_ci(empty, _reference(), n_iterations=50)

    assert ci.n_iterations == 0
    assert ci.lower == ci.upper == ci.point_estimate


def test_bootstrap_ci_custom_metric_fn():
    ci = bootstrap_ci(
        _predictions(),
        _reference(),
        metric_fn=lambda m: m.forced_choice_accuracy,
        n_iterations=30,
        seed=3,
    )

    direct = evaluate(_predictions(), _reference())
    assert ci.point_estimate == pytest.approx(direct.forced_choice_accuracy)
