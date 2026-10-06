import pandas as pd
import pytest
from types import SimpleNamespace

import evaluator_v2 as ev
from historical_baseline import build_historical_predictions, build_post_fix_baseline_predictions


def reference(n=10):
    return pd.DataFrame([dict(pair_id=str(i), contribution_id=str(i // 2),
                              resolution_status="resolved", reference_label="neutral") for i in range(n)])


def predictions(n=8):
    return pd.DataFrame([dict(pair_id=str(i), contribution_id=str(i // 2), status="accepted",
                              proposed_stance="neutral", final_stance="neutral", basis="no_position",
                              run_id="one", comment_evidence=["quote"], source_evidence=[])
                         for i in range(n)])


def test_explicit_ten_pair_cohort_with_eight_outputs_not_full_hundred():
    ref, preds = reference(100), predictions(8)
    metrics = ev.evaluate(preds, ref, expected_pairs=ref.iloc[:10])
    assert metrics.eligible_pairs == metrics.total_pairs == 10
    assert metrics.forced_choice_accuracy == .8
    assert metrics.missing_prediction_pairs == metrics.failed_pairs == 2
    assert ev.evaluate(preds, ref).forced_choice_accuracy == .08


def test_absent_reference_operational_only_and_unexpected_separate():
    metrics = ev.evaluate(predictions(10), reference(4), expected_pairs=reference(8))
    assert metrics.total_pairs == 8
    assert metrics.eligible_pairs == 4
    assert metrics.operational_coverage == 1
    assert metrics.unmatched_prediction_pairs == 6
    assert metrics.unexpected_prediction_pairs == 2


def test_annotation_duplicates_and_role_selection_do_not_multiply():
    ref = reference(2).assign(cohort_role="primary", tier="A")
    secondary = ref.assign(cohort_role="secondary")
    annotations = pd.concat([ref, ref, secondary], ignore_index=True)
    before = annotations.copy(deep=True)
    assert ev.evaluate(predictions(2), annotations).eligible_pairs == 2
    assert ev.evaluate(predictions(2), annotations, role="secondary").eligible_pairs == 2
    pd.testing.assert_frame_equal(annotations, before)


@pytest.mark.parametrize("role", ["primary", "secondary"])
def test_gold_conflict_detected_before_role_or_tier_selection(role):
    ref = reference(1).assign(cohort_role="primary", tier="A")
    other = ref.assign(reference_label="against", cohort_role=role, tier="B")
    metrics = ev.evaluate(predictions(1), pd.concat([ref, other]), tiers=("A",))
    assert metrics.eligible_pairs == 0
    assert metrics.quarantined_reference_pairs == 1


def test_secondary_never_supplements_unresolved_primary():
    ref = reference(1).assign(cohort_role="secondary", tier="A")
    primary = ref.assign(cohort_role="primary", reference_label=None, resolution_status="unresolved")
    annotations = pd.concat([ref, primary])
    assert ev.evaluate(predictions(1), annotations).eligible_pairs == 0
    assert ev.evaluate(predictions(1), annotations, role="secondary").eligible_pairs == 1


def test_tier_selection_preserves_operational_cohort():
    ref = reference(2).assign(tier=["A", "B"], cohort_role="primary")
    assert ev.evaluate(predictions(2), ref, tiers=("A",)).eligible_pairs == 1
    assert ev.evaluate(predictions(2), ref, tiers=("A",)).total_pairs == 2
    assert ev.evaluate(predictions(2), ref).eligible_pairs == 2


def test_identical_list_outputs_collapse_after_parquet_roundtrip(tmp_path):
    preds = pd.concat([predictions(2), predictions(2)], ignore_index=True)
    path = tmp_path / "predictions.parquet"
    preds.to_parquet(path)
    metrics = ev.evaluate(pd.read_parquet(path), reference(2))
    assert metrics.total_pairs == metrics.accepted_correct == 2
    assert metrics.identical_prediction_duplicates == 2


@pytest.mark.parametrize("column,value", [("proposed_stance", "against"), ("basis", "explicit_target"),
                                         ("comment_evidence", ["other"]), ("source_evidence", ["other"])])
def test_conflicting_predictions_fail_once_without_removing_gold(column, value):
    preds = pd.concat([predictions(1), predictions(1)], ignore_index=True)
    preds.at[1, column] = value
    metrics = ev.evaluate(preds, reference(1))
    assert metrics.eligible_pairs == metrics.failed_pairs == metrics.conflicting_prediction_pairs == 1
    assert metrics.forced_choice_correct == 0
    assert ev.accepted_accuracy_by_basis(preds, reference(1)) == {}


def test_mixed_runs_rejected_even_on_disjoint_pairs():
    preds = predictions(2)
    preds.at[1, "run_id"] = "two"
    with pytest.raises(ValueError, match="Mixed"):
        ev.evaluate(preds, reference(2))


def test_failed_stale_label_never_scores_but_review_proposal_does():
    preds = predictions(2)
    preds["status"] = ["failed", "needs_review"]
    metrics = ev.evaluate(preds, reference(2))
    assert metrics.forced_choice_correct == 1
    assert metrics.accepted_pairs == 0
    assert metrics.per_class["neutral"].recall == .5


def test_reliability_uses_accepted_not_eligible_count():
    metrics = ev.evaluate(predictions(4), reference(100))
    assert metrics.per_class["neutral"].insufficient_sample


def test_diagnostics_share_the_explicit_cohort():
    ref = reference(10)
    ref.loc[2:, "reference_label"] = "against"
    expected = ref.iloc[:2]
    assert ev.majority_class_baseline(ref, expected) == ("neutral", 1.0)
    assert ev.accepted_accuracy_by_basis(predictions(10), ref, expected)["no_position"].n == 2


class Folds:
    folds = [{"test_contribution_ids": ["0"]}, {"test_contribution_ids": ["1"]}]


def test_fold_with_no_surviving_predictions_still_scores_missing_expected_pairs():
    result = ev.evaluate_by_fold(predictions(1), reference(100), Folds(), reference(4))
    assert result["per_fold"][1]["metrics"].eligible_pairs == 2
    assert result["per_fold"][1]["metrics"].missing_prediction_pairs == 2
    assert result["pooled"].eligible_pairs == 4
    assert result["pooled"].forced_choice_accuracy == .25


def test_overlapping_folds_and_pooled_pairs_rejected():
    folds = Folds()
    folds.folds = [{"test_contribution_ids": ["0"]}] * 2
    with pytest.raises(ValueError, match="Overlapping"):
        ev.evaluate_by_fold(predictions(1), reference(1), folds)
    with pytest.raises(ValueError, match="Overlapping"):
        ev.pooled_out_of_fold([predictions(1), predictions(1)], reference(1))


def test_bootstrap_retains_repeated_block_weights_and_missing_contributions(monkeypatch):
    ref = reference(3).assign(contribution_id=["a", "b", "c"])
    draws = iter(["a", "a", "b"])
    monkeypatch.setattr(ev.random.Random, "choice", lambda self, values: next(draws))
    ci = ev.bootstrap_ci(predictions(1), ref, metric_fn=lambda m: m.forced_choice_accuracy,
                         n_iterations=1)
    assert ci.point_estimate == pytest.approx(1 / 3)
    assert ci.lower == ci.upper == pytest.approx(2 / 3)


@pytest.mark.parametrize("builder,column", [(build_historical_predictions, "established_original_label"),
                                           (build_post_fix_baseline_predictions, "post_fix_original_label")])
def test_historical_duplicates_unique_and_missing_or_conflicting_labels_fail(builder, column):
    ref = reference(2).assign(key_message="km", comment_text="comment")
    ref[column] = ["neutral", None]
    preds = builder(pd.concat([ref, ref]))
    assert len(preds) == 2
    assert list(preds.status) == ["accepted", "failed"]
    other = ref.iloc[:1].copy()
    other[column] = "against"
    preds = builder(pd.concat([ref, other]))
    assert len(preds) == 2
    assert set(preds.status) == {"failed"}


@pytest.mark.parametrize("explicit", [False, True])
def test_missing_eligible_fold_assignment_rejected_even_without_prediction(explicit):
    ref = reference(3).assign(contribution_id=["a", "b", "c"])
    ref.loc[2, ["resolution_status", "reference_label"]] = ["unresolved", None]
    folds = SimpleNamespace(folds=[{"test_contribution_ids": ["a"]}])
    with pytest.raises(ValueError, match="Missing fold assignments for 1 eligible expected pairs"):
        ev.evaluate_by_fold(predictions(1), ref, folds, ref if explicit else None)
    assert ev.evaluate(predictions(1), ref).forced_choice_accuracy == .5


def test_unassigned_unlabelled_inputs_remain_in_overall_operational_metrics():
    ref = reference(3).assign(contribution_id=["a", "b", "c"])
    ref.loc[2, ["resolution_status", "reference_label"]] = ["unresolved", None]
    folds = SimpleNamespace(folds=[{"test_contribution_ids": ["a", "b"]}])
    result = ev.evaluate_by_fold(predictions(1), ref, folds)
    assert result["overall"] == ev.evaluate(predictions(1), ref)
    assert result["overall"].total_pairs == 3
    assert result["overall"].operational_coverage == pytest.approx(1 / 3)
    assert result["overall"].missing_prediction_pairs == 2
    assert result["pooled"].total_pairs == 2
    assert result["pooled"].operational_coverage == .5
    assert result["overall"].forced_choice_accuracy == result["pooled"].forced_choice_accuracy == .5
    assert result["unassigned_expected_pairs"] == 1
    assert len(result["expected_pairs"]) == 3


def test_unreferenced_expected_inputs_still_operational_without_fold_assignment():
    ref = reference(2)
    expected = reference(3)
    result = ev.evaluate_by_fold(predictions(1), ref, SimpleNamespace(
        folds=[{"test_contribution_ids": ["0"]}]), expected)
    assert result["overall"].total_pairs == 3
    assert result["overall"].eligible_pairs == 2
    assert result["unassigned_expected_pairs"] == 1


@pytest.mark.parametrize("column", ["proposed_stance", "final_stance", "status"])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_nullable_accepted_outputs_fail_without_boolean_na_errors(tmp_path, column, roundtrip):
    preds = predictions(1).astype({c: "string" for c in ("proposed_stance", "final_stance", "status")})
    preds.loc[0, column] = pd.NA
    if roundtrip:
        path = tmp_path / "nullable.parquet"
        preds.to_parquet(path)
        preds = pd.read_parquet(path)
    metrics = ev.evaluate(preds, reference(3))
    assert metrics.failed_pairs == 3
    assert metrics.forced_choice_correct == metrics.accepted_pairs == 0


def test_fold_diagnostics_follow_pairs_not_global_attrs():
    ref = reference(5).assign(contribution_id=["a", "b", "c", "d", "e"])
    ref.loc[2:, ["resolution_status", "reference_label"]] = ["unresolved", None]
    base = predictions(5)
    preds = pd.concat([base, base.iloc[[0, 2]].assign(proposed_stance="against", final_stance="against"),
                       base.iloc[[1, 1, 3]]], ignore_index=True)
    folds = SimpleNamespace(folds=[{"test_contribution_ids": ["a"]},
                                   {"test_contribution_ids": ["b"]},
                                   {"test_contribution_ids": ["e"]}])
    result = ev.evaluate_by_fold(preds, ref, folds)
    a, b, e = [row["metrics"] for row in result["per_fold"]]
    assert (a.conflicting_prediction_pairs, a.identical_prediction_duplicates) == (1, 0)
    assert (b.conflicting_prediction_pairs, b.identical_prediction_duplicates) == (0, 2)
    assert (e.conflicting_prediction_pairs, e.identical_prediction_duplicates) == (0, 0)
    assert result["pooled"].conflicting_prediction_pairs == 1
    assert result["pooled"].identical_prediction_duplicates == 2
    assert result["overall"].conflicting_prediction_pairs == 2
    assert result["overall"].identical_prediction_duplicates == 3
