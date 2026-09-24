import pandas as pd

from historical_baseline import build_historical_predictions, build_post_fix_baseline_predictions
from whole_process import OUTPUT_COLUMNS_V2


def _reference_row(**overrides):
    base = dict(
        pair_id="p1",
        contribution_id=1,
        key_message="km1",
        comment_text="c1",
        resolution_status="resolved",
        reference_label="in_favor",
        established_original_label="in_favor",
        post_fix_original_label="in_favor",
    )
    base.update(overrides)
    return base


def test_build_historical_predictions_has_output_columns_v2_shape():
    reference = pd.DataFrame([_reference_row()])

    predictions = build_historical_predictions(reference)

    assert list(predictions.columns) == list(OUTPUT_COLUMNS_V2)
    assert len(predictions) == 1


def test_build_historical_predictions_uses_established_original_label():
    reference = pd.DataFrame([_reference_row(established_original_label="against")])

    predictions = build_historical_predictions(reference)

    assert predictions.iloc[0]["proposed_stance"] == "against"
    assert predictions.iloc[0]["final_stance"] == "against"


def test_build_historical_predictions_every_row_is_accepted():
    reference = pd.DataFrame(
        [
            _reference_row(pair_id="p1", resolution_status="resolved"),
            _reference_row(pair_id="p2", resolution_status="unresolved", established_original_label="neutral"),
        ]
    )

    predictions = build_historical_predictions(reference)

    assert (predictions["status"] == "accepted").all()


def test_build_historical_predictions_preserves_pair_id_and_contribution_id():
    reference = pd.DataFrame([_reference_row(pair_id="abc123", contribution_id=42)])

    predictions = build_historical_predictions(reference)

    assert predictions.iloc[0]["pair_id"] == "abc123"
    assert predictions.iloc[0]["contribution_id"] == "42"


def test_build_historical_predictions_row_count_matches_reference():
    reference = pd.DataFrame([_reference_row(pair_id="p1"), _reference_row(pair_id="p2")])

    predictions = build_historical_predictions(reference)

    assert len(predictions) == len(reference)


# ---------------------------------------------------------------------------
# build_post_fix_baseline_predictions
# ---------------------------------------------------------------------------


def test_build_post_fix_baseline_predictions_has_output_columns_v2_shape():
    reference = pd.DataFrame([_reference_row()])

    predictions = build_post_fix_baseline_predictions(reference)

    assert list(predictions.columns) == list(OUTPUT_COLUMNS_V2)
    assert len(predictions) == 1


def test_build_post_fix_baseline_predictions_uses_post_fix_original_label():
    reference = pd.DataFrame([_reference_row(post_fix_original_label="against")])

    predictions = build_post_fix_baseline_predictions(reference)

    assert predictions.iloc[0]["proposed_stance"] == "against"
    assert predictions.iloc[0]["final_stance"] == "against"
    assert predictions.iloc[0]["basis"] == "post_fix_historical"


def test_build_post_fix_baseline_predictions_excludes_uncovered_rows():
    """A row with no post_fix_original_label (not in stance_refit_20260908fix.csv)
    is excluded entirely, not guessed from established_original_label."""
    reference = pd.DataFrame(
        [
            _reference_row(pair_id="p1", post_fix_original_label="in_favor"),
            _reference_row(pair_id="p2", post_fix_original_label=None),
        ]
    )

    predictions = build_post_fix_baseline_predictions(reference)

    assert len(predictions) == 1
    assert predictions.iloc[0]["pair_id"] == "p1"


def test_build_post_fix_baseline_predictions_empty_when_no_coverage():
    reference = pd.DataFrame([_reference_row(post_fix_original_label=None)])

    predictions = build_post_fix_baseline_predictions(reference)

    assert len(predictions) == 0
