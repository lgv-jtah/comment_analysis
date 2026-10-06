"""Reshapes dataset_builder's historical label columns into predictions.

Brief section 10, step 1: "historical saved baseline" is the first entry
in the experiment matrix -- the already-deployed system's own predictions,
scored through the exact same evaluator_v2/fold path as every new policy-v2
candidate, so the comparison is apples-to-apples rather than against a
number quoted from an old report.

Two distinct historical baselines are built here, deliberately not merged:
  - build_historical_predictions(): established_original_label, the
    pipeline's predictions from *before* the Sept 8, 2026 code fix
    (cross-checked for agreement across all six evaluation_data CSVs).
  - build_post_fix_baseline_predictions(): post_fix_original_label, the
    validated Sept 8 fix's *own* predictions (new_stance from
    stance_refit_20260908fix.csv alone). This is the more recent, more
    relevant "what does the last deployed system actually score" baseline
    -- see dataset_builder.load_post_fix_predictions's docstring.

Available historical labels are accepted; missing or conflicting labels fail.
"""
from __future__ import annotations

import pandas as pd

from whole_process import OUTPUT_COLUMNS_V2
from evaluator_v2 import unique_predictions


def build_historical_predictions(reference: pd.DataFrame) -> pd.DataFrame:
    """Reshape dataset/pairs.parquet into an OUTPUT_COLUMNS_V2 predictions frame.

    Args:
        reference: dataset/pairs.parquet's contents -- must include
            pair_id, contribution_id, key_message, comment_text, and
            established_original_label.

    Returns:
        One row per reference row, in OUTPUT_COLUMNS_V2 column order.
    """
    n = len(reference)
    predictions = pd.DataFrame(
        {
            "contribution_id": reference["contribution_id"].astype(str),
            "contributionContent": None,
            "key_message_id": None,
            "key_message": reference["key_message"],
            "comment_text": reference["comment_text"],
            "pair_id": reference["pair_id"],
            "status": "accepted",
            "proposed_stance": reference["established_original_label"],
            "final_stance": reference["established_original_label"],
            "basis": "historical",
            "comment_evidence": [[] for _ in range(n)],
            "source_evidence": [[] for _ in range(n)],
            "explanation": "",
            "review_reasons": [[] for _ in range(n)],
            "policy_version": "legacy",
            "run_id": "historical_baseline",
            "extraction_model_name": None,
            "stance_model_name": "historical",
        }
    )
    return unique_predictions(predictions)[OUTPUT_COLUMNS_V2]


def build_post_fix_baseline_predictions(reference: pd.DataFrame) -> pd.DataFrame:
    """Reshape dataset/pairs.parquet's post_fix_original_label into predictions.

    This is the *actual* Sept 8, 2026 fix's own predictions -- one step
    more recent than build_historical_predictions()'s established_original_label,
    which is the pipeline's state *before* that fix (see
    dataset_builder.load_post_fix_predictions's docstring for why the two
    are kept separate rather than merged).

    Missing labels remain failed predictions in the expected cohort.

    Args:
        reference: dataset/pairs.parquet's contents -- must include
            pair_id, contribution_id, key_message, comment_text, and
            post_fix_original_label.

    Returns:
        One row per unique pair, in OUTPUT_COLUMNS_V2 column order.
    """
    covered = reference
    n = len(covered)
    predictions = pd.DataFrame(
        {
            "contribution_id": covered["contribution_id"].astype(str),
            "contributionContent": None,
            "key_message_id": None,
            "key_message": covered["key_message"],
            "comment_text": covered["comment_text"],
            "pair_id": covered["pair_id"],
            "status": "accepted",
            "proposed_stance": covered["post_fix_original_label"],
            "final_stance": covered["post_fix_original_label"],
            "basis": "post_fix_historical",
            "comment_evidence": [[] for _ in range(n)],
            "source_evidence": [[] for _ in range(n)],
            "explanation": "",
            "review_reasons": [[] for _ in range(n)],
            "policy_version": "legacy",
            "run_id": "post_fix_baseline",
            "extraction_model_name": None,
            "stance_model_name": "post_fix_historical",
        }
    )
    return unique_predictions(predictions)[OUTPUT_COLUMNS_V2]
