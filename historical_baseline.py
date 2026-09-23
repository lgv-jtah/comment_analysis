"""Reshapes dataset_builder's established-original labels into predictions.

Brief section 10, step 1: "historical saved baseline" is the first entry
in the experiment matrix -- the already-deployed system's own predictions,
scored through the exact same evaluator_v2/fold path as every new policy-v2
candidate, so the comparison is apples-to-apples rather than against a
number quoted from an old report.

The historical pipeline had no review-routing concept (no
accepted/needs_review/failed distinction) -- every prediction it made was
final, so every row here is "accepted" with proposed_stance == final_stance
== established_original_label.
"""
from __future__ import annotations

import pandas as pd

from whole_process import OUTPUT_COLUMNS_V2


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
    return predictions[OUTPUT_COLUMNS_V2]
