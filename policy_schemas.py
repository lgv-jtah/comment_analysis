"""Canonical stance labels for policy v2.

Internal v2 code (dataset builder, policy engine, evaluator) uses only these
three canonical values. Legacy vocabulary -- the German-mixed
"in favor"/"ablehnung"/"neutral" used throughout the historical annotation
data and prompts, and the LLM's own "IN FAVOR"/"AGAINST"/"NEUTRAL" -- is
mapped to/from canonical only at ingestion/output boundaries (see
``normalize_legacy_label`` below), never used internally.

More policy-v2 schemas (scope/attribution, per-target stance) are added here
in a later phase; this module starts minimal because the dataset builder
(an earlier phase) already needs a shared canonical vocabulary.
"""
from __future__ import annotations

from enum import Enum


class StanceLabelV2(str, Enum):
    """Canonical stance labels used throughout policy v2 code."""

    IN_FAVOR = "in_favor"
    AGAINST = "against"
    NEUTRAL = "neutral"


# Every legacy spelling seen in the historical data and prompts, lowercased,
# mapped to its canonical value. "ablehnung" (German for "rejection") is a
# deliberate translation to AGAINST, not a typo -- see llm_handler.py's
# normalize_stance_label, which documents the same mapping for the legacy
# LLM-facing vocabulary.
_LEGACY_LABEL_MAP: dict[str, StanceLabelV2] = {
    "in favor": StanceLabelV2.IN_FAVOR,
    "in_favor": StanceLabelV2.IN_FAVOR,
    "dafür": StanceLabelV2.IN_FAVOR,
    "against": StanceLabelV2.AGAINST,
    "ablehnung": StanceLabelV2.AGAINST,
    "dagegen": StanceLabelV2.AGAINST,
    "neutral": StanceLabelV2.NEUTRAL,
}


def normalize_legacy_label(raw: str | None) -> StanceLabelV2 | None:
    """Map a legacy-vocabulary label string to its canonical v2 value.

    Args:
        raw: A label string in any of the legacy spellings (case/whitespace
            insensitive), or None/empty. Non-string values (e.g. a pandas
            NaN float read from a missing CSV/SQL cell) are treated the same
            as None rather than raising.

    Returns:
        The canonical StanceLabelV2, or None if raw is empty/None/non-string
        or does not match any known legacy spelling. Unknown values are
        never guessed or coerced -- callers must handle None explicitly
        (e.g. by quarantining the row) rather than silently defaulting it.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    return _LEGACY_LABEL_MAP.get(raw.strip().lower())
