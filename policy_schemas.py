"""Canonical stance labels and structured decision schemas for policy v2.

Internal v2 code (dataset builder, policy engine, evaluator) uses only the
three canonical StanceLabelV2 values. Legacy vocabulary -- the German-mixed
"in favor"/"ablehnung"/"neutral" used throughout the historical annotation
data and prompts, and the LLM's own "IN FAVOR"/"AGAINST"/"NEUTRAL" -- is
mapped to/from canonical only at ingestion/output boundaries (see
``normalize_legacy_label`` below), never used internally.

The scope/attribution and per-target stance schemas below are the response
contracts for policy.py's two LLM stages. They are new classes, separate
from llm_handler.py's frozen ``Stance`` schema, so that legacy-prompt
historical-baseline reproduction is never affected by v2 schema changes.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


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


# ---------------------------------------------------------------------------
# Stage 1: scope and attribution
# ---------------------------------------------------------------------------


class ScopeType(str, Enum):
    """What portion of the contribution a comment's stance applies to."""

    WHOLE_CONTRIBUTION_APPROVAL = "whole_contribution_approval"
    TARGET_SPECIFIC = "target_specific"
    MIXED_QUALIFIED = "mixed_qualified"
    UNRESOLVED = "unresolved"


class ScopeAttributionResult(BaseModel):
    """Output of the scope/attribution stage: what a comment is about.

    Attributes:
        scope_type: Whole-contribution approval, specific target(s), a
            mixed/qualified approval, or unresolved.
        affected_key_message_ids: The key_message_id values this comment
            addresses. Empty for WHOLE_CONTRIBUTION_APPROVAL and UNRESOLVED
            (every key message is affected/none is resolvable, handled by
            the orchestrator rather than enumerated here).
        general_support_extends_to_unaddressed: Only meaningful when
            scope_type == MIXED_QUALIFIED: whether the qualified approval's
            general-support portion extends to key messages not explicitly
            mentioned (True) or those stay neutral (False/None).
        comment_evidence: Verbatim substrings of the comment text supporting
            this scope decision. Validated by the orchestrator as exact
            substrings of the comment; not accepted just because a quote is
            present -- the quote validates provenance, not correctness.
        uncertainty_flags: Machine-readable reasons this call is uncertain,
            e.g. "unresolved_reply_reference", "suspected_sarcasm".
        explanation: Concise German justification.
    """

    scope_type: ScopeType
    affected_key_message_ids: list[str] = Field(default_factory=list)
    general_support_extends_to_unaddressed: bool | None = None
    comment_evidence: list[str] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)
    explanation: str = ""


# ---------------------------------------------------------------------------
# Stage 2: per-target stance
# ---------------------------------------------------------------------------


class NeutralBasis(str, Enum):
    """Why a target-specific stance resolved to neutral."""

    UNADDRESSED = "unaddressed"
    NO_POSITION = "no_position"


class TargetStance(BaseModel):
    """One key message's resolved stance, from the per-target stance stage.

    Attributes:
        key_message_id: Which key message this applies to.
        stance: The resolved canonical stance.
        neutral_basis: Required when stance == NEUTRAL: whether the comment
            didn't address this key message at all, or addressed it without
            taking a position.
        comment_evidence: Verbatim substrings of the comment text.
        source_evidence: Verbatim substrings of that key message's
            related_sentences (grounding in the original contribution, not
            just the comment).
        explanation: Concise German justification.
    """

    key_message_id: str
    stance: StanceLabelV2
    neutral_basis: NeutralBasis | None = None
    comment_evidence: list[str] = Field(default_factory=list)
    source_evidence: list[str] = Field(default_factory=list)
    explanation: str = ""


class PerTargetStanceResult(BaseModel):
    """Batched output of the per-target stance stage for one comment."""

    targets: list[TargetStance] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# One-call comparison variant (brief section 10's experiment matrix)
# ---------------------------------------------------------------------------


class OneCallBasis(str, Enum):
    """Why a one-call result landed on its stance, mirroring the two-stage bases."""

    GENERAL_APPROVAL = "general_approval"
    EXPLICIT_TARGET = "explicit_target"
    UNADDRESSED = "unaddressed"
    NO_POSITION = "no_position"


class OneCallTargetResult(BaseModel):
    """One key message's stance from the single-call policy-v2 variant."""

    key_message_id: str
    stance: StanceLabelV2
    basis: OneCallBasis
    comment_evidence: list[str] = Field(default_factory=list)
    explanation: str = ""


class StanceV2OneCallResult(BaseModel):
    """Batched one-call output for one comment, covering every key message."""

    targets: list[OneCallTargetResult] = Field(default_factory=list)
