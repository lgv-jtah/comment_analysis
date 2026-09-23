"""Conservative parser for free-text annotation notes.

Recovers a canonical stance label from an evaluator's free-text ``comment``
field only when the note is genuinely unambiguous -- a whole note that is
exactly one label word, or a clear "Label: ..." prefix with no negation,
hedging, or second label mentioned anywhere in the note. Everything else is
left unresolved rather than guessed.

See STANCE_PIPELINE_IMPLEMENTATION_BRIEF.md section 5, Tier B: "Do not parse
by keyword presence. Notes such as 'nicht neutral,' 'neutral oder against,'
'eher ...,' questions, or explanations mentioning multiple labels are not
unambiguous corrections."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Bump this whenever the parsing rules below change; stored alongside every
# resolved row so a dataset snapshot records exactly which rule version
# produced it.
PARSER_RULE_VERSION = "tier_b_v1"

# Canonical v2 labels the parser can resolve to. Maps both English and the
# German legacy vocabulary onto the same canonical value.
_LABEL_ALIASES: dict[str, str] = {
    "neutral": "neutral",
    "in favor": "in_favor",
    "dafür": "in_favor",
    "against": "against",
    "ablehnung": "against",
    "dagegen": "against",
}

_NEGATION_MARKERS = ("nicht", "kein ", "keine ")
_HEDGE_MARKERS = (
    "eher",
    "vielleicht",
    "möglicherweise",
    "tendenziell",
    "wohl",
    "könnte",
    "eventuell",
    "wahrscheinlich",
)

ParsedNoteStatus = Literal["resolved", "unresolved", "empty"]
UnresolvedReason = Literal[
    "negation", "hedge", "question", "multi_label_mentioned", "no_unambiguous_pattern"
]


@dataclass(frozen=True)
class ParsedNote:
    """Result of attempting to recover a label from one free-text note.

    Attributes:
        status: "resolved" if a label was recovered, "unresolved" if the note
            was non-empty but not unambiguous, "empty" if there was no note.
        label: The recovered canonical label ("in_favor"/"against"/"neutral"),
            only set when status == "resolved".
        rule: The parser rule that resolved it (e.g.
            "tier_b_v1:whole_note_exact"), only set when status == "resolved".
        supporting_text: The verbatim note text that was resolved, only set
            when status == "resolved" -- kept for audit, never re-derived.
        reason: Why an unresolved note could not be parsed, only set when
            status == "unresolved".
    """

    status: ParsedNoteStatus
    label: str | None = None
    rule: str | None = None
    supporting_text: str | None = None
    reason: UnresolvedReason | None = None


def _has_negation_or_hedge(lowered: str) -> bool:
    return any(marker in lowered for marker in _NEGATION_MARKERS) or any(
        marker in lowered for marker in _HEDGE_MARKERS
    )


def _mentions_other_label(lowered: str, matched_alias: str) -> bool:
    return any(
        alias != matched_alias and alias in lowered for alias in _LABEL_ALIASES
    )


def parse_note(note: str | None) -> ParsedNote:
    """Conservatively parse an evaluator's free-text note into a label.

    Only two patterns resolve: the whole (trimmed, case-insensitive) note is
    exactly one label word, or it starts with "<label>:" and the rest of the
    note contains no negation, hedge, or mention of a second label. Anything
    else -- including a note that merely *contains* a label word without one
    of these two structures -- is left unresolved with a reason code, never
    guessed from keyword presence alone.

    Args:
        note: The evaluator's free-text comment, or None/empty. A non-string
            value (e.g. a pandas NaN float read from a missing SQL cell) is
            treated the same as None rather than raising.

    Returns:
        A ParsedNote describing whether and how the note resolved.
    """
    if not isinstance(note, str):
        return ParsedNote(status="empty")
    text = note.strip()
    if not text:
        return ParsedNote(status="empty")
    lowered = text.lower().rstrip(".! ")

    if lowered in _LABEL_ALIASES:
        return ParsedNote(
            status="resolved",
            label=_LABEL_ALIASES[lowered],
            rule=f"{PARSER_RULE_VERSION}:whole_note_exact",
            supporting_text=text,
        )

    for alias, label in _LABEL_ALIASES.items():
        prefix = f"{alias}:"
        if lowered.startswith(prefix):
            if not _has_negation_or_hedge(lowered) and not _mentions_other_label(
                lowered, alias
            ):
                return ParsedNote(
                    status="resolved",
                    label=label,
                    rule=f"{PARSER_RULE_VERSION}:prefix_colon",
                    supporting_text=text,
                )

    if any(marker in lowered for marker in _NEGATION_MARKERS):
        return ParsedNote(status="unresolved", reason="negation")
    if any(marker in lowered for marker in _HEDGE_MARKERS):
        return ParsedNote(status="unresolved", reason="hedge")
    if "?" in text:
        return ParsedNote(status="unresolved", reason="question")
    if sum(alias in lowered for alias in _LABEL_ALIASES) > 1:
        return ParsedNote(status="unresolved", reason="multi_label_mentioned")
    return ParsedNote(status="unresolved", reason="no_unambiguous_pattern")
