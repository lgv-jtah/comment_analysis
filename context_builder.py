"""Builds the shared per-comment context the policy v2 pipeline reasons over.

One CommentContext is built per comment, not per (comment, key message)
pair, and shared across every key message that comment might address. This
is what lets the scope/attribution stage see the full set of a
contribution's key messages at once (fixing the exclusive, one-key-message-
at-a-time attribution the legacy prompt implied) and what makes the CLI
benchmark path and the live whole_process.py path build their input the
same way -- both call build_comment_context().

Missing context is always represented explicitly (contribution_text_available,
parent_context.available, context_flags) rather than silently left blank or
invented. See docs/stance_policy_v2.md, "Known limitation: parent context",
for why parent_context.available is currently always False.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class KeyMessageRef(BaseModel):
    """One key message of a contribution, as seen by the policy engine.

    Attributes:
        key_message_id: Stable ID scoped to the contribution, e.g.
            "83:km1". Synthesized by build_key_message_refs() from
            extraction order since the upstream extractor does not assign
            its own IDs.
        text: The key message's text.
        key_message_type: The extractor's category for this key message
            (e.g. "Wunsch"), empty string if unavailable.
        related_sentences: Exact source sentences the key message was
            derived from. Empty list if unavailable -- never invented.
    """

    key_message_id: str
    text: str
    key_message_type: str = ""
    related_sentences: list[str] = Field(default_factory=list)


class ParentContext(BaseModel):
    """What is known about what a comment is replying to, if anything.

    Attributes:
        available: Whether real parent-context data was found. False for
            every comment built by build_comment_context() today -- see
            module docstring.
        parent_comment_text: The parent's text, only set when available.
        parent_kind: Whether the parent is the contribution itself, another
            comment, or unknown.
    """

    available: bool = False
    parent_comment_text: str | None = None
    parent_kind: Literal["contribution", "comment", "unknown"] = "unknown"


class CommentContext(BaseModel):
    """Everything the policy engine needs to resolve one comment's stances.

    Attributes:
        policy_version: Always "v2" for contexts built by this module.
        run_id: Identifier for the run this context was built for, used in
            PairOutput.run_id and cache keys.
        contribution_id: The contribution's ID (kept as a string; the
            legacy pipeline mixes int and str contribution IDs).
        project_id: The contribution's project, when known. None for
            sources (like stance_evaluations) that don't record it.
        contribution_text: Full contribution text, or None if unavailable.
        contribution_text_available: Explicit flag -- never infer
            availability from contribution_text being falsy alone, since an
            empty-but-present contribution is a different case from a
            missing one.
        comment_id: The comment's ID, when known.
        comment_text: The comment's raw text.
        parent_context: What is known about what this comment replies to.
        key_messages: Every key message of this comment's contribution.
        context_flags: Explicit machine-readable notes about what's missing
            or notable about this context, e.g. "missing_contribution_text".
    """

    policy_version: str = "v2"
    run_id: str
    contribution_id: str
    project_id: str | None = None
    contribution_text: str | None = None
    contribution_text_available: bool = False
    comment_id: str | None = None
    comment_text: str
    parent_context: ParentContext = Field(default_factory=ParentContext)
    key_messages: list[KeyMessageRef] = Field(default_factory=list)
    context_flags: list[str] = Field(default_factory=list)


def build_key_message_refs(
    contribution_id: str, key_messages: list[dict]
) -> list[KeyMessageRef]:
    """Synthesize stable KeyMessageRef IDs from an ordered list of raw dicts.

    Args:
        contribution_id: The owning contribution's ID.
        key_messages: Raw key-message dicts, each with at least a
            "key_message" (or "text") field; "key_message_type" and
            "related_sentences" are optional.

    Returns:
        One KeyMessageRef per input dict, in the same order, with IDs of the
        form "{contribution_id}:km{position}" (1-indexed).
    """
    refs: list[KeyMessageRef] = []
    for position, km in enumerate(key_messages, start=1):
        text = km.get("key_message") or km.get("text") or ""
        related = km.get("related_sentences") or []
        refs.append(
            KeyMessageRef(
                key_message_id=f"{contribution_id}:km{position}",
                text=str(text),
                key_message_type=str(km.get("key_message_type", "")),
                related_sentences=[str(s) for s in related],
            )
        )
    return refs


def build_comment_context(
    *,
    run_id: str,
    contribution_id: str,
    comment_text: str,
    key_messages: list[dict],
    contribution_text: str | None = None,
    project_id: str | None = None,
    comment_id: str | None = None,
    parent_context: ParentContext | None = None,
) -> CommentContext:
    """Build a CommentContext, making every missing piece of context explicit.

    This is the single construction path used both by whole_process.py (once
    wired in a later increment) and by the CLI benchmark path over
    dataset/pairs.parquet, so both call sites reason over identically-shaped
    input.

    Args:
        run_id: Identifier for the current run.
        contribution_id: The contribution's ID.
        comment_text: The comment's raw text.
        key_messages: Raw key-message dicts for this contribution, in
            extraction order (see build_key_message_refs).
        contribution_text: Full contribution text, if available.
        project_id: The contribution's project, if known.
        comment_id: The comment's ID, if known.
        parent_context: Known parent/reply context; defaults to
            ParentContext(available=False) when not supplied.

    Returns:
        A populated CommentContext with context_flags noting anything
        missing.
    """
    contribution_text_available = bool(contribution_text)
    flags: list[str] = []
    if not contribution_text_available:
        flags.append("missing_contribution_text")

    key_message_refs = build_key_message_refs(contribution_id, key_messages)
    if not key_message_refs:
        flags.append("no_key_messages")
    elif not any(km.related_sentences for km in key_message_refs):
        flags.append("missing_related_sentences")

    resolved_parent = parent_context if parent_context is not None else ParentContext()
    if not resolved_parent.available:
        flags.append("missing_parent_context")

    return CommentContext(
        run_id=run_id,
        contribution_id=str(contribution_id),
        project_id=project_id,
        contribution_text=contribution_text,
        contribution_text_available=contribution_text_available,
        comment_id=comment_id,
        comment_text=comment_text,
        parent_context=resolved_parent,
        key_messages=key_message_refs,
        context_flags=flags,
    )
