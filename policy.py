"""Deterministic orchestration for stance policy v2.

StancePolicyEngineV2.resolve_comment() is the single call site meant to be
used by both the CLI and whole_process.py (fixing the "SYNC NOTE: keep in
sync" duplication in main.py/whole_process.py's legacy fast-path logic --
wiring that in is a later, separately-reviewed increment; this module is
usable and fully tested standalone today via mocked LLMHandler.call_model).

It calls the two structured LLM stages (scope/attribution, then per-target
stance), validates their output against the actual context (real
key-message IDs, evidence that is an exact substring of the comment or that
key message's related sentences), and applies the general-approval rule
deterministically in code rather than trusting the LLM to remember it. See
docs/stance_policy_v2.md for the policy this implements.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from string import Template
from typing import Literal

import httpx

from context_builder import CommentContext, KeyMessageRef
from dataset_builder import compute_pair_id
from llm_handler import LLMHandler
from policy_schemas import (
    NeutralBasis,
    PerTargetStanceResult,
    ReactionPolarity,
    ReactionSplitResult,
    ScopeAttributionResult,
    ScopeType,
    StanceLabelV2,
    StanceV2OneCallResult,
    normalize_legacy_label,
)
from stance_detector import StanceDetector

POLICY_VERSION = "v2"

# Option C (decision change 2026-10-06, docs/stance_policy_v2.md): general
# approval of the whole contribution is reported as a contribution-level
# signal via basis="general_approval", not as support for each key message.
# Every general_approval row therefore carries this stance.
GENERAL_APPROVAL_STANCE = StanceLabelV2.NEUTRAL
# General rejection is the symmetric contribution-level signal, recorded
# with basis="general_rejection" (docs/reaction_split_plan.md). Only the
# opt-in reaction split stage produces it.
GENERAL_REJECTION_STANCE = StanceLabelV2.NEUTRAL
GENERAL_REACTION_BASES = frozenset({"general_approval", "general_rejection"})
_BASIS_BY_POLARITY = {ReactionPolarity.POSITIVE: "general_approval", ReactionPolarity.NEGATIVE: "general_rejection"}
_STANCE_BY_POLARITY = {ReactionPolarity.POSITIVE: GENERAL_APPROVAL_STANCE,
                       ReactionPolarity.NEGATIVE: GENERAL_REJECTION_STANCE}
REACTION_SPLIT_PROMPT_KEY = "reaction_split"

PairStatus = Literal["accepted", "needs_review", "failed"]

DEFAULT_PROMPT_CONFIG_PATH = "./prompts/config.json"


def _normalize_whitespace(text: str) -> str:
    """Collapses any run of whitespace (including newlines) to a single space.

    Used only for the evidence-substring grounding check, never for
    display/storage. A model quoting a multi-line comment often reproduces
    it with newlines folded to spaces (e.g. "a\\nb" -> "a b"); that is the
    same text, not a fabricated quote, so the grounding check should not
    reject it. It still rejects anything that differs by more than
    whitespace (typo "corrections", paraphrases, invented text).
    """
    return " ".join(text.split())


def restates_general_approval(target_evidence: list[str], general_evidence: list[str]) -> bool:
    """Whether target support rests only on the general-approval evidence.

    A model may label a blanket approval such as "Stimmt" as explicit support
    for every key message, quoting the same span each time. Support whose
    every quote lies inside the comment-wide approval evidence adds nothing
    target-specific, so its provenance is inherited general approval, not an
    explicit target position. Any quote outside that evidence keeps the
    explicit attribution.
    """
    general = [_normalize_whitespace(quote) for quote in general_evidence if quote.strip()]
    target = [_normalize_whitespace(quote) for quote in target_evidence if quote.strip()]
    return bool(general) and bool(target) and all(any(quote in span for span in general) for quote in target)


def _locate_spans(spans: list[str], comment: str) -> tuple[list[bool], bool]:
    """Mark where non-overlapping spans occur in a comment.

    Longer spans are placed first, so a short span that also occurs inside
    a longer one (e.g. "gut" in "gut gemacht") takes a later occurrence
    instead of blocking the longer span.

    Args:
        spans: Whitespace-normalized, non-empty spans, each a substring of
            comment.
        comment: The whitespace-normalized comment.

    Returns:
        A tuple of the per-character coverage mask and whether every span
        found a place that no other span covers.
    """
    covered = [False] * len(comment)
    for span in sorted(spans, key=len, reverse=True):
        start = comment.find(span)
        while start != -1 and any(covered[start:start + len(span)]):
            start = comment.find(span, start + 1)
        if start == -1:
            return covered, False
        covered[start:start + len(span)] = [True] * len(span)
    return covered, True


def validate_reaction_split(split: ReactionSplitResult, comment_text: str) -> list[str]:
    """Check a reaction split against its comment (pure).

    Every span must be a whitespace-normalized substring of the comment.
    When the model says the split is possible, the reaction and rest spans
    must also fit into the comment without overlapping and together cover
    every letter and digit of it (only punctuation, spaces and symbols may
    be left over). A reaction needs a polarity and evidence; no reaction
    means neither. A comment with neither a reaction nor a rest is invalid.

    Args:
        split: The split stage's output.
        comment_text: The full comment text.

    Returns:
        Machine-readable error codes; empty when the split is valid.
    """
    errors: list[str] = []
    comment = _normalize_whitespace(comment_text)
    reaction = [_normalize_whitespace(s) for s in split.reaction_spans if s.strip()]
    rest = [_normalize_whitespace(s) for s in split.rest_spans if s.strip()]
    if split.has_general_reaction:
        if split.polarity is None:
            errors.append("general_reaction_missing_polarity")
        if not reaction:
            errors.append("general_reaction_missing_evidence")
    else:
        if split.polarity is not None:
            errors.append("polarity_without_general_reaction")
        if reaction:
            errors.append("reaction_spans_without_general_reaction")
        if not rest:
            errors.append("empty_split")
    if any(span not in comment for span in reaction):
        errors.append("reaction_span_not_substring_of_comment")
    if any(span not in comment for span in rest):
        errors.append("rest_span_not_substring_of_comment")
    if errors or not split.split_possible:
        return errors
    covered, placed = _locate_spans(reaction + rest, comment)
    if not placed:
        errors.append("split_spans_overlap")
    elif any(char.isalnum() and not hit for char, hit in zip(comment, covered)):
        errors.append("split_does_not_cover_comment")
    return errors


@dataclass
class PairOutput:
    """One (comment, key message) result, matching the brief section 7 contract.

    Attributes:
        pair_id: Deterministic ID for (contribution, key_message, comment).
        key_message_id: Which key message this result is for.
        proposed_stance: The stance the pipeline arrived at, even if it was
            ultimately routed to review rather than accepted. None when
            inference failed or review has no canonical proposal.
        status: "accepted", "needs_review", or "failed". Failed inference
            never silently becomes "accepted" + neutral.
        final_stance: Non-null only when status == "accepted".
        basis: Attribution (e.g. "general_approval", "explicit_target",
            "explicit_exclusion", "unaddressed", "no_position"). Review may
            retain the attribution, including "unresolved"; failed has None.
        comment_evidence: Verbatim comment substrings supporting the result.
        source_evidence: Verbatim related_sentences substrings, if used.
        explanation: Concise German justification from the LLM.
        review_reasons: Review or failure reasons; empty for accepted results.
        policy_version: Always "v2" for results from this engine.
        run_id: Carried from the CommentContext this result came from.
        general_reaction: Comment-level result of the reaction split stage,
            repeated on every pair of the comment: whether the comment has a
            general reaction. None when the split stage is disabled or its
            output was unusable.
        general_reaction_polarity: "positive" or "negative" when
            general_reaction is True, else None.
        reaction_evidence: Verbatim comment substrings forming the general
            reaction; empty without one.
        rest_text: The comment without its general reaction (the text the
            stance stages saw); "" when nothing remains, None when the split
            stage is disabled or its output was unusable.
    """

    pair_id: str
    key_message_id: str
    proposed_stance: str | None
    status: PairStatus
    final_stance: str | None = None
    basis: str | None = None
    comment_evidence: list[str] = field(default_factory=list)
    source_evidence: list[str] = field(default_factory=list)
    explanation: str = ""
    review_reasons: list[str] = field(default_factory=list)
    policy_version: str = POLICY_VERSION
    run_id: str = ""
    general_reaction: bool | None = None
    general_reaction_polarity: str | None = None
    reaction_evidence: list[str] = field(default_factory=list)
    rest_text: str | None = None


def _normalize_stance_fields(raw: dict) -> dict:
    """Best-effort normalize raw LLM stance strings before Pydantic validation.

    llm_handler.normalize_stance_label only rewrites a top-level "stance"
    key, which none of the v2 schemas have (stance lives nested inside
    "targets" entries). This walks any "targets" list and normalizes each
    entry's "stance" value through the same legacy-tolerant mapping used
    everywhere else, so minor casing/spacing deviations from the model
    don't cause an avoidable validation failure. Values that don't resolve
    are left untouched, so Pydantic still rejects genuinely invalid output
    rather than silently coercing it.
    """
    if not isinstance(raw, dict):
        return raw
    targets = raw.get("targets")
    if isinstance(targets, list):
        for target in targets:
            if isinstance(target, dict) and "stance" in target:
                normalized = normalize_legacy_label(target["stance"])
                if normalized is not None:
                    target["stance"] = normalized.value
    return raw


def _format_key_messages_block(key_messages: list[KeyMessageRef]) -> str:
    """Render key messages (with their exact source sentences) for a prompt.

    related_sentences must be included here, not just the key-message text:
    the per-target stance stage is asked to ground "source_evidence" in
    these sentences, and the orchestrator validates that evidence is an
    exact substring of them -- the model has no way to produce a valid quote
    for text it was never shown.
    """
    if not key_messages:
        return "(keine Kernaussagen)"
    lines = []
    for km in key_messages:
        lines.append(f"- id: {km.key_message_id}\n  text: {km.text}")
        if km.related_sentences:
            for sentence in km.related_sentences:
                lines.append(f"  related_sentence: {sentence}")
    return "\n".join(lines)


# -- shared PairOutput builders, used by both StancePolicyEngineV2 and
# OneCallPolicyEngineV2 -------------------------------------------------


def _accepted(
    context: CommentContext,
    km: KeyMessageRef,
    stance: StanceLabelV2,
    basis: str,
    comment_evidence: list[str],
    source_evidence: list[str],
    explanation: str,
) -> PairOutput:
    return PairOutput(
        pair_id=compute_pair_id(context.contribution_id, km.text, context.comment_text),
        key_message_id=km.key_message_id,
        proposed_stance=stance.value,
        status="accepted",
        final_stance=stance.value,
        basis=basis,
        comment_evidence=comment_evidence,
        source_evidence=source_evidence,
        explanation=explanation,
        run_id=context.run_id,
    )


def _needs_review(context: CommentContext, km: KeyMessageRef, reasons: list[str]) -> PairOutput:
    return PairOutput(
        pair_id=compute_pair_id(context.contribution_id, km.text, context.comment_text),
        key_message_id=km.key_message_id,
        proposed_stance=None,
        status="needs_review",
        review_reasons=reasons,
        run_id=context.run_id,
    )


def _failed(context: CommentContext, km: KeyMessageRef, reasons: list[str] | None = None) -> PairOutput:
    return PairOutput(
        pair_id=compute_pair_id(context.contribution_id, km.text, context.comment_text),
        key_message_id=km.key_message_id,
        proposed_stance=None,
        status="failed",
        review_reasons=["llm_call_failed"] if reasons is None else reasons,
        run_id=context.run_id,
    )


class StancePolicyEngineV2:
    """Resolves stances for a whole comment against every key message at once.

    With reaction_split_enabled, a split stage first separates the comment's
    general reaction from the rest (docs/reaction_split_plan.md); the scope
    and stance stages then see only the rest. The split stage is opt-in until
    its prompt is written from the annotator's labels (plan phase 3), and
    only this engine supports it: OneCallPolicyEngineV2 and
    revisable_attribution.RevisableAttributionEngine do not.
    """

    def __init__(
        self,
        llm_handler: LLMHandler,
        base_url: str,
        model_name: str,
        api_key: str = "",
        enable_thinking: bool = False,
        prompt_config_path: str = DEFAULT_PROMPT_CONFIG_PATH,
        verifier_enabled: bool = False,
        reaction_split_enabled: bool = False,
    ) -> None:
        """Initialize the engine and load the policy v2 prompt templates.

        Args:
            llm_handler: Shared LLMHandler used for every stage call.
            base_url: Endpoint base URL for the configured model.
            model_name: Model identifier passed to the server.
            api_key: Bearer token for the endpoint.
            enable_thinking: Forwarded to LLMHandler.call_model.
            prompt_config_path: Path to prompts/config.json.
            verifier_enabled: When True, resolve_comment() also runs an
                independent verifier assessment and routes to review on any
                disagreement. Off by default -- see brief section 7:
                "Test whether it improves accepted accuracy ... before
                enabling it by default."
            reaction_split_enabled: When True, run the general-reaction split
                stage before scope and stance. Requires a "reaction_split"
                entry in the prompt config. Off by default.

        Raises:
            ValueError: If reaction_split_enabled is True and the prompt
                config has no "reaction_split" entry.
        """
        self._llm = llm_handler
        self._base_url = base_url
        self._model_name = model_name
        self._api_key = api_key
        self._enable_thinking = enable_thinking
        self._verifier_enabled = verifier_enabled
        self._reaction_split_enabled = reaction_split_enabled

        detector = StanceDetector()
        scope_cfg = detector.load_prompt_config(prompt_config_path, "scope_attribution")
        stance_cfg = detector.load_prompt_config(prompt_config_path, "per_target_stance")
        self._scope_system = scope_cfg["system"]
        self._scope_user_template = Template(scope_cfg["user_template"])
        self._stance_system = stance_cfg["system"]
        self._stance_user_template = Template(stance_cfg["user_template"])
        if reaction_split_enabled:
            with open(prompt_config_path, encoding="utf-8") as f:
                available = json.load(f)
            # load_prompt_config exits the process on a missing key; a clear exception is better here.
            if REACTION_SPLIT_PROMPT_KEY not in available:
                raise ValueError(f"Prompt config {prompt_config_path} has no {REACTION_SPLIT_PROMPT_KEY!r} entry; "
                                 "the split prompt is written in phase 3 of docs/reaction_split_plan.md")
            split_cfg = detector.load_prompt_config(prompt_config_path, REACTION_SPLIT_PROMPT_KEY)
            self._split_system = split_cfg["system"]
            self._split_user_template = Template(split_cfg["user_template"])

    # -- prompt building ----------------------------------------------------

    def _build_split_messages(self, context: CommentContext) -> tuple[str, list[dict]]:
        user = self._split_user_template.safe_substitute(
            contribution_text=context.contribution_text or "(nicht verfügbar)",
            key_messages_block=_format_key_messages_block(context.key_messages),
            comment_text=context.comment_text,
        )
        return self._split_system, [{"role": "user", "content": user}]

    def _build_scope_messages(self, context: CommentContext, full_comment_text: str) -> tuple[str, list[dict]]:
        # $full_comment_text lets a template show the whole comment as read-only context next to the
        # rest text in $comment_text; templates without the placeholder ignore it.
        user = self._scope_user_template.safe_substitute(
            contribution_text=context.contribution_text or "(nicht verfügbar)",
            key_messages_block=_format_key_messages_block(context.key_messages),
            comment_text=context.comment_text,
            full_comment_text=full_comment_text,
        )
        return self._scope_system, [{"role": "user", "content": user}]

    def _build_stance_messages(
        self, context: CommentContext, target_kms: list[KeyMessageRef], full_comment_text: str
    ) -> tuple[str, list[dict]]:
        user = self._stance_user_template.safe_substitute(
            contribution_text=context.contribution_text or "(nicht verfügbar)",
            key_messages_block=_format_key_messages_block(target_kms),
            comment_text=context.comment_text,
            full_comment_text=full_comment_text,
        )
        return self._stance_system, [{"role": "user", "content": user}]

    # -- stage calls ----------------------------------------------------------

    async def _call_structured(self, client, semaphore, system, messages, schema, normalize=None):
        """One structured LLM call, parsed into schema; None on failure or invalid output."""
        raw = await self._llm.call_model(
            client=client,
            base_url=self._base_url,
            model_name=self._model_name,
            system=system,
            messages=messages,
            semaphore=semaphore,
            enable_thinking=self._enable_thinking,
            api_key=self._api_key,
            response_schema=schema,
        )
        if raw is None:
            return None
        try:
            return schema.model_validate(normalize(raw) if normalize else raw)
        except Exception:
            return None

    async def _call_split_stage(
        self, client: httpx.AsyncClient, semaphore, context: CommentContext
    ) -> ReactionSplitResult | None:
        system, messages = self._build_split_messages(context)
        return await self._call_structured(client, semaphore, system, messages, ReactionSplitResult)

    async def _call_scope_stage(
        self, client: httpx.AsyncClient, semaphore, context: CommentContext, full_comment_text: str | None = None
    ) -> ScopeAttributionResult | None:
        system, messages = self._build_scope_messages(context, full_comment_text or context.comment_text)
        return await self._call_structured(client, semaphore, system, messages, ScopeAttributionResult)

    async def _call_stance_stage(
        self,
        client: httpx.AsyncClient,
        semaphore,
        context: CommentContext,
        target_kms: list[KeyMessageRef],
        full_comment_text: str | None = None,
    ) -> PerTargetStanceResult | None:
        system, messages = self._build_stance_messages(context, target_kms, full_comment_text or context.comment_text)
        return await self._call_structured(
            client, semaphore, system, messages, PerTargetStanceResult, _normalize_stance_fields)

    # -- validation -----------------------------------------------------------

    @staticmethod
    def _grounded(evidence: str, texts: list[str]) -> bool:
        """Whether a quote is a whitespace-normalized substring of every given text."""
        quote = _normalize_whitespace(evidence)
        return all(quote in _normalize_whitespace(text) for text in texts)

    @staticmethod
    def _validate_scope(
        scope: ScopeAttributionResult, context: CommentContext, extra_texts: list[str] | None = None
    ) -> list[str]:
        errors = []
        valid_ids = {km.key_message_id for km in context.key_messages}
        for kid in scope.affected_key_message_ids:
            if kid not in valid_ids:
                errors.append(f"unknown_key_message_id:{kid}")
        texts = [context.comment_text, *(extra_texts or [])]
        for evidence in scope.comment_evidence:
            if not StancePolicyEngineV2._grounded(evidence, texts):
                errors.append("evidence_not_substring_of_comment")
        return errors

    @staticmethod
    def _validate_stance(
        result: PerTargetStanceResult, context: CommentContext, allowed_ids: set[str],
        extra_texts: list[str] | None = None,
    ) -> list[str]:
        errors = []
        related_by_id = {km.key_message_id: set(km.related_sentences) for km in context.key_messages}
        texts = [context.comment_text, *(extra_texts or [])]
        for target in result.targets:
            if target.key_message_id not in allowed_ids:
                errors.append(f"unknown_key_message_id:{target.key_message_id}")
                continue
            for evidence in target.comment_evidence:
                if not StancePolicyEngineV2._grounded(evidence, texts):
                    errors.append("evidence_not_substring_of_comment")
            related = related_by_id.get(target.key_message_id, set())
            normalized_related = {_normalize_whitespace(s) for s in related}
            for evidence in target.source_evidence:
                if _normalize_whitespace(evidence) not in normalized_related:
                    errors.append("source_evidence_not_in_related_sentences")
            if target.stance == StanceLabelV2.NEUTRAL and target.neutral_basis is None:
                errors.append("neutral_missing_basis")
        return errors

    # -- orchestration ------------------------------------------------------

    async def _resolve_once(
        self, client: httpx.AsyncClient, semaphore, context: CommentContext
    ) -> dict[str, PairOutput]:
        """One full pass through all stages, without the verifier."""
        if not context.key_messages:
            return {}
        if self._reaction_split_enabled:
            return await self._resolve_with_split(client, semaphore, context)
        return await self._resolve_stages(client, semaphore, context, context)

    async def _resolve_with_split(
        self, client: httpx.AsyncClient, semaphore, context: CommentContext
    ) -> dict[str, PairOutput]:
        """Split stage, then scope and stance on the rest text.

        A comment that is only a general reaction gets basis
        general_approval or general_rejection (neutral) on every key message
        without further calls. Otherwise the scope and stance stages see the
        rest text as the comment, their evidence must lie in the rest and in
        the original comment, and the comment-level split result is attached
        to every pair.
        """
        kms = context.key_messages
        split = await self._call_split_stage(client, semaphore, context)
        if split is None:
            return {km.key_message_id: _failed(context, km) for km in kms}
        errors = validate_reaction_split(split, context.comment_text)
        if errors:
            return {km.key_message_id: _needs_review(context, km, ["invalid_split_output"] + errors) for km in kms}

        rest = " ".join(_normalize_whitespace(s) for s in split.rest_spans if s.strip())
        reaction_evidence = [s for s in split.reaction_spans if s.strip()]
        split_fields = dict(
            general_reaction=split.has_general_reaction,
            general_reaction_polarity=split.polarity.value if split.polarity else None,
            reaction_evidence=reaction_evidence,
            rest_text=rest,
        )
        if not split.split_possible:
            outputs = {km.key_message_id: _needs_review(context, km, ["split_not_possible", *split.uncertainty_flags])
                       for km in kms}
        elif not rest:
            # Validation guarantees a reaction with a polarity when nothing else remains.
            basis, stance = _BASIS_BY_POLARITY[split.polarity], _STANCE_BY_POLARITY[split.polarity]
            outputs = {km.key_message_id: _accepted(context, km, stance, basis, reaction_evidence, [],
                                                    split.hint_explanation) for km in kms}
        else:
            stage_context = context.model_copy(update={"comment_text": rest})
            outputs = await self._resolve_stages(client, semaphore, context, stage_context)
        return {kid: replace(output, **split_fields) for kid, output in outputs.items()}

    async def _resolve_stages(
        self, client: httpx.AsyncClient, semaphore, context: CommentContext, stage_context: CommentContext
    ) -> dict[str, PairOutput]:
        """Scope and stance stages over stage_context; outputs are identified by context.

        Without the split stage both contexts are the same. With it,
        stage_context carries the rest text as its comment: a general
        reaction found there is a split error (review), and general support
        never extends to unaddressed key messages, because the general
        reaction is recorded by the split stage instead.
        """
        kms = context.key_messages
        split_mode = stage_context is not context
        full_text = context.comment_text
        # In split mode evidence must lie in the rest text and in the original comment.
        extra_texts = [full_text] if split_mode else None

        scope = await self._call_scope_stage(client, semaphore, stage_context, full_text)
        if scope is None:
            return {km.key_message_id: _failed(context, km) for km in kms}

        scope_errors = self._validate_scope(scope, stage_context, extra_texts)
        if scope_errors:
            return {
                km.key_message_id: _needs_review(context, km, ["invalid_scope_output"] + scope_errors)
                for km in kms
            }

        outputs: dict[str, PairOutput] = {}

        if scope.scope_type == ScopeType.WHOLE_CONTRIBUTION_APPROVAL:
            if split_mode:
                return {km.key_message_id: _needs_review(context, km, ["general_reaction_in_rest_text"])
                        for km in kms}
            for km in kms:
                outputs[km.key_message_id] = _accepted(
                    context, km, GENERAL_APPROVAL_STANCE, "general_approval",
                    scope.comment_evidence, [], scope.explanation,
                )
            return outputs

        if scope.scope_type == ScopeType.UNRESOLVED:
            for km in kms:
                outputs[km.key_message_id] = _needs_review(
                    context, km, scope.uncertainty_flags or ["unresolved_scope"]
                )
            return outputs

        extends_to_unaddressed = (
            not split_mode
            and scope.scope_type == ScopeType.MIXED_QUALIFIED
            and bool(scope.general_support_extends_to_unaddressed)
        )

        # TARGET_SPECIFIC or MIXED_QUALIFIED
        affected = [km for km in kms if km.key_message_id in scope.affected_key_message_ids]
        if affected:
            stance_result = await self._call_stance_stage(client, semaphore, stage_context, affected, full_text)
            if stance_result is None:
                for km in affected:
                    outputs[km.key_message_id] = _failed(context, km)
            else:
                stance_errors = self._validate_stance(
                    stance_result, stage_context, {km.key_message_id for km in affected}, extra_texts
                )
                if stance_errors:
                    for km in affected:
                        outputs[km.key_message_id] = _needs_review(
                            context, km, ["invalid_stance_output"] + stance_errors
                        )
                else:
                    by_id = {t.key_message_id: t for t in stance_result.targets}
                    for km in affected:
                        target = by_id.get(km.key_message_id)
                        if target is None:
                            outputs[km.key_message_id] = _needs_review(
                                context, km, ["target_missing_from_stance_result"]
                            )
                            continue
                        evidence = target.comment_evidence
                        stance = target.stance
                        if target.stance == StanceLabelV2.NEUTRAL:
                            basis = target.neutral_basis.value if target.neutral_basis else NeutralBasis.NO_POSITION.value
                        elif (
                            target.stance == StanceLabelV2.IN_FAVOR
                            and extends_to_unaddressed
                            and restates_general_approval(target.comment_evidence, scope.comment_evidence)
                        ):
                            basis, evidence = "general_approval", scope.comment_evidence
                            stance = GENERAL_APPROVAL_STANCE
                        else:
                            basis = "explicit_target"
                        outputs[km.key_message_id] = _accepted(
                            context, km, stance, basis,
                            evidence, target.source_evidence, target.explanation,
                        )

        for km in kms:
            if km.key_message_id in outputs:
                continue
            if extends_to_unaddressed:
                outputs[km.key_message_id] = _accepted(
                    context, km, GENERAL_APPROVAL_STANCE, "general_approval",
                    scope.comment_evidence, [], scope.explanation,
                )
            else:
                outputs[km.key_message_id] = _accepted(
                    context, km, StanceLabelV2.NEUTRAL, NeutralBasis.UNADDRESSED.value, [], [], "",
                )

        return outputs

    async def resolve_comment(
        self, client: httpx.AsyncClient, semaphore, context: CommentContext
    ) -> list[PairOutput]:
        """Resolve every key message's stance for one comment.

        Runs the optional reaction split stage, the scope/attribution stage,
        then (when applicable) the per-target stance stage, validates each
        against the real context, and applies the general-reaction rules
        deterministically. When verifier_enabled is True, an independent
        second pass is compared against the first and any disagreement
        (final stance, general-reaction basis, or the split's general
        reaction and polarity) routes that pair to review.

        Args:
            client: Async HTTP client for the LLM requests.
            semaphore: Concurrency guard, shared with any other caller.
            context: The comment's built CommentContext.

        Returns:
            One PairOutput per key message in context.key_messages, in the
            same order.
        """
        primary = await self._resolve_once(client, semaphore, context)

        if self._verifier_enabled and context.key_messages:
            verifier = await self._resolve_once(client, semaphore, context)
            for km in context.key_messages:
                p = primary.get(km.key_message_id)
                v = verifier.get(km.key_message_id)
                if p is None or v is None:
                    continue
                if p.status == "accepted" and v.status == "accepted" and (
                    p.final_stance != v.final_stance
                    # A general reaction and an unaddressed message are both
                    # neutral; disagreeing on that is still a disagreement.
                    or _general_basis(p) != _general_basis(v)
                    or (p.general_reaction, p.general_reaction_polarity)
                    != (v.general_reaction, v.general_reaction_polarity)
                ):
                    primary[km.key_message_id] = _needs_review(
                        context, km, ["verifier_disagreement"]
                    )

        return [primary[km.key_message_id] for km in context.key_messages]


def _general_basis(output: PairOutput) -> str | None:
    """The output's basis when it is a general reaction, else None."""
    return output.basis if output.basis in GENERAL_REACTION_BASES else None


class OneCallPolicyEngineV2:
    """Single-call policy v2 variant: scope and stance decided together.

    Brief section 10's experiment matrix compares this against
    StancePolicyEngineV2's two-stage design. Uses the ``stance_v2_one_call``
    prompt and ``StanceV2OneCallResult`` schema (see policy_schemas.py),
    validated the same way -- every key_message_id must be a real ID for
    this comment, every target must be covered exactly once, and every
    evidence quote must be an exact substring of the comment text. Invalid
    or failed output never becomes a silent accepted+neutral result, same
    as the two-stage engine. It does not support the general-reaction split
    stage (docs/reaction_split_plan.md); only StancePolicyEngineV2 does.
    """

    def __init__(
        self,
        llm_handler: LLMHandler,
        base_url: str,
        model_name: str,
        api_key: str = "",
        enable_thinking: bool = False,
        prompt_config_path: str = DEFAULT_PROMPT_CONFIG_PATH,
    ) -> None:
        """Initialize the engine and load the one-call prompt template.

        Args:
            llm_handler: Shared LLMHandler used for the call.
            base_url: Endpoint base URL for the configured model.
            model_name: Model identifier passed to the server.
            api_key: Bearer token for the endpoint.
            enable_thinking: Forwarded to LLMHandler.call_model.
            prompt_config_path: Path to prompts/config.json.
        """
        self._llm = llm_handler
        self._base_url = base_url
        self._model_name = model_name
        self._api_key = api_key
        self._enable_thinking = enable_thinking

        detector = StanceDetector()
        cfg = detector.load_prompt_config(prompt_config_path, "stance_v2_one_call")
        self._system = cfg["system"]
        self._user_template = Template(cfg["user_template"])

    def _build_messages(self, context: CommentContext) -> tuple[str, list[dict]]:
        user = self._user_template.safe_substitute(
            contribution_text=context.contribution_text or "(nicht verfügbar)",
            key_messages_block=_format_key_messages_block(context.key_messages),
            comment_text=context.comment_text,
        )
        return self._system, [{"role": "user", "content": user}]

    @staticmethod
    def _validate(result: StanceV2OneCallResult, context: CommentContext) -> list[str]:
        errors: list[str] = []
        valid_ids = {km.key_message_id for km in context.key_messages}
        seen_ids: set[str] = set()
        normalized_comment = _normalize_whitespace(context.comment_text)
        for target in result.targets:
            if target.key_message_id not in valid_ids:
                errors.append(f"unknown_key_message_id:{target.key_message_id}")
                continue
            seen_ids.add(target.key_message_id)
            for evidence in target.comment_evidence:
                if _normalize_whitespace(evidence) not in normalized_comment:
                    errors.append("evidence_not_substring_of_comment")
        missing = valid_ids - seen_ids
        if missing:
            errors.append(f"missing_targets:{sorted(missing)}")
        return errors

    async def resolve_comment(
        self, client: httpx.AsyncClient, semaphore, context: CommentContext
    ) -> list[PairOutput]:
        """Resolve every key message's stance for one comment in a single LLM call.

        Args:
            client: Async HTTP client for the LLM request.
            semaphore: Concurrency guard.
            context: The comment's built CommentContext.

        Returns:
            One PairOutput per key message in context.key_messages, in the
            same order.
        """
        kms = context.key_messages
        if not kms:
            return []

        system, messages = self._build_messages(context)
        raw = await self._llm.call_model(
            client=client,
            base_url=self._base_url,
            model_name=self._model_name,
            system=system,
            messages=messages,
            semaphore=semaphore,
            enable_thinking=self._enable_thinking,
            api_key=self._api_key,
            response_schema=StanceV2OneCallResult,
        )
        if raw is None:
            return [_failed(context, km) for km in kms]

        try:
            result = StanceV2OneCallResult.model_validate(_normalize_stance_fields(raw))
        except Exception:
            return [_needs_review(context, km, ["invalid_one_call_output"]) for km in kms]

        errors = self._validate(result, context)
        if errors:
            return [_needs_review(context, km, ["invalid_one_call_output"] + errors) for km in kms]

        by_id = {t.key_message_id: t for t in result.targets}
        if any(t.basis.value == "general_approval" and t.stance == StanceLabelV2.AGAINST
               for t in by_id.values()):
            return [_needs_review(context, km, ["invalid_one_call_output", "general_approval_with_against"])
                    for km in kms]
        # The model decides the basis; the general-approval stance is applied
        # in code, as in the two-stage engine.
        return [
            _accepted(
                context,
                km,
                GENERAL_APPROVAL_STANCE if by_id[km.key_message_id].basis.value == "general_approval"
                else by_id[km.key_message_id].stance,
                by_id[km.key_message_id].basis.value,
                by_id[km.key_message_id].comment_evidence,
                [],
                by_id[km.key_message_id].explanation,
            )
            for km in kms
        ]
