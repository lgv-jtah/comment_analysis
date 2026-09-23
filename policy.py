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

from dataclasses import dataclass, field
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
    ScopeAttributionResult,
    ScopeType,
    StanceLabelV2,
    StanceV2OneCallResult,
    normalize_legacy_label,
)
from stance_detector import StanceDetector

POLICY_VERSION = "v2"

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


@dataclass
class PairOutput:
    """One (comment, key message) result, matching the brief section 7 contract.

    Attributes:
        pair_id: Deterministic ID for (contribution, key_message, comment).
        key_message_id: Which key message this result is for.
        proposed_stance: The stance the pipeline arrived at, even if it was
            ultimately routed to review rather than accepted. None only
            when the underlying LLM call itself failed.
        status: "accepted", "needs_review", or "failed". Failed inference
            never silently becomes "accepted" + neutral.
        final_stance: Non-null only when status == "accepted".
        basis: Why final_stance was assigned (e.g. "general_approval",
            "explicit_target", "unaddressed", "no_position"), only set for
            accepted results.
        comment_evidence: Verbatim comment substrings supporting the result.
        source_evidence: Verbatim related_sentences substrings, if used.
        explanation: Concise German justification from the LLM.
        review_reasons: Why this pair needs review, empty unless
            status == "needs_review".
        policy_version: Always "v2" for results from this engine.
        run_id: Carried from the CommentContext this result came from.
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


def _failed(context: CommentContext, km: KeyMessageRef) -> PairOutput:
    return PairOutput(
        pair_id=compute_pair_id(context.contribution_id, km.text, context.comment_text),
        key_message_id=km.key_message_id,
        proposed_stance=None,
        status="failed",
        review_reasons=["llm_call_failed"],
        run_id=context.run_id,
    )


class StancePolicyEngineV2:
    """Resolves stances for a whole comment against every key message at once."""

    def __init__(
        self,
        llm_handler: LLMHandler,
        base_url: str,
        model_name: str,
        api_key: str = "",
        enable_thinking: bool = False,
        prompt_config_path: str = DEFAULT_PROMPT_CONFIG_PATH,
        verifier_enabled: bool = False,
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
        """
        self._llm = llm_handler
        self._base_url = base_url
        self._model_name = model_name
        self._api_key = api_key
        self._enable_thinking = enable_thinking
        self._verifier_enabled = verifier_enabled

        detector = StanceDetector()
        scope_cfg = detector.load_prompt_config(prompt_config_path, "scope_attribution")
        stance_cfg = detector.load_prompt_config(prompt_config_path, "per_target_stance")
        self._scope_system = scope_cfg["system"]
        self._scope_user_template = Template(scope_cfg["user_template"])
        self._stance_system = stance_cfg["system"]
        self._stance_user_template = Template(stance_cfg["user_template"])

    # -- prompt building ----------------------------------------------------

    def _build_scope_messages(self, context: CommentContext) -> tuple[str, list[dict]]:
        user = self._scope_user_template.safe_substitute(
            contribution_text=context.contribution_text or "(nicht verfügbar)",
            key_messages_block=_format_key_messages_block(context.key_messages),
            comment_text=context.comment_text,
        )
        return self._scope_system, [{"role": "user", "content": user}]

    def _build_stance_messages(
        self, context: CommentContext, target_kms: list[KeyMessageRef]
    ) -> tuple[str, list[dict]]:
        user = self._stance_user_template.safe_substitute(
            contribution_text=context.contribution_text or "(nicht verfügbar)",
            key_messages_block=_format_key_messages_block(target_kms),
            comment_text=context.comment_text,
        )
        return self._stance_system, [{"role": "user", "content": user}]

    # -- stage calls ----------------------------------------------------------

    async def _call_scope_stage(
        self, client: httpx.AsyncClient, semaphore, context: CommentContext
    ) -> ScopeAttributionResult | None:
        system, messages = self._build_scope_messages(context)
        raw = await self._llm.call_model(
            client=client,
            base_url=self._base_url,
            model_name=self._model_name,
            system=system,
            messages=messages,
            semaphore=semaphore,
            enable_thinking=self._enable_thinking,
            api_key=self._api_key,
            response_schema=ScopeAttributionResult,
        )
        if raw is None:
            return None
        try:
            return ScopeAttributionResult.model_validate(raw)
        except Exception:
            return None

    async def _call_stance_stage(
        self,
        client: httpx.AsyncClient,
        semaphore,
        context: CommentContext,
        target_kms: list[KeyMessageRef],
    ) -> PerTargetStanceResult | None:
        system, messages = self._build_stance_messages(context, target_kms)
        raw = await self._llm.call_model(
            client=client,
            base_url=self._base_url,
            model_name=self._model_name,
            system=system,
            messages=messages,
            semaphore=semaphore,
            enable_thinking=self._enable_thinking,
            api_key=self._api_key,
            response_schema=PerTargetStanceResult,
        )
        if raw is None:
            return None
        try:
            return PerTargetStanceResult.model_validate(_normalize_stance_fields(raw))
        except Exception:
            return None

    # -- validation -----------------------------------------------------------

    @staticmethod
    def _validate_scope(scope: ScopeAttributionResult, context: CommentContext) -> list[str]:
        errors = []
        valid_ids = {km.key_message_id for km in context.key_messages}
        for kid in scope.affected_key_message_ids:
            if kid not in valid_ids:
                errors.append(f"unknown_key_message_id:{kid}")
        normalized_comment = _normalize_whitespace(context.comment_text)
        for evidence in scope.comment_evidence:
            if _normalize_whitespace(evidence) not in normalized_comment:
                errors.append("evidence_not_substring_of_comment")
        return errors

    @staticmethod
    def _validate_stance(
        result: PerTargetStanceResult, context: CommentContext, allowed_ids: set[str]
    ) -> list[str]:
        errors = []
        related_by_id = {km.key_message_id: set(km.related_sentences) for km in context.key_messages}
        normalized_comment = _normalize_whitespace(context.comment_text)
        for target in result.targets:
            if target.key_message_id not in allowed_ids:
                errors.append(f"unknown_key_message_id:{target.key_message_id}")
                continue
            for evidence in target.comment_evidence:
                if _normalize_whitespace(evidence) not in normalized_comment:
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
        """One full pass through both stages, without the verifier."""
        kms = context.key_messages
        if not kms:
            return {}

        scope = await self._call_scope_stage(client, semaphore, context)
        if scope is None:
            return {km.key_message_id: _failed(context, km) for km in kms}

        scope_errors = self._validate_scope(scope, context)
        if scope_errors:
            return {
                km.key_message_id: _needs_review(context, km, ["invalid_scope_output"] + scope_errors)
                for km in kms
            }

        outputs: dict[str, PairOutput] = {}

        if scope.scope_type == ScopeType.WHOLE_CONTRIBUTION_APPROVAL:
            for km in kms:
                outputs[km.key_message_id] = _accepted(
                    context, km, StanceLabelV2.IN_FAVOR, "general_approval",
                    scope.comment_evidence, [], scope.explanation,
                )
            return outputs

        if scope.scope_type == ScopeType.UNRESOLVED:
            for km in kms:
                outputs[km.key_message_id] = _needs_review(
                    context, km, scope.uncertainty_flags or ["unresolved_scope"]
                )
            return outputs

        # TARGET_SPECIFIC or MIXED_QUALIFIED
        affected = [km for km in kms if km.key_message_id in scope.affected_key_message_ids]
        if affected:
            stance_result = await self._call_stance_stage(client, semaphore, context, affected)
            if stance_result is None:
                for km in affected:
                    outputs[km.key_message_id] = _failed(context, km)
            else:
                stance_errors = self._validate_stance(
                    stance_result, context, {km.key_message_id for km in affected}
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
                        if target.stance == StanceLabelV2.NEUTRAL:
                            basis = target.neutral_basis.value if target.neutral_basis else NeutralBasis.NO_POSITION.value
                        else:
                            basis = "explicit_target"
                        outputs[km.key_message_id] = _accepted(
                            context, km, target.stance, basis,
                            target.comment_evidence, target.source_evidence, target.explanation,
                        )

        for km in kms:
            if km.key_message_id in outputs:
                continue
            if scope.scope_type == ScopeType.MIXED_QUALIFIED and scope.general_support_extends_to_unaddressed:
                outputs[km.key_message_id] = _accepted(
                    context, km, StanceLabelV2.IN_FAVOR, "general_approval",
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

        Runs the scope/attribution stage, then (when applicable) the
        per-target stance stage, validates both against the real context,
        and applies the general-approval rule deterministically. When
        verifier_enabled is True, an independent second pass is compared
        against the first and any disagreement routes that pair to review.

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
                if p.status == "accepted" and v.status == "accepted" and p.final_stance != v.final_stance:
                    primary[km.key_message_id] = _needs_review(
                        context, km, ["verifier_disagreement"]
                    )

        return [primary[km.key_message_id] for km in context.key_messages]


class OneCallPolicyEngineV2:
    """Single-call policy v2 variant: scope and stance decided together.

    Brief section 10's experiment matrix compares this against
    StancePolicyEngineV2's two-stage design. Uses the ``stance_v2_one_call``
    prompt and ``StanceV2OneCallResult`` schema (see policy_schemas.py),
    validated the same way -- every key_message_id must be a real ID for
    this comment, every target must be covered exactly once, and every
    evidence quote must be an exact substring of the comment text. Invalid
    or failed output never becomes a silent accepted+neutral result, same
    as the two-stage engine.
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
        return [
            _accepted(
                context,
                km,
                by_id[km.key_message_id].stance,
                by_id[km.key_message_id].basis.value,
                by_id[km.key_message_id].comment_evidence,
                [],
                by_id[km.key_message_id].explanation,
            )
            for km in kms
        ]
