"""Opt-in two-stage attribution: a hypothesis never gates target assessment."""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from copy import deepcopy
from pathlib import Path
from string import Template
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model, model_validator

from llm_handler import DuplicateJSONKeyError
from policy import (GENERAL_APPROVAL_STANCE, _accepted, _failed, _needs_review, _normalize_whitespace,
                    restates_general_approval)
from policy_schemas import ScopeAttributionResult, ScopeType, StanceLabelV2
from stance_detector import StanceDetector

PROMPT_CONFIG = str(Path(__file__).parent / "prompts/revisable_attribution_config.json")
REVISABLE_VARIANTS = ("baseline", "stage1_comment_only", "stage2_required_targets", "source_bound_evidence_ids")
VARIANT_VERSION = "revisable_diagnostics_v1"
EVIDENCE_CONTRACT_VERSION = "source_bound_evidence_ids_v1"
SEGMENTATION_VERSION = "conservative_sentence_newline_v1"
OFFSET_CONVENTION = "unicode_codepoint_halfopen"


def validate_revisable_variant(variant, mode="revisable_attribution", verifier_enabled=False):
    if variant not in REVISABLE_VARIANTS:
        raise ValueError("Unknown revisable variant")
    if variant != "baseline" and mode != "revisable_attribution":
        raise ValueError("revisable variant requires revisable_attribution")
    if mode == "revisable_attribution" and verifier_enabled:
        raise ValueError("revisable_attribution does not support a verifier")


def variant_prompt_config(variant):
    validate_revisable_variant(variant)
    if variant == "baseline":
        return PROMPT_CONFIG
    return str(Path(PROMPT_CONFIG).with_name(f"revisable_{variant}_config.json"))


class GlobalHypothesis(ScopeAttributionResult):
    model_config = ConfigDict(extra="forbid")


class TargetAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key_message_id: str
    attribution: Literal["explicit_target", "explicit_exclusion", "unaddressed", "no_position", "unresolved"]
    stance: StanceLabelV2 | None = None
    comment_evidence: list[str] = Field(default_factory=list)
    source_evidence: list[str] = Field(default_factory=list)
    uncertainty_flags: list[str] = Field(default_factory=list)
    explanation: str = ""

    @model_validator(mode="after")
    def consistent_decision(self):
        if self.attribution == "explicit_target" and self.stance not in (
            StanceLabelV2.IN_FAVOR, StanceLabelV2.AGAINST
        ):
            raise ValueError("explicit_target requires support or opposition")
        if self.attribution == "explicit_exclusion" and self.stance is None:
            raise ValueError("explicit_exclusion requires an explicit stance, including neutral")
        if self.attribution in ("unaddressed", "no_position") and self.stance not in (None, StanceLabelV2.NEUTRAL):
            raise ValueError("Non-explicit assessment cannot supply a directional stance")
        if (self.attribution.startswith("explicit_") or self.stance is not None
                and self.attribution == "unresolved") and not self.comment_evidence:
            raise ValueError("Explicit decisions and review proposals require evidence")
        return self


class RevisedAttributionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revised_scope: GlobalHypothesis
    targets: list[TargetAssessment]


def evidence_context_sha256(context):
    return hashlib.sha256(json.dumps(context.model_dump(exclude={"run_id"}), sort_keys=True).encode()).hexdigest()


def comment_evidence_spans(text):
    """Keep uncertain sentence boundaries joined; never discard non-whitespace."""
    abbreviations = {"dr", "prof", "mr", "mrs", "ms", "bzw", "ca", "etc", "usw", "vgl", "nr", "str", "sog"}
    start = 0
    for boundary in re.finditer(r'[\r\n\x85\u2028\u2029]+|[.!?]+[\"\'\u2019\u201d\u00bb\u203a)\]}]*(?=\s|$)', text):
        token = boundary.group()
        end = boundary.end()
        if token.startswith("."):
            word = re.search(r"[\w.]+$", text[start:boundary.start()])
            word = word.group() if word else ""
            if len(word) == 1 or "." in word or word.lower() in abbreviations:
                continue
            following = text[end:].lstrip()
            whitespace = text[end:len(text) - len(following)]
            next_word = following.lstrip('\"\'\u2018\u201c\u201e\u00ab\u2039([{')
            if next_word and not next_word[0].isupper() and not any(c in whitespace for c in "\r\n\x85\u2028\u2029"):
                continue
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            yield left, right
        start = end
    end = len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        yield start, end


def build_evidence_table(context):
    fingerprint = evidence_context_sha256(context)
    ids = [km.key_message_id for km in context.key_messages]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate target IDs")
    fields = [("c", "comment_text", None, None, None, context.comment_text,
               list(comment_evidence_spans(context.comment_text)))]
    for target_index, km in enumerate(context.key_messages):
        for source_index, text in enumerate(km.related_sentences):
            if text.strip():
                fields.append(("s", f"key_messages[{target_index}].related_sentences[{source_index}]",
                               km.key_message_id, km.text, source_index, text, [(0, len(text))]))
    table = []
    seen = set()
    for namespace, path, target, target_text, source_index, text, spans in fields:
        binding = hashlib.sha256(json.dumps({
            "contract_version": EVIDENCE_CONTRACT_VERSION, "segmentation_version": SEGMENTATION_VERSION,
            "offset_convention": OFFSET_CONVENTION, "context_sha256": fingerprint,
            "source_field_path": path, "target_binding": target, "target_text": target_text,
            "source_index": source_index, "exact_field": text,
        }, sort_keys=True).encode()).hexdigest()
        for start, end in spans:
            evidence_id = f"{namespace}:{binding}:{start}:{end}"
            if evidence_id in seen:
                raise ValueError("Evidence ID collision")
            seen.add(evidence_id)
            table.append({"id": evidence_id, "source_field_path": path, "target_binding": target,
                          "source_index": source_index, "start": start, "end": end, "text": text[start:end]})
    return table


class _EvidenceIDWire(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def unique_evidence_ids(self):
        for name in ("comment_evidence_ids", "source_evidence_ids"):
            values = getattr(self, name, [])
            if len(values) != len(set(values)):
                raise ValueError("Duplicate evidence IDs")
        return self


def evidence_id_schemas(context):
    table = build_evidence_table(context)
    comment_ids = tuple(entry["id"] for entry in table if entry["source_field_path"] == "comment_text")
    source_ids = tuple(entry["id"] for entry in table if entry["target_binding"] is not None)
    models = []
    for name, canonical in (("EvidenceIDHypothesis", GlobalHypothesis), ("EvidenceIDTarget", TargetAssessment)):
        fields = {}
        for key, field in canonical.model_fields.items():
            if key in ("comment_evidence", "source_evidence"):
                eligible = comment_ids if key == "comment_evidence" else source_ids
                # No string item (or invented sentinel) is eligible for an empty table.
                annotation = list[Literal[eligible]] if eligible else list[None]
                fields[key + "_ids"] = (annotation, Field(default_factory=list, **({} if eligible else {"max_length": 0})))
            else:
                fields[key] = (field.annotation, deepcopy(field))
        models.append(create_model(name, __base__=_EvidenceIDWire, **fields))
    hypothesis, target = models
    revised = create_model("EvidenceIDRevisedAttribution", __base__=_EvidenceIDWire,
                           revised_scope=(hypothesis, ...), targets=(list[target], ...))
    return hypothesis, revised


def resolve_evidence_ids(wire_result, context):
    """Resolve only exact current IDs, never text matches or alternate sources."""
    table = {entry["id"]: entry for entry in build_evidence_table(context)}
    data = wire_result.model_dump()
    records = [data] if "revised_scope" not in data else [data["revised_scope"], *data["targets"]]
    for record in records:
        for field in ("comment_evidence", "source_evidence"):
            key = field + "_ids"
            if key not in record:
                continue
            values = record.pop(key)
            if len(values) != len(set(values)) or field in record:
                raise ValueError("Invalid evidence IDs")
            quotes = []
            for evidence_id in values:
                entry = table.get(evidence_id)
                if entry is None or (field == "comment_evidence" and entry["source_field_path"] != "comment_text") or (
                    field == "source_evidence" and (entry["target_binding"] is None or
                                                   entry["target_binding"] != record.get("key_message_id"))):
                    raise ValueError("Invalid evidence binding")
                quotes.append(entry["text"])
            record[field] = quotes
    return data


def required_targets_schema(ids):
    ids = tuple(ids)
    if not ids or any(not isinstance(value, str) or not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Required targets need unique nonempty IDs")
    # Copy the canonical field definitions, but validate semantic decisions only
    # after restoring IDs in the unchanged canonical model.
    value = create_model("TargetAssessmentValue", __config__=ConfigDict(extra="forbid"), **{
        name: (field.annotation, deepcopy(field))
        for name, field in TargetAssessment.model_fields.items() if name != "key_message_id"
    })
    targets = create_model("RequiredTargets", __config__=ConfigDict(
        extra="forbid", validate_by_alias=True, validate_by_name=False), **{
        f"target_{index}": (value, Field(..., alias=target_id))
        for index, target_id in enumerate(ids)
    })
    return create_model("RequiredTargetsAttributionResult", __config__=ConfigDict(extra="forbid"),
                        revised_scope=(GlobalHypothesis, ...), targets=(targets, ...))


def canonical_revised_result(wire_result, ids):
    data = wire_result.model_dump(by_alias=True)
    return RevisedAttributionResult.model_validate({
        "revised_scope": data["revised_scope"],
        "targets": [{"key_message_id": target_id, **data["targets"][target_id]} for target_id in ids],
    })


def _stage1_quote_diagnostics(quote, context):
    """Describe a rejected quote without changing evidence acceptance or exposing text."""
    reasons = []
    normalized = _normalize_whitespace(quote)
    if any(char.isalnum() for char in normalized):
        fields = {
            "contribution": [context.contribution_text],
            "key_message": [km.text for km in context.key_messages],
            "related_sentence": [sentence for km in context.key_messages for sentence in km.related_sentences],
            "parent": [context.parent_context.parent_comment_text],
        }
        for field, values in fields.items():
            if any(value is not None and normalized in _normalize_whitespace(value) for value in values):
                reasons.append(f"stage1_quote_overlap_{field}")

        # NFC is local to diagnostics. Punctuation separates words; symbols remain
        # meaningful, and lower() deliberately does not equate sharp s with ss.
        quote_nfc = unicodedata.normalize("NFC", normalized)
        comment_nfc = unicodedata.normalize("NFC", _normalize_whitespace(context.comment_text))
        matches = {}
        for mode in ("exact", "case_only", "punctuation_only", "case_and_punctuation"):
            needle, haystack = quote_nfc, comment_nfc
            if mode in ("punctuation_only", "case_and_punctuation"):
                needle, haystack = (
                    _normalize_whitespace("".join(" " if unicodedata.category(c).startswith("P") else c
                                                  for c in text))
                    for text in (needle, haystack)
                )
            if mode in ("case_only", "case_and_punctuation"):
                needle, haystack = needle.lower(), haystack.lower()
            matches[mode] = False
            # Marks and invisible controls are not safe word boundaries, even
            # though Python's \w excludes them. Keep every character intact.
            for match in re.finditer(r"(?=" + re.escape(needle) + r")", haystack) if needle else ():
                start, end = match.start(), match.start() + len(needle)
                neighbors = haystack[max(0, start - 1):start] + haystack[end:end + 1]
                if not any(c.isalnum() or c == "_" or unicodedata.category(c).startswith("M")
                           or unicodedata.category(c) in ("Cc", "Cf") for c in neighbors):
                    matches[mode] = True
                    break
        if not matches["exact"]:
            for mode in ("case_only", "punctuation_only"):
                if matches[mode]:
                    reasons.append(f"stage1_quote_{mode}")
            if matches["case_and_punctuation"] and not (matches["case_only"] or matches["punctuation_only"]):
                reasons.append("stage1_quote_case_and_punctuation")
    return sorted(reasons) or ["stage1_quote_unexplained_text_mismatch"]


def _schema_error_reasons(error: ValidationError, stage: str, target_ids=()) -> list[str]:
    fields = {
        "revised_scope", "targets", "scope_type", "affected_key_message_ids",
        "general_support_extends_to_unaddressed", "comment_evidence", "source_evidence",
        "uncertainty_flags", "explanation", "key_message_id", "attribution", "stance",
        "comment_evidence_ids", "source_evidence_ids",
    }
    types = {
        "missing", "extra_forbidden", "enum", "literal_error", "value_error",
        "model_type", "model_attributes_type", "dict_type", "list_type", "string_type",
        "string_unicode", "bool_type", "bool_parsing",
    }
    reasons = []
    for detail in error.errors(include_url=False, include_context=False, include_input=False):
        kind = detail["type"] if detail["type"] in types else "other"
        location = []
        for index, part in enumerate(detail["loc"]):
            # Extra keys are input, even when they happen to name a known field.
            if kind == "extra_forbidden" and index == len(detail["loc"]) - 1:
                location.append("unknown_field")
            elif type(part) is int and part >= 0:
                location.append(str(part))
            elif index == 1 and detail["loc"][0] == "targets" and part in target_ids:
                location.append(str(target_ids.index(part)))
            else:
                location.append(part if isinstance(part, str) and part in fields else "unknown_field")
        reasons.append(f"{stage}_schema:{kind}:{'.'.join(location) or 'root'}")
    return sorted(set(reasons))


class RevisableAttributionEngine:
    def __init__(self, llm_handler, base_url, model_name, api_key="", enable_thinking=False,
                 prompt_config_path=PROMPT_CONFIG, verifier_enabled=False, revisable_variant="baseline"):
        validate_revisable_variant(revisable_variant, verifier_enabled=verifier_enabled)
        if revisable_variant != "baseline":
            if prompt_config_path != PROMPT_CONFIG:
                raise ValueError("Diagnostic variants require their fixed prompt configuration")
            prompt_config_path = variant_prompt_config(revisable_variant)
        self.revisable_variant = revisable_variant
        self._llm = llm_handler
        self._settings = dict(base_url=base_url, model_name=model_name, api_key=api_key,
                              enable_thinking=enable_thinking)
        detector = StanceDetector()
        self._prompts = [detector.load_prompt_config(prompt_config_path, key)
                         for key in ("global_hypothesis", "revised_attribution")]

    async def resolve_comment(self, client, semaphore, context):
        kms = context.key_messages
        if not kms:
            return []
        ids = [km.key_message_id for km in kms]
        if len(ids) != len(set(ids)):
            raise ValueError("Context contains duplicate target IDs")
        # Only source context goes to the model, never annotation labels or notes.
        data = context.model_dump(exclude={"run_id"})
        hypothesis = None
        wire_hypothesis = None
        evidence_ids = self.revisable_variant == "source_bound_evidence_ids"
        required_targets = self.revisable_variant == "stage2_required_targets"
        second_schema = required_targets_schema(ids) if required_targets else RevisedAttributionResult
        schemas = evidence_id_schemas(context) if evidence_ids else (GlobalHypothesis, second_schema)
        for config, schema in zip(self._prompts, schemas):
            stage = "stage1" if hypothesis is None else "stage2"
            user = Template(config["user_template"]).substitute(
                context_json=json.dumps(data, ensure_ascii=False),
                hypothesis_json=(wire_hypothesis if evidence_ids else hypothesis).model_dump_json() if hypothesis else "null",
                evidence_table_json=json.dumps(build_evidence_table(context), ensure_ascii=False) if evidence_ids else "[]",
            )
            try:
                raw = await self._llm.call_model(
                    client=client, semaphore=semaphore, system=config["system"],
                    messages=[{"role": "user", "content": user}], response_schema=schema,
                    **({"reject_duplicate_json_keys": True} if evidence_ids or required_targets and stage == "stage2" else {}),
                    **({"strict_json": True} if evidence_ids else {}),
                    **self._settings,
                )
            except DuplicateJSONKeyError:
                failure = _needs_review if evidence_ids and stage == "stage2" else _failed
                return [failure(context, km, [f"{stage}_duplicate_json_key"]) for km in kms]
            if raw is None:
                return [_failed(context, km, ["llm_call_failed", f"{stage}_llm_call_failed"]) for km in kms]
            try:
                result = schema.model_validate(raw)
            except ValidationError as error:
                reasons = _schema_error_reasons(error, stage, ids if required_targets else ())
                if hypothesis is None:
                    return [_failed(context, km, ["invalid_hypothesis_schema"] + reasons) for km in kms]
                return [_needs_review(context, km, ["invalid_revised_schema"] + reasons) for km in kms]
            if evidence_ids:
                wire_result = result
                failure = _failed if stage == "stage1" else _needs_review
                try:
                    canonical = resolve_evidence_ids(wire_result, context)
                except ValueError:
                    return [failure(context, km, [f"{stage}_invalid_evidence_binding"]) for km in kms]
                try:
                    result = (GlobalHypothesis if stage == "stage1" else RevisedAttributionResult).model_validate(canonical)
                except ValidationError as error:
                    return [failure(context, km, [f"{stage}_invalid_canonical_schema"] +
                                    _schema_error_reasons(error, stage)) for km in kms]
                if stage == "stage1":
                    wire_hypothesis = wire_result
            if stage == "stage2" and required_targets:
                try:
                    result = canonical_revised_result(result, ids)
                except ValidationError as error:
                    reasons = _schema_error_reasons(error, stage)
                    return [_needs_review(context, km, ["invalid_revised_schema"] + reasons) for km in kms]
            if hypothesis is None:
                errors = []
                for quote in result.comment_evidence:
                    if not quote.strip():
                        errors.append("stage1_empty_scope_evidence")
                    elif _normalize_whitespace(quote) not in _normalize_whitespace(context.comment_text):
                        errors.append("stage1_nonmatching_scope_evidence")
                        errors.extend(_stage1_quote_diagnostics(quote, context))
                if not set(result.affected_key_message_ids) <= set(ids):
                    errors.append("stage1_unknown_scope_target")
                if len(result.affected_key_message_ids) != len(set(result.affected_key_message_ids)):
                    errors.append("stage1_duplicate_scope_target")
                support = result.scope_type == ScopeType.WHOLE_CONTRIBUTION_APPROVAL or (
                    result.scope_type == ScopeType.MIXED_QUALIFIED
                    and result.general_support_extends_to_unaddressed is True
                )
                if support and not result.comment_evidence:
                    errors.append("stage1_general_support_missing_evidence")
                if errors:
                    return [_failed(context, km, sorted(set(errors))) for km in kms]
                hypothesis = result

        scope = result.revised_scope
        errors = []
        target_ids = [target.key_message_id for target in result.targets]
        if len(target_ids) != len(set(target_ids)) or set(target_ids) != set(ids):
            errors.append("invalid_exact_target_coverage")
        if set(ids) - set(target_ids):
            errors.append("missing_target_ids")
        if len(target_ids) != len(set(target_ids)):
            errors.append("duplicate_target_ids")
        if set(target_ids) - set(ids):
            errors.append("unknown_target_ids")
        if not set(scope.affected_key_message_ids) <= set(ids):
            errors.append("unknown_scope_target")
        if len(scope.affected_key_message_ids) != len(set(scope.affected_key_message_ids)):
            errors.append("duplicate_scope_target")
        general_support = scope.scope_type == ScopeType.WHOLE_CONTRIBUTION_APPROVAL or (
            scope.scope_type == ScopeType.MIXED_QUALIFIED and scope.general_support_extends_to_unaddressed is True
        )
        if general_support and not scope.comment_evidence:
            errors.append("general_support_missing_evidence")
        comment = _normalize_whitespace(context.comment_text)
        for quote in scope.comment_evidence:
            if not quote.strip() or _normalize_whitespace(quote) not in comment:
                errors.append("invalid_scope_evidence")
                errors.append("empty_scope_evidence" if not quote.strip() else "nonmatching_scope_evidence")
        by_id = {km.key_message_id: km for km in kms}
        for target in result.targets:
            for quote in target.comment_evidence:
                if not quote.strip() or _normalize_whitespace(quote) not in comment:
                    errors.append("invalid_comment_evidence")
                    errors.append("empty_comment_evidence" if not quote.strip() else "nonmatching_comment_evidence")
            km = by_id.get(target.key_message_id)
            for quote in target.source_evidence:
                if not quote.strip() or km is None or not any(
                    _normalize_whitespace(quote) in _normalize_whitespace(sentence)
                    for sentence in km.related_sentences
                ):
                    errors.append("invalid_target_source_evidence")
                    errors.append("empty_target_source_evidence" if not quote.strip() else
                                  "unknown_target_source_evidence" if km is None else
                                  "nonmatching_target_source_evidence")
        if errors:
            return [_needs_review(context, km, sorted(set(errors))) for km in kms]

        targets = {target.key_message_id: target for target in result.targets}
        outputs = []
        for km in kms:
            target = targets[km.key_message_id]
            basis = target.attribution
            evidence = target.comment_evidence
            stance = target.stance
            explanation = target.explanation
            if basis in ("unaddressed", "no_position"):
                stance = StanceLabelV2.NEUTRAL
                if general_support:
                    stance = GENERAL_APPROVAL_STANCE
                    basis, evidence = "general_approval", scope.comment_evidence
                    explanation = scope.explanation.strip() or (
                        "Allgemeine Zustimmung zum gesamten Beitrag, kein eigener Stance zu dieser Kernaussage."
                    )
            elif (basis == "explicit_target" and stance == StanceLabelV2.IN_FAVOR and general_support
                  and restates_general_approval(evidence, scope.comment_evidence)):
                # Support quoting only the comment-wide approval is inherited,
                # not target-specific: general-approval provenance and stance.
                stance = GENERAL_APPROVAL_STANCE
                basis, evidence = "general_approval", scope.comment_evidence
                explanation = scope.explanation.strip() or (
                    "Allgemeine Zustimmung zum gesamten Beitrag, kein eigener Stance zu dieser Kernaussage."
                )
            reasons = [flag if flag in {"unresolved_reply_reference", "suspected_sarcasm"}
                       else f"{origin}_uncertainty_flag:{index}"
                       for origin, flags in (("scope", scope.uncertainty_flags), ("target", target.uncertainty_flags))
                       for index, flag in enumerate(flags)]
            if scope.scope_type == ScopeType.UNRESOLVED:
                reasons = reasons + ["unresolved_scope"]
            if target.attribution == "unresolved":
                reasons = reasons + ["unresolved_attribution"]
            if reasons or stance is None:
                output = _needs_review(context, km, sorted(set(reasons or ["missing_stance"])))
                output.proposed_stance = stance.value if stance is not None else None
                output.basis = basis
                output.comment_evidence = evidence
                output.source_evidence = target.source_evidence
                output.explanation = explanation
            else:
                output = _accepted(context, km, stance, basis, evidence,
                                   target.source_evidence, explanation)
            outputs.append(output)
        return outputs
