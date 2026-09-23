"""Runs the policy v2 engine over an already-extracted source and scores it.

Deliberately does NOT re-run key-message extraction.
STANCE_PIPELINE_IMPLEMENTATION_BRIEF.md section 7: "Keep extraction fixed
during stance experiments." This session's own empirical check confirmed
why: re-extracting produces different key-message text (an LLM call is not
byte-reproducible), which changes dataset_builder.compute_pair_id's hash and
makes fresh predictions unmatchable against the historical reference dataset
built from the original extraction -- a live check found 23 of 28 fresh
predictions from WholeProcess.run_v2() (which re-extracts) had no match in
dataset/pairs.parquet at all.

So this module reads key messages (and their related_sentences) straight
from an already-extracted source (the same row shape
whole_process.OUTPUT_COLUMNS describes: one row per (key_message, comment)
pair, carrying contributionContent/key_message/related_sentences/
comment_text), groups it back into one entry per comment, builds one
CommentContext per comment from that frozen data, and calls
StancePolicyEngineV2.resolve_comment() directly -- never
WholeProcess.run_v2(), which always re-extracts.
"""
from __future__ import annotations

import asyncio
import time
from typing import Literal

import httpx
import pandas as pd

from context_builder import build_comment_context
from ionos_config import verify_ionos_only
from llm_handler import LLMHandler
from policy import OneCallPolicyEngineV2, StancePolicyEngineV2
from whole_process import OUTPUT_COLUMNS_V2

Mode = Literal["two_stage", "one_call"]


def load_frozen_comments(source: pd.DataFrame) -> list[dict]:
    """Group a whole_process-shaped source into one entry per comment.

    Args:
        source: A DataFrame shaped like whole_process.OUTPUT_COLUMNS: one
            row per (key_message, comment) pair, with contribution_id,
            contributionContent, key_message, key_message_type,
            related_sentences, and comment_text columns.

    Returns:
        One dict per (contribution_id, comment_text), each with
        contribution_id (str), contribution_text (str or None),
        comment_text, and key_messages (a list of {key_message,
        key_message_type, related_sentences} dicts, de-duplicated by
        key_message text, in first-seen order).
    """
    comments = []
    for (contribution_id, comment_text), group in source.groupby(
        ["contribution_id", "comment_text"], sort=False
    ):
        first = group.iloc[0]
        contribution_text = first.get("contributionContent")
        key_messages = []
        seen_km_texts: set[str] = set()
        for _, row in group.iterrows():
            km_text = row["key_message"]
            if km_text in seen_km_texts:
                continue
            seen_km_texts.add(km_text)
            related = row.get("related_sentences")
            key_messages.append(
                {
                    "key_message": km_text,
                    "key_message_type": row.get("key_message_type", ""),
                    "related_sentences": list(related) if related is not None else [],
                }
            )
        comments.append(
            {
                "contribution_id": str(contribution_id),
                "contribution_text": (
                    str(contribution_text) if pd.notna(contribution_text) else None
                ),
                "comment_text": comment_text,
                "key_messages": key_messages,
            }
        )
    return comments


async def run_benchmark(
    *,
    source: pd.DataFrame,
    base_url: str,
    model_name: str,
    api_key: str = "",
    enable_thinking: bool = False,
    verifier_enabled: bool = False,
    verify_ionos: bool = True,
    run_id: str | None = None,
    limit: int | None = None,
    mode: Mode = "two_stage",
) -> pd.DataFrame:
    """Run the policy v2 engine over a frozen, already-extracted source.

    Args:
        source: whole_process-shaped rows; see load_frozen_comments.
        base_url: LLM endpoint base URL.
        model_name: Model identifier.
        api_key: Bearer token for the endpoint.
        enable_thinking: Forwarded to LLMHandler.call_model.
        verifier_enabled: Forwarded to StancePolicyEngineV2; ignored when
            mode == "one_call" (OneCallPolicyEngineV2 has no verifier stage).
        verify_ionos: When True (default), fail fast via
            ionos_config.verify_ionos_only() before doing any work.
        run_id: Identifier recorded on every output row; auto-generated
            from the current time when None.
        limit: Optional cap on the number of comments processed.
        mode: "two_stage" (StancePolicyEngineV2, the default) or "one_call"
            (OneCallPolicyEngineV2) -- brief section 10's experiment matrix
            compares both against the same frozen source/reference.

    Returns:
        A DataFrame with whole_process.OUTPUT_COLUMNS_V2 columns, one row
        per (comment, key_message) pair.

    Raises:
        ionos_config.IonosVerificationError: If verify_ionos is True and
            model_name isn't IONOS-routed.
        ValueError: If mode is neither "two_stage" nor "one_call".
    """
    if verify_ionos:
        verify_ionos_only(model_name, base_url, api_key)

    run_id = run_id or f"benchmark_{int(time.time())}"
    if mode == "two_stage":
        engine = StancePolicyEngineV2(
            llm_handler=LLMHandler(),
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            enable_thinking=enable_thinking,
            verifier_enabled=verifier_enabled,
        )
    elif mode == "one_call":
        engine = OneCallPolicyEngineV2(
            llm_handler=LLMHandler(),
            base_url=base_url,
            model_name=model_name,
            api_key=api_key,
            enable_thinking=enable_thinking,
        )
    else:
        raise ValueError(f"Unknown mode {mode!r}; expected 'two_stage' or 'one_call'.")

    comments = load_frozen_comments(source)
    if limit is not None:
        comments = comments[:limit]

    semaphore = asyncio.Semaphore(1)
    rows: list[dict] = []
    async with httpx.AsyncClient() as client:
        for entry in comments:
            context = build_comment_context(
                run_id=run_id,
                contribution_id=entry["contribution_id"],
                comment_text=entry["comment_text"],
                key_messages=entry["key_messages"],
                contribution_text=entry["contribution_text"],
            )
            results = await engine.resolve_comment(client, semaphore, context)
            for pair_output, km_ref in zip(results, context.key_messages):
                rows.append(
                    {
                        "contribution_id": entry["contribution_id"],
                        "contributionContent": entry["contribution_text"],
                        "key_message_id": km_ref.key_message_id,
                        "key_message": km_ref.text,
                        "comment_text": entry["comment_text"],
                        "pair_id": pair_output.pair_id,
                        "status": pair_output.status,
                        "proposed_stance": pair_output.proposed_stance,
                        "final_stance": pair_output.final_stance,
                        "basis": pair_output.basis,
                        "comment_evidence": pair_output.comment_evidence,
                        "source_evidence": pair_output.source_evidence,
                        "explanation": pair_output.explanation,
                        "review_reasons": pair_output.review_reasons,
                        "policy_version": pair_output.policy_version,
                        "run_id": pair_output.run_id,
                        "extraction_model_name": None,
                        "stance_model_name": model_name,
                    }
                )

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS_V2)
