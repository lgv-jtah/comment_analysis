"""End-to-end "Whole Process" pipeline.

Joins key-message extraction with the general-reaction gated stance detection
flow over a parquet file of contributions and their comments, producing one
flattened row per (key message x comment) pair.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from rich.console import Console

from classifier import ReactionClassifier
from key_message_extractor import KeyMessageExtraction
from llm_handler import LLMHandler
from stance_detector import StanceDetector

console = Console()

# Resolve relative to this file (not the CWD) so the path is stable no matter
# which directory the app is launched from. Mirrors main.MODEL_PATH wiring.
MODEL_PATH = Path(__file__).resolve().parent / "model_output"

# NOTE: The fast-path gate constants and the should_* routing functions below
# are intentionally duplicated from main.py and MUST stay in sync with it. They
# live here too so the WholeProcess pipeline does not import from the
# interactive main module (which would pull in the whole menu/CLI surface).
GR_SCORE_THRESHOLD = 0.6
FAST_PATH_MAX_WORDS = 5
FAST_PATH_STANCE = "neutral"
FAST_PATH_EXPLANATION = "Allgemeine positive Reaktion — per KM neutral, Kommentar-Level positiv"

OUTPUT_COLUMNS = [
    "contribution_id",
    "key_message",
    "key_message_type",
    "comment_text",
    "stance",
    "explanation",
]


def should_fast_path(label: str, score: float, text: str) -> bool:
    """Decide whether a comment can skip the LLM via the fast path.

    SYNC NOTE: mirrors main.should_fast_path — keep both in sync.

    Args:
        label: General-reaction classifier label; only "ja" can qualify.
        score: Confidence of that label, in [0.0, 1.0].
        text: The raw comment text; word count gates short comments.

    Returns:
        True if the comment qualifies for the fast path, False otherwise.
    """
    return label == "ja" and score > GR_SCORE_THRESHOLD and len(text.split()) <= FAST_PATH_MAX_WORDS


def should_ensemble_path(label: str, score: float, text: str) -> bool:
    """Decide whether a comment takes the ensemble path.

    SYNC NOTE: mirrors main.should_ensemble_path — keep both in sync.

    Args:
        label: General-reaction classifier label; only "ja" can qualify.
        score: Confidence of that label, in [0.0, 1.0].
        text: The raw comment text; word count gates long comments.

    Returns:
        True if the comment qualifies for the ensemble path, False otherwise.
    """
    return label == "ja" and score > GR_SCORE_THRESHOLD and len(text.split()) > FAST_PATH_MAX_WORDS


class WholeProcess:
    """Run key-message extraction and gated stance detection end to end."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        prompt_config_path: str = "./prompts/config.json",
        prompt_key: str = "base",
        model_path: Path = MODEL_PATH,
    ) -> None:
        """Initialize the pipeline and load the stance prompt config.

        The BERT reaction classifier is NOT loaded here (it is lazily loaded in
        :meth:`run`) so constructing a WholeProcess stays cheap and test-friendly.

        Args:
            base_url: vLLM/OpenAI-compatible endpoint base URL.
            model_name: Model identifier passed to the server.
            prompt_config_path: Path to the stance prompt config JSON.
            prompt_key: Key within the config selecting the stance prompt.
            model_path: Directory holding the fine-tuned classifier weights.
        """
        self.base_url = base_url
        self.model_name = model_name
        self._model_path = model_path
        self._stance_detector = StanceDetector()
        self._llm_handler = LLMHandler()
        self._prompt_config = self._stance_detector.load_prompt_config(
            prompt_config_path, prompt_key
        )
        self._extractor = KeyMessageExtraction(base_url=base_url, model_name=model_name)
        self._classifier = ReactionClassifier()

    def _classify_comment(self, comment_text: str) -> tuple[str, float]:
        """Classify a comment with the general-reaction classifier.

        Args:
            comment_text: Raw comment text.

        Returns:
            A ``(label, score)`` tuple from the classifier result.
        """
        result = self._classifier.classify_comment(comment_text)
        return result["label"], result["score"]

    async def _resolve_stance(
        self,
        client: httpx.AsyncClient,
        gr_label: str,
        gr_score: float,
        comment_text: str,
        key_message: str,
        semaphore: asyncio.Semaphore,
    ) -> tuple[str | None, str | None]:
        """Resolve the stance for one comment against one key message.

        Routes through the fast path (canned neutral result), the ensemble path
        (LLM call with the general-reaction hint), or the plain LLM path.

        Args:
            client: Async HTTP client for the LLM request.
            gr_label: General-reaction classifier label.
            gr_score: General-reaction confidence score.
            comment_text: Raw comment text.
            key_message: The key message the comment is evaluated against.
            semaphore: Concurrency guard for the LLM call.

        Returns:
            A ``(stance, explanation)`` tuple; either element may be ``None`` if
            the LLM call failed.
        """
        if should_fast_path(gr_label, gr_score, comment_text):
            return FAST_PATH_STANCE, FAST_PATH_EXPLANATION

        item = {"comment_text": comment_text, "key_message": key_message}
        if should_ensemble_path(gr_label, gr_score, comment_text):
            hint = self._stance_detector.format_gr_hint(gr_score)
            system, messages = self._stance_detector.build_messages(
                self._prompt_config, item, gr_hint=hint
            )
        else:
            system, messages = self._stance_detector.build_messages(
                self._prompt_config, item
            )

        result = await self._llm_handler.call_model(
            client=client,
            base_url=self.base_url,
            model_name=self.model_name,
            system=system,
            messages=messages,
            semaphore=semaphore,
            enable_thinking=True,
        )
        if result:
            return result.get("stance"), result.get("explanation")
        return None, None

    async def run(
        self, df: pd.DataFrame, proceeding_description: str = ""
    ) -> pd.DataFrame:
        """Run the whole pipeline over a contributions DataFrame.

        For each contribution, extracts its key messages and then, for every
        (key message x comment) pair, classifies the comment and resolves a
        stance. Returns one flattened row per pair.

        Args:
            df: Contributions DataFrame with ``contributionContent`` and
                ``comments`` columns (and optionally ``contribution_id``).
            proceeding_description: Optional planning-proceeding context passed
                to the key-message extractor.

        Returns:
            A DataFrame with :data:`OUTPUT_COLUMNS`; empty if no rows are
            produced.
        """
        # Lazy load: defer the expensive BERT model load until run time.
        self._classifier.load_classifier(str(self._model_path))

        semaphore = asyncio.Semaphore(1)
        rows: list[dict] = []
        total = len(df)
        total_start = time.time()

        async with httpx.AsyncClient() as client:
            for contrib_idx, (_, row) in enumerate(df.iterrows(), start=1):
                # Prefer the stable contribution_id when present and scalar; fall
                # back to the loop index so every row still gets an identifier.
                contribution_id = (
                    int(row["contribution_id"])
                    if "contribution_id" in df.columns
                    and np.ndim(row["contribution_id"]) == 0
                    and not pd.isna(row["contribution_id"])
                    else contrib_idx
                )
                # Treat a NaN scalar contribution as an empty string so the
                # extractor never receives a float NaN.
                contribution_text = (
                    ""
                    if np.ndim(row["contributionContent"]) == 0
                    and pd.isna(row["contributionContent"])
                    else str(row["contributionContent"])
                )
                # Comments may be missing, a NaN scalar, or an array-like.
                raw_comments = row.get("comments", None)
                comments = (
                    []
                    if raw_comments is None
                    or (np.ndim(raw_comments) == 0 and pd.isna(raw_comments))
                    else list(raw_comments)
                )

                key_messages = await self._extractor.extract(
                    client=client,
                    contribution=contribution_text,
                    proceeding_description=proceeding_description,
                    semaphore=semaphore,
                )
                if not key_messages:
                    console.print(
                        f"[yellow][{contrib_idx}/{total}] contribution {contribution_id}: "
                        f"no key messages extracted — skipping[/yellow]"
                    )
                    continue

                rows_added = 0
                for km in key_messages:
                    for comment_text in comments:
                        gr_label, gr_score = self._classify_comment(str(comment_text))
                        stance, explanation = await self._resolve_stance(
                            client,
                            gr_label,
                            gr_score,
                            str(comment_text),
                            str(km.get("key_message", "")),
                            semaphore,
                        )
                        rows.append(
                            {
                                "contribution_id": contribution_id,
                                "key_message": str(km.get("key_message", "")),
                                "key_message_type": str(km.get("key_message_type", "")),
                                "comment_text": str(comment_text),
                                "stance": stance,
                                "explanation": explanation,
                            }
                        )
                        rows_added += 1

                console.print(
                    f"[{contrib_idx}/{total}] contribution {contribution_id}: "
                    f"{len(key_messages)} key messages, {len(comments)} comments "
                    f"→ {rows_added} rows"
                )

        total_elapsed = time.time() - total_start
        console.print(f"Total whole-process time: {total_elapsed:.2f}s")

        return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
