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
        api_key: str = "",
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
            api_key: API key forwarded to every LLM call (both key-message
                extraction and stance resolution). Defaults to an empty string so
                the pipeline can be constructed without a secret (e.g. in tests);
                callers thread the real key in from the environment.
            prompt_config_path: Path to the stance prompt config JSON.
            prompt_key: Key within the config selecting the stance prompt.
            model_path: Directory holding the fine-tuned classifier weights.
        """
        self.base_url = base_url
        self.model_name = model_name
        self._api_key = api_key
        self._model_path = model_path
        self._stance_detector = StanceDetector()
        self._llm_handler = LLMHandler()
        self._prompt_config = self._stance_detector.load_prompt_config(
            prompt_config_path, prompt_key
        )
        self._extractor = KeyMessageExtraction(base_url=base_url, model_name=model_name, api_key=api_key)
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
    ) -> tuple[str | None, str | None, str]:
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
            A ``(stance, explanation, path)`` tuple. ``stance`` and
            ``explanation`` may each be ``None`` if the LLM call failed. ``path``
            is the routing path taken: ``"fast_path"``, ``"ensemble"``, or
            ``"llm"``.
        """
        if should_fast_path(gr_label, gr_score, comment_text):
            return FAST_PATH_STANCE, FAST_PATH_EXPLANATION, "fast_path"

        item = {"comment_text": comment_text, "key_message": key_message}
        if should_ensemble_path(gr_label, gr_score, comment_text):
            hint = self._stance_detector.format_gr_hint(gr_score)
            system, messages = self._stance_detector.build_messages(
                self._prompt_config, item, gr_hint=hint
            )
            path = "ensemble"
        else:
            system, messages = self._stance_detector.build_messages(
                self._prompt_config, item
            )
            path = "llm"

        result = await self._llm_handler.call_model(
            client=client,
            base_url=self.base_url,
            model_name=self.model_name,
            system=system,
            messages=messages,
            semaphore=semaphore,
            enable_thinking=True,
            api_key=self._api_key,
        )
        if result:
            return result.get("stance"), result.get("explanation"), path
        return None, None, path

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

        Emits detailed per-step console output: for each contribution it prints
        a header, the key-message extraction time, and then — for every (key
        message x comment) pair — the routing path taken, the per-pair elapsed
        time, and the resolved stance. A per-contribution row count and elapsed
        time follow each contribution, and a grand total is printed at the end.

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
                contrib_start = time.time()
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

                console.print(
                    f"[bold][{contrib_idx}/{total}] Contribution {contribution_id}[/bold]"
                )

                # Step 1: key message extraction with timer.
                km_start = time.time()
                key_messages = await self._extractor.extract(
                    client=client,
                    contribution=contribution_text,
                    proceeding_description=proceeding_description,
                    semaphore=semaphore,
                )
                km_elapsed = time.time() - km_start
                if not key_messages:
                    console.print(
                        f"  [yellow]↳ Key message extraction: no messages "
                        f"({km_elapsed:.2f}s) — skipping[/yellow]"
                    )
                    continue
                console.print(
                    f"  [cyan]↳ Key message extraction: {len(key_messages)} messages "
                    f"({km_elapsed:.2f}s)[/cyan]"
                )

                rows_added = 0
                for km_idx, km in enumerate(key_messages, start=1):
                    km_text = str(km.get("key_message", ""))
                    km_type = str(km.get("key_message_type", ""))
                    # Truncate long key message text for display only.
                    display_km = km_text[:60] + "…" if len(km_text) > 60 else km_text
                    console.print(
                        f"  [cyan]↳ KM {km_idx}/{len(key_messages)} "
                        f"\"{display_km}\"[/cyan]"
                    )

                    for comment_idx, comment_text in enumerate(comments, start=1):
                        comment_str = str(comment_text)
                        # Per-comment: classify then resolve stance, timed together.
                        step_start = time.time()
                        gr_label, gr_score = self._classify_comment(comment_str)
                        stance, explanation, path = await self._resolve_stance(
                            client, gr_label, gr_score, comment_str, km_text, semaphore
                        )
                        step_elapsed = time.time() - step_start

                        # Color and label by path.
                        path_colors = {
                            "fast_path": "blue",
                            "ensemble": "magenta",
                            "llm": "white",
                        }
                        path_labels = {
                            "fast_path": "FAST-PATH",
                            "ensemble": "ENSEMBLE",
                            "llm": "LLM",
                        }
                        color = path_colors.get(path, "white")
                        label = path_labels.get(path, path.upper())
                        stance_display = stance if stance is not None else "FAILED"

                        console.print(
                            f"      Comment {comment_idx}/{len(comments)}: "
                            f"[{color}][{label}][/{color}]  "
                            f"({step_elapsed:.2f}s) → {stance_display}"
                        )

                        rows.append(
                            {
                                "contribution_id": contribution_id,
                                "key_message": km_text,
                                "key_message_type": km_type,
                                "comment_text": comment_str,
                                "stance": stance,
                                "explanation": explanation,
                            }
                        )
                        rows_added += 1

                contrib_elapsed = time.time() - contrib_start
                console.print(
                    f"  [green]→ {rows_added} rows  |  contribution: "
                    f"{contrib_elapsed:.2f}s[/green]"
                )

        total_elapsed = time.time() - total_start
        console.print(
            f"[bold green]Total: {len(rows)} rows in {total_elapsed:.2f}s[/bold green]"
        )

        return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
