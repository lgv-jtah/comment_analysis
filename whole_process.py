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
    "contributionContent",
    "key_message",
    "key_message_type",
    "related_sentences",
    "related_sentence_spans",
    "key_message_explanation",
    "comment_text",
    "stance",
    "explanation",
    "extraction_model_name",
    "stance_model_name",
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


def find_sentence_spans(
    text: str, sentences: list[str]
) -> list[list[tuple[int, int]] | None]:
    """Locate verbatim character spans of each sentence within ``text``.

    For each sentence, finds every non-overlapping verbatim occurrence in
    ``text`` and records its half-open ``(start, end)`` character span such that
    ``text[start:end] == sentence``.

    Args:
        text: The source text to search within.
        sentences: Candidate sentence strings to locate, in order.

    Returns:
        One element per input sentence, positionally aligned:
        - A non-empty ``list[tuple[int, int]]`` of half-open spans when the
          sentence occurs at least once.
        - ``None`` when the sentence is empty or is not found verbatim. The
          empty-string case returns ``None`` to avoid the ``''.find('') == 0``
          infinite-loop hazard.
    """
    result: list[list[tuple[int, int]] | None] = []
    for sentence in sentences:
        # Skip empty sentences: ''.find('') returns 0 (a match at every
        # position), which would make the str.find loop below never advance and
        # spin forever. Map this case to None instead.
        if not sentence:
            result.append(None)
            continue
        spans: list[tuple[int, int]] = []
        pos = text.find(sentence)
        while pos != -1:
            spans.append((pos, pos + len(sentence)))
            pos = text.find(sentence, pos + len(sentence))
        result.append(spans if spans else None)
    return result


class WholeProcess:
    """Run key-message extraction and gated stance detection end to end."""

    def __init__(
        self,
        extraction_base_url: str,
        extraction_model_name: str,
        stance_base_url: str,
        stance_model_name: str,
        extraction_api_key: str = "",
        stance_api_key: str = "",
        prompt_config_path: str = "./prompts/config.json",
        prompt_key: str = "base",
        model_path: Path = MODEL_PATH,
    ) -> None:
        """Initialize the pipeline and load the stance prompt config.

        Key-message extraction and stance detection use independently configured
        LLMs, so different models can be compared across the two roles. The BERT
        reaction classifier is NOT loaded here (it is lazily loaded in
        :meth:`run`) so constructing a WholeProcess stays cheap and test-friendly.

        Args:
            extraction_base_url: vLLM/OpenAI-compatible endpoint base URL for the
                key-message extraction model.
            extraction_model_name: Model identifier used for key-message
                extraction.
            stance_base_url: vLLM/OpenAI-compatible endpoint base URL for the
                stance-detection model.
            stance_model_name: Model identifier used for stance resolution.
            extraction_api_key: API key forwarded to the key-message extraction
                LLM calls. Defaults to an empty string so the pipeline can be
                constructed without a secret (e.g. in tests); callers thread the
                real key in from the environment.
            stance_api_key: API key forwarded to the stance-resolution LLM calls.
                Defaults to an empty string for the same reason.
            prompt_config_path: Path to the stance prompt config JSON.
            prompt_key: Key within the config selecting the stance prompt.
            model_path: Directory holding the fine-tuned classifier weights.
        """
        self.extraction_base_url = extraction_base_url
        self.extraction_model_name = extraction_model_name
        self._extraction_api_key = extraction_api_key
        self.stance_base_url = stance_base_url
        self.stance_model_name = stance_model_name
        self._stance_api_key = stance_api_key
        self._model_path = model_path
        self._stance_detector = StanceDetector()
        self._llm_handler = LLMHandler()
        self._prompt_config = self._stance_detector.load_prompt_config(
            prompt_config_path, prompt_key
        )
        self._extractor = KeyMessageExtraction(
            base_url=extraction_base_url,
            model_name=extraction_model_name,
            api_key=extraction_api_key,
        )
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
            base_url=self.stance_base_url,
            model_name=self.stance_model_name,
            system=system,
            messages=messages,
            semaphore=semaphore,
            enable_thinking=True,
            api_key=self._stance_api_key,
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
            produced. Each row carries the source contribution text in
            ``contributionContent`` and the key-message rationale in
            ``key_message_explanation``, alongside the contribution id, key
            message, key-message type, comment, stance, and stance explanation.
            Each row additionally carries ``related_sentences`` — the verbatim
            source sentences for the key message (a ``list[str]``, stored
            natively in the parquet output). This is untrusted model output and
            is defensively coerced to a list of strings. Each row also carries
            ``related_sentence_spans`` — the half-open character spans of those
            sentences within ``contributionContent`` (one element per related
            sentence, ``None`` where a sentence is empty or not found verbatim).
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
                    # Verbatim source sentences for this key message. This is
                    # untrusted model output, so coerce to a list of strings and
                    # default to an empty list if absent or malformed.
                    raw_rs = km.get("related_sentences", [])
                    related_sentences = (
                        [str(s) for s in raw_rs] if isinstance(raw_rs, list) else []
                    )
                    # Character-level spans of each related sentence within the
                    # contribution text. Computed once per key message and reused
                    # across every comment row below.
                    sentence_spans = find_sentence_spans(
                        contribution_text, related_sentences
                    )
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
                                "contributionContent": contribution_text,
                                "key_message": km_text,
                                "key_message_type": km_type,
                                "related_sentences": related_sentences,
                                "related_sentence_spans": sentence_spans,
                                "key_message_explanation": str(km.get("explanation", "")),
                                "comment_text": comment_str,
                                "stance": stance,
                                "explanation": explanation,
                                "extraction_model_name": self.extraction_model_name,
                                "stance_model_name": self.stance_model_name,
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
