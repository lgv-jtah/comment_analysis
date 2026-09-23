"""Stance-detection entry point with the general-reaction fast-path gate."""
from stance_detector import StanceDetector
from llm_handler import LLMHandler
from evaluator import (
    StanceEvaluator,
    RunMetadata,
    write_markdown_report,
    write_result_json,
    JudgeComparisonRow,
    write_judge_comparison_report,
)
import httpx
import asyncio
import time
import urllib.parse
import requests
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markup import escape
from rich.table import Table
from create_dataset import CreateDataset, API_URL
import pandas as pd
from sample_contributions import SampleContributions
from key_message_extractor import KeyMessageExtraction
from key_message_judge import KeyMessageJudge
from comment_fetcher import CommentFetcher
from contributions_with_comments_dataset import ContributionsWithCommentsDataset
from whole_process import WholeProcess
from dotenv import load_dotenv
from collections import defaultdict
import os
import re

load_dotenv()
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
# The judge feature authenticates against its own gateway (see JUDGE_LLM_*
# below), so it needs a credential distinct from the shared LLM_API_KEY.
JUDGE_LLM_API_KEY = os.getenv("JUDGE_LLM_API_KEY", "")

console = Console()

# Resolve relative to this file (not the CWD) so paths are stable no matter
# which directory the app is launched from.
DATA_DIR = Path(__file__).resolve().parent / "data"
CONTRIBUTIONS_PATH = DATA_DIR / "contributions.parquet"

# LLM endpoint configuration, shared by stance detection and key-message
# extraction. Kept at module level so both flows reference the same values.
# LLM_BASE_URL = "http://217.160.202.33:5555/v1"
# LLM_BASE_URL = "http://10.200.160.182:5555/v1"
LLM_BASE_URL = "https://api.lyceum.technology/api/v2/external/serverless"
# LLM_MODEL_NAME = "Qwen/Qwen3.5-27B-FP8"
LLM_MODEL_NAME = "Qwen/Qwen3-32B"

# Per-role model configuration. Key-message extraction and stance detection can
# each target an independently configured LLM so different models can be
# compared across roles. Each falls back to the shared LLM_* value above when its
# dedicated environment variable is unset, so existing .env setups keep working.
EXTRACTION_LLM_BASE_URL = os.getenv("EXTRACTION_LLM_BASE_URL", LLM_BASE_URL)
EXTRACTION_LLM_MODEL_NAME = os.getenv("EXTRACTION_LLM_MODEL_NAME", LLM_MODEL_NAME)
EXTRACTION_LLM_API_KEY = os.getenv("EXTRACTION_LLM_API_KEY", LLM_API_KEY)
STANCE_LLM_BASE_URL = os.getenv("STANCE_LLM_BASE_URL", LLM_BASE_URL)
STANCE_LLM_MODEL_NAME = os.getenv("STANCE_LLM_MODEL_NAME", LLM_MODEL_NAME)
STANCE_LLM_API_KEY = os.getenv("STANCE_LLM_API_KEY", LLM_API_KEY)
# Whether stance detection requests vLLM/Qwen's "thinking mode" extension
# (chat_template_kwargs.enable_thinking). Default true preserves the original
# vLLM/Qwen behaviour; set false for OpenAI-compatible backends/proxies (e.g.
# LiteLLM routing to non-vLLM models) that reject that vLLM-specific field.
STANCE_LLM_ENABLE_THINKING = os.getenv(
    "STANCE_LLM_ENABLE_THINKING", "true"
).strip().lower() not in {"false", "0", "no"}

# Judge-specific endpoint: the LLM-as-a-judge feature calls Claude via an
# OpenAI-compatible gateway, not the self-hosted Qwen vLLM endpoint above.
# Unlike LLM_BASE_URL, this is sourced from the environment (not hardcoded)
# because the gateway URL varies by deployment.
JUDGE_LLM_BASE_URL = os.getenv("JUDGE_LLM_BASE_URL", "")
JUDGE_LLM_MODEL_NAME = os.getenv("JUDGE_LLM_MODEL_NAME", "claude-opus-4-8")

# Fast-path gate thresholds (see general_reaction_gate_konzept.md, Task 1.3).
# NOTE: 0.6 is an interim placeholder, NOT a tuned production value. The
# calibration against gold labels is Task 1.1 in the concept doc; until that
# lands, do not treat this number as final.
GR_SCORE_THRESHOLD = 0.6
FAST_PATH_MAX_WORDS = 5
# The fast-path stance must be a value from the lowercase gold-label vocabulary
# ("neutral"), NOT a free-form phrase — the evaluator matches predictions against
# gold labels by exact string. The German explanation reads "general positive
# reaction — neutral per machine-classification (KM), positive at comment level":
# the per-row stance label is deliberately "neutral" even though the comment-level
# signal is positive, so the two values intentionally differ.
FAST_PATH_STANCE = "neutral"
FAST_PATH_EXPLANATION = "Allgemeine positive Reaktion — per KM neutral, Kommentar-Level positiv"

# DEBUG / TEMPORARY — for manual test runs only. Remove before any full
# production run: this caps the loop at the first few rows instead of all 155.
DEBUG_ROW_LIMIT = 10

# Maximum number of characters shown per table cell when inspecting a parquet
# file; longer values are truncated with an ellipsis so the table stays legible.
MAX_CELL_CHARS = 80


def should_fast_path(
    general_reaction_label: str, general_reaction_score: float, comment_text: str
) -> bool:
    """Decide whether a comment can skip the LLM via the fast path.

    Implements "Pfad 1 — Fast Path" from general_reaction_gate_konzept.md
    (Task 1.3): a short comment that the upstream general-reaction classifier
    confidently labelled as a positive reaction is assumed neutral per-KM
    (positive signal at comment level) without a dedicated stance-detection
    LLM call.

    Args:
        general_reaction_label: Label from the general-reaction classifier;
            only "ja" (yes, a general reaction) can qualify.
        general_reaction_score: Confidence of that label, in [0.0, 1.0].
        comment_text: The raw comment text; word count gates short comments.

    Returns:
        True if the comment qualifies for the fast path, False otherwise.
    """
    # Score uses strict `>` (a value exactly at the threshold is not confident
    # enough), while word count uses inclusive `<=` (a 5-word comment still
    # qualifies as short) — this asymmetry is intentional.
    return (
        general_reaction_label == "ja"
        and general_reaction_score > GR_SCORE_THRESHOLD
        and len(comment_text.split()) <= FAST_PATH_MAX_WORDS
    )


def should_ensemble_path(
    general_reaction_label: str, general_reaction_score: float, comment_text: str
) -> bool:
    """Decide whether a comment takes the ensemble path.

    Implements "Pfad 2 — Ensemble Path" from general_reaction_gate_konzept.md:
    a *longer* comment that the upstream general-reaction classifier
    confidently labelled as a positive reaction still gets a dedicated
    stance-detection LLM call, but with the general-reaction signal injected as
    a hint. This is the complement of the fast path on word count: the fast path
    takes short confident-positive comments, the ensemble path takes the long
    ones.

    Args:
        general_reaction_label: Label from the general-reaction classifier;
            only "ja" (yes, a general reaction) can qualify.
        general_reaction_score: Confidence of that label, in [0.0, 1.0].
        comment_text: The raw comment text; word count gates long comments.

    Returns:
        True if the comment qualifies for the ensemble path, False otherwise.
    """
    # Score uses strict `>` (mirrors the fast path), while word count uses
    # strict `>` here so the fast path (`<=`) and ensemble path partition the
    # confident-positive comments with no overlap.
    return (
        general_reaction_label == "ja"
        and general_reaction_score > GR_SCORE_THRESHOLD
        and len(comment_text.split()) > FAST_PATH_MAX_WORDS
    )


async def run_stance_detection():
    """Run stance detection over the gold-label dataset.

    Loads the data (capped at DEBUG_ROW_LIMIT for manual test runs), applies the
    fast-path gate per row, and either emits the canned fast-path result or calls
    the LLM, then prints a summary of how many rows took each path along with
    per-iteration and total loop timing. Finally, evaluates the predictions and
    writes a timestamped markdown report to disk at the end of each run.
    """
    stance_detector = StanceDetector()
    llm_handler = LLMHandler()

    data = stance_detector.load_data()
    # Build a per-contribution lookup of key_message_index -> key_message text so
    # each row can see the *other* key messages of its own contribution. This lets
    # the prompt route a comment to NEUTRAL when it more clearly addresses a
    # sibling key message than the one currently being classified.
    km_by_contrib: dict[object, dict] = defaultdict(dict)
    for it in data:
        cid = it.get("contribution_id")
        kidx = it.get("key_message_index")
        if cid is not None and kidx is not None:
            km_by_contrib[cid][kidx] = it.get("key_message", "")
    # data = data[:DEBUG_ROW_LIMIT]  # DEBUG / TEMPORARY — see DEBUG_ROW_LIMIT above
    prompt_key = "base"
    prompt_config = stance_detector.load_prompt_config(
        "./prompts/config.json", prompt_key
    )

    base_url = STANCE_LLM_BASE_URL
    # base_url = "http://10.200.160.182:5555/v1"
    model_name = STANCE_LLM_MODEL_NAME

    fast_path_count = 0
    ensemble_count = 0
    llm_count = 0
    # Key predictions by item["row_index"], NOT the loop's enumerate index:
    # row_index is the stable gold identifier and differs from list position
    # for many rows, so the evaluator must match on it.
    predictions: dict[int, str | None] = {}

    semaphore = asyncio.Semaphore(1)
    async with httpx.AsyncClient() as client:
        loop_start = time.time()
        for index, item in enumerate(data):
            iter_start = time.time()
            # Safe defaults so a malformed row fails the gate and routes to the
            # LLM instead of raising.
            comment_text = item.get("comment_text", "")
            gr_label = item.get("general_reaction_label", "")
            gr_score = item.get("general_reaction_score", 0.0)

            # Sibling key messages of this row's contribution (exclude this row's
            # own key message by its index, and drop empty texts).
            cid = item.get("contribution_id")
            kidx = item.get("key_message_index")
            siblings = km_by_contrib.get(cid, {})
            other_key_messages = [
                text for idx, text in siblings.items() if idx != kidx and text
            ]

            if should_fast_path(gr_label, gr_score, comment_text):
                result = {
                    "stance": FAST_PATH_STANCE,
                    "explanation": FAST_PATH_EXPLANATION,
                }
                predictions[item["row_index"]] = result["stance"]
                fast_path_count += 1
                iter_elapsed = time.time() - iter_start
                gold = item.get("gold_stance", "?")
                predicted = result["stance"]
                match_marker = "[green]✓[/green]" if predicted == gold else "[red]✗[/red]"
                console.print(
                    f"[{index}] [bold cyan]FAST-PATH[/bold cyan]  predicted=[yellow]{predicted}[/yellow]  gold=[blue]{gold}[/blue]  {match_marker}  ({iter_elapsed:.2f}s)"
                )
            elif should_ensemble_path(gr_label, gr_score, comment_text):
                # Pfad 2 — Ensemble Path: still call the LLM, but inject the
                # general-reaction signal as a German hint on the user prompt.
                hint = stance_detector.format_gr_hint(gr_score)
                system, messages = stance_detector.build_messages(
                    prompt_config=prompt_config,
                    item=item,
                    gr_hint=hint,
                    other_key_messages=other_key_messages,
                )
                result = await llm_handler.call_model(
                    client=client,
                    base_url=base_url,
                    model_name=model_name,
                    system=system,
                    messages=messages,
                    semaphore=semaphore,
                    enable_thinking=STANCE_LLM_ENABLE_THINKING,
                    api_key=STANCE_LLM_API_KEY,
                )
                predictions[item["row_index"]] = (
                    result.get("stance") if result else None
                )
                ensemble_count += 1
                iter_elapsed = time.time() - iter_start
                gold = item.get("gold_stance", "?")
                predicted = result.get("stance") if result else None
                if predicted is None:
                    console.print(
                        f"[{index}] [bold magenta]ENSEMBLE[/bold magenta]  [red]FAILED[/red]  gold=[blue]{gold}[/blue]  ({iter_elapsed:.2f}s)"
                    )
                else:
                    match_marker = "[green]✓[/green]" if predicted == gold else "[red]✗[/red]"
                    console.print(
                        f"[{index}] [bold magenta]ENSEMBLE[/bold magenta]  predicted=[yellow]{predicted}[/yellow]  gold=[blue]{gold}[/blue]  {match_marker}  ({iter_elapsed:.2f}s)"
                    )
            else:
                system, messages = stance_detector.build_messages(
                    prompt_config=prompt_config,
                    item=item,
                    other_key_messages=other_key_messages,
                )
                result = await llm_handler.call_model(
                    client=client,
                    base_url=base_url,
                    model_name=model_name,
                    system=system,
                    messages=messages,
                    semaphore=semaphore,
                    enable_thinking=STANCE_LLM_ENABLE_THINKING,
                    api_key=STANCE_LLM_API_KEY,
                )
                predictions[item["row_index"]] = (
                    result.get("stance") if result else None
                )
                llm_count += 1
                iter_elapsed = time.time() - iter_start
                gold = item.get("gold_stance", "?")
                predicted = result.get("stance") if result else None
                if predicted is None:
                    console.print(
                        f"[{index}] [bold white]LLM[/bold white]  [red]FAILED[/red]  gold=[blue]{gold}[/blue]  ({iter_elapsed:.2f}s)"
                    )
                else:
                    match_marker = "[green]✓[/green]" if predicted == gold else "[red]✗[/red]"
                    console.print(
                        f"[{index}] [bold white]LLM[/bold white]  predicted=[yellow]{predicted}[/yellow]  gold=[blue]{gold}[/blue]  {match_marker}  ({iter_elapsed:.2f}s)"
                    )

    # Measured after the `async with` exits so the HTTP client's connection
    # teardown is included in the total.
    total_elapsed = time.time() - loop_start
    console.print(
        f"Fast-pathed: {fast_path_count}  |  ensemble: {ensemble_count}  "
        f"|  LLM calls: {llm_count}  "
        f"|  total: {fast_path_count + ensemble_count + llm_count}"
    )
    console.print(f"Total loop time: {total_elapsed:.2f}s")

    evaluator = StanceEvaluator()
    report = evaluator.evaluate(predictions)
    console.print(Panel(evaluator.format_report(report), title="[bold]Evaluation Report[/bold]"))

    metadata = RunMetadata(
        model_name=model_name,
        base_url=base_url,
        prompt_key=prompt_key,
        fast_path_count=fast_path_count,
        ensemble_count=ensemble_count,
        llm_count=llm_count,
        total_elapsed=total_elapsed,
    )
    report_path = write_markdown_report(report, metadata)
    console.print(f"Markdown report written to {report_path}")

    result_path = write_result_json(data, predictions, stance_model_name=STANCE_LLM_MODEL_NAME)
    console.print(f"[green]Result JSON written to[/green] {result_path}")


def _sanitize_model_name(name: str) -> str:
    """Turn a model name into a filesystem-safe slug for output filenames.

    Strips every character outside ``[A-Za-z0-9]`` (collapsing runs to a single
    ``-``) so a model name can never inject path separators or ``..`` into an
    output filename. This is a security control, not just cosmetic.

    Args:
        name: The raw model name (e.g. ``"Qwen/Qwen3-32B"``).

    Returns:
        A slug containing only ``[A-Za-z0-9]`` and ``-`` separators, or the
        literal ``"model"`` when the input has no usable characters.
    """
    slug = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")
    return slug or "model"


def _prompt_overwrite(path: Path) -> bool:
    """Ask the user whether to overwrite an existing file.

    Args:
        path: Path to the file that already exists, shown in the prompt.

    Returns:
        True if the user explicitly confirms with "y"/"yes" (case-insensitive);
        False for any other input, so the default (blank) answer is a decline.
    """
    console.print(f"[yellow]A saved file already exists at {path}[/yellow]")
    answer = input("Overwrite? [y/N] ").strip().lower()
    return answer in ("y", "yes")


def _prompt_positive_int(prompt: str) -> int:
    """Prompt for a positive integer, re-prompting until one is given.

    Args:
        prompt: The text shown to the user when asking for input.

    Returns:
        The parsed integer, guaranteed to be greater than zero.
    """
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            console.print("[red]Please enter a whole number.[/red]")
            continue
        if value <= 0:
            console.print("[red]Please enter a number greater than zero.[/red]")
            continue
        return value


def _prompt_optional_non_negative_int(prompt: str) -> int | None:
    """Prompt for an optional non-negative integer, re-prompting on bad input.

    Args:
        prompt: The text shown to the user when asking for input.

    Returns:
        The parsed integer (zero or greater), or ``None`` if the user leaves the
        input blank.
    """
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return None
        try:
            value = int(raw)
        except ValueError:
            console.print("[red]Please enter a whole number (or leave blank).[/red]")
            continue
        if value < 0:
            console.print("[red]Please enter zero or a positive number (or leave blank).[/red]")
            continue
        return value


def _prompt_project_ids(available_ids: list[str]) -> list[str]:
    """Prompt for the project IDs to sample, or all of them.

    Shows the available IDs, then reads a comma-separated selection. The literal
    input ``"all"`` (case-insensitive) selects every available ID. Empty tokens
    are dropped and the prompt repeats until at least one ID is chosen.

    Args:
        available_ids: All project IDs present in the contributions data.

    Returns:
        The selected project IDs as a list of strings.
    """
    while True:
        console.print(f"[cyan]Available project IDs:[/cyan] {', '.join(available_ids)}")
        raw = input("Project IDs (comma-separated, or 'all'): ").strip()
        # Lowercase before comparing so "ALL", "All", etc. all trigger the
        # select-everything shortcut; the literal must match case-insensitively.
        if raw.lower() == "all":
            return list(available_ids)
        ids = [token.strip() for token in raw.split(",") if token.strip()]
        if ids:
            return ids
        console.print("[red]Please enter at least one project ID.[/red]")


def _pick_parquet_file(data_dir: Path) -> Path | None:
    """Prompt the user to choose a parquet file from a directory.

    Lists every ``*.parquet`` file in ``data_dir`` as a numbered (1-based) menu
    and reads a selection, re-prompting on non-integer or out-of-range input.

    Args:
        data_dir: Directory to scan for ``*.parquet`` files.

    Returns:
        The chosen file's :class:`~pathlib.Path`, or ``None`` if the directory
        does not exist or contains no parquet files.
    """
    if not data_dir.exists():
        console.print(f"[red]No data directory found at {data_dir}.[/red]")
        return None

    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        console.print(f"[red]No parquet files found in {data_dir}.[/red]")
        return None

    for number, file in enumerate(files, start=1):
        # file.name is a filesystem name, not model output, but printing it as a
        # separate argument keeps any stray rich-markup characters literal.
        console.print(f"  {number}.", file.name)

    while True:
        raw = input("Select a file by number: ").strip()
        try:
            choice = int(raw)
        except ValueError:
            console.print("[red]Please enter a whole number.[/red]")
            continue
        if not 1 <= choice <= len(files):
            console.print(f"[red]Please enter a number between 1 and {len(files)}.[/red]")
            continue
        return files[choice - 1]


def _pick_judged_parquet_files(data_dir: Path) -> list[Path] | None:
    """Prompt the user to multi-select ``judged_*.parquet`` files from a directory.

    Lists every ``judged_*.parquet`` file in ``data_dir`` as a numbered (1-based)
    menu and reads a comma-separated selection, or the literal ``"all"`` (case-
    insensitive) to select every file, re-prompting on invalid input.

    Args:
        data_dir: Directory to scan for ``judged_*.parquet`` files.

    Returns:
        The chosen files' :class:`~pathlib.Path` objects in selection order
        (duplicates removed), or ``None`` if the directory does not exist or
        contains no matching files.
    """
    if not data_dir.exists():
        console.print(f"[red]No data directory found at {data_dir}.[/red]")
        return None

    files = sorted(data_dir.glob("judged_*.parquet"))
    if not files:
        console.print(f"[red]No judged_*.parquet files found in {data_dir}.[/red]")
        return None

    for number, file in enumerate(files, start=1):
        # file.name is a filesystem name, not model output, but printing it as a
        # separate argument keeps any stray rich-markup characters literal.
        console.print(f"  {number}.", file.name)

    while True:
        raw = input("Select files (comma-separated numbers, or 'all'): ").strip()
        if raw.lower() == "all":
            return list(files)
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        if not tokens:
            console.print("[red]Please enter at least one number, or 'all'.[/red]")
            continue
        indices = []
        valid = True
        for token in tokens:
            try:
                idx = int(token)
            except ValueError:
                console.print("[red]Please enter whole numbers separated by commas.[/red]")
                valid = False
                break
            if not 1 <= idx <= len(files):
                console.print(f"[red]Please enter numbers between 1 and {len(files)}.[/red]")
                valid = False
                break
            indices.append(idx)
        if not valid:
            continue
        seen = []
        for idx in indices:
            if idx not in seen:
                seen.append(idx)
        return [files[i - 1] for i in seen]


def _summarize_judged_df(df: pd.DataFrame) -> dict:
    """Aggregate one ``judged_*.parquet`` file's ``judge_result`` scores.

    A ``judge_result`` cell is either a KeyMessageJudgment dict (judged) or
    ``None`` (either failed or skipped at judge time — that distinction was only
    an in-memory counter during the original judge run and was never persisted,
    so both collapse to "no result" here).

    Args:
        df: A loaded ``judged_*.parquet`` DataFrame.

    Returns:
        A dict with total/judged/no_result counts and mean_overall/
        mean_completeness/mean_semantic_precision/mean_downstream_viability,
        each ``None`` if no row contributed a numeric value for that dimension.
    """
    total = len(df)
    if "judge_result" not in df.columns:
        return {
            "total": total, "judged": 0, "no_result": total,
            "mean_overall": None, "mean_completeness": None,
            "mean_semantic_precision": None, "mean_downstream_viability": None,
        }

    overall_scores, completeness_scores, semantic_scores, viability_scores = [], [], [], []
    judged = 0

    def _numeric(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    for cell in df["judge_result"]:
        if cell is None or (np.ndim(cell) == 0 and pd.isna(cell)):
            continue
        judged += 1
        overall = cell.get("overall_score")
        if _numeric(overall):
            overall_scores.append(overall)
        for key, bucket in (
            ("completeness", completeness_scores),
            ("semantic_precision", semantic_scores),
            ("downstream_viability", viability_scores),
        ):
            score = (cell.get(key) or {}).get("score")
            if _numeric(score):
                bucket.append(score)

    def _mean(values):
        return sum(values) / len(values) if values else None

    return {
        "total": total,
        "judged": judged,
        "no_result": total - judged,
        "mean_overall": _mean(overall_scores),
        "mean_completeness": _mean(completeness_scores),
        "mean_semantic_precision": _mean(semantic_scores),
        "mean_downstream_viability": _mean(viability_scores),
    }


def _judge_model_label(df: pd.DataFrame, path: Path) -> str:
    """Derive a display label for a judged file: extraction model or filename.

    Prefers the ``extraction_model_name`` column (present on extraction-format
    judged files) since it names the model precisely; falls back to the filename
    stem with the ``judged_`` prefix stripped for whole-process-format files,
    which don't carry that column but always embed the model slug in their
    filename already (see :func:`_sanitize_model_name`).

    Args:
        df: A loaded ``judged_*.parquet`` DataFrame.
        path: The file's path, used for the filename fallback.

    Returns:
        A human-readable label identifying the model/run.
    """
    if "extraction_model_name" in df.columns:
        values = df["extraction_model_name"].dropna().unique()
        if len(values) == 1:
            return str(values[0])
        if len(values) > 1:
            return ", ".join(str(v) for v in values)
    return path.stem.removeprefix("judged_")


def run_compare_judge_results() -> None:
    """Compare multiple ``judged_*.parquet`` files to rank extraction models.

    Prompts the user to multi-select ``judged_*.parquet`` files, aggregates each
    file's judge scores, prints a ranked Rich table (highest mean overall score
    first), and writes a markdown comparison report to disk. Aborts early if no
    files are selected.
    """
    paths = _pick_judged_parquet_files(DATA_DIR)
    if paths is None:
        return

    rows = []
    for path in paths:
        df = pd.read_parquet(path)
        summary = _summarize_judged_df(df)
        label = _judge_model_label(df, path)
        rows.append(
            JudgeComparisonRow(
                label=label,
                file_name=path.name,
                total=summary["total"],
                judged=summary["judged"],
                no_result=summary["no_result"],
                mean_overall=summary["mean_overall"],
                mean_completeness=summary["mean_completeness"],
                mean_semantic_precision=summary["mean_semantic_precision"],
                mean_downstream_viability=summary["mean_downstream_viability"],
            )
        )

    # Stable sorts applied least-significant key first: label asc, then judged
    # desc, then mean_overall desc (None treated as -inf so all-unjudged files
    # rank last) — this fully determines order even when overall scores tie.
    rows.sort(key=lambda r: r.label)
    rows.sort(key=lambda r: r.judged, reverse=True)
    rows.sort(
        key=lambda r: r.mean_overall if r.mean_overall is not None else float("-inf"),
        reverse=True,
    )

    def _fmt(value):
        return "N/A" if value is None else f"{value:.2f}"

    table = Table(title="Judge result comparison")
    table.add_column("Model / file")
    table.add_column("Judged")
    table.add_column("No result")
    table.add_column("Completeness")
    table.add_column("Semantic precision")
    table.add_column("Downstream viability")
    table.add_column("Overall")
    for row in rows:
        table.add_row(
            escape(row.label),
            str(row.judged),
            str(row.no_result),
            _fmt(row.mean_completeness),
            _fmt(row.mean_semantic_precision),
            _fmt(row.mean_downstream_viability),
            _fmt(row.mean_overall),
        )
    console.print(table)

    report_path = write_judge_comparison_report(rows)
    console.print(f"[green]Comparison report written to {report_path}[/green]")


def _prompt_proceeding_description() -> str:
    """Prompt for an optional proceeding description used as LLM context.

    Returns:
        The entered description with surrounding whitespace stripped, or an empty
        string if the user left it blank.
    """
    description = input("Proceeding description (context, may be empty): ").strip()
    if not description:
        console.print("[yellow]No proceeding description given — continuing without context.[/yellow]")
        return ""
    return description


async def run_key_message_extraction() -> None:
    """Extract key messages from a chosen parquet file of contributions.

    Prompts the user to pick a source parquet file, validates that it contains a
    ``contributionContent`` column, then runs :class:`KeyMessageExtraction` over
    every row (serialized via a semaphore of 1) and writes a new parquet file. A
    single ``key_messages`` column is added: each cell holds the list of
    key-message dicts extracted for that row (or ``None`` if extraction failed) —
    not the former three separate ``key_message`` / ``key_message_type`` /
    ``explanation`` columns. The per-contribution elapsed time is printed after
    each row and the total extraction time is printed to the console once every
    row has been processed. Aborts early if no file is selected, the column is
    missing, or the user declines to overwrite an existing output.
    """
    source_path = _pick_parquet_file(DATA_DIR)
    if source_path is None:
        return

    df = pd.read_parquet(source_path)
    if "contributionContent" not in df.columns:
        console.print("[red]Selected file is missing the 'contributionContent' column.[/red]")
        return

    description = _prompt_proceeding_description()

    model_slug = _sanitize_model_name(EXTRACTION_LLM_MODEL_NAME)
    output_path = DATA_DIR / f"key_messages_{source_path.stem}_{model_slug}.parquet"
    if output_path.exists() and not _prompt_overwrite(output_path):
        console.print("[yellow]Skipping extraction — existing file kept.[/yellow]")
        return

    extractor = KeyMessageExtraction(
        base_url=EXTRACTION_LLM_BASE_URL,
        model_name=EXTRACTION_LLM_MODEL_NAME,
        api_key=EXTRACTION_LLM_API_KEY,
    )
    semaphore = asyncio.Semaphore(1)

    key_messages_col: list[list[dict] | None] = []

    total = len(df)
    total_start = time.time()
    async with httpx.AsyncClient() as client:
        for index, (_, row) in enumerate(df.iterrows(), start=1):
            iter_start = time.time()
            # Treat missing content as an empty string so a NaN cell does not
            # crash the extractor; the LLM simply receives no contribution text.
            contribution = "" if pd.isna(row["contributionContent"]) else str(row["contributionContent"])
            result = await extractor.extract(
                client=client,
                contribution=contribution,
                proceeding_description=description,
                semaphore=semaphore,
            )
            iter_elapsed = time.time() - iter_start
            # Contribution text is untrusted and may contain literal [ / ] that
            # rich would misread as console markup, so escape() it. Truncate the
            # RAW text first, then escape: escaping before slicing could split a
            # markup escape sequence.
            preview = contribution[:200] + "…" if len(contribution) > 200 else contribution
            console.print(f"[{index}/{total}] [dim]{escape(preview)}[/dim]")
            if result is not None:
                key_messages_col.append(result)
                console.print(f"[{index}/{total}] {len(result)} key messages found ({iter_elapsed:.2f}s)")
            else:
                key_messages_col.append(None)
                console.print(f"[red][{index}/{total}] FAILED ({iter_elapsed:.2f}s)[/red]")

    total_elapsed = time.time() - total_start
    console.print(f"Total extraction time: {total_elapsed:.2f}s")
    result_df = df.copy()
    result_df["key_messages"] = key_messages_col
    # Record the full (unsanitized) extraction model name as provenance so every
    # output row is traceable to the model that produced it.
    result_df["extraction_model_name"] = EXTRACTION_LLM_MODEL_NAME
    result_df.to_parquet(output_path)
    console.print(f"[green]Key messages saved to {output_path}[/green]")


def _is_whole_process_format(df: pd.DataFrame) -> bool:
    """Detect whole_process's flattened (key_message x comment) output shape.

    Extraction-format files (with a ``key_messages`` column) always take
    precedence over this check — this only identifies files that lack
    ``key_messages`` but carry whole_process's flattened per-comment columns.

    Args:
        df: The DataFrame loaded from the selected source parquet.

    Returns:
        ``True`` if the frame has no ``key_messages`` column but does carry the
        ``key_message`` and ``contribution_id`` columns produced by
        whole_process; ``False`` otherwise.
    """
    return (
        "key_messages" not in df.columns
        and "key_message" in df.columns
        and "contribution_id" in df.columns
    )


def _group_whole_process_to_key_messages(df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct a per-contribution key_messages view from whole_process output.

    whole_process.py flattens one row per (key message x comment) pair, so the
    same key message's fields repeat once per comment. This groups by
    ``contribution_id`` and, for each contribution, deduplicates the repeated
    rows back into a single ordered ``list[dict]`` of distinct key messages
    (first-seen order), renaming ``key_message_explanation`` to ``explanation``
    so each dict matches the shape :class:`KeyMessageJudge` expects (the same
    shape produced by :func:`run_key_message_extraction`).
    ``related_sentence_spans``, ``stance``, ``comment_text``, the (stance)
    ``explanation`` column, and model-name columns are intentionally dropped —
    they are either per-comment (no single correct value at the contribution
    level) or unused by the judge.

    Args:
        df: A whole_process-format DataFrame (one row per key-message x comment
            pair).

    Returns:
        A DataFrame with one row per contribution and the columns
        ``contribution_id``, ``contributionContent``, and ``key_messages``.
    """
    records = []
    # sort=False preserves the source file's first-seen contribution order in the
    # output; dropna=False keeps any rows whose contribution_id is NaN rather than
    # silently discarding them (pandas' groupby drops NaN keys by default).
    for contribution_id, group in df.groupby("contribution_id", sort=False, dropna=False):
        contribution_content = group["contributionContent"].iloc[0]
        seen = {}
        for _, row in group.iterrows():
            km_text = str(row["key_message"])
            km_type = str(row["key_message_type"])
            explanation = str(row["key_message_explanation"])
            raw_rs = row["related_sentences"]
            # A missing related_sentences cell may be None or a scalar NaN; an
            # array-valued cell has ndim > 0, so guard the pd.isna check with
            # np.ndim to avoid its ambiguous-truth-value error on arrays.
            related_sentences = (
                []
                if raw_rs is None or (np.ndim(raw_rs) == 0 and pd.isna(raw_rs))
                else [str(s) for s in raw_rs]
            )
            # These four fields together are the key message's identity: text,
            # type, explanation, and its set of related sentences. Two rows that
            # match on all four are the same key message flattened across
            # different comments, so keep only the first occurrence. (Message text
            # alone is insufficient — distinct key messages can share wording.)
            dedup_key = (km_text, km_type, explanation, tuple(related_sentences))
            if dedup_key not in seen:
                seen[dedup_key] = {
                    "key_message": km_text,
                    "key_message_type": km_type,
                    "related_sentences": related_sentences,
                    "explanation": explanation,
                }
        records.append(
            {
                "contribution_id": contribution_id,
                "contributionContent": contribution_content,
                "key_messages": list(seen.values()),
            }
        )
    return pd.DataFrame(records)


async def run_key_message_judge() -> None:
    """Judge the quality of already-extracted key messages in a parquet file.

    Prompts the user to pick a source parquet file, validates that it contains
    both a ``key_messages`` and a ``contributionContent`` column, then runs
    :class:`KeyMessageJudge` over every row (serialized via a semaphore of 1) and
    writes a new ``judged_<source-stem>.parquet`` file. A single ``judge_result``
    column is added: each cell holds the judgment dict produced by the LLM (or
    ``None`` if the row was skipped because it had no key messages, or if the
    judge call failed). The per-row elapsed time is printed after each row, and
    once every row has been processed an aggregate summary panel reports the mean
    overall score together with the judged, failed, and skipped counts. Aborts
    early if no file is selected, a required column is missing, or the user
    declines to overwrite an existing output.
    """
    source_path = _pick_parquet_file(DATA_DIR)
    if source_path is None:
        return

    df = pd.read_parquet(source_path)
    if "key_messages" in df.columns:
        pass  # extraction-format: already has the shape the judge loop expects
    elif _is_whole_process_format(df):
        df = _group_whole_process_to_key_messages(df)
    else:
        console.print(
            "[red]Selected file has neither a 'key_messages' column (extraction "
            "output) nor whole_process's 'key_message'/'contribution_id' columns "
            "— cannot judge.[/red]"
        )
        return
    if "contributionContent" not in df.columns:
        console.print("[red]Selected file is missing the 'contributionContent' column.[/red]")
        return

    output_path = DATA_DIR / f"judged_{source_path.stem}.parquet"
    if output_path.exists() and not _prompt_overwrite(output_path):
        console.print("[yellow]Skipping judging — existing file kept.[/yellow]")
        return

    judge = KeyMessageJudge(base_url=JUDGE_LLM_BASE_URL, model_name=JUDGE_LLM_MODEL_NAME, api_key=JUDGE_LLM_API_KEY)
    semaphore = asyncio.Semaphore(1)

    judge_col: list[dict | None] = []
    judged = 0
    failed = 0
    skipped = 0

    total = len(df)
    total_start = time.time()
    async with httpx.AsyncClient() as client:
        for index, (_, row) in enumerate(df.iterrows(), start=1):
            iter_start = time.time()
            km = row["key_messages"]
            # np.ndim avoids the ambiguous truth value that pd.isna raises on
            # array-valued cells; scalars have ndim 0 and are NaN-checked directly.
            if km is None or (np.ndim(km) == 0 and pd.isna(km)):
                judge_col.append(None)
                skipped += 1
                console.print(f"[yellow][{index}/{total}] skipped (no key messages)[/yellow]")
                continue

            contribution = "" if pd.isna(row["contributionContent"]) else str(row["contributionContent"])
            category = str(row["category"]) if "category" in df.columns and not pd.isna(row["category"]) else ""
            # Parquet may hand back a numpy array rather than a plain list.
            key_messages_list = list(km)

            result = await judge.judge(
                client=client,
                contribution=contribution,
                category=category,
                key_messages=key_messages_list,
                semaphore=semaphore,
            )
            iter_elapsed = time.time() - iter_start
            if result is not None:
                judge_col.append(result)
                judged += 1
                console.print(
                    f"[{index}/{total}] overall_score={result.get('overall_score', '?')} ({iter_elapsed:.2f}s)"
                )
            else:
                judge_col.append(None)
                failed += 1
                console.print(f"[red][{index}/{total}] FAILED ({iter_elapsed:.2f}s)[/red]")

    total_elapsed = time.time() - total_start
    console.print(f"Total judging time: {total_elapsed:.2f}s")

    result_df = df.copy()
    result_df["judge_result"] = judge_col
    result_df.to_parquet(output_path)
    console.print(f"[green]Judgments saved to {output_path}[/green]")

    # Collect only numeric overall_score values from successful judgments so a
    # malformed/missing score never poisons the mean or raises.
    scores = []
    for result in judge_col:
        if result is None:
            continue
        score = result.get("overall_score")
        if isinstance(score, (int, float)) and not isinstance(score, bool):
            scores.append(score)

    if scores:
        mean_line = f"[bold]Mean overall score:[/bold] {sum(scores) / len(scores):.2f}"
    else:
        mean_line = "[bold]Mean overall score:[/bold] no rows judged"

    console.print(Panel(
        f"{mean_line}\n"
        f"[bold]Judged:[/bold] {judged}\n"
        f"[bold]Failed:[/bold] {failed}\n"
        f"[bold]Skipped:[/bold] {skipped}",
        title="[bold cyan]Key message judge summary[/bold cyan]",
    ))


async def run_whole_process() -> None:
    """Run the end-to-end whole-process pipeline over a chosen parquet file.

    Prompts the user to pick a source parquet file and validates that it contains
    both a ``contributionContent`` and a ``comments`` column, aborting with a red
    error if either is missing. Delegates to :class:`WholeProcess`, which extracts
    key messages per contribution and, for every (key message x comment) pair,
    classifies the comment with the general-reaction classifier and resolves a
    stance (fast path, ensemble path, or plain LLM call). The flattened result —
    one row per pair with columns ``contribution_id``, ``key_message``,
    ``key_message_type``, ``comment_text``, ``stance``, and ``explanation`` — is
    written to ``whole_process_<source-stem>.parquet`` in the data directory.
    Before processing, the user is prompted for how many contributions to handle;
    a blank answer processes all of them, while a positive integer draws that
    many as a random sample (clamped to the total available). Aborts early if no
    file is selected, a required column is missing, the contribution count is not
    a positive whole number, or the user declines to overwrite an existing output.
    """
    source_path = _pick_parquet_file(DATA_DIR)
    if source_path is None:
        return

    df = pd.read_parquet(source_path)
    if "contributionContent" not in df.columns:
        console.print("[red]Selected file is missing the 'contributionContent' column.[/red]")
        return
    if "comments" not in df.columns:
        console.print("[red]Selected file is missing the 'comments' column.[/red]")
        return

    # Ask how many contributions to process; blank means all.
    n_raw = input("Process how many contributions? (blank for all): ").strip()
    if n_raw != "":
        try:
            n_contributions = int(n_raw)
        except ValueError:
            console.print("[red]Please enter a whole number or leave blank for all.[/red]")
            return
        if n_contributions <= 0:
            console.print("[red]Please enter a number greater than zero.[/red]")
            return
        total_available = len(df)
        if n_contributions < total_available:
            df = df.sample(n=n_contributions)
            console.print(
                f"[cyan]Processing {n_contributions} of {total_available} "
                f"contributions (random sample).[/cyan]"
            )
        else:
            console.print(
                f"[yellow]Requested {n_contributions} but only {total_available} "
                f"available — using all.[/yellow]"
            )

    extraction_slug = _sanitize_model_name(EXTRACTION_LLM_MODEL_NAME)
    stance_slug = _sanitize_model_name(STANCE_LLM_MODEL_NAME)
    output_path = DATA_DIR / f"whole_process_{source_path.stem}_{extraction_slug}_{stance_slug}.parquet"
    if output_path.exists() and not _prompt_overwrite(output_path):
        console.print("[yellow]Skipping whole process — existing file kept.[/yellow]")
        return

    processor = WholeProcess(
        extraction_base_url=EXTRACTION_LLM_BASE_URL,
        extraction_model_name=EXTRACTION_LLM_MODEL_NAME,
        stance_base_url=STANCE_LLM_BASE_URL,
        stance_model_name=STANCE_LLM_MODEL_NAME,
        extraction_api_key=EXTRACTION_LLM_API_KEY,
        stance_api_key=STANCE_LLM_API_KEY,
        stance_enable_thinking=STANCE_LLM_ENABLE_THINKING,
    )
    result_df = await processor.run(df, proceeding_description="")
    result_df.to_parquet(output_path)
    console.print(f"[green]Whole process saved to {output_path} — {len(result_df)} rows written[/green]")


async def run_whole_process_v2() -> None:
    """Run the end-to-end whole-process pipeline using the policy v2 engine.

    Mirrors run_whole_process() (same file picker, same row-count prompt,
    same overwrite confirmation) but delegates to
    :meth:`WholeProcess.run_v2`, which resolves stance via
    :class:`policy.StancePolicyEngineV2` -- one call per comment across all
    of its key messages at once, with the general-approval rule applied
    deterministically -- instead of the legacy per-(key message, comment)
    fast/ensemble/LLM routing. See docs/stance_policy_v2.md. Output columns
    (whole_process.OUTPUT_COLUMNS_V2) differ from run_whole_process()'s
    legacy shape: a validated status/basis per pair rather than a bare
    stance, so failures and review routing stay visible rather than being
    silently folded into a stance value.
    """
    source_path = _pick_parquet_file(DATA_DIR)
    if source_path is None:
        return

    df = pd.read_parquet(source_path)
    if "contributionContent" not in df.columns:
        console.print("[red]Selected file is missing the 'contributionContent' column.[/red]")
        return
    if "comments" not in df.columns:
        console.print("[red]Selected file is missing the 'comments' column.[/red]")
        return

    n_raw = input("Process how many contributions? (blank for all): ").strip()
    if n_raw != "":
        try:
            n_contributions = int(n_raw)
        except ValueError:
            console.print("[red]Please enter a whole number or leave blank for all.[/red]")
            return
        if n_contributions <= 0:
            console.print("[red]Please enter a number greater than zero.[/red]")
            return
        total_available = len(df)
        if n_contributions < total_available:
            df = df.sample(n=n_contributions)
            console.print(
                f"[cyan]Processing {n_contributions} of {total_available} "
                f"contributions (random sample).[/cyan]"
            )
        else:
            console.print(
                f"[yellow]Requested {n_contributions} but only {total_available} "
                f"available — using all.[/yellow]"
            )

    extraction_slug = _sanitize_model_name(EXTRACTION_LLM_MODEL_NAME)
    stance_slug = _sanitize_model_name(STANCE_LLM_MODEL_NAME)
    output_path = DATA_DIR / f"whole_process_v2_{source_path.stem}_{extraction_slug}_{stance_slug}.parquet"
    if output_path.exists() and not _prompt_overwrite(output_path):
        console.print("[yellow]Skipping whole process (v2) — existing file kept.[/yellow]")
        return

    processor = WholeProcess(
        extraction_base_url=EXTRACTION_LLM_BASE_URL,
        extraction_model_name=EXTRACTION_LLM_MODEL_NAME,
        stance_base_url=STANCE_LLM_BASE_URL,
        stance_model_name=STANCE_LLM_MODEL_NAME,
        extraction_api_key=EXTRACTION_LLM_API_KEY,
        stance_api_key=STANCE_LLM_API_KEY,
        stance_enable_thinking=STANCE_LLM_ENABLE_THINKING,
    )
    result_df = await processor.run_v2(df, proceeding_description="")
    result_df.to_parquet(output_path)
    console.print(f"[green]Whole process (v2) saved to {output_path} — {len(result_df)} rows written[/green]")


def inspect_key_messages() -> None:
    """Interactively page through a parquet file's extracted key messages.

    Prompts the user to pick a source parquet file and validates that it contains
    both a ``key_messages`` and a ``contributionContent`` column, aborting with a
    message if either is missing. Each row is then rendered in its own Rich panel
    showing the contribution text followed by its extracted key messages (or a
    ``FAILED`` marker when the cell holds no result). After each panel the user
    advances to the next row by entering "n" (or pressing Enter for the default)
    or exits early by entering "q".
    """
    source_path = _pick_parquet_file(DATA_DIR)
    if source_path is None:
        return

    df = pd.read_parquet(source_path)
    if "key_messages" not in df.columns:
        console.print("[red]Selected file is missing the 'key_messages' column.[/red]")
        return
    if "contributionContent" not in df.columns:
        console.print("[red]Selected file is missing the 'contributionContent' column.[/red]")
        return

    total = len(df)
    for index, (_, row) in enumerate(df.iterrows(), start=1):
        contribution = "" if pd.isna(row["contributionContent"]) else str(row["contributionContent"])
        km = row["key_messages"]
        is_failed = km is None or (np.ndim(km) == 0 and pd.isna(km))

        lines = [f"[bold]Contribution:[/bold] {escape(contribution)}\n"]
        if is_failed:
            lines.append("[red]FAILED[/red]")
        else:
            for i, msg in enumerate(km, start=1):
                lines.append(f"[bold cyan]Key Message {i}[/bold cyan]")
                lines.append(f"  [bold]Message:[/bold]     {escape(str(msg.get('key_message', '')))}")
                lines.append(f"  [bold]Type:[/bold]        {escape(str(msg.get('key_message_type', '')))}")
                lines.append(f"  [bold]Explanation:[/bold] {escape(str(msg.get('explanation', '')))}")
                sentences = msg.get("related_sentences", [])
                if len(sentences) > 0:
                    lines.append("  [bold]Sentences:[/bold]")
                    for s in sentences:
                        lines.append(f"    • {escape(str(s))}")
                lines.append("")

        console.print(Panel("\n".join(lines), title=f"[bold]Row {index}/{total}[/bold]"))

        answer = input("[n]ext / [q]uit (n): ").strip().lower()
        if answer == "q":
            break


def inspect_parquet() -> None:
    """Show metadata and the first rows of a chosen parquet file.

    Prompts the user to pick a source parquet file, then prints a metadata panel
    (file name, total row count, and column names) followed by a Rich table of
    the first ten rows. Each cell is stringified, NaN-guarded, truncated to
    ``MAX_CELL_CHARS`` characters, and escaped so stray markup stays literal.
    Aborts early if no file is selected.
    """
    source_path = _pick_parquet_file(DATA_DIR)
    if source_path is None:
        return

    df = pd.read_parquet(source_path)

    columns = "\n".join(f"  • {escape(str(col))}" for col in df.columns)
    console.print(Panel(
        f"[bold]File:[/bold] {escape(source_path.name)}\n"
        f"[bold]Rows:[/bold] {len(df)}\n"
        f"[bold]Columns:[/bold]\n{columns}",
        title="[bold cyan]Parquet metadata[/bold cyan]",
    ))

    def _to_cell(value) -> str:
        # np.ndim avoids the ambiguous truth value that pd.isna raises on
        # array-valued cells; scalars have ndim 0 and are NaN-checked directly.
        if np.ndim(value) == 0 and pd.isna(value):
            return ""
        text = str(value)
        if len(text) > MAX_CELL_CHARS:
            text = text[:MAX_CELL_CHARS - 1] + "…"
        return escape(text)

    table = Table()
    for col in df.columns:
        table.add_column(escape(str(col)))

    head = df.head(10)
    for row in head.itertuples(index=False):
        cells = [_to_cell(value) for value in row]
        table.add_row(*cells)

    console.print(table)


def export_parquet_to_excel() -> None:
    """Convert a chosen parquet file to an Excel workbook.

    Prompts the user to pick a source parquet file, then writes an ``.xlsx``
    file with the same stem next to it in ``DATA_DIR``. Aborts early if no
    file is selected.
    """
    source_path = _pick_parquet_file(DATA_DIR)
    if source_path is None:
        return

    df = pd.read_parquet(source_path)

    output_path = source_path.with_suffix(".xlsx")
    df.to_excel(output_path, index=False)
    console.print(f"[green]Wrote {len(df)} rows to {output_path}.[/green]")


def sample_contributions() -> None:
    """Draw a random sample of contributions for chosen projects.

    Loads the fetched contributions file, prompts for project IDs, a sample
    size, and an optional random seed for reproducibility, then delegates to
    :class:`SampleContributions`. Aborts with a message if the file is missing
    or lacks the ``project_id`` column.
    """
    if not CONTRIBUTIONS_PATH.exists():
        console.print(f"[red]No contributions file found at {CONTRIBUTIONS_PATH}. Run option 1 (Fetch contributions) first.[/red]")
        return

    df = pd.read_parquet(CONTRIBUTIONS_PATH)
    if "project_id" not in df.columns:
        console.print("[red]Contributions file is missing the 'project_id' column.[/red]")
        return

    available_ids = df["project_id"].astype(str).unique().tolist()
    ids = _prompt_project_ids(available_ids)
    n = _prompt_positive_int("Sample size (n): ")
    seed = _prompt_optional_non_negative_int("Random seed for reproducibility (blank for random): ")

    try:
        # Pass source_path explicitly rather than relying on SampleContributions'
        # default: tests redirect main.CONTRIBUTIONS_PATH into tmp_path, and the
        # sampler must read the same file this function already validated above.
        sampler = SampleContributions(project_ids=ids, n=n, source_path=CONTRIBUTIONS_PATH, random_state=seed)
        result_path = sampler.sample()
        console.print(f"[green]Sample written to {result_path}[/green]")
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Sampling failed: {e}[/red]")


def fetch_contributions() -> None:
    """Fetch DIPAS projects and their contributions, then save them to parquet.

    Prompts before overwriting an existing contributions file and aborts early if
    no projects are returned, so a failed upstream fetch never clobbers good data.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if CONTRIBUTIONS_PATH.exists():
        if not _prompt_overwrite(CONTRIBUTIONS_PATH):
            console.print("[yellow]Skipping fetch — existing file kept.[/yellow]")
            return

    dataset = CreateDataset()

    console.print("[bold]Fetching projects...[/bold]")
    df_projects = dataset.create()
    if df_projects.empty or "id" not in df_projects.columns:
        console.print("[red]No projects fetched — aborting.[/red]")
        return

    # Drop missing ids first, then stringify: the downstream API expects string
    # ids, and dropping NaN before the cast avoids "nan" entries in the list.
    project_ids = df_projects["id"].dropna().astype(str).tolist()
    console.print(f"[bold]Fetching contributions for {len(project_ids)} projects...[/bold]")
    console.print("=" * 60)
    df = dataset.fetch_all_contributions(project_ids)
    console.print("=" * 60)
    console.print(f"[green]Total contributions fetched: {len(df)}[/green]")

    summary = dataset.contributions_summary(df)
    if summary:
        console.print(Panel(summary, title="[bold]Contributions per project (top 10)[/bold]"))

    df.to_parquet(CONTRIBUTIONS_PATH)
    console.print(f"[green]Contributions saved to {CONTRIBUTIONS_PATH}[/green]")


def get_contribution_with_comments(contribution_id: int | None = None) -> dict | None:
    """Fetch a contribution and its related comments.

    When contribution_id is provided, looks up the project from the local
    contributions parquet, fetches all contributions for that project from the
    PDS API, and finds the matching contribution. When contribution_id is None,
    picks a random row from the parquet directly without an API call.
    In both cases, fetches comments for the contribution from the database.

    Args:
        contribution_id: The integer ID of the contribution to fetch, or None
            to pick one at random from the local contributions file.

    Returns:
        A dict with keys "contribution" (dict, containing a "contribution_id"
        key) and "comments" (list[str]), or None if any step fails.
    """
    if not CONTRIBUTIONS_PATH.exists():
        console.print(f"[red]No contributions file found at {CONTRIBUTIONS_PATH}. Run option 1 (Fetch contributions) first.[/red]")
        return None

    df = pd.read_parquet(CONTRIBUTIONS_PATH)

    if "contribution_id" not in df.columns:
        console.print("[red]Contributions file is missing the 'contribution_id' column.[/red]")
        return None

    if contribution_id is None:
        if df.empty:
            console.print("[red]Contributions file is empty.[/red]")
            return None
        row = df.sample(n=1).iloc[0]
        # Parquet yields numpy.int64 which CommentFetcher rejects; cast to native int.
        contribution_id = int(row["contribution_id"])
        project_id = row["project_id"]
        contribution = {
            "project_id": project_id,
            "contribution_id": contribution_id,
            "title": None if pd.isna(row["title"]) else str(row["title"]),
            "contributionContent": None if pd.isna(row["contributionContent"]) else str(row["contributionContent"]),
            "category": None if pd.isna(row["category"]) else str(row["category"]),
        }
        console.print(f"[bold]Randomly picked contribution {contribution_id} from project {project_id}[/bold]")
    else:
        matches = df[df["contribution_id"] == contribution_id]
        if matches.empty:
            console.print(f"[red]Contribution {contribution_id} not found in {CONTRIBUTIONS_PATH}.[/red]")
            return None
        project_id = matches.iloc[0]["project_id"]
        # Percent-encode project_id so it can't inject extra path segments into the URL.
        url = f"{API_URL}{urllib.parse.quote(str(project_id), safe='')}/contributions/"
        console.print(f"[bold]Fetching contributions for project {project_id}...[/bold]")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as e:
            console.print(f"[red]API request failed: {e}[/red]")
            return None
        contribution = None
        for feature in data.get("features", []):
            props = feature.get("properties", {})
            if props.get("id") == contribution_id:
                contribution = {
                    "project_id": project_id,
                    # Normalize to native int in case the API client returns numpy.int64.
                    "contribution_id": int(props["id"]),
                    "title": props.get("title"),
                    "contributionContent": props.get("contributionContent"),
                    "category": props.get("category"),
                }
                break
        if contribution is None:
            console.print(f"[yellow]Contribution {contribution_id} not returned by the API.[/yellow]")
            return None

    fetcher = CommentFetcher()
    try:
        # contribution_id may be numpy.int64 here; CommentFetcher requires a native int.
        comments = fetcher.get_comments(int(contribution_id))
    except Exception as e:
        console.print(f"[red]Failed to fetch comments: {e}[/red]")
        return None
    finally:
        # Release the DB engine even if get_comments raised, to avoid leaking connections.
        fetcher.close()

    comment_texts = [c["field_comment_value"] for c in comments]

    if not comment_texts:
        console.print(f"[yellow]No comments found for contribution {contribution_id}.[/yellow]")

    comment_lines = "\n".join(
        f"  {i}. {escape(text)}" for i, text in enumerate(comment_texts, start=1)
    ) if comment_texts else "  (none)"
    console.print(Panel(
        f"[bold]Title:[/bold] {escape(str(contribution.get('title') or ''))}\n"
        f"[bold]Comments ({len(comment_texts)}):[/bold]\n{comment_lines}",
        title=f"[bold cyan]Contribution {contribution_id}[/bold cyan]",
    ))
    return {"contribution": contribution, "comments": comment_texts}


def run_get_contribution_with_comments() -> None:
    """Prompt for an optional contribution ID and run the fetch+comments flow."""
    raw = input("Contribution ID (blank for random): ").strip()
    if raw == "":
        get_contribution_with_comments(None)
    else:
        try:
            cid = int(raw)
        except ValueError:
            console.print("[red]Invalid contribution ID — please enter a whole number.[/red]")
            return
        get_contribution_with_comments(cid)


def create_contributions_with_comments_dataset() -> None:
    """Build a dataset of sampled contributions joined with their comments.

    Loads the fetched contributions file, prompts for a sample size, then
    delegates to :class:`ContributionsWithCommentsDataset`. Aborts with a
    message if the file is missing, and skips if the user declines to overwrite
    an existing output.
    """
    if not CONTRIBUTIONS_PATH.exists():
        console.print(f"[red]No contributions file found at {CONTRIBUTIONS_PATH}. Run option 1 (Fetch contributions) first.[/red]")
        return

    n = _prompt_positive_int("Number of contributions to sample (n): ")

    # Pass source_path explicitly so tests that redirect main.CONTRIBUTIONS_PATH
    # into tmp_path read the same file this function validated above.
    dataset = ContributionsWithCommentsDataset(n=n, source_path=CONTRIBUTIONS_PATH)

    if dataset.output_path.exists() and not _prompt_overwrite(dataset.output_path):
        console.print("[yellow]Skipping — existing file kept.[/yellow]")
        return

    try:
        result_path = dataset.build()
        console.print(f"[green]Dataset written to {result_path}[/green]")
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Failed: {e}[/red]")


MENU_OPTIONS = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "q"}


def run_menu() -> str:
    """Display the interactive menu and return the user's validated choice.

    Loops until the user enters a valid option, re-prompting on bad input.

    Returns:
        The selected option, one of "1"-"13" or "q".
    """
    while True:
        console.print(Panel(
            "[bold]1[/bold] — Fetch contributions\n[bold]2[/bold] — Run stance detection\n[bold]3[/bold] — Sample contributions\n[bold]4[/bold] — Extract key messages\n[bold]5[/bold] — Inspect key messages\n[bold]6[/bold] — Fetch contribution with comments\n[bold]7[/bold] — Build contributions-with-comments dataset\n[bold]8[/bold] — Inspect parquet file\n[bold]9[/bold] — Run whole process\n[bold]10[/bold] — Export parquet file to Excel\n[bold]11[/bold] — Judge key messages\n[bold]12[/bold] — Compare judge results\n[bold]13[/bold] — Run whole process (policy v2)\n[bold]q[/bold] — Quit",
            title="[bold cyan]DIPAS Analysis[/bold cyan]",
        ))
        choice = input("Select option: ").strip().lower()
        if choice in MENU_OPTIONS:
            return choice
        console.print("[red]Invalid choice — enter 1-13 or q.[/red]")


def main() -> None:
    """Thin synchronous dispatcher: read the menu choice and run the action.

    Kept deliberately small so the menu, fetch, and stance-detection flows can be
    unit-tested in isolation.
    """
    choice = run_menu()
    if choice == "1":
        fetch_contributions()
    elif choice == "2":
        # run_stance_detection is a coroutine; asyncio.run() drives it to
        # completion from this synchronous entry point and owns the event loop.
        asyncio.run(run_stance_detection())
    elif choice == "3":
        sample_contributions()
    elif choice == "4":
        # run_key_message_extraction is a coroutine driven by asyncio.run() from
        # this synchronous entry point.
        asyncio.run(run_key_message_extraction())
    elif choice == "5":
        inspect_key_messages()
    elif choice == "6":
        run_get_contribution_with_comments()
    elif choice == "7":
        create_contributions_with_comments_dataset()
    elif choice == "8":
        inspect_parquet()
    elif choice == "9":
        # run_whole_process is a coroutine driven by asyncio.run() from this
        # synchronous entry point.
        asyncio.run(run_whole_process())
    elif choice == "10":
        export_parquet_to_excel()
    elif choice == "11":
        # run_key_message_judge is a coroutine driven by asyncio.run() from this
        # synchronous entry point.
        asyncio.run(run_key_message_judge())
    elif choice == "12":
        # run_compare_judge_results is synchronous (no LLM/network calls), so it
        # is called directly rather than driven by asyncio.run().
        run_compare_judge_results()
    elif choice == "13":
        # run_whole_process_v2 is a coroutine driven by asyncio.run() from this
        # synchronous entry point.
        asyncio.run(run_whole_process_v2())
    else:
        console.print("[yellow]Goodbye.[/yellow]")


if __name__ == "__main__":
    main()
