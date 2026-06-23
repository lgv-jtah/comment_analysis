"""Stance-detection entry point with the general-reaction fast-path gate."""
from stance_detector import StanceDetector
from llm_handler import LLMHandler
from evaluator import StanceEvaluator, RunMetadata, write_markdown_report, write_result_json
import httpx
import asyncio
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from create_dataset import CreateDataset
import pandas as pd
from sample_contributions import SampleContributions
from key_message_extractor import KeyMessageExtraction

console = Console()

# Resolve relative to this file (not the CWD) so paths are stable no matter
# which directory the app is launched from.
DATA_DIR = Path(__file__).resolve().parent / "data"
CONTRIBUTIONS_PATH = DATA_DIR / "contributions.parquet"

# LLM endpoint configuration, shared by stance detection and key-message
# extraction. Kept at module level so both flows reference the same values.
LLM_BASE_URL = "http://217.160.202.33:5555/v1"
LLM_MODEL_NAME = "Qwen/Qwen3.5-27B-FP8"

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
    # data = data[:DEBUG_ROW_LIMIT]  # DEBUG / TEMPORARY — see DEBUG_ROW_LIMIT above
    prompt_key = "base"
    prompt_config = stance_detector.load_prompt_config(
        "./prompts/config.json", prompt_key
    )

    base_url = LLM_BASE_URL
    # base_url = "http://10.200.160.182:5555/v1"
    model_name = LLM_MODEL_NAME

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
                    prompt_config=prompt_config, item=item, gr_hint=hint
                )
                result = await llm_handler.call_model(
                    client=client,
                    base_url=base_url,
                    model_name=model_name,
                    system=system,
                    messages=messages,
                    semaphore=semaphore,
                    enable_thinking=True,
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
                    prompt_config=prompt_config, item=item
                )
                result = await llm_handler.call_model(
                    client=client,
                    base_url=base_url,
                    model_name=model_name,
                    system=system,
                    messages=messages,
                    semaphore=semaphore,
                    enable_thinking=True,
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

    result_path = write_result_json(data, predictions)
    console.print(f"[green]Result JSON written to[/green] {result_path}")


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
    ``explanation`` columns. Aborts early if no file is selected, the column is
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

    output_path = DATA_DIR / f"key_messages_{source_path.stem}.parquet"
    if output_path.exists() and not _prompt_overwrite(output_path):
        console.print("[yellow]Skipping extraction — existing file kept.[/yellow]")
        return

    extractor = KeyMessageExtraction(base_url=LLM_BASE_URL, model_name=LLM_MODEL_NAME)
    semaphore = asyncio.Semaphore(1)

    key_messages_col: list[list[dict] | None] = []

    total = len(df)
    async with httpx.AsyncClient() as client:
        for index, (_, row) in enumerate(df.iterrows(), start=1):
            # Treat missing content as an empty string so a NaN cell does not
            # crash the extractor; the LLM simply receives no contribution text.
            contribution = "" if pd.isna(row["contributionContent"]) else str(row["contributionContent"])
            result = await extractor.extract(
                client=client,
                contribution=contribution,
                proceeding_description=description,
                semaphore=semaphore,
            )
            if result is not None:
                key_messages_col.append(result)
                console.print(f"[{index}/{total}] {len(result)} key messages found")
            else:
                key_messages_col.append(None)
                console.print(f"[red][{index}/{total}] FAILED[/red]")

    result_df = df.copy()
    result_df["key_messages"] = key_messages_col
    result_df.to_parquet(output_path)
    console.print(f"[green]Key messages saved to {output_path}[/green]")


def sample_contributions() -> None:
    """Draw a random sample of contributions for chosen projects.

    Loads the fetched contributions file, prompts for project IDs and a sample
    size, then delegates to :class:`SampleContributions`. Aborts with a message
    if the file is missing or lacks the ``project_id`` column.
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

    try:
        # Pass source_path explicitly rather than relying on SampleContributions'
        # default: tests redirect main.CONTRIBUTIONS_PATH into tmp_path, and the
        # sampler must read the same file this function already validated above.
        sampler = SampleContributions(project_ids=ids, n=n, source_path=CONTRIBUTIONS_PATH)
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


MENU_OPTIONS = {"1", "2", "3", "4", "q"}


def run_menu() -> str:
    """Display the interactive menu and return the user's validated choice.

    Loops until the user enters a valid option, re-prompting on bad input.

    Returns:
        The selected option, one of "1", "2", "3", "4", or "q".
    """
    while True:
        console.print(Panel(
            "[bold]1[/bold] — Fetch contributions\n[bold]2[/bold] — Run stance detection\n[bold]3[/bold] — Sample contributions\n[bold]4[/bold] — Extract key messages\n[bold]q[/bold] — Quit",
            title="[bold cyan]DIPAS Analysis[/bold cyan]",
        ))
        choice = input("Select option: ").strip().lower()
        if choice in MENU_OPTIONS:
            return choice
        console.print("[red]Invalid choice — enter 1, 2, 3, 4, or q.[/red]")


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
    else:
        console.print("[yellow]Goodbye.[/yellow]")


if __name__ == "__main__":
    main()
