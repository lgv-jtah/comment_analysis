"""Stance-detection entry point with the general-reaction fast-path gate."""
from stance_detector import StanceDetector
from llm_handler import LLMHandler
from evaluator import StanceEvaluator, RunMetadata, write_markdown_report
import httpx
import asyncio
import time

# Fast-path gate thresholds (see general_reaction_gate_konzept.md, Task 1.3).
# NOTE: 0.6 is an interim placeholder, NOT a tuned production value. The
# calibration against gold labels is Task 1.1 in the concept doc; until that
# lands, do not treat this number as final.
GR_SCORE_THRESHOLD = 0.6
FAST_PATH_MAX_WORDS = 5
FAST_PATH_STANCE = "in favor"
FAST_PATH_EXPLANATION = "Allgemeine positive Reaktion auf den gesamten Beitrag"

# DEBUG / TEMPORARY — for manual test runs only. Remove before any full
# production run: this caps the loop at the first few rows instead of all 155.
DEBUG_ROW_LIMIT = 10


def should_fast_path(
    general_reaction_label: str, general_reaction_score: float, comment_text: str
) -> bool:
    """Decide whether a comment can skip the LLM via the fast path.

    Implements "Pfad 1 — Fast Path" from general_reaction_gate_konzept.md
    (Task 1.3): a short comment that the upstream general-reaction classifier
    confidently labelled as a positive reaction is assumed "in favor" without
    a dedicated stance-detection LLM call.

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


async def main():
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
    data = data[:DEBUG_ROW_LIMIT]  # DEBUG / TEMPORARY — see DEBUG_ROW_LIMIT above
    prompt_key = "base"
    prompt_config = stance_detector.load_prompt_config(
        "./prompts/config.json", prompt_key
    )

    base_url = "http://10.200.160.182:5555/v1"
    model_name = "Qwen/Qwen3.5-27B-FP8"

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
                print(
                    f"[{index}] FAST-PATH  stance={FAST_PATH_STANCE}  (gr_score={gr_score})  (elapsed={iter_elapsed:.2f}s)"
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
                print(
                    f"[{index}] ENSEMBLE   stance={result.get('stance') if result else None}  (gr_score={gr_score})  (elapsed={iter_elapsed:.2f}s)"
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
                print(
                    f"[{index}] LLM        stance={result.get('stance') if result else None}  (elapsed={iter_elapsed:.2f}s)"
                )

    # Measured after the `async with` exits so the HTTP client's connection
    # teardown is included in the total.
    total_elapsed = time.time() - loop_start
    print(
        f"Fast-pathed: {fast_path_count}  |  ensemble: {ensemble_count}  "
        f"|  LLM calls: {llm_count}  "
        f"|  total: {fast_path_count + ensemble_count + llm_count}"
    )
    print(f"Total loop time: {total_elapsed:.2f}s")

    evaluator = StanceEvaluator()
    report = evaluator.evaluate(predictions)
    print(evaluator.format_report(report))

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
    print(f"Markdown report written to {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
