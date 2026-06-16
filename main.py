"""Stance-detection entry point with the general-reaction fast-path gate."""
from stance_detector import StanceDetector
from llm_handler import LLMHandler
from evaluator import StanceEvaluator
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
DEBUG_ROW_LIMIT = 3


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


async def main():
    """Run stance detection over the gold-label dataset.

    Loads the data (capped at DEBUG_ROW_LIMIT for manual test runs), applies the
    fast-path gate per row, and either emits the canned fast-path result or calls
    the LLM, then prints a summary of how many rows took each path along with
    per-iteration and total loop timing.
    """
    stance_detector = StanceDetector()
    llm_handler = LLMHandler()

    data = stance_detector.load_data()
    data = data[:DEBUG_ROW_LIMIT]  # DEBUG / TEMPORARY — see DEBUG_ROW_LIMIT above
    prompt_config = stance_detector.load_prompt_config("./prompts/config.json", "base")

    base_url = "http://10.200.160.182:5555/v1"
    model_name = "Qwen/Qwen3.5-27B-FP8"

    fast_path_count = 0
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
        f"Fast-pathed: {fast_path_count}  |  LLM calls: {llm_count}  "
        f"|  total: {fast_path_count + llm_count}"
    )
    print(f"Total loop time: {total_elapsed:.2f}s")

    evaluator = StanceEvaluator()
    report = evaluator.evaluate(predictions)
    print(evaluator.format_report(report))


if __name__ == "__main__":
    asyncio.run(main())
