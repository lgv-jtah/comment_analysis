# Phase 5 screening results: policy-v2 vs. historical baseline (mistral-nemo)

STANCE_PIPELINE_IMPLEMENTATION_BRIEF.md section 10, step 3: compare the
historical saved baseline, the policy-v2 one-call classifier, the policy-v2
scope/attribution-plus-stance (two-stage) engine, and a best-development
configuration with an optional independent verifier, on the existing
grouped folds (`dataset/folds_v1.json`).

## Screening subset

A full run on all 257 comments takes 30-70+ minutes per variant (the
pipeline runs single-concurrency: `asyncio.Semaphore(1)` in
`benchmark.py`, by design, to respect IONOS/proxy rate limits). Per the
brief's own allowance ("If costs require a staged search, define the
development-only screening subset before inspecting results"), fold 0's
23 test contributions (138 pairs / 40 comments) were fixed as the
screening subset *before* any candidate-model result was inspected. The
historical baseline itself was already scored on all 5 folds (see below)
since it requires no LLM calls.

Artifacts:
- `dataset/experiments/screening_subset_fold0_source.parquet` -- the
  frozen extraction source restricted to fold 0's test contributions.
- `dataset/experiments/screening_fold0_mistral-nemo_{one_call,two_stage,two_stage_verifier}.parquet`
  -- raw predictions per variant.
- `dataset/experiments/historical_baseline_predictions.parquet` -- the
  full-cohort historical baseline (all 804 rows).

## Results (mistral-nemo, fold 0 test set, 119 eligible pairs)

| Variant | forced_choice_accuracy | accepted_accuracy | labelled_coverage | review_rate |
|---|---|---|---|---|
| **Historical baseline** (all 5 folds pooled, 721 eligible) | **0.732** (528/721) | 0.732 | 1.000 | 0.000 |
| Historical baseline (fold 0 only, for direct comparison) | **0.748** (89/119) | 0.748 | 1.000 | 0.000 |
| policy-v2 one-call | 0.504 (60/119) | 0.583 (60/103) | 0.866 | 0.159 |
| policy-v2 two-stage | 0.429 (51/119) | 0.560 (51/91) | 0.765 | 0.254 |
| policy-v2 two-stage + verifier | 0.454 (54/119) | 0.574 (54/94) | 0.790 | 0.232 |

**All three new policy-v2 variants substantially underperform the
historical baseline on this screening subset.** This held consistently
across all three variants, so it is very unlikely to be noise in any one
run.

## Root-cause breakdown

Two distinct, well-understood effects explain most (not all) of the gap.

### 1. Historical labels don't follow the new general-approval policy

Per brief section 3: "Existing annotations do not fully match the new
general-approval policy. Report results against historical labels
separately from policy-compliance checks." This predicted exactly what
was observed: when a comment is a blanket agreement ("Stimmt.", "In der
Tat.") with no reference to a specific key message, the new policy
correctly (per its own specification) propagates `in_favor` /
`general_approval` to *every* key message in the contribution. The
historical human annotation (Antonie) did not apply this rule -- she
labelled many of the same pairs `neutral`.

Quantified: excluding **every** error whose `basis == "general_approval"`
(i.e. best case, crediting the policy engine for all of them):

| Variant | errors | general_approval errors | accuracy if those are excluded |
|---|---|---|---|
| one_call | 59 | 8 (14%) | 0.571 |
| two_stage | 68 | 25 (37%) | 0.639 |
| two_stage+verifier | 65 | 25 (38%) | 0.664 |

So this effect is real and non-trivial (up to 38% of errors for the
two-stage variants), but **even crediting it fully still leaves accuracy
well below the historical baseline's 0.748** -- something else is also
driving the gap.

### 2. Missing parent/reply context drives high review-routing

This was flagged as an open risk before any Phase-5 run (see the
implementation plan's "Open risks" #1, confirmed by reading
`comment_fetcher.py`: no parent/reply-linkage data source exists, so
`ParentContext.available` is always `False`). It shows up concretely now:
of the two-stage variant's wrong predictions, 28 of 68 (41%) have
`status=needs_review` with `review_reasons=["missing_context"]` --  the
scope-attribution stage, honestly, cannot resolve whether a short reply
("Mit freundlichen Grüßen...", "Die Baustelle der U5 erfordert
leider...") is agreeing/disagreeing with a specific key message without
knowing what it's replying to, and correctly declines rather than
guessing. Since `forced_choice_accuracy` counts every `needs_review` pair
as incorrect (no proposed_stance to compare), this abstention behavior
directly costs accuracy under this metric even though it is not a wrong
guess.

## Interpretation

This is not (as far as investigated) an implementation bug: the policy
engine is doing what sections 2 and 4 of the brief specify (general
approval propagates; missing context routes to review rather than
guessing). The gap versus the historical baseline reflects two things the
brief itself anticipated as risks: a genuine policy/label semantics
mismatch, and a real data-availability gap (no reply-context source).
After controlling for the first, a real accuracy shortfall remains
(0.57-0.66 vs. 0.748), which needs further investigation before this
architecture can be called competitive with the existing system.

## Status

**Not yet decided which direction to pursue.** Before spending further
IONOS budget on the section 10 step 4 cross-model comparison
(llama-31-405b-fp8, gptoss-120b) -- which would repeat this same
screening-subset methodology per model, another 15-25 min of live calls
each -- this result needs to be discussed, since running the same
setup on more models won't resolve either root cause above (both are
prompt/architecture/data issues, not model-choice issues). Candidate next
steps, not yet started:
- Tighten the scope-attribution prompt's `missing_context` trigger
  (currently fires on ordinary short replies with no parent-context
  signal at all, which may be over-conservative given
  `parent_context.available` is always `False` anyway -- the model has
  no way to *not* be missing that context).
- Decide how to report the general-approval/historical-label divergence
  in the final report: as brief section 3 requires, report both numbers
  (against-historical and policy-compliant) rather than picking one.
- Only after one of the above is addressed, re-run the fold-0 screening
  comparison to confirm improvement, then proceed to the full-cohort and
  cross-model comparisons.
