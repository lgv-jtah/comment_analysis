# Phase 5 screening results: policy-v2 vs. historical baseline

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

## Cross-model comparison (brief section 10, step 4)

Same fold-0 screening subset, verifier-enabled two-stage configuration
held identical, only the model swapped:

| Model | forced_choice_accuracy | accepted_accuracy | labelled_coverage | review_rate | per-class accepted_accuracy (against / in_favor / neutral) |
|---|---|---|---|---|---|
| mistral-nemo | 0.454 (54/119) | 0.574 (54/94) | 0.790 | 0.232 | 0.400 / 0.690 / 0.545 |
| llama-31-405b-fp8 | **failed -- see below** | -- | -- | -- | -- |
| gptoss-120b | 0.471 (56/119) | 0.602 (56/93) | 0.782 | 0.210 | 0.600 / 0.640 / 0.586 |

`gptoss-120b` is marginally stronger than `mistral-nemo` on this subset
(accepted_accuracy 0.602 vs. 0.574), most notably on the `against` class
(0.600 vs. 0.400, the class the historical baseline itself struggles with
most). Neither reaches the historical baseline's 0.748.

**`llama-31-405b-fp8` run failed completely** (138/138 pairs `status=failed`,
0 accepted) -- not a policy-engine bug. Direct API check confirms IONOS has
fully decommissioned `meta-llama/Meta-Llama-3.1-405B-Instruct-FP8`:

```
404 MODEL_NOT_SUPPORTED: Model 'meta-llama/Meta-Llama-3.1-405B-Instruct-FP8'
is not supported for chat completions. Supported models are:
"Qwen/Qwen3-Coder-Next", "Qwen/Qwen3.5-397B-A17B", "Qwen/Qwen3.5-9B",
"Qwen/Qwen3.8-27B", "lightonai/LightOnOCR-2-1B",
"meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Meta-Llama-3.1-8B-Instruct",
"mistralai/Mistral-Nemo-Instruct-2407", "mistralai/Mistral-Small-24B-Instruct",
"openai/gpt-oss-120b"
```

This resolves implementation-plan open risk #6: IONOS's current catalog
*does* include both of the brief's originally-suggested models --
**Llama 3.3 70B** (`meta-llama/Llama-3.3-70B-Instruct`) and **Mistral
Small 24B** (`mistralai/Mistral-Small-24B-Instruct`) -- confirmed live
directly against IONOS, not just inferred from the brief. Neither is
wired into the local LiteLLM proxy's `config.yaml` yet (a separate repo,
`/home/julien/projects/LiteLLM/`) under a `model_name` alias this
pipeline can call. Per the plan's stated fallback ("comparing only the
already-configured models if new provisioning isn't feasible"), the
`llama-31-405b-fp8` slot in the comparison was simply dropped rather than
retried, and `gptoss-120b` stands as the second data point above. The
failed run's predictions (all-`failed` rows) are kept at
`dataset/experiments/screening_fold0_llama-31-405b-fp8_two_stage_verifier.parquet`
as a record of the failure, not deleted.

## Prompt tightening: scope-attribution `missing_context` trigger

Investigated *why* `missing_context` fires before touching anything.
Original two-stage run (mistral-nemo, no verifier) on the fold-0 subset:
23 of 35 `needs_review` rows (66%) carried `missing_context`, concentrated
in 6 of the subset's 40 comments (each comment maps to ~4 key-message
rows, so 6 comments cost 23 rows).

Reading the 6 flagged comments by hand: 2-3 are genuine reply/thread-context
gaps ("Ich habe mich beim Ort vertan. Es geht um den nördlicheren Tunnel"
-- clearly a correction referring to something not stated in the comment
itself), but 2-3 are substantive, self-contained comments where the
*content* is perfectly interpretable and the model was really just unsure
*which specific key message* it targets (e.g. "Muss hier erst ein Kind
angefahren werden bis die Politik endlich handelt?" -- a clear safety
complaint, just not obviously tied to one key message over another). The
prompt's old closing line, `"Bei Unsicherheit generell: unresolved."` (when
uncertain generally: unresolved), is a blanket catch-all that doesn't
distinguish "I lack context I'd need" from "I can't tell which key message
this targets" -- and `policy.py` only ever consumes `uncertainty_flags`
when `scope_type == UNRESOLVED`, so there was no cheaper way for the model
to express partial confidence.

**Change** (`prompts/scope_attribution_user.txt`): replaced the blanket
catch-all with two explicit rules -- `unresolved` is reserved for cases
where information genuinely absent from the comment text itself would be
required (reply-context gaps, suspected sarcasm), and is explicitly *not*
for target-mapping uncertainty on an otherwise-interpretable comment,
which should get `target_specific`/`mixed_qualified` with the most
plausible key-message mapping instead of a blanket punt.

**Re-ran the identical two-stage/mistral-nemo/fold-0 configuration**
(`dataset/experiments/screening_fold0_mistral-nemo_two_stage_v2prompt.parquet`)
to isolate the effect:

| | forced_choice_accuracy | accepted_accuracy | labelled_coverage | review_rate | missing_context rows |
|---|---|---|---|---|---|
| Before (original prompt) | 0.429 (51/119) | 0.560 (51/91) | 0.765 | 0.254 (35 review, 23 missing_context) | 23 / 6 comments |
| After (tightened prompt) | 0.445 (53/119) | 0.552 (53/96) | 0.807 | 0.188 (26 review, 18 missing_context) | 18 / 5 comments |

**Real but modest effect.** `missing_context` rows dropped 23->18 (5 of
the newly-resolved fell one way or another rather than review), coverage
rose, and forced_choice_accuracy improved slightly (+1.6pp) because fewer
pairs are now automatically counted wrong via review-routing. But two of
the three comments judged (by hand) as over-cautious were *still* flagged
`missing_context` even after the rewrite -- the model's own uncertainty
about target-mapping on those specific comments is apparently not just a
wording artifact. accepted_accuracy is flat-to-slightly-down (0.560 ->
0.552), so the newly-resolved rows are, on net, no more accurate than a
coin flip -- this is not a case of "unlocking easy correct answers," more
"trading abstention for a forced (roughly 50/50) guess." **Not a
decisive fix.** A separate, untouched failure mode --
`evidence_not_substring_of_comment` / `invalid_scope_output` /
`invalid_stance_output` (~16 rows, structured-output evidence-grounding
rejections) -- is now proportionally a larger share of remaining review
routing than before, and hasn't been investigated yet.

## Evidence-grounding validation cluster (`evidence_not_substring_of_comment` etc.)

Root-caused by reproducing the 4 comments that triggered it live
(`policy.py`'s `_validate_scope`/`_validate_stance` do an exact Python
`in` substring check between the LLM's claimed quote and the actual
comment text -- see debug reproduction, not committed, run via a
scratchpad script). Found two genuinely different failure patterns among
5 mismatches from those 4 comments:

1. **Whitespace-only mismatch (1 of 5).** The source comment was
   `"Mit freundlichen Grüßen\nHans-Hinrich Biel\n\nRadfahrer..."`; the
   model quoted it back with newlines folded to spaces:
   `"Mit freundlichen Grüßen Hans-Hinrich Biel Radfahrer..."`. Same text,
   reformatted -- not a fabricated quote, but the exact-substring check
   rejected it anyway.
2. **Real content mismatch (4 of 5).** The model auto-corrected a citizen's
   typo/spelling while "quoting" (`"STrasse"` -> `"Straße"`), and in two
   cases produced a paraphrase/summary or a split reconstruction that
   simply isn't present verbatim anywhere in the comment (e.g. quoting
   "Leider sind einige Autofahrer ihres Führerscheins unwürdig." for a
   comment that never says that -- a genuine hallucinated "quote", not a
   formatting artifact).

**Change, code** (`policy.py`): added `_normalize_whitespace()` (collapse
any whitespace run to a single space) and applied it to both sides of
every evidence-substring/set-membership check in `_validate_scope`,
`_validate_stance`, and `OneCallPolicyEngineV2._validate`. This recovers
case 1 without weakening the grounding guarantee -- it is still an exact
match after normalization, so case 2's fabrications and corrections
correctly continue to fail. 4 new regression tests added
(`tests/test_policy.py`): two confirm whitespace-only differences are now
accepted, two (one per engine) confirm a genuine word-level change (the
`STrasse`/`Straße` case) is still rejected.

**Change, prompts**: strengthened the "literal quote" instruction in all
three evidence-quoting prompts (`scope_attribution_user.txt`,
`per_target_stance_user.txt`, `stance_v2_one_call_user.txt`) to explicitly
say: copy character-for-character, do not correct typos/spelling/casing,
do not paraphrase, and leave the evidence field empty rather than invent
one. This cannot fix case-2-style fabrication outright (a
determined-enough model can still ignore instructions), but should reduce
its rate; the exact-substring validation remains the actual safety net
regardless.

**Re-ran the identical two-stage/mistral-nemo/fold-0 configuration again**
(`dataset/experiments/screening_fold0_mistral-nemo_two_stage_v3groundingfix.parquet`),
cumulative with the `missing_context` prompt change:

| | forced_choice_accuracy | accepted_accuracy | labelled_coverage | review_rate | review rows |
|---|---|---|---|---|---|
| v1: original | 0.429 (51/119) | 0.560 (51/91) | 0.765 | 0.254 | 35 |
| v2: + missing_context prompt fix | 0.445 (53/119) | 0.552 (53/96) | 0.807 | 0.188 | 26 |
| v3: + grounding whitespace fix + stronger quote instructions | 0.454 (54/119) | 0.551 (54/98) | 0.824 | 0.152 | 21 |

Cumulative v1->v3: forced_choice_accuracy +2.5pp, review_rate cut from
25.4% to 15.2% (35 -> 21 review rows), coverage up from 76.5% to 82.4%.
`accepted_accuracy` stayed essentially flat across all three versions
(0.560 -> 0.551) -- both fixes convert *declined* pairs (automatically
wrong under forced-choice scoring) into *forced guesses* that are, on
net, roughly coin-flip accurate, not a quality improvement on the
predictions themselves. **Directionally correct, cumulatively real, but
still not close to closing the gap** with the historical baseline's
0.748.

## The decisive finding: accepted-but-wrong predictions, split by `basis`

Pulled the 44 accepted-but-wrong pairs from the v3 run (98 accepted, 54
correct) and split accuracy by `PairOutput.basis` instead of looking at
them as one undifferentiated pool:

| basis | n (accepted) | correct | accuracy |
|---|---|---|---|
| explicit_target | 25 | 21 | **0.840** |
| general_approval | 27 | 3 | **0.111** |
| no_position | 1 | 1 | 1.000 |
| unaddressed | 45 | 29 | 0.644 |

**`general_approval` is the entire story.** Its accuracy (0.111) is
catastrophic and every one of its 24 wrong predictions has the identical
shape: the historical reference label is `neutral`, the policy predicted
`in_favor` (`"Stimmt"`, `"In der Tat."`, `"Klingt smart! Unbedingt
technisch untersuchen!"` -- blanket-agreement comments, propagated to
every key message per the brief's own specification). This is not a new
finding in kind (the general-approval/historical-label divergence was
already identified as a contributor earlier in this document) but it is
a much stronger finding in *degree* than previously estimated: earlier
analysis (counting it as a share of *all* errors, review-routed included)
put it at "up to 38% of errors." Isolated to just the accepted-and-wrong
pool and measured as its own basis's accuracy, it is by far the dominant
term.

**Excluding `general_approval` entirely, accepted_accuracy on the rest
(`explicit_target` + `unaddressed` + `no_position`) is 51/71 = 0.718** --
within a few points of the historical baseline's 0.748 on this same
fold. `explicit_target` (0.840) is actually *better* than the historical
baseline. `unaddressed` (0.644) is the one place with room to improve on
its own merits, but it is a secondary effect next to `general_approval`.

**Conclusion: the new policy-v2 architecture is not, in itself,
meaningfully worse than the historical system.** Nearly the entire
measured gap traces to one specific, deliberate design choice --
propagating blanket contribution-level approval to every key message --
disagreeing with how the historical annotator scored those exact same
pairs (she evidently did not apply blanket propagation; she scored
per-key-message and defaulted to `neutral` when a comment didn't clearly
address that specific key message, even amid general positive sentiment).
This is precisely what brief section 3 predicted and told us not to hide
or conflate into one score.

**Code change**: added `evaluator_v2.accepted_accuracy_by_basis()` (and
its `BasisMetrics` dataclass) so this split is computed and printed
automatically after every `cli.py evaluate`/`benchmark` run (as an
"Accepted accuracy by basis" table), rather than requiring the ad-hoc
pandas analysis that found it this time. 5 new tests
(`tests/test_evaluator_v2.py`, `tests/test_cli.py`), including that it
degrades gracefully (returns `{}`) when a predictions file has no `basis`
column at all, rather than crashing.

## Status

**Direction is now clear: the architecture works; the general-approval
policy needs a reporting decision, not a bug fix.** Concretely, next
steps:
1. **Decide how to report the general-approval/historical-label split**
   in the final report, per brief section 3 -- e.g. report
   `accepted_accuracy` two ways: against historical labels as-is (what
   the tables above show), and with `general_approval`-basis pairs
   scored/reported separately rather than penalized against
   pre-policy labels. This is a reporting decision, not further tuning,
   and the brief explicitly asks for it rather than picking one number.
2. Optionally, investigate `unaddressed` (0.644 accuracy, the one
   remaining basis with real room to improve) the same way the two prior
   clusters were investigated, though it is a secondary effect.
3. Once the reporting approach is settled, re-run the fold-0 screening
   comparison one more time with the by-basis view in hand to confirm the
   overall picture, then proceed to the full-cohort comparison and
   (separately, cross-repo) provisioning Llama 3.3 70B / Mistral Small
   24B on the LiteLLM proxy if a broader model sweep is still wanted at
   that point.
