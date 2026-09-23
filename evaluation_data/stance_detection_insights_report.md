# Stance Detection Evaluation: Insights from Antonie's Annotations

2026-09-22 · Julien

Across 804 stance judgments on 119 contributions, the stance-detection stage is correct 64.7% of the time — the clear bottleneck, since key-message extraction alone is 97.2% accurate. The dominant error is a false positive: the model invents a directional stance (in favor / against) when a comment doesn't actually address that specific key message, most often by cross-applying one comment's sentiment across every key message in a multi-key-message contribution.

## Dataset & methodology

Two related sources were used:

| Source | Rows | Content |
| --- | --- | --- |
| `evaluation_data/evaluations_backup20260922.sq3` (`stance_evaluations` table) | 804 | Current, complete annotation set: for each (key message, comment) pair the model scored, an evaluator recorded `correct` / `incorrect`, and optionally a `corrected_label` and free-text `comment` explaining the call |
| `evaluations_export_mistral-nemo_20260908_131352.pkl` | 296 (subset) | An earlier export that additionally carries the model's actual predicted label (`stance`) and its explanation — used here only to build the precision-by-label breakdown, since the live sqlite table doesn't store the raw prediction |

Annotations were done almost entirely by **Antonie** (797 of 804 rows); Julien annotated 7 rows early on as a calibration check on contribution 83. Evaluations span **119 distinct contributions**. For comparison, the same evaluator pool judged **356 key-message extractions**, correct 97.2% of the time — confirming that extraction is not where the pipeline is losing accuracy.

## Headline results

**Overall stance-detection accuracy: 64.7%** (520 correct / 284 incorrect of 804 judgments).

Accuracy is not uniform across the three stance labels. Using the 296-row subset that carries the model's actual predicted label:

| Predicted label | Share of predictions | Precision (verdict = correct) |
| --- | --- | --- |
| neutral | 34% | **76.5%** |
| in favor | 56% | 56.3% |
| ablehnung (against) | 9% | **44.4%** |

`neutral` is the model's most trustworthy call; `ablehnung` (against) is the least trustworthy, despite being predicted rarely. The model also predicts "in favor" far more than it predicts "against" (56% vs. 9% of all predictions), consistent with participatory-platform comments skewing supportive — but this same skew is also where most of the errors concentrate (see next section).

## Dominant failure mode: false directional stance

When the model is wrong, the true label is overwhelmingly **neutral** — it isn't confusing "in favor" with "against," it's inventing a directional stance where the comment doesn't actually take one on that specific key message.

- Of errors where Antonie specified the correct label, **\~70%+ should have been neutral** (136 explicit `corrected_label = neutral` rows, plus dozens more where she wrote "neutral" in the free-text comment instead of using the dropdown — see Data quality below).
- In the 296-row prediction subset, of all incorrect predictions **65% were predicted "in favor"** (vs. a 56% base rate) — the model skews toward over-eager positive stance-taking specifically when it's wrong.
- Missed genuine directional stances are comparatively rare: only 21 errors had a true label of "in favor" and 32 had a true label of "ablehnung" (against) — far fewer than the 131+ errors whose true label was neutral.

In short: the model's main weakness is a **false-positive problem**, not a left/right confusion. It is too willing to read a stance into a comment that isn't actually engaging with the key message in question.

## Root cause: cross-key-message contamination

Antonie's free-text notes on incorrect cases point to a specific, recurring mechanism: in contributions with **multiple key messages**, the model frequently takes a comment that's topically about key message A and also assigns it a directional stance on key messages B, C, ... in the same contribution — broadcasting one sentiment across every key message instead of checking topical relevance per key message.

- **\~28%** of commented errors literally say the stance "bezieht sich auf andere Keymessage" (refers to a different key message).
- **\~19%** describe a general positive reaction ("generelle/allgemeine zustimmende Reaktion") that isn't a specific stance on *this* claim — including cases like a comment reading only "Siehe oben" ("see above"), or a comment complaining about an unrelated site bug, both scored as "in favor."

Contribution-level error rates make the pattern concrete: contributions with many key messages sharing few comments are where accuracy collapses, while contributions with narrowly-scoped comments are near-perfect.

| Contribution | Stance judgments | Errors | Error rate | Pattern |
| --- | --- | --- | --- | --- |
| 22643 | 6 | 6 | 100% | Same comments re-scored against every key message in the contribution |
| 8627 | 8 | 7 | 88% | Off-topic remarks (e.g. about traffic-light timing) scored against unrelated key messages |
| 4573 | 6 | 5 | 83% | Cross-key-message contamination |
| 2545 | 12 | 9 | 75% | Cross-key-message contamination |
| 11005 | 20 | 12 | 60% | Cross-key-message contamination |
| 5525 | 30 | 17 | 57% | Largest contribution in the set; same pattern at scale |
| 1339, 1906, 2046, 2638, 2780, 2833, 3552, 4572, 4924, 14491 | 4–7 each | 0 | 0% | Few, narrowly-scoped key messages per comment |

This is the single most actionable finding: the failure isn't in judging sentiment, it's in judging **relevance** — whether a given comment is even about the key message being scored.

## Secondary findings & data-quality gaps

- **"Against" is the least reliable label** (44% precision) despite being predicted rarely (9% of cases) — likely because negative comments are scarce in the data, so the model may be triggering on general criticism *somewhere* in a contribution rather than criticism of the specific claim being scored.
- **Key-message extraction is not the bottleneck.** At 97.2% accuracy it is already near ceiling, so further gains from improving extraction quality will be small; the return on effort is much higher on the stance-classification stage.
- **`corrected_label` is under-used in the annotation tool.** 101 of 284 "incorrect" rows have `corrected_label = NULL`, but Antonie recorded the correct label in the free-text `comment` field almost every time instead. This isn't a judgment gap, just a structured-data gap — worth enforcing the field on every "incorrect" verdict, or backfilling this batch by parsing the comment text.
- **Label vocabulary is inconsistently mixed German/English**: `ablehnung` alongside `in favor` / `neutral`. Worth normalizing to one vocabulary before running further quantitative analysis on this table.

## Recommendations

1. **Add an explicit relevance gate before stance classification**: for each (key message, comment) pair, first decide whether the comment actually engages with that key message; default to neutral when relevance is low. This directly targets the \~70%+ of errors that are false positives on neutral, and should be the highest-leverage single change.
2. **Treat "against" predictions with extra scrutiny** — at 44% precision they're a coin flip; consider a stricter threshold or a secondary check specifically for this label until precision improves.
3. **Clean up the annotation record**: enforce `corrected_label` on every "incorrect" stance verdict going forward, backfill the 101 existing null rows from the free-text comments, and normalize the German/English label mix.
4. **Re-run this evaluation after the relevance-gate change** using the same contributions, especially the worst performers (22643, 8627, 4573, 2545, 11005, 5525), to confirm the fix closes the gap without introducing new neutral-when-should-be-directional errors.

## Addendum: the root cause was already found and already fixed in code — but never re-validated

A follow-up check of the pipeline code (`stance-detection-app/`) against git history shows the cross-key-message bleed diagnosed above is not a new finding: it was independently identified and patched **before** this report was written, but the fix has never been checked against real predictions.

**The predictions Antonie evaluated are from 2026-07-09.** `data/whole_process_contributions_with_comments_250_mistral-nemo_mistral-nemo.parquet`, the source of all 804 stance judgments, was generated on that date. Her annotation work then ran from 2026-07-28 to 2026-09-21 — two months spent working through that one static batch; no new predictions were generated in that window.

**Two fixes landed on 2026-09-08**, mid-way through her annotation work:

| Commit | Date | What it fixed |
| --- | --- | --- |
| `0971e36` | 2026-09-08 14:12 | The prompt's own few-shot examples contradicted its own rule: "Beispiel 4" showed a bare generic-agreement comment ("Die Ansicht teile ich") labeled **IN FAVOR**, contradicting the rule that generic agreement alone isn't enough. Commit message: this was "causing IN-FAVOR over-firing on generic agreement." |
| `33d65e9` | 2026-09-08 14:37 | Before this commit, the stance prompt had **no visibility into a contribution's other key messages at all** — `StanceDetector.build_messages()` didn't even accept an `other_key_messages` parameter. This commit adds it, plus an explicit prompt rule and worked example, so the model can now answer NEUTRAL when a comment more clearly matches a sibling key message. Commit message, verbatim: "Cross-key-message bleed was the single largest failure category in the prior human-reviewed evaluation (**48%** of all wrong stance verdicts)." |

That 48% figure lines up closely with what this report derived independently from the sqlite data (\~28% "andere Keymessage" + \~19% general-reaction errors ≈ 47%) — strong convergent confirmation that both analyses are looking at the same underlying defect.

**Net effect: the diagnosis in this report describes pipeline behavior that predates both fixes.** The fix design is sound (explicit rule + worked example + sibling-key-message injection directly targets the false-positive mechanism found above), but:

- No `whole_process` output, `results/*.json`, or any file under `data/` postdates 2026-09-08 — the fixed prompt has never actually been run.
- Nobody has re-evaluated fresh predictions against gold labels or against Antonie's judgment since the fix landed.
- A separate finding from this report — "against" (ablehnung) precision at only 44% — is **not** addressed by either fix and remains open.
- `prompts/user_prompt.txt` was not tracked in git before `0971e36`; it was edited on disk outside version control, so the exact wording live during the July run isn't fully reconstructable.

**Re-run results (2026-09-22).** The fixed prompt was re-run over the same 797 evaluated (key message, comment) pairs from the 119 contributions, holding key-message extraction constant (reused verbatim from the July run) so only the stance-classification change is being measured. **Verdict: the fix helps, but only marginally — it does not resolve the false-positive problem.**

| Metric | Before (July run) | After (fixed prompt) |
| --- | --- | --- |
| Accuracy vs. Antonie's judgments (proxy metric, see note below) | 64.5% (514/797) | \~66.6% (516/775, projected) |
| Predictions that changed at all | — | 79 of 802 (9.9%) |
| Previously-wrong rows now fixed | — | 41 of 261 with a known true label (15.7%) |
| Previously-right rows that regressed | — | 39 of 514 (7.6%) |

The projected-accuracy metric is a proxy: it re-uses Antonie's existing correct/incorrect verdicts and corrected labels as ground truth against the *new* predictions, rather than collecting fresh human judgments — a fast way to validate direction and magnitude without waiting on a new annotation pass, though a real re-evaluation would be more definitive.

The fix rate breaks down unevenly by the true label the row should have had:

| True label (row was wrong) | Rows | Now fixed |
| --- | --- | --- |
| neutral (the dominant failure mode) | 195 | 29 (14.9%) |
| in favor | 34 | 9 (26.5%) |
| ablehnung (against) | 32 | 3 (9.4%) |

The dominant failure mode identified above — the model inventing a directional stance when the true answer is neutral — is barely moved: 85% of those cases (166 of 195) are still wrong after the fix.

**A new, smaller regression also appeared.** Of the 39 previously-correct predictions that changed, 31 flipped from neutral to a directional label (20 to "in favor", 11 to "ablehnung") — giving the model sibling-key-message context sometimes makes it *more* willing to assign a directional stance, not less, partially offsetting the gain elsewhere.

**Bottom line:** the two Sept 8 fixes are a real, measurable improvement (+\~2 points of accuracy) and correctly diagnosed a real defect, but they are not sufficient on their own to fix the false-positive problem this report describes. Something beyond sibling-key-message context and one corrected few-shot example is needed — e.g. an explicit two-step prompt ("is this comment even about this key message? yes/no" before asking for a stance), or few-shot examples specifically drawn from the still-unfixed neutral-true-label cases (raw data: `evaluation_data/stance_refit_20260908fix.csv`).
