# Plan: General Reaction Split

Status: phases 1 and 2 implemented 2026-10-12; phase 3 waits for the annotator's labels.
Decision: option A (her labels define the split prompt, the prompt is tuned only after the table comes back).

## Implementation notes (phases 1 and 2)

Where the implementation differs from or refines the plan below:

- The whole `general_approval_review.py` module and its tests were removed, not only the rule-specific parts.
  Without its subcommand the diagnostic code had no caller left; it stays in the git history up to commit d45fd04.
- The population is read from the frozen `diagnostic_pairs.parquet` of the 2026-10-07 run, after checking its hash against that run's `artifact_hashes.json`.
  Only ID and text columns are read, so no model output or earlier label can leak into the pack.
- The population has 57 comments: 12 in fold 0, 38 in folds 1-4 and 3 in no fold.
  Prompt tuning in phase 3 would therefore have had only 12 development comments.
- To get more development data, the pack also contains the 28 remaining fold-0 comments of `dataset/experiments/screening_subset_fold0_source.parquet`, so all 40 fold-0 comments are in it.
  Their key-message, pair and comment IDs are derived as in the benchmark runs; the derivation is checked on the 12 fold-0 comments both parts share.
  Most of them have no general reaction, which gives the split stage negative examples.
  The pack now has 85 comments and 275 key-message rows; `--no-extra-fold0` builds the 57-comment version.
- A larger, label-free development pool would need a fresh DIPAS sample on contributions outside the gold set, with one key-message extraction call per contribution; not done.
- `review_pack_common.py` has no collapsed column groups, because the blank pack has no background columns.
- The pack has a sheet "Anleitung" with six invented examples, "Kommentare", "Kernbotschaften", a hidden "Listen" and a very hidden "Meta" sheet.
- Without a general reaction the annotator may leave the rest text empty; the reader then takes the whole comment as the rest.
- `uv run cli.py reaction-split-read --pack <file>` writes her labels to a new `reaction_split_labels_<timestamp>` directory.
  Quotes not found in the comment, contradictory answers and unanswered rows are listed per row in an `issues` column instead of failing.
- The schema uses `rest_spans` (a list of quotes) instead of a single `rest_text`; the output column `rest_text` holds the spans joined by spaces.
- The split stage is opt-in (`StancePolicyEngineV2(reaction_split_enabled=True)`, `WholeProcess.run_v2(reaction_split_enabled=True)`) and needs a `reaction_split` entry in `prompts/config.json`, which phase 3 adds.
  Until then the default pipeline behaves exactly as before.
- In split mode the old general-approval logic is bypassed: a whole-contribution approval found in the rest text goes to review, and general support never extends to unaddressed key messages.
  Removing that logic from the default path (plan step 2.4) is deferred to phase 3, when split mode becomes the default.
- Scope and stance templates may use `$full_comment_text` to show the whole comment as context next to the rest text in `$comment_text`; the current templates do not use it yet.
- In split mode, stance evidence must lie in the rest text and in the original comment.
- `OneCallPolicyEngineV2` and `RevisableAttributionEngine` do not support the split stage, and the `benchmark` subcommand has no flag for it yet (phase 4).
- The new output columns are `general_reaction`, `general_reaction_polarity`, `reaction_evidence` and `rest_text`.
  The summary adds `general_rejection` and `contribution_general_rejection_comments`.
- `contribution_general_approval_comments` and `contribution_general_rejection_comments` also count reactions followed by own content (split polarity), not only comments that are pure reactions.
- Benchmark checkpoints from before this change cannot be resumed, because the output columns changed.

## Goal

Split every comment into a general reaction part and a rest part.
The general reaction is recorded once per comment with a polarity (positive or negative).
Stance per key message is decided only on the rest part.
This replaces the open rule 1/2/3 question from 2026-10-07: a general reaction never counts as stance, own content is judged per key message.

## Phase 1 - Annotation table (blocking for phase 3)

1. Extract the shared pack infrastructure from `general_approval_review.py` into a new module `review_pack_common.py`: dropdown lists, row fingerprints, new-directory-per-run writing, artifact hashes, sheet protection, column groups.
2. Delete the rule 1/2/3 specific code from `general_approval_review.py` (option sets `einordnung`, `regel`, `reichweite`, `ursache`, global answer cells, `ANLEITUNG_TEXT`, the pack writer and `read_review_pack`) and the `general-approval-review` CLI subcommand, plus their tests.
   The generated pack under `dataset/experiments/general_approval_review_20261007T133543733476Z/` stays as historical record.
3. New module `reaction_split_review.py` and CLI subcommand `uv run cli.py reaction-split-pack`, offline, no inference, no `.env`, annotation DB read-only.
4. Population: the same 57 comments as the 2026-10-07 review pack.
5. Sheet "Kommentare", one row per comment:
   - shown: `nr`, comment text, contribution text, key message overview
   - editable: general reaction (ja, nein, unklar), polarity (positiv, negativ, unklar), hint span (exact quote), hint explanation, rest text (exact quote), split possible (ja, nein), free comment
   - hidden: `comment_id`, `row_fingerprint`
6. Sheet "Kernbotschaften", one row per comment x key message:
   - shown: `nr`, comment text, key message
   - editable: stance of the rest text (dafür, dagegen, neutral, nicht angesprochen, unklar), free comment
   - hidden: `pair_id`, `comment_id`, `row_fingerprint`
7. The pack is blank: no model output, no earlier labels, no heuristic signals, not even in collapsed groups.
8. Sheet "Anleitung" with invented examples only: blanket approval, approval with own content, general rejection, sarcasm, non-splittable comment.
9. Reader `read_reaction_split_pack`: validates schema version, IDs, fingerprints, option values, and that hint span and rest text are whitespace-normalized substrings of the comment; flags violations instead of failing silently.
10. Her returned labels are written to a new versioned file; existing reference labels are never overwritten.

## Phase 2 - Pipeline scaffolding (can start immediately)

1. `policy_schemas.py`: new `ReactionSplitResult` with `has_general_reaction`, `polarity` (positive, negative, none), `reaction_spans`, `hint_explanation`, `rest_text`, `split_possible`, `uncertainty_flags`.
2. `policy.py`: new `_call_split_stage` and `_validate_split` (spans and rest are substrings, reaction spans plus rest cover the comment without overlap); invalid output goes to `needs_review`, never to a silent neutral.
3. `_resolve_once` routing:
   - general reaction and empty rest: every key message `neutral` with basis `general_approval` or new basis `general_rejection`
   - otherwise scope and stance stages run on the rest text, with the full comment as read-only context
   - split not possible: `needs_review` with reason `split_not_possible`
4. Remove `WHOLE_CONTRIBUTION_APPROVAL` handling, `general_support_extends_to_unaddressed` and `restates_general_approval` once the split stage replaces them; adjust the scope prompt accordingly.
5. Verifier: disagreement on general reaction or polarity routes to review.
6. Apply the same split to `OneCallPolicyEngineV2` and `revisable_attribution.py`, or explicitly mark them as not supporting it.
7. `whole_process.py`: add `general_reaction`, `general_reaction_polarity`, `reaction_evidence`, `rest_text` to `OUTPUT_COLUMNS_V2`; add `general_rejection` and `contribution_general_rejection_comments` to `SUMMARY_COLUMNS_V2`.
8. Unit tests for schema, validator, routing and summary counts with a stubbed LLM.

## Phase 3 - Split prompt (after her labels are back)

1. Write `prompts/reaction_split_system.txt` and `prompts/reaction_split_user.txt` from her fold-0 labels and the "Anleitung" definitions.
2. Tune only on fold-0 comments; comments from folds 1-4 are evaluation only.

## Phase 4 - Evaluation

1. Script comparing model split against her labels: agreement on general reaction and polarity, span overlap, per key message stance on the rest.
2. Fold-0 benchmark with `uv run cli.py benchmark`, then one held-out run on folds 1-4.
3. Report blended accuracy together with the basis breakdown, as required by `docs/stance_policy_v2.md`.

## Documentation

1. New decision section in `docs/stance_policy_v2.md`: general reaction split, general rejection marker.
2. Mark "Open question 2026-10-07" as superseded by this decision.

## Constraints

- Folds 1-4 are held out and never used for tuning.
- Reference labels are never overwritten.
- The split stage adds one LLM call per comment (about one third more calls in two-stage mode).
