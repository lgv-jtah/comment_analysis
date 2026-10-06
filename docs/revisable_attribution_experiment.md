# Revisable Attribution Experiment

## Status And Scope

`--mode revisable_attribution` opts into an experimental architecture implementing the same [stance policy v2](stance_policy_v2.md), not a new labeling policy.
The initial validation handoff reported 679 passing tests from `uv run pytest --cov` plus eight independent probes; this was software validation, not an inference or accuracy result.
No broad experiment run has been completed; the [checkpointed model comparison](#checkpointed-model-comparison) is a 40-comment fold-0 development run; the later [matched fold-0 screening summary](../dataset/experiments/screening_fold0_mistral-nemo_matched_20260928T102005Z.summary.json) records verified routing and schema-valid responses for that historical run, not a current readiness guarantee.
Pre-variant report numbers remain historical results under their original code, prompts, cohorts, and scoring contracts, not measurements of the A/B variants described below.
Fold 0 has been used for development and tuning and is not an untouched holdout, even where CLI output calls its grouping "out-of-fold".
Do not rewrite old results or claim an independently validated accuracy improvement from policy fixtures or these tests.

## Engine Contract

The experiment addresses irreversible early attribution: the original two-stage engine restricts target assessment using its first scope decision, whereas this engine treats that decision as revisable advice.
Both stages use the same configured model and original per-comment context, including all frozen targets, related source sentences, contribution text, and supplied parent context.
Annotation labels and annotation notes are not sent to either stage, and the serialized context excludes `run_id`.

1. `GlobalHypothesis` proposes `whole_contribution_approval`, `target_specific`, `mixed_qualified`, or `unresolved`, with affected IDs, general-support scope, comment evidence, and uncertainty flags.
2. `RevisedAttributionResult` returns a replacement `revised_scope` and exactly one assessment for every original target, regardless of the hypothesis's affected IDs, global approval, or unresolved scope.

Stage 2 can revise scope and uncertainty in either direction; the first stage's flags are not permanent gates.
The orchestrator uses only the revised scope for global policy application, and affected IDs never filter the final target set.
Multiple targets may have explicit directional stances at once.

| Attribution | Required decision and deterministic treatment |
| --- | --- |
| `explicit_target` | Requires `in_favor` or `against` and comment evidence; retains its stance ahead of global approval. Under valid revised general support, `in_favor` whose every quote lies inside the revised scope evidence records `basis=general_approval` with stance `neutral` instead. |
| `explicit_exclusion` | Requires a stance, including an allowed `neutral`, and comment evidence for the exception; retains its stance ahead of global approval. |
| `unaddressed` | Allows `neutral` or null from the model; stays `neutral`, recorded as `basis=general_approval` under valid revised general support. |
| `no_position` | Allows `neutral` or null from the model; follows the same inheritance rule as `unaddressed`. |
| `unresolved` | Always requires review; a non-null proposed stance requires comment evidence and is retained only as a review proposal. |

General support means revised `whole_contribution_approval`, or revised `mixed_qualified` with `general_support_extends_to_unaddressed=true`, supported by nonempty comment evidence.
Inherited general approval records stance `neutral` and `basis=general_approval` with revised scope evidence and explanation, per the 2026-10-06 decision change in [Stance policy v2](stance_policy_v2.md); merely omitting a target is not an explicit exclusion.
Models may instead label a blanket approval such as "Stimmt" as `explicit_target`/`in_favor` for every target, quoting the same span.
Support that quotes only text inside the revised general-support evidence (`policy.restates_general_approval`, whitespace-normalized containment) is therefore recorded as `general_approval` with stance `neutral`, the scope evidence, and the scope explanation, since the target's explanation of explicit support would contradict the label.
Any quote outside that evidence, any `against` stance, and any `explicit_exclusion` keep their explicit attribution.
General rejection never propagates automatically, and topic overlap, sentiment, or confirmation of a problem does not establish support for a solution.

Every supplied quote must be nonempty and match a substring after whitespace normalization: comment evidence against the comment, and source evidence against that target's own `related_sentences`.
Source evidence is optional, but any supplied source quote is checked; quotation provenance does not prove semantic correctness.
Revised scope uncertainty affects every target, target uncertainty affects that target, and revised unresolved scope routes every target to review.
Review outputs have no final stance, but structurally valid target decisions can preserve their proposed stance, basis, evidence, and explanation.
Missing context is represented explicitly, supplied parent context is preserved, and the prompts require unresolved references or sarcasm to produce uncertainty rather than invented certainty.

| Failure boundary | Result for the comment's targets |
| --- | --- |
| Either LLM stage returns `None`, including handled request or JSON parsing failures | Every target is `failed`, without a stance fallback. |
| Stage 1 schema, evidence, or affected-ID validation fails | Every target is `failed`; stage 2 is not called. |
| Stage 2 schema validation fails | Every target is `needs_review` with `invalid_revised_schema`, without salvaging labels. |
| Stage 2 has missing, duplicate, or extra targets; invalid scope IDs; missing general-support evidence; or invalid quotes | Every target is `needs_review` with validation reasons, without salvaging labels. |

There is no fallback that silently converts a failed stage to accepted neutral or inherited approval.
Duplicate input target IDs raise an error; an empty target list returns no outputs.
Live benchmark CLI runs now use durable per-comment checkpoints by default, with resume but no stage cache; Python `run_benchmark(..., checkpoint=...)` support is optional and callers must manage the `BenchmarkCheckpoint` lifecycle.
The experimental mode rejects `--verifier` before provider checks rather than offering a homogeneous or heterogeneous verifier.
Existing interactive and whole-process defaults are untouched; benchmark still defaults to `two_stage`, with verifier and dry-run disabled.
Without `--enable-thinking`, the request omits the thinking override: reasoning is `backend_default`, not explicitly disabled, even when older manifests say `enable_thinking=false`.

## Commands And Artifacts

Run these commands from the repository root with the existing frozen fold-0 development source.
Extraction is never rerun: pairs are identified from contribution ID, frozen key-message text, and comment text.
`--limit` caps comments in first-seen order, not pairs, and retains every frozen target for each selected comment; zero selects none and negative values are rejected.
Conflicting duplicate target context, conflicting contribution or parent context, null pair identity inputs, and detected cross-project contribution collisions are rejected rather than arbitrarily resolved.

Offline preparation needs no credentials and performs no provider discovery, schema readiness check, inference, or scoring.
Dry-run does not load `.env` or read the stance configuration from environment defaults; optional `--model-name` and `--base-url` can record non-secret intended identities explicitly.

```bash
uv run cli.py benchmark \
  --source dataset/experiments/screening_subset_fold0_source.parquet \
  --reference dataset/pairs.parquet --folds dataset/folds_v1.json \
  --mode revisable_attribution --role primary --tiers A+B \
  --revisable-variant source_bound_evidence_ids --model-name gptoss-120b \
  --dry-run --output dataset/experiments/revisable_fold0_dry_01.parquet
```

For live execution, provision `STANCE_LLM_BASE_URL`, `STANCE_LLM_MODEL_NAME`, and `STANCE_LLM_API_KEY` through the runtime environment without displaying or recording credential values.
The CLI also loads dotenv for live execution, but these instructions require no secret-file edits or command-line API keys.
Use only IONOS-routed execution and do not pass `--no-verify-ionos` for this experiment.
Verification accepts direct HTTPS access to `openai.inference.de-txl.ionos.com` on the default port or 443 without URL user credentials, or checks proxy `/v1/model/info` routing against that approved host.
Every discovered route for a duplicated model alias must be IONOS-approved; a model name alone is not proof of routing, and route verification is not a live schema capability check.
The following live command is a future execution recipe, not evidence that readiness or inference has been completed.

```bash
uv run cli.py benchmark \
  --source dataset/experiments/screening_subset_fold0_source.parquet \
  --reference dataset/pairs.parquet --folds dataset/folds_v1.json \
  --mode revisable_attribution --role primary --tiers A+B \
  --revisable-variant source_bound_evidence_ids --model-name gptoss-120b \
  --output dataset/experiments/revisable_fold0_live_01.parquet
```

For an output named `run.parquet`, preparation creates `run.expected.parquet` and `run.manifest.json`; durable live execution also uses `run.checkpoint.json` and `run.lock`, and publishes `run.parquet` only after validated complete coverage.
Fresh durable runs reject existing prediction, manifest, expected-cohort, execution, checkpoint, partial, or telemetry artifacts at that basename.
Dry-run and live therefore require DISTINCT basenames, as above; dry-run is not a resumable reservation for the live run.
A new experiment or repeat needs a fresh basename; resuming the same durable run requires its original basename and matching contract.
`evaluate --output` also uses exclusive creation and refuses to overwrite its JSON output.

The durable manifest records ordered comment/target/pair identities, role/tier/limit selection, source/reference/fold hashes, schemas, all prompt-file hashes, root Python implementation hashes, dependency-file hashes and installed versions, model/endpoint identity, verification, decoding, execution settings, and enabled resumability with disabled caching.
Unlike historical manifests, durable code identity uses implementation bytes rather than Git commit/diff metadata, so documentation-only edits do not invalidate resume.
Missing reference or fold files are represented by null input hashes, not a readiness guarantee.
`experiment_id` hashes the specification; dry-run uses that deterministic ID as `run_id`, while a fresh live run gets a new run ID and resume preserves it.
Preparation-only state and configuration differences can change experiment identity, so dry and live IDs need not match.
Durable execution stores session records inside the checkpoint, not a new standalone `run.execution.json`; historical non-checkpoint runs retain their separate execution records.
Session records store sanitized endpoint and route identity, `verification_requested`, `provider_verified`, and `verification_status` (`verified`, `failed`, or `not_requested`) before inference.
Verification is repeated before resumed inference and route changes are rejected; a verification record is not an inference-completion, accuracy, or schema-readiness result.
These JSON provenance records omit API keys, URL credentials, query/fragment data, and raw provider error bodies; this is not a guarantee that arbitrary console logs are secret-free.
The prediction parquet keeps `OUTPUT_COLUMNS_V2`, including statuses, proposed/final labels, basis, evidence, review reasons, policy version, model name, and run ID, rather than raw stage transcripts.

### Resume And Partial Evaluation

The checksum-validated checkpoint is authoritative and commits all canonical rows for one completed comment together using same-directory atomic replacement, file `fsync`, and directory `fsync`.
An OS `flock` enforces a single writer; `--recover-stale-lock` requires `--resume` and only acknowledges a stale saved session after acquiring the lock, never overrides a live writer.
Do not delete the lock file or edit checkpoint state to force recovery; there is no unsafe force option.
Completed `accepted`, `needs_review`, and `failed` results are all skipped on resume, not retried for better labels.
An interrupted comment repeats from stage 1 because there is no stage cache; uncommitted calls may repeat and incur usage again.

Resume the example above with the same model, evidence-ID variant, source, reference, folds, endpoint, selection, and inference settings:

```bash
uv run cli.py benchmark \
  --source dataset/experiments/screening_subset_fold0_source.parquet \
  --reference dataset/pairs.parquet --folds dataset/folds_v1.json \
  --mode revisable_attribution --role primary --tiers A+B \
  --revisable-variant source_bound_evidence_ids --model-name gptoss-120b \
  --output dataset/experiments/revisable_fold0_live_01.parquet --resume
```

After a hard kill, add `--recover-stale-lock` to that command only once the prior writer has stopped.
For explicit partial scoring without provider discovery or inference, use the same command with `--evaluate-partial`, retaining `--resume` and adding `--recover-stale-lock` if the session is stale.
These flags cannot be combined with `--dry-run`; recovery and partial evaluation both require `--resume`.
Resume rejects changed model, source/order/context, prompts/schemas, code, dependencies, reference/folds, or other frozen contract fields before inference rather than mixing results.
API credentials are not part of the contract and may rotate; model and endpoint configuration must still match, including for partial evaluation.

`run.partial.parquet` is a derived export of the saved completed prefix, not the recovery source or a per-comment write-ahead artifact.
It is exported on handled interruption or explicit partial evaluation; after a hard kill it may be absent or stale and can be regenerated from the checkpoint.
Existing exports must have matching ownership and rows; corrupt, unrelated, or conflicting artifacts are rejected rather than overwritten.
Final publication requires complete validated comment/target coverage, preserves a matching existing final artifact, and never promotes an incomplete partial export to final output.
Partial scoring retains the full original expected cohort and eligible denominator, counting missing predictions as failures rather than scoring only survivors.
Completion counts and matches to historical reference labels are not counts of human-verified correct predictions.
Historical runs without durable checkpoints cannot be resumed, including the interrupted old GPT-OSS comparison; telemetry cannot reconstruct its lost canonical predictions.

After an actual live run, score its own prepared cohort explicitly:

```bash
uv run cli.py evaluate \
  --predictions dataset/experiments/revisable_fold0_live_01.parquet \
  --reference dataset/pairs.parquet --folds dataset/folds_v1.json \
  --expected-cohort dataset/experiments/revisable_fold0_live_01.expected.parquet \
  --role primary --tiers A+B \
  --output dataset/experiments/revisable_fold0_live_01.primary_ab.json
```

Use the same predictions and expected cohort for primary Tier A, primary Tier B, and secondary diagnostics, keeping separate metric files:

```bash
for selection in primary:A primary:B secondary:A+B; do
  uv run cli.py evaluate \
    --predictions dataset/experiments/revisable_fold0_live_01.parquet \
    --reference dataset/pairs.parquet --folds dataset/folds_v1.json \
    --expected-cohort dataset/experiments/revisable_fold0_live_01.expected.parquet \
    --role "${selection%%:*}" --tiers "${selection#*:}" \
    --output "dataset/experiments/revisable_fold0_live_01.${selection/:/_}.json"
done
```

Both subcommands accept `--role primary|secondary|all` and `--tiers A|A+B|B`, defaulting to primary and A+B; only `evaluate` accepts `--expected-cohort`.
Benchmark prepares the full source-selected expected cohort independently of role/tier choices, which affect scoring eligibility rather than which targets receive inference.
Omitting `--expected-cohort` in evaluation selects the reference cohort, not the surviving prediction rows, so a subset run can accrue failures for the rest of the reference.
Use `--folds ""` only when intentionally scoring without fold grouping; a missing specified folds file also falls back to ungrouped evaluation with a warning.

## Scoring Contract

- Score unique expected pairs, left-joining predictions so absent predictions become `failed` and remain in eligible accuracy denominators.
- Expected pairs absent from the reference, unresolved pairs, and quarantined pairs remain in operational metrics but not label accuracy.
- Report `unmatched_prediction_pairs` for outputs absent from the full reference separately from `unexpected_prediction_pairs` for outputs outside the expected cohort; neither expands the denominator.
- Identical full prediction rows collapse and increment `identical_prediction_duplicates`; conflicting rows for one pair fail that pair once and increment `conflicting_prediction_pairs`.
- Mixed `run_id` values are rejected, and null pair IDs or conflicting expected/reference contribution identities are errors.
- Invalid statuses and accepted outputs without equal canonical proposed/final labels become failed; failed proposals are cleared, and non-accepted final labels are cleared.
- Reference annotation duplicates count once; conflicting resolved labels or any quarantined annotation quarantine the pair before role/tier selection, so selecting Tier A or primary cannot hide a disagreement.
- Primary labels are not supplemented with secondary labels, and out-of-tier pairs stay operational but become ineligible for accuracy; missing role/tier columns cannot enforce that distinction.
- Report primary A+B as the main historical-label view, primary A and B separately as tier diagnostics, and secondary separately rather than silently pooling it with primary; `--role all` is an explicit combined diagnostic.

Forced-choice accuracy compares proposed labels over all eligible pairs, including valid review proposals; failed and missing outputs count as incorrect.
Accepted accuracy uses final labels only among accepted eligible pairs and is null if none were accepted.
Labelled coverage is accepted eligible pairs divided by eligible pairs; operational coverage is all accepted expected pairs divided by all expected pairs.
Review and failure rates use all expected pairs, and per-class review rate uses eligible pairs of that reference class.
Per-class precision, recall, and F1 use proposed labels over eligible pairs, while per-class accepted accuracy and coverage use accepted labels.
Macro-F1 averages classes present in eligible reference labels or canonical proposed labels, not an unconditional fixed three-class average.
Classes with fewer than five accepted examples are flagged as insufficient, and the majority baseline and accepted-accuracy-by-basis breakdown use the same selected expected cohort.
Always report general-approval basis results alongside blended historical-label accuracy, not as a selectively improved replacement for it.

Fold scoring aligns the expected cohort first, then partitions by expected contribution ID, so folds with no surviving predictions still include their missing outputs.
Overlapping test contributions are rejected, and any eligible expected pair without a fold assignment raises an error rather than disappearing from pooled accuracy.
Unassigned non-eligible pairs remain in overall operational metrics and `unassigned_expected_pairs`, but not in the pooled fold subset.
Evaluation prints per-fold, pooled, and overall metrics; JSON contains overall `metrics`, `pooled_out_of_fold`, and `unassigned_expected_pairs`, but not per-fold metric objects.
JSON also exposes macro-F1, per-class precision/recall/F1, and duplicate/quarantine diagnostics beyond the compact console tables.
The Python `bootstrap_ci` API resamples expected contributions with replacement, preserving missing pairs and repeated draw weights; the CLI does not generate confidence intervals automatically.

## Comparison Order

1. Compare `two_stage`, `one_call`, and `revisable_attribution` with one fixed model, frozen extraction, identical expected pairs, reference/folds, decoding, and verifier disabled to isolate architecture.
2. Freeze the selected engine and its prompts, then compare IONOS models with the same inputs and scoring contract to isolate model effects.

Use new basenames for every configuration and repeat, and retain manifests, execution records, predictions, coverage/failure diagnostics, and historical-label basis breakdowns together.
Development screening is not holdout evidence; future evaluation must distinguish untouched test data from fold-0 tuning and report uncertainty without claiming a completed broad run.

## Implementation Map

| Files | Change overview |
| --- | --- |
| `revisable_attribution.py` | Experimental schemas, two-stage engine, revised-policy orchestration, target/evidence validation, and review/failure routing. |
| `prompts/revisable_attribution_config.json`, `prompts/revisable_attribution_system.txt`, `prompts/global_hypothesis_user.txt`, `prompts/revised_attribution_user.txt` | Separate experimental prompt configuration and versioned stage contracts, without changing default prompts. |
| `benchmark.py`, `benchmark_checkpoint.py`, `cli.py` | Opt-in mode, frozen-input checks, offline preparation, durable whole-comment checkpoints, safe resume/publication, provenance, and cohort-aware CLI scoring. |
| `evaluator_v2.py` | Expected-pair alignment, duplicate/conflict handling, role/tier selection, status normalization, fold completeness, class metrics, and contribution bootstrap alignment. |
| `dataset_builder.py`, `historical_baseline.py` | Annotation-conflict quarantine and unique historical outputs that retain missing or conflicting labels as failures. |
| `ionos_config.py` | Exact approved-host checks, direct endpoint handling, normalized proxy discovery URL, and duplicate-route safety. |
| `tests/test_revisable_attribution.py`, `tests/test_benchmark_readiness.py`, `tests/test_scoring_contracts.py`, and updated existing tests | Offline engine, preparation, provenance, routing, CLI, annotation, baseline, and scoring contract coverage. |

The user's untracked `prompts/user_prompt_v1.txt`, `prompts/user_prompt_v2.txt`, and `prompts/user_prompt_v3.txt` are not part of this experimental configuration and remain untouched.

## Validation Diagnostics

The earlier 2026-09-28 validation handoff reported 746 passing project tests, 46 helper checks, and 24 independent checks.
The quote-diagnostics validation handoff reported 841 passing project tests, 61 helper checks, and 20 independent checks.
The earlier source-bound evidence-ID validation handoff reported 932 passing project tests, 68 helper checks, and 22 independent checks.
The checkpoint basis-fix validation handoff reports 1,029 passing project tests, 31 helper checks, 10 independent basis/reservation checks, and 37 checkpoint regression checks: 1,107 total.
These are offline software-validation results reported by the validator, not tests rerun for this documentation update, live inference, or performance evidence.
The earlier reason-reporting update in `policy.py` and `revisable_attribution.py` did not change acceptance predicates, failure/review routing, prompts, stage schemas, or prediction columns.
`policy._failed` preserves supplied reasons and defaults to `llm_call_failed` only when no reasons are supplied, so context failures need not be mislabeled as request failures.

| Boundary | Diagnostic reasons and interpretation |
| --- | --- |
| Request result | A null result retains `llm_call_failed` and adds `stage1_llm_call_failed` or `stage2_llm_call_failed`; all targets fail, but these codes alone do not distinguish transport, HTTP, or JSON parsing causes. |
| Schema validation | `invalid_hypothesis_schema` fails all targets before stage 2; `invalid_revised_schema` sends all targets to review, with additional `stage1_schema:<type>:<location>` or `stage2_schema:<type>:<location>` reasons. |
| Stage 1 context validation | `stage1_empty_scope_evidence`, `stage1_nonmatching_scope_evidence`, `stage1_unknown_scope_target`, `stage1_duplicate_scope_target`, and `stage1_general_support_missing_evidence` identify schema-valid hypotheses that fail context checks; stage 2 is not called. |
| Stage 2 target coverage | `missing_target_ids`, `duplicate_target_ids`, and `unknown_target_ids` identify each applicable defect while retaining aggregate `invalid_exact_target_coverage`; all targets require review without salvaged labels. |
| Stage 2 scope and evidence | `unknown_scope_target`, `duplicate_scope_target`, and `general_support_missing_evidence` remain distinct from schema errors; `invalid_scope_evidence` and `invalid_comment_evidence` retain their aggregates with corresponding `empty_*` or `nonmatching_*` reasons. |
| Stage 2 source evidence | `invalid_target_source_evidence` retains its aggregate with `empty_target_source_evidence`, `unknown_target_source_evidence` for an unknown assessment target, or `nonmatching_target_source_evidence` for a quote absent from that target's source context. |

Schema diagnostics retain only allowlisted error types, known field names, and nonnegative list indices, using `other`, `unknown_field`, and `root` where appropriate; they omit input values, validation messages, error context, and URLs.
Free-form uncertainty text becomes `scope_uncertainty_flag:<index>` or `target_uncertainty_flag:<index>`, except for the recognized `unresolved_reply_reference` and `suspected_sarcasm` codes.
These diagnostics do not persist raw model responses or provider error bodies; ordinary prediction evidence and explanations remain part of the existing output contract.
The local helper further allowlists telemetry reasons, replaces unrecognized reasons with `other_review_reason`, and suppresses ordinary stdout/stderr and logging rather than storing raw response strings.

### Stage-1 Quote Diagnostics

`revisable_attribution._stage1_quote_diagnostics` adds the following reasons only for nonempty evidence already rejected with `stage1_nonmatching_scope_evidence`.

| Reasons | Interpretation |
| --- | --- |
| `stage1_quote_overlap_contribution`, `stage1_quote_overlap_key_message`, `stage1_quote_overlap_related_sentence`, `stage1_quote_overlap_parent` | Whitespace-normalized substring overlap with the named context field, not proof that the model copied from it. |
| `stage1_quote_case_only`, `stage1_quote_punctuation_only`, `stage1_quote_case_and_punctuation` | Conservative, boundary-checked matches against comment text after the indicated diagnostic transformations. |
| `stage1_quote_unexplained_text_mismatch` | No diagnostic category matched; not proof of paraphrase, translation, or fabrication. |

Formatting diagnostics use local NFC normalization, `lower()` rather than case folding, and punctuation-to-space conversion rather than deletion; symbols, accents, combining marks, and invisible controls are not stripped.
Adjacent alphanumeric characters, underscores, marks, and control/format characters prevent a diagnostic boundary match; overlap checks require alphanumeric content and do not join separate context fields.
Reasons can overlap and are deduplicated per comment, so affected-pair counts are not quote counts.
Acceptance and failure criteria remain fixed: evidence must still be a whitespace-normalized comment substring, and a stage-1 mismatch still fails every target without calling stage 2.
No quote repair, prompt/schema change, or label salvage is introduced, and diagnostic reasons contain no raw quotes or context text.

### Completed Baseline Diagnostic

Reservation `20260928T113121045969Z` was prepared offline, with provenance in its [selection record](../dataset/experiments/revisable_validation_subset_20260928T113121045969Z.selection.json) and paths in its [reservation record](../dataset/experiments/revisable_validation_subset_20260928T113121045969Z.reservation.json).
It retains all frozen targets for six comments and 29 pairs in source order: three prior stage-1 failures and three prior exact-target-coverage failures from matched run `20260928T102005Z`.
The selected stage-1 calls were non-null and schema-valid in the prior telemetry; the new contextual reasons must not be inferred retroactively from the old aggregate failure code.
This is a failure-selected diagnostic subset, not a representative evaluation or untouched holdout, and supports no performance or accuracy-improvement claim.

This baseline diagnostic has now completed, as recorded in its [saved analysis](../dataset/experiments/revisable_validation_subset_20260928T113121045969Z.analysis_20260928T113831969348Z.json).
Four comments failed stage 1 with `stage1_nonmatching_scope_evidence`, blocking 20 pairs; two comments reached stage 2 and required review for `missing_target_ids`, blocking nine pairs.
These are affected-pair counts, not counts of mismatched quotes or omitted targets.
All eight completion attempts returned HTTP 200 and schema-valid responses, with no null responses; client-visible usage totaled 9,131 tokens over 41.88 seconds (about 42 seconds).
No pairs were accepted, no proposed or final labels survived, and accepted accuracy is undefined, not zero.
The saved integrity checks report unchanged protected original artifacts.
Stage-1 failures hid stage-2 behavior for four comments, and neither HTTP success nor schema validity establishes valid evidence or complete target coverage.

## Separate A/B Variants

Benchmark accepts `--revisable-variant baseline|stage1_comment_only|stage2_required_targets|source_bound_evidence_ids`, defaulting to `baseline`.
Non-baseline variants require `--mode revisable_attribution` and are rejected with `two_stage` or `one_call`; revisable attribution rejects `--verifier`, including with the baseline variant.
The variants use fixed prompt configurations, and non-baseline variants reject a custom prompt configuration.
These are separate interventions, not a combined A+B implementation; variant letters here are unrelated to annotation tiers.
The evidence-ID variant is described separately below and does not incorporate B's required-target wire schema.

| Variant | Isolated intervention |
| --- | --- |
| A: `stage1_comment_only` | Changes only the stage-1 user prompt to require nonempty, contiguous verbatim evidence from top-level `comment_text`, excluding contribution, target, source, and parent text, paraphrases, translations, and joined spans. |
| B: `stage2_required_targets` | Keeps baseline stage 1 and changes the stage-2 prompt and wire schema to a per-context `targets` object keyed by every original target ID, with all keys required and additional keys forbidden. |

A still receives the full original context; "comment only" restricts evidence provenance, not model input.
A leaves both schemas, stage 2, parsing, and validation unchanged.
B omits `key_message_id` inside each keyed value, restores IDs in original context order, and validates the converted list through the unchanged `RevisedAttributionResult` before policy application.
B rejects duplicate JSON object keys during stage-2 parsing rather than silently keeping the last value; this fails all comment targets with `stage2_duplicate_json_key`.
Missing or extra wire keys and invalid canonical decisions require review with `invalid_revised_schema`; no quote repair, target filling, or neutral fallback is introduced.
Canonical stance, evidence, uncertainty, exact-target coverage, and inheritance semantics remain unchanged, as do prediction columns and baseline prompt assets.

### Variant Provenance

The manifest's `revisable_diagnostic` records variant/version, configuration hash, Pydantic dependency versions, canonical schema hash, stage-1 response-format hash, and ordered per-context metadata.
Each context records its anchor, contribution, pair and target IDs, context hash excluding `run_id`, and stage-2 wire schema and full response-format hashes.
For B, the canonical stage metadata is not the dynamic wire schema; use the per-context hashes to identify the actual required-target request contract.
The local helper checks these hashes against outgoing requests and records stage reachability, wire validation, canonical validation, sanitized reasons, usage, and timing separately without raw response strings.

### Completed Variant Runs

Both variants completed against the same saved six-comment, 29-pair source in its original order, with all frozen targets retained and no new extraction.
The existing [verified comparison summary](../dataset/experiments/revisable_variants_comparison_20260928T122808143133758Z_verified.summary.json) records A with 10 accepted, 15 failed, and four review pairs, and B with four accepted, 20 failed, and five review pairs.
These are outcomes on a failure-selected diagnostic subset, not general performance metrics or evidence of an accuracy improvement.
The historical commands below used `/tmp/opencode/revisable_validation_variants.py`, a local session helper, not a permanent project CLI; their consumed reservations must not be rerun:

```bash
uv run /tmp/opencode/revisable_validation_variants.py run \
  --reservation dataset/experiments/revisable_variant_stage1_comment_only_20260928T121916619209Z.reservation.json
```

```bash
uv run /tmp/opencode/revisable_validation_variants.py run \
  --reservation dataset/experiments/revisable_variant_stage2_required_targets_20260928T121923762350Z.reservation.json
```

Each historical reservation has separate selection, dry-run, live, claim, and telemetry paths; these non-checkpoint artifacts are never overwritten or resumed.
The helper checks prepared inputs, implementation and prompt hashes, context order, inference settings, and the saved baseline's verified IONOS route, and protects original artifacts with before/after hashes.
Each historical run retains `mistral-nemo`, temperature `0.1`, `max_tokens=20000`, JSON-schema responses, concurrency one, omitted thinking override, disabled verifier/cache/resume, and required IONOS route verification.
Per-run limits are 12 logical calls, 16 completion attempts including retries, and 300 seconds, with soft observed budgets of 30,000 total tokens and 10,000 completion tokens.
Token accounting covers only client-visible responses, missing usage remains unknown, proxy-internal usage is unknown, and one in-flight request can overshoot a token budget; these are not hard billing caps.

B retains baseline stage 1, so a stage-1 failure masks its stage-2 intervention and must not be counted as an observed stage-2 result.
The comparator is the saved baseline diagnostic, not a contemporaneous baseline rerun; matching inputs and inference settings do not remove response variability or time-dependent provider effects.
This failure-selected fold-0 subset supports only bounded failure-mechanism diagnostics, not broad accuracy, generalization, or combined-intervention improvement claims.

### Completed Quote Diagnostic

The completed A-only diagnostic selected the three comments and all 15 frozen pairs that failed with `stage1_nonmatching_scope_evidence` in the completed A run, preserving source order and original context without new extraction.
Its [saved analysis](../dataset/experiments/revisable_quote_diagnostics_20260928T124716743166Z.analysis_20260928T125307011357231Z.safe_summary.json) records two stage-1 failures (10 pairs) and one comment reaching stage 2 with five accepted pairs and no review pairs.
One rejected quote overlapped both contribution and related-sentence text; the other had an unexplained mismatch, with no case/punctuation categories observed.
These are two rejected quotes with nonexclusive diagnostic categories, not proof of copying or fabrication; categories cannot be assigned retroactively to the older A responses.
The [primary A+B metrics](../dataset/experiments/revisable_quote_diagnostics_20260928T124716743166Z_live.primary_AB.metrics_20260928T125307011357231Z.json) record four reference matches: 4/15 forced-choice accuracy, 4/5 accepted accuracy, and 5/15 coverage.
All four completion attempts returned HTTP 200 and schema-valid responses, with no null results or client retries; observed usage was 4,713 tokens over 23.81 seconds.
This tiny failure-selected fold-0 cohort is not general accuracy evidence, and a fresh completion recovering one comment does not establish an instrumentation benefit.
Consumed reservation `20260928T124716743166Z`, with its [selection record](../dataset/experiments/revisable_quote_diagnostics_20260928T124716743166Z.selection.json), replaced superseded reservation `20260928T124259921528Z`.
Preserve both reservations and all existing artifacts unchanged; the superseded reservation is not a fallback or resumable run.
The historical command used a local session helper, not a permanent project CLI; do not rerun this consumed reservation:

```bash
uv run /tmp/opencode/revisable_quote_diagnostics.py run \
  --reservation dataset/experiments/revisable_quote_diagnostics_20260928T124716743166Z.reservation.json
```

This historical reservation retains A's original prompts, schemas, and inference settings: `mistral-nemo`, temperature `0.1`, `max_tokens=20000`, JSON-schema responses, concurrency one, omitted thinking override, disabled verifier/cache/resume, and required IONOS route verification.
Limits are six logical calls, eight completion attempts including retries, and 180 seconds, with soft observed budgets of 15,000 total tokens and 5,000 completion tokens.
Client-visible usage only is counted, missing usage remains unknown, proxy-internal usage is unknown, and an in-flight request can overshoot a token budget; these are not hard billing caps.
The helper verifies frozen inputs and request contracts, preserves protected artifacts, and allowlists diagnostic reasons without retaining raw quotes, responses, or provider error bodies in telemetry.

## Source-Bound Evidence IDs

Opt in with `--mode revisable_attribution --revisable-variant source_bound_evidence_ids`; baseline and other variants remain unchanged.
Both stages replace every evidence field on the wire with IDs: stage-1 and revised-scope `comment_evidence_ids`, plus target `comment_evidence_ids` and `source_evidence_ids`.
Both stages still receive full context, but only `comment_text` spans are eligible for comment evidence and each target's own `related_sentences` entries for source evidence.
Contribution text, parent context, and key-message summaries have no eligible evidence IDs.

`source_bound_evidence_ids_v1` uses deterministic `conservative_sentence_newline_v1` comment segmentation, keeping uncertain boundaries joined, and whole nonblank related-sentence entries for source spans.
Each table entry binds an ID to its source-field path, target/source index where applicable, and exact original-text slice at Unicode code-point half-open offsets `[start, end)` within that source field, not byte or UTF-16 offsets.
IDs bind the full context and contract versions but ignore `run_id`; identical text in another context, field, or target cannot substitute for the current eligible ID.
No Unicode normalization, quote repair, or reconstructed text is used during resolution.

Context-specific wire schemas constrain evidence strings to eligible ID enums; an empty eligible set permits only an empty array, with no sentinel ID or arbitrary string escape hatch.
Extra quote fields, unknown IDs, duplicate IDs within an evidence list, and wrong-source or wrong-target IDs are rejected, including when their text is identical to valid evidence.
Source enums cover the context's sources, with per-target ownership enforced during resolution.
Resolved original text passes through the unchanged canonical `GlobalHypothesis` or `RevisedAttributionResult` and policy validators; prediction columns and canonical evidence text remain unchanged.
Stage 2 retains a generic target list, not B's required-target object; exact target coverage is still checked after canonical conversion.
Both evidence-ID stages alone opt into `strict_json`: parse the untouched JSON response, reject duplicate object keys, and do not strip fences/thinking text, extract embedded objects, or normalize labels.
Stage-1 wire/binding/canonical failures fail all targets before stage 2; corresponding stage-2 failures require review without salvaged labels, while null request/parse results fail all targets.
There is no free-text evidence fallback, target filling, neutral fallback, or relaxed acceptance; valid provenance still does not establish semantic correctness.

### Provenance And Reservation

The manifest stores both per-context wire schemas and response formats with hashes and serialized byte sizes, both canonical schemas with hashes, and evidence-table hash, count, byte size, contract/segmentation versions, and offset convention.
It also records ordered context/pair/target identities, context hashes excluding `run_id`, prompt/configuration and implementation hashes, Pydantic versions, `uv.lock`/`pyproject.toml` hashes, parser flags, and inference settings.
The table's raw text is not persisted in this metadata, but schemas retain IDs and other artifacts retain source or canonical evidence text; hashes and sanitized telemetry are provenance/privacy controls, not anonymization.

Historical reservation `20260928T135618996064Z` completed; see its [saved analysis](../dataset/experiments/revisable_evidence_ids_20260928T135618996064Z.analysis_20260928T141306296300Z.safe_summary.json), [selection record](../dataset/experiments/revisable_evidence_ids_20260928T135618996064Z.selection.json), and [reservation](../dataset/experiments/revisable_evidence_ids_20260928T135618996064Z.reservation.json).
It preserved the same three comments, 15 pairs, all frozen targets, source order, and original context without new extraction, comparing against the completed quote diagnostic's 10 failed, five accepted, and four reference-matching pairs, not the earlier all-failed A subset.
The historical command used the local session helper, not a permanent project CLI; do not rerun this consumed reservation:

```bash
uv run /tmp/opencode/revisable_evidence_ids_20260928T131719607967115Z.py run \
  --reservation dataset/experiments/revisable_evidence_ids_20260928T135618996064Z.reservation.json
```

This reservation superseded `20260928T133439246378Z`; preserve both and all historical artifacts, with no overwrite, resume, or fallback to an older reservation.
Historical settings were `mistral-nemo`, temperature `0.1`, `max_tokens=20000`, JSON-schema responses, concurrency one, omitted thinking override, disabled verifier/cache/resume, and required IONOS route verification.
Limits are six logical calls, eight completion attempts including retries, and 180 seconds, with soft observed budgets of 15,000 total tokens and 5,000 completion tokens; the request's `max_tokens=20000` is unchanged.
Only client-visible usage is counted, missing usage remains unknown, proxy-internal usage is unknown, and one in-flight request can overshoot a token budget; these are not hard billing caps.
The helper checks reserved inputs, outgoing contracts and settings, records bounded sanitized diagnostics, and protects prior artifacts with hashes.
Offline byte overhead is recorded in the selection, not measured live token overhead or provider schema readiness; this tiny selected comparison supports no broad accuracy or generalization claim.

## Checkpointed Model Comparison

The checkpointed comparison ran on 2026-09-29 under reservation prefix `model_comparison_checkpointed_20260929T064921105737Z`.
GPT-OSS-120B completed and published; Qwen3.5-397B stopped partway because of provider errors and cannot be resumed.
Both runs used `source_bound_evidence_ids` on the frozen fold-0 cohort of 40 comments, 138 pairs, 23 contributions, and 119 eligible primary A+B reference pairs; fold 0 remains development data.
Both runs predate the 2026-10-06 general-approval changes described in [Stance policy v2](stance_policy_v2.md): their `general_approval` rows carry `in_favor`, and blanket approval quoted per target is still labelled `explicit_target`.
Their saved scores are historical results under the old rule, not measurements of the current engine.

### Execution

| Run | Outcome | Calls and HTTP attempts | Known usage |
| --- | --- | --- | --- |
| `gptoss-120b` | Published: 40/40 comments, 138/138 pairs. | 80 logical calls, all returned; 81 HTTP requests (1 metadata, 80 completions, all HTTP 200 with `finish_reason=stop`), no client retries; about 916 seconds. | 244,283 total tokens, 100,934 completion. |
| `qwen35-397b` | Not published: 36/40 comments, 125/138 pairs saved as a partial checkpoint. | 54 logical calls (31 returned, 22 null, 1 reserved and never completed); all 120 HTTP attempts used: 1 metadata, 32 HTTP 200, 6 HTTP 502, 81 HTTP 503 (the last 81 attempts consecutive), 65 client retries; about 5,947 seconds. | 257,554 total tokens, 200,823 completion; usage unknown for 87 error responses. |

The Qwen run stopped when its 120-attempt HTTP budget was exhausted by IONOS 502/503 responses.
Its persisted absolute deadline (2026-09-29 11:10:55 CEST) has also passed, so the reservation can never be resumed.
The run says nothing reliable about Qwen's stance quality: its full-cohort score mixes availability with semantics.
Before any Qwen rerun, investigate the LiteLLM/IONOS serving path; no serving configuration was changed.

### Results Against Historical Labels

Read-only analysis: [`saved_model_comparison_analysis_20260929T090403579070Z`](../dataset/experiments/saved_model_comparison_analysis_20260929T090403579070Z/comparison.safe_summary.json), produced with no new inference requests and no label changes.
Scores are on the 119 eligible primary A+B pairs; a correct automatic answer is an accepted pair whose final stance matches the reference label.

| Run | Correct automatic | Incorrect automatic | Not accepted | Correct review proposals | Forced-choice correct |
| --- | --- | --- | --- | --- | --- |
| GPT-OSS-120B | 63 (52.9%) | 41 | 15 review | 14 | 77 |
| Mistral-Nemo evidence-ID screening | 51 (42.9%) | 26 | 42 review | 16 | 67 |
| Qwen3.5-397B partial | 14 (11.8%) | 15 | 13 review, 77 failed (12 eligible pairs missing) | 6 | 20 |
| Legacy, pre-fix (`established_original_label`) | 89 (74.8%) | 30 | 0 | 0 | 89 |
| Legacy, post-fix (`post_fix_original_label`) | 87 (73.1%) | 32 | 0 | 0 | 87 |

The legacy rows are not an independent baseline.
Reference labels come from a human verifying the legacy model's own output: a "correct" verdict makes the legacy label the reference (89 of these 119 pairs), and an "incorrect" verdict makes legacy wrong by construction.
The legacy pre-fix score therefore equals the number of verdict-correct pairs, and the 19 cohort pairs where legacy was judged wrong without a usable correction are excluded as unresolved.

The 2026-10-06 gap investigation decomposed GPT-OSS's 24-pair deficit against legacy post-fix as follows:

- -17 pairs on 12 blanket-approval comments ("Stimmt", "Warum nicht!", ...), which the old rule labelled `in_favor` and the annotator mostly labelled `neutral`.
- -14 pairs from review routing on `unresolved_scope` comments, where the withheld proposal matched the reference.
- +7 pairs on all other comments, where GPT-OSS's proposals beat legacy post-fix.

This led to the basis-provenance fix and the option C decision in [Stance policy v2](stance_policy_v2.md).
[`basis_rescore.py`](../basis_rescore.py) re-scores these saved predictions offline.
The basis fix alone moves GPT-OSS `explicit_target` accepted accuracy from 35/66 to 27/31 without changing any label.
Simulating option C, GPT-OSS gets 65 to 82 correct automatic answers out of 119 ([summary](../dataset/experiments/basis_rescore_20261006T084723083505Z.safe_summary.json)).
The range exists because saved rows do not record the model's scope decision.
The rule was chosen after inspecting this cohort, so these are development-set simulations; fresh runs under the current engine are required for reportable numbers.

### Artifacts And Reuse

Committed: the GPT-OSS predictions and expected cohort, the Qwen partial predictions and expected cohort, the Nemo screening predictions and expected cohort, and the saved analysis directory.
The checkpoints, manifests, locks, and telemetry of both runs remain local, untracked files under `dataset/experiments/`.
The earlier interrupted GPT-OSS attempt `model_comparison_checkpointed_20260929T062257210845Z_gptoss-120b_full` (four comments, nine pairs) is preserved unchanged and was not merged into the published run.
The previously passed Qwen smoke from `20260929T062257210845Z` was reused through read-only hash, route, and same-wire-contract checks, not repeated.

The run helper `/tmp/opencode/model_comparison_checkpointed_20260929.py` (SHA-256 `f7c085a276cb6193293949fed5fccbd7c31cfc66d91c54ed6badd76669e62773`) and both prepared JSON files lived in a temporary directory and no longer exist.
Together with the changed implementation and prompt hashes, this means neither reservation can be checked, run, or resumed again.
Do not edit recorded hashes or migrate rows into a new run.
A new comparison needs a fresh reservation and a helper kept in the repository rather than in a temporary directory.

Checkpoint validation accepts the engine's `explicit_exclusion` basis, including neutral exclusions, alongside `general_approval`, `explicit_target`, `unaddressed`, and `no_position`.
Valid review rows retain their proposed stance, basis (including `unresolved` or null), evidence, and explanation, but never a final stance; unknown bases and invalid status/stance combinations remain rejected.
Offline tests exercise actual engine outputs with mocked model responses through checkpoint commit, reload/resume, partial export, and final publication.

### Execution Contract Used

The helper's `check`, `run`, and `resume` commands required `--prepared PATH --prepared-sha256 SHA256` with `--kind full|smoke`; only `resume` accepted `--recover-stale-lock`, and `--evaluate-partial` required `resume --kind full`.
It validated prepared/helper hashes, protected historical artifacts, frozen contracts, approved routes, and outgoing request bodies before proceeding.
Terminal failure summaries distinguished `row_validation_failed`, `checkpoint_persistence_failed`, and allowlisted stop reasons, retaining safe counters without exception messages, tracebacks, raw model responses, or provider response text.

| Helper execution | Limits |
| --- | --- |
| Full comparison, per model | 100 logical calls, 120 client-visible HTTP attempts, 7,200 seconds, 1,000,000 observed completion tokens, 2,000,000 observed total tokens. |
| Qwen smoke | Two logical stage calls, three HTTP attempts including route metadata, at most 1,300 seconds, zero retries, unchanged `max_tokens=20000` per request. |

Full runs allowed 80 expected two-stage calls plus 20 replay calls, with at most three retries per logical call and retry delays capped at 30 seconds, including `Retry-After`.
The per-request contract used temperature `0.1`, `max_tokens=20000`, JSON-schema responses, concurrency one, backend-default reasoning with no thinking override, and no verifier or stage cache.
The smoke used one synthetic three-target comment and expected `in_favor`, `against`, `in_favor` under the pre-2026-10-06 general-approval rule; a rerun under the current rule would expect `neutral` for the inherited targets.
Logical calls and HTTP attempts were reserved durably before dispatch; HTTP counts include metadata GETs and completion POSTs, including retries, but not unknown proxy-internal attempts.
The absolute persisted deadline includes downtime and is never reset on resume, and saved counters survive resume, so a consumed budget cannot be refreshed by restarting.
Observed token budgets are soft lower bounds, missing usage remains unknown, and an in-flight request may overshoot; they are not hard billing caps.
The built-in `cli.py benchmark` shares durable checkpointing and bounded per-request retries/timeouts but has no aggregate logical-call, HTTP-attempt, token-budget, or absolute-deadline flags and does not enforce these helper-specific budgets.
