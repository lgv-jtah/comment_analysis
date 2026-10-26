# Vote-ratio hypothesis: pre-registration note

This note is written and committed **before** any result is computed. It
pre-registers a single equivalence test and all of its sensitivity analyses,
so that the acceptance criteria cannot be chosen after seeing the numbers.
It contains **no results**: the files that report numbers are produced by
`vote_ratio_hypothesis.py` and written only into fresh, timestamped
directories under `dataset/experiments/`.

The analysis is implemented in `vote_ratio_hypothesis.py` and exposed through
two CLI subcommands, `vote-ratio-fetch` (the only step that touches the
network) and `vote-ratio-analyse` (fully offline).

## Background

The DIPAS participation platform stores, for every contribution, two kinds of
public reaction:

- **Comments.** Free-text replies. Our stance-detection pipeline labels each
  comment's stance toward each of the contribution's key messages as
  `in_favor`, `against` or `neutral`.
- **Votes.** A simple thumbs-up / thumbs-down signal per contribution,
  exposed by the API as `votingPro` and `votingContra`.

A recurring question from stakeholders is whether the expensive,
comment-level stance detection tells us anything that the cheap vote counts
do not already tell us at the aggregate level. This note pre-registers one
narrow, falsifiable version of that question.

## Hypothesis and test

We compare two aggregate shares, both expressed as a percentage in
`[0, 100]`:

- **Comment favor share** `F`: among comments that take a directional stance
  (`in_favor` or `against`), the share that is `in_favor`. Computed over the
  pooled set of qualifying comments (see *Units and pooling*).
- **Vote pro share** `V`: among all votes cast (`votingPro + votingContra`),
  the share that is `votingPro`. Computed over the pooled vote totals of the
  same contributions.

The quantity of interest is the difference in percentage points
`d = F - V`.

**Null / alternative.** We run a **TOST** (two one-sided tests) equivalence
test. The equivalence margin is pre-registered at
`EQUIVALENCE_MARGIN_PP = 10.0` percentage points. We declare the two shares
**equivalent** if, and only if, the confidence interval for `d` lies strictly
inside the band `(-10, +10)` — that is, `-10 < low` **and** `high < +10`.
Otherwise the result is **non-equivalent** (inconclusive or a genuine
difference). The primary decision uses the **90%** confidence interval
(the level that matches a two-sided 5% TOST); the **95%** interval is reported
alongside for transparency.

**Confidence interval.** `d` is estimated with a **clustered bootstrap**:
`N_BOOTSTRAP = 10_000` resamples, seeded with `BOOTSTRAP_SEED = 20261026`,
resampling **contributions** (not individual comments or votes) with
replacement, so that the dependence between comments on the same contribution
is respected. Each resample recomputes both `F` and `V` from the *same*
resampled set of contributions, and the reported interval is the percentile
interval of the resampled `d` values.

### Interpretation rules (verbatim)

The following caveat is reproduced verbatim in the machine summary and in the
final report, and applies in both directions:

> Equivalence does not prove that stance detection is unnecessary;
> non-equivalence does not prove that it is necessary. This test answers only
> whether the overall favor share of comments matches the overall share of
> thumbs up among votes.

## Units and pooling

### The fixed vote-side contribution set (decided rule)

The vote side is pooled over a **fixed set of contributions** that does
**not** depend on any stance outcome:

> The vote-side set is every contribution that (a) has at least one comment in
> the primary_AB pair-label set **and** (b) is present in the vote snapshot.
> This set is identical for the primary analysis and for every sensitivity
> variant. The stance labels can never select which votes are counted.

**Why.** If the set of contributions whose votes we pool were allowed to shift
with the stance labelling (for example, if we only counted votes for
contributions that happened to have a directional comment under one labelling
rule), then the vote side would become a function of the stance outcome and
the comparison would be circular. Fixing the set up front keeps `V`
independent of `F` and of every sensitivity choice, so that each sensitivity
analysis changes only the comment side. In the bootstrap, these same
contributions are resampled as clusters and **both** `F` and `V` are
recomputed from the one resampled set, preserving the pairing between the two
shares within each contribution.

The stance side uses the comments of exactly those same contributions.

### Primary analysis

- **Unit:** the comment (one directional comment contributes one observation
  to `F`).
- **Mixed comments:** a comment that is `in_favor` toward one key message and
  `against` toward another is **excluded** from `F` in the primary analysis.
- **Labels:** taken as-is from the pair-label set; no override.
- **Pooling:** `F` and `V` are pooled totals over the fixed contribution set.

## Pre-registered sensitivity analyses

Each sensitivity analysis has exactly the same shape as the primary (both CI
levels, same fixed vote-side set, same bootstrap seed) and changes only one
choice on the comment side:

- **(a) Pair level.** The unit of observation is the (comment, key-message)
  pair rather than the comment: each directional pair contributes one
  observation to `F`.
- **(b) Generic-approval override.** The override unit is the **flagged pair**,
  not the whole comment: a generic-approval note re-labels only the pair it is
  attached to, before the comment is rolled up, so the comment keeps the
  stance of its substantive pairs. (A comment with a flagged `in_favor` pair
  and a substantive `against` pair therefore becomes `against` once the
  flagged pair is set to neutral — never a wholesale `neutral`.) Two variants
  are reported: flagged pairs re-labelled to `neutral` (so they drop out of
  the directional `F`), and flagged pairs re-labelled to `in_favor`. The
  **strict** note matcher drives this sensitivity; the **broad** matcher's
  flagged count is reported for context. The summary reports both the flagged
  **pair** counts (`generic_approval_pairs_strict` / `_broad`, the override
  unit) and the **comment** counts of comments with at least one flagged pair
  (`generic_approval_comments_strict` / `_broad`). (See *Generic-approval
  matcher*.)
- **(c) Per-contribution unweighted mean.** Instead of pooling, `F` and `V`
  are computed per contribution and then averaged with equal weight across
  contributions (large contributions no longer dominate).
- **(d) Mixed comments counted as a half.** Mixed comments are not excluded
  but contribute 0.5 to the `in_favor` count and 0.5 to the `against` count.

## Data sources and exclusions

- **Gold contributions.** Derived from the primary_AB pair-label set:
  `evaluator_v2.select_reference(reference, role="primary", tiers=("A","B"))`
  restricted to `resolution_status == "resolved"`. This reuses the existing
  selection (excluding the 91 NaN-tier rows and 7 secondary rows) rather than
  reimplementing it.
- **`contribution_id → project_id`.** Resolved from the local contributions
  table; the fetch confirms each contribution against the feature `id`
  returned by its project's contributions endpoint. Contributions that map to
  no project (**unmapped**) or to more than one project (**collision**) are
  reported and excluded.
- **Contributions missing from the live API.** A gold contribution whose id is
  not returned by the API is **excluded from both sides**, counted, and its id
  listed in the summary.
- **Zero-vote contributions.** A contribution that is found in the API but has
  `votingPro + votingContra == 0` is **kept** and counted as a zero-vote
  contribution. No extra sensitivity analysis is defined for it.
- **Zero-denominator bootstrap resamples.** A bootstrap resample in which the
  pooled denominator of either share is zero is **dropped**, and the number of
  dropped resamples is reported. (With many contributions this is essentially
  never triggered, but it is handled explicitly for correctness.)

### Generic-approval matcher

Sensitivity (b) depends on annotator notes in the evaluation SQLite database
(`dataset_builder.DEFAULT_EVAL_DB`, primary evaluator
`dataset_builder.PRIMARY_EVALUATOR`), opened **read-only**
(`?mode=ro`, `uri=True`). Two matchers over the free-text note are used:

- **strict:** `(?:generell|allgemein)\w*\s+zustimmend` — this drives
  sensitivity (b).
- **broad:** `zustimm` **and** (`generell` or `allgemein`) anywhere in the
  note — its flagged count is reported for context and is always `>=` the
  strict count.

If the evaluation database is missing, the analysis **fails with a clear
error**; sensitivity (b) is never silently skipped.

## Snapshot timing (a limitation)

Votes are a live, monotonically growing quantity, whereas the comment labels
are frozen gold data. The fetch therefore records the UTC time of the vote
snapshot, and the report states plainly that the two sides are measured at
different times. This is a **representativeness limitation**, not a bug: the
comparison is between a frozen comment-label set and a later vote snapshot.

## Reproducibility and security

- Every run writes into a brand-new timestamped directory under
  `dataset/experiments/`; nothing is ever overwritten.
- Every input is hashed; `artifact_hashes.json` is written **last** as the
  completion marker.
- The fetch is the only networked step. Its API host is fixed in code, the
  project id is percent-encoded, and no credentials or `.env` are used.
- The analysis is fully offline, reads the evaluation database read-only, and
  never evaluates any payload.

## Outputs

- `vote_ratio_fetch.summary.json` — snapshot provenance and per-project fetch
  status.
- `contribution_votes.parquet` — the vote snapshot.
- `vote_ratio_analysis.summary.json` — all counts, the primary result and
  every sensitivity result.
- `vote_ratio_report.md` — an English report with the TOST verdict (90%, with
  95% reported), both interpretation caveats, the counts table, the neutral
  share, the snapshot time, the representativeness limitation and every
  sensitivity analysis.
