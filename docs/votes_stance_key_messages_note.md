# Votes vs. stance: key-message artifacts and the shared-contributions block

This note documents the artifacts the `votes-stance` export adds for the
per-key-message views, and — importantly — the difference in standing between
the two summary blocks the export now carries. It does **not** restate the
pre-registered analysis (see `docs/vote_ratio_hypothesis.md` for that); it only
explains what is new and how to read it.

## What the new artifacts are

The exporter (`votes_stance_export.py`, `run_export`) now writes, alongside the
existing per-comment table, vote snapshot and summary:

- **`votes_stance_key_messages.parquet`** — one row per gold scoring pair, at the
  **key-message grain** (a `contribution_id`, `key_message`, `comment_id`,
  `comment_text` and resolved `reference_label`). It is built by
  `build_key_messages_table`, which reuses `vote_ratio_hypothesis.pair_label_set`
  verbatim, so the key messages, comments and labels are exactly the gold set the
  pre-registered test uses. Row order is the gold set's first-appearance order,
  so the backend can list a contribution's key messages in a stable sequence.
  In the frozen export this is 707 rows over 320 distinct (contribution, key
  message) pairs.

The summary (`votes_stance_export.summary.json`) gains two blocks:

- **`overall_summary`** — the headline thumbs-vs-comment-stance figures.
- **`shared_contributions`** — an exploratory narrower comparison.

The backend serves these through three gold-only endpoints (see the backend
`README.md`): `/votes-stance/summary`, `/contributions/{id}/votes-stance/key-messages`
and `/votes-stance/key-messages/overview`.

## `overall_summary` is copied; `shared_contributions` is computed fresh

This distinction is the reason this note exists.

- **`overall_summary` is pre-registered and copied verbatim.** `load_overall_summary`
  reads the frozen `vote_ratio_analysis.summary.json` and copies its `primary`
  block (favor/against/neutral/mixed counts, favor share, pooled vote figures,
  the point difference and its 90% interval with the equivalence verdict) without
  recomputing anything from the raw pairs or votes. Recomputing would risk
  silently reporting a number that differs from the one actually pre-registered,
  so the export deliberately never does. This block carries `pre_registered: true`.

- **`shared_contributions` is computed fresh here and is NOT pre-registered.**
  `compute_shared_contributions` restricts **both** the thumbs side and the
  comment side to the same contributions — those with at least one thumbs vote
  **and** at least one comment carrying a clear stance. That restriction was
  chosen after seeing the data, so it is exploratory. The block carries
  `pre_registered: false` and an explicit note.

### Why the shared set does not replace the main result

The three contribution sets are different sizes (figures from the frozen export):

- **107** contributions have thumbs votes (the thumbs side);
- **85** contributions have at least one comment with a clear stance (the comment side);
- **80** contributions have **both**.

The pre-registered headline pools each side over its own fixed set (thumbs over
all 107, comments over all 85). Those two sets are fixed independently of the
stance labels. Restricting to the 80 shared contributions instead lets the
stance labels decide which contributions — and therefore which votes — count.
That is exactly the data-dependent selection the pre-registered fixed sets are
designed to avoid, so the shared-contributions figures are extra context, not a
replacement for the main result.

The block reports two gaps over the shared set, each with a seeded
clustered-bootstrap 90% range (whole contributions resampled together, seed
`20261026`, 10,000 draws, so the published range is exactly reproducible):

- `pooled` — sums all in-favour / all directional comments and all thumbs-up /
  all thumbs, so busier contributions weigh more;
- `unweighted` — averages each contribution's own two shares, so every
  contribution counts once.

## Key-message counts are descriptive only

The per-key-message agreement/disagreement counts (and the "most agreed" / "most
rejected" overview) are **descriptive**: they report how many gold-labelled
comments take each stance toward a key message. They do not establish causation
and are not a significance test. The overview lists counts only (never
percentages) and never groups key messages across contributions; a key message
with fewer than three comments carrying a clear stance is flagged `too_few` and
excluded from the ranked lists, because a key message backed by one or two
comments is too thin to rank.
