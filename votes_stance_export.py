"""Build the "votes vs. comment stance" export for the evaluation backend.

This is a small, offline, provenance-tracked exporter. It joins three things
that already exist in this repo and freezes them into one self-describing
directory the DIPAS evaluation backend can load:

1. The frozen gold comment labels (the primary_AB resolved pair-label set, the
   same 707 pairs / 115 contributions used by the pre-registered vote-ratio
   test), via :func:`vote_ratio_hypothesis.pair_label_set`.
2. One stance column per evaluated model, read from the frozen
   ``whole_process`` prediction parquets and normalised to the canonical
   in_favor/against/neutral vocabulary.
3. A copy of a frozen vote snapshot (never a live API call), hash-verified via
   :func:`vote_ratio_hypothesis.load_snapshot`.

Nothing here is networked and nothing mutates its inputs. The per-contribution
figures this feeds are descriptive only; they are not part of the
pre-registered vote-ratio equivalence test (see :data:`DESCRIPTIVE_ONLY_NOTE`).
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

import vote_ratio_hypothesis as vr
from dataset_builder import compute_pair_id
from reaction_split_review import comment_id_for
from review_pack_common import (
    CREATED_UTC_FORMAT,
    sha256_file,
    write_artifact_hashes,
    write_json,
)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# All runs write a new timestamped directory under here, next to the
# vote-ratio artifacts, and never overwrite an existing one.
EXPERIMENTS_DIR = Path("dataset/experiments")
# Where the frozen model-prediction parquets live.
DATA_DIR = Path("data")
# The frozen vote snapshot this export is pinned to (no API call is ever made).
DEFAULT_VOTES_SNAPSHOT_DIR = (
    EXPERIMENTS_DIR / "vote_ratio_votes_20261026T094846675419Z"
)
# The gold pairs table (same default as the vote-ratio test).
DEFAULT_PAIRS_PATH = vr.DEFAULT_PAIRS_PATH
# The frozen, PRE-REGISTERED vote-ratio analysis summary. The overall headline
# figures in this export are copied verbatim from here and are never recomputed
# from the raw pairs/votes, so the export can never silently drift from the
# number that was actually pre-registered and reported.
DEFAULT_ANALYSIS_SUMMARY_PATH = (
    EXPERIMENTS_DIR
    / "vote_ratio_analysis_20261026T094931373229Z"
    / "vote_ratio_analysis.summary.json"
)

# Directory prefix and artifact file names, mirroring the vote-ratio artifacts.
EXPORT_DIR_PREFIX = "votes_stance_export_"
COMMENTS_FILE = "votes_stance_comments.parquet"
KEY_MESSAGES_FILE = "votes_stance_key_messages.parquet"
VOTES_FILE = "votes_snapshot.parquet"
SUMMARY_FILE = "votes_stance_export.summary.json"
SCHEMA_VERSION = "votes_stance_export_v1"

# The clustered-bootstrap settings for the *exploratory* shared-contributions
# figures below. The seed is fixed so the published confidence range is exactly
# reproducible; it matches the seed used by the frozen prototype these numbers
# were first produced with.
SHARED_BOOTSTRAP_N = 10_000
SHARED_BOOTSTRAP_SEED = 20261026

# The model prediction files to offer, by model name. Only these are exported:
# the empty single-s "deepseek-v4-pro" file and the xlsx variants are excluded
# deliberately (see the user's decisions). "base" is the un-suffixed run.
DEFAULT_MODELS: dict[str, Path] = {
    "base": DATA_DIR / "whole_process_contributions_with_comments_250.parquet",
    "deepsseek-v4-pro": DATA_DIR
    / "whole_process_contributions_with_comments_250_deepsseek-v4-pro_deepsseek-v4-pro.parquet",
    "gptoss-120b": DATA_DIR
    / "whole_process_contributions_with_comments_250_gptoss-120b_gptoss-120b.parquet",
    "llama-31-405b-fp8": DATA_DIR
    / "whole_process_contributions_with_comments_250_llama-31-405b-fp8_llama-31-405b-fp8.parquet",
    "mistral-nemo": DATA_DIR
    / "whole_process_contributions_with_comments_250_mistral-nemo_mistral-nemo.parquet",
}

# The raw stance vocabulary used in the prediction parquets, mapped to the
# canonical labels used everywhere else (gold labels, roll-up, backend).
STANCE_NORMALIZATION = {
    "in favor": "in_favor",
    "ablehnung": "against",
    "neutral": "neutral",
}

# The prefix of a model's rolled-up comment-category column in the export.
CATEGORY_PREFIX = "category__"

# Verbatim caveat carried into the summary and shown in the UI; it stops the
# per-contribution figures being read as part of the pre-registered test.
DESCRIPTIVE_ONLY_NOTE = (
    "These per-contribution figures are descriptive only and are not part of "
    "the pre-registered vote-ratio test."
)

# Verbatim caveat for the shared-contributions block. Unlike the pre-registered
# headline figures (which compare two independently fixed sets), this block
# restricts both sides to the SAME contributions. That restriction was decided
# after seeing the data, so it is exploratory, not pre-registered, and must be
# labelled as such everywhere it is shown.
NOT_PRE_REGISTERED_NOTE = (
    "This shared-contributions comparison was not pre-registered. It restricts "
    "both the thumbs side and the comment side to the same contributions (those "
    "with at least one thumbs vote and at least one comment carrying a clear "
    "stance), which was decided after seeing the data. Read it as exploratory "
    "context, not as a confirmatory result."
)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _created_utc(now: datetime | None) -> str:
    """Return a creation timestamp in CREATED_UTC_FORMAT.

    Args:
        now: Creation time; defaults to the current UTC time. A naive datetime
            is read as local time and converted to UTC.

    Returns:
        The formatted UTC timestamp used to name the run directory.
    """
    # Normalising to UTC keeps directory names independent of the local zone,
    # exactly as the vote-ratio artifacts do.
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime(
        CREATED_UTC_FORMAT
    )


def normalize_stance(raw: object) -> str:
    """Map a raw prediction stance to the canonical in_favor/against/neutral.

    Args:
        raw: A stance string from a prediction parquet ("in favor",
            "ablehnung" or "neutral").

    Returns:
        The canonical label.

    Raises:
        ValueError: If ``raw`` is not one of the three known stance strings;
            an unexpected label must never be silently dropped or miscounted.
    """
    try:
        return STANCE_NORMALIZATION[raw]
    except KeyError:
        raise ValueError(f"Unexpected stance label {raw!r}") from None


def load_models(model_paths: Mapping[str, Path]) -> dict[str, pd.DataFrame]:
    """Load each model's predictions into a (pair_id -> model_label) frame.

    Every model file is read independently. Comment and pair ids are derived
    with the pipeline's own helpers (:func:`comment_id_for`,
    :func:`compute_pair_id`) over the *stringified* contribution id, so the ids
    produced here join to the gold label set and the vote snapshot without any
    re-implementation of the id rule.

    Args:
        model_paths: Maps a model name to its prediction parquet path.

    Returns:
        Maps each model name to a DataFrame with columns ``pair_id``,
        ``comment_id``, ``contribution_id`` (str) and ``model_label``
        (canonical), one row per prediction pair.

    Raises:
        FileNotFoundError: If a model file does not exist.
        ValueError: If a file is empty (0 rows) or carries an unknown stance.
    """
    loaded: dict[str, pd.DataFrame] = {}
    for name, path in model_paths.items():
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file for {name!r} not found: {path}")
        df = pd.read_parquet(path)
        # An empty file (e.g. the excluded single-s deepseek export) carries no
        # usable predictions; refuse it rather than emit an all-NaN column.
        if df.empty:
            raise ValueError(f"Model file for {name!r} is empty: {path}")
        # Stringify once so every derived id uses the same key type as the gold
        # set and the votes.
        contribution_id = df["contribution_id"].astype(str)
        model_label = df["stance"].map(normalize_stance)
        frame = pd.DataFrame(
            {
                "contribution_id": contribution_id,
                "comment_id": [
                    comment_id_for(cid, text)
                    for cid, text in zip(contribution_id, df["comment_text"])
                ],
                "pair_id": [
                    compute_pair_id(cid, km, text)
                    for cid, km, text in zip(
                        contribution_id, df["key_message"], df["comment_text"]
                    )
                ],
                "model_label": model_label.to_numpy(),
            }
        )
        loaded[name] = _collapse_duplicate_pairs(frame, name)
    return loaded


def _collapse_duplicate_pairs(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Collapse rows that share a pair_id to one row per pair.

    The prediction files contain a handful of exact-duplicate rows (the same
    contribution/key-message/comment predicted twice with the same label). If
    they were left in, the per-comment roll-up in :func:`build_comments_table`
    would double-count. Identical duplicates are collapsed to one row; a
    genuine label conflict (the same pair predicted two different labels) is a
    data error and is never silently resolved.

    Args:
        frame: A per-model frame with pair_id/comment_id/contribution_id/
            model_label columns.
        name: The model name, for error messages.

    Returns:
        The frame with one row per pair_id.

    Raises:
        ValueError: If a pair_id carries more than one distinct model_label.
    """
    conflicting = frame.groupby("pair_id").model_label.nunique()
    conflicting = conflicting[conflicting > 1]
    if not conflicting.empty:
        raise ValueError(
            f"Model {name!r} has conflicting labels for pair(s): "
            f"{sorted(conflicting.index)}"
        )
    return frame.drop_duplicates(subset="pair_id").reset_index(drop=True)


def _rollup_by_comment(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Roll a pair-level label frame up to one category per comment.

    Reuses :func:`vote_ratio_hypothesis.rollup_comments` verbatim (renaming the
    label column to the name it expects), so the in_favor/against/mixed/neutral
    rule is never reimplemented.

    Args:
        df: Rows with ``comment_id``, ``contribution_id`` and ``label_col``.
        label_col: The column holding the canonical per-pair label.

    Returns:
        One row per comment with ``comment_id``, ``contribution_id`` and
        ``category``.
    """
    tmp = df[["comment_id", "contribution_id", label_col]].rename(
        columns={label_col: "reference_label"}
    )
    return vr.rollup_comments(tmp)


def build_comments_table(
    pairs_path: Path, models: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Build the per-comment category table across gold and every model.

    The roll-up unit is the comment, identified by ``comment_id`` (a hash of
    contribution id and comment text, independent of the key message). This is
    the only unit comparable across sources: different models extract different
    key messages, so a key-message-dependent ``pair_id`` does not join across
    models, but the comment set does.

    The comment universe is the frozen gold comment set (the comments of the
    primary_AB resolved pairs). Each model's own pairs are rolled up by comment
    and left-joined onto that universe, so a comment a model never labelled
    simply gets a missing category rather than shrinking the universe.

    Args:
        pairs_path: Path to the gold pairs parquet.
        models: The per-model frames from :func:`load_models`.

    Returns:
        One row per gold comment with ``comment_id``, ``contribution_id``
        (str), ``comment_text``, ``gold_category`` and one
        ``category__<model>`` column per model (in_favor/against/mixed/neutral,
        or missing where the model has no pair for that comment).
    """
    gold = vr.pair_label_set(pairs_path)
    comments = _rollup_by_comment(gold, "reference_label").rename(
        columns={"category": "gold_category"}
    )
    # One representative comment text per comment id (identical within a group).
    comment_text = gold.drop_duplicates("comment_id").set_index("comment_id")[
        "comment_text"
    ]
    comments["comment_text"] = comments.comment_id.map(comment_text)
    for name, frame in models.items():
        model_categories = _rollup_by_comment(frame, "model_label")[
            ["comment_id", "category"]
        ].rename(columns={"category": f"{CATEGORY_PREFIX}{name}"})
        comments = comments.merge(model_categories, on="comment_id", how="left")
    return comments.reset_index(drop=True)


def copy_votes(snapshot_dir: Path) -> tuple[pd.DataFrame, str, str]:
    """Load (and hash-verify) the frozen vote snapshot, with its snapshot time.

    No network call is made: the snapshot is read from a completed
    ``vote-ratio-fetch`` directory and verified against its recorded hash by
    :func:`vote_ratio_hypothesis.load_snapshot`.

    Args:
        snapshot_dir: A completed ``vote-ratio-fetch`` directory.

    Returns:
        A tuple ``(votes, snapshot_utc, snapshot_sha256)``. ``snapshot_utc`` is
        the fetch run's creation timestamp (when the votes were read).

    Raises:
        RuntimeError: If the snapshot's hash differs from the recorded one.
        ValueError: If the directory does not record a completed snapshot.
    """
    snapshot_dir = Path(snapshot_dir)
    votes, sha = vr.load_snapshot(snapshot_dir)
    # The snapshot time is the fetch run's created_utc, recorded in its summary.
    summary_path = snapshot_dir / vr.FETCH_SUMMARY_FILE
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    snapshot_utc = summary.get("created_utc")
    return votes, snapshot_utc, sha


def build_key_messages_table(pairs_path: Path) -> pd.DataFrame:
    """Build the per-gold-pair key-message table for the per-key-message view.

    The backend's per-key-message agreement view needs the gold labels at the
    *key-message* grain, not the comment grain the headline figures roll up to.
    Each row here is one gold scoring pair: a (contribution, key message,
    comment) triple with its single resolved label.

    This reuses :func:`vote_ratio_hypothesis.pair_label_set` verbatim, so the
    key messages, comment ids and labels are exactly the gold set the
    pre-registered test uses; nothing about the gold selection is
    re-implemented here.

    The row order is preserved exactly as ``pair_label_set`` yields it, which is
    first-appearance order of the key messages within the audited pairs table.
    Keeping that order lets the backend list a contribution's key messages in a
    stable, human-meaningful sequence rather than an arbitrary one.

    Args:
        pairs_path: Path to the gold pairs parquet.

    Returns:
        One row per gold pair with columns ``contribution_id`` (str),
        ``key_message``, ``comment_id``, ``comment_text`` and
        ``reference_label`` (in_favor/against/neutral).
    """
    # Load the gold pair-label set; it already stringifies the contribution id
    # and derives the per-comment id with the shared helper.
    gold = vr.pair_label_set(pairs_path)
    # Keep only the columns the per-key-message view needs, in a fixed order,
    # dropping the internal pair_id. The index is reset so the parquet carries a
    # clean 0..n-1 row order that mirrors first-appearance key-message order.
    return gold[
        ["contribution_id", "key_message", "comment_id", "comment_text", "reference_label"]
    ].reset_index(drop=True)


def load_overall_summary(analysis_summary_path: Path) -> dict:
    """Load the PRE-REGISTERED overall headline figures, copied verbatim.

    The overall "thumbs vs. comment stance" headline numbers shown to
    stakeholders are the pre-registered vote-ratio result. They are read
    verbatim from the frozen analysis summary and are **never** recomputed from
    the raw pairs or votes here: recomputing would risk quietly reporting a
    number that differs from the one that was actually pre-registered.

    Args:
        analysis_summary_path: Path to the frozen
            ``vote_ratio_analysis.summary.json``.

    Returns:
        A dict with ``pre_registered`` True, the full ``primary`` block copied
        verbatim, the comment total ``n_comments``, a ``no_clear_stance``
        breakdown (neutral + mixed comments), ``n_contributions_with_votes`` and
        the verbatim ``interpretation_rules`` text.
    """
    # Read the frozen analysis summary as-is.
    analysis = json.loads(Path(analysis_summary_path).read_text(encoding="utf-8"))
    # The "primary" block holds the pre-registered headline: favor/against/
    # neutral/mixed counts, favor_share, the pooled vote figures, the point
    # difference and its 90% interval with the equivalence verdict. Copy it
    # verbatim (a deep copy, so a later caller can never mutate the source
    # values we read).
    primary = copy.deepcopy(analysis["primary"])
    counts = analysis["counts"]
    # "No clear stance" is every comment that is neither purely in favour nor
    # purely against: the neutral comments plus the mixed comments. Both counts
    # are taken verbatim from the pre-registered primary block; only their sum
    # is formed here, purely for display convenience.
    neutral_comments = primary["neutral"]
    mixed_comments = primary["mixed"]
    return {
        "pre_registered": True,
        "primary": primary,
        "n_comments": counts["n_comments"],
        "no_clear_stance": {
            "total": neutral_comments + mixed_comments,
            "neutral": neutral_comments,
            "mixed": mixed_comments,
        },
        "n_contributions_with_votes": counts["n_contributions_with_votes"],
        "interpretation_rules": analysis["interpretation_rules"],
    }


def compute_shared_contributions(pairs_path: Path, votes: pd.DataFrame) -> dict:
    """Compute the EXPLORATORY shared-contributions comparison (not pre-registered).

    The pre-registered headline pools the thumbs side over every contribution
    with any votes and the comment side over every contribution with any
    directional comment, and those two sets are not the same contributions. This
    block instead asks a narrower, exploratory question: among the
    contributions that have BOTH a thumbs vote and a comment with a clear
    stance, how close are the favour share and the thumbs-up share? Because that
    "both sides present" restriction was chosen after seeing the data, this is
    not pre-registered and is labelled accordingly.

    Two summaries are reported, both over the same shared set:

    * ``pooled`` adds up all in-favour comments / all directional comments and
      all thumbs-up / all thumbs, so busy contributions weigh more.
    * ``unweighted`` averages each contribution's own two shares, so every
      contribution counts once regardless of size.

    Each gap is bracketed by a clustered bootstrap that resamples whole
    contributions (keeping each contribution's comments and votes together), so
    the uncertainty reflects variation between contributions.

    Args:
        pairs_path: Path to the gold pairs parquet.
        votes: The frozen vote snapshot (contribution_id as str, with
            ``votingPro`` / ``votingContra`` counts). Its rows define the
            contribution universe and their order fixes the bootstrap draw.

    Returns:
        A dict with ``pre_registered`` False, the not-pre-registered note, the
        three contribution counts (thumbs side / comment side / both), the
        bootstrap settings, and a ``pooled`` and ``unweighted`` block each
        carrying ``difference_pp`` and a 90% ``ci90`` low/high range.
    """
    # Step 1 -- gold comment categories.
    # Roll the gold pairs up to one category per comment with the SAME
    # pre-registered rule the headline uses (in_favor / against / neutral /
    # mixed). We never re-implement that rule here.
    rollup = vr.rollup_comments(vr.pair_label_set(pairs_path))

    # Step 2 -- per-contribution in-favour / against comment counts.
    # Pivot the category counts so each contribution has an in_favor and an
    # against column. Neutral and mixed comments deliberately do not feed the
    # favour share: a contribution's directional total is in_favor + against.
    category_counts = (
        rollup.groupby("contribution_id")
        .category.value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for column in ("in_favor", "against"):
        # A category can be entirely absent from the data; guarantee the column
        # exists so the arithmetic below never raises on a missing column.
        if column not in category_counts:
            category_counts[column] = 0

    # Step 3 -- join the comment counts onto the vote snapshot.
    # The vote snapshot's rows are the contribution universe. A left join keeps
    # every contribution that has votes even when it has no comment (its counts
    # become 0), and the fixed snapshot row order is what makes the seeded
    # bootstrap below reproducible.
    merged = votes.copy()
    merged["tot"] = merged.votingPro.fillna(0) + merged.votingContra.fillna(0)
    merged = merged.merge(
        category_counts[["contribution_id", "in_favor", "against"]],
        on="contribution_id",
        how="left",
    )
    merged[["in_favor", "against"]] = merged[["in_favor", "against"]].fillna(0)
    # n is each contribution's directional comment count (the favour-share
    # denominator); tot is its total thumbs (the thumbs-share denominator).
    merged["n"] = merged.in_favor + merged.against

    # Step 4 -- pull the four quantities out as plain float arrays, aligned by
    # contribution. favour numerator (f), directional comments (n), thumbs-up
    # (p) and total thumbs (t).
    favor = merged.in_favor.to_numpy(dtype=float)
    directional = merged.n.to_numpy(dtype=float)
    thumbs_up = merged.votingPro.fillna(0).to_numpy(dtype=float)
    thumbs_total = merged.tot.to_numpy(dtype=float)
    n_contributions = len(merged)

    def _pooled(index: np.ndarray) -> float:
        """Pooled favour share minus pooled thumbs-up share, over the shared set.

        Args:
            index: Row indices selecting the contributions in this (possibly
                resampled) draw.

        Returns:
            The gap in percentage points, or NaN when this draw happens to
            contain no shared contribution (an undefined draw, dropped from the
            interval). The shared-set mask is applied inside the draw so only
            contributions present on both sides are pooled.
        """
        # Keep only contributions that have at least one directional comment AND
        # at least one thumbs vote in this draw.
        both = (directional[index] > 0) & (thumbs_total[index] > 0)
        # A resample can, in principle, contain no shared contribution; that
        # draw is undefined, so return NaN rather than divide by zero.
        if not both.any():
            return float("nan")
        favor_share = 100.0 * favor[index][both].sum() / directional[index][both].sum()
        thumbs_share = 100.0 * thumbs_up[index][both].sum() / thumbs_total[index][both].sum()
        return favor_share - thumbs_share

    def _unweighted(index: np.ndarray) -> float:
        """Mean per-contribution favour share minus mean thumbs-up share.

        Args:
            index: Row indices selecting the contributions in this (possibly
                resampled) draw.

        Returns:
            The gap in percentage points, or NaN when this draw contains no
            shared contribution. Each shared contribution's own two shares are
            averaged, so every contribution weighs the same.
        """
        both = (directional[index] > 0) & (thumbs_total[index] > 0)
        if not both.any():
            return float("nan")
        favor_share = 100.0 * (favor[index][both] / directional[index][both]).mean()
        thumbs_share = 100.0 * (thumbs_up[index][both] / thumbs_total[index][both]).mean()
        return favor_share - thumbs_share

    # Step 5 -- point estimates over the full (unresampled) set. arange selects
    # every contribution exactly once, so this is the observed gap.
    full = np.arange(n_contributions)
    pooled_point = _pooled(full)
    unweighted_point = _unweighted(full)

    # Step 6 -- clustered bootstrap. Draw whole contributions with replacement
    # (so a contribution's comments and votes always move together), recompute
    # each gap, and read the 5th/95th percentiles as the 90% range. All draws
    # are generated from one seeded generator and reused for both estimators, so
    # the published range is exactly reproducible.
    rng = np.random.default_rng(SHARED_BOOTSTRAP_SEED)
    draws = [
        rng.integers(0, n_contributions, n_contributions)
        for _ in range(SHARED_BOOTSTRAP_N)
    ]
    pooled_draws = np.array([_pooled(draw) for draw in draws])
    unweighted_draws = np.array([_unweighted(draw) for draw in draws])
    # nanpercentile ignores any undefined (NaN) draw. On the real data no draw
    # is undefined (80 of 115 contributions are shared), so this is identical
    # to a plain percentile there; the guard only matters for tiny synthetic
    # inputs where a resample can miss the shared set entirely.
    pooled_low, pooled_high = np.nanpercentile(pooled_draws, [5, 95])
    unweighted_low, unweighted_high = np.nanpercentile(unweighted_draws, [5, 95])

    # Step 7 -- the three headline counts, for the "107 / 85 / 80" line in the
    # UI. Thumbs side = any votes; comment side = any directional comment; both
    # = the shared set the figures above are computed over.
    thumbs_side = int((thumbs_total > 0).sum())
    comment_side = int((directional > 0).sum())
    both_sides = int(((directional > 0) & (thumbs_total > 0)).sum())

    return {
        "pre_registered": False,
        "note": NOT_PRE_REGISTERED_NOTE,
        "thumbs_side_contributions": thumbs_side,
        "comment_side_contributions": comment_side,
        "both_sides_contributions": both_sides,
        "bootstrap": {"n": SHARED_BOOTSTRAP_N, "seed": SHARED_BOOTSTRAP_SEED},
        "pooled": {
            "difference_pp": float(pooled_point),
            "ci90": {"low": float(pooled_low), "high": float(pooled_high)},
        },
        "unweighted": {
            "difference_pp": float(unweighted_point),
            "ci90": {"low": float(unweighted_low), "high": float(unweighted_high)},
        },
    }


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def run_export(
    *,
    pairs_path: Path = DEFAULT_PAIRS_PATH,
    model_paths: Mapping[str, Path] | None = None,
    votes_snapshot_dir: Path = DEFAULT_VOTES_SNAPSHOT_DIR,
    analysis_summary_path: Path = DEFAULT_ANALYSIS_SUMMARY_PATH,
    output_dir: Path | None = None,
    now: datetime | None = None,
) -> vr.ArtifactRun:
    """Freeze the votes/stance export into a fresh timestamped directory.

    Args:
        pairs_path: Gold pairs parquet. Defaults to the vote-ratio default.
        model_paths: Model name -> prediction parquet. Defaults to
            :data:`DEFAULT_MODELS`.
        votes_snapshot_dir: A completed vote-ratio-fetch directory (only read).
        analysis_summary_path: The frozen, pre-registered vote-ratio analysis
            summary; its headline figures are copied verbatim into the export.
        output_dir: Directory to create (its parent must exist). Defaults to
            ``EXPERIMENTS_DIR / "votes_stance_export_<created_utc>"``.
        now: Creation time (for tests). Defaults to the current UTC time.

    Returns:
        The output directory, the written files and the summary mapping.

    Raises:
        FileExistsError: If the output directory already exists.
    """
    model_paths = dict(DEFAULT_MODELS if model_paths is None else model_paths)
    created_utc = _created_utc(now)

    models = load_models(model_paths)
    comments_table = build_comments_table(pairs_path, models)
    # Per-key-message gold labels for the per-key-message agreement view.
    key_messages_table = build_key_messages_table(pairs_path)
    votes, snapshot_utc, snapshot_sha = copy_votes(votes_snapshot_dir)

    # Gold favor counts over the comment categories, reusing the pre-registered
    # rule so the recorded counts can never drift from the hypothesis module.
    gold_favor = vr.favor_share(comments_table.gold_category)

    # The PRE-REGISTERED headline, copied verbatim from the frozen analysis
    # summary (never recomputed), and the EXPLORATORY shared-contributions
    # block, computed fresh here and clearly flagged not-pre-registered.
    overall_summary = load_overall_summary(analysis_summary_path)
    shared_contributions = compute_shared_contributions(pairs_path, votes)

    available_models = sorted(models)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created_utc,
        "module_sha256": sha256_file(Path(__file__).resolve()),
        "descriptive_only_note": DESCRIPTIVE_ONLY_NOTE,
        "inputs": {
            "pairs": {
                "path": str(pairs_path),
                "sha256": sha256_file(Path(pairs_path)),
            },
            "models": {
                name: {
                    "path": str(model_paths[name]),
                    "sha256": sha256_file(Path(model_paths[name])),
                }
                for name in available_models
            },
            "votes_snapshot": {
                "path": str(votes_snapshot_dir),
                "snapshot_utc": snapshot_utc,
                "sha256": snapshot_sha,
            },
            # The frozen pre-registered analysis summary is a real input: its
            # headline figures are copied verbatim, so hash-verify it too.
            "overall_analysis": {
                "path": str(analysis_summary_path),
                "sha256": sha256_file(Path(analysis_summary_path)),
            },
        },
        "available_models": available_models,
        "snapshot_utc": snapshot_utc,
        "counts": {
            "comments": int(comments_table.comment_id.nunique()),
            "contributions": int(comments_table.contribution_id.nunique()),
            "key_message_rows": int(len(key_messages_table)),
            "key_messages": int(
                key_messages_table.drop_duplicates(
                    ["contribution_id", "key_message"]
                ).shape[0]
            ),
            "votes_rows": int(len(votes)),
            "gold_favor_counts": {
                k: gold_favor[k]
                for k in ("favor", "against", "neutral", "mixed")
            },
        },
        # The pre-registered headline and the exploratory shared-set figures.
        "overall_summary": overall_summary,
        "shared_contributions": shared_contributions,
    }

    output_dir = (
        Path(output_dir)
        if output_dir
        else EXPERIMENTS_DIR / f"{EXPORT_DIR_PREFIX}{created_utc}"
    )
    # Refuse to touch an existing directory: every run is a fresh snapshot.
    output_dir.mkdir(parents=False, exist_ok=False)
    files = {
        "comments": output_dir / COMMENTS_FILE,
        "key_messages": output_dir / KEY_MESSAGES_FILE,
        "votes": output_dir / VOTES_FILE,
        "summary": output_dir / SUMMARY_FILE,
    }
    comments_table.to_parquet(files["comments"], index=False)
    key_messages_table.to_parquet(files["key_messages"], index=False)
    votes.to_parquet(files["votes"], index=False)
    write_json(files["summary"], summary)
    # artifact_hashes.json is written LAST: it is the completion marker and
    # covers every other artifact, so a half-written run has no hashes file.
    files["hashes"] = write_artifact_hashes(
        output_dir,
        created_utc,
        [files["comments"], files["key_messages"], files["votes"], files["summary"]],
    )
    return vr.ArtifactRun(output_dir, files, summary)
