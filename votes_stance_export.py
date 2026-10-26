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

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

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

# Directory prefix and artifact file names, mirroring the vote-ratio artifacts.
EXPORT_DIR_PREFIX = "votes_stance_export_"
COMMENTS_FILE = "votes_stance_comments.parquet"
VOTES_FILE = "votes_snapshot.parquet"
SUMMARY_FILE = "votes_stance_export.summary.json"
SCHEMA_VERSION = "votes_stance_export_v1"

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


# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------


def run_export(
    *,
    pairs_path: Path = DEFAULT_PAIRS_PATH,
    model_paths: Mapping[str, Path] | None = None,
    votes_snapshot_dir: Path = DEFAULT_VOTES_SNAPSHOT_DIR,
    output_dir: Path | None = None,
    now: datetime | None = None,
) -> vr.ArtifactRun:
    """Freeze the votes/stance export into a fresh timestamped directory.

    Args:
        pairs_path: Gold pairs parquet. Defaults to the vote-ratio default.
        model_paths: Model name -> prediction parquet. Defaults to
            :data:`DEFAULT_MODELS`.
        votes_snapshot_dir: A completed vote-ratio-fetch directory (only read).
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
    votes, snapshot_utc, snapshot_sha = copy_votes(votes_snapshot_dir)

    # Gold favor counts over the comment categories, reusing the pre-registered
    # rule so the recorded counts can never drift from the hypothesis module.
    gold_favor = vr.favor_share(comments_table.gold_category)

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
        },
        "available_models": available_models,
        "snapshot_utc": snapshot_utc,
        "counts": {
            "comments": int(comments_table.comment_id.nunique()),
            "contributions": int(comments_table.contribution_id.nunique()),
            "votes_rows": int(len(votes)),
            "gold_favor_counts": {
                k: gold_favor[k]
                for k in ("favor", "against", "neutral", "mixed")
            },
        },
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
        "votes": output_dir / VOTES_FILE,
        "summary": output_dir / SUMMARY_FILE,
    }
    comments_table.to_parquet(files["comments"], index=False)
    votes.to_parquet(files["votes"], index=False)
    write_json(files["summary"], summary)
    # artifact_hashes.json is written LAST: it is the completion marker and
    # covers every other artifact, so a half-written run has no hashes file.
    files["hashes"] = write_artifact_hashes(
        output_dir, created_utc, [files["comments"], files["votes"], files["summary"]]
    )
    return vr.ArtifactRun(output_dir, files, summary)
