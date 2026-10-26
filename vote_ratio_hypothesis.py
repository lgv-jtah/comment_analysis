"""Vote-ratio equivalence test: fetch DIPAS votes, then compare offline.

This module answers one narrow, pre-registered question (see
``docs/vote_ratio_hypothesis.md``): does the overall *favor share* of the
gold comments match the overall *pro share* of the thumbs-up/down votes on
the same contributions? It is deliberately split into two phases so that the
only networked step is isolated and the statistics are fully reproducible:

1. ``vote-ratio-fetch`` (networked): read the current DIPAS vote counts for
   the gold contributions and freeze them into a timestamped snapshot
   directory. This is the *only* step that touches the internet.
2. ``vote-ratio-analyse`` (offline): run a TOST equivalence test comparing
   the comment-level favor share against the vote pro share, with a clustered
   bootstrap confidence interval and every pre-registered sensitivity
   analysis. No network, no ``.env``, read-only database access.

The comments in this module are intentionally far denser than the rest of the
repository: every function and every logical block states *what* it does and
*why*, so the analysis can be reported to stakeholders step by step. This
density is specific to this module; ``cli.py`` and the tests follow the
normal repository style.
"""
from __future__ import annotations

# Standard library: hashing for the module fingerprint, regex for the
# annotation-note matchers, sqlite3 for the read-only evaluation database,
# dataclasses for the small result object, datetime for the timestamps and
# urllib for percent-encoding the project id in the fetch URL.
import json
import re
import sqlite3
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
# `requests` is imported at module level (not inside the function) so that the
# tests can monkeypatch `vote_ratio_hypothesis.requests.get` exactly as the
# create_dataset tests do; the analysis phase never touches it.
import requests

import dataset_builder
import evaluator_v2
# comment_id_for is the single source of truth for the deterministic comment
# identity; it is reused (never reimplemented) so our IDs match the rest of
# the pipeline.
from reaction_split_review import comment_id_for
from review_pack_common import (
    CREATED_UTC_FORMAT,
    HASHES_FILE,
    sha256_file,
    write_artifact_hashes,
    write_json,
)

# --------------------------------------------------------------------------
# Constants (pre-registered; see docs/vote_ratio_hypothesis.md)
# --------------------------------------------------------------------------

# All artifacts live under this directory; each run makes its own new,
# timestamped subdirectory and never overwrites an existing one.
EXPERIMENTS_DIR = Path("dataset/experiments")
# The DIPAS contributions endpoint; the host is fixed in code (never taken
# from user input) and {project_id} is percent-encoded before substitution.
CONTRIBUTIONS_API_TEMPLATE = (
    "https://beteiligung.hamburg/drupal/dipas-pds/projects/{project_id}/contributions/"
)
# Default input locations (overridable on the CLI).
DEFAULT_PAIRS_PATH = Path("dataset/pairs.parquet")
DEFAULT_CONTRIBUTIONS_PATH = Path("data/contributions.parquet")
DEFAULT_EVAL_DB = dataset_builder.DEFAULT_EVAL_DB

# Directory prefixes and file names, mirroring the reaction_split artifacts.
FETCH_DIR_PREFIX = "vote_ratio_votes_"
ANALYSIS_DIR_PREFIX = "vote_ratio_analysis_"
SNAPSHOT_FILE = "contribution_votes.parquet"
FETCH_SUMMARY_FILE = "vote_ratio_fetch.summary.json"
ANALYSIS_SUMMARY_FILE = "vote_ratio_analysis.summary.json"
REPORT_FILE = "vote_ratio_report.md"
FETCH_SCHEMA_VERSION = "vote_ratio_votes_v1"
ANALYSIS_SCHEMA_VERSION = "vote_ratio_analysis_v1"

# The equivalence band half-width in percentage points: the two shares are
# declared equivalent only if the CI for their difference lies inside
# (-margin, +margin).
EQUIVALENCE_MARGIN_PP = 10.0
# The primary decision uses the 90% CI (matches a two-sided 5% TOST); the 95%
# CI is reported alongside for transparency.
CI_LEVELS = (0.90, 0.95)
# Clustered bootstrap configuration; the seed makes every run reproducible.
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20261026
# The DIPAS feature-property fields we read for each contribution.
VOTE_FIELDS = ("id", "votingPro", "votingContra", "commentsNumber", "dateCreated")

# The verbatim interpretation caveat, reproduced in the doc, the summary and
# the report. It applies in both directions and prevents over-reading the
# verdict.
INTERPRETATION_RULES = (
    "Equivalence does not prove that stance detection is unnecessary; "
    "non-equivalence does not prove that it is necessary. This test answers only "
    "whether the overall favor share of comments matches the overall share of "
    "thumbs up among votes."
)
# Plain-language statement of the snapshot-timing limitation.
REPRESENTATIVENESS_LIMIT = (
    "The comment labels are frozen gold data, while the votes are a live, "
    "monotonically growing quantity read at the snapshot time below. The two "
    "sides are therefore measured at different times; treat the comparison as "
    "between a frozen comment-label set and a later vote snapshot."
)

# Generic-approval note matchers, re-vendored from the historical
# general_approval_review.py (commit d45fd04). The STRICT matcher drives
# sensitivity (b); the BROAD matcher's count is reported for context and is
# always >= the strict count.
GA_NOTE_STRICT = re.compile(r"(?:generell|allgemein)\w*\s+zustimmend", re.IGNORECASE)
GA_NOTE_BROAD = (re.compile(r"zustimm", re.IGNORECASE),
                 re.compile(r"generell|allgemein", re.IGNORECASE))

# The three canonical directional/neutral stance labels.
_IN_FAVOR, _AGAINST, _NEUTRAL = "in_favor", "against", "neutral"


# --------------------------------------------------------------------------
# Small result object (shaped like reaction_split_review.ArtifactRun)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArtifactRun:
    """Where one run wrote its artifacts, plus the summary it computed.

    Attributes:
        output_dir: The newly created, timestamped output directory.
        files: Artifact role (e.g. "snapshot", "summary") to its path.
        summary: The summary mapping that was written as JSON.
    """

    output_dir: Path
    files: dict[str, Path]
    summary: dict


def _created_utc(now: datetime | None) -> str:
    """Return a creation timestamp in CREATED_UTC_FORMAT.

    Args:
        now: Creation time; a naive datetime is read as local time. Defaults
            to the current UTC time.

    Returns:
        The formatted UTC timestamp string used to name the run directory.
    """
    # Normalising to UTC keeps directory names and recorded timestamps
    # independent of the machine's local timezone.
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime(CREATED_UTC_FORMAT)


# --------------------------------------------------------------------------
# Shared input loading
# --------------------------------------------------------------------------


def pair_label_set(pairs_path: Path) -> pd.DataFrame:
    """Load the primary_AB resolved pair-label set (the gold comment labels).

    This is the single definition of "the gold labels" used by both phases.
    It reuses ``evaluator_v2.select_reference`` so the selection (primary
    cohort, tiers A/B, resolved only) is identical to the rest of the
    pipeline and never reimplemented here.

    Args:
        pairs_path: Path to ``dataset/pairs.parquet`` (or a test fixture of
            the same shape).

    Returns:
        One row per scoring pair with columns ``pair_id``,
        ``contribution_id`` (as str), ``comment_id``, ``key_message``,
        ``comment_text`` and ``reference_label`` (one of in_favor/against/
        neutral).
    """
    # Load the full audited pairs table.
    reference = pd.read_parquet(pairs_path)
    # Select the primary cohort over tiers A and B exactly as scoring does;
    # this already drops the 91 NaN-tier rows and the 7 secondary rows.
    selected = evaluator_v2.select_reference(reference, role="primary", tiers=("A", "B"))
    # Keep only rows whose reference label is actually resolved: those carry a
    # real in_favor/against/neutral label; unresolved/quarantined rows carry
    # None and must not enter the favor share.
    selected = selected[selected.resolution_status == "resolved"].copy()
    # Contribution ids are int64 in the parquet; stringify once so every join
    # (votes, contributions, comment_id) uses one consistent key type.
    selected["contribution_id"] = selected.contribution_id.astype(str)
    # Derive the deterministic per-comment id from (contribution_id, text),
    # reusing the pipeline's own helper so our IDs match everywhere.
    selected["comment_id"] = [
        comment_id_for(cid, text)
        for cid, text in zip(selected.contribution_id, selected.comment_text)
    ]
    # Return only the columns this module needs, with a fresh index.
    return selected[
        ["pair_id", "contribution_id", "comment_id", "key_message", "comment_text", "reference_label"]
    ].reset_index(drop=True)


# --------------------------------------------------------------------------
# Phase 1: fetch (the only networked step)
# --------------------------------------------------------------------------


def gold_contribution_projects(pairs_path: Path, contributions_path: Path) -> pd.DataFrame:
    """Map each gold contribution to its DIPAS project, flagging problems.

    The vote endpoint is per project, so each gold contribution must first be
    located in a project. The mapping is read from the local contributions
    table. Contributions that map to no project (unmapped) or to more than one
    project (collision) are flagged rather than crashing the run; only the
    cleanly-mapped ones become fetch targets.

    Args:
        pairs_path: Path to the pairs parquet (gold contributions come from
            its primary_AB resolved set).
        contributions_path: Path to the contributions parquet holding the
            ``project_id`` / ``contribution_id`` mapping.

    Returns:
        One row per gold contribution with columns ``contribution_id``,
        ``project_id`` (the sole project, else None) and
        ``n_project_candidates`` (0 = unmapped, 1 = clean, >1 = collision).
    """
    # The gold contributions are the distinct contribution ids in the label
    # set; this does not depend on any stance outcome.
    gold_ids = sorted(set(pair_label_set(pairs_path).contribution_id))
    # Load the local project<->contribution mapping and stringify both keys so
    # they match the stringified gold ids.
    contributions = pd.read_parquet(contributions_path, columns=["project_id", "contribution_id"])
    contributions = contributions.astype({"project_id": str, "contribution_id": str})
    # For each contribution id, collect the distinct projects it appears in.
    by_contribution = contributions.groupby("contribution_id").project_id.agg(lambda s: sorted(set(s)))
    rows = []
    for contribution_id in gold_ids:
        # projects is the list of candidate project ids for this contribution.
        projects = by_contribution.get(contribution_id, [])
        rows.append({
            "contribution_id": contribution_id,
            # Only a unique candidate becomes a usable target; 0 or >1 leaves
            # project_id None so the caller excludes and reports it.
            "project_id": projects[0] if len(projects) == 1 else None,
            "n_project_candidates": len(projects),
        })
    return pd.DataFrame(rows, columns=["contribution_id", "project_id", "n_project_candidates"])


def fetch_votes(targets: pd.DataFrame, *, timeout: int = 30,
                get: Callable = requests.get) -> tuple[pd.DataFrame, dict]:
    """Fetch current vote counts for the target contributions, per project.

    One HTTP GET is made per distinct project. Each project's contributions
    feature collection is parsed, and only the features whose id is a gold
    contribution of that project are kept (the stance labels never select the
    votes, but a project also contains non-gold contributions we ignore).

    Args:
        targets: Rows with ``contribution_id`` and a non-null ``project_id``
            (the cleanly-mapped gold contributions).
        timeout: Per-request timeout in seconds. Defaults to 30.
        get: The HTTP getter; injectable so tests pass a stub instead of
            ``requests.get``. Defaults to ``requests.get``.

    Returns:
        A tuple ``(votes_df, per_project_status)``. ``votes_df`` has one row
        per matched gold contribution with ``contribution_id``, ``project_id``
        and the ``VOTE_FIELDS`` (pro/contra as nullable Int64).
        ``per_project_status`` maps each project id to a status dict
        (``status``, ``features_returned``, ``gold_matched``).
    """
    # Group the wanted gold ids by project so each project is fetched once.
    wanted_by_project: dict[str, set[str]] = {}
    for row in targets.itertuples():
        wanted_by_project.setdefault(str(row.project_id), set()).add(str(row.contribution_id))

    collected: list[dict] = []
    per_project_status: dict[str, dict] = {}
    for project_id, wanted in wanted_by_project.items():
        # Percent-encode the project id so it cannot break out of the URL path
        # or inject extra components (security: the host is fixed, the id is
        # the only variable part).
        safe_id = urllib.parse.quote(project_id, safe="")
        url = CONTRIBUTIONS_API_TEMPLATE.format(project_id=safe_id)
        try:
            # Fetch and require a 2xx status; JSON decode errors are NOT caught
            # here (a malformed body signals an API-contract change that must
            # surface loudly, exactly as create_dataset does).
            response = get(url, timeout=timeout)
            response.raise_for_status()
            features = response.json().get("features", [])
        except requests.RequestException as error:
            # A per-project network/HTTP failure is recorded and skipped so one
            # bad project does not abort the whole fetch.
            per_project_status[project_id] = {
                "status": f"error: {error}", "features_returned": 0, "gold_matched": 0,
            }
            continue
        matched = 0
        for feature in features:
            props = feature.get("properties", {})
            # Only keep features whose id is a gold contribution of this
            # project; everything else in the project is irrelevant here.
            contribution_id = str(props.get("id"))
            if contribution_id not in wanted:
                continue
            matched += 1
            collected.append({
                "contribution_id": contribution_id,
                "project_id": project_id,
                "votingPro": props.get("votingPro"),
                "votingContra": props.get("votingContra"),
                "commentsNumber": props.get("commentsNumber"),
                "dateCreated": props.get("dateCreated"),
            })
        # Record how many features the project returned and how many were gold.
        per_project_status[project_id] = {
            "status": "ok", "features_returned": len(features), "gold_matched": matched,
        }

    # Assemble the snapshot frame with a fixed column order even when empty.
    columns = ["contribution_id", "project_id", "votingPro", "votingContra", "commentsNumber", "dateCreated"]
    votes_df = pd.DataFrame(collected, columns=columns)
    # Coerce the two vote counts to nullable integers: unparseable or missing
    # values become <NA> rather than raising, so a single odd field cannot
    # fail the fetch.
    for field in ("votingPro", "votingContra"):
        votes_df[field] = pd.to_numeric(votes_df[field], errors="coerce").astype("Int64")
    return votes_df, per_project_status


def run_vote_fetch(*, pairs_path: Path = DEFAULT_PAIRS_PATH,
                   contributions_path: Path = DEFAULT_CONTRIBUTIONS_PATH,
                   output_dir: Path | None = None, now: datetime | None = None,
                   timeout: int = 30, get: Callable = requests.get) -> ArtifactRun:
    """Fetch the vote snapshot and write it to a fresh timestamped directory.

    Args:
        pairs_path: Pairs parquet with the gold contributions.
        contributions_path: Contributions parquet with the project mapping.
        output_dir: Directory to create (its parent must exist). Defaults to
            ``EXPERIMENTS_DIR / "vote_ratio_votes_<created_utc>"``.
        now: Creation time (for tests). Defaults to the current UTC time.
        timeout: Per-request timeout in seconds. Defaults to 30.
        get: HTTP getter; injectable for tests. Defaults to ``requests.get``.

    Returns:
        The output directory, the written files and the fetch summary.

    Raises:
        FileExistsError: If the output directory already exists.
    """
    created_utc = _created_utc(now)
    # Resolve which gold contributions map cleanly to a single project.
    mapping = gold_contribution_projects(pairs_path, contributions_path)
    unmapped = mapping[mapping.n_project_candidates == 0]
    collisions = mapping[mapping.n_project_candidates > 1]
    targets = mapping[mapping.n_project_candidates == 1]
    # Fetch the votes for the clean targets (the single networked step).
    votes_df, per_project_status = fetch_votes(targets, timeout=timeout, get=get)
    # A contribution was "fetched" iff a matching feature was found.
    fetched_ids = set(votes_df.contribution_id)
    missing_from_api = sorted(set(targets.contribution_id) - fetched_ids)

    # Build the provenance summary. network_calls counts one GET per project.
    summary = {
        "schema_version": FETCH_SCHEMA_VERSION,
        "created_utc": created_utc,
        "network_calls": int(len(per_project_status)),
        "api_template": CONTRIBUTIONS_API_TEMPLATE,
        "module_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {
            "pairs": {"path": str(pairs_path), "sha256": sha256_file(Path(pairs_path))},
            "contributions": {"path": str(contributions_path),
                              "sha256": sha256_file(Path(contributions_path))},
        },
        "requested_contributions": int(len(mapping)),
        "contributions_fetched": int(len(fetched_ids)),
        "contributions_missing_from_api": {"count": len(missing_from_api), "ids": missing_from_api},
        "contributions_without_project": {"count": int(len(unmapped)),
                                          "ids": sorted(unmapped.contribution_id)},
        "contributions_with_project_collision": {"count": int(len(collisions)),
                                                 "ids": sorted(collisions.contribution_id)},
        "per_project": per_project_status,
        "snapshot_file": SNAPSHOT_FILE,
        "note": "Only the gold contributions that map to exactly one project are fetched.",
    }

    # Create the run directory (never overwrite) and write the artifacts, with
    # artifact_hashes.json written LAST as the completion marker.
    output_dir = Path(output_dir) if output_dir else EXPERIMENTS_DIR / f"{FETCH_DIR_PREFIX}{created_utc}"
    output_dir.mkdir(parents=False, exist_ok=False)
    files = {"snapshot": output_dir / SNAPSHOT_FILE, "summary": output_dir / FETCH_SUMMARY_FILE}
    votes_df.to_parquet(files["snapshot"], index=False)
    write_json(files["summary"], summary)
    files["hashes"] = write_artifact_hashes(output_dir, created_utc, [files["snapshot"], files["summary"]])
    return ArtifactRun(output_dir, files, summary)


# --------------------------------------------------------------------------
# Phase 2: analyse (offline)
# --------------------------------------------------------------------------


def load_snapshot(snapshot_dir: Path) -> tuple[pd.DataFrame, str]:
    """Load a vote snapshot, verifying it against its run's artifact hashes.

    Args:
        snapshot_dir: A completed ``vote-ratio-fetch`` directory (holding
            SNAPSHOT_FILE and HASHES_FILE).

    Returns:
        A tuple of the votes frame (contribution_id as str) and the file's
        SHA-256.

    Raises:
        FileNotFoundError: If the snapshot or hashes file is missing.
        ValueError: If the hashes file does not record a completed snapshot.
        RuntimeError: If the snapshot's hash differs from the recorded one.
    """

    snapshot_dir = Path(snapshot_dir)
    path, hashes_path = snapshot_dir / SNAPSHOT_FILE, snapshot_dir / HASHES_FILE
    # The hashes file is the completion marker; without a complete record for
    # the snapshot the directory is treated as an incomplete run.
    recorded = json.loads(hashes_path.read_text(encoding="utf-8"))
    expected = recorded.get("files", {}).get(SNAPSHOT_FILE)
    if not recorded.get("complete") or expected is None:
        raise ValueError(f"{hashes_path} does not record a completed {SNAPSHOT_FILE}")
    # Re-hash the file and refuse to proceed if it changed since it was frozen.
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{path} changed since it was written (sha256 {actual}, recorded {expected})")
    votes = pd.read_parquet(path)
    # Stringify the contribution id so it joins to the label set's key type.
    votes["contribution_id"] = votes.contribution_id.astype(str)
    return votes.reset_index(drop=True), actual


def generic_approval_pair_ids(pairs_path: Path, eval_db_path: Path, *,
                              matcher: str = "strict") -> set[str]:
    """Return the gold PAIR ids whose annotator note flags generic approval.

    The override in sensitivity (b) is applied per pair, not per comment: a
    generic-approval note on one pair must re-label only that pair, leaving
    the comment's other (substantive) pairs untouched. This function therefore
    returns flagged *pair* ids, restricted to the gold pair-label set (pairs
    not in the gold set cannot be overridden and are irrelevant here).

    The evaluation database is opened read-only. Each row's free-text note is
    matched with the strict or broad matcher; the row's pair id is recomputed
    exactly as the dataset builder does, so the flagged ids join to the pairs
    table.

    Args:
        pairs_path: Pairs parquet (used to restrict the flagged pair ids to
            the gold label set).
        eval_db_path: The SQLite annotation database.
        matcher: "strict" (drives sensitivity b) or "broad" (reported for
            context). Defaults to "strict".

    Returns:
        The set of flagged gold pair ids (possibly empty).

    Raises:
        ValueError: If ``matcher`` is not "strict" or "broad".
        FileNotFoundError: If the evaluation database file does not exist.
    """
    if matcher not in ("strict", "broad"):
        raise ValueError(f"Unknown matcher {matcher!r}")
    eval_db_path = Path(eval_db_path)
    # Fail loudly (never silently skip sensitivity b) when the DB is absent.
    if not eval_db_path.exists():
        raise FileNotFoundError(
            f"Evaluation database {eval_db_path} is required for the generic-approval "
            "sensitivity analysis but was not found"
        )
    # A file: URI with mode=ro makes SQLite refuse every write and never
    # creates an empty database when the path is wrong.
    uri = f"{eval_db_path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        frame = pd.read_sql_query(
            "select id, contribution_id, key_message, comment_text, verdict, "
            "comment, evaluator, corrected_label from stance_evaluations order by id",
            conn,
        )
    finally:
        conn.close()
    # Recompute the pair id exactly as the dataset builder does (contribution
    # id stringified first), so the flagged pair ids join to the pairs table.
    frame["contribution_id"] = frame.contribution_id.astype(str)
    frame["pair_id"] = [
        dataset_builder.compute_pair_id(r.contribution_id, r.key_message, r.comment_text)
        for r in frame.itertuples()
    ]
    # Flag each note with the requested matcher.
    flagged_mask = frame.comment.map(lambda note: _note_is_generic_approval(note, matcher))
    flagged_pair_ids = set(frame.loc[flagged_mask, "pair_id"])
    # Keep only flagged pairs that actually appear in the gold label set.
    gold_pair_ids = set(pair_label_set(pairs_path).pair_id)
    return flagged_pair_ids & gold_pair_ids


def _note_is_generic_approval(note: object, matcher: str) -> bool:
    """Return whether one annotator note matches the generic-approval matcher.

    Args:
        note: The free-text note; anything not a string is an empty note.
        matcher: "strict" or "broad".

    Returns:
        True if the note is flagged as generic approval.
    """
    # Non-string notes (None/NaN) are treated as empty and never match.
    text = note if isinstance(note, str) else ""
    if matcher == "strict":
        # Strict: one regex requiring "generell/allgemein ... zustimmend".
        return bool(GA_NOTE_STRICT.search(text))
    # Broad: both "zustimm" and ("generell" or "allgemein") anywhere.
    return all(pattern.search(text) for pattern in GA_NOTE_BROAD)


def _apply_pair_override(pair_labels: pd.DataFrame, approval_as: str,
                         generic_approval_pair_ids: set[str] | None) -> pd.DataFrame:
    """Re-label each flagged pair's ``reference_label`` (the sensitivity-b override).

    The override is per pair, not per comment: only the generic-approval pairs
    change, so a comment keeps the stance of its substantive pairs. For
    example, a comment with a flagged in_favor pair and a substantive against
    pair becomes, after overriding the flagged pair to neutral, an "against"
    comment -- never a wholesale "neutral".

    Args:
        pair_labels: Rows from :func:`pair_label_set`.
        approval_as: "as_labeled" (no change), "neutral" (flagged pairs become
            neutral) or "in_favor" (flagged pairs become in_favor).
        generic_approval_pair_ids: The flagged pair ids; ignored when
            ``approval_as`` is "as_labeled".

    Returns:
        ``pair_labels`` unchanged for "as_labeled", otherwise a copy with the
        flagged pairs' ``reference_label`` replaced.

    Raises:
        ValueError: If ``approval_as`` is unknown.
    """
    if approval_as not in ("as_labeled", "neutral", "in_favor"):
        raise ValueError(f"Unknown approval_as {approval_as!r}")
    flagged = generic_approval_pair_ids or set()
    # No override requested, or nothing flagged: return the input untouched.
    if approval_as == "as_labeled" or not flagged:
        return pair_labels
    # Replace only the flagged pairs' labels; every other pair is left as-is.
    overridden = pair_labels.copy()
    mask = overridden.pair_id.isin(flagged)
    overridden.loc[mask, "reference_label"] = _NEUTRAL if approval_as == _NEUTRAL else _IN_FAVOR
    return overridden


def _category_from_labels(labels: set) -> str:
    """Collapse the set of directional labels of a unit into one category.

    Args:
        labels: The reference labels present among a comment's pairs.

    Returns:
        "mixed" (both directions), "in_favor", "against" or "neutral".
    """
    has_favor, has_against = _IN_FAVOR in labels, _AGAINST in labels
    if has_favor and has_against:
        return "mixed"              # both directions present -> mixed
    if has_favor:
        return _IN_FAVOR
    if has_against:
        return _AGAINST
    return _NEUTRAL                 # only neutral pairs (or none) -> neutral


def rollup_comments(pair_labels: pd.DataFrame, *, approval_as: str = "as_labeled",
                    generic_approval_pair_ids: set[str] | None = None) -> pd.DataFrame:
    """Roll each comment's pair labels up into a single comment-level category.

    Any generic-approval override is applied per pair first (see
    :func:`_apply_pair_override`), then each comment's category is derived from
    its (possibly overridden) pair labels.

    Args:
        pair_labels: Rows from :func:`pair_label_set`, restricted to the
            contributions of interest.
        approval_as: "as_labeled" (no override, the primary analysis),
            "neutral" or "in_favor". Defaults to "as_labeled".
        generic_approval_pair_ids: The flagged pair ids; used when
            ``approval_as`` is not "as_labeled".

    Returns:
        One row per comment with ``comment_id``, ``contribution_id`` and
        ``category`` (one of in_favor/against/mixed/neutral).

    Raises:
        ValueError: If ``approval_as`` is unknown.
    """
    # Apply the per-pair override up front; the roll-up below is then identical
    # for the primary and every (b) variant.
    labels = _apply_pair_override(pair_labels, approval_as, generic_approval_pair_ids)
    rows = []
    # Group all pairs of one comment together; the rows of a comment share a
    # contribution id.
    for comment_id, group in labels.groupby("comment_id", sort=False):
        rows.append({
            "comment_id": comment_id,
            "contribution_id": group.contribution_id.iloc[0],
            "category": _category_from_labels(set(group.reference_label)),
        })
    return pd.DataFrame(rows, columns=["comment_id", "contribution_id", "category"])


def pair_categories(pair_labels: pd.DataFrame, *, approval_as: str = "as_labeled",
                    generic_approval_pair_ids: set[str] | None = None) -> pd.DataFrame:
    """Return per-pair categories for the pair-level sensitivity analysis (a).

    The same per-pair override is available here as in the roll-up, so the
    pair-level counting uses the identical override mechanism.

    Args:
        pair_labels: Rows from :func:`pair_label_set`.
        approval_as: "as_labeled" (no override, the plan's sensitivity a),
            "neutral" or "in_favor". Defaults to "as_labeled".
        generic_approval_pair_ids: The flagged pair ids; used when
            ``approval_as`` is not "as_labeled".

    Returns:
        One row per pair with ``contribution_id`` and ``category`` (the pair's
        own, possibly overridden, reference label: in_favor/against/neutral).
    """
    # Apply the per-pair override, then take each pair's label as its category
    # (a pair carries exactly one label, so there is no "mixed").
    labels = _apply_pair_override(pair_labels, approval_as, generic_approval_pair_ids)
    out = labels[["contribution_id", "reference_label"]].rename(columns={"reference_label": "category"})
    return out.reset_index(drop=True)


def favor_share(categories, *, mixed_mode: str = "exclude") -> dict:
    """Compute the favor share and category counts from comment/pair categories.

    Args:
        categories: Iterable of category strings (in_favor/against/neutral/
            mixed).
        mixed_mode: "exclude" (mixed comments drop out, the primary rule) or
            "half" (a mixed comment contributes 0.5 to each side). Defaults to
            "exclude".

    Returns:
        A dict with integer counts ``favor``, ``against``, ``neutral``,
        ``mixed``; ``favor_share`` (percent of the directional total that is
        in_favor, or None when the denominator is zero); and ``neutral_share``
        (percent of all categories that are neutral, or None when empty).

    Raises:
        ValueError: If ``mixed_mode`` is unknown.
    """
    if mixed_mode not in ("exclude", "half"):
        raise ValueError(f"Unknown mixed_mode {mixed_mode!r}")
    cats = list(categories)
    favor = cats.count(_IN_FAVOR)
    against = cats.count(_AGAINST)
    neutral = cats.count(_NEUTRAL)
    mixed = cats.count("mixed")
    # In "half" mode a mixed comment counts as half in_favor and half against;
    # in "exclude" mode it is left out of the directional totals entirely.
    if mixed_mode == "half":
        favor_num = favor + 0.5 * mixed
        denominator = favor + against + mixed
    else:
        favor_num = favor
        denominator = favor + against
    total = favor + against + neutral + mixed
    return {
        "favor": favor, "against": against, "neutral": neutral, "mixed": mixed,
        "favor_share": None if denominator == 0 else 100.0 * favor_num / denominator,
        "neutral_share": None if total == 0 else 100.0 * neutral / total,
    }


def vote_share(votes: pd.DataFrame, contribution_ids: Sequence[str]) -> dict:
    """Compute the pooled vote pro share over a fixed set of contributions.

    Args:
        votes: The vote snapshot.
        contribution_ids: The fixed vote-side contribution set (independent of
            any stance outcome).

    Returns:
        A dict with pooled ``vote_pro`` and ``vote_contra`` integers and
        ``vote_share`` (percent of pooled votes that are pro, or None when no
        votes were cast).
    """
    # Restrict to the fixed set, then pool the two counts (missing counts as 0).
    subset = votes[votes.contribution_id.isin(set(contribution_ids))]
    pro = int(subset.votingPro.fillna(0).sum())
    contra = int(subset.votingContra.fillna(0).sum())
    denominator = pro + contra
    return {"vote_pro": pro, "vote_contra": contra,
            "vote_share": None if denominator == 0 else 100.0 * pro / denominator}


def _comment_numerator_denominator(category: str, mixed_mode: str) -> tuple[float, float]:
    """Return the (favor numerator, directional denominator) for one category.

    Args:
        category: in_favor/against/neutral/mixed.
        mixed_mode: "exclude" or "half".

    Returns:
        A pair of floats: how much this unit adds to the in_favor numerator
        and to the directional denominator.
    """
    if category == _IN_FAVOR:
        return 1.0, 1.0            # a pure in_favor unit: 1 of 1
    if category == _AGAINST:
        return 0.0, 1.0            # a pure against unit: 0 of 1
    if category == "mixed" and mixed_mode == "half":
        return 0.5, 1.0            # mixed split half/half when not excluded
    # neutral, or mixed under "exclude": contributes nothing to either total.
    return 0.0, 0.0


def contribution_aggregates(categories: pd.DataFrame, votes: pd.DataFrame,
                            fixed_ids: Sequence[str], *, mixed_mode: str) -> pd.DataFrame:
    """Aggregate comment/pair categories and votes to one row per contribution.

    This is the single frame both the point estimate and the clustered
    bootstrap are computed from, so the two always agree.

    Args:
        categories: A frame with ``contribution_id`` and ``category`` (one row
            per comment for the comment-level variants, or per pair for the
            pair-level variant).
        votes: The vote snapshot.
        fixed_ids: The fixed vote-side contribution set; the output is indexed
            over exactly these contributions so the vote side never depends on
            the stance categories.
        mixed_mode: "exclude" or "half" (how mixed units count).

    Returns:
        A frame with one row per fixed contribution and float columns
        ``favor_num``, ``favor_den``, ``vote_pro`` and ``vote_contra``.
    """
    fixed_ids = list(fixed_ids)
    # Translate every category into its (numerator, denominator) contribution.
    nd = categories.category.map(lambda c: _comment_numerator_denominator(c, mixed_mode))
    comment_side = pd.DataFrame({
        "contribution_id": categories.contribution_id.to_numpy(),
        "favor_num": [x[0] for x in nd],
        "favor_den": [x[1] for x in nd],
    })
    # Sum the comment side per contribution, then reindex onto the fixed set so
    # contributions with no comments still appear (with zero comment side).
    comment_grouped = comment_side.groupby("contribution_id")[["favor_num", "favor_den"]].sum()
    comment_grouped = comment_grouped.reindex(fixed_ids, fill_value=0.0)
    # Pool the votes per contribution over the fixed set (missing votes -> 0).
    vote_grouped = (votes[votes.contribution_id.isin(set(fixed_ids))]
                    .groupby("contribution_id")[["votingPro", "votingContra"]].sum())
    vote_grouped = vote_grouped.reindex(fixed_ids, fill_value=0)
    # Assemble the aligned per-contribution frame as plain floats.
    return pd.DataFrame({
        "contribution_id": fixed_ids,
        "favor_num": comment_grouped.favor_num.to_numpy(dtype=float),
        "favor_den": comment_grouped.favor_den.to_numpy(dtype=float),
        "vote_pro": vote_grouped.votingPro.fillna(0).to_numpy(dtype=float),
        "vote_contra": vote_grouped.votingContra.fillna(0).to_numpy(dtype=float),
    })


def _pooled_difference(cols: Mapping[str, np.ndarray]) -> float | None:
    """Pooled favor share minus pooled vote share, in percentage points.

    Args:
        cols: Arrays ``favor_num``, ``favor_den``, ``vote_pro``,
            ``vote_contra`` (one entry per contribution in the sample).

    Returns:
        The difference in percentage points, or None if either pooled
        denominator is zero (an undefined resample, to be dropped).
    """
    favor_den = cols["favor_den"].sum()
    vote_den = cols["vote_pro"].sum() + cols["vote_contra"].sum()
    if favor_den == 0 or vote_den == 0:
        return None
    favor = 100.0 * cols["favor_num"].sum() / favor_den
    vote = 100.0 * cols["vote_pro"].sum() / vote_den
    return favor - vote


def _unweighted_difference(cols: Mapping[str, np.ndarray]) -> float | None:
    """Unweighted mean favor share minus mean vote share (sensitivity c).

    Each contribution contributes equally: per-contribution shares are
    averaged rather than pooled, so large contributions no longer dominate.

    Args:
        cols: Arrays ``favor_num``, ``favor_den``, ``vote_pro``,
            ``vote_contra`` (one entry per contribution in the sample).

    Returns:
        The difference of means in percentage points, or None if no
        contribution has a defined favor share or none has any votes.
    """
    favor_den = cols["favor_den"]
    vote_den = cols["vote_pro"] + cols["vote_contra"]
    favor_mask = favor_den > 0           # contributions with a directional comment
    vote_mask = vote_den > 0             # contributions with at least one vote
    if not favor_mask.any() or not vote_mask.any():
        return None
    favor = 100.0 * (cols["favor_num"][favor_mask] / favor_den[favor_mask]).mean()
    vote = 100.0 * (cols["vote_pro"][vote_mask] / vote_den[vote_mask]).mean()
    return favor - vote


def per_contribution_shares(aggregates: pd.DataFrame) -> pd.DataFrame:
    """Per-contribution favor and vote shares, for the unweighted analysis (c).

    Args:
        aggregates: A frame from :func:`contribution_aggregates`.

    Returns:
        A frame with ``contribution_id`` and the per-contribution
        ``favor_share`` / ``vote_share`` in percent (NaN where undefined).
    """
    vote_den = aggregates.vote_pro + aggregates.vote_contra
    # Divide only where the denominator is positive; elsewhere leave NaN. The
    # inner ``.where(den > 0)`` turns every zero denominator into NaN *before*
    # the division runs, so the division never actually divides by zero and
    # numpy emits no "divide by zero" / "invalid value" RuntimeWarning; the
    # outer ``np.where`` then selects the real share for positive denominators
    # and NaN for the rest. Both guards are needed: the inner one silences the
    # warning, the outer one decides the result.
    return pd.DataFrame({
        "contribution_id": aggregates.contribution_id,
        "favor_share": np.where(aggregates.favor_den > 0,
                                100.0 * aggregates.favor_num / aggregates.favor_den.where(aggregates.favor_den > 0),
                                np.nan),
        "vote_share": np.where(vote_den > 0,
                               100.0 * aggregates.vote_pro / vote_den.where(vote_den > 0),
                               np.nan),
    })


def bootstrap_difference(contribution_frame: pd.DataFrame, *, n: int, seed: int,
                         estimator: Callable[[Mapping[str, np.ndarray]], float | None]
                         ) -> tuple[np.ndarray, int]:
    """Clustered bootstrap of the share difference, resampling contributions.

    Contributions (not comments or votes) are resampled with replacement, so
    the dependence between a contribution's comments and its votes is
    preserved, and both shares are recomputed from the same resampled set.

    Args:
        contribution_frame: One row per contribution with ``favor_num``,
            ``favor_den``, ``vote_pro`` and ``vote_contra``.
        n: Number of bootstrap resamples.
        seed: Seed for ``numpy.random.default_rng`` (reproducibility).
        estimator: Maps the resampled column arrays to a difference in
            percentage points, or None for an undefined resample.

    Returns:
        A tuple of the array of defined differences and the number of
        resamples dropped for a zero denominator.
    """
    # Extract the needed columns once as float arrays for fast indexing.
    cols = {c: contribution_frame[c].to_numpy(dtype=float)
            for c in ("favor_num", "favor_den", "vote_pro", "vote_contra")}
    m = len(contribution_frame)
    rng = np.random.default_rng(seed)
    diffs: list[float] = []
    dropped = 0
    # Each iteration draws a fresh cluster resample of the contributions.
    for _ in range(n):
        if m == 0:
            dropped += 1
            continue
        index = rng.integers(0, m, size=m)
        sample = {c: arr[index] for c, arr in cols.items()}
        value = estimator(sample)
        # A zero-denominator resample is undefined and dropped (and counted).
        if value is None:
            dropped += 1
        else:
            diffs.append(value)
    return np.asarray(diffs, dtype=float), dropped


def tost_equivalence(diffs: np.ndarray, *, margin: float, level: float) -> dict:
    """TOST equivalence verdict from a bootstrap distribution of differences.

    Args:
        diffs: The bootstrap differences in percentage points.
        margin: The equivalence half-width in percentage points.
        level: Confidence level (e.g. 0.90).

    Returns:
        A dict with the percentile-interval ``low`` and ``high`` and
        ``equivalent`` (True iff ``-margin < low and high < margin``). The
        bounds are None when there are no defined resamples.
    """
    # With no defined resamples the interval is undefined and not equivalent.
    if diffs.size == 0:
        return {"low": None, "high": None, "equivalent": False}
    # Two-sided percentile interval at the requested level.
    alpha = 1.0 - level
    low = float(np.percentile(diffs, 100.0 * alpha / 2.0))
    high = float(np.percentile(diffs, 100.0 * (1.0 - alpha / 2.0)))
    # Equivalent only if the whole interval sits strictly inside the band.
    return {"low": low, "high": high, "equivalent": bool(-margin < low and high < margin)}


def _variant_block(*, categories: pd.DataFrame, votes: pd.DataFrame, fixed_ids: Sequence[str],
                   mixed_mode: str, mode: str, n_bootstrap: int, seed: int, margin: float) -> dict:
    """Compute one result block (point estimate + both CIs) for a variant.

    Args:
        categories: Per-comment or per-pair categories for this variant.
        votes: The vote snapshot.
        fixed_ids: The fixed vote-side contribution set.
        mixed_mode: "exclude" or "half".
        mode: "pooled" or "unweighted" (how the shares are aggregated).
        n_bootstrap: Number of bootstrap resamples.
        seed: Bootstrap seed.
        margin: Equivalence half-width in percentage points.

    Returns:
        A block dict with the counts, the point shares/difference and the 90%
        and 95% CI verdicts.
    """
    # Build the single per-contribution frame this variant is computed from.
    aggregates = contribution_aggregates(categories, votes, fixed_ids, mixed_mode=mixed_mode)
    estimator = _pooled_difference if mode == "pooled" else _unweighted_difference
    cols = {c: aggregates[c].to_numpy(dtype=float)
            for c in ("favor_num", "favor_den", "vote_pro", "vote_contra")}
    # Point favor/vote shares and their difference, from the full (unresampled)
    # frame using the same estimator as the bootstrap (so they agree).
    if mode == "pooled":
        favor_point, vote_point = _pooled_shares(cols)
    else:
        shares = per_contribution_shares(aggregates)
        favor_point = None if shares.favor_share.notna().sum() == 0 else float(np.nanmean(shares.favor_share))
        vote_point = None if shares.vote_share.notna().sum() == 0 else float(np.nanmean(shares.vote_share))
    difference = estimator(cols)
    # Descriptive category counts for this variant's unit.
    counts = favor_share(categories.category, mixed_mode=mixed_mode)
    # Clustered bootstrap once, then read both CI levels off the same draws.
    diffs, dropped = bootstrap_difference(aggregates, n=n_bootstrap, seed=seed, estimator=estimator)
    ci = {level: tost_equivalence(diffs, margin=margin, level=level) for level in CI_LEVELS}
    votes_dict = vote_share(votes, fixed_ids)
    return {
        "favor": counts["favor"], "against": counts["against"],
        "neutral": counts["neutral"], "mixed": counts["mixed"],
        "favor_share": favor_point, "neutral_share": counts["neutral_share"],
        "vote_pro": votes_dict["vote_pro"], "vote_contra": votes_dict["vote_contra"],
        "vote_share": vote_point,
        "difference_pp": difference,
        "resamples_dropped_zero_denominator": dropped,
        "ci90": ci[0.90], "ci95": ci[0.95],
    }


def _pooled_shares(cols: Mapping[str, np.ndarray]) -> tuple[float | None, float | None]:
    """Pooled favor share and vote share as a pair (None where undefined).

    Args:
        cols: Arrays ``favor_num``, ``favor_den``, ``vote_pro``,
            ``vote_contra``.

    Returns:
        A tuple ``(favor_share, vote_share)`` in percent.
    """
    favor_den = cols["favor_den"].sum()
    vote_den = cols["vote_pro"].sum() + cols["vote_contra"].sum()
    favor = None if favor_den == 0 else 100.0 * cols["favor_num"].sum() / favor_den
    vote = None if vote_den == 0 else 100.0 * cols["vote_pro"].sum() / vote_den
    return favor, vote


def run_vote_analysis(*, snapshot_dir: Path, pairs_path: Path = DEFAULT_PAIRS_PATH,
                      contributions_path: Path = DEFAULT_CONTRIBUTIONS_PATH,
                      eval_db_path: Path = DEFAULT_EVAL_DB, output_dir: Path | None = None,
                      now: datetime | None = None, n_bootstrap: int = N_BOOTSTRAP,
                      seed: int = BOOTSTRAP_SEED) -> ArtifactRun:
    """Run the full offline equivalence analysis and write its artifacts.

    Args:
        snapshot_dir: A completed ``vote-ratio-fetch`` directory.
        pairs_path: Pairs parquet with the gold labels.
        contributions_path: Contributions parquet (recorded for provenance).
        eval_db_path: The evaluation SQLite database (sensitivity b).
        output_dir: Directory to create (its parent must exist). Defaults to
            ``EXPERIMENTS_DIR / "vote_ratio_analysis_<created_utc>"``.
        now: Creation time (for tests). Defaults to the current UTC time.
        n_bootstrap: Number of bootstrap resamples. Defaults to N_BOOTSTRAP.
        seed: Bootstrap seed. Defaults to BOOTSTRAP_SEED.

    Returns:
        The output directory, the written files and the analysis summary.

    Raises:
        FileExistsError: If the output directory already exists.
        FileNotFoundError: If the evaluation database is missing.
        RuntimeError: If the snapshot changed since it was written.
    """

    created_utc = _created_utc(now)
    # --- Load the inputs ---------------------------------------------------
    votes, snapshot_sha = load_snapshot(snapshot_dir)
    labels = pair_label_set(pairs_path)
    # The vote snapshot's creation time and note come from the fetch summary
    # next to it (if present); used only for the snapshot-timing statement.
    snapshot_created, snapshot_note = _read_snapshot_meta(Path(snapshot_dir))

    # --- Build the fixed vote-side contribution set ------------------------
    # Gold contributions are those with >=1 comment in the label set; the fixed
    # set additionally requires presence in the snapshot. This set is identical
    # for the primary analysis and every sensitivity variant, so stance labels
    # can never select which votes are counted.
    gold_ids = set(labels.contribution_id)
    snapshot_ids = set(votes.contribution_id)
    fixed_ids = sorted(gold_ids & snapshot_ids)
    missing_votes = sorted(gold_ids - snapshot_ids)
    # Restrict the label set to the fixed contributions for every comment-side
    # computation.
    fixed_labels = labels[labels.contribution_id.isin(set(fixed_ids))].reset_index(drop=True)

    # --- Generic-approval flags (sensitivity b) ---------------------------
    # The override is per PAIR: strict drives it, broad is reported for
    # context. Each is a set of flagged gold pair ids.
    strict_pair_ids = generic_approval_pair_ids(pairs_path, eval_db_path, matcher="strict")
    broad_pair_ids = generic_approval_pair_ids(pairs_path, eval_db_path, matcher="broad")
    # Restrict the flagged pairs to the fixed set, and also count the comments
    # that have at least one flagged pair within the fixed set (both are
    # reported, with matching names, so neither is mislabelled).
    fixed_pair_ids = set(fixed_labels.pair_id)
    strict_fixed_pairs = strict_pair_ids & fixed_pair_ids
    broad_fixed_pairs = broad_pair_ids & fixed_pair_ids
    strict_fixed_comments = set(fixed_labels.loc[fixed_labels.pair_id.isin(strict_fixed_pairs), "comment_id"])
    broad_fixed_comments = set(fixed_labels.loc[fixed_labels.pair_id.isin(broad_fixed_pairs), "comment_id"])

    # --- Primary and sensitivity categories -------------------------------
    primary_categories = rollup_comments(fixed_labels, approval_as="as_labeled")
    pair_level_categories = pair_categories(fixed_labels)
    # (b) overrides only the flagged pairs (strict), then rolls the comment up.
    approval_neutral_categories = rollup_comments(
        fixed_labels, approval_as="neutral", generic_approval_pair_ids=strict_pair_ids)
    approval_favor_categories = rollup_comments(
        fixed_labels, approval_as="in_favor", generic_approval_pair_ids=strict_pair_ids)

    # --- Compute every result block ---------------------------------------
    common = {"votes": votes, "fixed_ids": fixed_ids, "n_bootstrap": n_bootstrap,
              "seed": seed, "margin": EQUIVALENCE_MARGIN_PP}
    primary = _variant_block(categories=primary_categories, mixed_mode="exclude", mode="pooled", **common)
    sensitivity = {
        # (a) pair as the unit of observation.
        "pair_level": _variant_block(categories=pair_level_categories, mixed_mode="exclude",
                                     mode="pooled", **common),
        # (b) generic approval re-labelled as neutral / as in_favor.
        "approval_as_neutral": _variant_block(categories=approval_neutral_categories, mixed_mode="exclude",
                                              mode="pooled", **common),
        "approval_as_in_favor": _variant_block(categories=approval_favor_categories, mixed_mode="exclude",
                                               mode="pooled", **common),
        # (c) per-contribution unweighted mean.
        "per_contribution_unweighted": _variant_block(categories=primary_categories, mixed_mode="exclude",
                                                       mode="unweighted", **common),
        # (d) mixed comments counted as a half.
        "mixed_half": _variant_block(categories=primary_categories, mixed_mode="half",
                                     mode="pooled", **common),
    }

    # --- Descriptive counts ------------------------------------------------
    pooled_votes = vote_share(votes, fixed_ids)
    # A "zero-vote" contribution is one that IS in the snapshot but received no
    # votes at all (pro + contra == 0, treating missing counts as 0). These are
    # kept in the fixed set (never dropped) and only reported, so stakeholders
    # can see how much of the vote side carries no signal.
    zero_votes = [cid for cid in fixed_ids
                  if int(votes.loc[votes.contribution_id == cid, "votingPro"].fillna(0).sum()
                         + votes.loc[votes.contribution_id == cid, "votingContra"].fillna(0).sum()) == 0]
    counts = {
        "n_comments": int(fixed_labels.comment_id.nunique()),
        "n_pairs": int(len(fixed_labels)),
        "n_contributions": int(len(fixed_ids)),
        "n_contributions_with_votes": int(len(fixed_ids) - len(zero_votes)),
        "n_contributions_missing_votes": {"count": len(missing_votes), "ids": missing_votes},
        "n_contributions_zero_votes": len(zero_votes),
        "total_votingPro": pooled_votes["vote_pro"],
        "total_votingContra": pooled_votes["vote_contra"],
        "mixed_comments": primary["mixed"],
        "neutral_comments": primary["neutral"],
        # Flagged PAIRS within the fixed set (the override unit)...
        "generic_approval_pairs_strict": len(strict_fixed_pairs),
        "generic_approval_pairs_broad": len(broad_fixed_pairs),
        # ...and the COMMENTS that have at least one flagged pair.
        "generic_approval_comments_strict": len(strict_fixed_comments),
        "generic_approval_comments_broad": len(broad_fixed_comments),
    }

    # --- Assemble the summary ---------------------------------------------
    summary = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "created_utc": created_utc,
        "module_sha256": sha256_file(Path(__file__).resolve()),
        "inputs": {
            "snapshot": {"path": str(Path(snapshot_dir) / SNAPSHOT_FILE), "sha256": snapshot_sha},
            "pairs": {"path": str(pairs_path), "sha256": sha256_file(Path(pairs_path))},
            "contributions": {"path": str(contributions_path),
                              "sha256": sha256_file(Path(contributions_path))},
            "eval_db": {"path": str(eval_db_path), "sha256": sha256_file(Path(eval_db_path))},
        },
        "bootstrap": {"n": n_bootstrap, "seed": seed,
                      "resamples_dropped_zero_denominator": primary["resamples_dropped_zero_denominator"]},
        "margin_pp": EQUIVALENCE_MARGIN_PP,
        "interpretation_rules": INTERPRETATION_RULES,
        "vote_side_rule": (
            "The vote side is pooled over a fixed set = every contribution with >=1 comment in the "
            "primary_AB label set and present in the vote snapshot. This set does not depend on the "
            "stance outcome and is identical for the primary analysis and every sensitivity variant, "
            "so the stance labels can never select which votes are counted."
        ),
        "counts": counts,
        "vote_snapshot": {"created_utc": snapshot_created, "note": snapshot_note},
        "representativeness_limit": REPRESENTATIVENESS_LIMIT,
        "primary": primary,
        "sensitivity": sensitivity,
    }

    # --- Write artifacts (hashes last) ------------------------------------
    output_dir = Path(output_dir) if output_dir else EXPERIMENTS_DIR / f"{ANALYSIS_DIR_PREFIX}{created_utc}"
    output_dir.mkdir(parents=False, exist_ok=False)
    files = {"summary": output_dir / ANALYSIS_SUMMARY_FILE, "report": output_dir / REPORT_FILE}
    write_json(files["summary"], summary)
    write_report(files["report"], summary)
    files["hashes"] = write_artifact_hashes(output_dir, created_utc, [files["summary"], files["report"]])
    return ArtifactRun(output_dir, files, summary)


def _read_snapshot_meta(snapshot_dir: Path) -> tuple[str | None, str | None]:
    """Read the vote snapshot's creation time and note from its fetch summary.

    Args:
        snapshot_dir: The fetch directory.

    Returns:
        A tuple ``(created_utc, note)``; both None when the fetch summary is
        absent or unreadable.
    """

    summary_path = snapshot_dir / FETCH_SUMMARY_FILE
    if not summary_path.exists():
        return None, None
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None, None
    return data.get("created_utc"), data.get("note")


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def _fmt(value: object) -> str:
    """Format a share/difference for the report (one decimal, or n/a).

    Args:
        value: A number or None.

    Returns:
        The number to one decimal place, or "n/a" for None.
    """
    return "n/a" if value is None else f"{value:.1f}"


def _verdict(ci: Mapping) -> str:
    """One-line verdict for a CI block.

    Args:
        ci: A CI dict with ``low``, ``high`` and ``equivalent``.

    Returns:
        A human-readable verdict string.
    """
    word = "EQUIVALENT" if ci["equivalent"] else "NOT equivalent"
    return f"{word} (CI [{_fmt(ci['low'])}, {_fmt(ci['high'])}] pp)"


def _block_lines(block: Mapping) -> list[str]:
    """Render one result block (point estimates + both CIs) as report lines.

    Args:
        block: A result block from :func:`_variant_block`.

    Returns:
        A list of Markdown lines.
    """
    return [
        f"- Favor share: {_fmt(block['favor_share'])}% "
        f"(in_favor {block['favor']}, against {block['against']}, "
        f"neutral {block['neutral']}, mixed {block['mixed']})",
        f"- Neutral share: {_fmt(block['neutral_share'])}%",
        f"- Vote pro share: {_fmt(block['vote_share'])}% "
        f"(pro {block['vote_pro']}, contra {block['vote_contra']})",
        f"- Difference (favor - vote): {_fmt(block['difference_pp'])} pp",
        f"- 90% (primary): {_verdict(block['ci90'])}",
        f"- 95%: {_verdict(block['ci95'])}",
    ]


def format_report(summary: Mapping) -> str:
    """Render the analysis summary as an English Markdown report.

    Args:
        summary: The analysis summary mapping.

    Returns:
        The full report as a single Markdown string.
    """
    counts = summary["counts"]
    primary = summary["primary"]
    lines: list[str] = []
    # Title and the null/margin/test description.
    lines.append("# Vote-ratio equivalence test: report")
    lines.append("")
    lines.append(f"Run: {summary['created_utc']}")
    lines.append("")
    lines.append(
        "H0 (non-equivalence) vs H1 (equivalence): the comment favor share and the vote pro share "
        f"are declared equivalent iff the confidence interval for their difference lies strictly "
        f"inside (-{summary['margin_pp']:.0f}, +{summary['margin_pp']:.0f}) percentage points. "
        "The test is a clustered-bootstrap TOST; the 90% interval drives the primary verdict and the "
        "95% interval is reported alongside."
    )
    lines.append("")
    # The two interpretation caveats (both directions).
    lines.append("## Interpretation (read first)")
    lines.append("")
    lines.append(f"> {summary['interpretation_rules']}")
    lines.append("")
    lines.append(f"> {summary['representativeness_limit']}")
    lines.append("")
    # Snapshot timing.
    snap = summary["vote_snapshot"]
    lines.append(f"Vote snapshot taken at: {snap.get('created_utc') or 'unknown'}.")
    lines.append("")
    # Counts table.
    lines.append("## Counts")
    lines.append("")
    lines.append("| Quantity | Value |")
    lines.append("| --- | --- |")
    lines.append(f"| Comments | {counts['n_comments']} |")
    lines.append(f"| Pairs | {counts['n_pairs']} |")
    lines.append(f"| Contributions (fixed set) | {counts['n_contributions']} |")
    lines.append(f"| Contributions with votes | {counts['n_contributions_with_votes']} |")
    lines.append(f"| Contributions with zero votes | {counts['n_contributions_zero_votes']} |")
    lines.append(f"| Contributions missing votes | {counts['n_contributions_missing_votes']['count']} |")
    lines.append(f"| Total thumbs up (pro) | {counts['total_votingPro']} |")
    lines.append(f"| Total thumbs down (contra) | {counts['total_votingContra']} |")
    lines.append(f"| Mixed comments | {counts['mixed_comments']} |")
    lines.append(f"| Neutral comments | {counts['neutral_comments']} |")
    lines.append(f"| Generic-approval pairs (strict) | {counts['generic_approval_pairs_strict']} |")
    lines.append(f"| Generic-approval pairs (broad) | {counts['generic_approval_pairs_broad']} |")
    lines.append(f"| Generic-approval comments (strict) | {counts['generic_approval_comments_strict']} |")
    lines.append(f"| Generic-approval comments (broad) | {counts['generic_approval_comments_broad']} |")
    lines.append("")
    # Primary result.
    lines.append("## Primary result (comment unit, mixed excluded, labels as-is)")
    lines.append("")
    lines.extend(_block_lines(primary))
    lines.append("")
    lines.append(
        f"Bootstrap: {summary['bootstrap']['n']} resamples, seed {summary['bootstrap']['seed']}, "
        f"{summary['bootstrap']['resamples_dropped_zero_denominator']} dropped for a zero denominator."
    )
    lines.append("")
    # Sensitivity analyses.
    titles = {
        "pair_level": "(a) Pair level",
        "approval_as_neutral": "(b) Generic approval -> neutral",
        "approval_as_in_favor": "(b) Generic approval -> in_favor",
        "per_contribution_unweighted": "(c) Per-contribution unweighted mean",
        "mixed_half": "(d) Mixed comments counted as a half",
    }
    lines.append("## Sensitivity analyses")
    lines.append("")
    for key, title in titles.items():
        lines.append(f"### {title}")
        lines.append("")
        lines.extend(_block_lines(summary["sensitivity"][key]))
        lines.append("")
    return "\n".join(lines)


def write_report(path: Path, summary: Mapping) -> None:
    """Write the Markdown report to a new file.

    Args:
        path: Target path; must not exist yet (mode "x").
        summary: The analysis summary mapping.

    Raises:
        FileExistsError: If the path already exists.
    """
    # Exclusive create so a run never overwrites an existing report.
    with open(path, "x", encoding="utf-8") as handle:
        handle.write(format_report(summary))
        handle.write("\n")
