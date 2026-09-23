"""Grouped cross-validation fold assignment for the resolved dataset.

STANCE_PIPELINE_IMPLEMENTATION_BRIEF.md section 6: "Use grouped evaluation:
all key messages and comments belonging to a contribution must stay in one
split. Prevent substantive near-duplicate source material from crossing
folds when detectable. [...] Implement reproducible five-fold grouped outer
evaluation. Tune routing thresholds and any example selection only within
each training partition, using an inner grouped validation split. Save fold
assignments before experiments. Keep the experiment matrix small and fixed
to avoid repeated optimization against outer results."

Folds are computed once and persisted (dataset/folds_v1.json); every later
experiment loads them rather than re-splitting, so results across variants
are comparable and the split can't be silently re-rolled to favor one.

Known limitation: dataset/pairs.parquet (dataset_builder.py's output) does
not carry contribution text -- none of its sources (stance_evaluations, the
six experiment CSVs) include it. The near-duplicate check below is
therefore a best-effort, opt-in step: callers who have contribution text
available separately (e.g. from a whole_process contributions-with-comments
source parquet) can pass it in via `contribution_texts`; without it, folds
still split correctly by contribution_id, just without the extra merge, and
that is recorded explicitly in the resulting FoldAssignment rather than
silently skipped.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

FOLDS_SCHEMA_VERSION = "folds_v1"
DEFAULT_N_SPLITS = 5
DEFAULT_SEED = 42
DEFAULT_INNER_TEST_SIZE = 0.2


def normalize_contribution_text(text: str) -> str:
    """Normalize contribution text for near-duplicate hashing."""
    return " ".join(text.lower().split())


def find_near_duplicate_groups(contribution_texts: dict[str, str] | None) -> dict[str, str]:
    """Map each contribution_id to a dedup-group id via normalized-text hash equality.

    Args:
        contribution_texts: contribution_id -> full contribution text, or
            None/empty when contribution text isn't available (see module
            docstring).

    Returns:
        A dict mapping every contribution_id present in contribution_texts
        to a group id (its own id, or another contribution's id if the two
        normalize to an identical hash). Empty when contribution_texts is
        falsy -- callers must treat that as "no merging performed," not
        "no duplicates found."
    """
    if not contribution_texts:
        return {}
    first_id_for_hash: dict[str, str] = {}
    group_of: dict[str, str] = {}
    for contribution_id, text in contribution_texts.items():
        digest = hashlib.sha256(normalize_contribution_text(text).encode("utf-8")).hexdigest()
        if digest not in first_id_for_hash:
            first_id_for_hash[digest] = contribution_id
        group_of[contribution_id] = first_id_for_hash[digest]
    return group_of


@dataclass
class FoldAssignment:
    """Persisted, reproducible outer-fold split.

    Attributes:
        schema_version: Format version, for forward compatibility.
        seed: Random seed used.
        n_splits: Number of outer folds.
        grouping_key: Always "contribution_id" today.
        near_duplicate_check: "performed" or
            "skipped_no_contribution_text_available" -- see module docstring.
        dup_groups_merged: How many distinct contribution_ids were merged
            into an existing group (0 when the check was skipped).
        folds: One entry per fold: {"train_contribution_ids": [...],
            "test_contribution_ids": [...]}.
    """

    seed: int
    n_splits: int
    grouping_key: str
    near_duplicate_check: str
    dup_groups_merged: int
    folds: list[dict] = field(default_factory=list)
    schema_version: str = FOLDS_SCHEMA_VERSION


def _eligible_rows(pairs: pd.DataFrame) -> pd.DataFrame:
    """Rows that can enter a fold: resolved (Tier A/B), not quarantined/unresolved."""
    resolved = pairs[pairs["resolution_status"] == "resolved"].copy()
    resolved["contribution_id"] = resolved["contribution_id"].astype(str)
    return resolved


def build_folds(
    pairs: pd.DataFrame,
    *,
    n_splits: int = DEFAULT_N_SPLITS,
    seed: int = DEFAULT_SEED,
    contribution_texts: dict[str, str] | None = None,
) -> FoldAssignment:
    """Build grouped outer folds over the resolved (Tier A/B) cohort.

    Each fold's test set is balanced by *row count* (GroupKFold splits on
    the actual pair rows, not the list of unique contributions), while never
    splitting a contribution's rows across train and test -- and never
    splitting two near-duplicate contributions from each other, when
    contribution_texts makes that detectable.

    Args:
        pairs: dataset/pairs.parquet's contents (or an equivalent
            DataFrame) with at least contribution_id and resolution_status
            columns.
        n_splits: Number of outer folds.
        seed: Random seed for reproducibility.
        contribution_texts: Optional contribution_id -> text lookup; see
            find_near_duplicate_groups.

    Returns:
        A FoldAssignment with one entry per fold.

    Raises:
        ValueError: If there are fewer distinct contribution groups than
            n_splits.
    """
    resolved = _eligible_rows(pairs)

    dup_groups = find_near_duplicate_groups(contribution_texts)
    resolved["fold_group"] = resolved["contribution_id"].map(lambda cid: dup_groups.get(cid, cid))

    unique_groups = resolved["fold_group"].nunique()
    if unique_groups < n_splits:
        raise ValueError(
            f"Only {unique_groups} distinct contribution group(s) available, "
            f"need at least {n_splits} for {n_splits}-fold CV."
        )

    gkf = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X = resolved.index.to_numpy()
    groups = resolved["fold_group"].to_numpy()

    fold_entries = []
    for train_idx, test_idx in gkf.split(X, groups=groups):
        train_ids = sorted(resolved.iloc[train_idx]["contribution_id"].unique().tolist())
        test_ids = sorted(resolved.iloc[test_idx]["contribution_id"].unique().tolist())
        fold_entries.append({"train_contribution_ids": train_ids, "test_contribution_ids": test_ids})

    merged_count = sum(1 for cid, g in dup_groups.items() if cid != g)

    return FoldAssignment(
        seed=seed,
        n_splits=n_splits,
        grouping_key="contribution_id",
        near_duplicate_check="performed" if contribution_texts else "skipped_no_contribution_text_available",
        dup_groups_merged=merged_count,
        folds=fold_entries,
    )


def build_inner_split(
    pairs: pd.DataFrame,
    train_contribution_ids: list[str],
    *,
    test_size: float = DEFAULT_INNER_TEST_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[list[str], list[str]]:
    """Split one outer fold's training contributions into inner-train/inner-val.

    Used to tune review-routing thresholds and select few-shot demonstration
    examples using only the inner-train partition, before the outer test
    fold (or even the inner-val partition) is touched -- per the brief,
    demonstrations must come only from the current training partition.

    Args:
        pairs: dataset/pairs.parquet's contents.
        train_contribution_ids: The outer fold's training contribution IDs
            (FoldAssignment.folds[i]["train_contribution_ids"]).
        test_size: Fraction of groups (by row count) held out as inner-val.
        seed: Random seed for reproducibility.

    Returns:
        (inner_train_contribution_ids, inner_val_contribution_ids), both
        sorted lists, disjoint, together covering train_contribution_ids.
    """
    resolved = _eligible_rows(pairs)
    subset = resolved[resolved["contribution_id"].isin(set(train_contribution_ids))]

    unique_groups = subset["contribution_id"].nunique()
    if unique_groups < 2:
        # Too few groups to hold anything out; everything stays in inner-train.
        return sorted(subset["contribution_id"].unique().tolist()), []

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    X = subset.index.to_numpy()
    groups = subset["contribution_id"].to_numpy()
    inner_train_idx, inner_val_idx = next(gss.split(X, groups=groups))

    inner_train_ids = sorted(subset.iloc[inner_train_idx]["contribution_id"].unique().tolist())
    inner_val_ids = sorted(subset.iloc[inner_val_idx]["contribution_id"].unique().tolist())
    return inner_train_ids, inner_val_ids


def save_folds(assignment: FoldAssignment, path: Path) -> None:
    """Persist a FoldAssignment to disk as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(assignment), f, indent=2)


def load_folds(path: Path) -> FoldAssignment:
    """Load a previously-persisted FoldAssignment from disk."""
    with open(Path(path), "r", encoding="utf-8") as f:
        data = json.load(f)
    return FoldAssignment(**data)
