import pandas as pd
import pytest

from fold_builder import (
    build_folds,
    build_inner_split,
    find_near_duplicate_groups,
    load_folds,
    save_folds,
)


def _pairs_df(rows_per_contribution: dict[str, int], resolved: bool = True) -> pd.DataFrame:
    rows = []
    pair_counter = 0
    for cid, n in rows_per_contribution.items():
        for _ in range(n):
            rows.append(
                {
                    "pair_id": f"p{pair_counter}",
                    "contribution_id": cid,
                    "resolution_status": "resolved" if resolved else "unresolved",
                }
            )
            pair_counter += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# find_near_duplicate_groups
# ---------------------------------------------------------------------------


def test_find_near_duplicate_groups_none_returns_empty():
    assert find_near_duplicate_groups(None) == {}


def test_find_near_duplicate_groups_empty_dict_returns_empty():
    assert find_near_duplicate_groups({}) == {}


def test_find_near_duplicate_groups_merges_identical_normalized_text():
    texts = {
        "1": "Hello World.  Extra   spaces.",
        "2": "hello world. extra spaces.",  # same after normalization
        "3": "Completely different text.",
    }

    groups = find_near_duplicate_groups(texts)

    assert groups["1"] == groups["2"]
    assert groups["3"] != groups["1"]


def test_find_near_duplicate_groups_no_duplicates_maps_each_to_itself():
    texts = {"1": "aaa", "2": "bbb"}

    groups = find_near_duplicate_groups(texts)

    assert groups == {"1": "1", "2": "2"}


# ---------------------------------------------------------------------------
# build_folds
# ---------------------------------------------------------------------------


def test_build_folds_keeps_each_contribution_entirely_in_one_split():
    # 10 contributions with varying row counts, so grouping actually matters.
    pairs = _pairs_df({str(i): (i % 4) + 1 for i in range(10)})

    assignment = build_folds(pairs, n_splits=5, seed=1)

    assert len(assignment.folds) == 5
    for fold in assignment.folds:
        train = set(fold["train_contribution_ids"])
        test = set(fold["test_contribution_ids"])
        assert train.isdisjoint(test)
        assert train | test == set(pairs["contribution_id"].astype(str).unique())


def test_build_folds_every_contribution_appears_in_exactly_one_test_fold():
    pairs = _pairs_df({str(i): 2 for i in range(10)})

    assignment = build_folds(pairs, n_splits=5, seed=1)

    test_appearances: dict[str, int] = {}
    for fold in assignment.folds:
        for cid in fold["test_contribution_ids"]:
            test_appearances[cid] = test_appearances.get(cid, 0) + 1

    assert all(count == 1 for count in test_appearances.values())
    assert set(test_appearances) == set(pairs["contribution_id"].astype(str).unique())


def test_build_folds_excludes_unresolved_rows():
    resolved = _pairs_df({str(i): 2 for i in range(6)}, resolved=True)
    unresolved = _pairs_df({"unresolved-contrib": 5}, resolved=False)
    pairs = pd.concat([resolved, unresolved], ignore_index=True)

    assignment = build_folds(pairs, n_splits=3, seed=1)

    all_ids = {cid for fold in assignment.folds for cid in fold["train_contribution_ids"] + fold["test_contribution_ids"]}
    assert "unresolved-contrib" not in all_ids


def test_build_folds_raises_when_too_few_groups():
    pairs = _pairs_df({"1": 3, "2": 3})

    with pytest.raises(ValueError, match="need at least"):
        build_folds(pairs, n_splits=5, seed=1)


def test_build_folds_is_reproducible_for_the_same_seed():
    pairs = _pairs_df({str(i): (i % 3) + 1 for i in range(12)})

    a = build_folds(pairs, n_splits=4, seed=7)
    b = build_folds(pairs, n_splits=4, seed=7)

    assert a.folds == b.folds


def test_build_folds_records_near_duplicate_check_status():
    pairs = _pairs_df({str(i): 2 for i in range(6)})

    without_texts = build_folds(pairs, n_splits=3, seed=1)
    assert without_texts.near_duplicate_check == "skipped_no_contribution_text_available"
    assert without_texts.dup_groups_merged == 0

    with_texts = build_folds(
        pairs, n_splits=3, seed=1, contribution_texts={str(i): f"text {i}" for i in range(6)}
    )
    assert with_texts.near_duplicate_check == "performed"


def test_build_folds_merges_near_duplicate_contributions_into_the_same_fold():
    pairs = _pairs_df({str(i): 2 for i in range(8)})
    # Contributions "0" and "1" are near-duplicates (same normalized text).
    texts = {str(i): (f"identical text" if i in (0, 1) else f"unique text {i}") for i in range(8)}

    assignment = build_folds(pairs, n_splits=4, seed=3, contribution_texts=texts)

    for fold in assignment.folds:
        train, test = set(fold["train_contribution_ids"]), set(fold["test_contribution_ids"])
        # "0" and "1" must always land on the same side.
        assert ("0" in train) == ("1" in train)
        assert ("0" in test) == ("1" in test)
    assert assignment.dup_groups_merged == 1


# ---------------------------------------------------------------------------
# build_inner_split
# ---------------------------------------------------------------------------


def test_build_inner_split_is_disjoint_and_covers_train_ids():
    pairs = _pairs_df({str(i): 2 for i in range(10)})
    train_ids = [str(i) for i in range(10)]

    inner_train, inner_val = build_inner_split(pairs, train_ids, seed=1)

    assert set(inner_train).isdisjoint(inner_val)
    assert set(inner_train) | set(inner_val) == set(train_ids)
    assert len(inner_val) > 0  # test_size=0.2 default over 10 groups holds some out


def test_build_inner_split_too_few_groups_keeps_everything_in_train():
    pairs = _pairs_df({"1": 3})

    inner_train, inner_val = build_inner_split(pairs, ["1"], seed=1)

    assert inner_train == ["1"]
    assert inner_val == []


# ---------------------------------------------------------------------------
# save_folds / load_folds
# ---------------------------------------------------------------------------


def test_save_then_load_round_trips(tmp_path):
    pairs = _pairs_df({str(i): 2 for i in range(6)})
    assignment = build_folds(pairs, n_splits=3, seed=1)
    path = tmp_path / "folds_v1.json"

    save_folds(assignment, path)
    loaded = load_folds(path)

    assert loaded == assignment


def test_save_folds_creates_parent_directory(tmp_path):
    pairs = _pairs_df({str(i): 2 for i in range(6)})
    assignment = build_folds(pairs, n_splits=3, seed=1)
    path = tmp_path / "nested" / "dir" / "folds_v1.json"

    save_folds(assignment, path)

    assert path.exists()
