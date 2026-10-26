"""Tests for votes_stance_export (synthetic data only).

These tests never touch the real dataset/pairs.parquet, the real model
prediction files or the real vote snapshot. Every input is built in tmp_path:
synthetic pairs and model parquets and a synthetic, hash-marked vote snapshot
directory (same spirit as test_vote_ratio_hypothesis.py). No real favor or
vote ratio is ever computed here.
"""
import json

import pandas as pd
import pytest

import votes_stance_export as vse
import vote_ratio_hypothesis as vr
from dataset_builder import compute_pair_id
from reaction_split_review import comment_id_for
from review_pack_common import write_artifact_hashes, write_json


# --------------------------------------------------------------------------
# Synthetic fixtures
# --------------------------------------------------------------------------


def _pairs_frame():
    """Build a minimal gold pairs frame shaped like dataset/pairs.parquet.

    Two contributions, four comments, with a deliberate mix so the roll-up has
    one in_favor comment, one against comment, one mixed comment (both labels)
    and one neutral comment. pair_id is computed with the shared helper so the
    synthetic model frames join to it.
    """
    rows = [
        # contribution 10, comment A: one in_favor pair -> comment is in_favor
        (10, "kmA", "comment A", "in_favor"),
        # contribution 10, comment B: one against pair -> comment is against
        (10, "kmB", "comment B", "against"),
        # contribution 20, comment C: in_favor + against -> comment is mixed
        (20, "kmC1", "comment C", "in_favor"),
        (20, "kmC2", "comment C", "against"),
        # contribution 20, comment D: one neutral pair -> comment is neutral
        (20, "kmD", "comment D", "neutral"),
    ]
    records = []
    for cid, km, comment, label in rows:
        records.append(
            {
                "pair_id": compute_pair_id(str(cid), km, comment),
                "contribution_id": cid,
                "key_message": km,
                "comment_text": comment,
                "reference_label": label,
                "resolution_status": "resolved",
                "cohort_role": "primary",
                "tier": "A",
            }
        )
    return pd.DataFrame(records)


def _model_frame(stance_by_comment):
    """Build a prediction parquet frame using the raw (German) stance words.

    Args:
        stance_by_comment: Maps (contribution_id, key_message, comment_text) to
            the raw stance string to emit.
    """
    records = []
    for (cid, km, comment), raw in stance_by_comment.items():
        records.append(
            {
                "contribution_id": cid,
                "key_message": km,
                "comment_text": comment,
                "stance": raw,
            }
        )
    return pd.DataFrame(records)


@pytest.fixture
def pairs_path(tmp_path):
    path = tmp_path / "pairs.parquet"
    _pairs_frame().to_parquet(path, index=False)
    return path


@pytest.fixture
def model_paths(tmp_path):
    """Two synthetic model files with the raw stance vocabulary."""
    # gptoss agrees with gold on every comment.
    gptoss = _model_frame(
        {
            (10, "kmA", "comment A"): "in favor",
            (10, "kmB", "comment B"): "ablehnung",
            (20, "kmC1", "comment C"): "in favor",
            (20, "kmC2", "comment C"): "ablehnung",
            (20, "kmD", "comment D"): "neutral",
        }
    )
    # base is missing one pair, to exercise the left-join keeping gold rows.
    base = _model_frame(
        {
            (10, "kmA", "comment A"): "neutral",
            (10, "kmB", "comment B"): "neutral",
            (20, "kmC1", "comment C"): "in favor",
            (20, "kmD", "comment D"): "neutral",
        }
    )
    gpath = tmp_path / "gptoss.parquet"
    bpath = tmp_path / "base.parquet"
    gptoss.to_parquet(gpath, index=False)
    base.to_parquet(bpath, index=False)
    return {"gptoss-120b": gpath, "base": bpath}


@pytest.fixture
def snapshot_dir(tmp_path):
    """A synthetic, hash-marked vote-ratio-fetch directory."""
    directory = tmp_path / "votes_snap"
    directory.mkdir()
    votes = pd.DataFrame(
        {
            "contribution_id": [10, 20],
            "project_id": ["proj-x", "proj-y"],
            "votingPro": [7, 2],
            "votingContra": [3, 8],
            "commentsNumber": [2, 3],
            "dateCreated": ["2022-01-01", "2022-02-02"],
        }
    )
    snapshot = directory / vr.SNAPSHOT_FILE
    summary = directory / vr.FETCH_SUMMARY_FILE
    votes.to_parquet(snapshot, index=False)
    write_json(summary, {"created_utc": "20260102T030405060708Z", "note": "synthetic"})
    # Completion marker written last, exactly as the real fetch does.
    write_artifact_hashes(directory, "20260102T030405060708Z", [snapshot, summary])
    return directory


# --------------------------------------------------------------------------
# normalize_stance
# --------------------------------------------------------------------------


def test_normalize_stance_maps_each_known_label():
    assert vse.normalize_stance("in favor") == "in_favor"
    assert vse.normalize_stance("ablehnung") == "against"
    assert vse.normalize_stance("neutral") == "neutral"


def test_normalize_stance_rejects_unknown_label():
    with pytest.raises(ValueError):
        vse.normalize_stance("undecided")


# --------------------------------------------------------------------------
# load_models
# --------------------------------------------------------------------------


def test_load_models_reuses_shared_id_helpers(model_paths):
    models = vse.load_models(model_paths)
    frame = models["gptoss-120b"]
    # pair_id / comment_id must match the pipeline's own derivation exactly.
    row = frame[frame.comment_id == comment_id_for("10", "comment A")].iloc[0]
    assert row.pair_id == compute_pair_id("10", "kmA", "comment A")
    assert row.contribution_id == "10"
    # Labels are normalised to the canonical vocabulary.
    assert set(frame.model_label) <= {"in_favor", "against", "neutral"}


def test_load_models_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        vse.load_models({"x": tmp_path / "nope.parquet"})


def test_load_models_collapses_identical_duplicates(tmp_path):
    # The same pair predicted twice with the same label collapses to one row.
    dup = _model_frame({(10, "kmA", "comment A"): "in favor"})
    dup = pd.concat([dup, dup], ignore_index=True)
    path = tmp_path / "dup.parquet"
    dup.to_parquet(path, index=False)
    frame = vse.load_models({"x": path})["x"]
    assert len(frame) == 1


def test_load_models_rejects_conflicting_duplicates(tmp_path):
    # The same pair predicted with two different labels is a data error.
    conflict = pd.DataFrame(
        {
            "contribution_id": [10, 10],
            "key_message": ["kmA", "kmA"],
            "comment_text": ["comment A", "comment A"],
            "stance": ["in favor", "ablehnung"],
        }
    )
    path = tmp_path / "conflict.parquet"
    conflict.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        vse.load_models({"x": path})


def test_load_models_empty_file_raises(tmp_path):
    empty = tmp_path / "empty.parquet"
    pd.DataFrame(
        {"contribution_id": [], "key_message": [], "comment_text": [], "stance": []}
    ).to_parquet(empty, index=False)
    with pytest.raises(ValueError):
        vse.load_models({"x": empty})


# --------------------------------------------------------------------------
# build_comments_table
# --------------------------------------------------------------------------


def test_build_comments_table_has_gold_and_model_columns(pairs_path, model_paths):
    models = vse.load_models(model_paths)
    table = vse.build_comments_table(pairs_path, models)
    # Four distinct comments (A, B, C, D), one row each.
    assert len(table) == 4
    for col in (
        "comment_id",
        "contribution_id",
        "comment_text",
        "gold_category",
        "category__gptoss-120b",
        "category__base",
    ):
        assert col in table.columns
    by_comment = table.set_index("comment_text")
    # Comment C has both in_favor and against gold pairs -> mixed.
    assert by_comment.loc["comment C", "gold_category"] == "mixed"
    # The agreeing model reproduces every gold comment category.
    assert (table["category__gptoss-120b"] == table["gold_category"]).all()


def test_build_comments_table_left_join_keeps_unpredicted_comments(pairs_path, model_paths):
    models = vse.load_models(model_paths)
    table = vse.build_comments_table(pairs_path, models)
    # base rolls comment C to in_favor (it only labelled the in_favor pair),
    # while gold had it mixed - the join keeps the gold comment either way.
    row = table[table.comment_text == "comment C"].iloc[0]
    assert row["gold_category"] == "mixed"
    assert row["category__base"] == "in_favor"


# --------------------------------------------------------------------------
# copy_votes
# --------------------------------------------------------------------------


def test_copy_votes_returns_snapshot_time(snapshot_dir):
    votes, snapshot_utc, sha = vse.copy_votes(snapshot_dir)
    assert snapshot_utc == "20260102T030405060708Z"
    assert set(votes.contribution_id) == {"10", "20"}
    assert isinstance(sha, str) and len(sha) == 64


def test_copy_votes_detects_tampering(snapshot_dir):
    # Rewriting the snapshot after the hashes file must be caught.
    (snapshot_dir / vr.SNAPSHOT_FILE).write_bytes(b"corrupted")
    with pytest.raises((RuntimeError, Exception)):
        vse.copy_votes(snapshot_dir)


# --------------------------------------------------------------------------
# run_export
# --------------------------------------------------------------------------


def _run(tmp_path, pairs_path, model_paths, snapshot_dir, name="out"):
    return vse.run_export(
        pairs_path=pairs_path,
        model_paths=model_paths,
        votes_snapshot_dir=snapshot_dir,
        output_dir=tmp_path / name,
    )


def test_run_export_writes_all_artifacts(tmp_path, pairs_path, model_paths, snapshot_dir):
    result = _run(tmp_path, pairs_path, model_paths, snapshot_dir)
    for role in ("comments", "votes", "summary", "hashes"):
        assert result.files[role].exists()
    summary = result.summary
    assert summary["schema_version"] == vse.SCHEMA_VERSION
    assert summary["snapshot_utc"] == "20260102T030405060708Z"
    assert summary["available_models"] == ["base", "gptoss-120b"]
    assert summary["descriptive_only_note"] == vse.DESCRIPTIVE_ONLY_NOTE
    # Provenance: every input is hashed.
    assert "pairs" in summary["inputs"]
    assert set(summary["inputs"]["models"]) == {"base", "gptoss-120b"}


def test_run_export_rollup_parity_with_hypothesis(tmp_path, pairs_path, model_paths, snapshot_dir):
    result = _run(tmp_path, pairs_path, model_paths, snapshot_dir)
    comments_table = pd.read_parquet(result.files["comments"])
    # The summary counts must equal a direct favor_share over the gold
    # categories via the hypothesis module (same rule table).
    expected = vr.favor_share(comments_table.gold_category)
    gold = result.summary["counts"]["gold_favor_counts"]
    # one in_favor (A), one against (B), one mixed (C), one neutral (D)
    assert gold["favor"] == expected["favor"] == 1
    assert gold["against"] == expected["against"] == 1
    assert gold["neutral"] == expected["neutral"] == 1
    assert gold["mixed"] == expected["mixed"] == 1


def test_run_export_hashes_written_last_and_verify(tmp_path, pairs_path, model_paths, snapshot_dir):
    result = _run(tmp_path, pairs_path, model_paths, snapshot_dir)
    recorded = json.loads(result.files["hashes"].read_text(encoding="utf-8"))
    assert recorded["complete"] is True
    # The hashes file records every other artifact, so re-loading the snapshot
    # it wrote verifies cleanly.
    assert set(recorded["files"]) == {vse.COMMENTS_FILE, vse.VOTES_FILE, vse.SUMMARY_FILE}


def test_run_export_refuses_existing_directory(tmp_path, pairs_path, model_paths, snapshot_dir):
    target = tmp_path / "already"
    target.mkdir()
    with pytest.raises(FileExistsError):
        vse.run_export(
            pairs_path=pairs_path,
            model_paths=model_paths,
            votes_snapshot_dir=snapshot_dir,
            output_dir=target,
        )
