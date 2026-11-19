"""Tests for votes_stance_export (synthetic data only).

These tests never touch the real dataset/pairs.parquet, the real model
prediction files or the real vote snapshot. Every input is built in tmp_path:
synthetic pairs and model parquets and a synthetic, hash-marked vote snapshot
directory (same spirit as test_vote_ratio_hypothesis.py). No real favor or
vote ratio is ever computed here.
"""
import json
from pathlib import Path

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


def _synthetic_analysis_summary():
    """A minimal frozen analysis summary shaped like the real pre-registered one.

    Only the fields :func:`votes_stance_export.load_overall_summary` reads are
    populated; the values are deliberately *not* the real ones so a test can
    prove they are copied verbatim rather than recomputed.
    """
    return {
        "interpretation_rules": "SYNTHETIC RULES: this is not the real text.",
        "counts": {"n_comments": 9, "n_contributions_with_votes": 4},
        "primary": {
            "favor": 5,
            "against": 2,
            "neutral": 1,
            "mixed": 1,
            "favor_share": 71.4,
            "vote_pro": 40,
            "vote_contra": 10,
            "vote_share": 80.0,
            "difference_pp": -8.6,
            "ci90": {"low": -15.0, "high": -2.0, "equivalent": False},
        },
    }


@pytest.fixture
def analysis_summary_path(tmp_path):
    """A synthetic frozen analysis summary JSON written into tmp_path."""
    path = tmp_path / "vote_ratio_analysis.summary.json"
    write_json(path, _synthetic_analysis_summary())
    return path


def _run(tmp_path, pairs_path, model_paths, snapshot_dir, name="out",
         analysis_summary_path=None):
    # Keep run_export fully synthetic: if the caller gave no analysis summary,
    # write a throwaway synthetic one so the real frozen file is never read.
    if analysis_summary_path is None:
        analysis_summary_path = tmp_path / f"{name}_analysis.summary.json"
        write_json(analysis_summary_path, _synthetic_analysis_summary())
    return vse.run_export(
        pairs_path=pairs_path,
        model_paths=model_paths,
        votes_snapshot_dir=snapshot_dir,
        analysis_summary_path=analysis_summary_path,
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
    assert set(recorded["files"]) == {
        vse.COMMENTS_FILE,
        vse.KEY_MESSAGES_FILE,
        vse.SPANS_FILE,
        vse.VOTES_FILE,
        vse.SUMMARY_FILE,
    }


def test_run_export_refuses_existing_directory(
    tmp_path, pairs_path, model_paths, snapshot_dir, analysis_summary_path
):
    target = tmp_path / "already"
    target.mkdir()
    with pytest.raises(FileExistsError):
        vse.run_export(
            pairs_path=pairs_path,
            model_paths=model_paths,
            votes_snapshot_dir=snapshot_dir,
            analysis_summary_path=analysis_summary_path,
            output_dir=target,
        )


# --------------------------------------------------------------------------
# build_key_messages_table
# --------------------------------------------------------------------------


def test_build_key_messages_table_one_row_per_pair(pairs_path):
    table = vse.build_key_messages_table(pairs_path)
    # One row per gold pair: the synthetic pairs frame has five pairs.
    assert len(table) == 5
    assert list(table.columns) == [
        "contribution_id",
        "key_message",
        "comment_id",
        "comment_text",
        "reference_label",
    ]
    # Contribution ids are strings (joinable to votes and the comment set).
    assert table.contribution_id.map(type).eq(str).all()
    # comment_id matches the shared helper's derivation.
    row = table[table.comment_text == "comment A"].iloc[0]
    assert row.comment_id == comment_id_for("10", "comment A")
    assert row.reference_label == "in_favor"


def test_build_key_messages_table_counts_per_key_message(pairs_path):
    table = vse.build_key_messages_table(pairs_path)
    # Contribution 20 has two key messages (kmC1, kmC2, kmD) -> three gold pairs.
    cid20 = table[table.contribution_id == "20"]
    assert len(cid20) == 3
    assert set(cid20.key_message) == {"kmC1", "kmC2", "kmD"}
    # Distinct (contribution, key_message) pairs across the whole table.
    distinct = table.drop_duplicates(["contribution_id", "key_message"])
    assert len(distinct) == 5


def test_build_key_messages_table_preserves_first_appearance_order(pairs_path):
    table = vse.build_key_messages_table(pairs_path)
    # The order mirrors pair_label_set (first-appearance order), so a key
    # message's first row never appears after a later key message's first row.
    first_seen = table.key_message.drop_duplicates().tolist()
    assert first_seen == ["kmA", "kmB", "kmC1", "kmC2", "kmD"]


def test_build_key_messages_table_thin_boundary(pairs_path):
    # A key message backed by only two stance-bearing comments is "thin" (the
    # backend will mark it too_few at <3); three is the first non-thin count.
    # Here kmA is backed by a single pair, proving the per-pair grain the
    # backend's 3-comment boundary is computed from is preserved faithfully.
    table = vse.build_key_messages_table(pairs_path)
    kmA = table[(table.contribution_id == "10") & (table.key_message == "kmA")]
    assert len(kmA) == 1  # one stance-bearing comment behind kmA


# --------------------------------------------------------------------------
# load_overall_summary (pre-registered, copied verbatim)
# --------------------------------------------------------------------------


def test_load_overall_summary_is_verbatim(analysis_summary_path):
    overall = vse.load_overall_summary(analysis_summary_path)
    source = _synthetic_analysis_summary()
    assert overall["pre_registered"] is True
    # The primary block is copied verbatim, field for field.
    assert overall["primary"] == source["primary"]
    assert overall["n_comments"] == source["counts"]["n_comments"]
    assert (
        overall["n_contributions_with_votes"]
        == source["counts"]["n_contributions_with_votes"]
    )
    assert overall["interpretation_rules"] == source["interpretation_rules"]
    # no_clear_stance is neutral + mixed from the verbatim primary block.
    assert overall["no_clear_stance"] == {"total": 2, "neutral": 1, "mixed": 1}


def test_load_overall_summary_does_not_mutate_source(analysis_summary_path):
    # Mutating the returned primary block must not change a second read.
    first = vse.load_overall_summary(analysis_summary_path)
    first["primary"]["favor"] = 999
    second = vse.load_overall_summary(analysis_summary_path)
    assert second["primary"]["favor"] == 5


# --------------------------------------------------------------------------
# compute_shared_contributions (exploratory, not pre-registered)
# --------------------------------------------------------------------------


def _shared_votes_frame():
    """Votes for three contributions exercising each shared-set category.

    - "10": has directional comments (A in_favor, B against) AND votes -> both.
    - "20": has only non-directional comments (C mixed, D neutral) but votes ->
      thumbs-side only.
    - "30": has no row in the pairs frame but has votes -> thumbs-side only.
    """
    return pd.DataFrame(
        {
            "contribution_id": ["10", "20", "30"],
            "votingPro": [7, 2, 5],
            "votingContra": [3, 8, 5],
        }
    )


def test_compute_shared_contributions_counts_and_flag(pairs_path):
    votes = _shared_votes_frame()
    shared = vse.compute_shared_contributions(pairs_path, votes)
    assert shared["pre_registered"] is False
    assert shared["note"] == vse.NOT_PRE_REGISTERED_NOTE
    # All three contributions have votes; only "10" has a directional comment.
    assert shared["thumbs_side_contributions"] == 3
    assert shared["comment_side_contributions"] == 1
    assert shared["both_sides_contributions"] == 1


def test_compute_shared_contributions_pooled_point_estimate(pairs_path):
    votes = _shared_votes_frame()
    shared = vse.compute_shared_contributions(pairs_path, votes)
    # Only "10" is on both sides: favour share = 1/2 = 50%, thumbs-up share =
    # 7/10 = 70%, so the gap is -20 pp for both pooled and unweighted (one
    # contribution, so pooling and averaging coincide).
    assert shared["pooled"]["difference_pp"] == pytest.approx(-20.0)
    assert shared["unweighted"]["difference_pp"] == pytest.approx(-20.0)
    assert shared["bootstrap"] == {
        "n": vse.SHARED_BOOTSTRAP_N,
        "seed": vse.SHARED_BOOTSTRAP_SEED,
    }


# --------------------------------------------------------------------------
# run_export: new artifacts and provenance
# --------------------------------------------------------------------------


def test_run_export_writes_key_messages_parquet(
    tmp_path, pairs_path, model_paths, snapshot_dir
):
    result = _run(tmp_path, pairs_path, model_paths, snapshot_dir)
    assert result.files["key_messages"].exists()
    table = pd.read_parquet(result.files["key_messages"])
    assert len(table) == 5
    assert "reference_label" in table.columns
    # Counts recorded in the summary match the parquet.
    counts = result.summary["counts"]
    assert counts["key_message_rows"] == 5
    assert counts["key_messages"] == 5


def test_run_export_hashes_list_key_messages_parquet(
    tmp_path, pairs_path, model_paths, snapshot_dir
):
    result = _run(tmp_path, pairs_path, model_paths, snapshot_dir)
    recorded = json.loads(result.files["hashes"].read_text(encoding="utf-8"))
    assert vse.KEY_MESSAGES_FILE in recorded["files"]
    assert set(recorded["files"]) == {
        vse.COMMENTS_FILE,
        vse.KEY_MESSAGES_FILE,
        vse.SPANS_FILE,
        vse.VOTES_FILE,
        vse.SUMMARY_FILE,
    }


def test_run_export_records_analysis_input_hash(
    tmp_path, pairs_path, model_paths, snapshot_dir, analysis_summary_path
):
    result = _run(
        tmp_path, pairs_path, model_paths, snapshot_dir,
        analysis_summary_path=analysis_summary_path,
    )
    overall_input = result.summary["inputs"]["overall_analysis"]
    assert overall_input["path"] == str(analysis_summary_path)
    assert len(overall_input["sha256"]) == 64


def test_run_export_embeds_overall_and_shared_blocks(
    tmp_path, pairs_path, model_paths, snapshot_dir, analysis_summary_path
):
    result = _run(
        tmp_path, pairs_path, model_paths, snapshot_dir,
        analysis_summary_path=analysis_summary_path,
    )
    summary = result.summary
    # Pre-registered headline is present and copied verbatim.
    assert summary["overall_summary"]["pre_registered"] is True
    assert summary["overall_summary"]["primary"]["favor"] == 5
    # Exploratory shared-set block is present and flagged not pre-registered.
    assert summary["shared_contributions"]["pre_registered"] is False
    assert "thumbs_side_contributions" in summary["shared_contributions"]


# --------------------------------------------------------------------------
# Integration test on the REAL frozen inputs
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not (
        vse.DEFAULT_PAIRS_PATH.exists()
        and vse.DEFAULT_VOTES_SNAPSHOT_DIR.exists()
        and vse.DEFAULT_ANALYSIS_SUMMARY_PATH.exists()
    ),
    reason="real frozen inputs not present in this checkout",
)
def test_real_inputs_reproduce_published_figures():
    """The real frozen inputs must reproduce the published headline figures."""
    votes, _, _ = vse.copy_votes(vse.DEFAULT_VOTES_SNAPSHOT_DIR)

    # Key-message grain: 707 gold pairs, 320 distinct (contribution, key_message).
    km = vse.build_key_messages_table(vse.DEFAULT_PAIRS_PATH)
    assert len(km) == 707
    assert km.drop_duplicates(["contribution_id", "key_message"]).shape[0] == 320

    # Pre-registered headline, copied verbatim.
    overall = vse.load_overall_summary(vse.DEFAULT_ANALYSIS_SUMMARY_PATH)
    primary = overall["primary"]
    assert (primary["favor"], primary["against"], primary["neutral"], primary["mixed"]) == (
        130, 31, 74, 11,
    )
    assert round(primary["favor_share"], 3) == 80.745
    assert round(primary["vote_share"], 3) == 88.940
    assert round(primary["difference_pp"], 3) == -8.194
    assert round(primary["ci90"]["low"], 3) == -14.642
    assert round(primary["ci90"]["high"], 3) == -1.066
    assert primary["ci90"]["equivalent"] is False
    assert overall["n_comments"] == 246
    assert overall["no_clear_stance"] == {"total": 85, "neutral": 74, "mixed": 11}
    assert overall["n_contributions_with_votes"] == 107

    # Exploratory shared-set figures, computed fresh and reproducible.
    shared = vse.compute_shared_contributions(vse.DEFAULT_PAIRS_PATH, votes)
    assert shared["thumbs_side_contributions"] == 107
    assert shared["comment_side_contributions"] == 85
    assert shared["both_sides_contributions"] == 80
    assert round(shared["pooled"]["difference_pp"], 1) == -4.9
    assert round(shared["pooled"]["ci90"]["low"], 1) == -11.5
    assert round(shared["pooled"]["ci90"]["high"], 1) == 3.5
    assert round(shared["unweighted"]["difference_pp"], 1) == -4.0
    assert round(shared["unweighted"]["ci90"]["low"], 1) == -10.2
    assert round(shared["unweighted"]["ci90"]["high"], 1) == 2.0


# --------------------------------------------------------------------------
# Key-message source-sentence spans
# --------------------------------------------------------------------------


# A synthetic contribution text and the related sentences that resolve against it
# by each method. Offsets are half-open code-point spans into CONTENT_10 / 20.
CONTENT_10 = "Die Straße ist zu laut. Autos fahren hier zu schnell."
CONTENT_20 = (
    "Ich wünsche mir mehr Bäume. Das Parkhaus muss abgerissen werden. "
    "Deshalb ist Tempo 30 wichtig. Auch Lärmschutz hilft."
)


def _extraction_frame():
    """A synthetic mistral-nemo-shaped extraction frame for the gold pairs.

    Columns mirror the real prediction parquet's span-relevant fields. The gold
    pairs frame (``_pairs_frame``) uses contributions 10/20 and key messages
    kmA/kmB/kmC1/kmC2/kmD; this gives each one a related sentence that resolves
    by a specific method against CONTENT_10 / CONTENT_20:

    * kmA  -> verbatim  ("Die Straße ist zu laut.")
    * kmB  -> normalized (only case/whitespace differ)
    * kmC1 -> verbatim  ("Ich wünsche mir mehr Bäume.")
    * kmC2 -> fuzzy     (a one-letter typo: "abgerisen")
    * kmD  -> unmatched (a sentence that does not occur)

    Each key message is duplicated across two comment rows (as in the real file)
    so the loader's dedup-per-(cid, km) is exercised.
    """
    rows = [
        (10, "kmA", CONTENT_10, ["Die Straße ist zu laut."]),
        (10, "kmB", CONTENT_10, ["AUTOS   fahren hier zu schnell."]),
        (20, "kmC1", CONTENT_20, ["Ich wünsche mir mehr Bäume."]),
        (20, "kmC2", CONTENT_20, ["Das Parkhaus muss abgerisen werden."]),
        (20, "kmD", CONTENT_20, ["Ein Satz der hier gar nicht vorkommt."]),
    ]
    records = []
    for cid, km, content, sentences in rows:
        for _ in range(2):  # two comment rows per key message
            records.append(
                {
                    "contribution_id": cid,
                    "contributionContent": content,
                    "key_message": km,
                    "related_sentences": list(sentences),
                    "related_sentence_spans": [None],
                    # Extra columns the loader must ignore.
                    "comment_text": "a comment",
                    "stance": "neutral",
                }
            )
    return pd.DataFrame(records)


@pytest.fixture
def extraction_path(tmp_path):
    path = tmp_path / "mistral.parquet"
    _extraction_frame().to_parquet(path, index=False)
    return path


# -- _normalize_text / _normalized_index --


def test_normalize_text_collapses_and_lowers():
    assert vse._normalize_text("  Die   STRASSE\nist\tlaut  ") == "die strasse ist laut"


def test_normalized_index_maps_back_to_raw_offsets():
    content = "Die  Straße   ist laut"
    normalized, starts, ends = vse._normalized_index(content)
    assert normalized == "die straße ist laut"
    # Every raw offset is within the content and monotrack the source char.
    assert starts[0] == 0 and ends[0] == 1
    assert all(0 <= s < e <= len(content) for s, e in zip(starts, ends))


# -- _segments --


def test_segments_split_on_sentence_punctuation():
    content = "Erster Satz. Zweiter! Dritter?"
    segments = vse._segments(content)
    texts = [content[s:e] for s, e in segments]
    assert texts == ["Erster Satz.", "Zweiter!", "Dritter?"]


def test_segments_trailing_unterminated_segment():
    content = "Satz eins. Satz zwei ohne Punkt"
    texts = [content[s:e] for s, e in vse._segments(content)]
    assert texts == ["Satz eins.", "Satz zwei ohne Punkt"]


# -- per-method span resolution --


def test_verbatim_span_exact_match():
    span, method = vse._resolve_sentence_span(CONTENT_10, "Die Straße ist zu laut.")
    assert method == "verbatim"
    assert CONTENT_10[span[0]:span[1]] == "Die Straße ist zu laut."


def test_normalized_span_case_and_whitespace_only():
    span, method = vse._resolve_sentence_span(CONTENT_10, "AUTOS   fahren hier zu schnell.")
    assert method == "normalized"
    assert CONTENT_10[span[0]:span[1]] == "Autos fahren hier zu schnell."


def test_fuzzy_span_recovers_typo_sentence():
    span, method = vse._resolve_sentence_span(
        CONTENT_20, "Das Parkhaus muss abgerisen werden."
    )
    assert method == "fuzzy"
    assert CONTENT_20[span[0]:span[1]] == "Das Parkhaus muss abgerissen werden."


def test_fuzzy_span_spans_two_segments():
    # A related sentence that runs across two of the text's own sentences (the
    # Tempo-30 shape) is recovered by a multi-segment window.
    content = "Deshalb ist Tempo 30 wichtig. Auch Lärmschutz hilft hier."
    sentence = "Deshalb ist Tempo 30 wichtig und auch Lärmschutz hilft hier."
    span, method = vse._resolve_sentence_span(content, sentence)
    assert method == "fuzzy"
    matched = content[span[0]:span[1]]
    assert "Tempo 30" in matched and "Lärmschutz" in matched


def test_unmatched_sentence_returns_none():
    span, method = vse._resolve_sentence_span(CONTENT_20, "Ein Satz der hier gar nicht vorkommt.")
    assert span is None
    assert method == "unmatched"


def test_fuzzy_span_below_threshold_is_unmatched():
    # A sentence sharing only a couple of words must not be force-matched.
    span, method = vse._resolve_sentence_span(CONTENT_10, "Völlig anderer Inhalt ueber Katzen und Hunde im Garten.")
    assert span is None
    assert method == "unmatched"


# -- load_extraction_df --


def test_load_extraction_df_dedups_per_key_message(extraction_path):
    df = vse.load_extraction_df({vse.EXTRACTION_SPANS_MODEL: extraction_path})
    # Two comment rows per (cid, km) collapse to one row each: 5 gold key messages.
    assert len(df) == 5
    assert df.contribution_id.map(type).eq(str).all()
    assert set(df.columns) == {
        "contribution_id",
        "contributionContent",
        "key_message",
        "related_sentences",
        "related_sentence_spans",
    }


def test_load_extraction_df_missing_model_returns_empty():
    # No span-source model in the paths -> an empty (typed) frame, not an error.
    df = vse.load_extraction_df({"base": Path("/nonexistent.parquet")})
    assert df.empty
    assert "contributionContent" in df.columns


def test_load_extraction_df_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        vse.load_extraction_df({vse.EXTRACTION_SPANS_MODEL: tmp_path / "nope.parquet"})


# -- build_key_message_spans_table --


def test_build_key_message_spans_table_resolves_each_method(pairs_path, extraction_path):
    key_messages = vse.build_key_messages_table(pairs_path)
    extraction = vse.load_extraction_df({vse.EXTRACTION_SPANS_MODEL: extraction_path})
    table = vse.build_key_message_spans_table(key_messages, extraction)
    # One row per distinct (contribution, key message): five gold key messages.
    assert len(table) == 5
    by_km = {(r.contribution_id, r.key_message): r for r in table.itertuples(index=False)}
    assert by_km[("10", "kmA")].methods == ["verbatim"]
    assert by_km[("10", "kmB")].methods == ["normalized"]
    assert by_km[("20", "kmC2")].methods == ["fuzzy"]
    assert by_km[("20", "kmD")].methods == ["unmatched"]
    # The unmatched key message has an empty span list and is counted as such.
    kmd = by_km[("20", "kmD")]
    assert list(kmd.spans) == []
    assert kmd.unmatched == 1 and kmd.n_matched == 0
    # Every recorded span is a valid half-open range inside its content.
    assert all(r.extraction_model == vse.EXTRACTION_SPANS_MODEL for r in table.itertuples(index=False))


def test_build_key_message_spans_table_span_points_at_right_text(pairs_path, extraction_path):
    key_messages = vse.build_key_messages_table(pairs_path)
    extraction = vse.load_extraction_df({vse.EXTRACTION_SPANS_MODEL: extraction_path})
    table = vse.build_key_message_spans_table(key_messages, extraction)
    kma = table[(table.contribution_id == "10") & (table.key_message == "kmA")].iloc[0]
    s, e = kma.spans[0]
    assert CONTENT_10[s:e] == "Die Straße ist zu laut."


def test_build_key_message_spans_table_empty_extraction(pairs_path):
    # With no extraction rows, every gold key message is covered but empty.
    key_messages = vse.build_key_messages_table(pairs_path)
    empty = vse.load_extraction_df({"base": Path("/nope.parquet")})
    table = vse.build_key_message_spans_table(key_messages, empty)
    assert len(table) == 5
    assert all(len(list(r.spans)) == 0 for r in table.itertuples(index=False))
    assert all(r.n_sentences == 0 for r in table.itertuples(index=False))


# -- summarize_key_message_spans --


def test_summarize_key_message_spans_counts(pairs_path, extraction_path):
    key_messages = vse.build_key_messages_table(pairs_path)
    extraction = vse.load_extraction_df({vse.EXTRACTION_SPANS_MODEL: extraction_path})
    table = vse.build_key_message_spans_table(key_messages, extraction)
    counts = vse.summarize_key_message_spans(table)
    assert counts["extraction_model"] == vse.EXTRACTION_SPANS_MODEL
    assert counts["key_messages"] == 5
    assert counts["sentences_total"] == 5
    assert counts["verbatim"] == 2
    assert counts["normalized"] == 1
    assert counts["fuzzy"] == 1
    assert counts["unmatched"] == 1
    # Four key messages fully located, one (kmD) with none.
    assert counts["km_all"] == 4
    assert counts["km_some"] == 0
    assert counts["km_none"] == 1


# -- run_export with spans --


@pytest.fixture
def model_paths_with_spans(tmp_path, model_paths):
    """The base model_paths plus a span-source mistral-nemo extraction file."""
    path = tmp_path / "mistral.parquet"
    _extraction_frame().to_parquet(path, index=False)
    return {**model_paths, vse.EXTRACTION_SPANS_MODEL: path}


def test_run_export_writes_spans_parquet(
    tmp_path, pairs_path, model_paths_with_spans, snapshot_dir
):
    result = _run(tmp_path, pairs_path, model_paths_with_spans, snapshot_dir)
    assert result.files["key_message_spans"].exists()
    table = pd.read_parquet(result.files["key_message_spans"])
    assert len(table) == 5
    # The summary carries the per-method span counts and the span-source model.
    spans_counts = result.summary["counts"]["key_message_spans"]
    assert spans_counts["extraction_model"] == vse.EXTRACTION_SPANS_MODEL
    assert spans_counts["verbatim"] == 2
    assert spans_counts["unmatched"] == 1


def test_run_export_hashes_list_spans_parquet(
    tmp_path, pairs_path, model_paths_with_spans, snapshot_dir
):
    result = _run(tmp_path, pairs_path, model_paths_with_spans, snapshot_dir)
    recorded = json.loads(result.files["hashes"].read_text(encoding="utf-8"))
    assert vse.SPANS_FILE in recorded["files"]
    assert set(recorded["files"]) == {
        vse.COMMENTS_FILE,
        vse.KEY_MESSAGES_FILE,
        vse.SPANS_FILE,
        vse.VOTES_FILE,
        vse.SUMMARY_FILE,
    }


def test_run_export_without_span_source_still_writes_empty_spans(
    tmp_path, pairs_path, model_paths, snapshot_dir
):
    # The default synthetic model_paths has no mistral-nemo: the spans artifact is
    # still written (every key message empty) and hash-covered.
    result = _run(tmp_path, pairs_path, model_paths, snapshot_dir)
    assert result.files["key_message_spans"].exists()
    table = pd.read_parquet(result.files["key_message_spans"])
    assert len(table) == 5
    spans_counts = result.summary["counts"]["key_message_spans"]
    assert spans_counts["km_none"] == 5
