import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from dataset_builder import (
    build_cohort_b,
    build_dataset,
    build_policy_fixtures,
    collapse_duplicates,
    compute_pair_id,
    establish_original_predictions,
    load_csv_predictions,
    load_post_fix_predictions,
    load_stance_evaluations,
    resolve_cohort_a,
    write_dataset,
)
from policy_schemas import StanceLabelV2


# ---------------------------------------------------------------------------
# compute_pair_id
# ---------------------------------------------------------------------------


def test_compute_pair_id_is_deterministic():
    a = compute_pair_id(1, "km", "comment")
    b = compute_pair_id(1, "km", "comment")

    assert a == b


def test_compute_pair_id_differs_on_any_field():
    base = compute_pair_id(1, "km", "comment")

    assert compute_pair_id(2, "km", "comment") != base
    assert compute_pair_id(1, "other km", "comment") != base
    assert compute_pair_id(1, "km", "other comment") != base


# ---------------------------------------------------------------------------
# collapse_duplicates
# ---------------------------------------------------------------------------


def test_collapse_duplicates_identical_rows_collapse_to_one():
    df = pd.DataFrame(
        [
            {"pair_id": "p1", "contribution_id": 1, "old_stance": "in favor"},
            {"pair_id": "p1", "contribution_id": 1, "old_stance": "in favor"},
            {"pair_id": "p2", "contribution_id": 2, "old_stance": "neutral"},
        ]
    )

    result = collapse_duplicates(df, "src")

    assert len(result.collapsed) == 2
    assert result.identical_groups == [{"source": "src", "pair_id": "p1", "row_count": 2}]
    assert result.conflicting_keys == []


def test_collapse_duplicates_conflicting_rows_are_reported_not_collapsed():
    df = pd.DataFrame(
        [
            {"pair_id": "p1", "contribution_id": 1, "old_stance": "in favor"},
            {"pair_id": "p1", "contribution_id": 1, "old_stance": "neutral"},
        ]
    )

    result = collapse_duplicates(df, "src")

    # Neither conflicting row is arbitrarily kept.
    assert "p1" not in result.collapsed["pair_id"].tolist()
    assert result.conflicting_keys == [{"source": "src", "pair_id": "p1", "row_count": 2}]


# ---------------------------------------------------------------------------
# establish_original_predictions
# ---------------------------------------------------------------------------


def test_establish_original_predictions_agreement_resolves():
    frames = {
        "a": pd.DataFrame([{"pair_id": "p1", "old_stance": "neutral"}]),
        "b": pd.DataFrame([{"pair_id": "p1", "old_stance": "neutral"}]),
    }

    result = establish_original_predictions(frames)

    assert result.established["p1"] == StanceLabelV2.NEUTRAL
    assert result.disagreements == []


def test_establish_original_predictions_disagreement_is_unresolved():
    frames = {
        "a": pd.DataFrame([{"pair_id": "p1", "old_stance": "neutral"}]),
        "b": pd.DataFrame([{"pair_id": "p1", "old_stance": "in favor"}]),
    }

    result = establish_original_predictions(frames)

    assert "p1" not in result.established
    assert len(result.disagreements) == 1
    assert result.disagreements[0]["pair_id"] == "p1"


# ---------------------------------------------------------------------------
# load_post_fix_predictions
# ---------------------------------------------------------------------------


def test_load_post_fix_predictions_reads_new_stance():
    df = pd.DataFrame(
        [
            {"pair_id": "p1", "new_stance": "in favor"},
            {"pair_id": "p2", "new_stance": "neutral"},
        ]
    )

    result = load_post_fix_predictions(df)

    assert result == {"p1": StanceLabelV2.IN_FAVOR, "p2": StanceLabelV2.NEUTRAL}


def test_load_post_fix_predictions_skips_unrecognized_values():
    df = pd.DataFrame([{"pair_id": "p1", "new_stance": "not a real label"}])

    result = load_post_fix_predictions(df)

    assert result == {}


def test_load_post_fix_predictions_does_not_cross_check_agreement():
    """Unlike establish_original_predictions, a single source is trusted
    directly -- there is nothing to cross-check it against here."""
    df = pd.DataFrame([{"pair_id": "p1", "new_stance": "against"}])

    result = load_post_fix_predictions(df)

    assert result == {"p1": StanceLabelV2.AGAINST}


# ---------------------------------------------------------------------------
# resolve_cohort_a -- synthetic stance_evaluations rows
# ---------------------------------------------------------------------------


def _eval_row(**overrides):
    base = dict(
        id=1,
        contribution_id=1,
        key_message="km",
        comment_text="comment",
        verdict="correct",
        comment=None,
        evaluator="Antonie",
        corrected_label=None,
    )
    base.update(overrides)
    base["pair_id"] = compute_pair_id(
        base["contribution_id"], base["key_message"], base["comment_text"]
    )
    return base


def test_correct_verdict_uses_established_original():
    df = pd.DataFrame([_eval_row(verdict="correct")])
    established = {df.iloc[0]["pair_id"]: StanceLabelV2.IN_FAVOR}

    resolved = resolve_cohort_a(df, established)

    assert len(resolved) == 1
    assert resolved[0].resolution_status == "resolved"
    assert resolved[0].tier == "A"
    assert resolved[0].reference_label == "in_favor"
    assert resolved[0].basis == "verdict_correct"


def test_same_evaluator_conflicts_quarantined_without_losing_annotation_provenance():
    df = pd.DataFrame([
        _eval_row(id=1, verdict="incorrect", corrected_label="neutral", comment="first note"),
        _eval_row(id=2, verdict="incorrect", corrected_label="against", comment="second note"),
    ])
    before = df.copy(deep=True)
    resolved = resolve_cohort_a(df, {df.iloc[0].pair_id: StanceLabelV2.IN_FAVOR})
    assert len(resolved) == 2
    assert all(p.resolution_status == "quarantined" for p in resolved)
    assert all(p.quarantine_reason == "conflicting_annotations" for p in resolved)
    pd.testing.assert_frame_equal(df, before)


def test_resolve_cohort_a_threads_post_fix_original_label():
    df = pd.DataFrame([_eval_row(verdict="correct")])
    pair_id = df.iloc[0]["pair_id"]
    established = {pair_id: StanceLabelV2.IN_FAVOR}
    post_fix = {pair_id: StanceLabelV2.AGAINST}

    resolved = resolve_cohort_a(df, established, post_fix)

    assert resolved[0].post_fix_original_label == "against"


def test_resolve_cohort_a_post_fix_defaults_to_none_when_uncovered():
    df = pd.DataFrame([_eval_row(verdict="correct")])
    established = {df.iloc[0]["pair_id"]: StanceLabelV2.IN_FAVOR}

    resolved = resolve_cohort_a(df, established)  # no post_fix argument at all

    assert resolved[0].post_fix_original_label is None


def test_correct_verdict_without_established_original_is_unresolved():
    df = pd.DataFrame([_eval_row(verdict="correct")])

    resolved = resolve_cohort_a(df, established={})

    assert resolved[0].resolution_status == "unresolved"
    assert resolved[0].unresolved_reason == "ambiguous_original_prediction_provenance"


def test_incorrect_with_corrected_label_resolves():
    df = pd.DataFrame([_eval_row(verdict="incorrect", corrected_label="neutral")])
    established = {df.iloc[0]["pair_id"]: StanceLabelV2.IN_FAVOR}

    resolved = resolve_cohort_a(df, established)

    assert resolved[0].resolution_status == "resolved"
    assert resolved[0].reference_label == "neutral"
    assert resolved[0].basis == "verdict_incorrect_corrected"


def test_id_768_style_contradiction_is_quarantined():
    """verdict=incorrect but corrected_label equals the established original."""
    df = pd.DataFrame([_eval_row(id=768, verdict="incorrect", corrected_label="neutral")])
    established = {df.iloc[0]["pair_id"]: StanceLabelV2.NEUTRAL}

    resolved = resolve_cohort_a(df, established)

    assert resolved[0].resolution_status == "quarantined"
    assert resolved[0].quarantine_reason == "contradiction_correction_equals_original"
    assert resolved[0].reference_label is None


def test_incorrect_without_correction_falls_back_to_tier_b():
    df = pd.DataFrame(
        [_eval_row(verdict="incorrect", corrected_label=None, comment="Neutral")]
    )

    resolved = resolve_cohort_a(df, established={})

    assert resolved[0].resolution_status == "resolved"
    assert resolved[0].tier == "B"
    assert resolved[0].reference_label == "neutral"
    assert resolved[0].basis == "tier_b_parsed"
    assert resolved[0].supporting_text == "Neutral"


def test_incorrect_with_ambiguous_note_stays_unresolved():
    df = pd.DataFrame(
        [_eval_row(verdict="incorrect", corrected_label=None, comment="eher neutral")]
    )

    resolved = resolve_cohort_a(df, established={})

    assert resolved[0].resolution_status == "unresolved"
    assert resolved[0].unresolved_reason == "hedge"


def test_unknown_verdict_value_is_unresolved():
    df = pd.DataFrame([_eval_row(verdict="maybe")])

    resolved = resolve_cohort_a(df, established={})

    assert resolved[0].resolution_status == "unresolved"
    assert resolved[0].unresolved_reason == "unknown_verdict_value"


def test_evaluator_agreement_keeps_both_rows_separately():
    rows = [
        _eval_row(id=1, evaluator="Antonie", verdict="correct"),
        _eval_row(id=2, evaluator="Julien", verdict="correct"),
    ]
    df = pd.DataFrame(rows)
    established = {df.iloc[0]["pair_id"]: StanceLabelV2.IN_FAVOR}

    resolved = resolve_cohort_a(df, established)

    assert len(resolved) == 2
    statuses = {p.cohort_role: p.resolution_status for p in resolved}
    assert statuses == {"primary": "resolved", "secondary": "resolved"}
    labels = {p.cohort_role: p.reference_label for p in resolved}
    assert labels == {"primary": "in_favor", "secondary": "in_favor"}


def test_evaluator_disagreement_quarantines_both_rows():
    rows = [
        _eval_row(id=1, evaluator="Antonie", verdict="incorrect", corrected_label="in favor"),
        _eval_row(id=2, evaluator="Julien", verdict="incorrect", corrected_label="ablehnung"),
    ]
    df = pd.DataFrame(rows)
    established = {df.iloc[0]["pair_id"]: StanceLabelV2.NEUTRAL}

    resolved = resolve_cohort_a(df, established)

    assert len(resolved) == 2
    assert all(p.resolution_status == "quarantined" for p in resolved)
    assert all(p.quarantine_reason == "evaluator_disagreement" for p in resolved)


# ---------------------------------------------------------------------------
# Cohort B
# ---------------------------------------------------------------------------


def test_build_cohort_b_loads_and_normalizes(tmp_path):
    records = [
        {
            "row_index": 0,
            "project_id": "proj-a",
            "contribution_id": 10,
            "key_message": "km",
            "comment_text": "text",
            "gold_stance": "ablehnung",
        },
        {
            "row_index": 1,
            "project_id": "proj-a",
            "contribution_id": 11,
            "key_message": "km2",
            "comment_text": "text2",
            "gold_stance": "unknown_value",
        },
    ]
    path = tmp_path / "gold_labels.json"
    path.write_text(json.dumps(records), encoding="utf-8")

    df = build_cohort_b(path)

    assert len(df) == 2
    assert df.iloc[0]["reference_label"] == "against"
    assert df.iloc[0]["resolution_status"] == "resolved"
    assert pd.isna(df.iloc[1]["reference_label"])
    assert df.iloc[1]["resolution_status"] == "unresolved"
    assert set(df["label_source"]) == {"human_gold"}


# ---------------------------------------------------------------------------
# Policy fixtures
# ---------------------------------------------------------------------------


def test_policy_fixtures_are_tagged_policy_derived_not_human_gold():
    fixtures = build_policy_fixtures()

    assert len(fixtures) > 0
    assert all(fx["label_source"] == "policy_derived" for fx in fixtures)


# ---------------------------------------------------------------------------
# End-to-end build_dataset / write_dataset over a small synthetic fixture
# ---------------------------------------------------------------------------


def _make_sqlite_db(path: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        create table stance_evaluations (
            id integer primary key,
            contribution_id integer,
            key_message text,
            comment_text text,
            verdict text,
            comment text,
            evaluator text,
            corrected_label text
        )
        """
    )
    for row in rows:
        conn.execute(
            "insert into stance_evaluations "
            "(id, contribution_id, key_message, comment_text, verdict, comment, evaluator, corrected_label) "
            "values (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["id"],
                row["contribution_id"],
                row["key_message"],
                row["comment_text"],
                row["verdict"],
                row.get("comment"),
                row["evaluator"],
                row.get("corrected_label"),
            ),
        )
    conn.commit()
    conn.close()


def test_build_dataset_end_to_end(tmp_path):
    eval_db = tmp_path / "eval.sq3"
    _make_sqlite_db(
        eval_db,
        [
            dict(id=1, contribution_id=1, key_message="km1", comment_text="c1", verdict="correct", evaluator="Antonie"),
            dict(id=2, contribution_id=1, key_message="km1", comment_text="c2", verdict="incorrect", evaluator="Antonie", corrected_label="neutral"),
            dict(id=3, contribution_id=2, key_message="km2", comment_text="c3", verdict="incorrect", evaluator="Antonie", comment="Neutral"),
        ],
    )

    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    csv_sources = {}
    for name in ("src_a", "src_b"):
        df = pd.DataFrame(
            [
                {"contribution_id": 1, "key_message": "km1", "comment_text": "c1", "old_stance": "in favor"},
                {"contribution_id": 1, "key_message": "km1", "comment_text": "c2", "old_stance": "in favor"},
                {"contribution_id": 2, "key_message": "km2", "comment_text": "c3", "old_stance": "ablehnung"},
            ]
        )
        path = csv_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        csv_sources[name] = path

    gold_path = tmp_path / "gold.json"
    gold_path.write_text(
        json.dumps(
            [
                {
                    "row_index": 0,
                    "project_id": "proj-x",
                    "contribution_id": 99,
                    "key_message": "gkm",
                    "comment_text": "gc",
                    "gold_stance": "neutral",
                }
            ]
        ),
        encoding="utf-8",
    )

    result = build_dataset(eval_db_path=eval_db, csv_sources=csv_sources, gold_labels_path=gold_path)

    assert len(result.pairs_a) == 3
    statuses = result.pairs_a.set_index("source_row_id")["resolution_status"].to_dict()
    assert statuses[1] == "resolved"  # correct, established original = in_favor
    assert statuses[2] == "resolved"  # incorrect w/ corrected_label=neutral, differs from original
    assert statuses[3] == "resolved"  # incorrect, Tier B parsed "Neutral"

    assert len(result.pairs_b) == 1
    assert result.pairs_b.iloc[0]["reference_label"] == "neutral"

    output_dir = tmp_path / "dataset_out"
    write_dataset(result, output_dir)

    assert (output_dir / "pairs.parquet").exists()
    assert (output_dir / "pairs_cohort_b.parquet").exists()
    assert (output_dir / "quarantine.csv").exists()
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "policy_fixtures.json").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["cohort_a"]["row_count"] == 3
    assert manifest["cohort_b"]["row_count"] == 1

    # Raw sources were never modified.
    assert eval_db.exists()
    for path in csv_sources.values():
        assert path.exists()


def test_build_dataset_threads_post_fix_original_label_when_sept8_fix_present(tmp_path):
    eval_db = tmp_path / "eval.sq3"
    _make_sqlite_db(
        eval_db,
        [dict(id=1, contribution_id=1, key_message="km1", comment_text="c1", verdict="correct", evaluator="Antonie")],
    )

    csv_dir = tmp_path / "csvs"
    csv_dir.mkdir()
    df = pd.DataFrame(
        [{"contribution_id": 1, "key_message": "km1", "comment_text": "c1", "old_stance": "in favor", "new_stance": "against"}]
    )
    sept8_path = csv_dir / "sept8_fix.csv"
    df.to_csv(sept8_path, index=False)

    gold_path = tmp_path / "gold.json"
    gold_path.write_text(
        json.dumps(
            [{"row_index": 0, "project_id": "proj-x", "contribution_id": 99, "key_message": "gkm", "comment_text": "gc", "gold_stance": "neutral"}]
        ),
        encoding="utf-8",
    )

    result = build_dataset(
        eval_db_path=eval_db, csv_sources={"sept8_fix": sept8_path}, gold_labels_path=gold_path
    )

    assert result.pairs_a.iloc[0]["established_original_label"] == "in_favor"
    assert result.pairs_a.iloc[0]["post_fix_original_label"] == "against"
    assert result.manifest["cohort_a"]["post_fix_coverage"]["resolved_rows_with_post_fix_label"] == 1
