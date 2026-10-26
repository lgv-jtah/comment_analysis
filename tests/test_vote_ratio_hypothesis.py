"""Tests for vote_ratio_hypothesis (synthetic data only).

These tests never touch the real API, the real dataset/pairs.parquet, or the
real evaluation database. Every input is built in tmp_path: synthetic parquet
fixtures, a temporary SQLite evaluation database and a monkeypatched
``requests.get`` stand-in (same convention as test_create_dataset.py). No real
favor or vote ratio is ever computed.
"""
import json
import re
import sqlite3

import numpy as np
import pandas as pd
import pytest
import requests

import vote_ratio_hypothesis as v
from dataset_builder import compute_pair_id
from reaction_split_review import comment_id_for


# --------------------------------------------------------------------------
# Fake HTTP layer
# --------------------------------------------------------------------------


class FakeResponse:
    """Minimal requests.Response stand-in (mirrors test_create_dataset.py)."""

    def __init__(self, json_data=None, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def _feature(contribution_id, pro, contra, *, comments=0, date="2026-01-01"):
    """Build one GeoJSON-style contribution feature with vote counts."""
    return {"properties": {"id": contribution_id, "votingPro": pro, "votingContra": contra,
                           "commentsNumber": comments, "dateCreated": date}}


def _make_get(project_payloads):
    """Return a fake `get` that dispatches by the project id in the URL.

    Args:
        project_payloads: Maps project id to either a list of features or an
            Exception/HTTPError status to simulate.
    """
    def fake_get(url, timeout=None):
        match = re.search(r"/projects/([^/]+)/contributions/", url)
        project_id = match.group(1)
        payload = project_payloads[project_id]
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, int):
            return FakeResponse(status_code=payload)
        return FakeResponse(json_data={"features": payload})

    return fake_get


# --------------------------------------------------------------------------
# Synthetic input fixtures
# --------------------------------------------------------------------------


def _write_pairs(tmp_path, records):
    """Write a synthetic pairs.parquet from record dicts.

    Each record: contribution_id (int), key_message, comment_text,
    reference_label, and optional cohort_role/tier/resolution_status.
    """
    rows = []
    for rec in records:
        cid = rec["contribution_id"]
        rows.append({
            "pair_id": compute_pair_id(str(cid), rec["key_message"], rec["comment_text"]),
            "contribution_id": cid,
            "key_message": rec["key_message"],
            "comment_text": rec["comment_text"],
            "cohort_role": rec.get("cohort_role", "primary"),
            "tier": rec.get("tier", "A"),
            "resolution_status": rec.get("resolution_status", "resolved"),
            "reference_label": rec["reference_label"],
        })
    path = tmp_path / "pairs.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _write_contributions(tmp_path, mapping_rows):
    """Write a synthetic contributions.parquet from (project_id, contribution_id) rows."""
    rows = [{"project_id": pid, "contribution_id": str(cid), "title": "t",
             "contributionContent": "c", "category": "k"} for pid, cid in mapping_rows]
    path = tmp_path / "contributions.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _write_eval_db(tmp_path, records):
    """Write a temporary SQLite stance_evaluations database.

    Each record may carry a free-text ``note`` (the annotator note matched by
    the generic-approval matchers).
    """
    path = tmp_path / "eval.sq3"
    conn = sqlite3.connect(str(path))
    conn.execute(
        "create table stance_evaluations (id integer primary key, contribution_id text, "
        "key_message text, comment_text text, verdict text, comment text, evaluator text, "
        "corrected_label text)"
    )
    for i, rec in enumerate(records, start=1):
        conn.execute(
            "insert into stance_evaluations values (?,?,?,?,?,?,?,?)",
            (i, str(rec["contribution_id"]), rec["key_message"], rec["comment_text"],
             "correct", rec.get("note"), rec.get("evaluator", "Antonie"), None),
        )
    conn.commit()
    conn.close()
    return path


# A small, fully synthetic world used across several tests.
# Contribution 1: two key messages for one comment -> favor + against = mixed.
# Contribution 2: one favor comment.
# Contribution 3: one against comment (its project returns no feature -> missing).
BASE_RECORDS = [
    {"contribution_id": 1, "key_message": "km1", "comment_text": "c1 text",
     "reference_label": "in_favor"},
    {"contribution_id": 1, "key_message": "km2", "comment_text": "c1 text",
     "reference_label": "against"},
    {"contribution_id": 2, "key_message": "km1", "comment_text": "c2 text",
     "reference_label": "in_favor", "note": "generell zustimmend"},
    {"contribution_id": 3, "key_message": "km1", "comment_text": "c3 text",
     "reference_label": "against"},
]
BASE_CONTRIB_ROWS = [("pA", 1), ("pA", 2), ("pB", 3)]


# --------------------------------------------------------------------------
# 2. gold_contribution_projects
# --------------------------------------------------------------------------


def test_gold_contribution_projects_flags_unmapped_and_collision(tmp_path):
    records = BASE_RECORDS + [
        {"contribution_id": 4, "key_message": "km1", "comment_text": "c4", "reference_label": "neutral"},
        {"contribution_id": 5, "key_message": "km1", "comment_text": "c5", "reference_label": "neutral"},
    ]
    pairs = _write_pairs(tmp_path, records)
    # 4 appears in two projects (collision); 5 appears in none (unmapped).
    contributions = _write_contributions(tmp_path, BASE_CONTRIB_ROWS + [("pA", 4), ("pB", 4)])
    mapping = v.gold_contribution_projects(pairs, contributions)
    by_id = mapping.set_index("contribution_id")
    assert by_id.loc["1", "n_project_candidates"] == 1
    assert by_id.loc["1", "project_id"] == "pA"
    assert by_id.loc["4", "n_project_candidates"] == 2
    assert pd.isna(by_id.loc["4", "project_id"])
    assert by_id.loc["5", "n_project_candidates"] == 0
    assert pd.isna(by_id.loc["5", "project_id"])


# --------------------------------------------------------------------------
# 1. fetch_votes
# --------------------------------------------------------------------------


def test_fetch_votes_filters_non_gold_and_records_errors(tmp_path):
    targets = pd.DataFrame([
        {"contribution_id": "1", "project_id": "pA"},
        {"contribution_id": "2", "project_id": "pA"},
        {"contribution_id": "3", "project_id": "pB"},
    ])
    payloads = {
        # pA also returns a non-gold feature (99) that must be filtered out,
        # plus contribution 2 with missing vote fields (coerced to <NA>).
        "pA": [_feature("1", 7, 3), _feature("99", 100, 100),
               {"properties": {"id": "2", "dateCreated": "2026-01-02"}}],
        # pB raises a network error -> recorded, not fatal.
        "pB": requests.ConnectionError("boom"),
    }
    votes, status = v.fetch_votes(targets, get=_make_get(payloads))
    assert set(votes.contribution_id) == {"1", "2"}  # 99 filtered, 3 unreachable
    assert status["pA"] == {"status": "ok", "features_returned": 3, "gold_matched": 2}
    assert status["pB"]["status"].startswith("error:")
    # Missing vote fields on contribution 2 are nullable <NA>, not a crash.
    row2 = votes.set_index("contribution_id").loc["2"]
    assert pd.isna(row2.votingPro) and pd.isna(row2.votingContra)
    assert str(votes.votingPro.dtype) == "Int64"


def test_fetch_votes_does_not_swallow_json_errors(tmp_path):
    targets = pd.DataFrame([{"contribution_id": "1", "project_id": "pA"}])

    def bad_get(url, timeout=None):
        class R:
            def raise_for_status(self):
                pass

            def json(self):
                raise ValueError("not json")

        return R()

    with pytest.raises(ValueError):
        v.fetch_votes(targets, get=bad_get)


# --------------------------------------------------------------------------
# 3. run_vote_fetch
# --------------------------------------------------------------------------


def _fetch_payloads():
    return {
        "pA": [_feature("1", 7, 3), _feature("2", 4, 4)],
        # pB returns features but not contribution 3 -> 3 is missing from API.
        "pB": [_feature("999", 1, 1)],
    }


def test_run_vote_fetch_writes_artifacts_hashes_last_and_refuses_overwrite(tmp_path):
    pairs = _write_pairs(tmp_path, BASE_RECORDS)
    contributions = _write_contributions(tmp_path, BASE_CONTRIB_ROWS)
    out = tmp_path / "votes"
    result = v.run_vote_fetch(pairs_path=pairs, contributions_path=contributions,
                              output_dir=out, get=_make_get(_fetch_payloads()))
    # All three files exist, hashes file present and complete.
    assert (out / v.SNAPSHOT_FILE).exists()
    assert (out / v.FETCH_SUMMARY_FILE).exists()
    hashes = json.loads((out / "artifact_hashes.json").read_text())
    assert hashes["complete"] is True
    assert set(hashes["files"]) == {v.SNAPSHOT_FILE, v.FETCH_SUMMARY_FILE}
    # Contribution 3 maps to pB but was not returned -> missing from API.
    assert result.summary["contributions_missing_from_api"]["ids"] == ["3"]
    assert result.summary["contributions_fetched"] == 2
    assert result.summary["network_calls"] == 2
    # Re-running into the same directory must refuse to overwrite.
    with pytest.raises(FileExistsError):
        v.run_vote_fetch(pairs_path=pairs, contributions_path=contributions,
                         output_dir=out, get=_make_get(_fetch_payloads()))


# --------------------------------------------------------------------------
# 4. pair_label_set / rollup
# --------------------------------------------------------------------------


def test_pair_label_set_excludes_nan_tier_and_secondary(tmp_path):
    records = BASE_RECORDS + [
        {"contribution_id": 10, "key_message": "km1", "comment_text": "secondary",
         "reference_label": "in_favor", "cohort_role": "secondary"},
        {"contribution_id": 11, "key_message": "km1", "comment_text": "nan tier",
         "reference_label": "in_favor", "tier": None},
    ]
    pairs = _write_pairs(tmp_path, records)
    labels = v.pair_label_set(pairs)
    assert "10" not in set(labels.contribution_id)  # secondary dropped
    assert "11" not in set(labels.contribution_id)  # NaN tier dropped
    # comment_id matches the pipeline helper exactly.
    row = labels[labels.contribution_id == "2"].iloc[0]
    assert row.comment_id == comment_id_for("2", "c2 text")


def test_rollup_comments_all_categories(tmp_path):
    records = [
        {"contribution_id": 1, "key_message": "km1", "comment_text": "mix", "reference_label": "in_favor"},
        {"contribution_id": 1, "key_message": "km2", "comment_text": "mix", "reference_label": "against"},
        {"contribution_id": 2, "key_message": "km1", "comment_text": "fav", "reference_label": "in_favor"},
        {"contribution_id": 3, "key_message": "km1", "comment_text": "ag", "reference_label": "against"},
        {"contribution_id": 4, "key_message": "km1", "comment_text": "neu", "reference_label": "neutral"},
    ]
    pairs = _write_pairs(tmp_path, records)
    labels = v.pair_label_set(pairs)
    rolled = v.rollup_comments(labels).set_index("comment_id").category
    cats = set(rolled)
    assert cats == {"mixed", "in_favor", "against", "neutral"}
    assert rolled[comment_id_for("1", "mix")] == "mixed"


def test_rollup_approval_override(tmp_path):
    records = [
        {"contribution_id": 2, "key_message": "km1", "comment_text": "fav", "reference_label": "in_favor"},
    ]
    pairs = _write_pairs(tmp_path, records)
    labels = v.pair_label_set(pairs)
    flagged = {compute_pair_id("2", "km1", "fav")}
    assert v.rollup_comments(labels, approval_as="neutral",
                             generic_approval_pair_ids=flagged).category.iloc[0] == "neutral"
    assert v.rollup_comments(labels, approval_as="in_favor",
                             generic_approval_pair_ids=flagged).category.iloc[0] == "in_favor"
    # Without the override (primary) the label stays as-is.
    assert v.rollup_comments(labels).category.iloc[0] == "in_favor"


def test_rollup_override_is_per_pair_not_wholesale(tmp_path):
    # One comment with two pairs: a flagged generic-approval in_favor pair and
    # a substantive against pair. Overriding only the flagged pair to neutral
    # must leave the comment "against" (a wholesale comment override would have
    # forced "neutral").
    records = [
        {"contribution_id": 1, "key_message": "km_approval", "comment_text": "same",
         "reference_label": "in_favor"},
        {"contribution_id": 1, "key_message": "km_substance", "comment_text": "same",
         "reference_label": "against"},
    ]
    pairs = _write_pairs(tmp_path, records)
    labels = v.pair_label_set(pairs)
    # Only the generic-approval pair (km_approval) is flagged.
    flagged = {compute_pair_id("1", "km_approval", "same")}
    rolled = v.rollup_comments(labels, approval_as="neutral", generic_approval_pair_ids=flagged)
    assert rolled.category.iloc[0] == "against"
    # Sanity: without the override the comment is mixed, and a wholesale
    # (every-pair) override would have made it neutral -- both differ from the
    # correct per-pair result above.
    assert v.rollup_comments(labels).category.iloc[0] == "mixed"
    both_flagged = {compute_pair_id("1", "km_approval", "same"),
                    compute_pair_id("1", "km_substance", "same")}
    assert v.rollup_comments(labels, approval_as="neutral",
                             generic_approval_pair_ids=both_flagged).category.iloc[0] == "neutral"


# --------------------------------------------------------------------------
# 5. favor_share, approval sensitivity, broad >= strict
# --------------------------------------------------------------------------


def test_favor_share_exclude_vs_half():
    cats = ["in_favor", "against", "mixed", "neutral"]
    excl = v.favor_share(cats, mixed_mode="exclude")
    assert excl["favor_share"] == pytest.approx(50.0)  # 1 of 2 directional
    assert excl["mixed"] == 1 and excl["neutral"] == 1
    half = v.favor_share(cats, mixed_mode="half")
    # favor 1 + 0.5 mixed = 1.5 of (1+1+1)=3 -> 50%
    assert half["favor_share"] == pytest.approx(50.0)
    # neutral share is over all four categories.
    assert excl["neutral_share"] == pytest.approx(25.0)
    assert v.favor_share([], mixed_mode="exclude")["favor_share"] is None


def test_generic_approval_strict_and_broad(tmp_path):
    records = [
        {"contribution_id": 1, "key_message": "km1", "comment_text": "a", "reference_label": "in_favor"},
        {"contribution_id": 2, "key_message": "km1", "comment_text": "b", "reference_label": "in_favor"},
    ]
    pairs = _write_pairs(tmp_path, records)
    # Note 1 matches only the broad matcher (terms in reverse order, no
    # "...zustimmend"); note 2 matches the strict matcher.
    eval_db = _write_eval_db(tmp_path, [
        {"contribution_id": 1, "key_message": "km1", "comment_text": "a",
         "note": "zustimmende generelle Reaktion"},
        {"contribution_id": 2, "key_message": "km1", "comment_text": "b",
         "note": "generell zustimmend"},
    ])
    strict = v.generic_approval_pair_ids(pairs, eval_db, matcher="strict")
    broad = v.generic_approval_pair_ids(pairs, eval_db, matcher="broad")
    # The function now returns flagged PAIR ids, restricted to the gold set.
    assert strict == {compute_pair_id("2", "km1", "b")}
    assert strict <= broad  # broad always a superset
    assert broad == {compute_pair_id("1", "km1", "a"), compute_pair_id("2", "km1", "b")}


def test_generic_approval_missing_db_raises(tmp_path):
    pairs = _write_pairs(tmp_path, BASE_RECORDS)
    with pytest.raises(FileNotFoundError):
        v.generic_approval_pair_ids(pairs, tmp_path / "nope.sq3")


# --------------------------------------------------------------------------
# 6. vote_share
# --------------------------------------------------------------------------


def test_vote_share_pooled_zero_and_missing():
    votes = pd.DataFrame({
        "contribution_id": ["1", "2", "3"],
        "votingPro": pd.array([7, 0, 5], dtype="Int64"),
        "votingContra": pd.array([3, 0, pd.NA], dtype="Int64"),
    })
    # Pooled over 1 and 2 only (3 excluded from the fixed set here).
    out = v.vote_share(votes, ["1", "2"])
    assert out["vote_pro"] == 7 and out["vote_contra"] == 3
    assert out["vote_share"] == pytest.approx(70.0)
    # A zero-vote-only set has an undefined share.
    assert v.vote_share(votes, ["2"])["vote_share"] is None
    # Missing contra counts as 0.
    assert v.vote_share(votes, ["3"])["vote_contra"] == 0


# --------------------------------------------------------------------------
# 7. bootstrap / TOST
# --------------------------------------------------------------------------


def _frame(favor_num, favor_den, pro, contra):
    return pd.DataFrame({"favor_num": favor_num, "favor_den": favor_den,
                         "vote_pro": pro, "vote_contra": contra})


def test_bootstrap_reproducible_and_equivalent():
    # Favor share ~ vote share on every contribution -> difference ~ 0.
    frame = _frame([6, 7, 5, 8], [10, 10, 10, 10], [60, 70, 50, 80], [40, 30, 50, 20])
    diffs1, dropped1 = v.bootstrap_difference(frame, n=500, seed=1, estimator=v._pooled_difference)
    diffs2, dropped2 = v.bootstrap_difference(frame, n=500, seed=1, estimator=v._pooled_difference)
    assert np.array_equal(diffs1, diffs2)  # same seed -> identical draws
    assert dropped1 == 0 and dropped2 == 0
    verdict = v.tost_equivalence(diffs1, margin=v.EQUIVALENCE_MARGIN_PP, level=0.90)
    assert verdict["equivalent"] is True


def test_bootstrap_clear_non_equivalence():
    # Favor always 100%, votes always 0% pro -> difference ~ 100 pp.
    frame = _frame([10, 10], [10, 10], [0, 0], [10, 10])
    diffs, _ = v.bootstrap_difference(frame, n=300, seed=2, estimator=v._pooled_difference)
    verdict = v.tost_equivalence(diffs, margin=v.EQUIVALENCE_MARGIN_PP, level=0.90)
    assert verdict["equivalent"] is False
    assert verdict["low"] == pytest.approx(100.0)


def test_bootstrap_zero_denominator_resamples_dropped():
    # No directional comments anywhere -> every resample is undefined.
    frame = _frame([0.0, 0.0], [0.0, 0.0], [5, 5], [5, 5])
    diffs, dropped = v.bootstrap_difference(frame, n=100, seed=3, estimator=v._pooled_difference)
    assert diffs.size == 0 and dropped == 100
    verdict = v.tost_equivalence(diffs, margin=v.EQUIVALENCE_MARGIN_PP, level=0.90)
    assert verdict == {"low": None, "high": None, "equivalent": False}


# --------------------------------------------------------------------------
# 8. run_vote_analysis end to end
# --------------------------------------------------------------------------


@pytest.fixture
def analysis(tmp_path):
    pairs = _write_pairs(tmp_path, BASE_RECORDS)
    contributions = _write_contributions(tmp_path, BASE_CONTRIB_ROWS)
    eval_db = _write_eval_db(tmp_path, [
        {"contribution_id": r["contribution_id"], "key_message": r["key_message"],
         "comment_text": r["comment_text"], "note": r.get("note")} for r in BASE_RECORDS
    ])
    fetch = v.run_vote_fetch(pairs_path=pairs, contributions_path=contributions,
                             output_dir=tmp_path / "votes", get=_make_get(_fetch_payloads()))
    result = v.run_vote_analysis(snapshot_dir=fetch.output_dir, pairs_path=pairs,
                                 contributions_path=contributions, eval_db_path=eval_db,
                                 output_dir=tmp_path / "analysis", n_bootstrap=200, seed=5)
    return result, pairs, contributions, eval_db, tmp_path


def test_run_vote_analysis_structure_and_fixed_vote_side(analysis):
    result, *_ = analysis
    summary = result.summary
    # Required top-level keys.
    for key in ("schema_version", "counts", "primary", "sensitivity", "bootstrap",
                "margin_pp", "interpretation_rules", "vote_side_rule", "representativeness_limit",
                "vote_snapshot"):
        assert key in summary
    primary = summary["primary"]
    for key in ("favor", "against", "neutral", "mixed", "favor_share", "neutral_share",
                "vote_pro", "vote_contra", "vote_share", "difference_pp", "ci90", "ci95"):
        assert key in primary
    # Fixed set = {1, 2} (3 is missing from the API snapshot).
    assert summary["counts"]["n_contributions"] == 2
    assert summary["counts"]["n_contributions_missing_votes"]["ids"] == ["3"]
    # The vote side is identical across the primary and every sensitivity.
    for block in summary["sensitivity"].values():
        assert block["vote_pro"] == primary["vote_pro"]
        assert block["vote_contra"] == primary["vote_contra"]
    # Pooled votes over contributions 1 (7/3) and 2 (4/4) = 11 pro, 7 contra.
    assert primary["vote_pro"] == 11 and primary["vote_contra"] == 7
    # Generic-approval strict flagged contribution 2's single pair/comment;
    # both the pair and the comment counts are reported with matching names.
    assert summary["counts"]["generic_approval_pairs_strict"] == 1
    assert summary["counts"]["generic_approval_comments_strict"] == 1
    assert "generic_approval_pairs_broad" in summary["counts"]
    assert "generic_approval_comments_broad" in summary["counts"]


def test_run_vote_analysis_report_and_no_overwrite(analysis):
    result, pairs, contributions, eval_db, tmp_path = analysis
    report = (result.output_dir / v.REPORT_FILE).read_text()
    assert "Vote-ratio equivalence test: report" in report
    assert v.INTERPRETATION_RULES in report
    assert v.REPRESENTATIVENESS_LIMIT in report
    assert "(a) Pair level" in report and "(d) Mixed comments counted as a half" in report
    assert "90% (primary)" in report
    # Report labels distinguish flagged pairs from flagged comments.
    assert "Generic-approval pairs (strict)" in report
    assert "Generic-approval comments (strict)" in report
    hashes = json.loads((result.output_dir / "artifact_hashes.json").read_text())
    assert set(hashes["files"]) == {v.ANALYSIS_SUMMARY_FILE, v.REPORT_FILE}
    # Re-running into the same directory refuses to overwrite.
    with pytest.raises(FileExistsError):
        v.run_vote_analysis(snapshot_dir=result.output_dir.parent / "votes", pairs_path=pairs,
                            contributions_path=contributions, eval_db_path=eval_db,
                            output_dir=result.output_dir, n_bootstrap=10, seed=5)


# --------------------------------------------------------------------------
# 9. CLI wiring
# --------------------------------------------------------------------------


def test_cli_vote_ratio_fetch_and_analyse(tmp_path, monkeypatch):
    import cli

    pairs = _write_pairs(tmp_path, BASE_RECORDS)
    contributions = _write_contributions(tmp_path, BASE_CONTRIB_ROWS)
    eval_db = _write_eval_db(tmp_path, [
        {"contribution_id": r["contribution_id"], "key_message": r["key_message"],
         "comment_text": r["comment_text"], "note": r.get("note")} for r in BASE_RECORDS
    ])
    # Route the fetch's network call through the fake get.
    monkeypatch.setattr("vote_ratio_hypothesis.requests.get", _make_get(_fetch_payloads()))
    votes_dir = tmp_path / "votes"
    cli.main(["vote-ratio-fetch", "--pairs", str(pairs), "--contributions", str(contributions),
              "--output-dir", str(votes_dir)])
    assert (votes_dir / v.SNAPSHOT_FILE).exists()

    analysis_dir = tmp_path / "analysis"
    cli.main(["vote-ratio-analyse", "--snapshot-dir", str(votes_dir), "--pairs", str(pairs),
              "--contributions", str(contributions), "--eval-db", str(eval_db),
              "--output-dir", str(analysis_dir), "--n-bootstrap", "20", "--seed", "5"])
    assert (analysis_dir / v.ANALYSIS_SUMMARY_FILE).exists()
    assert (analysis_dir / v.REPORT_FILE).exists()
