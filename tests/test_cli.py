import json
from pathlib import Path

import pandas as pd
import pytest
from types import SimpleNamespace

import cli
from dataset_builder import AuditResult


def test_evaluate_reports_all_expected_operational_pairs_separately_from_oof(tmp_path, monkeypatch):
    reference = pd.DataFrame([
        dict(pair_id="p1", contribution_id="c1", resolution_status="resolved", reference_label="neutral"),
        dict(pair_id="p2", contribution_id="c2", resolution_status="resolved", reference_label="neutral"),
        dict(pair_id="p3", contribution_id="c3", resolution_status="unresolved", reference_label=None),
    ])
    preds = pd.DataFrame([dict(pair_id="p1", status="accepted", proposed_stance="neutral", final_stance="neutral")])
    ref_path, pred_path = tmp_path / "ref.parquet", tmp_path / "pred.parquet"
    reference.to_parquet(ref_path)
    preds.to_parquet(pred_path)
    folds_path = tmp_path / "folds.json"
    folds_path.write_text("{}")
    folds = SimpleNamespace(folds=[dict(test_contribution_ids=["c1", "c2"])])
    monkeypatch.setattr(cli, "load_folds", lambda path: folds)
    output = tmp_path / "metrics.json"
    command = ["evaluate", "--reference", str(ref_path), "--predictions", str(pred_path),
               "--folds", str(folds_path), "--output", str(output)]
    cli.main(command)
    report = json.loads(output.read_text())
    assert report["metrics"]["total_pairs"] == 3
    assert report["metrics"]["operational_coverage"] == pytest.approx(1 / 3)
    assert report["pooled_out_of_fold"]["total_pairs"] == 2
    assert report["unassigned_expected_pairs"] == 1
    folds.folds[0]["test_contribution_ids"] = ["c1"]
    with pytest.raises(ValueError, match="Missing fold assignments"):
        cli.main(command)


def _fake_audit_result() -> AuditResult:
    pairs_a = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "resolution_status": "resolved",
                "tier": "A",
            }
        ]
    )
    pairs_b = pd.DataFrame([{"pair_id": "g1", "resolution_status": "resolved"}])
    quarantine = pairs_a.iloc[0:0].copy()
    manifest = {
        "cohort_a": {
            "row_count": 1,
            "primary_rows": 1,
            "secondary_rows": 0,
            "resolution_status_counts": {"resolved": 1, "unresolved": 0, "quarantined": 0},
            "tier_counts": {"A": 1},
            "quarantine_reasons": {},
            "unresolved_reasons": {},
        },
        "cohort_b": {
            "row_count": 1,
            "resolved": 1,
            "unresolved": 0,
            "distinct_projects": ["proj-a"],
        },
    }
    return AuditResult(pairs_a, pairs_b, quarantine, manifest, policy_fixtures=[])


def test_audit_command_invokes_build_and_write_dataset(monkeypatch, tmp_path, capsys):
    calls = {}

    def fake_build_dataset():
        calls["built"] = True
        return _fake_audit_result()

    def fake_write_dataset(result, output_dir):
        calls["written_to"] = output_dir

    monkeypatch.setattr(cli, "build_dataset", fake_build_dataset)
    monkeypatch.setattr(cli, "write_dataset", fake_write_dataset)

    output_dir = tmp_path / "dataset_out"
    cli.main(["audit", "--output-dir", str(output_dir)])

    assert calls["built"] is True
    assert calls["written_to"] == output_dir

    out = capsys.readouterr().out
    assert "Cohort A" in out
    assert "Wrote dataset artifacts" in out


def test_no_subcommand_raises_usage_error():
    with pytest.raises(SystemExit):
        cli.main([])


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------


def _write_reference(tmp_path) -> Path:
    reference = pd.DataFrame(
        [
            {"pair_id": "p1", "contribution_id": "1", "resolution_status": "resolved", "reference_label": "in_favor"},
            {"pair_id": "p2", "contribution_id": "1", "resolution_status": "resolved", "reference_label": "neutral"},
        ]
    )
    path = tmp_path / "pairs.parquet"
    reference.to_parquet(path)
    return path


def _write_predictions(tmp_path) -> Path:
    predictions = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "contribution_id": "1",
                "status": "accepted",
                "proposed_stance": "in_favor",
                "final_stance": "in_favor",
            },
            {
                "pair_id": "p2",
                "contribution_id": "1",
                "status": "accepted",
                "proposed_stance": "in_favor",
                "final_stance": "in_favor",
            },
        ]
    )
    path = tmp_path / "predictions.parquet"
    predictions.to_parquet(path)
    return path


def test_evaluate_command_without_folds_prints_overall_metrics(tmp_path, capsys):
    reference_path = _write_reference(tmp_path)
    predictions_path = _write_predictions(tmp_path)

    cli.main(
        [
            "evaluate",
            "--predictions",
            str(predictions_path),
            "--reference",
            str(reference_path),
            "--folds",
            str(tmp_path / "no_such_folds.json"),
        ]
    )

    out = capsys.readouterr().out
    assert "Overall" in out
    assert "Majority-class baseline" in out


def test_evaluate_command_writes_json_output(tmp_path):
    reference_path = _write_reference(tmp_path)
    predictions_path = _write_predictions(tmp_path)
    output_path = tmp_path / "metrics.json"

    cli.main(
        [
            "evaluate",
            "--predictions",
            str(predictions_path),
            "--reference",
            str(reference_path),
            "--folds",
            str(tmp_path / "no_such_folds.json"),
            "--output",
            str(output_path),
        ]
    )

    assert output_path.exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert "metrics" in data
    assert data["metrics"]["eligible_pairs"] == 2
    assert data["accepted_accuracy_by_basis"] == {}


def test_evaluate_command_json_output_includes_basis_breakdown(tmp_path):
    reference = pd.DataFrame(
        [
            {"pair_id": "p1", "contribution_id": "1", "resolution_status": "resolved", "reference_label": "neutral"},
        ]
    )
    reference_path = tmp_path / "pairs.parquet"
    reference.to_parquet(reference_path)

    predictions = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "contribution_id": "1",
                "status": "accepted",
                "proposed_stance": "in_favor",
                "final_stance": "in_favor",
                "basis": "general_approval",
            },
        ]
    )
    predictions_path = tmp_path / "predictions.parquet"
    predictions.to_parquet(predictions_path)
    output_path = tmp_path / "metrics.json"

    cli.main(
        [
            "evaluate",
            "--predictions",
            str(predictions_path),
            "--reference",
            str(reference_path),
            "--folds",
            str(tmp_path / "no_such_folds.json"),
            "--output",
            str(output_path),
        ]
    )

    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["accepted_accuracy_by_basis"] == {
        "general_approval": {"n": 1, "correct": 0, "accuracy": 0.0}
    }


def test_evaluate_command_prints_basis_breakdown_when_basis_column_present(tmp_path, capsys):
    reference = pd.DataFrame(
        [
            {"pair_id": "p1", "contribution_id": "1", "resolution_status": "resolved", "reference_label": "neutral"},
        ]
    )
    reference_path = tmp_path / "pairs.parquet"
    reference.to_parquet(reference_path)

    predictions = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "contribution_id": "1",
                "status": "accepted",
                "proposed_stance": "in_favor",
                "final_stance": "in_favor",
                "basis": "general_approval",
            },
        ]
    )
    predictions_path = tmp_path / "predictions.parquet"
    predictions.to_parquet(predictions_path)

    cli.main(
        [
            "evaluate",
            "--predictions",
            str(predictions_path),
            "--reference",
            str(reference_path),
            "--folds",
            str(tmp_path / "no_such_folds.json"),
        ]
    )

    out = capsys.readouterr().out
    assert "Accepted accuracy by basis" in out
    assert "general_approval" in out


def test_evaluate_command_with_folds_prints_per_fold_and_pooled(tmp_path, capsys):
    reference_path = _write_reference(tmp_path)
    predictions_path = _write_predictions(tmp_path)
    folds_path = tmp_path / "folds_v1.json"
    folds_path.write_text(
        json.dumps(
            {
                "seed": 1,
                "n_splits": 1,
                "grouping_key": "contribution_id",
                "near_duplicate_check": "skipped_no_contribution_text_available",
                "dup_groups_merged": 0,
                "folds": [{"train_contribution_ids": [], "test_contribution_ids": ["1"]}],
                "schema_version": "folds_v1",
            }
        ),
        encoding="utf-8",
    )

    cli.main(
        [
            "evaluate",
            "--predictions",
            str(predictions_path),
            "--reference",
            str(reference_path),
            "--folds",
            str(folds_path),
        ]
    )

    out = capsys.readouterr().out
    assert "Fold 0" in out
    assert "Pooled out-of-fold" in out


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


def test_benchmark_command_invokes_run_benchmark_and_writes_predictions(monkeypatch, tmp_path, capsys):
    source_path = tmp_path / "source.parquet"
    pd.DataFrame(
        [
            {
                "contribution_id": 1,
                "contributionContent": "Beitrag",
                "key_message": "km1",
                "key_message_type": "Wunsch",
                "related_sentences": ["s1"],
                "comment_text": "c1",
            }
        ]
    ).to_parquet(source_path)

    calls = {}

    async def fake_run_benchmark(**kwargs):
        calls["kwargs"] = kwargs
        frame = pd.DataFrame(
            [
                {
                    "contribution_id": "1",
                    "contributionContent": "Beitrag",
                    "key_message_id": "1:km1",
                    "key_message": "km1",
                    "comment_text": "c1",
                    "pair_id": "abc",
                    "status": "accepted",
                    "proposed_stance": "in_favor",
                    "final_stance": "in_favor",
                    "basis": "general_approval",
                    "comment_evidence": [],
                    "source_evidence": [],
                    "explanation": "",
                    "review_reasons": [],
                    "general_reaction": None,
                    "general_reaction_polarity": None,
                    "reaction_evidence": [],
                    "rest_text": None,
                    "policy_version": "v2",
                    "run_id": "run-1",
                    "extraction_model_name": None,
                    "stance_model_name": "test-model",
                }
            ]
        )
        checkpoint = kwargs["checkpoint"]
        frame["run_id"] = kwargs["run_id"]
        frame["pair_id"] = checkpoint.expected.pair_id.tolist()
        checkpoint.commit_comment(frame.to_dict("records"))
        return frame

    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)

    output_path = tmp_path / "predictions.parquet"
    reference_path = tmp_path / "nonexistent_reference.parquet"

    cli.main(
        [
            "benchmark",
            "--source",
            str(source_path),
            "--output",
            str(output_path),
            "--reference",
            str(reference_path),
            "--model-name",
            "test-model",
            "--base-url",
            "http://testserver",
            "--no-verify-ionos",
        ]
    )

    assert output_path.exists()
    assert calls["kwargs"]["verify_ionos"] is False
    assert calls["kwargs"]["model_name"] == "test-model"

    out = capsys.readouterr().out
    assert "Wrote 1 predictions" in out
    assert "Reference file not found" in out


def test_benchmark_command_scores_against_existing_reference_and_folds(monkeypatch, tmp_path, capsys):
    source_path = tmp_path / "source.parquet"
    pd.DataFrame(
        [
            {
                "contribution_id": 1,
                "contributionContent": "Beitrag",
                "key_message": "km1",
                "key_message_type": "Wunsch",
                "related_sentences": ["s1"],
                "comment_text": "c1",
            }
        ]
    ).to_parquet(source_path)

    reference_path = tmp_path / "pairs.parquet"
    pd.DataFrame(
        [{"pair_id": "abc", "contribution_id": "1", "resolution_status": "resolved", "reference_label": "in_favor"}]
    ).to_parquet(reference_path)

    folds_path = tmp_path / "folds_v1.json"
    folds_path.write_text(
        json.dumps(
            {
                "seed": 1,
                "n_splits": 1,
                "grouping_key": "contribution_id",
                "near_duplicate_check": "skipped_no_contribution_text_available",
                "dup_groups_merged": 0,
                "folds": [{"train_contribution_ids": [], "test_contribution_ids": ["1"]}],
                "schema_version": "folds_v1",
            }
        ),
        encoding="utf-8",
    )

    async def fake_run_benchmark(**kwargs):
        frame = pd.DataFrame(
            [
                {
                    "contribution_id": "1",
                    "contributionContent": "Beitrag",
                    "key_message_id": "1:km1",
                    "key_message": "km1",
                    "comment_text": "c1",
                    "pair_id": "abc",
                    "status": "accepted",
                    "proposed_stance": "in_favor",
                    "final_stance": "in_favor",
                    "basis": "general_approval",
                    "comment_evidence": [],
                    "source_evidence": [],
                    "explanation": "",
                    "review_reasons": [],
                    "general_reaction": None,
                    "general_reaction_polarity": None,
                    "reaction_evidence": [],
                    "rest_text": None,
                    "policy_version": "v2",
                    "run_id": "run-1",
                    "extraction_model_name": None,
                    "stance_model_name": "test-model",
                }
            ]
        )
        checkpoint = kwargs["checkpoint"]
        frame["run_id"] = kwargs["run_id"]
        frame["pair_id"] = checkpoint.expected.pair_id.tolist()
        checkpoint.commit_comment(frame.to_dict("records"))
        return frame

    monkeypatch.setattr(cli, "run_benchmark", fake_run_benchmark)

    cli.main(
        [
            "benchmark",
            "--source",
            str(source_path),
            "--output",
            str(tmp_path / "predictions.parquet"),
            "--reference",
            str(reference_path),
            "--folds",
            str(folds_path),
            "--model-name",
            "test-model",
            "--base-url",
            "http://testserver",
            "--no-verify-ionos",
            "--limit",
            "5",
        ]
    )

    out = capsys.readouterr().out
    assert "Limiting to the first 5 comments" in out
    assert "Pooled out-of-fold" in out


# ---------------------------------------------------------------------------
# historical-baseline
# ---------------------------------------------------------------------------


def test_historical_baseline_command_writes_predictions_and_scores_them(tmp_path, capsys):
    reference_path = tmp_path / "pairs.parquet"
    pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "contribution_id": 1,
                "key_message": "km1",
                "comment_text": "c1",
                "resolution_status": "resolved",
                "reference_label": "in_favor",
                "established_original_label": "in_favor",
            },
            {
                "pair_id": "p2",
                "contribution_id": 1,
                "key_message": "km2",
                "comment_text": "c2",
                "resolution_status": "resolved",
                "reference_label": "neutral",
                "established_original_label": "against",
            },
        ]
    ).to_parquet(reference_path)

    output_path = tmp_path / "historical_predictions.parquet"

    cli.main(
        [
            "historical-baseline",
            "--output",
            str(output_path),
            "--reference",
            str(reference_path),
            "--folds",
            str(tmp_path / "no_such_folds.json"),
        ]
    )

    assert output_path.exists()
    written = pd.read_parquet(output_path)
    assert len(written) == 2
    assert set(written["status"]) == {"accepted"}

    out = capsys.readouterr().out
    assert "Wrote 2 historical-baseline predictions" in out
    assert "Overall" in out


# ---------------------------------------------------------------------------
# post-fix-baseline
# ---------------------------------------------------------------------------


def test_post_fix_baseline_command_fails_uncovered_rows_and_scores(tmp_path, capsys):
    reference_path = tmp_path / "pairs.parquet"
    pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "contribution_id": 1,
                "key_message": "km1",
                "comment_text": "c1",
                "resolution_status": "resolved",
                "reference_label": "in_favor",
                "established_original_label": "in_favor",
                "post_fix_original_label": "in_favor",
            },
            {
                "pair_id": "p2",
                "contribution_id": 1,
                "key_message": "km2",
                "comment_text": "c2",
                "resolution_status": "resolved",
                "reference_label": "neutral",
                "established_original_label": "against",
                "post_fix_original_label": None,  # not covered by stance_refit_20260908fix.csv
            },
        ]
    ).to_parquet(reference_path)

    output_path = tmp_path / "post_fix_predictions.parquet"

    cli.main(
        [
            "post-fix-baseline",
            "--output",
            str(output_path),
            "--reference",
            str(reference_path),
            "--folds",
            str(tmp_path / "no_such_folds.json"),
        ]
    )

    assert output_path.exists()
    written = pd.read_parquet(output_path)
    assert len(written) == 2
    assert written.iloc[0]["pair_id"] == "p1"
    assert written.iloc[1]["status"] == "failed"

    out = capsys.readouterr().out
    assert "Wrote 2 post-fix-baseline predictions" in out
    assert "Overall" in out
