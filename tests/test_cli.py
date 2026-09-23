from pathlib import Path

import pandas as pd
import pytest

import cli
from dataset_builder import AuditResult


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
