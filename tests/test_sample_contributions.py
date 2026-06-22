"""Tests for sample_contributions.SampleContributions.

These tests write a real Parquet fixture under ``tmp_path`` so the
load/filter/sample/save pipeline is exercised end to end without touching the
package's real ``data/`` directory.
"""
import pandas as pd
import pytest

from sample_contributions import CONTRIBUTIONS_PATH, SampleContributions


def _make_contributions_df() -> pd.DataFrame:
    """Build a small contributions DataFrame spanning projects A, B and C.

    Project A has 5 rows, B has 5 rows, C has 2 rows. IDs are unique strings so
    individual sampled rows can be identified across runs.
    """
    rows = []
    counts = {"A": 5, "B": 5, "C": 2}
    next_id = 1
    for project_id, count in counts.items():
        for _ in range(count):
            rows.append(
                {
                    "project_id": project_id,
                    "id": str(next_id),
                    "title": f"Title {next_id}",
                    "contributionContent": f"Content {next_id}",
                    "category": f"Cat{next_id}",
                }
            )
            next_id += 1
    return pd.DataFrame(rows)


@pytest.fixture
def source_path(tmp_path):
    """Write the contributions fixture to a Parquet file and return its path."""
    path = tmp_path / "contributions.parquet"
    _make_contributions_df().to_parquet(path)
    return path


def test_sample_happy_path(source_path):
    sampler = SampleContributions(
        ["A", "B"], n=2, source_path=source_path, random_state=42
    )
    out = sampler.sample()
    assert out.exists()
    result = pd.read_parquet(out)
    assert len(result) == 2
    assert set(result["project_id"]).issubset({"A", "B"})


def test_sample_is_reproducible(source_path):
    first = SampleContributions(
        ["A", "B"], n=3, source_path=source_path, random_state=7
    ).sample()
    first_ids = sorted(pd.read_parquet(first)["id"].tolist())

    second = SampleContributions(
        ["A", "B"], n=3, source_path=source_path, random_state=7
    ).sample()
    second_ids = sorted(pd.read_parquet(second)["id"].tolist())

    assert first_ids == second_ids


def test_n_exceeds_available_clamps(source_path):
    # Project C only has 2 rows; requesting 10 should clamp to all of them.
    sampler = SampleContributions(
        ["C"], n=10, source_path=source_path, random_state=1
    )
    out = sampler.sample()
    result = pd.read_parquet(out)
    assert len(result) == 2
    assert set(result["project_id"]) == {"C"}


def test_filter_drops_unrequested_projects(source_path):
    sampler = SampleContributions(
        ["A", "B"], n=12, source_path=source_path, random_state=1
    )
    out = sampler.sample()
    result = pd.read_parquet(out)
    assert "C" not in result["project_id"].tolist()


def test_missing_project_id_does_not_raise(source_path):
    sampler = SampleContributions(
        ["A", "MISSING"], n=2, source_path=source_path, random_state=3
    )
    out = sampler.sample()
    result = pd.read_parquet(out)
    assert len(result) == 2
    assert set(result["project_id"]) == {"A"}


def test_all_ids_missing_raises_valueerror(source_path):
    sampler = SampleContributions(
        ["NOPE", "ALSO_NOPE"], n=2, source_path=source_path
    )
    with pytest.raises(ValueError):
        sampler.sample()


def test_source_file_missing_raises_filenotfound(tmp_path):
    missing = tmp_path / "does_not_exist.parquet"
    sampler = SampleContributions(["A"], n=1, source_path=missing)
    with pytest.raises(FileNotFoundError):
        sampler.sample()


@pytest.mark.parametrize("bad_n", [0, -1])
def test_invalid_n_raises(bad_n):
    with pytest.raises(ValueError):
        SampleContributions(["A"], n=bad_n)


def test_empty_project_ids_raises():
    with pytest.raises(ValueError):
        SampleContributions([], n=1)


def test_output_path_naming(source_path):
    sampler = SampleContributions(["A"], n=3, source_path=source_path)
    assert sampler.output_path.name == "sample_3.parquet"
    assert sampler.output_path.parent == source_path.parent


def test_string_coercion_of_ids(source_path):
    # The project_id column holds strings; integers must still match after coercion.
    sampler = SampleContributions(
        [1234], n=1, source_path=source_path, random_state=1
    )
    df = sampler.load()
    # Sanity check: no integer-valued project IDs exist in the fixture.
    assert sampler.project_ids == ["1234"]

    # Real coercion test: pass an integer that matches a string project_id.
    df2 = _make_contributions_df()
    df2["project_id"] = df2["project_id"].replace({"A": "100"})
    df2.to_parquet(source_path)
    int_sampler = SampleContributions(
        [100], n=2, source_path=source_path, random_state=1
    )
    out = int_sampler.sample()
    result = pd.read_parquet(out)
    assert len(result) == 2
    assert set(result["project_id"]) == {"100"}


def test_default_source_path():
    assert SampleContributions(["A"], 1).source_path == CONTRIBUTIONS_PATH
