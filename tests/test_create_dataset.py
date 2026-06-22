"""Tests for create_dataset.CreateDataset.

These tests never make a real network request. Following the project's
convention (see test_llm_handler.py), requests.get is monkeypatched to
return a minimal FakeResponse stand-in instead of relying on an external
mocking library.
"""
import pandas as pd
import pytest
import requests

from create_dataset import API_URL, CreateDataset


class FakeResponse:
    """Minimal stand-in for requests.Response used by the monkeypatch."""

    def __init__(self, json_data=None, status_code=200, json_exc=None):
        self._json_data = json_data
        self.status_code = status_code
        self._json_exc = json_exc

    def json(self):
        """Return the canned JSON body, or raise the configured exception."""
        if self._json_exc is not None:
            raise self._json_exc
        return self._json_data

    def raise_for_status(self):
        """Raise HTTPError for >=400 status codes, mirroring requests."""
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def _feature(
    feature_id="1",
    name="Project One",
    date_start="2020-01-01T00:00:00+00:00",
    date_end="2020-01-11T00:00:00+00:00",
    geometry_type="Polygon",
):
    """Build a representative GeoJSON feature for tests."""
    return {
        "geometry": {"type": geometry_type},
        "properties": {
            "id": feature_id,
            "nameFull": name,
            "description": "A description.",
            "dateStart": date_start,
            "dateEnd": date_end,
            "proceedingState": "finished",
            "website": "https://example.org",
            "owner": "Owner",
            "publisher": "Publisher",
            "referenceSystem": "EPSG:25832",
            "isDipasProceeding": True,
            "projectOwner": {"a": {"name": "Alice"}, "b": {"name": "Bob"}},
            "standardCategories": {"c": {"name": "Cat1"}},
            "projectContributionType": {"d": {"name": "Idea"}},
            "dipasMainDistrict": {"e": {"name": "Mitte"}},
            "projectTopics": {"f": {"name": "Mobility"}},
            "hasParticipatoryText": [{}, {}, {}],
        },
    }


# ── fetch_projects ─────────────────────────────────────────────────


def test_fetch_projects_happy_path(monkeypatch):
    payload = {"type": "FeatureCollection", "features": [_feature()]}
    # Patch the name as looked up inside create_dataset (create_dataset.requests.get),
    # not requests.get globally, so only this module's calls are intercepted.
    monkeypatch.setattr(
        "create_dataset.requests.get",
        lambda url, timeout: FakeResponse(json_data=payload),
    )
    result = CreateDataset().fetch_projects()
    assert result == payload


def test_fetch_projects_http_error(monkeypatch):
    monkeypatch.setattr(
        "create_dataset.requests.get",
        lambda url, timeout: FakeResponse(status_code=500),
    )
    assert CreateDataset().fetch_projects() is None


def test_fetch_projects_connection_error(monkeypatch):
    def _raise(url, timeout):
        raise requests.RequestException("connection failed")

    monkeypatch.setattr("create_dataset.requests.get", _raise)
    assert CreateDataset().fetch_projects() is None


def test_fetch_projects_invalid_json(monkeypatch):
    monkeypatch.setattr(
        "create_dataset.requests.get",
        lambda url, timeout: FakeResponse(json_exc=ValueError("bad json")),
    )
    assert CreateDataset().fetch_projects() is None


# ── _extract_names_from_dict ───────────────────────────────────────


@pytest.mark.parametrize(
    "nested, expected",
    [
        (None, None),
        ({}, None),
        ({"a": {"name": "Alice"}, "b": {"name": "Bob"}}, "Alice, Bob"),
        # Mixed invalid values: non-dicts skipped, empty names filtered out.
        (
            {"a": {"name": "Alice"}, "b": "not a dict", "c": {"name": ""}},
            "Alice",
        ),
    ],
)
def test_extract_names_from_dict(nested, expected):
    assert CreateDataset()._extract_names_from_dict(nested) == expected


# ── _extract_project_properties ────────────────────────────────────


def test_extract_project_properties_full_feature():
    flat = CreateDataset()._extract_project_properties(_feature())
    assert flat["id"] == "1"
    assert flat["nameFull"] == "Project One"
    assert flat["proceedingState"] == "finished"
    assert flat["isDipasProceeding"] is True
    assert flat["projectOwner"] == "Alice, Bob"
    assert flat["standardCategories"] == "Cat1"
    assert flat["contributionTypes"] == "Idea"
    assert flat["mainDistrict"] == "Mitte"
    assert flat["projectTopics"] == "Mobility"
    assert flat["contributionCount"] == 3
    assert flat["geometryType"] == "Polygon"


def test_extract_project_properties_empty_feature():
    flat = CreateDataset()._extract_project_properties({})
    assert flat["id"] is None
    assert flat["contributionCount"] == 0
    assert flat["geometryType"] is None
    assert flat["projectOwner"] is None


# ── build_dataframe ────────────────────────────────────────────────


def test_build_dataframe_happy_path():
    geojson = {
        "features": [
            _feature(feature_id="1"),
            _feature(
                feature_id="2",
                date_start="2021-01-01T00:00:00+00:00",
                date_end="2021-01-06T00:00:00+00:00",
            ),
        ]
    }
    df = CreateDataset().build_dataframe(geojson)
    assert df.shape[0] == 2
    assert isinstance(df["dateStart"].dtype, pd.DatetimeTZDtype)
    assert str(df["dateStart"].dt.tz) == "UTC"
    assert df.loc[0, "durationDays"] == 10
    assert df.loc[1, "durationDays"] == 5


def test_build_dataframe_none_input():
    df = CreateDataset().build_dataframe(None)
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_build_dataframe_empty_features():
    df = CreateDataset().build_dataframe({"features": []})
    assert isinstance(df, pd.DataFrame)
    assert df.empty


# ── create ─────────────────────────────────────────────────────────


def test_create_end_to_end(monkeypatch):
    payload = {"features": [_feature()]}
    monkeypatch.setattr(
        "create_dataset.requests.get",
        lambda url, timeout: FakeResponse(json_data=payload),
    )
    df = CreateDataset().create()
    assert df.shape[0] == 1
    assert df.loc[0, "nameFull"] == "Project One"


def test_create_failure_path(monkeypatch):
    def _raise(url, timeout):
        raise requests.RequestException("down")

    monkeypatch.setattr("create_dataset.requests.get", _raise)
    df = CreateDataset().create()
    assert df.empty


# ── durationDays with NaT ──────────────────────────────────────────


def test_duration_days_with_missing_date_does_not_raise():
    geojson = {"features": [_feature(date_end=None)]}
    df = CreateDataset().build_dataframe(geojson)
    assert pd.isna(df.loc[0, "durationDays"])


def test_api_url_default_is_used():
    assert CreateDataset().api_url == API_URL
    assert CreateDataset(api_url="http://x", timeout=5).timeout == 5
