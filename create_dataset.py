"""Fetch DIPAS participation projects and build a tabular dataset.

The Hamburg DIPAS participation data service exposes project metadata as a
GeoJSON FeatureCollection. This module retrieves that data and flattens the
nested project properties into a :class:`pandas.DataFrame` suitable for
analysis.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pandas as pd
import requests

API_URL = "https://beteiligung.hamburg/drupal/dipas-pds/projects/"


class CreateDataset:
    """Download DIPAS project data and convert it into a DataFrame."""

    def __init__(self, api_url: str = API_URL, timeout: int = 30) -> None:
        """Initialize the dataset builder.

        Args:
            api_url: Endpoint to fetch the DIPAS GeoJSON payload from.
                Defaults to the public DIPAS projects endpoint.
            timeout: Per-request timeout in seconds. Defaults to 30.
        """
        self.api_url = api_url
        self.timeout = timeout

    # ── Public API ─────────────────────────────────────────────────

    def fetch_projects(self) -> Optional[Dict[str, Any]]:
        """Fetch the raw GeoJSON payload from the DIPAS API.

        Network and decode failures are swallowed so that callers can treat
        an unavailable API as "no data" rather than handling exceptions.

        Returns:
            The parsed JSON object on success, or ``None`` if the request
            fails or the response body is not valid JSON.
        """
        try:
            response = requests.get(self.api_url, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError, json.JSONDecodeError):
            return None

    def build_dataframe(
        self, geojson_data: Optional[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Flatten a GeoJSON FeatureCollection into a DataFrame.

        Args:
            geojson_data: Parsed GeoJSON FeatureCollection, or ``None``.

        Returns:
            A DataFrame with one row per feature, or an empty DataFrame when
            the input is ``None`` or contains no ``"features"`` key.
        """
        if geojson_data is None or "features" not in geojson_data:
            return pd.DataFrame()

        projects_data = [
            self._extract_project_properties(feature)
            for feature in geojson_data.get("features", [])
        ]
        df = pd.DataFrame(projects_data)
        return self._convert_dtypes(df)

    def create(self) -> pd.DataFrame:
        """Fetch the API data and build the dataset in one step.

        Returns:
            The assembled DataFrame, empty if the API call failed.
        """
        return self.build_dataframe(self.fetch_projects())

    # ── Private helpers ────────────────────────────────────────────

    def _extract_names_from_dict(
        self, nested_dict: Optional[Dict]
    ) -> Optional[str]:
        """Join the ``name`` values of a nested mapping into one string.

        DIPAS encodes related entities (owners, topics, ...) as a dict keyed
        by opaque ids, each value being a dict with a ``name`` field. This
        collapses them into a single comma-separated label.

        Args:
            nested_dict: Mapping of id -> entity dict, or ``None``.

        Returns:
            A comma-separated string of names, or ``None`` if there are no
            usable names.
        """
        if not nested_dict:
            return None
        # Skip values that are not dicts; missing names default to "".
        names = [
            v.get("name", "")
            for v in nested_dict.values()
            if isinstance(v, dict)
        ]
        # filter(None, ...) drops empty strings so they don't become ", , ".
        return ", ".join(filter(None, names)) if names else None

    def _extract_project_properties(
        self, feature: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Flatten a single GeoJSON feature into a flat property dict.

        Args:
            feature: A single GeoJSON feature object.

        Returns:
            A flat dict of scalar project attributes, with nested entity
            collections reduced to comma-separated name strings.
        """
        props = feature.get("properties", {})
        keys = [
            "id",
            "nameFull",
            "description",
            "dateStart",
            "dateEnd",
            "proceedingState",
            "website",
            "owner",
            "publisher",
            "referenceSystem",
            "isDipasProceeding",
        ]
        flat_props = {key: props.get(key) for key in keys}

        flat_props["projectOwner"] = self._extract_names_from_dict(
            props.get("projectOwner")
        )
        flat_props["standardCategories"] = self._extract_names_from_dict(
            props.get("standardCategories")
        )
        flat_props["contributionTypes"] = self._extract_names_from_dict(
            props.get("projectContributionType")
        )
        flat_props["mainDistrict"] = self._extract_names_from_dict(
            props.get("dipasMainDistrict")
        )
        flat_props["projectTopics"] = self._extract_names_from_dict(
            props.get("projectTopics")
        )
        # `or 0` is a defensive fallback: len() already yields 0 for an empty
        # list, so this only matters if the value is ever a falsy non-list.
        flat_props["contributionCount"] = (
            len(props.get("hasParticipatoryText", [])) or 0
        )
        flat_props["geometryType"] = feature.get("geometry", {}).get("type")
        return flat_props

    def _convert_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse date columns and derive the project duration in days.

        Args:
            df: DataFrame produced from the flattened project properties.

        Returns:
            The same DataFrame with ``dateStart``/``dateEnd`` converted to
            UTC datetimes and a new ``durationDays`` column. Returned
            unchanged if it is empty or lacks the date columns.
        """
        if not df.empty and "dateStart" in df.columns and "dateEnd" in df.columns:
            # errors="coerce" turns unparseable/missing dates into NaT instead
            # of raising, so a single bad date can't fail the whole build.
            df["dateStart"] = pd.to_datetime(
                df["dateStart"], errors="coerce", utc=True
            )
            df["dateEnd"] = pd.to_datetime(
                df["dateEnd"], errors="coerce", utc=True
            )
            # Subtracting NaT propagates NaT, so durationDays stays NaN here.
            df["durationDays"] = (df["dateEnd"] - df["dateStart"]).dt.days
        return df
