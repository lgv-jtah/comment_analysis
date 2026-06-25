"""Fetch DIPAS participation projects and build a tabular dataset.

The Hamburg DIPAS participation data service exposes project metadata as a
GeoJSON FeatureCollection. This module retrieves that data and flattens the
nested project properties into a :class:`pandas.DataFrame` suitable for
analysis.
"""
from __future__ import annotations

import json
import urllib.parse
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from rich.console import Console


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

    def fetch_all_contributions(
        self, project_ids: List[str]
    ) -> pd.DataFrame:
        """Fetch contributions for many projects into a single DataFrame.

        For each project id, this calls the DIPAS contributions endpoint
        (``{api_url}{project_id}/contributions/``) and flattens the returned
        GeoJSON features into rows.

        Args:
            project_ids: Project ids to fetch contributions for. These are
                typically obtained from ``create()["id"].tolist()``.
                ``self.api_url`` must end with ``/`` for the URL to be built
                correctly (the default endpoint does).

        Returns:
            A DataFrame with one row per contribution and the columns
            ``project_id``, ``contribution_id``, ``title``,
            ``contributionContent`` and ``category``. The DataFrame is empty if
            no contributions were collected.

        Notes:
            Per-project :class:`requests.RequestException` errors are caught
            and logged, and the partial results gathered so far are still
            returned. JSON decode errors are *not* caught and will propagate
            to the caller.
        """
        all_contributions = []

        for project_id in project_ids:
            # quote(..., safe="") percent-encodes the id so it cannot break
            # out of the path segment or inject extra URL components.
            safe_id = urllib.parse.quote(str(project_id), safe="")
            url = f"{self.api_url}{safe_id}/contributions/"
            try:
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                for feature in data.get("features", []):
                    props = feature.get("properties", {})
                    all_contributions.append(
                        {
                            "project_id": project_id,
                            "contribution_id": props.get("id"),
                            "title": props.get("title"),
                            "contributionContent": props.get(
                                "contributionContent"
                            ),
                            "category": props.get("category"),
                        }
                    )
                print(
                    f"OK {project_id}: "
                    f"{len(data.get('features', []))} contributions"
                )
            except requests.RequestException as e:
                # Only network/HTTP errors are caught per-project so a single
                # bad project doesn't abort the whole run; the partial results
                # gathered so far are still returned below. JSON decode errors
                # are deliberately NOT caught here -- a malformed payload signals
                # an unexpected API contract change that should surface loudly
                # rather than be silently treated as "no contributions".
                print(f"FAIL {project_id}: {e}")

        return pd.DataFrame(all_contributions)

    def contributions_summary(
        self, df: pd.DataFrame, top_n: int = 10
    ) -> str:
        """Summarize contribution counts per project.

        Args:
            df: A DataFrame produced by :meth:`fetch_all_contributions`.
            top_n: Maximum number of projects to include, ordered by
                descending contribution count. Defaults to 10.

        Returns:
            A human-readable string mapping project ids to their contribution
            counts, or an empty string when ``df`` is empty or lacks the
            ``project_id`` column.
        """
        if df.empty or "project_id" not in df.columns:
            return ""
        return (
            df.groupby("project_id")
            .size()
            .sort_values(ascending=False)
            .head(top_n)
            .to_string()
        )

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
