"""Draw a random sample of contributions for a set of projects.

This module exposes :class:`SampleContributions`, a small helper that loads the
fetched contributions Parquet file, filters it down to a requested set of
project IDs, draws a reproducible random sample, and writes the result back to
disk next to the source file.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console

DATA_DIR = Path(__file__).resolve().parent / "data"
CONTRIBUTIONS_PATH = DATA_DIR / "contributions.parquet"

console = Console()


class SampleContributions:
    """Sample contributions for a chosen set of projects.

    The workflow is: load the contributions Parquet file, keep only rows whose
    ``project_id`` is in the requested set, draw ``n`` rows at random, and save
    the sample to a sibling Parquet file named ``sample_<n>.parquet``.

    Attributes:
        project_ids: Requested project IDs, coerced to ``str`` so they compare
            correctly against the string-typed ``project_id`` column.
        n: Number of rows to sample.
        source_path: Path to the source contributions Parquet file.
        random_state: Optional seed for reproducible sampling.
    """

    def __init__(
        self,
        project_ids: list[str],
        n: int,
        source_path: Path = CONTRIBUTIONS_PATH,
        random_state: int | None = None,
    ) -> None:
        """Initialize the sampler and validate its arguments.

        Args:
            project_ids: Project IDs to keep. Each value is coerced to ``str``.
            n: Number of rows to sample. Must be greater than zero.
            source_path: Path to the contributions Parquet file. Defaults to
                the package's ``data/contributions.parquet``.
            random_state: Optional seed passed to ``DataFrame.sample`` for
                reproducible results.

        Raises:
            ValueError: If ``n`` is not greater than zero, or if
                ``project_ids`` is empty.
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        if not project_ids:
            raise ValueError("project_ids must not be empty")
        # The Parquet ``project_id`` column is stored as strings. Callers may
        # pass ints (e.g. from a CLI or numeric source), so coerce every ID to
        # str up front — otherwise ``isin`` would silently match nothing.
        self.project_ids = [str(pid) for pid in project_ids]
        self.n = n
        self.source_path = Path(source_path)
        self.random_state = random_state

    @property
    def output_path(self) -> Path:
        """Path the sample is written to.

        Returns:
            A sibling of ``source_path`` named ``sample_<n>.parquet``.
        """
        # Write next to the source file so the sample lives in the same data
        # directory it was drawn from, regardless of where source_path points.
        return self.source_path.parent / f"sample_{self.n}.parquet"

    def load(self) -> pd.DataFrame:
        """Load the source contributions Parquet file.

        Returns:
            The full contributions DataFrame.

        Raises:
            FileNotFoundError: If ``source_path`` does not exist.
        """
        if not self.source_path.exists():
            raise FileNotFoundError(
                f"Contributions file not found: {self.source_path}. "
                "Run option 1 (Fetch contributions) first."
            )
        return pd.read_parquet(self.source_path)

    def filter_projects(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows whose ``project_id`` is in the requested set.

        Args:
            df: The full contributions DataFrame.

        Returns:
            The filtered DataFrame containing only requested projects.

        Raises:
            ValueError: If ``df`` lacks a ``project_id`` column, or if no rows
                remain after filtering.
        """
        if "project_id" not in df.columns:
            raise ValueError("DataFrame missing required column 'project_id'")
        present = set(df["project_id"].unique())
        missing = [pid for pid in self.project_ids if pid not in present]
        if missing:
            console.print(
                f"[yellow]Warning: {len(missing)} project ID(s) not found in "
                f"data: {missing[:5]}{'...' if len(missing) > 5 else ''}[/yellow]"
            )
        filtered = df[df["project_id"].isin(self.project_ids)]
        if filtered.empty:
            raise ValueError(
                "No contributions found for the requested project IDs: "
                f"{self.project_ids}"
            )
        return filtered

    def draw_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Draw a random sample from the filtered DataFrame.

        If fewer than ``n`` rows are available the entire DataFrame is returned
        and a warning is printed, rather than raising.

        Args:
            df: The filtered contributions DataFrame.

        Returns:
            A DataFrame with ``min(n, len(df))`` randomly sampled rows.
        """
        # Clamp instead of raising: a smaller-than-requested pool is a normal,
        # recoverable situation (the user just gets everything available), so
        # don't abort the pipeline over it.
        effective_n = min(self.n, len(df))
        if effective_n < self.n:
            console.print(
                f"[yellow]Warning: requested {self.n} rows but only "
                f"{effective_n} available after filtering — sampling all.[/yellow]"
            )
        return df.sample(n=effective_n, random_state=self.random_state)

    def save(self, df: pd.DataFrame) -> Path:
        """Write the sampled DataFrame to ``output_path``.

        Args:
            df: The sampled DataFrame to persist.

        Returns:
            The path the sample was written to.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path)
        console.print(
            f"[green]Sample ({len(df)} rows) saved to {self.output_path}[/green]"
        )
        return self.output_path

    def sample(self) -> Path:
        """Run the full load → filter → sample → save pipeline.

        Returns:
            The path the sample was written to.

        Raises:
            FileNotFoundError: If the source file is missing.
            ValueError: If filtering produces no rows or the data is malformed.
        """
        df = self.load()
        filtered = self.filter_projects(df)
        sampled = self.draw_sample(filtered)
        return self.save(sampled)
