"""Build a dataset of sampled contributions joined with their comments.

This module exposes :class:`ContributionsWithCommentsDataset`, a helper that
loads the fetched contributions Parquet file, draws a reproducible random
sample, fetches the comments for each sampled contribution from the DIPAS
database, drops contributions that have no comments, and writes the result to
a sibling Parquet file named ``contributions_with_comments_<n>.parquet``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from rich.console import Console

DATA_DIR = Path(__file__).resolve().parent / "data"
CONTRIBUTIONS_PATH = DATA_DIR / "contributions.parquet"

console = Console()


class ContributionsWithCommentsDataset:
    """Build a contributions-with-comments dataset from a random sample.

    The workflow is: load the contributions Parquet file, draw ``n`` rows at
    random, fetch comments for each sampled contribution, drop rows whose
    comment list is empty, and save the survivors (each augmented with a
    ``comments`` column) to a sibling Parquet file.

    Attributes:
        n: Number of contributions to sample.
        source_path: Path to the source contributions Parquet file.
        random_state: Optional seed for reproducible sampling.
    """

    def __init__(
        self,
        n: int = 100,
        source_path: Path = CONTRIBUTIONS_PATH,
        fetcher=None,
        random_state: int | None = None,
    ) -> None:
        """Initialize the dataset builder and validate its arguments.

        Args:
            n: Number of contributions to sample. Must be greater than zero.
            source_path: Path to the contributions Parquet file. Defaults to
                the package's ``data/contributions.parquet``.
            fetcher: Optional comment fetcher. When ``None``, a
                :class:`~comment_fetcher.CommentFetcher` is created lazily in
                :meth:`build`. Injecting a fake here keeps tests off the DB.
            random_state: Optional seed passed to ``DataFrame.sample`` for
                reproducible results.

        Raises:
            ValueError: If ``n`` is not greater than zero.
        """
        if n <= 0:
            raise ValueError(f"n must be > 0, got {n}")
        self.n = n
        self.source_path = Path(source_path)
        self.random_state = random_state
        self._fetcher = fetcher

    @property
    def output_path(self) -> Path:
        """Path the dataset is written to.

        Returns:
            A sibling of ``source_path`` named
            ``contributions_with_comments_<n>.parquet``.
        """
        return self.source_path.parent / f"contributions_with_comments_{self.n}.parquet"

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

    def draw_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """Draw a random sample from the contributions DataFrame.

        If fewer than ``n`` rows are available the entire DataFrame is sampled
        and a warning is printed, rather than raising.

        Args:
            df: The full contributions DataFrame.

        Returns:
            A DataFrame with ``min(n, len(df))`` randomly sampled rows.

        Raises:
            ValueError: If ``df`` lacks the required ``id`` column.
        """
        if "id" not in df.columns:
            raise ValueError("DataFrame missing required column 'id'")
        effective_n = min(self.n, len(df))
        if effective_n < self.n:
            console.print(
                f"[yellow]Warning: requested {self.n} rows but only "
                f"{effective_n} available — sampling all.[/yellow]"
            )
        return df.sample(n=effective_n, random_state=self.random_state)

    def fetch_comments_for(self, contribution_id: int) -> list[dict]:
        """Fetch the comments for a single contribution.

        Args:
            contribution_id: The contribution ID; coerced to a native ``int``
                because Parquet yields ``numpy.int64`` which the fetcher rejects.

        Returns:
            The list of comment dicts for the contribution.
        """
        cid = int(contribution_id)
        return self._fetcher.get_comments(cid)

    def build(self) -> Path:
        """Run the full load → sample → fetch → save pipeline.

        Loads and samples the contributions, fetches comments for each sampled
        row, drops rows with no comments, and writes the survivors to
        :attr:`output_path`. The fetcher is always closed, even on error.

        Returns:
            The path the dataset was written to.

        Raises:
            FileNotFoundError: If the source file is missing.
            ValueError: If the data is malformed (missing ``id`` column).
        """
        df = self.load()
        sampled = self.draw_sample(df)

        if self._fetcher is None:
            from comment_fetcher import CommentFetcher

            self._fetcher = CommentFetcher()

        kept_records: list[dict] = []
        try:
            for _, row in sampled.iterrows():
                contribution_id = int(row["id"])
                comments = self.fetch_comments_for(contribution_id)
                if not comments:
                    console.print(
                        f"[yellow]Skipping contribution {contribution_id} — "
                        "no comments found.[/yellow]"
                    )
                    continue
                record = row.to_dict()
                record["comments"] = comments
                kept_records.append(record)
        finally:
            # Always release the DB engine, even if fetching raised.
            self._fetcher.close()

        result_df = pd.DataFrame(kept_records)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(self.output_path)
        console.print(
            f"[green]Dataset ({len(result_df)} rows) saved to {self.output_path}[/green]"
        )
        return self.output_path
