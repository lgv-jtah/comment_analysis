from __future__ import annotations

"""Fetches comments from the DIPAS PostgreSQL database for a given contribution."""

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

DEFAULT_DB_URL = "postgresql://dipas:dipas@localhost:5435/dipas"

_COMMENTS_QUERY = text("""
    SELECT cfd.cid, comment_type, cfd.entity_id, cfc.field_comment_value
    FROM comment_field_data AS cfd
    INNER JOIN comment__field_comment AS cfc ON cfc.entity_id = cfd.cid
    WHERE comment_type = 'default'
    AND cfd.entity_id = :contribution_id
""")


class CommentFetcher:
    """Fetches comments related to a contribution from the DIPAS database."""

    def __init__(self, db_url: str = DEFAULT_DB_URL) -> None:
        """Create a fetcher backed by a SQLAlchemy engine for the given database.

        Args:
            db_url: SQLAlchemy connection URL. Defaults to the local DIPAS database.
        """
        self._engine: Engine = create_engine(db_url)

    def get_comments(self, contribution_id: int) -> list[dict]:
        """Return all default-type comments for the given contribution ID.

        Args:
            contribution_id: The integer ID of the contribution whose comments
                should be fetched.

        Returns:
            A list of dicts with keys cid, comment_type, entity_id,
            field_comment_value. Empty list if no comments found.

        Raises:
            TypeError: If contribution_id is not a plain int (rejects bool too).
        """
        # bool is a subclass of int, so reject it explicitly (True/False are not valid IDs).
        if not isinstance(contribution_id, int) or isinstance(contribution_id, bool):
            raise TypeError(f"contribution_id must be int, got {type(contribution_id).__name__}")
        with self._engine.connect() as conn:
            result = conn.execute(_COMMENTS_QUERY, {"contribution_id": contribution_id})
            return [dict(row) for row in result.mappings()]

    def close(self) -> None:
        """Dispose the underlying SQLAlchemy engine and release connections."""
        self._engine.dispose()
