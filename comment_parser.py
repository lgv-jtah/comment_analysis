"""Utilities for inspecting and validating comment text."""


class CommentParser:
    """Provides helper methods for analysing raw comment strings."""

    def is_shorter_than(self, comment: str, max_words: int) -> bool:
        """Check whether a comment contains fewer than a given number of words.

        Word boundaries are determined by whitespace; consecutive spaces count
        as a single separator (``str.split()`` behaviour).

        Args:
            comment: The raw comment text to evaluate.
            max_words: The exclusive upper word-count threshold.

        Returns:
            True if the comment has fewer than ``max_words`` words, False otherwise.
        """
        words = comment.split()  # split() strips leading/trailing space and collapses runs
        return len(words) < max_words
