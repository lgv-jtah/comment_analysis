"""Tests for the fast-path gate in main.should_fast_path.

should_fast_path is pure and synchronous, so no async harness or network
mocking is needed. The data-driven test loads gold_labels_corrected.json
directly (resolved relative to this file) rather than via
StanceDetector.load_data(), which prints distribution side effects and reads
from a cwd-relative path.
"""
import json
from pathlib import Path

from main import (
    FAST_PATH_MAX_WORDS,
    GR_SCORE_THRESHOLD,
    should_fast_path,
)

GOLD_LABELS_PATH = (
    Path(__file__).resolve().parent.parent / "gold_labels_corrected.json"
)


def test_fires_for_short_positive_comment():
    assert should_fast_path("ja", 0.7, "Sehr coole Idee!") is True


def test_wrong_label_does_not_fire():
    assert should_fast_path("nein", 0.9, "Sehr coole Idee!") is False


def test_score_equal_to_threshold_does_not_fire():
    # Strict `>`: a score exactly at the threshold must not fire.
    assert should_fast_path("ja", GR_SCORE_THRESHOLD, "Sehr coole Idee!") is False


def test_score_just_above_threshold_fires():
    assert should_fast_path("ja", 0.6001, "Sehr coole Idee!") is True


def test_word_count_at_boundary_fires():
    comment = "eins zwei drei vier fuenf"
    assert len(comment.split()) == FAST_PATH_MAX_WORDS
    assert should_fast_path("ja", 0.7, comment) is True


def test_word_count_above_boundary_does_not_fire():
    comment = "eins zwei drei vier fuenf sechs"
    assert len(comment.split()) == FAST_PATH_MAX_WORDS + 1
    assert should_fast_path("ja", 0.7, comment) is False


def test_empty_comment_fires():
    assert should_fast_path("ja", 0.7, "") is True


def test_against_real_gold_labels():
    with open(GOLD_LABELS_PATH, encoding="utf-8") as f:
        rows = json.load(f)

    assert len(rows) == 155

    fired = [
        row
        for row in rows
        if should_fast_path(
            row.get("general_reaction_label", ""),
            row.get("general_reaction_score", 0.0),
            row.get("comment_text", ""),
        )
    ]
    not_fired = [row for row in rows if row not in fired]

    assert len(fired) == 13
    assert len(not_fired) == 142

    for row in fired:
        assert row["general_reaction_label"] == "ja"
        assert row["general_reaction_score"] > GR_SCORE_THRESHOLD
        assert len(row["comment_text"].split()) <= FAST_PATH_MAX_WORDS

    nein_rows = [r for r in rows if r.get("general_reaction_label") == "nein"]
    for row in nein_rows:
        assert (
            should_fast_path(
                row.get("general_reaction_label", ""),
                row.get("general_reaction_score", 0.0),
                row.get("comment_text", ""),
            )
            is False
        )
