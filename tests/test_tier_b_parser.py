import pytest

from tier_b_parser import PARSER_RULE_VERSION, parse_note


@pytest.mark.parametrize(
    "note,expected_label",
    [
        ("Neutral", "neutral"),
        ("neutral", "neutral"),
        ("NEUTRAL.", "neutral"),
        ("In Favor", "in_favor"),
        ("Ablehnung", "against"),
        ("Dagegen", "against"),
    ],
)
def test_whole_note_exact_match_resolves(note, expected_label):
    result = parse_note(note)

    assert result.status == "resolved"
    assert result.label == expected_label
    assert result.rule == f"{PARSER_RULE_VERSION}:whole_note_exact"
    assert result.supporting_text == note


def test_unambiguous_prefix_resolves():
    result = parse_note("In Favor: stimmt inhaltlich zu")

    assert result.status == "resolved"
    assert result.label == "in_favor"
    assert result.rule == f"{PARSER_RULE_VERSION}:prefix_colon"
    assert result.supporting_text == "In Favor: stimmt inhaltlich zu"


@pytest.mark.parametrize(
    "note,expected_reason",
    [
        ("nicht neutral", "negation"),
        ("Neutral: aber nicht sicher", "negation"),
        ("kein eindeutiger Bezug", "negation"),
        ("eher neutral", "hedge"),
        ("Neutral: eher unklar", "hedge"),
        ("neutral oder against?", "question"),  # "?" is checked before multi-label
        ("Ist das neutral?", "question"),
        ("neutral oder against", "multi_label_mentioned"),
        ("In Favor: bezieht sich aber auch auf against", "multi_label_mentioned"),
        ("Der Kommentar bezieht sich auf eine andere Keymessage", "no_unambiguous_pattern"),
    ],
)
def test_ambiguous_notes_stay_unresolved(note, expected_reason):
    result = parse_note(note)

    assert result.status == "unresolved"
    assert result.label is None
    assert result.reason == expected_reason


def test_question_mark_alone_is_unresolved():
    result = parse_note("Ist das neutral?")

    assert result.status == "unresolved"
    assert result.reason == "question"


@pytest.mark.parametrize("note", [None, "", "   "])
def test_missing_or_blank_note_is_empty(note):
    result = parse_note(note)

    assert result.status == "empty"
    assert result.label is None


def test_pandas_nan_note_is_empty_not_a_crash():
    # A missing SQL/CSV cell read through pandas becomes a NaN float, not
    # None -- must not raise AttributeError on .strip().
    result = parse_note(float("nan"))

    assert result.status == "empty"


def test_note_mentioning_unrelated_text_is_unresolved_not_guessed():
    result = parse_note("Bezieht sich zustimmend auf andere Keymessage. Hier NEUTRAL")

    # Not a whole-note match, not a clean prefix -- must not be silently
    # resolved just because "NEUTRAL" appears in the text.
    assert result.status == "unresolved"
