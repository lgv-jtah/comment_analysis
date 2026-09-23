import pytest

from policy_schemas import StanceLabelV2, normalize_legacy_label


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("in favor", StanceLabelV2.IN_FAVOR),
        ("In Favor", StanceLabelV2.IN_FAVOR),
        ("  in favor  ", StanceLabelV2.IN_FAVOR),
        ("dafür", StanceLabelV2.IN_FAVOR),
        ("against", StanceLabelV2.AGAINST),
        ("ablehnung", StanceLabelV2.AGAINST),
        ("Ablehnung", StanceLabelV2.AGAINST),
        ("dagegen", StanceLabelV2.AGAINST),
        ("neutral", StanceLabelV2.NEUTRAL),
        ("NEUTRAL", StanceLabelV2.NEUTRAL),
    ],
)
def test_normalize_legacy_label_known_values(raw, expected):
    assert normalize_legacy_label(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "unknown_value", "in favour"])
def test_normalize_legacy_label_unknown_or_empty_returns_none(raw):
    assert normalize_legacy_label(raw) is None


def test_normalize_legacy_label_handles_pandas_nan_without_raising():
    # A missing SQL/CSV cell read through pandas becomes a NaN float.
    assert normalize_legacy_label(float("nan")) is None
