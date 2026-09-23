import pytest
from pydantic import ValidationError

from policy_schemas import (
    NeutralBasis,
    OneCallBasis,
    OneCallTargetResult,
    PerTargetStanceResult,
    ScopeAttributionResult,
    ScopeType,
    StanceLabelV2,
    StanceV2OneCallResult,
    TargetStance,
    normalize_legacy_label,
)


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


def test_scope_attribution_result_rejects_unknown_scope_type():
    with pytest.raises(ValidationError):
        ScopeAttributionResult(scope_type="not_a_real_scope")


def test_scope_attribution_result_defaults():
    result = ScopeAttributionResult(scope_type=ScopeType.WHOLE_CONTRIBUTION_APPROVAL)

    assert result.affected_key_message_ids == []
    assert result.general_support_extends_to_unaddressed is None
    assert result.comment_evidence == []
    assert result.uncertainty_flags == []


def test_target_stance_requires_valid_stance_enum():
    with pytest.raises(ValidationError):
        TargetStance(key_message_id="1:km1", stance="not_a_label")


def test_target_stance_neutral_with_basis_round_trips():
    ts = TargetStance(
        key_message_id="1:km1",
        stance=StanceLabelV2.NEUTRAL,
        neutral_basis=NeutralBasis.UNADDRESSED,
    )

    assert ts.stance == StanceLabelV2.NEUTRAL
    assert ts.neutral_basis == NeutralBasis.UNADDRESSED


def test_per_target_stance_result_holds_multiple_targets():
    result = PerTargetStanceResult(
        targets=[
            TargetStance(key_message_id="1:km1", stance=StanceLabelV2.IN_FAVOR),
            TargetStance(
                key_message_id="1:km2",
                stance=StanceLabelV2.NEUTRAL,
                neutral_basis=NeutralBasis.NO_POSITION,
            ),
        ]
    )

    assert len(result.targets) == 2
    assert result.targets[1].neutral_basis == NeutralBasis.NO_POSITION


def test_one_call_result_holds_basis_enum():
    result = StanceV2OneCallResult(
        targets=[
            OneCallTargetResult(
                key_message_id="1:km1",
                stance=StanceLabelV2.IN_FAVOR,
                basis=OneCallBasis.GENERAL_APPROVAL,
            )
        ]
    )

    assert result.targets[0].basis == OneCallBasis.GENERAL_APPROVAL


def test_schemas_serialize_to_plain_json_values():
    result = ScopeAttributionResult(
        scope_type=ScopeType.TARGET_SPECIFIC, affected_key_message_ids=["1:km1"]
    )

    dumped = result.model_dump(mode="json")

    assert dumped["scope_type"] == "target_specific"
