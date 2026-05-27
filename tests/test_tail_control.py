import polars as pl
import pytest

from janestreet.tail_control import (
    TailSwitchPolicy,
    add_tail_switch_prediction,
    fit_grouped_tail_advantage_policy,
    with_batch_missing_fraction,
)


def test_tail_switch_prediction_uses_candidate_only_for_tail_buckets() -> None:
    frame = pl.DataFrame(
        {
            "weight_bucket": ["q00_q50", "q90_q99", "q99_q100"],
            "base_prediction": [1.0, 2.0, 3.0],
            "candidate_prediction": [10.0, 20.0, 30.0],
        }
    )

    switched = add_tail_switch_prediction(
        frame,
        base_prediction="base_prediction",
        candidate_prediction="candidate_prediction",
        tail_buckets=("q90_q99", "q99_q100"),
        output="prediction",
    )

    assert switched["prediction"].to_list() == [1.0, 20.0, 30.0]


def test_tail_switch_policy_uses_default_tail_buckets() -> None:
    frame = pl.DataFrame(
        {
            "weight_bucket": ["q50_q90", "q90_q99", "q99_q100"],
            "ensemble_prediction": [1.0, 2.0, 3.0],
            "clock_simplex_prediction": [10.0, 20.0, 30.0],
        }
    )

    result = TailSwitchPolicy().apply(frame)

    assert result["tail_control_prediction"].to_list() == [1.0, 20.0, 30.0]


def test_tail_switch_rejects_empty_tail_buckets() -> None:
    frame = pl.DataFrame(
        {
            "weight_bucket": ["q99_q100"],
            "base_prediction": [1.0],
            "candidate_prediction": [2.0],
        }
    )

    with pytest.raises(ValueError, match="tail_buckets"):
        add_tail_switch_prediction(
            frame,
            base_prediction="base_prediction",
            candidate_prediction="candidate_prediction",
            tail_buckets=(),
            output="prediction",
        )


def test_batch_missing_fraction_matches_gateway_batch_mean() -> None:
    frame = pl.DataFrame(
        {
            "date_id": [1, 1, 1, 1, 1],
            "time_id": [0, 0, 0, 1, 1],
            "feature_00": [None, 1.0, 2.0, None, 5.0],
            "feature_01": [None, None, 2.0, 4.0, 5.0],
        }
    )

    result = with_batch_missing_fraction(frame, source_columns=("feature_00", "feature_01"))

    assert result["batch_missing_frac"].to_list() == [0.5, 0.5, 0.5, 0.25, 0.25]


def test_grouped_tail_advantage_uses_candidate_only_where_calibration_improves() -> None:
    calibration = pl.DataFrame(
        {
            "clock_bucket": [0, 0, 1, 1, 1, 1],
            "weight_bucket": ["q90_q99", "q90_q99", "q90_q99", "q90_q99", "q00_q50", "q00_q50"],
            "weight": [1.0] * 6,
            "responder_6": [10.0, 10.0, 0.0, 0.0, 10.0, 10.0],
            "base_prediction": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "candidate_prediction": [10.0, 10.0, 5.0, 5.0, 10.0, 10.0],
        }
    )
    validation = pl.DataFrame(
        {
            "clock_bucket": [0, 1, 0],
            "weight_bucket": ["q90_q99", "q90_q99", "q00_q50"],
            "base_prediction": [1.0, 2.0, 3.0],
            "candidate_prediction": [10.0, 20.0, 30.0],
        }
    )

    policy = fit_grouped_tail_advantage_policy(
        calibration,
        group_columns=("clock_bucket",),
        base_prediction="base_prediction",
        candidate_prediction="candidate_prediction",
        min_group_rows=2,
        output="prediction",
    )
    result = policy.apply(validation)

    assert policy.parameters.select(["clock_bucket", "_tail_use_candidate"]).sort("clock_bucket").to_dicts() == [
        {"clock_bucket": 0, "_tail_use_candidate": True},
        {"clock_bucket": 1, "_tail_use_candidate": False},
    ]
    assert result["prediction"].to_list() == [10.0, 2.0, 3.0]


def test_grouped_tail_advantage_matches_fixed_tail_when_all_tail_groups_use_candidate() -> None:
    calibration = pl.DataFrame(
        {
            "clock_bucket": [0, 0, 1, 1],
            "weight_bucket": ["q90_q99", "q90_q99", "q99_q100", "q99_q100"],
            "weight": [1.0] * 4,
            "responder_6": [10.0, 10.0, -5.0, -5.0],
            "base_prediction": [0.0, 0.0, 0.0, 0.0],
            "candidate_prediction": [10.0, 10.0, -5.0, -5.0],
        }
    )
    validation = pl.DataFrame(
        {
            "clock_bucket": [0, 1, 2, 0],
            "weight_bucket": ["q90_q99", "q99_q100", "q90_q99", "q00_q50"],
            "base_prediction": [1.0, 2.0, 3.0, 4.0],
            "candidate_prediction": [10.0, 20.0, 30.0, 40.0],
        }
    )

    policy = fit_grouped_tail_advantage_policy(
        calibration,
        group_columns=("clock_bucket",),
        base_prediction="base_prediction",
        candidate_prediction="candidate_prediction",
        min_group_rows=2,
        output="advantage_prediction",
    )
    fixed = add_tail_switch_prediction(
        validation,
        base_prediction="base_prediction",
        candidate_prediction="candidate_prediction",
        tail_buckets=("q90_q99", "q99_q100"),
        output="fixed_prediction",
    )
    advantage = policy.apply(fixed)

    assert policy.fallback_use_candidate is True
    assert policy.parameters.select("_tail_use_candidate").to_series().to_list() == [True, True]
    assert advantage["advantage_prediction"].to_list() == advantage["fixed_prediction"].to_list()
