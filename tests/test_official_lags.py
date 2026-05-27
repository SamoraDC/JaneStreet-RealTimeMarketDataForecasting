import polars as pl
import pytest

from janestreet.official_lags import (
    GatewayResponderLagCache,
    daily_last_responder_lag_columns,
    daily_last_responder_lags,
    gateway_lags_from_training_date,
    responder_lag_columns,
    with_responder_lag_features,
)


def test_with_responder_lag_features_shifts_previous_date_by_time_and_symbol():
    frame = pl.DataFrame(
        {
            "date_id": [0, 0, 0, 1, 1, 1],
            "time_id": [0, 1, 0, 0, 1, 0],
            "symbol_id": [7, 7, 8, 7, 7, 8],
            "responder_0": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            "responder_1": [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
        }
    )

    result = with_responder_lag_features(
        frame.lazy(),
        responder_columns=("responder_0", "responder_1"),
    ).collect()

    assert result["responder_0_lag_1"].to_list() == pytest.approx([None, None, None, 1.0, 2.0, 3.0])
    assert result["responder_1_lag_1"].to_list() == pytest.approx([None, None, None, 4.0, 5.0, 6.0])


def test_gateway_lags_from_training_date_sets_current_date_id():
    frame = pl.DataFrame(
        {
            "date_id": [3, 3, 4],
            "time_id": [0, 1, 0],
            "symbol_id": [1, 1, 1],
            "responder_6": [0.5, -0.25, 9.0],
        }
    )

    lags = gateway_lags_from_training_date(
        frame.lazy(),
        current_date_id=4,
        responder_columns=("responder_6",),
    )

    assert lags.to_dict(as_series=False) == {
        "date_id": [4, 4],
        "time_id": [0, 1],
        "symbol_id": [1, 1],
        "responder_6_lag_1": [0.5, -0.25],
    }


def test_gateway_lag_cache_reuses_lags_after_time_zero():
    lags = pl.DataFrame(
        {
            "date_id": [4, 4],
            "time_id": [0, 1],
            "symbol_id": [1, 1],
            "responder_6_lag_1": [0.5, -0.25],
        }
    )
    batch_0 = pl.DataFrame({"date_id": [4], "time_id": [0], "symbol_id": [1]})
    batch_1 = pl.DataFrame({"date_id": [4], "time_id": [1], "symbol_id": [1]})

    cache = GatewayResponderLagCache(responder_columns=("responder_6",))

    first = cache.add_to_batch(batch_0, lags)
    second = cache.add_to_batch(batch_1, None)

    assert first["responder_6_lag_1"].to_list() == pytest.approx([0.5])
    assert second["responder_6_lag_1"].to_list() == pytest.approx([-0.25])


def test_daily_last_responder_lags_shift_last_time_per_symbol():
    frame = pl.DataFrame(
        {
            "date_id": [0, 0, 0, 1],
            "time_id": [0, 2, 1, 0],
            "symbol_id": [7, 7, 8, 7],
            "responder_6": [1.0, 2.0, 3.0, 10.0],
        }
    )

    result = daily_last_responder_lags(
        frame.lazy(),
        responder_columns=("responder_6",),
    ).collect()

    assert result.to_dict(as_series=False) == {
        "date_id": [1, 1, 2],
        "symbol_id": [7, 8, 7],
        "responder_6_daily_last_lag_1": [2.0, 3.0, 10.0],
    }


def test_responder_lag_columns_validates_lag():
    assert responder_lag_columns(("responder_0",)) == ("responder_0_lag_1",)
    assert daily_last_responder_lag_columns(("responder_0",)) == ("responder_0_daily_last_lag_1",)
    with pytest.raises(ValueError, match="positive"):
        responder_lag_columns(("responder_0",), date_lag=0)
