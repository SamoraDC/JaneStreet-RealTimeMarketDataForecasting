from __future__ import annotations

import polars as pl
import pytest

from janestreet.submission_models import build_lagged_tabm_update_frame


def test_build_lagged_tabm_update_frame_uses_shifted_previous_day_cache() -> None:
    cached = pl.DataFrame(
        {
            "date_id": [10],
            "time_id": [3],
            "symbol_id": [2],
            "weight": [4.0],
            "feature_00": [1.5],
        }
    )
    lags = pl.DataFrame(
        {
            "date_id": [11],
            "time_id": [3],
            "symbol_id": [2],
            "responder_6_lag_1": [0.7],
            "responder_0_lag_1": [-0.2],
        }
    )

    update = build_lagged_tabm_update_frame(
        [cached],
        lags,
        target_columns=("responder_6", "responder_0"),
    )

    assert update.select(["date_id", "time_id", "symbol_id", "weight", "feature_00"]).row(0) == (
        11,
        3,
        2,
        4.0,
        1.5,
    )
    assert update.select(["responder_6", "responder_0"]).row(0) == pytest.approx((0.7, -0.2))


def test_build_lagged_tabm_update_frame_rejects_missing_target_lag() -> None:
    cached = pl.DataFrame({"date_id": [10], "time_id": [0], "symbol_id": [1], "weight": [1.0]})
    lags = pl.DataFrame({"date_id": [11], "time_id": [0], "symbol_id": [1]})

    with pytest.raises(ValueError, match="responder_6_lag_1"):
        build_lagged_tabm_update_frame([cached], lags, target_columns=("responder_6",))
