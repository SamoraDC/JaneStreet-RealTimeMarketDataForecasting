from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from janestreet.submission_inference import (
    DynamicRLSMetaState,
    KaggleRLSSubmissionPredictor,
    PredictionFeatureCache,
    make_prior_rls_state,
)


class LinearFeaturePredictor:
    def __init__(self) -> None:
        self.update_calls = 0

    def update_from_lags(self, lags: pl.DataFrame | None) -> None:
        if lags is not None:
            self.update_calls += 1

    def predict_features(self, test_with_lags: pl.DataFrame) -> pl.DataFrame:
        return test_with_lags.select(
            "date_id",
            "time_id",
            "symbol_id",
            "weight",
            pl.col("feature_00").cast(pl.Float64).alias("tabm_prediction"),
            pl.col("feature_01").cast(pl.Float64).alias("xgboost_prediction"),
        )


def test_prediction_feature_cache_updates_from_shifted_gateway_lags() -> None:
    cache = PredictionFeatureCache(("tabm_prediction", "xgboost_prediction"))
    cache.cache_batch(
        pl.DataFrame(
            {
                "date_id": [4, 4],
                "time_id": [0, 1],
                "symbol_id": [7, 7],
                "weight": [2.0, 3.0],
                "tabm_prediction": [1.0, 2.0],
                "xgboost_prediction": [0.5, 0.25],
            }
        )
    )
    lags = pl.DataFrame(
        {
            "date_id": [5],
            "time_id": [1],
            "symbol_id": [7],
            "responder_6_lag_1": [4.0],
        }
    )

    update = cache.build_lag_update_frame(lags)

    assert update.select(["date_id", "time_id", "symbol_id", "weight", "responder_6"]).row(0) == (
        5,
        1,
        7,
        3.0,
        4.0,
    )
    assert update.select(["tabm_prediction", "xgboost_prediction"]).row(0) == pytest.approx((2.0, 0.25))


def test_dynamic_rls_meta_state_applies_forgetting_update() -> None:
    state = DynamicRLSMetaState(
        ("tabm_prediction",),
        precision=np.array([[10.0]], dtype=np.float64),
        rhs=np.array([5.0], dtype=np.float64),
        forgetting_factor=0.5,
    )

    state.update(pl.DataFrame({"tabm_prediction": [2.0], "responder_6": [3.0], "weight": [4.0]}))

    assert state.precision[0, 0] == pytest.approx(0.5 * 10.0 + 4.0 * 2.0 * 2.0)
    assert state.rhs[0] == pytest.approx(0.5 * 5.0 + 4.0 * 2.0 * 3.0)


def test_kaggle_submission_predictor_updates_meta_only_from_previous_day_lags() -> None:
    base = LinearFeaturePredictor()
    state = DynamicRLSMetaState(
        ("tabm_prediction", "xgboost_prediction"),
        precision=np.eye(2, dtype=np.float64),
        rhs=np.array([1.0, 0.0], dtype=np.float64),
        forgetting_factor=1.0,
    )
    predictor = KaggleRLSSubmissionPredictor(base, state)
    date0 = pl.DataFrame(
        {
            "row_id": [0],
            "date_id": [0],
            "time_id": [0],
            "symbol_id": [1],
            "weight": [1.0],
            "feature_00": [1.0],
            "feature_01": [0.0],
        }
    )

    first = predictor.predict(date0, None)

    assert first["responder_6"].to_list() == pytest.approx([1.0])
    date1 = pl.DataFrame(
        {
            "row_id": [1],
            "date_id": [1],
            "time_id": [0],
            "symbol_id": [1],
            "weight": [1.0],
            "feature_00": [2.0],
            "feature_01": [0.0],
        }
    )
    lags = pl.DataFrame(
        {
            "date_id": [1],
            "time_id": [0],
            "symbol_id": [1],
            "responder_0_lag_1": [0.0],
            "responder_1_lag_1": [0.0],
            "responder_2_lag_1": [0.0],
            "responder_3_lag_1": [0.0],
            "responder_4_lag_1": [0.0],
            "responder_5_lag_1": [0.0],
            "responder_6_lag_1": [2.0],
            "responder_7_lag_1": [0.0],
            "responder_8_lag_1": [0.0],
        }
    )

    second = predictor.predict(date1, lags)

    assert base.update_calls == 1
    assert predictor.last_meta_update_date_id == 1
    assert second.select(["row_id", "responder_6"]).row(0) == pytest.approx((1, 3.0))


def test_make_prior_rls_state_sets_tabm_prior_weight() -> None:
    state = make_prior_rls_state(
        ("tabm_prediction", "xgboost_prediction"),
        ridge_alpha=1000.0,
        forgetting_factor=0.995,
    )

    assert state.beta.tolist() == pytest.approx([1.0, 0.0])
