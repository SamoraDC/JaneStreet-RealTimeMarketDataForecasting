from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_clock_tournament.py"
    spec = importlib.util.spec_from_file_location("run_clock_tournament", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_quantile_clock_binner_uses_fitted_thresholds() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "ot_source_activity": [0.0, 1.0, 2.0, 3.0],
            "time_id": [0, 1, 2, 3],
        }
    )

    binner = module.fit_clock_binner(
        frame,
        clock_name="row_activity",
        bucket_count=2,
        max_time_id=3,
    )
    output = module.apply_clock_binner(frame, binner)

    assert binner.thresholds == (1.5,)
    assert output["clock_bucket"].to_list() == [0, 0, 1, 1]


def test_grouped_simplex_prefers_different_components_by_clock() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "clock_bucket": [0, 0, 1, 1],
            "left_prediction": [1.0, 2.0, 0.0, 0.0],
            "right_prediction": [0.0, 0.0, 3.0, 4.0],
            "responder_6": [1.0, 2.0, 3.0, 4.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )

    weights = module.fit_grouped_simplex_weights(
        frame,
        group_columns=("clock_bucket",),
        prediction_columns=("left_prediction", "right_prediction"),
        min_group_rows=2,
    )
    scored = weights.apply(frame, output="prediction")

    assert weights.parameters.height == 2
    assert scored["prediction"].to_list() == [1.0, 2.0, 3.0, 4.0]


def test_weight_bucket_slice_aggregates_weighted_r2_components() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "weight_bucket": ["low", "low", "high"],
            "responder_6": [1.0, 2.0, 4.0],
            "prediction": [1.0, 1.0, 2.0],
            "weight": [1.0, 2.0, 3.0],
        }
    )

    sliced = module._weight_bucket_slice(frame, prediction="prediction")

    low = sliced.filter(pl.col("weight_bucket") == "low").row(0, named=True)
    high = sliced.filter(pl.col("weight_bucket") == "high").row(0, named=True)
    assert low["numerator"] == 2.0
    assert low["denominator"] == 9.0
    assert low["weighted_zero_mean_r2"] == 1.0 - 2.0 / 9.0
    assert high["numerator"] == 12.0
    assert high["denominator"] == 48.0
    assert high["weighted_zero_mean_r2"] == 0.75

