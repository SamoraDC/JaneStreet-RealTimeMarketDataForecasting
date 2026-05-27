import polars as pl
import pytest

from janestreet.calibration import (
    add_abs_prediction_bucket,
    fit_abs_prediction_thresholds,
    fit_shrinkage_calibrator,
)


def test_fit_global_shrinkage_recovers_weighted_alpha():
    frame = pl.DataFrame(
        {
            "responder_6": [1.0, 2.0, -1.0],
            "prediction": [2.0, 4.0, -2.0],
            "weight": [1.0, 2.0, 3.0],
        }
    )

    calibrator = fit_shrinkage_calibrator(frame, name="global", min_group_rows=1)
    result = calibrator.apply(frame)

    assert calibrator.fallback_alpha == pytest.approx(0.5)
    assert result["calibrated_prediction"].to_list() == pytest.approx([1.0, 2.0, -1.0])


def test_group_shrinkage_falls_back_for_small_groups():
    frame = pl.DataFrame(
        {
            "group": ["a", "a", "b"],
            "responder_6": [1.0, 2.0, 10.0],
            "prediction": [2.0, 4.0, 10.0],
            "weight": [1.0, 1.0, 1.0],
        }
    )

    calibrator = fit_shrinkage_calibrator(
        frame,
        name="by_group",
        group_columns=["group"],
        min_group_rows=2,
    )
    result = calibrator.apply(frame).sort("group")

    assert calibrator.parameters.filter(pl.col("group") == "a")["_calibration_alpha"][0] == pytest.approx(0.5)
    assert calibrator.parameters.filter(pl.col("group") == "b")["_calibration_alpha"][0] == pytest.approx(
        calibrator.fallback_alpha
    )
    assert "calibrated_prediction" in result.columns


def test_shrinkage_clip_bounds_extreme_predictions():
    frame = pl.DataFrame(
        {
            "responder_6": [1.0, 1.0],
            "prediction": [1.0, 100.0],
            "weight": [1.0, 1.0],
        }
    )

    calibrator = fit_shrinkage_calibrator(frame, name="clipped", clip_abs=2.0, min_group_rows=1)
    result = calibrator.apply(frame)

    assert result["calibrated_prediction"].max() <= 2.0


def test_abs_prediction_buckets_use_calibration_thresholds():
    frame = pl.DataFrame({"prediction": [-10.0, -1.0, 0.0, 1.0, 10.0]})

    thresholds = fit_abs_prediction_thresholds(frame)
    result = add_abs_prediction_bucket(frame, thresholds)

    assert set(result["prediction_abs_bucket"]) <= {"p00_p50", "p50_p90", "p90_p99", "p99_p100"}
