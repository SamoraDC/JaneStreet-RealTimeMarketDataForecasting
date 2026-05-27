import math

import polars as pl
import pytest

from janestreet.metrics import weighted_zero_mean_r2, weighted_zero_mean_r2_polars


def test_weighted_zero_mean_r2_perfect_prediction_is_one():
    score = weighted_zero_mean_r2(
        y_true=[1.0, -2.0, 3.0],
        y_pred=[1.0, -2.0, 3.0],
        weights=[1.0, 2.0, 0.5],
    )

    assert score == pytest.approx(1.0)


def test_weighted_zero_mean_r2_zero_prediction_is_zero():
    score = weighted_zero_mean_r2(
        y_true=[1.0, -2.0, 3.0],
        y_pred=[0.0, 0.0, 0.0],
        weights=[1.0, 2.0, 0.5],
    )

    assert score == pytest.approx(0.0)


def test_weighted_zero_mean_r2_matches_manual_formula():
    y_true = [2.0, -1.0, 0.5]
    y_pred = [1.5, -0.5, 0.0]
    weights = [3.0, 1.0, 2.0]
    numerator = sum(w * (y - p) ** 2 for y, p, w in zip(y_true, y_pred, weights, strict=True))
    denominator = sum(w * y**2 for y, w in zip(y_true, weights, strict=True))

    assert weighted_zero_mean_r2(y_true, y_pred, weights) == pytest.approx(1.0 - numerator / denominator)


def test_weighted_zero_mean_r2_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="non-negative"):
        weighted_zero_mean_r2([1.0], [1.0], [-1.0])

    with pytest.raises(ValueError, match="non-finite"):
        weighted_zero_mean_r2([math.nan], [1.0], [1.0])

    with pytest.raises(ValueError, match="positive"):
        weighted_zero_mean_r2([0.0], [0.0], [1.0])


def test_weighted_zero_mean_r2_polars_matches_sequence_metric():
    frame = pl.DataFrame(
        {
            "responder_6": [2.0, -1.0, 0.5],
            "prediction": [1.5, -0.5, 0.0],
            "weight": [3.0, 1.0, 2.0],
        }
    )

    assert weighted_zero_mean_r2_polars(frame) == pytest.approx(
        weighted_zero_mean_r2(
            frame["responder_6"].to_list(),
            frame["prediction"].to_list(),
            frame["weight"].to_list(),
        )
    )
