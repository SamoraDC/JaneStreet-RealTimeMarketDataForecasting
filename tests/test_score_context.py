import polars as pl
import pytest

from janestreet.score_context import add_prediction_context, fit_prediction_context_combiner


def test_add_prediction_context_uses_leave_one_out_mean():
    frame = pl.DataFrame(
        {
            "date_id": [1, 1, 1],
            "time_id": [0, 0, 0],
            "prediction": [1.0, 2.0, 4.0],
        }
    )

    result = add_prediction_context(frame)

    assert result["score_market_loo"].to_list() == pytest.approx([3.0, 2.5, 1.5])
    assert result["score_deviation"].to_list() == pytest.approx([-2.0, -0.5, 2.5])


def test_add_prediction_context_clips_before_market_context():
    frame = pl.DataFrame(
        {
            "date_id": [1, 1],
            "time_id": [0, 0],
            "prediction": [1.0, 100.0],
        }
    )

    result = add_prediction_context(frame, clip_abs=2.0)

    assert result["score_prediction"].to_list() == pytest.approx([1.0, 2.0])
    assert result["score_market_loo"].to_list() == pytest.approx([2.0, 1.0])


def test_fit_prediction_context_combiner_recovers_bounded_weights():
    frame = pl.DataFrame(
        {
            "score_market_loo": [1.0, 2.0, 3.0],
            "score_deviation": [0.5, 0.0, -0.5],
            "responder_6": [1.5, 2.0, 2.5],
            "weight": [1.0, 1.0, 1.0],
        }
    )

    model = fit_prediction_context_combiner(frame, alpha=0.0)
    result = model.apply(frame)

    assert all(0.0 <= coefficient <= 1.0 for coefficient in model.coefficients)
    assert "score_context_prediction" in result.columns
