import polars as pl
import pytest

from janestreet.baselines import evaluate_constant_prediction_by_fold
from janestreet.folds import DateFold


def test_evaluate_constant_prediction_by_fold_scores_zero_baseline():
    frame = pl.DataFrame(
        {
            "date_id": [0, 0, 1, 1],
            "weight": [1.0, 2.0, 3.0, 4.0],
            "responder_6": [1.0, -2.0, 0.5, 1.5],
        }
    )
    folds = [DateFold("wf_01", train_start=0, train_end=0, valid_start=1, valid_end=1)]

    result = evaluate_constant_prediction_by_fold(frame.lazy(), folds, prediction_value=0.0)

    assert result["rows"][0] == 2
    assert result["valid_days_present"][0] == 1
    assert result["weighted_zero_mean_r2"][0] == pytest.approx(0.0)

