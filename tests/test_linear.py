import polars as pl
import numpy as np

from janestreet.folds import DateFold
from janestreet.linear import build_weighted_ridge_fit_data, evaluate_ridge, fit_weighted_ridge, solve_weighted_ridge


def test_weighted_ridge_fits_simple_linear_relation():
    frame = pl.DataFrame(
        {
            "date_id": [0, 0, 1, 1, 2, 2],
            "feature_00": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "feature_01": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "responder_6": [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
        }
    )
    fold = DateFold("toy", train_start=0, train_end=1, valid_start=2, valid_end=2)

    model = fit_weighted_ridge(
        frame.lazy(),
        fold,
        feature_columns=("feature_00", "feature_01"),
        alpha=0.0,
        chunk_days=1,
    )
    result = evaluate_ridge(frame.lazy(), fold, model, chunk_days=1)

    assert result["weighted_zero_mean_r2"] > 0.99


def test_ridge_fit_data_can_be_reused_across_alphas():
    frame = pl.DataFrame(
        {
            "date_id": [0, 0, 1, 1],
            "feature_00": [0.0, 1.0, 2.0, 3.0],
            "weight": [1.0, 2.0, 3.0, 4.0],
            "responder_6": [1.0, 3.0, 5.0, 7.0],
        }
    )
    fold = DateFold("toy", train_start=0, train_end=1, valid_start=1, valid_end=1)

    fit_data = build_weighted_ridge_fit_data(
        frame.lazy(),
        fold,
        feature_columns=("feature_00",),
        chunk_days=1,
    )
    low_alpha = solve_weighted_ridge(fit_data, alpha=0.0)
    high_alpha = solve_weighted_ridge(fit_data, alpha=1000.0)

    assert low_alpha.alpha == 0.0
    assert high_alpha.alpha == 1000.0
    assert abs(low_alpha.coefficients[0]) > abs(high_alpha.coefficients[0])


def test_ridge_fit_data_matches_explicit_weighted_design_matrix():
    frame = pl.DataFrame(
        {
            "date_id": [0, 0, 1, 1, 2],
            "feature_00": [0.0, 1.0, 2.0, 3.0, 4.0],
            "feature_01": [1.0, -2.0, 0.5, 4.0, -1.0],
            "weight": [1.0, 0.5, 3.0, 2.0, 1.5],
            "responder_6": [0.2, -0.4, 0.8, 1.2, -0.1],
        }
    )
    fold = DateFold("toy", train_start=0, train_end=2, valid_start=2, valid_end=2)

    fit_data = build_weighted_ridge_fit_data(
        frame.lazy(),
        fold,
        feature_columns=("feature_00", "feature_01"),
        chunk_days=1,
    )

    x = frame.select(["feature_00", "feature_01"]).to_numpy()
    x = (x - fit_data.means) / fit_data.scales
    design = np.column_stack([np.ones(x.shape[0]), x])
    y = frame["responder_6"].to_numpy()
    w = frame["weight"].to_numpy()

    assert np.allclose(fit_data.xtwx, design.T @ (design * w[:, None]))
    assert np.allclose(fit_data.xtwy, design.T @ (y * w))
