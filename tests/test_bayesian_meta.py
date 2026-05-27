import numpy as np
import polars as pl
import pytest

from janestreet.bayesian_meta import (
    fit_empirical_bayes_scales,
    fit_hierarchical_means,
    score_prediction_by_fold,
    softmax_from_log_weights,
    summarize_fold_scores,
    weighted_normal_stats,
)


def test_empirical_bayes_scale_shrinks_group_to_global() -> None:
    frame = pl.DataFrame(
        {
            "bucket": ["a", "a", "b", "b"],
            "responder_6": [1.0, 2.0, 2.0, 4.0],
            "prediction": [1.0, 2.0, 1.0, 2.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )

    model = fit_empirical_bayes_scales(
        frame,
        group_columns=["bucket"],
        prediction="prediction",
        prior_strength=1.0,
        min_group_rows=1,
        max_scale=3.0,
    )

    scales = dict(model.parameters.select(["bucket", "_eb_scale"]).iter_rows())
    assert 1.0 < model.fallback_scale < 2.0
    assert scales["a"] > 1.0
    assert scales["a"] < model.fallback_scale
    assert scales["b"] > model.fallback_scale
    assert scales["b"] < 2.0


def test_hierarchical_mean_falls_back_for_unseen_group() -> None:
    train = pl.DataFrame(
        {
            "bucket": ["a", "a"],
            "responder_6": [1.0, 3.0],
            "weight": [1.0, 1.0],
        }
    )
    valid = pl.DataFrame({"bucket": ["a", "new"]})

    model = fit_hierarchical_means(
        train,
        group_columns=["bucket"],
        prior_strength=0.0,
        min_group_rows=1,
    )
    result = model.apply(valid, output="prediction")

    assert result["prediction"].to_list() == pytest.approx([2.0, 2.0])


def test_score_prediction_summary_matches_manual_formula() -> None:
    frame = pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01"],
            "responder_6": [1.0, 2.0],
            "prediction": [0.0, 2.0],
            "weight": [2.0, 1.0],
        }
    )

    scores = score_prediction_by_fold(frame, prediction="prediction")
    summary = summarize_fold_scores(scores)

    assert summary["global_r2"] == pytest.approx(1.0 - 2.0 / 6.0)


def test_weighted_normal_stats_matches_numpy() -> None:
    frame = pl.DataFrame(
        {
            "x0": [1.0, 2.0, 3.0],
            "x1": [2.0, 0.0, 1.0],
            "responder_6": [1.0, -1.0, 2.0],
            "weight": [1.0, 2.0, 3.0],
        }
    )

    gram, rhs = weighted_normal_stats(frame, feature_columns=["x0", "x1"])
    arrays = frame.select(["x0", "x1", "responder_6", "weight"]).to_numpy()
    x = arrays[:, :2]
    y = arrays[:, 2]
    weight = arrays[:, 3]

    assert gram == pytest.approx(x.T @ (x * weight[:, None]))
    assert rhs == pytest.approx(x.T @ (weight * y))


def test_softmax_from_log_weights_is_stable() -> None:
    weights = softmax_from_log_weights(np.array([1000.0, 999.0], dtype=np.float64))

    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] > weights[1]
