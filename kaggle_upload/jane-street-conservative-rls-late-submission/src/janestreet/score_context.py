"""Cross-sectional context features derived from model scores."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl


@dataclass(frozen=True)
class PredictionContextCombiner:
    """Bounded linear combiner over cross-sectional prediction context."""

    feature_columns: tuple[str, ...]
    coefficients: np.ndarray
    alpha: float

    def apply(self, frame: pl.DataFrame, *, output: str = "score_context_prediction") -> pl.DataFrame:
        expr = pl.lit(0.0)
        for column, coefficient in zip(self.feature_columns, self.coefficients, strict=True):
            expr = expr + pl.col(column) * float(coefficient)
        return frame.with_columns(expr.alias(output))


def add_prediction_context(
    frame: pl.DataFrame,
    *,
    prediction: str = "prediction",
    group_columns: Sequence[str] = ("date_id", "time_id"),
    clip_abs: float | None = None,
    prefix: str = "score",
) -> pl.DataFrame:
    """Add leave-one-out cross-sectional prediction context.

    The transform is causal when the full current timestamp batch is available.
    It uses no target information.
    """

    groups = list(group_columns)
    if not groups:
        raise ValueError("group_columns must not be empty")
    clipped = _clipped_expr(prediction, clip_abs)
    pred_col = f"{prefix}_prediction"
    count_col = f"{prefix}_group_size"
    sum_col = f"__{prefix}_prediction_sum"
    return (
        frame.with_columns(clipped.alias(pred_col))
        .with_columns(
            [
                pl.len().over(groups).alias(count_col),
                pl.col(pred_col).sum().over(groups).alias(sum_col),
            ]
        )
        .with_columns(
            pl.when(pl.col(count_col) > 1)
            .then((pl.col(sum_col) - pl.col(pred_col)) / (pl.col(count_col) - 1))
            .otherwise(pl.col(pred_col))
            .alias(f"{prefix}_market_loo")
        )
        .with_columns((pl.col(pred_col) - pl.col(f"{prefix}_market_loo")).alias(f"{prefix}_deviation"))
        .drop(sum_col)
    )


def fit_prediction_context_combiner(
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str] = ("score_market_loo", "score_deviation"),
    target: str = "responder_6",
    weight: str = "weight",
    alpha: float = 1e-3,
    coefficient_min: float = 0.0,
    coefficient_max: float = 1.0,
) -> PredictionContextCombiner:
    """Fit a bounded weighted Ridge combiner for score context."""

    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if coefficient_min > coefficient_max:
        raise ValueError("coefficient_min must be <= coefficient_max")
    features = tuple(feature_columns)
    if not features:
        raise ValueError("feature_columns must not be empty")
    x = frame.select(list(features)).to_numpy().astype(np.float64, copy=False)
    y = frame[target].to_numpy().astype(np.float64, copy=False)
    w = frame[weight].to_numpy().astype(np.float64, copy=False)
    xtwx = x.T @ (x * w[:, None])
    xtwy = x.T @ (y * w)
    system = xtwx + np.eye(len(features), dtype=np.float64) * alpha
    try:
        coefficients = np.linalg.solve(system, xtwy)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, xtwy, rcond=None)[0]
    coefficients = np.clip(coefficients, coefficient_min, coefficient_max)
    return PredictionContextCombiner(
        feature_columns=features,
        coefficients=coefficients,
        alpha=alpha,
    )


def _clipped_expr(prediction: str, clip_abs: float | None) -> pl.Expr:
    expr = pl.col(prediction)
    if clip_abs is None:
        return expr
    if clip_abs <= 0.0:
        raise ValueError("clip_abs must be positive")
    return pl.when(expr > clip_abs).then(clip_abs).when(expr < -clip_abs).then(-clip_abs).otherwise(expr)
