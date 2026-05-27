"""Scoring helpers for the multi-model lab."""

from __future__ import annotations

from math import isfinite

import numpy as np
import polars as pl


TARGET = "responder_6"
WEIGHT = "weight"
KEY_COLUMNS = ("date_id", "time_id", "symbol_id")


def weighted_zero_mean_r2_arrays(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray) -> float:
    """Return the Jane Street weighted zero-mean R2."""

    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    if y.shape != pred.shape or y.shape != w.shape:
        raise ValueError("y_true, y_pred and weight must have the same shape")
    mask = np.isfinite(y) & np.isfinite(pred) & np.isfinite(w) & (w >= 0.0)
    if not np.all(mask):
        y = y[mask]
        pred = pred[mask]
        w = w[mask]
    denominator = float(np.sum(w * y * y))
    if denominator <= 0.0 or not isfinite(denominator):
        raise ValueError("weighted target energy must be positive")
    numerator = float(np.sum(w * (y - pred) * (y - pred)))
    return 1.0 - numerator / denominator


def weighted_scale(y_true: np.ndarray, y_pred: np.ndarray, weight: np.ndarray, *, prior: float = 0.0) -> float:
    """Fit y ~= scale * prediction under weighted squared loss."""

    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    rhs = float(np.sum(w * pred * y))
    lhs = float(np.sum(w * pred * pred))
    if prior > 0.0:
        rhs += prior
        lhs += prior
    if lhs <= 0.0 or not isfinite(lhs):
        return 0.0
    return rhs / lhs


def score_arrays(
    *,
    fold: str,
    candidate: str,
    family: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float | int | str]:
    """Return fold-level sufficient statistics and R2 for one candidate."""

    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    numerator = float(np.sum(w * (y - pred) * (y - pred)))
    denominator = float(np.sum(w * y * y))
    if denominator <= 0.0:
        raise ValueError(f"{fold} has non-positive target energy")
    return {
        "fold": fold,
        "candidate": candidate,
        "family": family,
        "rows": int(y.size),
        "weight_sum": float(np.sum(w)),
        "numerator": numerator,
        "denominator": denominator,
        "weighted_zero_mean_r2": 1.0 - numerator / denominator,
    }


def summarize_scores(scores: pl.DataFrame) -> pl.DataFrame:
    """Aggregate fold scores into candidate-level summaries."""

    if scores.height == 0:
        return pl.DataFrame()
    return (
        scores.group_by(["candidate", "family"])
        .agg(
            pl.col("rows").sum().alias("rows"),
            pl.col("weight_sum").sum().alias("weight_sum"),
            pl.col("numerator").sum().alias("numerator"),
            pl.col("denominator").sum().alias("denominator"),
            pl.col("weighted_zero_mean_r2").mean().alias("mean_fold_r2"),
            pl.col("weighted_zero_mean_r2").min().alias("min_fold_r2"),
            pl.col("weighted_zero_mean_r2").std().fill_null(0.0).alias("std_fold_r2"),
        )
        .with_columns((1.0 - pl.col("numerator") / pl.col("denominator")).alias("global_r2"))
        .sort(["global_r2", "min_fold_r2"], descending=[True, True])
    )
