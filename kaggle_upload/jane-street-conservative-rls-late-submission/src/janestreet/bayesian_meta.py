"""Cheap causal Bayesian-style meta models for saved predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl


TARGET = "responder_6"
WEIGHT = "weight"


@dataclass(frozen=True)
class EmpiricalBayesScaleModel:
    """Shrunk multiplicative scales for a prediction column."""

    group_columns: tuple[str, ...]
    parameters: pl.DataFrame
    fallback_scale: float

    def apply(self, frame: pl.DataFrame, *, prediction: str, output: str) -> pl.DataFrame:
        if not self.group_columns:
            return frame.with_columns((self.fallback_scale * pl.col(prediction)).alias(output))
        joined = frame.join(self.parameters, on=list(self.group_columns), how="left")
        scale = pl.coalesce(pl.col("_eb_scale"), pl.lit(self.fallback_scale))
        return joined.with_columns((scale * pl.col(prediction)).alias(output)).drop(
            [name for name in ("_eb_rows", "_eb_den", "_eb_scale") if name in joined.columns]
        )


@dataclass(frozen=True)
class HierarchicalMeanModel:
    """Shrunk conditional means for a minimal Bayesian baseline."""

    group_columns: tuple[str, ...]
    parameters: pl.DataFrame
    fallback_mean: float

    def apply(self, frame: pl.DataFrame, *, output: str) -> pl.DataFrame:
        if not self.group_columns:
            return frame.with_columns(pl.lit(self.fallback_mean).alias(output))
        joined = frame.join(self.parameters, on=list(self.group_columns), how="left")
        mean = pl.coalesce(pl.col("_eb_mean"), pl.lit(self.fallback_mean))
        return joined.with_columns(mean.alias(output)).drop(
            [name for name in ("_eb_rows", "_eb_weight_sum", "_eb_mean") if name in joined.columns]
        )


def score_prediction_by_fold(
    frame: pl.DataFrame,
    *,
    prediction: str,
    target: str = TARGET,
    weight: str = WEIGHT,
) -> pl.DataFrame:
    """Return per-fold weighted zero-mean R2 numerators and denominators."""

    target_expr = pl.col(target).cast(pl.Float64)
    prediction_expr = pl.col(prediction).cast(pl.Float64)
    weight_expr = pl.col(weight).cast(pl.Float64)
    return (
        frame.group_by("fold")
        .agg(
            pl.len().cast(pl.Int64).alias("rows"),
            weight_expr.sum().alias("weight_sum"),
            (weight_expr * (target_expr - prediction_expr).pow(2)).sum().alias("numerator"),
            (weight_expr * target_expr.pow(2)).sum().alias("denominator"),
        )
        .with_columns((1.0 - pl.col("numerator") / pl.col("denominator")).alias("weighted_zero_mean_r2"))
        .sort("fold")
    )


def summarize_fold_scores(scores: pl.DataFrame) -> dict[str, float | int]:
    """Summarize fold-level metric rows into the global competition metric."""

    totals = scores.select(
        pl.col("rows").sum().alias("rows"),
        pl.col("weight_sum").sum().alias("weight_sum"),
        pl.col("numerator").sum().alias("numerator"),
        pl.col("denominator").sum().alias("denominator"),
        pl.col("weighted_zero_mean_r2").mean().alias("mean_fold_r2"),
        pl.col("weighted_zero_mean_r2").std().alias("std_fold_r2"),
        pl.col("weighted_zero_mean_r2").min().alias("min_fold_r2"),
        pl.col("weighted_zero_mean_r2").max().alias("max_fold_r2"),
    ).row(0, named=True)
    denominator = float(totals["denominator"])
    if denominator <= 0.0:
        raise ValueError("denominator must be positive")
    return {
        "folds": int(scores.height),
        "rows": int(totals["rows"]),
        "weight_sum": float(totals["weight_sum"]),
        "numerator": float(totals["numerator"]),
        "denominator": denominator,
        "global_r2": 1.0 - float(totals["numerator"]) / denominator,
        "mean_fold_r2": float(totals["mean_fold_r2"]),
        "std_fold_r2": 0.0 if totals["std_fold_r2"] is None else float(totals["std_fold_r2"]),
        "min_fold_r2": float(totals["min_fold_r2"]),
        "max_fold_r2": float(totals["max_fold_r2"]),
    }


def fit_empirical_bayes_scales(
    frame: pl.DataFrame,
    *,
    group_columns: Sequence[str],
    prediction: str,
    target: str = TARGET,
    weight: str = WEIGHT,
    prior_strength: float = 8.0,
    min_group_rows: int = 1_000,
    min_scale: float = 0.0,
    max_scale: float = 2.0,
) -> EmpiricalBayesScaleModel:
    """Fit shrunk scales for `prediction`.

    This is equivalent to a conjugate empirical Bayes shrink around the global
    scale, using prediction energy as the evidence unit.
    """

    if prior_strength < 0.0:
        raise ValueError("prior_strength must be non-negative")
    if min_group_rows <= 0:
        raise ValueError("min_group_rows must be positive")
    if min_scale > max_scale:
        raise ValueError("min_scale must be <= max_scale")
    groups = tuple(group_columns)
    stats = frame.select(
        (pl.col(weight) * pl.col(prediction) * pl.col(target)).sum().alias("num"),
        (pl.col(weight) * pl.col(prediction).pow(2)).sum().alias("den"),
    ).row(0, named=True)
    global_scale = _safe_clipped_ratio(
        float(stats["num"]),
        float(stats["den"]),
        fallback=1.0,
        min_value=min_scale,
        max_value=max_scale,
    )
    if not groups:
        return EmpiricalBayesScaleModel(groups, pl.DataFrame({"_eb_scale": [global_scale]}), global_scale)

    raw = (
        frame.group_by(list(groups))
        .agg(
            pl.len().alias("_eb_rows"),
            (pl.col(weight) * pl.col(prediction) * pl.col(target)).sum().alias("_eb_num"),
            (pl.col(weight) * pl.col(prediction).pow(2)).sum().alias("_eb_den"),
        )
    )
    positive_den = raw.filter(pl.col("_eb_den") > 1e-12)
    typical_den = float(positive_den["_eb_den"].median()) if positive_den.height else 0.0
    prior_den = max(0.0, prior_strength * typical_den)
    parameters = (
        raw.with_columns(
            pl.when((pl.col("_eb_rows") >= min_group_rows) & (pl.col("_eb_den") > 1e-12))
            .then(((pl.col("_eb_num") + prior_den * global_scale) / (pl.col("_eb_den") + prior_den)).clip(min_scale, max_scale))
            .otherwise(pl.lit(global_scale))
            .alias("_eb_scale")
        )
        .select(list(groups) + ["_eb_rows", "_eb_den", "_eb_scale"])
        .sort(list(groups))
    )
    return EmpiricalBayesScaleModel(groups, parameters, global_scale)


def fit_hierarchical_means(
    frame: pl.DataFrame,
    *,
    group_columns: Sequence[str],
    target: str = TARGET,
    weight: str = WEIGHT,
    prior_strength: float = 8.0,
    min_group_rows: int = 1_000,
) -> HierarchicalMeanModel:
    """Fit shrunk weighted means for a minimal conditional expectation model."""

    if prior_strength < 0.0:
        raise ValueError("prior_strength must be non-negative")
    if min_group_rows <= 0:
        raise ValueError("min_group_rows must be positive")
    groups = tuple(group_columns)
    stats = frame.select(
        (pl.col(weight) * pl.col(target)).sum().alias("target_sum"),
        pl.col(weight).sum().alias("weight_sum"),
    ).row(0, named=True)
    weight_sum = float(stats["weight_sum"])
    fallback_mean = 0.0 if weight_sum <= 1e-12 else float(stats["target_sum"]) / weight_sum
    if not groups:
        return HierarchicalMeanModel(groups, pl.DataFrame({"_eb_mean": [fallback_mean]}), fallback_mean)

    raw = frame.group_by(list(groups)).agg(
        pl.len().alias("_eb_rows"),
        pl.col(weight).sum().alias("_eb_weight_sum"),
        (pl.col(weight) * pl.col(target)).sum().alias("_eb_target_sum"),
    )
    positive_weight = raw.filter(pl.col("_eb_weight_sum") > 1e-12)
    typical_weight = float(positive_weight["_eb_weight_sum"].median()) if positive_weight.height else 0.0
    prior_weight = max(0.0, prior_strength * typical_weight)
    parameters = (
        raw.with_columns(
            pl.when((pl.col("_eb_rows") >= min_group_rows) & (pl.col("_eb_weight_sum") > 1e-12))
            .then((pl.col("_eb_target_sum") + prior_weight * fallback_mean) / (pl.col("_eb_weight_sum") + prior_weight))
            .otherwise(pl.lit(fallback_mean))
            .alias("_eb_mean")
        )
        .select(list(groups) + ["_eb_rows", "_eb_weight_sum", "_eb_mean"])
        .sort(list(groups))
    )
    return HierarchicalMeanModel(groups, parameters, fallback_mean)


def softmax_from_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """Return a stable softmax for Bayesian model averaging weights."""

    if log_weights.ndim != 1:
        raise ValueError("log_weights must be one-dimensional")
    if not np.all(np.isfinite(log_weights)):
        raise ValueError("log_weights must be finite")
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(log_weights.shape[0], 1.0 / log_weights.shape[0], dtype=np.float64)
    return weights / total


def weighted_normal_stats(
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    target: str = TARGET,
    weight: str = WEIGHT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `X'WX` and `X'Wy` for small dense meta models."""

    features = tuple(feature_columns)
    if not features:
        raise ValueError("feature_columns must not be empty")
    expressions: list[pl.Expr] = []
    for i, left in enumerate(features):
        for j, right in enumerate(features):
            if j < i:
                continue
            expressions.append((pl.col(weight) * pl.col(left) * pl.col(right)).sum().alias(f"g_{i}_{j}"))
    for i, column in enumerate(features):
        expressions.append((pl.col(weight) * pl.col(column) * pl.col(target)).sum().alias(f"r_{i}"))
    row = frame.select(expressions).row(0, named=True)
    gram = np.zeros((len(features), len(features)), dtype=np.float64)
    rhs = np.zeros(len(features), dtype=np.float64)
    for i in range(len(features)):
        for j in range(i, len(features)):
            value = float(row[f"g_{i}_{j}"])
            gram[i, j] = value
            gram[j, i] = value
        rhs[i] = float(row[f"r_{i}"])
    return gram, rhs


def _safe_clipped_ratio(
    numerator: float,
    denominator: float,
    *,
    fallback: float,
    min_value: float,
    max_value: float,
) -> float:
    if denominator <= 1e-12:
        return min(max(fallback, min_value), max_value)
    value = numerator / denominator
    return min(max(value, min_value), max_value)
