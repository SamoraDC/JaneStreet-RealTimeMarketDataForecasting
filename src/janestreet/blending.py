"""Prediction blending utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl


@dataclass(frozen=True)
class GroupedBlendWeights:
    """Convex blend weights fitted globally with optional group overrides."""

    group_columns: tuple[str, ...]
    parameters: pl.DataFrame
    fallback_weight: float

    def apply(
        self,
        frame: pl.DataFrame,
        *,
        left_prediction: str,
        right_prediction: str,
        output: str = "blend_prediction",
    ) -> pl.DataFrame:
        """Apply grouped blend weights, falling back to the global weight."""

        if not self.group_columns:
            return add_convex_blend_prediction(
                frame,
                blend_weight=self.fallback_weight,
                left_prediction=left_prediction,
                right_prediction=right_prediction,
                output=output,
            )
        joined = frame.join(self.parameters, on=list(self.group_columns), how="left")
        blend_weight = pl.coalesce(pl.col("_blend_weight"), pl.lit(self.fallback_weight))
        return joined.with_columns(
            (
                blend_weight * pl.col(left_prediction)
                + (1.0 - blend_weight) * pl.col(right_prediction)
            ).alias(output)
        ).drop([name for name in ("_blend_rows", "_blend_weight") if name in joined.columns])


def fit_convex_blend_weight(
    frame: pl.DataFrame,
    *,
    left_prediction: str,
    right_prediction: str,
    target: str = "responder_6",
    weight: str = "weight",
) -> float:
    """Fit lambda for `lambda * left + (1 - lambda) * right`.

    The closed-form weighted least-squares solution is clipped to `[0, 1]`.
    """

    row = frame.select(
        [
            (
                pl.col(weight)
                * (pl.col(left_prediction) - pl.col(right_prediction))
                * (pl.col(target) - pl.col(right_prediction))
            )
            .sum()
            .alias("numerator"),
            (
                pl.col(weight) * (pl.col(left_prediction) - pl.col(right_prediction)).pow(2)
            )
            .sum()
            .alias("denominator"),
        ]
    ).row(0, named=True)
    denominator = float(row["denominator"])
    if denominator <= 1e-12:
        return 0.0
    blend_weight = float(row["numerator"]) / denominator
    return min(max(blend_weight, 0.0), 1.0)


def fit_grouped_convex_blend_weights(
    frame: pl.DataFrame,
    *,
    group_columns: Sequence[str],
    left_prediction: str,
    right_prediction: str,
    target: str = "responder_6",
    weight: str = "weight",
    min_group_rows: int = 1_000,
) -> GroupedBlendWeights:
    """Fit convex blend weights by group with a global fallback."""

    if min_group_rows <= 0:
        raise ValueError("min_group_rows must be positive")
    groups = tuple(group_columns)
    fallback_weight = fit_convex_blend_weight(
        frame,
        left_prediction=left_prediction,
        right_prediction=right_prediction,
        target=target,
        weight=weight,
    )
    if not groups:
        return GroupedBlendWeights(
            group_columns=groups,
            parameters=pl.DataFrame({"_blend_weight": [fallback_weight]}),
            fallback_weight=fallback_weight,
        )

    diff = pl.col(left_prediction) - pl.col(right_prediction)
    target_diff = pl.col(target) - pl.col(right_prediction)
    parameters = (
        frame.group_by(list(groups))
        .agg(
            pl.len().alias("_blend_rows"),
            (pl.col(weight) * diff * target_diff).sum().alias("_blend_num"),
            (pl.col(weight) * diff.pow(2)).sum().alias("_blend_den"),
        )
        .with_columns(
            pl.when((pl.col("_blend_rows") >= min_group_rows) & (pl.col("_blend_den") > 1e-12))
            .then((pl.col("_blend_num") / pl.col("_blend_den")).clip(0.0, 1.0))
            .otherwise(pl.lit(fallback_weight))
            .alias("_blend_weight")
        )
        .select(list(groups) + ["_blend_rows", "_blend_weight"])
    )
    return GroupedBlendWeights(
        group_columns=groups,
        parameters=parameters,
        fallback_weight=fallback_weight,
    )


def fit_simplex_blend_weights(
    frame: pl.DataFrame,
    *,
    prediction_columns: Sequence[str],
    target: str = "responder_6",
    weight: str = "weight",
) -> dict[str, float]:
    """Fit non-negative weights summing to 1 over multiple predictions."""

    columns = tuple(prediction_columns)
    if not columns:
        raise ValueError("prediction_columns must not be empty")
    arrays = frame.select(list(columns) + [target, weight]).to_numpy()
    predictions = arrays[:, : len(columns)].astype(np.float64, copy=False)
    y = arrays[:, len(columns)].astype(np.float64, copy=False)
    sample_weight = arrays[:, len(columns) + 1].astype(np.float64, copy=False)
    if np.any(sample_weight < 0.0):
        raise ValueError("weights must be non-negative")
    if float(np.sum(sample_weight)) <= 0.0:
        raise ValueError("sum of weights must be positive")

    best_weights: np.ndarray | None = None
    best_loss = np.inf
    n_columns = len(columns)
    for mask in range(1, 1 << n_columns):
        active = [idx for idx in range(n_columns) if mask & (1 << idx)]
        candidate = _fit_simplex_active_set(predictions[:, active], y, sample_weight)
        if candidate is None:
            continue
        full = np.zeros(n_columns, dtype=np.float64)
        full[active] = candidate
        residual = y - predictions @ full
        loss = float(np.sum(sample_weight * residual * residual))
        if loss < best_loss:
            best_loss = loss
            best_weights = full

    if best_weights is None:
        best_weights = np.full(n_columns, 1.0 / n_columns, dtype=np.float64)
    best_weights[best_weights < 1e-12] = 0.0
    best_weights = best_weights / np.sum(best_weights)
    return {column: float(value) for column, value in zip(columns, best_weights, strict=True)}


def add_simplex_blend_prediction(
    frame: pl.DataFrame,
    *,
    weights: dict[str, float],
    output: str = "ensemble_prediction",
) -> pl.DataFrame:
    """Add a prediction column from fitted simplex weights."""

    if not weights:
        raise ValueError("weights must not be empty")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1")
    if any(value < -1e-12 for value in weights.values()):
        raise ValueError("weights must be non-negative")
    expression = sum(float(value) * pl.col(column) for column, value in weights.items())
    return frame.with_columns(expression.alias(output))


def add_convex_blend_prediction(
    frame: pl.DataFrame,
    *,
    blend_weight: float,
    left_prediction: str,
    right_prediction: str,
    output: str = "blend_prediction",
) -> pl.DataFrame:
    """Add `blend_weight * left + (1 - blend_weight) * right`."""

    if blend_weight < 0.0 or blend_weight > 1.0:
        raise ValueError("blend_weight must be in [0, 1]")
    return frame.with_columns(
        (
            blend_weight * pl.col(left_prediction)
            + (1.0 - blend_weight) * pl.col(right_prediction)
        ).alias(output)
    )


def _fit_simplex_active_set(
    predictions: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
) -> np.ndarray | None:
    n_active = predictions.shape[1]
    weighted_predictions = predictions * sample_weight[:, None]
    gram = predictions.T @ weighted_predictions
    rhs = predictions.T @ (sample_weight * target)
    system = np.zeros((n_active + 1, n_active + 1), dtype=np.float64)
    system[:n_active, :n_active] = gram
    system[:n_active, n_active] = 1.0
    system[n_active, :n_active] = 1.0
    target_rhs = np.zeros(n_active + 1, dtype=np.float64)
    target_rhs[:n_active] = rhs
    target_rhs[n_active] = 1.0
    try:
        solution = np.linalg.solve(system, target_rhs)[:n_active]
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(system, target_rhs, rcond=None)[0][:n_active]
    if np.any(solution < -1e-9):
        return None
    solution = np.clip(solution, 0.0, None)
    total = float(np.sum(solution))
    if total <= 0.0:
        return None
    return solution / total
