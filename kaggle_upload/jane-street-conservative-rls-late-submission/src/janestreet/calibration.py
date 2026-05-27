"""Conservative prediction calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl


@dataclass(frozen=True)
class ShrinkageCalibrator:
    """Multiplicative shrinkage fitted on a calibration window."""

    name: str
    group_columns: tuple[str, ...]
    parameters: pl.DataFrame
    fallback_alpha: float
    clip_abs: float | None = None

    def apply(
        self,
        frame: pl.DataFrame,
        *,
        prediction: str = "prediction",
        output: str = "calibrated_prediction",
    ) -> pl.DataFrame:
        """Apply clipping and multiplicative shrinkage to predictions."""

        calibrated = frame.with_columns(
            _clipped_prediction_expr(prediction, self.clip_abs).alias("_calibration_prediction")
        )
        if self.group_columns:
            calibrated = calibrated.join(self.parameters, on=list(self.group_columns), how="left")
            alpha_expr = pl.coalesce(pl.col("_calibration_alpha"), pl.lit(self.fallback_alpha))
            drop_cols = ["_calibration_prediction", "_calibration_alpha"]
        else:
            alpha_expr = pl.lit(self.fallback_alpha)
            drop_cols = ["_calibration_prediction"]

        return calibrated.with_columns((pl.col("_calibration_prediction") * alpha_expr).alias(output)).drop(
            drop_cols
        )


def fit_shrinkage_calibrator(
    frame: pl.DataFrame,
    *,
    name: str,
    group_columns: Sequence[str] = (),
    target: str = "responder_6",
    prediction: str = "prediction",
    weight: str = "weight",
    min_group_rows: int = 1_000,
    alpha_min: float = 0.0,
    alpha_max: float = 1.0,
    clip_abs: float | None = None,
) -> ShrinkageCalibrator:
    """Fit alpha in `target ~= alpha * prediction` with conservative bounds.

    The fit is weighted least squares on a calibration window. Group alphas with
    too few rows or non-positive prediction energy fall back to the global alpha.
    """

    if min_group_rows <= 0:
        raise ValueError("min_group_rows must be positive")
    if alpha_min > alpha_max:
        raise ValueError("alpha_min must be <= alpha_max")
    if frame.height == 0:
        raise ValueError("calibration frame must not be empty")

    groups = tuple(group_columns)
    working = frame.with_columns(_clipped_prediction_expr(prediction, clip_abs).alias("_calibration_prediction"))
    fallback_alpha = _bounded_alpha(
        numerator=working.select(
            (pl.col(weight) * pl.col(target) * pl.col("_calibration_prediction")).sum()
        ).item(),
        denominator=working.select(
            (pl.col(weight) * pl.col("_calibration_prediction").pow(2)).sum()
        ).item(),
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )

    if not groups:
        return ShrinkageCalibrator(
            name=name,
            group_columns=groups,
            parameters=pl.DataFrame({"_calibration_alpha": [fallback_alpha]}),
            fallback_alpha=fallback_alpha,
            clip_abs=clip_abs,
        )

    parameters = (
        working.group_by(list(groups))
        .agg(
            pl.len().alias("_calibration_rows"),
            (pl.col(weight) * pl.col(target) * pl.col("_calibration_prediction")).sum().alias("_alpha_num"),
            (pl.col(weight) * pl.col("_calibration_prediction").pow(2)).sum().alias("_alpha_den"),
        )
        .with_columns(
            pl.when((pl.col("_calibration_rows") >= min_group_rows) & (pl.col("_alpha_den") > 1e-12))
            .then((pl.col("_alpha_num") / pl.col("_alpha_den")).clip(alpha_min, alpha_max))
            .otherwise(pl.lit(fallback_alpha))
            .alias("_calibration_alpha")
        )
        .select(list(groups) + ["_calibration_alpha"])
    )
    return ShrinkageCalibrator(
        name=name,
        group_columns=groups,
        parameters=parameters,
        fallback_alpha=fallback_alpha,
        clip_abs=clip_abs,
    )


def fit_abs_prediction_thresholds(
    frame: pl.DataFrame,
    *,
    prediction: str = "prediction",
    quantiles: tuple[float, float, float] = (0.50, 0.90, 0.99),
) -> dict[str, float]:
    """Fit absolute-prediction bucket thresholds on a calibration window."""

    if len(quantiles) != 3:
        raise ValueError("quantiles must contain exactly three values")
    if any(q <= 0.0 or q >= 1.0 for q in quantiles):
        raise ValueError("quantiles must be between 0 and 1")
    row = frame.select(
        [
            pl.col(prediction).abs().quantile(quantile).alias(f"q{idx}")
            for idx, quantile in enumerate(quantiles)
        ]
    ).row(0, named=True)
    return {"q50": float(row["q0"]), "q90": float(row["q1"]), "q99": float(row["q2"])}


def add_abs_prediction_bucket(
    frame: pl.DataFrame,
    thresholds: dict[str, float],
    *,
    prediction: str = "prediction",
    output: str = "prediction_abs_bucket",
) -> pl.DataFrame:
    """Add a causal bucket for prediction magnitude."""

    abs_pred = pl.col(prediction).abs()
    return frame.with_columns(
        pl.when(abs_pred <= thresholds["q50"])
        .then(pl.lit("p00_p50"))
        .when(abs_pred <= thresholds["q90"])
        .then(pl.lit("p50_p90"))
        .when(abs_pred <= thresholds["q99"])
        .then(pl.lit("p90_p99"))
        .otherwise(pl.lit("p99_p100"))
        .alias(output)
    )


def _bounded_alpha(
    *,
    numerator: float | int | None,
    denominator: float | int | None,
    alpha_min: float,
    alpha_max: float,
) -> float:
    if denominator is None or float(denominator) <= 1e-12:
        return 0.0
    if numerator is None:
        return 0.0
    alpha = float(numerator) / float(denominator)
    return min(max(alpha, alpha_min), alpha_max)


def _clipped_prediction_expr(prediction: str, clip_abs: float | None) -> pl.Expr:
    expr = pl.col(prediction)
    if clip_abs is None:
        return expr
    if clip_abs <= 0.0:
        raise ValueError("clip_abs must be positive")
    return pl.when(expr > clip_abs).then(clip_abs).when(expr < -clip_abs).then(-clip_abs).otherwise(expr)
