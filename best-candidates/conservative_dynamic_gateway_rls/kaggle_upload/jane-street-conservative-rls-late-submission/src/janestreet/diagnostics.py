"""Diagnostics for validation slices."""

from collections.abc import Sequence

import polars as pl


def aggregate_weighted_r2_by_slice(
    frame: pl.DataFrame | pl.LazyFrame,
    by: str | Sequence[str],
    *,
    target: str = "responder_6",
    prediction: str = "prediction",
    weight: str = "weight",
) -> pl.DataFrame:
    """Aggregate weighted zero-mean R2 components by slice."""

    group_cols = [by] if isinstance(by, str) else list(by)
    if not group_cols:
        raise ValueError("by must contain at least one column")

    lazy = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    return (
        lazy.group_by(group_cols)
        .agg(
            pl.len().alias("rows"),
            pl.col(weight).sum().alias("weight_sum"),
            (pl.col(weight) * (pl.col(target) - pl.col(prediction)).pow(2)).sum().alias("numerator"),
            (pl.col(weight) * pl.col(target).pow(2)).sum().alias("denominator"),
            pl.col(target).mean().alias("target_mean"),
            pl.col(prediction).mean().alias("prediction_mean"),
        )
        .with_columns(
            (1.0 - pl.col("numerator") / pl.col("denominator")).alias("weighted_zero_mean_r2")
        )
        .sort("weighted_zero_mean_r2")
        .collect()
    )


def combine_slice_aggregates(frames: Sequence[pl.DataFrame], by: str | Sequence[str]) -> pl.DataFrame:
    """Combine per-chunk slice aggregates by summing additive components."""

    group_cols = [by] if isinstance(by, str) else list(by)
    if not frames:
        raise ValueError("frames must not be empty")

    return (
        pl.concat(frames)
        .group_by(group_cols)
        .agg(
            pl.col("rows").sum(),
            pl.col("weight_sum").sum(),
            pl.col("numerator").sum(),
            pl.col("denominator").sum(),
        )
        .with_columns(
            (1.0 - pl.col("numerator") / pl.col("denominator")).alias("weighted_zero_mean_r2")
        )
        .sort("weighted_zero_mean_r2")
    )

