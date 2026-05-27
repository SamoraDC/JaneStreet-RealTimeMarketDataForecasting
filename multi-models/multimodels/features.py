"""Observable feature construction for the multi-model lab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl


BASE_KEYS = ("date_id", "time_id", "symbol_id")
TARGET = "responder_6"
WEIGHT = "weight"


@dataclass(frozen=True)
class FeatureSpec:
    base_features: tuple[str, ...]
    model_features: tuple[str, ...]
    context_features: tuple[str, ...]
    lag_features: tuple[str, ...]


def scan_parquet_dir(path: Path) -> pl.LazyFrame:
    """Scan a parquet directory or a single parquet file."""

    if path.is_dir():
        return pl.scan_parquet(str(path / "**" / "*.parquet"))
    return pl.scan_parquet(str(path))


def feature_columns_from_schema(schema: pl.Schema, *, max_features: int | None = None) -> tuple[str, ...]:
    """Return ordered feature columns from a Polars schema."""

    features = tuple(name for name in schema.names() if name.startswith("feature_"))
    if max_features is not None and max_features > 0:
        return features[:max_features]
    return features


def lag_columns_from_schema(schema: pl.Schema) -> tuple[str, ...]:
    """Return responder lag columns available in a processed training frame."""

    return tuple(name for name in schema.names() if name.startswith("responder_") and name.endswith("_lag_1"))


def add_context_features(
    lazy: pl.LazyFrame,
    *,
    base_features: tuple[str, ...],
    lag_features: tuple[str, ...] = (),
    cross_sectional_features: tuple[str, ...],
    time_bucket_size: int,
) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    """Add current-row and current-batch context features.

    These features are target-free and observable at inference time from the
    current batch. Cross-sectional ranks are computed within `(date_id,time_id)`.
    """

    if time_bucket_size <= 0:
        raise ValueError("time_bucket_size must be positive")
    selected = tuple(name for name in cross_sectional_features if name in base_features)
    exprs: list[pl.Expr] = [
        (pl.col("time_id") // time_bucket_size).cast(pl.Float64).alias("ctx_time_bucket"),
        pl.col("symbol_id").cast(pl.Float64).alias("ctx_symbol_id"),
        pl.col("weight").cast(pl.Float64).log1p().alias("ctx_log1p_weight"),
        pl.sum_horizontal([pl.col(name).is_null().cast(pl.Float64) for name in base_features]).alias("ctx_missing_count"),
        pl.mean_horizontal([pl.col(name).fill_null(0.0).abs() for name in base_features]).alias("ctx_abs_feature_mean"),
    ]
    context_names = [
        "ctx_time_bucket",
        "ctx_symbol_id",
        "ctx_log1p_weight",
        "ctx_missing_count",
        "ctx_abs_feature_mean",
    ]
    if lag_features:
        exprs.append(pl.mean_horizontal([pl.col(name).fill_null(0.0).abs() for name in lag_features]).alias("ctx_lag_energy"))
    else:
        exprs.append(pl.lit(0.0).alias("ctx_lag_energy"))
    context_names.append("ctx_lag_energy")
    group = ["date_id", "time_id"]
    for name in selected:
        rank_name = f"{name}_ctx_cs_rank"
        z_name = f"{name}_ctx_cs_z"
        filled = pl.col(name).fill_null(0.0)
        count = pl.len().over(group).cast(pl.Float64)
        mean = filled.mean().over(group)
        std = filled.std(ddof=0).over(group)
        exprs.extend(
            [
                ((filled.rank("average").over(group).cast(pl.Float64) - 0.5) / count - 0.5).alias(rank_name),
                ((filled - mean) / pl.when(std > 1e-12).then(std).otherwise(1.0)).alias(z_name),
            ]
        )
        context_names.extend([rank_name, z_name])
    return lazy.with_columns(exprs), tuple(context_names)


def add_raw_preprocessing_features(
    lazy: pl.LazyFrame,
    *,
    raw_feature_columns: tuple[str, ...],
    modes: tuple[str, ...],
) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    """Add causal raw-feature preprocessing for primary alpha models.

    Batch transforms use only rows in the current `(date_id,time_id)` gateway
    batch. Row summaries use only current-row raw feature values.
    """

    columns = tuple(dict.fromkeys(raw_feature_columns))
    selected_modes = tuple(dict.fromkeys(modes))
    if not columns or not selected_modes:
        return lazy, ()
    unsafe = [name for name in columns if name == TARGET or name.startswith("responder_")]
    if unsafe:
        raise ValueError(f"raw_feature_columns cannot include target/responder columns: {unsafe}")
    allowed = {
        "batch_rank",
        "batch_demean",
        "batch_zscore",
        "batch_abs_zscore",
        "batch_top_bottom",
        "row_missing_count",
        "row_abs_mean",
        "row_l2_energy",
    }
    unknown = sorted(set(selected_modes) - allowed)
    if unknown:
        raise ValueError(f"unknown raw_preprocess_modes: {unknown}")

    exprs: list[pl.Expr] = []
    names: list[str] = []
    group = ["date_id", "time_id"]
    filled_columns = [pl.col(name).cast(pl.Float64).fill_null(0.0) for name in columns]

    for column in columns:
        value = pl.col(column).cast(pl.Float64).fill_null(0.0)
        safe = column.replace(".", "_")
        if "batch_rank" in selected_modes or "batch_top_bottom" in selected_modes:
            rank = value.rank(method="average").over(group).cast(pl.Float64)
            count = pl.len().over(group).cast(pl.Float64)
            centered_rank = pl.when(count > 1.0).then(((rank - 1.0) / (count - 1.0)) - 0.5).otherwise(0.0)
            if "batch_rank" in selected_modes:
                name = f"{safe}__raw_batch_rank"
                exprs.append(_finite_expr(centered_rank).alias(name))
                names.append(name)
            if "batch_top_bottom" in selected_modes:
                top = f"{safe}__raw_batch_top10"
                bottom = f"{safe}__raw_batch_bottom10"
                exprs.append(_finite_expr((centered_rank >= 0.4).cast(pl.Float64)).alias(top))
                exprs.append(_finite_expr((centered_rank <= -0.4).cast(pl.Float64)).alias(bottom))
                names.extend([top, bottom])
        if "batch_demean" in selected_modes:
            name = f"{safe}__raw_batch_demean"
            exprs.append(_finite_expr(value - value.mean().over(group)).alias(name))
            names.append(name)
        if "batch_zscore" in selected_modes or "batch_abs_zscore" in selected_modes:
            mean = value.mean().over(group)
            std = value.std(ddof=0).over(group)
            z = (value - mean) / pl.when(std > 1e-12).then(std).otherwise(1.0)
            if "batch_zscore" in selected_modes:
                name = f"{safe}__raw_batch_zscore"
                exprs.append(_finite_expr(z).alias(name))
                names.append(name)
            if "batch_abs_zscore" in selected_modes:
                name = f"{safe}__raw_batch_abs_zscore"
                exprs.append(_finite_expr(z.abs()).alias(name))
                names.append(name)

    if "row_missing_count" in selected_modes:
        name = "raw_row_missing_count"
        exprs.append(pl.sum_horizontal([pl.col(column).is_null().cast(pl.Float64) for column in columns]).alias(name))
        names.append(name)
    if "row_abs_mean" in selected_modes:
        name = "raw_row_abs_mean"
        exprs.append(_finite_expr(pl.mean_horizontal([value.abs() for value in filled_columns])).alias(name))
        names.append(name)
    if "row_l2_energy" in selected_modes:
        name = "raw_row_l2_energy"
        exprs.append(_finite_expr(pl.sum_horizontal([value * value for value in filled_columns])).alias(name))
        names.append(name)

    return lazy.with_columns(exprs), tuple(names)


def _finite_expr(expr: pl.Expr) -> pl.Expr:
    return pl.when(expr.is_finite()).then(expr).otherwise(0.0)
