"""Operational-time diagnostics and causal feature transforms."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Sequence

import polars as pl


@dataclass(frozen=True)
class OperationalTimeSpec:
    """Causal operational-time features for the original row grid."""

    source_columns: tuple[str, ...]
    windows: tuple[int, ...] = (16, 64)
    max_time_id: int = 967

    @property
    def output_columns(self) -> tuple[str, ...]:
        names = [
            "ot_time_frac",
            "ot_time_sin",
            "ot_time_cos",
            "ot_symbol_tick_index",
            "ot_symbol_weight_cum",
            "ot_missing_count",
            "ot_missing_frac",
            "ot_source_activity",
        ]
        for window in self.windows:
            names.extend(
                [
                    f"ot_symbol_weight_ewm_{window}",
                    f"ot_source_activity_ewm_{window}",
                    f"ot_missing_frac_ewm_{window}",
                ]
            )
        return tuple(names)


def with_operational_time_features(data: pl.LazyFrame, spec: OperationalTimeSpec) -> pl.LazyFrame:
    """Add causal row-level clocks without changing the competition rows.

    The features use only current and past rows within each `date_id,symbol_id`
    trajectory. They are safe as model inputs when `weight` is available at
    inference, which is true for the local competition API schema.
    """

    _validate_spec(spec)
    partition = ["date_id", "symbol_id"]
    order_by = "time_id"
    source_count = float(len(spec.source_columns))
    current_missing = sum(pl.col(name).is_null().cast(pl.Float32) for name in spec.source_columns)
    current_activity = sum(pl.col(name).fill_null(0.0).abs().cast(pl.Float32) for name in spec.source_columns) / source_count
    time_frac = pl.col("time_id").cast(pl.Float32) / float(spec.max_time_id)

    base = data.with_columns(
        [
            time_frac.alias("ot_time_frac"),
            (time_frac * (2.0 * pi)).sin().cast(pl.Float32).alias("ot_time_sin"),
            (time_frac * (2.0 * pi)).cos().cast(pl.Float32).alias("ot_time_cos"),
            pl.col("time_id").cum_count().over(partition, order_by=order_by).cast(pl.Float32).alias("ot_symbol_tick_index"),
            pl.col("weight").fill_null(0.0).cum_sum().over(partition, order_by=order_by).cast(pl.Float32).alias("ot_symbol_weight_cum"),
            current_missing.cast(pl.Float32).alias("ot_missing_count"),
            (current_missing / source_count).cast(pl.Float32).alias("ot_missing_frac"),
            current_activity.cast(pl.Float32).alias("ot_source_activity"),
        ]
    )
    ewm_exprs: list[pl.Expr] = []
    for window in spec.windows:
        ewm_exprs.extend(
            [
                pl.col("weight")
                .fill_null(0.0)
                .ewm_mean(span=window, adjust=False, min_samples=1)
                .over(partition, order_by=order_by)
                .cast(pl.Float32)
                .alias(f"ot_symbol_weight_ewm_{window}"),
                pl.col("ot_source_activity")
                .ewm_mean(span=window, adjust=False, min_samples=1)
                .over(partition, order_by=order_by)
                .cast(pl.Float32)
                .alias(f"ot_source_activity_ewm_{window}"),
                pl.col("ot_missing_frac")
                .ewm_mean(span=window, adjust=False, min_samples=1)
                .over(partition, order_by=order_by)
                .cast(pl.Float32)
                .alias(f"ot_missing_frac_ewm_{window}"),
            ]
        )
    return base.with_columns(ewm_exprs)


def parse_operational_time_columns(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def parse_operational_time_windows(raw: str) -> tuple[int, ...]:
    windows = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not windows:
        raise ValueError("operational-time windows must not be empty")
    if any(window <= 0 for window in windows):
        raise ValueError("operational-time windows must be positive")
    return windows


def require_operational_time_columns(columns: Sequence[str], available: Sequence[str]) -> None:
    available_set = set(available)
    missing = [column for column in columns if column not in available_set]
    if missing:
        raise ValueError(f"unknown operational-time columns: {', '.join(missing)}")


def _validate_spec(spec: OperationalTimeSpec) -> None:
    if not spec.source_columns:
        raise ValueError("OperationalTimeSpec.source_columns must not be empty")
    if not spec.windows:
        raise ValueError("OperationalTimeSpec.windows must not be empty")
    if any(window <= 0 for window in spec.windows):
        raise ValueError("OperationalTimeSpec.windows must be positive")
    if spec.max_time_id <= 0:
        raise ValueError("OperationalTimeSpec.max_time_id must be positive")
