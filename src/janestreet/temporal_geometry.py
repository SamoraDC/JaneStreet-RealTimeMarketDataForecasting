"""Causal intraday temporal-geometry feature transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl


@dataclass(frozen=True)
class TemporalGeometrySpec:
    """Feature columns and rolling windows for intraday path summaries."""

    columns: tuple[str, ...]
    windows: tuple[int, ...] = (5, 20)

    @property
    def output_columns(self) -> tuple[str, ...]:
        names: list[str] = []
        for column in self.columns:
            names.append(f"{column}_tg_diff1")
            for window in self.windows:
                names.extend(
                    [
                        f"{column}_tg_tv_{window}",
                        f"{column}_tg_qv_{window}",
                        f"{column}_tg_rough_{window}",
                    ]
                )
        return tuple(names)


def with_temporal_geometry_features(data: pl.LazyFrame, spec: TemporalGeometrySpec) -> pl.LazyFrame:
    """Add causal intraday variation features per `date_id, symbol_id`.

    Features use current and past values within the same trading date and symbol.
    They do not use future `time_id` rows or target/responders.
    """

    _validate_spec(spec)
    partition = ["date_id", "symbol_id"]
    order_by = "time_id"

    diff_exprs = []
    for column in spec.columns:
        current = pl.col(column).fill_null(0.0)
        previous = current.shift(1).over(partition, order_by=order_by)
        diff_exprs.append(
            (current - previous)
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias(f"{column}_tg_diff1")
        )

    with_diff = data.with_columns(diff_exprs)
    geometry_exprs = []
    for column in spec.columns:
        diff = pl.col(f"{column}_tg_diff1").fill_null(0.0)
        for window in spec.windows:
            tv = diff.abs().rolling_sum(window_size=window, min_samples=1).over(partition, order_by=order_by)
            qv = diff.pow(2).rolling_sum(window_size=window, min_samples=1).over(partition, order_by=order_by)
            geometry_exprs.extend(
                [
                    tv.cast(pl.Float32).alias(f"{column}_tg_tv_{window}"),
                    qv.cast(pl.Float32).alias(f"{column}_tg_qv_{window}"),
                    (tv / (qv.sqrt() + 1e-6)).cast(pl.Float32).alias(f"{column}_tg_rough_{window}"),
                ]
            )

    return with_diff.with_columns(geometry_exprs)


def parse_temporal_geometry_columns(raw: str) -> tuple[str, ...]:
    """Parse a comma-separated feature list."""

    return tuple(part.strip() for part in raw.split(",") if part.strip())


def parse_temporal_geometry_windows(raw: str) -> tuple[int, ...]:
    """Parse comma-separated positive rolling windows."""

    windows = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not windows:
        raise ValueError("temporal geometry windows must not be empty")
    if any(window <= 0 for window in windows):
        raise ValueError("temporal geometry windows must be positive")
    return windows


def require_temporal_geometry_columns(columns: Sequence[str], available: Sequence[str]) -> None:
    """Validate that requested temporal columns exist in model features."""

    available_set = set(available)
    missing = [column for column in columns if column not in available_set]
    if missing:
        raise ValueError(f"unknown temporal geometry columns: {', '.join(missing)}")


def _validate_spec(spec: TemporalGeometrySpec) -> None:
    if not spec.columns:
        raise ValueError("TemporalGeometrySpec.columns must not be empty")
    if not spec.windows:
        raise ValueError("TemporalGeometrySpec.windows must not be empty")
    if any(window <= 0 for window in spec.windows):
        raise ValueError("TemporalGeometrySpec.windows must be positive")
