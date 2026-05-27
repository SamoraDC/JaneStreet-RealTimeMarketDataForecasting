"""Causal Koopman/EDMD-lite observable features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl


@dataclass(frozen=True)
class KoopmanSpec:
    """Source columns and memory span for nonlinear observables."""

    columns: tuple[str, ...]
    span: int = 16

    @property
    def output_columns(self) -> tuple[str, ...]:
        names: list[str] = []
        for column in self.columns:
            names.extend(
                [
                    f"{column}_kp_square",
                    f"{column}_kp_delta",
                    f"{column}_kp_value_delta",
                    f"{column}_kp_value_ewm_{self.span}",
                ]
            )
        return tuple(names)


def parse_koopman_columns(raw: str) -> tuple[str, ...]:
    """Parse comma-separated Koopman source columns."""

    return tuple(part.strip() for part in raw.split(",") if part.strip())


def require_koopman_columns(columns: Sequence[str], available: Sequence[str]) -> None:
    """Validate Koopman source columns."""

    available_set = set(available)
    missing = [column for column in columns if column not in available_set]
    if missing:
        raise ValueError(f"unknown koopman columns: {', '.join(missing)}")


def with_koopman_features(data: pl.LazyFrame, spec: KoopmanSpec) -> pl.LazyFrame:
    """Add causal nonlinear observables per `date_id, symbol_id`."""

    _validate_spec(spec)
    partition = ["date_id", "symbol_id"]
    order_by = "time_id"
    exprs: list[pl.Expr] = []
    for column in spec.columns:
        value = pl.col(column).fill_null(0.0).cast(pl.Float64)
        delta = (value - value.shift(1).over(partition, order_by=order_by)).fill_null(0.0)
        ewm = value.ewm_mean(span=spec.span, adjust=False, min_samples=1).over(
            partition,
            order_by=order_by,
        )
        exprs.extend(
            [
                value.pow(2).cast(pl.Float32).alias(f"{column}_kp_square"),
                delta.cast(pl.Float32).alias(f"{column}_kp_delta"),
                (value * delta).cast(pl.Float32).alias(f"{column}_kp_value_delta"),
                (value * ewm).cast(pl.Float32).alias(f"{column}_kp_value_ewm_{spec.span}"),
            ]
        )
    return data.with_columns(exprs)


def _validate_spec(spec: KoopmanSpec) -> None:
    if not spec.columns:
        raise ValueError("KoopmanSpec.columns must not be empty")
    if spec.span <= 0:
        raise ValueError("KoopmanSpec.span must be positive")
