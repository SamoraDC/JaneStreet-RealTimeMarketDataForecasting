"""Cross-sectional market-state feature transforms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl


@dataclass(frozen=True)
class RandomProjectionSpec:
    """Fixed random projections for cross-sectional state features."""

    feature_columns: tuple[str, ...]
    weights: np.ndarray
    seed: int

    @property
    def n_projections(self) -> int:
        return int(self.weights.shape[0])

    @property
    def market_columns(self) -> tuple[str, ...]:
        return tuple(f"xs_market_rp_{idx:02d}" for idx in range(self.n_projections))

    @property
    def deviation_columns(self) -> tuple[str, ...]:
        return tuple(f"xs_dev_rp_{idx:02d}" for idx in range(self.n_projections))

    @property
    def output_columns(self) -> tuple[str, ...]:
        return self.market_columns + self.deviation_columns


def make_random_projection_spec(
    feature_columns: Sequence[str],
    *,
    n_projections: int = 8,
    seed: int = 17,
) -> RandomProjectionSpec:
    """Create normalized Gaussian projection weights."""

    features = tuple(feature_columns)
    if not features:
        raise ValueError("feature_columns must not be empty")
    if n_projections <= 0:
        raise ValueError("n_projections must be positive")
    rng = np.random.default_rng(seed)
    weights = rng.normal(loc=0.0, scale=1.0 / np.sqrt(len(features)), size=(n_projections, len(features)))
    return RandomProjectionSpec(feature_columns=features, weights=weights.astype(np.float64), seed=seed)


def with_cross_sectional_random_projections(
    data: pl.LazyFrame,
    spec: RandomProjectionSpec,
    *,
    group_columns: Sequence[str] = ("date_id", "time_id"),
) -> pl.LazyFrame:
    """Add market-state and symbol-deviation random projection features.

    The transform is causal if all rows for the current `group_columns` batch are
    available at prediction time. For the Jane Street local data this corresponds
    to cross-sectional state at the same `date_id,time_id`.
    """

    groups = list(group_columns)
    if not groups:
        raise ValueError("group_columns must not be empty")

    projection_columns = tuple(f"__xs_rp_{idx:02d}" for idx in range(spec.n_projections))
    with_projections = data.with_columns(
        [
            _projection_expr(spec.feature_columns, spec.weights[idx]).alias(projection_columns[idx])
            for idx in range(spec.n_projections)
        ]
    )
    with_market = with_projections.with_columns(
        [
            pl.col(projection_columns[idx]).mean().over(groups).alias(spec.market_columns[idx])
            for idx in range(spec.n_projections)
        ]
    )
    return with_market.with_columns(
        [
            (pl.col(projection_columns[idx]) - pl.col(spec.market_columns[idx])).alias(
                spec.deviation_columns[idx]
            )
            for idx in range(spec.n_projections)
        ]
    )


def _projection_expr(feature_columns: tuple[str, ...], weights: np.ndarray) -> pl.Expr:
    expr = pl.lit(0.0)
    for feature, weight in zip(feature_columns, weights, strict=True):
        expr = expr + pl.col(feature).fill_null(0.0).cast(pl.Float64) * float(weight)
    return expr
