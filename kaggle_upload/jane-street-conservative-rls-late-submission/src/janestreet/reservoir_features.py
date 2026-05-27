"""Randomized causal reservoir-style features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl


@dataclass(frozen=True)
class ReservoirSpec:
    """Random projection reservoir specification."""

    columns: tuple[str, ...]
    weights: np.ndarray
    spans: tuple[int, ...]

    @property
    def n_states(self) -> int:
        return int(self.weights.shape[0])

    @property
    def output_columns(self) -> tuple[str, ...]:
        names: list[str] = []
        for state_idx in range(self.n_states):
            for span in self.spans:
                names.append(f"reservoir_s{state_idx:02d}_ewm_{span}")
        return tuple(names)


def make_reservoir_spec(
    columns: Sequence[str],
    *,
    n_states: int = 8,
    spans: Sequence[int] = (5, 20),
    seed: int = 17,
) -> ReservoirSpec:
    """Create deterministic random projection weights."""

    source_columns = tuple(columns)
    if not source_columns:
        raise ValueError("reservoir columns must not be empty")
    if n_states <= 0:
        raise ValueError("n_states must be positive")
    span_values = tuple(int(span) for span in spans)
    if not span_values or any(span <= 0 for span in span_values):
        raise ValueError("reservoir spans must be positive")
    rng = np.random.default_rng(seed)
    weights = rng.normal(0.0, 1.0, size=(n_states, len(source_columns))).astype(np.float64)
    weights /= np.sqrt(len(source_columns))
    return ReservoirSpec(columns=source_columns, weights=weights, spans=span_values)


def parse_reservoir_columns(raw: str) -> tuple[str, ...]:
    """Parse comma-separated reservoir source columns."""

    return tuple(part.strip() for part in raw.split(",") if part.strip())


def parse_reservoir_spans(raw: str) -> tuple[int, ...]:
    """Parse comma-separated positive EWM spans."""

    spans = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not spans:
        raise ValueError("reservoir spans must not be empty")
    if any(span <= 0 for span in spans):
        raise ValueError("reservoir spans must be positive")
    return spans


def require_reservoir_columns(columns: Sequence[str], available: Sequence[str]) -> None:
    """Validate reservoir source columns."""

    available_set = set(available)
    missing = [column for column in columns if column not in available_set]
    if missing:
        raise ValueError(f"unknown reservoir columns: {', '.join(missing)}")


def with_reservoir_features(data: pl.LazyFrame, spec: ReservoirSpec) -> pl.LazyFrame:
    """Add nonlinear randomized EWM state features per `date_id, symbol_id`."""

    partition = ["date_id", "symbol_id"]
    order_by = "time_id"
    exprs = []
    for state_idx in range(spec.n_states):
        weighted_terms = [
            float(weight) * pl.col(column).fill_null(0.0).cast(pl.Float64)
            for weight, column in zip(spec.weights[state_idx], spec.columns)
        ]
        projection = weighted_terms[0]
        for term in weighted_terms[1:]:
            projection = projection + term
        projection = projection.tanh()
        for span in spec.spans:
            exprs.append(
                projection.ewm_mean(span=span, adjust=False, min_samples=1)
                .over(partition, order_by=order_by)
                .cast(pl.Float32)
                .alias(f"reservoir_s{state_idx:02d}_ewm_{span}")
            )
    return data.with_columns(exprs)
