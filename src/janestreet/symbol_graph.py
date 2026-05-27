"""Fold-local symbol graph features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl


@dataclass(frozen=True)
class SymbolGraphSpec:
    """Static symbol-neighbor graph fitted on a training window."""

    columns: tuple[str, ...]
    neighbors: dict[int, tuple[int, ...]]

    @property
    def output_columns(self) -> tuple[str, ...]:
        names = ["symbol_graph_neighbor_count"]
        for column in self.columns:
            names.append(f"{column}_sg_neighbor_mean")
            names.append(f"{column}_sg_deviation")
        return tuple(names)


def parse_symbol_graph_columns(raw: str) -> tuple[str, ...]:
    """Parse comma-separated graph source columns."""

    return tuple(part.strip() for part in raw.split(",") if part.strip())


def require_symbol_graph_columns(columns: Sequence[str], available: Sequence[str]) -> None:
    """Validate graph source columns against available model features."""

    available_set = set(available)
    missing = [column for column in columns if column not in available_set]
    if missing:
        raise ValueError(f"unknown symbol graph columns: {', '.join(missing)}")


def fit_symbol_graph_spec(
    data: pl.LazyFrame,
    *,
    start: int,
    end: int,
    columns: Sequence[str],
    n_neighbors: int = 5,
    profile_columns: Sequence[str] = (
        "responder_0",
        "responder_1",
        "responder_2",
        "responder_3",
        "responder_4",
        "responder_5",
        "responder_6",
        "responder_7",
        "responder_8",
    ),
) -> SymbolGraphSpec:
    """Fit nearest-symbol graph from historical weighted responder profiles."""

    source_columns = tuple(columns)
    if not source_columns:
        raise ValueError("symbol graph columns must not be empty")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")
    profiles = _symbol_profiles(data, start=start, end=end, profile_columns=tuple(profile_columns))
    if profiles.height <= n_neighbors:
        raise ValueError("n_neighbors must be smaller than the number of symbols")

    symbols = profiles["symbol_id"].to_numpy().astype(np.int16, copy=False)
    matrix = profiles.select(list(profile_columns)).to_numpy().astype(np.float64, copy=False)
    matrix = _standardize_profile_matrix(matrix)
    distances = _pairwise_squared_distances(matrix)
    neighbors: dict[int, tuple[int, ...]] = {}
    for row_idx, symbol in enumerate(symbols):
        order = np.argsort(distances[row_idx])
        selected = [int(symbols[idx]) for idx in order if idx != row_idx][:n_neighbors]
        neighbors[int(symbol)] = tuple(selected)
    return SymbolGraphSpec(columns=source_columns, neighbors=neighbors)


def add_symbol_graph_features(frame: pl.DataFrame, spec: SymbolGraphSpec) -> pl.DataFrame:
    """Add current-timestamp neighbor means and symbol deviations."""

    if frame.height == 0:
        return frame
    if not spec.neighbors:
        raise ValueError("symbol graph must contain neighbors")

    edge_frame = _edge_frame(spec)
    source_frame = frame.select(
        [
            pl.col("date_id"),
            pl.col("time_id"),
            pl.col("symbol_id").alias("_neighbor_symbol_id"),
        ]
        + [
            pl.col(column).fill_null(0.0).cast(pl.Float64).alias(f"__sg_source_{column}")
            for column in spec.columns
        ]
    )
    base = frame.with_row_index("__sg_row").with_columns(pl.col("symbol_id").cast(pl.Int16))
    neighbor_values = (
        base.select(["__sg_row", "date_id", "time_id", "symbol_id"])
        .join(edge_frame, on="symbol_id", how="left")
        .join(source_frame, on=["date_id", "time_id", "_neighbor_symbol_id"], how="left")
    )

    aggregates = neighbor_values.group_by("__sg_row").agg(
        [pl.col("_neighbor_symbol_id").count().alias("symbol_graph_neighbor_count")]
        + [
            pl.col(f"__sg_source_{column}")
            .drop_nulls()
            .mean()
            .fill_null(0.0)
            .cast(pl.Float32)
            .alias(f"{column}_sg_neighbor_mean")
            for column in spec.columns
        ]
    )
    output = base.join(aggregates, on="__sg_row", how="left")
    output = output.with_columns(
        pl.col("symbol_graph_neighbor_count").fill_null(0).cast(pl.Int16)
    )
    output = output.with_columns(
        [
            (
                pl.col(column).fill_null(0.0).cast(pl.Float32)
                - pl.col(f"{column}_sg_neighbor_mean").fill_null(0.0).cast(pl.Float32)
            ).alias(f"{column}_sg_deviation")
            for column in spec.columns
        ]
    )
    return output.drop("__sg_row")


def _symbol_profiles(
    data: pl.LazyFrame,
    *,
    start: int,
    end: int,
    profile_columns: tuple[str, ...],
) -> pl.DataFrame:
    denominator = pl.col("weight").sum()
    return (
        data.filter(pl.col("date_id").is_between(start, end))
        .group_by("symbol_id")
        .agg(
            [
                (
                    (pl.col("weight") * pl.col(column).fill_null(0.0)).sum()
                    / denominator
                )
                .fill_null(0.0)
                .alias(column)
                for column in profile_columns
            ]
        )
        .sort("symbol_id")
        .collect()
    )


def _standardize_profile_matrix(matrix: np.ndarray) -> np.ndarray:
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[~np.isfinite(scales)] = 1.0
    scales[scales <= 1e-12] = 1.0
    return (matrix - means) / scales


def _pairwise_squared_distances(matrix: np.ndarray) -> np.ndarray:
    diff = matrix[:, None, :] - matrix[None, :, :]
    return np.sum(diff * diff, axis=2)


def _edge_frame(spec: SymbolGraphSpec) -> pl.DataFrame:
    rows = [
        {"symbol_id": symbol, "_neighbor_symbol_id": neighbor}
        for symbol, neighbors in spec.neighbors.items()
        for neighbor in neighbors
    ]
    return pl.DataFrame(rows).with_columns(
        [
            pl.col("symbol_id").cast(pl.Int16),
            pl.col("_neighbor_symbol_id").cast(pl.Int16),
        ]
    )
