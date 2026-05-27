"""Causal feature transforms used by the modular models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


def frame_to_matrix(frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    """Collect selected columns as a finite float64 matrix."""

    if not columns:
        raise ValueError("columns must not be empty")
    selected = frame.select([pl.col(name).cast(pl.Float64).alias(name) for name in columns])
    return selected.to_numpy().astype(np.float64, copy=False)


@dataclass(frozen=True)
class ZScoreEncoder:
    """Mean-impute and z-score columns using training statistics only."""

    means: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "ZScoreEncoder":
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("values must be 2D")
        finite = np.isfinite(x)
        counts = finite.sum(axis=0)
        sums = np.where(finite, x, 0.0).sum(axis=0)
        means = np.divide(sums, counts, out=np.zeros(x.shape[1], dtype=np.float64), where=counts > 0)
        filled = np.where(finite, x, means)
        centered = filled - means
        variances = np.divide(
            (centered * centered).sum(axis=0),
            counts,
            out=np.ones(x.shape[1], dtype=np.float64),
            where=counts > 1,
        )
        scales = np.sqrt(np.maximum(variances, 0.0))
        scales[~np.isfinite(scales)] = 1.0
        scales[scales <= 1e-12] = 1.0
        return cls(means=means, scales=scales)

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        filled = np.where(np.isfinite(x), x, self.means)
        return ((filled - self.means) / self.scales).astype(np.float32, copy=False)


@dataclass(frozen=True)
class RankQuantileEncoder:
    """Approximate rank-normalize columns using train-only quantile edges."""

    edges: tuple[np.ndarray, ...]
    fill_values: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, *, n_bins: int = 255) -> "RankQuantileEncoder":
        if n_bins < 4:
            raise ValueError("n_bins must be at least 4")
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("values must be 2D")
        quantiles = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)[1:-1]
        fill_values = np.zeros(x.shape[1], dtype=np.float64)
        all_edges: list[np.ndarray] = []
        for col_idx in range(x.shape[1]):
            col = x[:, col_idx]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                fill_values[col_idx] = 0.0
                all_edges.append(np.empty(0, dtype=np.float64))
                continue
            fill_values[col_idx] = float(np.median(finite))
            edges = np.unique(np.quantile(finite, quantiles))
            all_edges.append(edges.astype(np.float64, copy=False))
        return cls(edges=tuple(all_edges), fill_values=fill_values)

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("values must be 2D")
        if x.shape[1] != len(self.edges):
            raise ValueError("values column count does not match encoder")
        out = np.empty(x.shape, dtype=np.float32)
        for col_idx, edges in enumerate(self.edges):
            col = x[:, col_idx]
            filled = np.where(np.isfinite(col), col, self.fill_values[col_idx])
            if edges.size == 0:
                out[:, col_idx] = 0.0
                continue
            codes = np.searchsorted(edges, filled, side="right").astype(np.float64)
            q = (codes + 0.5) / float(edges.size + 1)
            out[:, col_idx] = (2.0 * q - 1.0).astype(np.float32, copy=False)
        return out


def fit_encoder(kind: str, values: np.ndarray, *, rank_bins: int) -> ZScoreEncoder | RankQuantileEncoder:
    """Fit one supported encoder kind."""

    if kind == "z":
        return ZScoreEncoder.fit(values)
    if kind == "rank":
        return RankQuantileEncoder.fit(values, n_bins=rank_bins)
    raise ValueError(f"unknown encoder kind: {kind}")
