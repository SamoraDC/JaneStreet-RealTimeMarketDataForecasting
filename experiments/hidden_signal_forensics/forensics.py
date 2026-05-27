"""Utilities for hidden-signal forensics on noisy tabular time series."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MarchenkoPasturBounds:
    """Theoretical eigenvalue support for a standardized noise covariance."""

    n_samples: int
    n_features: int
    aspect_ratio: float
    lambda_minus: float
    lambda_plus: float


def marchenko_pastur_bounds(n_samples: int, n_features: int, sigma2: float = 1.0) -> MarchenkoPasturBounds:
    """Return Marchenko-Pastur bounds for p/n aspect ratio."""

    if n_samples <= 1:
        raise ValueError("n_samples must be greater than 1")
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if not isfinite(sigma2) or sigma2 <= 0.0:
        raise ValueError("sigma2 must be finite and positive")
    q = n_features / n_samples
    root = sqrt(q)
    lower = sigma2 * max(0.0, (1.0 - root) ** 2)
    upper = sigma2 * (1.0 + root) ** 2
    return MarchenkoPasturBounds(
        n_samples=n_samples,
        n_features=n_features,
        aspect_ratio=q,
        lambda_minus=lower,
        lambda_plus=upper,
    )


def standardize_matrix(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-impute and z-score columns without changing column count."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("values must be two-dimensional")
    finite = np.isfinite(x)
    counts = finite.sum(axis=0)
    sums = np.where(finite, x, 0.0).sum(axis=0)
    means = np.divide(sums, counts, out=np.zeros(x.shape[1], dtype=np.float64), where=counts > 0)
    filled = np.where(finite, x, means)
    centered = filled - means
    variances = np.divide((centered * centered).sum(axis=0), counts, out=np.ones(x.shape[1], dtype=np.float64), where=counts > 1)
    stds = np.sqrt(np.maximum(variances, 0.0))
    safe_stds = np.where(stds > 0.0, stds, 1.0)
    return centered / safe_stds, means, safe_stds


def correlation_eigenvalues(values: np.ndarray) -> np.ndarray:
    """Return sorted eigenvalues of the standardized feature covariance."""

    z, _, _ = standardize_matrix(values)
    if z.shape[0] <= 1:
        raise ValueError("at least two rows are required")
    covariance = (z.T @ z) / float(z.shape[0] - 1)
    return np.linalg.eigvalsh(covariance)


def weighted_centered_corr(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> float:
    """Return weighted centered correlation; zero if either side is constant."""

    x, y, weight = _valid_triplet(x, y, weight)
    if x.size == 0:
        return 0.0
    w_sum = float(weight.sum())
    if w_sum <= 0.0:
        return 0.0
    mx = float(np.sum(weight * x) / w_sum)
    my = float(np.sum(weight * y) / w_sum)
    xc = x - mx
    yc = y - my
    vx = float(np.sum(weight * xc * xc))
    vy = float(np.sum(weight * yc * yc))
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    cov = float(np.sum(weight * xc * yc))
    return cov / sqrt(vx * vy)


def average_rank(values: np.ndarray) -> np.ndarray:
    """Return average ranks for one-dimensional values, preserving ties."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("values must be one-dimensional")
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(x.size, dtype=np.float64)
    start = 0
    while start < x.size:
        end = start + 1
        while end < x.size and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def binned_mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Estimate mutual information using quantile bins for x and y."""

    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < n_bins or np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
        return 0.0
    xb = quantile_codes(x, n_bins)
    yb = quantile_codes(y, n_bins)
    if xb.max(initial=0) == 0 or yb.max(initial=0) == 0:
        return 0.0
    table = np.zeros((int(xb.max()) + 1, int(yb.max()) + 1), dtype=np.float64)
    np.add.at(table, (xb, yb), 1.0)
    total = table.sum()
    if total <= 0.0:
        return 0.0
    pxy = table / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    positive = pxy > 0.0
    return float(np.sum(pxy[positive] * np.log(pxy[positive] / (px @ py)[positive])))


def quantile_codes(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Map values to quantile bin codes with duplicate-safe edges."""

    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("values must be one-dimensional")
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(x, quantiles))
    if edges.size == 0:
        return np.zeros(x.size, dtype=np.int16)
    return np.searchsorted(edges, x, side="right").astype(np.int16)


def tail_mean_spread(
    x: np.ndarray,
    y: np.ndarray,
    weight: np.ndarray,
    *,
    lower_q: float = 0.05,
    upper_q: float = 0.95,
) -> tuple[float, float, float]:
    """Return weighted target means in lower/upper x tails and upper-lower spread."""

    if not (0.0 < lower_q < upper_q < 1.0):
        raise ValueError("tail quantiles must satisfy 0 < lower_q < upper_q < 1")
    x, y, weight = _valid_triplet(x, y, weight)
    if x.size == 0:
        return 0.0, 0.0, 0.0
    lower = float(np.quantile(x, lower_q))
    upper = float(np.quantile(x, upper_q))
    low_mean = weighted_mean(y[x <= lower], weight[x <= lower])
    high_mean = weighted_mean(y[x >= upper], weight[x >= upper])
    return low_mean, high_mean, high_mean - low_mean


def weighted_mean(values: np.ndarray, weight: np.ndarray) -> float:
    """Return weighted mean, or zero when no positive weight exists."""

    values = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    mask = np.isfinite(values) & np.isfinite(weight) & (weight >= 0.0)
    values = values[mask]
    weight = weight[mask]
    w_sum = float(weight.sum())
    if values.size == 0 or w_sum <= 0.0:
        return 0.0
    return float(np.sum(weight * values) / w_sum)


def optimal_univariate_fit(train_phi: np.ndarray, train_y: np.ndarray, train_weight: np.ndarray) -> float:
    """Fit y_hat = alpha * phi under weighted squared loss."""

    phi, y, weight = _valid_triplet(train_phi, train_y, train_weight)
    denom = float(np.sum(weight * phi * phi))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(weight * phi * y) / denom)


def weighted_zero_mean_r2_arrays(y: np.ndarray, pred: np.ndarray, weight: np.ndarray) -> float:
    """Array implementation of the competition's weighted zero-mean R2."""

    pred, y, weight = _valid_triplet(pred, y, weight)
    denominator = float(np.sum(weight * y * y))
    if denominator <= 0.0:
        return 0.0
    numerator = float(np.sum(weight * (y - pred) * (y - pred)))
    return 1.0 - numerator / denominator


def autocorrelation(values: np.ndarray, lags: Iterable[int]) -> dict[int, float]:
    """Return centered autocorrelation for requested positive lags."""

    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    result: dict[int, float] = {}
    if x.size < 2:
        return {int(lag): 0.0 for lag in lags}
    x = x - float(np.mean(x))
    variance = float(np.dot(x, x))
    for lag in lags:
        lag = int(lag)
        if lag <= 0:
            raise ValueError("lags must be positive")
        if lag >= x.size or variance <= 0.0:
            result[lag] = 0.0
        else:
            result[lag] = float(np.dot(x[:-lag], x[lag:]) / variance)
    return result


def top_periodogram_peaks(values: np.ndarray, *, top_k: int = 10) -> list[dict[str, float]]:
    """Return top nonzero periodogram peaks from an ordered one-dimensional series."""

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return []
    x = x - float(np.mean(x))
    spectrum = np.fft.rfft(x)
    power = (spectrum.real * spectrum.real + spectrum.imag * spectrum.imag) / x.size
    if power.size <= 1:
        return []
    power[0] = 0.0
    candidate_count = min(top_k, power.size - 1)
    indices = np.argpartition(power, -candidate_count)[-candidate_count:]
    indices = indices[np.argsort(power[indices])[::-1]]
    total_power = float(power.sum())
    rows = []
    for idx in indices:
        frequency = float(idx / x.size)
        rows.append(
            {
                "frequency_index": int(idx),
                "frequency": frequency,
                "period_rows": float(x.size / idx) if idx > 0 else 0.0,
                "power": float(power[idx]),
                "power_share": float(power[idx] / total_power) if total_power > 0.0 else 0.0,
            }
        )
    return rows


def hill_tail_index_abs(values: np.ndarray, k: int) -> float:
    """Return Hill tail index estimate for absolute values."""

    if k <= 0:
        raise ValueError("k must be positive")
    x = np.abs(np.asarray(values, dtype=np.float64))
    x = x[np.isfinite(x) & (x > 0.0)]
    if x.size <= k:
        return 0.0
    sorted_x = np.sort(x)
    top = sorted_x[-k:]
    threshold = sorted_x[-k - 1]
    if threshold <= 0.0:
        return 0.0
    denominator = float(np.sum(np.log(top / threshold)))
    if denominator <= 0.0:
        return 0.0
    return float(k / denominator)


def final_digit_counts(values: np.ndarray, decimals: int = 6) -> np.ndarray:
    """Count final decimal digits after fixed decimal rounding."""

    if decimals <= 0:
        raise ValueError("decimals must be positive")
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros(10, dtype=np.int64)
    scale = 10**decimals
    rounded = np.rint(np.abs(x) * scale).astype(np.int64)
    digits = rounded % 10
    return np.bincount(digits, minlength=10)


def _valid_triplet(x: np.ndarray, y: np.ndarray, weight: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)
    if x.shape != y.shape or x.shape != weight.shape:
        raise ValueError("x, y, and weight must have the same shape")
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(weight) & (weight >= 0.0)
    return x[mask], y[mask], weight[mask]
