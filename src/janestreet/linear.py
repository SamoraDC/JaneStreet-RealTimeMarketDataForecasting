"""Linear baselines fitted with explicit weighted normal equations."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl

from janestreet.folds import DateFold


@dataclass(frozen=True)
class RidgeModel:
    feature_columns: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    intercept: float
    coefficients: np.ndarray
    alpha: float

    def predict_array(self, features: np.ndarray) -> np.ndarray:
        x = (features.astype(np.float64, copy=False) - self.means) / self.scales
        return self.intercept + x @ self.coefficients


@dataclass(frozen=True)
class RidgeFitData:
    """Sufficient statistics for solving Ridge models on one fold."""

    feature_columns: tuple[str, ...]
    means: np.ndarray
    scales: np.ndarray
    xtwx: np.ndarray
    xtwy: np.ndarray


def fit_weighted_ridge(
    data: pl.LazyFrame,
    fold: DateFold,
    *,
    feature_columns: Sequence[str],
    target: str = "responder_6",
    weight: str = "weight",
    alpha: float = 100.0,
    chunk_days: int = 10,
) -> RidgeModel:
    """Fit weighted Ridge over a fold's training dates.

    Null feature values are filled with 0.0 before standardization. The intercept
    is not regularized.
    """

    fit_data = build_weighted_ridge_fit_data(
        data,
        fold,
        feature_columns=feature_columns,
        target=target,
        weight=weight,
        chunk_days=chunk_days,
    )
    return solve_weighted_ridge(fit_data, alpha=alpha)


def build_weighted_ridge_fit_data(
    data: pl.LazyFrame,
    fold: DateFold,
    *,
    feature_columns: Sequence[str],
    target: str = "responder_6",
    weight: str = "weight",
    chunk_days: int = 10,
) -> RidgeFitData:
    """Accumulate weighted normal-equation statistics for one fold."""

    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")
    features = tuple(feature_columns)
    if not features:
        raise ValueError("feature_columns must not be empty")

    means, scales = _feature_means_scales(data.filter(fold.train_filter()), features)
    p = len(features)
    xtwx = np.zeros((p + 1, p + 1), dtype=np.float64)
    xtwy = np.zeros(p + 1, dtype=np.float64)

    for chunk_start, chunk_end in _date_chunks(fold.train_start, fold.train_end, chunk_days):
        frame = _collect_model_frame(
            data.filter(pl.col("date_id").is_between(chunk_start, chunk_end)),
            features,
            target,
            weight,
        )
        if frame.height == 0:
            continue
        x, y, w = _frame_to_arrays(frame, features, target, weight)
        x = (x - means) / scales
        weighted_x = x * w[:, None]
        xtwx[0, 0] += float(np.sum(w))
        xtwx[0, 1:] += np.sum(weighted_x, axis=0)
        xtwx[1:, 0] = xtwx[0, 1:]
        xtwx[1:, 1:] += x.T @ weighted_x
        weighted_y = y * w
        xtwy[0] += float(np.sum(weighted_y))
        xtwy[1:] += x.T @ weighted_y

    return RidgeFitData(
        feature_columns=features,
        means=means,
        scales=scales,
        xtwx=xtwx,
        xtwy=xtwy,
    )


def solve_weighted_ridge(fit_data: RidgeFitData, *, alpha: float) -> RidgeModel:
    """Solve one Ridge model from precomputed fold statistics."""

    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")

    p = len(fit_data.feature_columns)
    penalty = np.eye(p + 1, dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    system = fit_data.xtwx + penalty
    try:
        params = np.linalg.solve(system, fit_data.xtwy)
    except np.linalg.LinAlgError:
        params = np.linalg.lstsq(system, fit_data.xtwy, rcond=None)[0]

    return RidgeModel(
        feature_columns=fit_data.feature_columns,
        means=fit_data.means,
        scales=fit_data.scales,
        intercept=float(params[0]),
        coefficients=params[1:],
        alpha=alpha,
    )


def evaluate_ridge(
    data: pl.LazyFrame,
    fold: DateFold,
    model: RidgeModel,
    *,
    target: str = "responder_6",
    weight: str = "weight",
    chunk_days: int = 10,
) -> dict[str, float | int | str]:
    """Evaluate a fitted Ridge model over a fold's validation dates."""

    if chunk_days <= 0:
        raise ValueError("chunk_days must be positive")

    numerator = 0.0
    denominator = 0.0
    rows = 0
    weight_sum = 0.0

    for chunk_start, chunk_end in _date_chunks(fold.valid_start, fold.valid_end, chunk_days):
        frame = _collect_model_frame(
            data.filter(pl.col("date_id").is_between(chunk_start, chunk_end)),
            model.feature_columns,
            target,
            weight,
        )
        if frame.height == 0:
            continue
        x, y, w = _frame_to_arrays(frame, model.feature_columns, target, weight)
        pred = model.predict_array(x)
        err = y - pred
        numerator += float(np.sum(w * err * err))
        denominator += float(np.sum(w * y * y))
        rows += frame.height
        weight_sum += float(np.sum(w))

    if denominator <= 0.0:
        raise ValueError(f"{fold.name} has non-positive weighted target energy")

    return {
        "fold": fold.name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "valid_start": fold.valid_start,
        "valid_end": fold.valid_end,
        "train_days": fold.train_days,
        "valid_days": fold.valid_days,
        "rows": rows,
        "weight_sum": weight_sum,
        "numerator": numerator,
        "denominator": denominator,
        "weighted_zero_mean_r2": 1.0 - numerator / denominator,
        "alpha": model.alpha,
        "n_features": len(model.feature_columns),
    }


def feature_columns_from_schema(schema: pl.Schema) -> tuple[str, ...]:
    return tuple(name for name in schema.names() if name.startswith("feature_"))


def _feature_means_scales(data: pl.LazyFrame, features: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    means_frame = data.select([pl.col(name).fill_null(0.0).mean().alias(name) for name in features]).collect()
    stds_frame = data.select([pl.col(name).fill_null(0.0).std(ddof=0).alias(name) for name in features]).collect()
    means = np.asarray(means_frame.row(0), dtype=np.float64)
    scales = np.asarray(stds_frame.row(0), dtype=np.float64)
    scales[~np.isfinite(scales)] = 1.0
    scales[scales <= 1e-12] = 1.0
    return means, scales


def _collect_model_frame(
    data: pl.LazyFrame,
    features: Sequence[str],
    target: str,
    weight: str,
) -> pl.DataFrame:
    return data.select(
        [pl.col(name).fill_null(0.0).cast(pl.Float64).alias(name) for name in features]
        + [pl.col(target).cast(pl.Float64), pl.col(weight).cast(pl.Float64)]
    ).collect()


def _frame_to_arrays(
    frame: pl.DataFrame,
    features: Sequence[str],
    target: str,
    weight: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = frame.select(list(features)).to_numpy()
    y = frame[target].to_numpy()
    w = frame[weight].to_numpy()
    return x, y, w

def _date_chunks(start: int, end: int, chunk_days: int) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    current = start
    while current <= end:
        chunk_end = min(current + chunk_days - 1, end)
        chunks.append((current, chunk_end))
        current = chunk_end + 1
    return chunks
