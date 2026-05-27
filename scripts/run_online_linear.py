"""Evaluate online linear learning baselines on rolling folds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

from janestreet.folds import DateFold, make_rolling_folds
from janestreet.linear import feature_columns_from_schema
from janestreet.paths import TRAIN_PARQUET_DIR


@dataclass
class NormalEquationOnlineRidge:
    """Chunk-updated ridge normal equations with optional forgetting."""

    n_features: int
    alpha: float
    decay: float

    def __post_init__(self) -> None:
        self.xtwx = np.zeros((self.n_features + 1, self.n_features + 1), dtype=np.float64)
        self.xtwy = np.zeros(self.n_features + 1, dtype=np.float64)
        self.coef_ = np.zeros(self.n_features + 1, dtype=np.float64)

    def partial_fit(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> None:
        self.xtwx *= self.decay
        self.xtwy *= self.decay
        design = np.empty((x.shape[0], x.shape[1] + 1), dtype=np.float64)
        design[:, 0] = 1.0
        design[:, 1:] = x
        self.xtwx += design.T @ (design * sample_weight[:, None])
        self.xtwy += design.T @ (y * sample_weight)
        penalty = np.eye(self.n_features + 1, dtype=np.float64) * self.alpha
        penalty[0, 0] = 0.0
        system = self.xtwx + penalty
        try:
            self.coef_ = np.linalg.solve(system, self.xtwy)
        except np.linalg.LinAlgError:
            self.coef_ = np.linalg.lstsq(system, self.xtwy, rcond=None)[0]

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.coef_[0] + x @ self.coef_[1:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--valid-window", type=int, default=60)
    parser.add_argument("--chunk-days", type=int, default=5)
    parser.add_argument("--methods", default="sgd,sgd_huber_avg,passive_aggressive,online_ridge_decay")
    parser.add_argument("--id-columns", default="")
    parser.add_argument("--sgd-alpha", type=float, default=1e-5)
    parser.add_argument("--sgd-eta0", type=float, default=0.01)
    parser.add_argument("--pa-c", type=float, default=0.01)
    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--ridge-decay", type=float, default=0.98)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/online_linear"))
    args = parser.parse_args()
    methods = _parse_methods(args.methods)
    id_columns = _parse_id_columns(args.id_columns)
    if args.chunk_days <= 0:
        raise ValueError("--chunk-days must be positive")

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    base_features = feature_columns_from_schema(train.collect_schema())
    feature_columns = tuple(dict.fromkeys([*base_features, *id_columns]))
    folds = _make_folds(train, args)
    rows: list[dict[str, float | int | str]] = []

    for fold in folds:
        scaler = _fit_scaler(train, fold, feature_columns, args.chunk_days)
        for method in methods:
            model = _make_model(method, n_features=len(feature_columns), args=args)
            for chunk_start, chunk_end in _date_chunks(fold.train_start, fold.train_end, args.chunk_days):
                frame = _collect_frame(train, chunk_start, chunk_end, feature_columns)
                if frame.height == 0:
                    continue
                x, y, weight = _frame_arrays(frame, feature_columns)
                x_scaled = scaler.transform(x)
                model.partial_fit(x_scaled, y, sample_weight=weight)
            rows.append(
                {
                    **_fold_metadata(fold),
                    "method": method,
                    "n_features": len(feature_columns),
                    **_evaluate_model(train, fold, feature_columns, scaler, model, args.chunk_days),
                }
            )

    results = pl.DataFrame(rows)
    summary = _summary_by_method(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(args.output_dir / "online_linear_by_fold.csv")
    summary.write_csv(args.output_dir / "online_linear_summary.csv")
    print(results)
    print(summary)
    print(f"Wrote {args.output_dir}")


def _make_model(method: str, *, n_features: int, args: argparse.Namespace):
    if method == "sgd":
        return SGDRegressor(
            loss="squared_error",
            penalty="l2",
            alpha=args.sgd_alpha,
            learning_rate="constant",
            eta0=args.sgd_eta0,
            fit_intercept=True,
            random_state=17,
        )
    if method == "sgd_huber_avg":
        return SGDRegressor(
            loss="huber",
            epsilon=1.0,
            penalty="l2",
            alpha=args.sgd_alpha,
            learning_rate="adaptive",
            eta0=args.sgd_eta0,
            fit_intercept=True,
            average=True,
            random_state=17,
        )
    if method == "passive_aggressive":
        return SGDRegressor(
            loss="epsilon_insensitive",
            penalty=None,
            learning_rate="pa1",
            eta0=args.pa_c,
            epsilon=0.01,
            fit_intercept=True,
            random_state=17,
        )
    if method == "online_ridge_decay":
        return NormalEquationOnlineRidge(
            n_features=n_features,
            alpha=args.ridge_alpha,
            decay=args.ridge_decay,
        )
    raise ValueError(f"unknown method: {method}")


def _fit_scaler(
    train: pl.LazyFrame,
    fold: DateFold,
    feature_columns: tuple[str, ...],
    chunk_days: int,
) -> StandardScaler:
    scaler = StandardScaler()
    for chunk_start, chunk_end in _date_chunks(fold.train_start, fold.train_end, chunk_days):
        frame = _collect_frame(train, chunk_start, chunk_end, feature_columns)
        if frame.height == 0:
            continue
        x, _, weight = _frame_arrays(frame, feature_columns)
        scaler.partial_fit(x, sample_weight=weight)
    return scaler


def _evaluate_model(
    train: pl.LazyFrame,
    fold: DateFold,
    feature_columns: tuple[str, ...],
    scaler: StandardScaler,
    model: object,
    chunk_days: int,
) -> dict[str, float | int]:
    numerator = 0.0
    denominator = 0.0
    rows = 0
    weight_sum = 0.0
    prediction_sum = 0.0
    prediction_sq_sum = 0.0
    for chunk_start, chunk_end in _date_chunks(fold.valid_start, fold.valid_end, chunk_days):
        frame = _collect_frame(train, chunk_start, chunk_end, feature_columns)
        if frame.height == 0:
            continue
        x, y, weight = _frame_arrays(frame, feature_columns)
        pred = model.predict(scaler.transform(x))
        err = y - pred
        numerator += float(np.sum(weight * err * err))
        denominator += float(np.sum(weight * y * y))
        rows += frame.height
        weight_sum += float(np.sum(weight))
        prediction_sum += float(np.sum(pred))
        prediction_sq_sum += float(np.sum(pred * pred))
    if denominator <= 0.0:
        raise ValueError(f"{fold.name} has non-positive weighted target energy")
    prediction_mean = prediction_sum / rows
    prediction_var = max(prediction_sq_sum / rows - prediction_mean * prediction_mean, 0.0)
    return {
        "rows": rows,
        "weight_sum": weight_sum,
        "numerator": numerator,
        "denominator": denominator,
        "weighted_zero_mean_r2": 1.0 - numerator / denominator,
        "prediction_mean": prediction_mean,
        "prediction_std": float(np.sqrt(prediction_var)),
    }


def _collect_frame(
    train: pl.LazyFrame,
    start: int,
    end: int,
    feature_columns: Sequence[str],
) -> pl.DataFrame:
    return (
        train.filter(pl.col("date_id").is_between(start, end))
        .select(
            [pl.col(name).fill_null(0.0).cast(pl.Float32).alias(name) for name in feature_columns]
            + [pl.col("responder_6").cast(pl.Float32), pl.col("weight").cast(pl.Float32)]
        )
        .collect()
    )


def _frame_arrays(
    frame: pl.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        frame.select(list(feature_columns)).to_numpy(),
        frame["responder_6"].to_numpy(),
        frame["weight"].to_numpy(),
    )


def _make_folds(train: pl.LazyFrame, args: argparse.Namespace) -> list[DateFold]:
    bounds = train.select(
        pl.min("date_id").alias("min_date_id"),
        pl.max("date_id").alias("max_date_id"),
    ).collect()
    return make_rolling_folds(
        min_date_id=int(bounds["min_date_id"][0]),
        max_date_id=int(bounds["max_date_id"][0]),
        n_folds=args.n_folds,
        train_window=args.train_window,
        valid_window=args.valid_window,
    )


def _summary_by_method(results: pl.DataFrame) -> pl.DataFrame:
    return (
        results.group_by("method")
        .agg(
            pl.len().alias("folds"),
            pl.col("weighted_zero_mean_r2").mean().alias("mean_r2"),
            pl.col("weighted_zero_mean_r2").std().alias("std_r2"),
            pl.col("weighted_zero_mean_r2").min().alias("min_r2"),
            pl.col("weighted_zero_mean_r2").max().alias("max_r2"),
            (1.0 - pl.col("numerator").sum() / pl.col("denominator").sum()).alias("global_r2"),
            pl.col("rows").sum().alias("validation_rows"),
        )
        .sort(["global_r2", "min_r2"], descending=[True, True])
    )


def _fold_metadata(fold: DateFold) -> dict[str, int | str]:
    return {
        "fold": fold.name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "valid_start": fold.valid_start,
        "valid_end": fold.valid_end,
    }


def _date_chunks(start: int, end: int, chunk_days: int) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    current = start
    while current <= end:
        chunk_end = min(current + chunk_days - 1, end)
        chunks.append((current, chunk_end))
        current = chunk_end + 1
    return chunks


def _parse_methods(raw: str) -> tuple[str, ...]:
    methods = tuple(part.strip() for part in raw.split(",") if part.strip())
    allowed = {"sgd", "sgd_huber_avg", "passive_aggressive", "online_ridge_decay"}
    unknown = set(methods) - allowed
    if unknown:
        raise ValueError(f"unknown methods: {', '.join(sorted(unknown))}")
    if not methods:
        raise ValueError("--methods must contain at least one method")
    return methods


def _parse_id_columns(raw: str) -> tuple[str, ...]:
    columns = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    allowed = {"time_id", "symbol_id"}
    unknown = set(columns) - allowed
    if unknown:
        raise ValueError(f"unknown id columns: {', '.join(sorted(unknown))}")
    return columns


if __name__ == "__main__":
    main()
