"""Run a small sklearn HistGradientBoosting baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor

from janestreet.folds import make_rolling_folds
from janestreet.linear import feature_columns_from_schema
from janestreet.paths import TRAIN_PARQUET_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--valid-window", type=int, default=60)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--train-sample-frac", type=float, default=0.10)
    parser.add_argument("--chunk-days", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/sklearn_gbdt"))
    args = parser.parse_args()

    if not 0.0 < args.train_sample_frac <= 1.0:
        raise ValueError("--train-sample-frac must be in (0, 1]")

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    features = feature_columns_from_schema(train.collect_schema())
    bounds = train.select(
        pl.min("date_id").alias("min_date_id"),
        pl.max("date_id").alias("max_date_id"),
    ).collect()
    folds = make_rolling_folds(
        min_date_id=int(bounds["min_date_id"][0]),
        max_date_id=int(bounds["max_date_id"][0]),
        n_folds=args.n_folds,
        train_window=args.train_window,
        valid_window=args.valid_window,
        gap=args.gap,
    )

    rows: list[dict[str, float | int | str]] = []
    for fold in folds:
        x_train, y_train, w_train = _collect_train_sample(
            train,
            features,
            fold.train_start,
            fold.train_end,
            sample_frac=args.train_sample_frac,
            seed=args.random_state,
        )
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2_regularization,
            early_stopping=False,
            random_state=args.random_state,
        )
        model.fit(x_train, y_train, sample_weight=w_train)
        rows.append(
            {
                "fold": fold.name,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "valid_start": fold.valid_start,
                "valid_end": fold.valid_end,
                "train_sample_rows": int(x_train.shape[0]),
                **_evaluate_model(
                    train,
                    features,
                    model,
                    fold.valid_start,
                    fold.valid_end,
                    chunk_days=args.chunk_days,
                ),
            }
        )

    results = pl.DataFrame(rows)
    summary = _summary(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(args.output_dir / "sklearn_gbdt_by_fold.csv")
    summary.write_csv(args.output_dir / "sklearn_gbdt_summary.csv")
    report = {
        "experiment": "sklearn_hist_gradient_boosting",
        "n_folds": args.n_folds,
        "train_window": args.train_window,
        "valid_window": args.valid_window,
        "train_sample_frac": args.train_sample_frac,
        "max_iter": args.max_iter,
        "learning_rate": args.learning_rate,
        "max_leaf_nodes": args.max_leaf_nodes,
        "l2_regularization": args.l2_regularization,
        "random_state": args.random_state,
        "summary": summary.row(0, named=True),
    }
    (args.output_dir / "sklearn_gbdt_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(results)
    print(summary)
    print(f"Wrote {args.output_dir}")


def _collect_train_sample(
    data: pl.LazyFrame,
    features: tuple[str, ...],
    start: int,
    end: int,
    *,
    sample_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    threshold = int(sample_frac * 1_000_000)
    sample_filter = (pl.struct(["date_id", "time_id", "symbol_id"]).hash(seed=seed) % 1_000_000) < threshold
    frame = (
        data.filter(pl.col("date_id").is_between(start, end) & sample_filter)
        .select(
            [pl.col(name).cast(pl.Float32).alias(name) for name in features]
            + [pl.col("responder_6").cast(pl.Float32), pl.col("weight").cast(pl.Float32)]
        )
        .collect()
    )
    if frame.height == 0:
        raise ValueError(f"empty train sample for dates {start}-{end}")
    return (
        frame.select(list(features)).to_numpy(),
        frame["responder_6"].to_numpy(),
        frame["weight"].to_numpy(),
    )


def _evaluate_model(
    data: pl.LazyFrame,
    features: tuple[str, ...],
    model: HistGradientBoostingRegressor,
    start: int,
    end: int,
    *,
    chunk_days: int,
) -> dict[str, float | int]:
    numerator = 0.0
    denominator = 0.0
    rows = 0
    weight_sum = 0.0
    prediction_sum = 0.0
    prediction_sumsq = 0.0
    for chunk_start, chunk_end in _date_chunks(start, end, chunk_days):
        frame = (
            data.filter(pl.col("date_id").is_between(chunk_start, chunk_end))
            .select(
                [pl.col(name).cast(pl.Float32).alias(name) for name in features]
                + [pl.col("responder_6").cast(pl.Float32), pl.col("weight").cast(pl.Float32)]
            )
            .collect()
        )
        x = frame.select(list(features)).to_numpy()
        y = frame["responder_6"].to_numpy().astype(np.float64, copy=False)
        w = frame["weight"].to_numpy().astype(np.float64, copy=False)
        pred = model.predict(x).astype(np.float64, copy=False)
        err = y - pred
        numerator += float(np.sum(w * err * err))
        denominator += float(np.sum(w * y * y))
        rows += frame.height
        weight_sum += float(np.sum(w))
        prediction_sum += float(np.sum(pred))
        prediction_sumsq += float(np.sum(pred * pred))
    prediction_mean = prediction_sum / rows
    prediction_var = max(prediction_sumsq / rows - prediction_mean * prediction_mean, 0.0)
    return {
        "rows": rows,
        "weight_sum": weight_sum,
        "numerator": numerator,
        "denominator": denominator,
        "weighted_zero_mean_r2": 1.0 - numerator / denominator,
        "prediction_mean": prediction_mean,
        "prediction_std": float(np.sqrt(prediction_var)),
    }


def _summary(results: pl.DataFrame) -> pl.DataFrame:
    return results.select(
        pl.len().alias("folds"),
        pl.col("weighted_zero_mean_r2").mean().alias("mean_r2"),
        pl.col("weighted_zero_mean_r2").std().alias("std_r2"),
        pl.col("weighted_zero_mean_r2").min().alias("min_r2"),
        pl.col("weighted_zero_mean_r2").max().alias("max_r2"),
        (1.0 - pl.col("numerator").sum() / pl.col("denominator").sum()).alias("global_r2"),
        pl.col("train_sample_rows").sum().alias("train_sample_rows"),
        pl.col("rows").sum().alias("validation_rows"),
    )


def _date_chunks(start: int, end: int, chunk_days: int) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    current = start
    while current <= end:
        chunk_end = min(current + chunk_days - 1, end)
        chunks.append((current, chunk_end))
        current = chunk_end + 1
    return chunks


if __name__ == "__main__":
    main()
