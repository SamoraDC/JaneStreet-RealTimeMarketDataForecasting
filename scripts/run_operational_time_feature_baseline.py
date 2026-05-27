"""Sampled GBDT tests for operational-time features on original rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from janestreet.folds import DateFold, make_rolling_folds
from janestreet.linear import feature_columns_from_schema
from janestreet.paths import TRAIN_PARQUET_DIR
from janestreet.time_geometry import OperationalTimeSpec, with_operational_time_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--valid-window", type=int, default=20)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--engine", choices=["lightgbm", "xgboost"], default="lightgbm")
    parser.add_argument("--feature-set", choices=["clock", "operational"], default="operational")
    parser.add_argument("--n-operational-source-features", type=int, default=32)
    parser.add_argument("--operational-windows", default="16,64")
    parser.add_argument("--train-sample-frac", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--chunk-days", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/time_geometry/reports/operational_lightgbm_smoke"))
    args = parser.parse_args()
    _validate_args(args)

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    schema = train.collect_schema()
    base_features = feature_columns_from_schema(schema)
    source_features = base_features[: args.n_operational_source_features]
    bounds = train.select(
        pl.min("date_id").alias("min_date_id"),
        pl.max("date_id").alias("max_date_id"),
        pl.max("time_id").alias("max_time_id"),
    ).collect()
    max_time_id = int(bounds["max_time_id"][0])
    data = train
    operational_features: tuple[str, ...] = ()
    if args.feature_set == "operational":
        spec = OperationalTimeSpec(
            source_columns=source_features,
            windows=tuple(int(part) for part in args.operational_windows.split(",") if part.strip()),
            max_time_id=max_time_id,
        )
        data = with_operational_time_features(data, spec)
        operational_features = spec.output_columns
    model_features = tuple(dict.fromkeys([*base_features, "time_id", "symbol_id", *operational_features]))
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
            data,
            model_features,
            fold,
            sample_frac=args.train_sample_frac,
            seed=args.seed,
        )
        model = _make_model(args)
        model.fit(x_train, y_train, sample_weight=w_train)
        rows.append(
            {
                **_fold_metadata(fold),
                "engine": args.engine,
                "feature_set": args.feature_set,
                "n_features": len(model_features),
                "train_sample_rows": int(x_train.shape[0]),
                **_evaluate_model(data, model_features, model, fold, args.chunk_days),
            }
        )

    results = pl.DataFrame(rows)
    summary = _summary(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(args.output_dir / "operational_time_by_fold.csv")
    summary.write_csv(args.output_dir / "operational_time_summary.csv")
    report = {
        "experiment": "operational_time_feature_baseline",
        "engine": args.engine,
        "feature_set": args.feature_set,
        "n_folds": args.n_folds,
        "train_window": args.train_window,
        "valid_window": args.valid_window,
        "train_sample_frac": args.train_sample_frac,
        "model_features": len(model_features),
        "operational_features": list(operational_features),
        "summary": summary.to_dicts(),
    }
    (args.output_dir / "operational_time_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(results)
    print(summary)
    print(f"Wrote {args.output_dir}")


def _make_model(args: argparse.Namespace) -> LGBMRegressor | XGBRegressor:
    if args.engine == "lightgbm":
        return LGBMRegressor(
            objective="regression",
            n_estimators=args.max_iter,
            learning_rate=args.learning_rate,
            num_leaves=args.max_leaf_nodes,
            min_child_samples=args.min_child_samples,
            subsample=1.0,
            colsample_bytree=1.0,
            random_state=args.seed,
            n_jobs=args.n_jobs,
            verbosity=-1,
        )
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=args.max_iter,
        learning_rate=args.learning_rate,
        max_leaves=args.max_leaf_nodes,
        tree_method="hist",
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbosity=0,
    )


def _collect_train_sample(
    data: pl.LazyFrame,
    model_features: tuple[str, ...],
    fold: DateFold,
    *,
    sample_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    threshold = int(sample_frac * 1_000_000)
    sample_filter = (pl.struct(["date_id", "time_id", "symbol_id"]).hash(seed=seed) % 1_000_000) < threshold
    frame = (
        data.filter(pl.col("date_id").is_between(fold.train_start, fold.train_end) & sample_filter)
        .select(
            [pl.col(name).fill_null(0.0).cast(pl.Float32).alias(name) for name in model_features]
            + [pl.col("responder_6").cast(pl.Float32), pl.col("weight").cast(pl.Float32)]
        )
        .collect()
    )
    if frame.height == 0:
        raise ValueError(f"empty train sample for {fold.name}")
    return (
        frame.select(list(model_features)).to_numpy(),
        frame["responder_6"].to_numpy(),
        frame["weight"].to_numpy(),
    )


def _evaluate_model(
    data: pl.LazyFrame,
    model_features: tuple[str, ...],
    model: LGBMRegressor | XGBRegressor,
    fold: DateFold,
    chunk_days: int,
) -> dict[str, float | int]:
    numerator = 0.0
    denominator = 0.0
    rows = 0
    for chunk_start, chunk_end in _date_chunks(fold.valid_start, fold.valid_end, chunk_days):
        frame = (
            data.filter(pl.col("date_id").is_between(chunk_start, chunk_end))
            .select(
                [pl.col(name).fill_null(0.0).cast(pl.Float32).alias(name) for name in model_features]
                + [pl.col("responder_6").cast(pl.Float64), pl.col("weight").cast(pl.Float64)]
            )
            .collect()
        )
        if frame.height == 0:
            continue
        x = frame.select(list(model_features)).to_numpy()
        y = frame["responder_6"].to_numpy()
        w = frame["weight"].to_numpy()
        pred = model.predict(x).astype(np.float64, copy=False)
        numerator += float(np.sum(w * (y - pred) ** 2))
        denominator += float(np.sum(w * y * y))
        rows += frame.height
    if denominator <= 0.0:
        raise ValueError(f"{fold.name} has non-positive target energy")
    return {
        "rows": rows,
        "numerator": numerator,
        "denominator": denominator,
        "weighted_zero_mean_r2": 1.0 - numerator / denominator,
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


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_folds <= 0 or args.train_window <= 0 or args.valid_window <= 0:
        raise ValueError("fold counts and windows must be positive")
    if not 0.0 < args.train_sample_frac <= 1.0:
        raise ValueError("--train-sample-frac must be in (0, 1]")
    if args.n_operational_source_features <= 0:
        raise ValueError("--n-operational-source-features must be positive")
    if args.chunk_days <= 0:
        raise ValueError("--chunk-days must be positive")


if __name__ == "__main__":
    main()
