"""Evaluate a small TabPFN regression baseline on rolling folds."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor

from janestreet.folds import DateFold, make_rolling_folds
from janestreet.linear import feature_columns_from_schema
from janestreet.paths import PROJECT_ROOT, TRAIN_PARQUET_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=1)
    parser.add_argument("--train-window", type=int, default=30)
    parser.add_argument("--valid-window", type=int, default=5)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--id-columns", default="time_id,symbol_id")
    parser.add_argument("--max-train-rows", type=int, default=4096)
    parser.add_argument("--max-valid-rows", type=int, default=50_000)
    parser.add_argument("--n-estimators", type=int, default=2)
    parser.add_argument("--predict-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-path", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/tabpfn_smoke"))
    args = parser.parse_args()
    _validate_args(args)
    _load_dotenv(PROJECT_ROOT / ".env")

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    schema = train.collect_schema()
    id_columns = _parse_id_columns(args.id_columns)
    feature_columns = _select_feature_columns(schema, args.n_features, id_columns)
    folds = _make_folds(train, args)
    rows: list[dict[str, float | int | str]] = []

    for fold in folds:
        train_frame = _collect_sample(
            train,
            feature_columns,
            fold.train_start,
            fold.train_end,
            max_rows=args.max_train_rows,
            seed=args.seed,
        )
        valid_frame = _collect_sample(
            train,
            feature_columns,
            fold.valid_start,
            fold.valid_end,
            max_rows=args.max_valid_rows,
            seed=args.seed + 1,
        )
        x_train, y_train, w_train = _frame_arrays(train_frame, feature_columns)
        x_valid, y_valid, w_valid = _frame_arrays(valid_frame, feature_columns)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train, sample_weight=w_train).astype(np.float32, copy=False)
        x_valid = scaler.transform(x_valid).astype(np.float32, copy=False)
        model = TabPFNRegressor(
            n_estimators=args.n_estimators,
            device=args.device,
            model_path=args.model_path,
            ignore_pretraining_limits=True,
            fit_mode="low_memory",
            memory_saving_mode="auto",
            random_state=args.seed,
            show_progress_bar=False,
        )
        model.fit(x_train, y_train)
        pred = _predict_in_batches(model, x_valid, batch_size=args.predict_batch_size)
        rows.append(
            {
                **_fold_metadata(fold),
                "train_rows": int(x_train.shape[0]),
                "valid_rows": int(x_valid.shape[0]),
                "n_features": len(feature_columns),
                **_evaluate_prediction(y_valid, pred, w_valid),
            }
        )

    results = pl.DataFrame(rows)
    summary = _summary(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(args.output_dir / "tabpfn_by_fold.csv")
    summary.write_csv(args.output_dir / "tabpfn_summary.csv")
    report = {
        "experiment": "tabpfn_baseline",
        "n_folds": args.n_folds,
        "train_window": args.train_window,
        "valid_window": args.valid_window,
        "n_features": args.n_features,
        "id_columns": id_columns,
        "max_train_rows": args.max_train_rows,
        "max_valid_rows": args.max_valid_rows,
        "n_estimators": args.n_estimators,
        "predict_batch_size": args.predict_batch_size,
        "device": args.device,
        "model_path": args.model_path,
        "summary": summary.to_dicts(),
    }
    (args.output_dir / "tabpfn_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(results)
    print(summary)
    print(f"Wrote {args.output_dir}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_folds <= 0:
        raise ValueError("--n-folds must be positive")
    if args.train_window <= 0 or args.valid_window <= 0:
        raise ValueError("--train-window and --valid-window must be positive")
    if args.n_features <= 0:
        raise ValueError("--n-features must be positive")
    if args.max_train_rows <= 0 or args.max_valid_rows <= 0:
        raise ValueError("TabPFN row limits must be positive")
    if args.n_estimators <= 0:
        raise ValueError("--n-estimators must be positive")
    if args.predict_batch_size <= 0:
        raise ValueError("--predict-batch-size must be positive")


def _load_dotenv(path: Path) -> tuple[str, ...]:
    """Load key-value pairs from a .env file without returning secret values."""

    if not path.exists():
        return ()
    loaded: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return tuple(loaded)


def _parse_id_columns(raw: str) -> tuple[str, ...]:
    allowed = {"time_id", "symbol_id"}
    columns = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    unknown = sorted(set(columns) - allowed)
    if unknown:
        raise ValueError(f"unknown id columns: {', '.join(unknown)}")
    return columns


def _select_feature_columns(
    schema: pl.Schema,
    n_features: int,
    id_columns: tuple[str, ...],
) -> tuple[str, ...]:
    base = feature_columns_from_schema(schema)[:n_features]
    if len(base) < n_features:
        raise ValueError(f"requested {n_features} features, only {len(base)} available")
    return tuple(dict.fromkeys([*base, *id_columns]))


def _collect_sample(
    data: pl.LazyFrame,
    feature_columns: Sequence[str],
    start: int,
    end: int,
    *,
    max_rows: int,
    seed: int,
) -> pl.DataFrame:
    row_hash = pl.struct(["date_id", "time_id", "symbol_id"]).hash(seed=seed)
    return (
        data.filter(pl.col("date_id").is_between(start, end))
        .with_columns(row_hash.alias("_sample_hash"))
        .select(
            [pl.col(name).fill_null(0.0).cast(pl.Float32).alias(name) for name in feature_columns]
            + [pl.col("responder_6").cast(pl.Float32), pl.col("weight").cast(pl.Float32), pl.col("_sample_hash")]
        )
        .sort("_sample_hash")
        .head(max_rows)
        .drop("_sample_hash")
        .collect()
    )


def _frame_arrays(
    frame: pl.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frame.height == 0:
        raise ValueError("empty TabPFN sample")
    return (
        frame.select(list(feature_columns)).to_numpy().astype(np.float32, copy=False),
        frame["responder_6"].to_numpy().astype(np.float32, copy=False),
        frame["weight"].to_numpy().astype(np.float32, copy=False),
    )


def _evaluate_prediction(
    y: np.ndarray,
    pred: np.ndarray,
    weight: np.ndarray,
) -> dict[str, float]:
    y64 = y.astype(np.float64, copy=False)
    w64 = weight.astype(np.float64, copy=False)
    err = y64 - pred
    numerator = float(np.sum(w64 * err * err))
    denominator = float(np.sum(w64 * y64 * y64))
    if denominator <= 0.0:
        raise ValueError("validation target energy must be positive")
    return {
        "numerator": numerator,
        "denominator": denominator,
        "weighted_zero_mean_r2": 1.0 - numerator / denominator,
        "weight_sum": float(np.sum(w64)),
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
    }


def _predict_in_batches(model: TabPFNRegressor, x: np.ndarray, *, batch_size: int) -> np.ndarray:
    predictions: list[np.ndarray] = []
    for start in range(0, x.shape[0], batch_size):
        end = min(start + batch_size, x.shape[0])
        predictions.append(model.predict(x[start:end]).astype(np.float64, copy=False))
    return np.concatenate(predictions)


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
        gap=args.gap,
    )


def _summary(results: pl.DataFrame) -> pl.DataFrame:
    return results.select(
        pl.len().alias("folds"),
        pl.col("weighted_zero_mean_r2").mean().alias("mean_r2"),
        pl.col("weighted_zero_mean_r2").std().alias("std_r2"),
        pl.col("weighted_zero_mean_r2").min().alias("min_r2"),
        pl.col("weighted_zero_mean_r2").max().alias("max_r2"),
        (1.0 - pl.col("numerator").sum() / pl.col("denominator").sum()).alias("global_r2"),
        pl.col("train_rows").sum().alias("train_rows"),
        pl.col("valid_rows").sum().alias("validation_rows"),
    )


def _fold_metadata(fold: DateFold) -> dict[str, int | str]:
    return {
        "fold": fold.name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "valid_start": fold.valid_start,
        "valid_end": fold.valid_end,
    }


if __name__ == "__main__":
    main()
