"""Analyze Ridge validation residuals by operational slices."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from janestreet.diagnostics import aggregate_weighted_r2_by_slice, combine_slice_aggregates
from janestreet.folds import DateFold, make_expanding_folds, make_rolling_folds
from janestreet.linear import (
    build_weighted_ridge_fit_data,
    evaluate_ridge,
    feature_columns_from_schema,
    solve_weighted_ridge,
)
from janestreet.paths import TRAIN_PARQUET_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-type", choices=["rolling", "expanding"], default="rolling")
    parser.add_argument("--fold-name", default="rw_02")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--valid-window", type=int, default=60)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--chunk-days", type=int, default=10)
    parser.add_argument("--date-bucket-days", type=int, default=10)
    parser.add_argument("--time-bucket-size", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/diagnostics/ridge_rw_02"))
    args = parser.parse_args()

    if args.chunk_days <= 0:
        raise ValueError("--chunk-days must be positive")
    if args.date_bucket_days <= 0:
        raise ValueError("--date-bucket-days must be positive")
    if args.time_bucket_size <= 0:
        raise ValueError("--time-bucket-size must be positive")

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    schema = train.collect_schema()
    features = feature_columns_from_schema(schema)
    folds = _make_folds(train, args)
    fold = _select_fold(folds, args.fold_name)
    weight_thresholds = _weight_thresholds(train.filter(fold.valid_filter()))

    fit_data = build_weighted_ridge_fit_data(
        train,
        fold,
        feature_columns=features,
        chunk_days=args.chunk_days,
    )
    model = solve_weighted_ridge(fit_data, alpha=args.alpha)
    overall = evaluate_ridge(train, fold, model, chunk_days=args.chunk_days)

    slice_specs: dict[str, str | tuple[str, ...]] = {
        "by_date_id": "date_id",
        "by_date_bucket": "date_bucket",
        "by_symbol_id": "symbol_id",
        "by_time_bucket": "time_bucket",
        "by_weight_bucket": "weight_bucket",
        "by_missing_bucket": "missing_bucket",
        "by_date_id_symbol_id": ("date_id", "symbol_id"),
        "by_date_bucket_symbol_id": ("date_bucket", "symbol_id"),
        "by_date_bucket_time_bucket": ("date_bucket", "time_bucket"),
        "by_date_bucket_weight_bucket": ("date_bucket", "weight_bucket"),
        "by_symbol_id_weight_bucket": ("symbol_id", "weight_bucket"),
        "by_symbol_id_missing_bucket": ("symbol_id", "missing_bucket"),
    }
    partials: dict[str, list[pl.DataFrame]] = {name: [] for name in slice_specs}

    for chunk_start, chunk_end in _date_chunks(fold.valid_start, fold.valid_end, args.chunk_days):
        frame = _collect_validation_chunk(train, features, chunk_start, chunk_end)
        if frame.height == 0:
            continue
        predictions = model.predict_array(frame.select(list(features)).to_numpy())
        slice_frame = (
            frame.select(["date_id", "time_id", "symbol_id", "weight", "responder_6", "missing_count"])
            .with_columns(pl.Series("prediction", predictions))
            .with_columns(
                _slice_expressions(
                    fold=fold,
                    weight_thresholds=weight_thresholds,
                    date_bucket_days=args.date_bucket_days,
                    time_bucket_size=args.time_bucket_size,
                )
            )
        )
        for name, by in slice_specs.items():
            partials[name].append(aggregate_weighted_r2_by_slice(slice_frame, by))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined: dict[str, pl.DataFrame] = {}
    for name, by in slice_specs.items():
        combined[name] = combine_slice_aggregates(partials[name], by)
        combined[name].write_csv(args.output_dir / f"{name}.csv")

    summary = {
        "model": "weighted_ridge",
        "fold_type": args.fold_type,
        "fold": _fold_to_dict(fold),
        "alpha": args.alpha,
        "chunk_days": args.chunk_days,
        "date_bucket_days": args.date_bucket_days,
        "time_bucket_size": args.time_bucket_size,
        "n_features": len(features),
        "weight_thresholds": weight_thresholds,
        "overall": overall,
        "worst_slices": {
            name: _head_records(frame, limit=10) for name, frame in combined.items()
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"overall_r2": overall["weighted_zero_mean_r2"], "summary": str(summary_path)}, indent=2))
    for name, frame in combined.items():
        print(f"\n{name}")
        print(frame.head(10))


def _make_folds(train: pl.LazyFrame, args: argparse.Namespace) -> list[DateFold]:
    date_bounds = train.select(
        pl.min("date_id").alias("min_date_id"),
        pl.max("date_id").alias("max_date_id"),
    ).collect()
    min_date_id = int(date_bounds["min_date_id"][0])
    max_date_id = int(date_bounds["max_date_id"][0])
    if args.fold_type == "rolling":
        return make_rolling_folds(
            min_date_id=min_date_id,
            max_date_id=max_date_id,
            n_folds=args.n_folds,
            train_window=args.train_window,
            valid_window=args.valid_window,
            gap=args.gap,
        )
    return make_expanding_folds(
        min_date_id=min_date_id,
        max_date_id=max_date_id,
        n_folds=args.n_folds,
        valid_window=args.valid_window,
        gap=args.gap,
        min_train_window=args.train_window,
    )


def _select_fold(folds: Sequence[DateFold], fold_name: str) -> DateFold:
    for fold in folds:
        if fold.name == fold_name:
            return fold
    available = ", ".join(fold.name for fold in folds)
    raise ValueError(f"unknown fold {fold_name!r}; available folds: {available}")


def _weight_thresholds(valid: pl.LazyFrame) -> dict[str, float]:
    row = valid.select(
        pl.col("weight").quantile(0.50).alias("q50"),
        pl.col("weight").quantile(0.90).alias("q90"),
        pl.col("weight").quantile(0.99).alias("q99"),
    ).collect()
    return {name: float(row[name][0]) for name in ("q50", "q90", "q99")}


def _collect_validation_chunk(
    train: pl.LazyFrame,
    features: tuple[str, ...],
    chunk_start: int,
    chunk_end: int,
) -> pl.DataFrame:
    missing_expr = pl.sum_horizontal(
        *[pl.col(name).is_null().cast(pl.UInt16) for name in features]
    ).alias("missing_count")
    return (
        train.filter(pl.col("date_id").is_between(chunk_start, chunk_end))
        .select(
            [
                pl.col("date_id").cast(pl.Int32),
                pl.col("time_id").cast(pl.Int32),
                pl.col("symbol_id").cast(pl.Int16),
                pl.col("weight").cast(pl.Float64),
                pl.col("responder_6").cast(pl.Float64),
                missing_expr,
            ]
            + [pl.col(name).fill_null(0.0).cast(pl.Float64).alias(name) for name in features]
        )
        .collect()
    )


def _slice_expressions(
    *,
    fold: DateFold,
    weight_thresholds: dict[str, float],
    date_bucket_days: int,
    time_bucket_size: int,
) -> list[pl.Expr]:
    return [
        ((pl.col("date_id") - fold.valid_start) // date_bucket_days)
        .cast(pl.Int16)
        .alias("date_bucket"),
        (pl.col("time_id") // time_bucket_size).cast(pl.Int16).alias("time_bucket"),
        (
            pl.when(pl.col("weight") <= weight_thresholds["q50"])
            .then(pl.lit("q00_q50"))
            .when(pl.col("weight") <= weight_thresholds["q90"])
            .then(pl.lit("q50_q90"))
            .when(pl.col("weight") <= weight_thresholds["q99"])
            .then(pl.lit("q90_q99"))
            .otherwise(pl.lit("q99_q100"))
            .alias("weight_bucket")
        ),
        (
            pl.when(pl.col("missing_count") == 0)
            .then(pl.lit("m00"))
            .when(pl.col("missing_count") <= 5)
            .then(pl.lit("m01_m05"))
            .when(pl.col("missing_count") <= 20)
            .then(pl.lit("m06_m20"))
            .otherwise(pl.lit("m21_plus"))
            .alias("missing_bucket")
        ),
    ]


def _date_chunks(start: int, end: int, chunk_days: int) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    current = start
    while current <= end:
        chunk_end = min(current + chunk_days - 1, end)
        chunks.append((current, chunk_end))
        current = chunk_end + 1
    return chunks


def _fold_to_dict(fold: DateFold) -> dict[str, int | str]:
    return {
        "name": fold.name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "valid_start": fold.valid_start,
        "valid_end": fold.valid_end,
        "train_days": fold.train_days,
        "valid_days": fold.valid_days,
    }


def _head_records(frame: pl.DataFrame, *, limit: int) -> list[dict[str, int | float | str]]:
    return [dict(row) for row in frame.head(limit).iter_rows(named=True)]


if __name__ == "__main__":
    main()
