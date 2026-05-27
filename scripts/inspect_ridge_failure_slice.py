"""Inspect the dominant Ridge failure slice at row level."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from janestreet.diagnostics import aggregate_weighted_r2_by_slice
from janestreet.folds import DateFold, make_rolling_folds
from janestreet.linear import (
    build_weighted_ridge_fit_data,
    feature_columns_from_schema,
    solve_weighted_ridge,
)
from janestreet.paths import TRAIN_PARQUET_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-name", default="rw_02")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--valid-window", type=int, default=60)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--chunk-days", type=int, default=10)
    parser.add_argument("--date-id", type=int, default=1489)
    parser.add_argument("--symbol-id", type=int, default=25)
    parser.add_argument("--time-bucket-size", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/diagnostics/ridge_rw_02_failure_slice"))
    args = parser.parse_args()

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    schema = train.collect_schema()
    features = feature_columns_from_schema(schema)
    fold = _select_fold(_make_rolling_folds(train, args), args.fold_name)

    fit_data = build_weighted_ridge_fit_data(
        train,
        fold,
        feature_columns=features,
        chunk_days=args.chunk_days,
    )
    model = solve_weighted_ridge(fit_data, alpha=args.alpha)

    target_filter = (pl.col("date_id") == args.date_id) & (pl.col("symbol_id") == args.symbol_id)
    groups = {
        "target_slice": target_filter,
        "same_date_other_symbols": (pl.col("date_id") == args.date_id) & (pl.col("symbol_id") != args.symbol_id),
        "same_symbol_other_valid_dates": fold.valid_filter()
        & (pl.col("symbol_id") == args.symbol_id)
        & (pl.col("date_id") != args.date_id),
        "same_symbol_train_window": fold.train_filter() & (pl.col("symbol_id") == args.symbol_id),
        "same_date_all_symbols": pl.col("date_id") == args.date_id,
    }

    prediction_frames = {
        name: _with_predictions(train.filter(expr), features, model, args.time_bucket_size)
        for name, expr in groups.items()
    }
    summaries = [
        _prediction_summary(frame, name)
        for name, frame in prediction_frames.items()
    ]

    target = prediction_frames["target_slice"]
    if target.height == 0:
        raise ValueError("target slice is empty")

    by_time_bucket = aggregate_weighted_r2_by_slice(target, "time_bucket")
    feature_drift = _feature_drift(
        target=target,
        references={
            "same_symbol_train_window": prediction_frames["same_symbol_train_window"],
            "same_date_other_symbols": prediction_frames["same_date_other_symbols"],
            "same_symbol_other_valid_dates": prediction_frames["same_symbol_other_valid_dates"],
        },
        features=features,
        model_means=model.means,
        model_scales=model.scales,
        model_coefficients=model.coefficients,
    )
    feature_contributions = (
        feature_drift.select(
            [
                "feature",
                "coefficient",
                "target_z_global_train",
                "target_linear_contribution",
                "contribution_diff_vs_same_symbol_train_window",
                "contribution_diff_vs_same_date_other_symbols",
                "contribution_diff_vs_same_symbol_other_valid_dates",
                "target_mean",
                "target_null_rate",
            ]
        )
        .with_columns(pl.col("target_linear_contribution").abs().alias("abs_target_linear_contribution"))
        .sort("abs_target_linear_contribution", descending=True)
    )
    top_errors = (
        target.select(
            [
                "date_id",
                "time_id",
                "symbol_id",
                "weight",
                "responder_6",
                "prediction",
                "error",
                "weighted_squared_error",
                "missing_count",
                "time_bucket",
            ]
        )
        .sort("weighted_squared_error", descending=True)
        .head(100)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(summaries).write_csv(args.output_dir / "group_summaries.csv")
    by_time_bucket.write_csv(args.output_dir / "target_by_time_bucket.csv")
    feature_drift.write_csv(args.output_dir / "feature_drift.csv")
    feature_contributions.write_csv(args.output_dir / "feature_contributions.csv")
    top_errors.write_csv(args.output_dir / "target_top_errors.csv")
    target.select(
        [
            "date_id",
            "time_id",
            "symbol_id",
            "weight",
            "responder_6",
            "prediction",
            "error",
            "weighted_squared_error",
            "missing_count",
            "time_bucket",
        ]
    ).write_csv(args.output_dir / "target_rows.csv")

    summary = {
        "model": "weighted_ridge",
        "fold": _fold_to_dict(fold),
        "alpha": args.alpha,
        "target": {"date_id": args.date_id, "symbol_id": args.symbol_id},
        "group_summaries": summaries,
        "worst_time_buckets": _records(by_time_bucket.head(10)),
        "largest_feature_drifts": _records(feature_drift.head(20)),
        "largest_feature_contributions": _records(feature_contributions.head(20)),
        "top_error_rows": _records(top_errors.head(20)),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"summary": str(summary_path), "target_summary": summaries[0]}, indent=2))
    print("\nGroup summaries")
    print(pl.DataFrame(summaries))
    print("\nTarget by time bucket")
    print(by_time_bucket)
    print("\nLargest feature drifts")
    print(feature_drift.head(20))
    print("\nLargest feature contributions")
    print(feature_contributions.head(20))


def _make_rolling_folds(train: pl.LazyFrame, args: argparse.Namespace) -> list[DateFold]:
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


def _select_fold(folds: list[DateFold], name: str) -> DateFold:
    for fold in folds:
        if fold.name == name:
            return fold
    available = ", ".join(fold.name for fold in folds)
    raise ValueError(f"unknown fold {name!r}; available folds: {available}")


def _with_predictions(
    data: pl.LazyFrame,
    features: tuple[str, ...],
    model,
    time_bucket_size: int,
) -> pl.DataFrame:
    missing_expr = pl.sum_horizontal(
        *[pl.col(name).is_null().cast(pl.UInt16) for name in features]
    ).alias("missing_count")
    frame = data.select(
        [
            pl.col("date_id").cast(pl.Int32),
            pl.col("time_id").cast(pl.Int32),
            pl.col("symbol_id").cast(pl.Int16),
            pl.col("weight").cast(pl.Float64),
            pl.col("responder_6").cast(pl.Float64),
            missing_expr,
        ]
        + [pl.col(name).fill_null(0.0).cast(pl.Float64).alias(name) for name in features]
        + [pl.col(name).is_null().cast(pl.Float64).alias(f"{name}__is_null") for name in features]
    ).collect()
    if frame.height == 0:
        return frame.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("prediction"),
                pl.lit(None, dtype=pl.Float64).alias("error"),
                pl.lit(None, dtype=pl.Float64).alias("weighted_squared_error"),
                pl.lit(None, dtype=pl.Int16).alias("time_bucket"),
            ]
        )
    predictions = model.predict_array(frame.select(list(features)).to_numpy())
    return frame.with_columns(
        [
            pl.Series("prediction", predictions),
            (pl.col("time_id") // time_bucket_size).cast(pl.Int16).alias("time_bucket"),
        ]
    ).with_columns(
        [
            (pl.col("responder_6") - pl.col("prediction")).alias("error"),
            (pl.col("weight") * (pl.col("responder_6") - pl.col("prediction")).pow(2)).alias(
                "weighted_squared_error"
            ),
        ]
    )


def _prediction_summary(frame: pl.DataFrame, name: str) -> dict[str, float | int | str | None]:
    if frame.height == 0:
        return {"group": name, "rows": 0}
    row = frame.select(
        [
            pl.len().alias("rows"),
            pl.col("weight").sum().alias("weight_sum"),
            pl.col("weighted_squared_error").sum().alias("numerator"),
            (pl.col("weight") * pl.col("responder_6").pow(2)).sum().alias("denominator"),
            pl.col("responder_6").mean().alias("target_mean"),
            pl.col("responder_6").std().alias("target_std"),
            pl.col("prediction").mean().alias("prediction_mean"),
            pl.col("prediction").std().alias("prediction_std"),
            pl.col("error").mean().alias("error_mean"),
            pl.col("error").std().alias("error_std"),
            pl.col("weighted_squared_error").max().alias("max_weighted_squared_error"),
            pl.col("missing_count").mean().alias("missing_mean"),
            pl.col("missing_count").max().alias("missing_max"),
        ]
    ).row(0, named=True)
    denominator = float(row["denominator"])
    row["group"] = name
    row["weighted_zero_mean_r2"] = None if denominator <= 0.0 else 1.0 - float(row["numerator"]) / denominator
    return row


def _feature_drift(
    *,
    target: pl.DataFrame,
    references: dict[str, pl.DataFrame],
    features: tuple[str, ...],
    model_means: np.ndarray,
    model_scales: np.ndarray,
    model_coefficients: np.ndarray,
) -> pl.DataFrame:
    target_profile = _feature_profile(target, features)
    reference_profiles = {name: _feature_profile(frame, features) for name, frame in references.items()}
    rows: list[dict[str, float | str]] = []
    for idx, feature in enumerate(features):
        target_stats = target_profile[feature]
        coefficient = float(model_coefficients[idx])
        model_scale = float(model_scales[idx])
        target_z_global = (target_stats["mean"] - float(model_means[idx])) / model_scale
        row: dict[str, float | str] = {
            "feature": feature,
            "coefficient": coefficient,
            "target_mean": target_stats["mean"],
            "target_std": target_stats["std"],
            "target_null_rate": target_stats["null_rate"],
            "target_z_global_train": target_z_global,
            "z_vs_global_train": target_z_global,
            "target_linear_contribution": coefficient * target_z_global,
        }
        for name, profile in reference_profiles.items():
            ref = profile[feature]
            scale = max(abs(ref["std"]), 1e-12)
            row[f"z_vs_{name}"] = (target_stats["mean"] - ref["mean"]) / scale
            row[f"null_rate_diff_vs_{name}"] = target_stats["null_rate"] - ref["null_rate"]
            row[f"contribution_diff_vs_{name}"] = coefficient * (
                (target_stats["mean"] - ref["mean"]) / model_scale
            )
        rows.append(row)
    return (
        pl.DataFrame(rows)
        .with_columns(pl.col("z_vs_same_symbol_train_window").abs().alias("abs_z_vs_same_symbol_train_window"))
        .sort("abs_z_vs_same_symbol_train_window", descending=True)
    )


def _feature_profile(frame: pl.DataFrame, features: tuple[str, ...]) -> dict[str, dict[str, float]]:
    if frame.height == 0:
        return {
            feature: {"mean": float("nan"), "std": float("nan"), "null_rate": float("nan")}
            for feature in features
        }
    raw = frame.select(
        [
            expr
            for feature in features
            for expr in (
                pl.col(feature).mean().alias(f"{feature}__mean"),
                pl.col(feature).std().alias(f"{feature}__std"),
                pl.col(f"{feature}__is_null").mean().alias(f"{feature}__null_rate"),
            )
        ]
    ).row(0, named=True)
    return {
        feature: {
            "mean": _finite_or_zero(raw[f"{feature}__mean"]),
            "std": max(_finite_or_zero(raw[f"{feature}__std"]), 1e-12),
            "null_rate": _finite_or_zero(raw[f"{feature}__null_rate"]),
        }
        for feature in features
    }


def _finite_or_zero(value: object) -> float:
    result = float(value) if value is not None else 0.0
    return result if np.isfinite(result) else 0.0


def _fold_to_dict(fold: DateFold) -> dict[str, int | str]:
    return {
        "name": fold.name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "valid_start": fold.valid_start,
        "valid_end": fold.valid_end,
    }


def _records(frame: pl.DataFrame) -> list[dict[str, int | float | str | None]]:
    return [dict(row) for row in frame.iter_rows(named=True)]


if __name__ == "__main__":
    main()
