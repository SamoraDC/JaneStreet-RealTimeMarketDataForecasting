"""Sample-first audit of clock time, trading time, and operational-time structure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from janestreet.linear import feature_columns_from_schema
from janestreet.paths import TRAIN_PARQUET_DIR
from janestreet.time_geometry import OperationalTimeSpec, with_operational_time_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent-days", type=int, default=180)
    parser.add_argument("--n-features", type=int, default=24)
    parser.add_argument("--operational-windows", default="16,64")
    parser.add_argument("--bucket-count", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/time_geometry/reports/audit_recent180"))
    args = parser.parse_args()
    _validate_args(args)

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    schema = train.collect_schema()
    feature_columns = feature_columns_from_schema(schema)[: args.n_features]
    bounds = train.select(
        pl.min("date_id").alias("min_date_id"),
        pl.max("date_id").alias("max_date_id"),
        pl.max("time_id").alias("max_time_id"),
    ).collect()
    max_date_id = int(bounds["max_date_id"][0])
    start_date_id = max(int(bounds["min_date_id"][0]), max_date_id - args.recent_days + 1)
    max_time_id = int(bounds["max_time_id"][0])
    data = train.filter(pl.col("date_id").is_between(start_date_id, max_date_id))
    feature_null_expr = sum(pl.col(name).is_null().cast(pl.Float64) for name in feature_columns) / float(len(feature_columns))

    date_grid = _date_grid_profile(data, feature_null_expr)
    time_profile = _time_id_profile(data, feature_null_expr)
    symbol_profile = _symbol_profile(data, feature_null_expr)
    operational_profile = _operational_profile(
        data,
        feature_columns,
        max_time_id=max_time_id,
        bucket_count=args.bucket_count,
        windows=tuple(int(part) for part in args.operational_windows.split(",") if part.strip()),
    )
    summary = _summary_payload(
        bounds=bounds,
        start_date_id=start_date_id,
        recent_days=args.recent_days,
        date_grid=date_grid,
        time_profile=time_profile,
        feature_columns=feature_columns,
        max_time_id=max_time_id,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    date_grid.write_csv(args.output_dir / "date_grid_profile.csv")
    time_profile.write_csv(args.output_dir / "time_id_profile.csv")
    symbol_profile.write_csv(args.output_dir / "symbol_profile.csv")
    operational_profile.write_csv(args.output_dir / "operational_bucket_profile.csv")
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Wrote {args.output_dir}")


def _date_grid_profile(data: pl.LazyFrame, feature_null_expr: pl.Expr) -> pl.DataFrame:
    return (
        data.group_by("date_id")
        .agg(
            pl.len().alias("rows"),
            pl.col("time_id").n_unique().alias("n_time_ids"),
            pl.col("symbol_id").n_unique().alias("n_symbols"),
            pl.col("time_id").min().alias("min_time_id"),
            pl.col("time_id").max().alias("max_time_id"),
            pl.col("weight").sum().alias("weight_sum"),
            (pl.col("weight") * pl.col("responder_6").pow(2)).sum().alias("target_energy"),
            pl.col("responder_6").abs().mean().alias("target_abs_mean"),
            feature_null_expr.mean().alias("feature_null_frac"),
        )
        .with_columns(
            (pl.col("rows") / (pl.col("n_time_ids") * pl.col("n_symbols"))).alias("date_symbol_time_completeness"),
            (pl.col("rows") / pl.col("n_time_ids")).alias("rows_per_time_id"),
        )
        .sort("date_id")
        .collect()
    )


def _time_id_profile(data: pl.LazyFrame, feature_null_expr: pl.Expr) -> pl.DataFrame:
    return (
        data.group_by("time_id")
        .agg(
            pl.len().alias("rows"),
            pl.col("date_id").n_unique().alias("n_dates"),
            pl.col("symbol_id").n_unique().alias("n_symbols_seen"),
            pl.col("weight").sum().alias("weight_sum"),
            (pl.col("weight") * pl.col("responder_6").pow(2)).sum().alias("target_energy"),
            pl.col("responder_6").abs().mean().alias("target_abs_mean"),
            feature_null_expr.mean().alias("feature_null_frac"),
        )
        .sort("time_id")
        .collect()
    )


def _symbol_profile(data: pl.LazyFrame, feature_null_expr: pl.Expr) -> pl.DataFrame:
    return (
        data.group_by("symbol_id")
        .agg(
            pl.len().alias("rows"),
            pl.col("date_id").n_unique().alias("n_dates"),
            pl.col("time_id").n_unique().alias("n_time_ids_seen"),
            pl.col("weight").sum().alias("weight_sum"),
            (pl.col("weight") * pl.col("responder_6").pow(2)).sum().alias("target_energy"),
            pl.col("responder_6").abs().mean().alias("target_abs_mean"),
            feature_null_expr.mean().alias("feature_null_frac"),
        )
        .sort("symbol_id")
        .collect()
    )


def _operational_profile(
    data: pl.LazyFrame,
    feature_columns: tuple[str, ...],
    *,
    max_time_id: int,
    bucket_count: int,
    windows: tuple[int, ...],
) -> pl.DataFrame:
    spec = OperationalTimeSpec(source_columns=feature_columns, windows=windows, max_time_id=max_time_id)
    with_features = with_operational_time_features(data, spec)
    bucket_size = max(max_time_id // bucket_count, 1)
    return (
        with_features.with_columns(
            [
                (pl.col("time_id") // bucket_size).clip(0, bucket_count - 1).cast(pl.Int16).alias("clock_bucket"),
                ((pl.col("ot_symbol_tick_index") - 1) // bucket_size).clip(0, bucket_count - 1).cast(pl.Int16).alias("symbol_tick_bucket"),
                pl.when(pl.col("ot_symbol_weight_cum") <= 10.0)
                .then(pl.lit("w_cum_000_010"))
                .when(pl.col("ot_symbol_weight_cum") <= 50.0)
                .then(pl.lit("w_cum_010_050"))
                .when(pl.col("ot_symbol_weight_cum") <= 100.0)
                .then(pl.lit("w_cum_050_100"))
                .otherwise(pl.lit("w_cum_100_plus"))
                .alias("symbol_weight_clock_bucket"),
            ]
        )
        .group_by(["clock_bucket", "symbol_tick_bucket", "symbol_weight_clock_bucket"])
        .agg(
            pl.len().alias("rows"),
            pl.col("date_id").n_unique().alias("n_dates"),
            pl.col("symbol_id").n_unique().alias("n_symbols"),
            pl.col("weight").sum().alias("weight_sum"),
            (pl.col("weight") * pl.col("responder_6").pow(2)).sum().alias("target_energy"),
            pl.col("responder_6").abs().mean().alias("target_abs_mean"),
            pl.col("ot_missing_frac").mean().alias("missing_frac_mean"),
            pl.col("ot_source_activity").mean().alias("source_activity_mean"),
        )
        .sort(["clock_bucket", "symbol_tick_bucket", "symbol_weight_clock_bucket"])
        .collect()
    )


def _summary_payload(
    *,
    bounds: pl.DataFrame,
    start_date_id: int,
    recent_days: int,
    date_grid: pl.DataFrame,
    time_profile: pl.DataFrame,
    feature_columns: tuple[str, ...],
    max_time_id: int,
) -> dict[str, object]:
    n_time_min = int(date_grid["n_time_ids"].min())
    n_time_max = int(date_grid["n_time_ids"].max())
    completeness_min = float(date_grid["date_symbol_time_completeness"].min())
    completeness_max = float(date_grid["date_symbol_time_completeness"].max())
    rows_per_time_std = float(date_grid["rows_per_time_id"].std())
    time_energy = time_profile["target_energy"].to_numpy()
    energy_cv = float(time_energy.std() / time_energy.mean()) if float(time_energy.mean()) > 0.0 else float("nan")
    return {
        "scope": {
            "start_date_id": start_date_id,
            "max_date_id": int(bounds["max_date_id"][0]),
            "recent_days_requested": recent_days,
            "feature_columns_used": feature_columns,
        },
        "grid": {
            "max_time_id": max_time_id,
            "n_dates": date_grid.height,
            "n_time_ids_min": n_time_min,
            "n_time_ids_max": n_time_max,
            "time_grid_constant_in_sample": n_time_min == n_time_max,
            "date_symbol_time_completeness_min": completeness_min,
            "date_symbol_time_completeness_max": completeness_max,
            "rows_per_time_id_std_across_dates": rows_per_time_std,
        },
        "target_geometry": {
            "time_id_target_energy_cv": energy_cv,
            "top_5_time_ids_by_target_energy": (
                time_profile.sort("target_energy", descending=True)
                .head(5)
                .select(["time_id", "target_energy", "target_abs_mean", "feature_null_frac"])
                .to_dicts()
            ),
        },
        "interpretation": [
            "A constant time_id grid supports a regular intraday index, but does not prove physical clock time.",
            "Operational-time transforms are valid only as internal row-level features; final predictions must remain on original rows.",
            "Target-energy and responder diagnostics are for research only and must not be used as online features.",
        ],
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.recent_days <= 0:
        raise ValueError("--recent-days must be positive")
    if args.n_features <= 0:
        raise ValueError("--n-features must be positive")
    if args.bucket_count <= 1:
        raise ValueError("--bucket-count must be greater than 1")


if __name__ == "__main__":
    main()
