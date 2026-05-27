"""Compare statistical properties of alternative time clocks on small aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from janestreet.linear import feature_columns_from_schema
from janestreet.paths import TRAIN_PARQUET_DIR
from janestreet.time_geometry import OperationalTimeSpec, with_operational_time_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recent-days", type=int, default=180)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--bucket-count", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/time_geometry/reports/clock_property_recent180"))
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
    base = train.filter(pl.col("date_id").is_between(start_date_id, max_date_id))
    spec = OperationalTimeSpec(source_columns=feature_columns, windows=(16, 64), max_time_id=max_time_id)
    with_ot = with_operational_time_features(base, spec)
    with_batch_state = _add_batch_state(with_ot)
    clocked = _add_clock_buckets(with_batch_state, bucket_count=args.bucket_count, max_time_id=max_time_id)

    detail_frames = []
    summary_rows = []
    for clock_name in _clock_columns():
        detail = _aggregate_by_clock(clocked, clock_name).with_columns(pl.lit(clock_name).alias("clock"))
        detail_frames.append(detail)
        summary_rows.append(_summarize_clock(clocked, detail, clock_name))

    detail_frame = pl.concat(detail_frames).select(
        ["clock", "bucket", "rows", "n_dates", "n_symbols", "weight_sum", "target_energy", "target_abs_mean", "feature_null_frac"]
    )
    summary = pl.DataFrame(summary_rows).sort(["energy_cv", "date_bucket_energy_cv_mean"])
    report = {
        "scope": {
            "start_date_id": start_date_id,
            "max_date_id": max_date_id,
            "recent_days_requested": args.recent_days,
            "bucket_count": args.bucket_count,
            "feature_columns_used": feature_columns,
        },
        "summary": summary.to_dicts(),
        "interpretation": [
            "Lower row_cv means the clock distributes rows more evenly across buckets.",
            "Lower energy_cv means weighted responder_6 energy is more evenly distributed across clock buckets.",
            "Lower date_bucket_energy_cv_mean means each bucket is more stable across dates.",
            "Target-energy metrics are diagnostics only; target-derived clocks are not valid online features.",
        ],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.output_dir / "clock_property_summary.csv")
    detail_frame.write_csv(args.output_dir / "clock_property_by_bucket.csv")
    (args.output_dir / "clock_property_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary)
    print(f"Wrote {args.output_dir}")


def _add_batch_state(data: pl.LazyFrame) -> pl.LazyFrame:
    groups = ["date_id", "time_id"]
    return data.with_columns(
        [
            pl.col("weight").sum().over(groups).cast(pl.Float32).alias("batch_weight_sum"),
            pl.col("ot_source_activity").mean().over(groups).cast(pl.Float32).alias("batch_activity_mean"),
            pl.col("ot_missing_frac").mean().over(groups).cast(pl.Float32).alias("batch_missing_frac"),
        ]
    )


def _add_clock_buckets(data: pl.LazyFrame, *, bucket_count: int, max_time_id: int) -> pl.LazyFrame:
    clock_bucket_size = max(max_time_id // bucket_count, 1)
    return data.with_columns(
        [
            (pl.col("time_id") // clock_bucket_size).clip(0, bucket_count - 1).cast(pl.Int16).alias("clock_time_bucket"),
            _rank_bucket("ot_symbol_weight_cum", bucket_count).alias("symbol_weight_cum_rank_bucket"),
            _rank_bucket("ot_source_activity", bucket_count).alias("row_activity_rank_bucket"),
            _rank_bucket("ot_source_activity_ewm_64", bucket_count).alias("symbol_activity_ewm_rank_bucket"),
            _rank_bucket("batch_weight_sum", bucket_count).alias("batch_weight_rank_bucket"),
            _rank_bucket("batch_activity_mean", bucket_count).alias("batch_activity_rank_bucket"),
            _rank_bucket("batch_missing_frac", bucket_count).alias("batch_missing_rank_bucket"),
        ]
    )


def _rank_bucket(column: str, bucket_count: int) -> pl.Expr:
    return (
        (((pl.col(column).rank(method="average") - 1.0) / pl.len()) * bucket_count)
        .floor()
        .clip(0, bucket_count - 1)
        .cast(pl.Int16)
    )


def _clock_columns() -> tuple[str, ...]:
    return (
        "clock_time_bucket",
        "symbol_weight_cum_rank_bucket",
        "row_activity_rank_bucket",
        "symbol_activity_ewm_rank_bucket",
        "batch_weight_rank_bucket",
        "batch_activity_rank_bucket",
        "batch_missing_rank_bucket",
    )


def _aggregate_by_clock(data: pl.LazyFrame, clock_name: str) -> pl.DataFrame:
    feature_null_frac = pl.col("ot_missing_frac").mean()
    return (
        data.group_by(clock_name)
        .agg(
            pl.len().alias("rows"),
            pl.col("date_id").n_unique().alias("n_dates"),
            pl.col("symbol_id").n_unique().alias("n_symbols"),
            pl.col("weight").sum().alias("weight_sum"),
            (pl.col("weight") * pl.col("responder_6").pow(2)).sum().alias("target_energy"),
            pl.col("responder_6").abs().mean().alias("target_abs_mean"),
            feature_null_frac.alias("feature_null_frac"),
        )
        .rename({clock_name: "bucket"})
        .sort("bucket")
        .collect()
    )


def _summarize_clock(data: pl.LazyFrame, detail: pl.DataFrame, clock_name: str) -> dict[str, float | int | str]:
    row_values = detail["rows"].to_numpy().astype(np.float64)
    energy_values = detail["target_energy"].to_numpy().astype(np.float64)
    return {
        "clock": clock_name,
        "buckets": detail.height,
        "row_cv": _cv(row_values),
        "energy_cv": _cv(energy_values),
        "energy_share_max": float(energy_values.max() / energy_values.sum()),
        "target_abs_mean_spread": float(detail["target_abs_mean"].max() - detail["target_abs_mean"].min()),
        "date_bucket_energy_cv_mean": _date_bucket_energy_cv_mean(data, clock_name),
    }


def _date_bucket_energy_cv_mean(data: pl.LazyFrame, clock_name: str) -> float:
    date_bucket = (
        data.group_by(["date_id", clock_name])
        .agg((pl.col("weight") * pl.col("responder_6").pow(2)).sum().alias("target_energy"))
        .collect()
    )
    by_bucket = (
        date_bucket.group_by(clock_name)
        .agg(
            pl.col("target_energy").mean().alias("energy_mean"),
            pl.col("target_energy").std(ddof=0).alias("energy_std"),
        )
        .with_columns((pl.col("energy_std") / pl.col("energy_mean")).alias("energy_cv"))
        .filter(pl.col("energy_mean") > 0.0)
    )
    if by_bucket.height == 0:
        return float("nan")
    return float(by_bucket["energy_cv"].mean())


def _cv(values: np.ndarray) -> float:
    mean = float(values.mean())
    if mean <= 0.0:
        return float("nan")
    return float(values.std(ddof=0) / mean)


def _validate_args(args: argparse.Namespace) -> None:
    if args.recent_days <= 0:
        raise ValueError("--recent-days must be positive")
    if args.n_features <= 0:
        raise ValueError("--n-features must be positive")
    if args.bucket_count <= 1:
        raise ValueError("--bucket-count must be greater than 1")


if __name__ == "__main__":
    main()
