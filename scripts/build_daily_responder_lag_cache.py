"""Materialize daily-last official responder lags with bounded memory."""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Iterator
from pathlib import Path

import polars as pl

from janestreet.official_lags import RESPONDER_COLUMNS, daily_last_responder_lag_columns
from janestreet.paths import DAILY_RESPONDER_LAGS_LAST_PARQUET, TRAIN_PARQUET_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=TRAIN_PARQUET_DIR)
    parser.add_argument("--output", type=Path, default=DAILY_RESPONDER_LAGS_LAST_PARQUET)
    parser.add_argument("--chunk-days", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.chunk_days <= 0:
        raise ValueError("--chunk-days must be positive")
    if args.output.exists() and not args.force:
        raise FileExistsError(f"{args.output} already exists; pass --force to rebuild")
    if args.output.exists():
        if args.output.is_dir():
            shutil.rmtree(args.output)
        else:
            args.output.unlink()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    bounds = pl.scan_parquet(str(args.input)).select(
        pl.col("date_id").min().alias("min_date_id"),
        pl.col("date_id").max().alias("max_date_id"),
    ).collect()
    min_date_id = int(bounds["min_date_id"][0])
    max_date_id = int(bounds["max_date_id"][0])
    lag_columns = daily_last_responder_lag_columns()

    for part_idx, (chunk_start, chunk_end) in enumerate(_date_chunks(min_date_id, max_date_id, args.chunk_days)):
        lags = (
            pl.scan_parquet(str(args.input))
            .filter(pl.col("date_id").is_between(chunk_start, chunk_end))
            .select(["date_id", "time_id", "symbol_id", *RESPONDER_COLUMNS])
            .sort(["date_id", "symbol_id", "time_id"])
            .group_by(["date_id", "symbol_id"], maintain_order=True)
            .agg(
                [
                    pl.col(source).last().alias(target)
                    for source, target in zip(RESPONDER_COLUMNS, lag_columns, strict=True)
                ]
            )
            .with_columns((pl.col("date_id") + 1).cast(pl.Int16).alias("date_id"))
            .collect()
        )
        part_path = args.output / f"part-{part_idx:04d}.parquet"
        lags.write_parquet(part_path, compression="zstd")
        print(f"part={part_idx:04d} source_dates={chunk_start}-{chunk_end} rows={lags.height}")

    summary = pl.scan_parquet(str(args.output / "*.parquet")).select(
        pl.len().alias("rows"),
        pl.col("date_id").min().alias("min_date_id"),
        pl.col("date_id").max().alias("max_date_id"),
        pl.col("responder_6_daily_last_lag_1").std().alias("responder_6_daily_last_lag_1_std"),
    ).collect()
    print(summary)
    print(f"lag_columns={lag_columns}")
    print(f"wrote {args.output}")


def _date_chunks(start: int, end: int, chunk_days: int) -> Iterator[tuple[int, int]]:
    current = start
    while current <= end:
        chunk_end = min(current + chunk_days - 1, end)
        yield current, chunk_end
        current = chunk_end + 1


if __name__ == "__main__":
    main()
