"""Run a low-cost weighted Ridge smoke baseline on recent real data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from janestreet.folds import DateFold
from janestreet.linear import evaluate_ridge, feature_columns_from_schema, fit_weighted_ridge
from janestreet.paths import TRAIN_PARQUET_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-window", type=int, default=30)
    parser.add_argument("--valid-window", type=int, default=30)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--chunk-days", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/baselines"))
    args = parser.parse_args()

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    schema = train.collect_schema()
    features = feature_columns_from_schema(schema)
    max_date_id = int(train.select(pl.max("date_id")).collect().item())
    valid_end = max_date_id
    valid_start = valid_end - args.valid_window + 1
    train_end = valid_start - args.gap - 1
    train_start = train_end - args.train_window + 1
    if train_start < 0:
        raise ValueError("requested windows extend before date_id=0")

    fold = DateFold(
        name="ridge_smoke_recent",
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
    )
    model = fit_weighted_ridge(
        train,
        fold,
        feature_columns=features,
        alpha=args.alpha,
        chunk_days=args.chunk_days,
    )
    result = evaluate_ridge(
        train,
        fold,
        model,
        chunk_days=args.chunk_days,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "ridge_smoke_summary.json"
    json_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

