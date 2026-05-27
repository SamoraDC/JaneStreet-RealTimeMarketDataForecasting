"""Run the zero-prediction baseline on real validation folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from janestreet.baselines import evaluate_constant_prediction_by_fold
from janestreet.folds import make_expanding_folds
from janestreet.paths import TRAIN_PARQUET_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--valid-window", type=int, default=120)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/baselines"))
    args = parser.parse_args()

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    date_bounds = train.select(
        pl.min("date_id").alias("min_date_id"),
        pl.max("date_id").alias("max_date_id"),
    ).collect()
    min_date_id = int(date_bounds["min_date_id"][0])
    max_date_id = int(date_bounds["max_date_id"][0])

    folds = make_expanding_folds(
        min_date_id=min_date_id,
        max_date_id=max_date_id,
        n_folds=args.n_folds,
        valid_window=args.valid_window,
        gap=args.gap,
        min_train_window=365,
    )
    results = evaluate_constant_prediction_by_fold(
        train,
        folds,
        prediction_value=0.0,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "zero_baseline.csv"
    json_path = args.output_dir / "zero_baseline_summary.json"
    results.write_csv(csv_path)

    summary = {
        "baseline": "constant_zero",
        "n_folds": args.n_folds,
        "valid_window": args.valid_window,
        "gap": args.gap,
        "min_date_id": min_date_id,
        "max_date_id": max_date_id,
        "mean_weighted_zero_mean_r2": results["weighted_zero_mean_r2"].mean(),
        "total_validation_rows": int(results["rows"].sum()),
        "csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(results)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

