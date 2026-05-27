"""Run weighted Ridge over multiple temporal folds and alphas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from janestreet.folds import make_expanding_folds, make_rolling_folds
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
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--valid-window", type=int, default=60)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--alphas", type=str, default="10,100,1000")
    parser.add_argument("--chunk-days", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/baselines"))
    args = parser.parse_args()

    alphas = _parse_alphas(args.alphas)
    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    schema = train.collect_schema()
    features = feature_columns_from_schema(schema)
    date_bounds = train.select(
        pl.min("date_id").alias("min_date_id"),
        pl.max("date_id").alias("max_date_id"),
    ).collect()
    min_date_id = int(date_bounds["min_date_id"][0])
    max_date_id = int(date_bounds["max_date_id"][0])

    if args.fold_type == "rolling":
        folds = make_rolling_folds(
            min_date_id=min_date_id,
            max_date_id=max_date_id,
            n_folds=args.n_folds,
            train_window=args.train_window,
            valid_window=args.valid_window,
            gap=args.gap,
        )
    else:
        folds = make_expanding_folds(
            min_date_id=min_date_id,
            max_date_id=max_date_id,
            n_folds=args.n_folds,
            valid_window=args.valid_window,
            gap=args.gap,
            min_train_window=args.train_window,
        )

    rows: list[dict[str, float | int | str]] = []
    for fold in folds:
        fit_data = build_weighted_ridge_fit_data(
            train,
            fold,
            feature_columns=features,
            chunk_days=args.chunk_days,
        )
        for alpha in alphas:
            model = solve_weighted_ridge(fit_data, alpha=alpha)
            rows.append(
                evaluate_ridge(
                    train,
                    fold,
                    model,
                    chunk_days=args.chunk_days,
                )
            )

    results = pl.DataFrame(rows)
    summary_by_alpha = (
        results.group_by("alpha")
        .agg(
            pl.len().alias("folds"),
            pl.col("weighted_zero_mean_r2").mean().alias("mean_r2"),
            pl.col("weighted_zero_mean_r2").std().alias("std_r2"),
            pl.col("weighted_zero_mean_r2").min().alias("min_r2"),
            pl.col("weighted_zero_mean_r2").max().alias("max_r2"),
            (1.0 - pl.col("numerator").sum() / pl.col("denominator").sum()).alias("global_r2"),
            pl.col("numerator").sum().alias("numerator_sum"),
            pl.col("denominator").sum().alias("denominator_sum"),
            pl.col("rows").sum().alias("validation_rows"),
        )
        .sort("global_r2", descending=True)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ridge_sweep.csv"
    summary_csv_path = args.output_dir / "ridge_sweep_by_alpha.csv"
    json_path = args.output_dir / "ridge_sweep_summary.json"
    results.write_csv(csv_path)
    summary_by_alpha.write_csv(summary_csv_path)

    best = summary_by_alpha.row(0, named=True)
    summary = {
        "baseline": "weighted_ridge",
        "fold_type": args.fold_type,
        "n_folds": args.n_folds,
        "train_window": args.train_window if args.fold_type == "rolling" else "expanding",
        "valid_window": args.valid_window,
        "gap": args.gap,
        "alphas": alphas,
        "chunk_days": args.chunk_days,
        "n_features": len(features),
        "best_alpha": float(best["alpha"]),
        "best_mean_r2": float(best["mean_r2"]),
        "best_global_r2": float(best["global_r2"]),
        "csv": str(csv_path),
        "summary_csv": str(summary_csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(results)
    print(summary_by_alpha)
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_csv_path}")
    print(f"Wrote {json_path}")


def _parse_alphas(raw: str) -> list[float]:
    alphas = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not alphas:
        raise ValueError("--alphas must contain at least one value")
    if any(alpha < 0.0 for alpha in alphas):
        raise ValueError("--alphas must be non-negative")
    return alphas


if __name__ == "__main__":
    main()
