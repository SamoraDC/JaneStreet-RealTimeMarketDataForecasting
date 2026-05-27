"""Evaluate cross-sectional context derived from Ridge scores."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from janestreet.calibration import fit_abs_prediction_thresholds, fit_shrinkage_calibrator
from janestreet.diagnostics import aggregate_weighted_r2_by_slice, combine_slice_aggregates
from janestreet.folds import DateFold, make_rolling_folds
from janestreet.linear import (
    build_weighted_ridge_fit_data,
    feature_columns_from_schema,
    solve_weighted_ridge,
)
from janestreet.paths import TRAIN_PARQUET_DIR
from janestreet.score_context import add_prediction_context, fit_prediction_context_combiner


@dataclass(frozen=True)
class StrategyOutput:
    name: str
    frame: pl.DataFrame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--valid-window", type=int, default=60)
    parser.add_argument("--calibration-window", type=int, default=30)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--chunk-days", type=int, default=10)
    parser.add_argument("--time-bucket-size", type=int, default=100)
    parser.add_argument("--min-group-rows", type=int, default=2_000)
    parser.add_argument("--clip-target-abs-quantile", type=float, default=0.999)
    parser.add_argument("--context-alpha", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/ridge_score_context"))
    args = parser.parse_args()

    if args.calibration_window <= 0:
        raise ValueError("--calibration-window must be positive")
    if args.calibration_window >= args.train_window:
        raise ValueError("--calibration-window must be smaller than --train-window")

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    features = feature_columns_from_schema(train.collect_schema())
    folds = _make_folds(train, args)

    rows: list[dict[str, float | int | str | list[float]]] = []
    parameter_rows: list[dict[str, float | int | str]] = []
    slice_partials = {"time_bucket": [], "weight_bucket": [], "missing_bucket": []}

    for fold in folds:
        calibration_fold = _calibration_fold(fold, args.calibration_window)
        inner_model = solve_weighted_ridge(
            build_weighted_ridge_fit_data(
                train,
                calibration_fold,
                feature_columns=features,
                chunk_days=args.chunk_days,
            ),
            alpha=args.alpha,
        )
        calibration = _collect_prediction_frame(
            train,
            features,
            inner_model,
            calibration_fold.valid_start,
            calibration_fold.valid_end,
            chunk_days=args.chunk_days,
            time_bucket_size=args.time_bucket_size,
        )
        clip_abs = _target_abs_quantile(calibration, args.clip_target_abs_quantile)
        weight_thresholds = _weight_thresholds(calibration)
        pred_abs_thresholds = fit_abs_prediction_thresholds(calibration)
        calibration = _add_regime_columns(calibration, weight_thresholds, pred_abs_thresholds)

        base_time_weight = fit_shrinkage_calibrator(
            calibration,
            name="base_time_weight",
            group_columns=["time_bucket", "weight_bucket"],
            min_group_rows=args.min_group_rows,
            clip_abs=clip_abs,
        )
        calibration_context = add_prediction_context(calibration, clip_abs=clip_abs)
        context_model = fit_prediction_context_combiner(
            calibration_context,
            alpha=args.context_alpha,
        )
        calibration_context = context_model.apply(calibration_context)
        context_time_weight = fit_shrinkage_calibrator(
            calibration_context,
            name="score_context_time_weight",
            group_columns=["time_bucket", "weight_bucket"],
            prediction="score_context_prediction",
            min_group_rows=args.min_group_rows,
            clip_abs=clip_abs,
        )
        parameter_rows.append(
            {
                "fold": fold.name,
                "clip_abs": clip_abs,
                "score_market_loo_coef": float(context_model.coefficients[0]),
                "score_deviation_coef": float(context_model.coefficients[1]),
                "base_fallback_alpha": base_time_weight.fallback_alpha,
                "context_fallback_alpha": context_time_weight.fallback_alpha,
            }
        )

        full_model = solve_weighted_ridge(
            build_weighted_ridge_fit_data(
                train,
                fold,
                feature_columns=features,
                chunk_days=args.chunk_days,
            ),
            alpha=args.alpha,
        )
        validation = _collect_prediction_frame(
            train,
            features,
            full_model,
            fold.valid_start,
            fold.valid_end,
            chunk_days=args.chunk_days,
            time_bucket_size=args.time_bucket_size,
        )
        validation = _add_regime_columns(validation, weight_thresholds, pred_abs_thresholds)
        validation_context = context_model.apply(add_prediction_context(validation, clip_abs=clip_abs))

        outputs = [
            StrategyOutput("raw", validation.with_columns(pl.col("prediction").alias("strategy_prediction"))),
            StrategyOutput(
                "base_time_weight",
                base_time_weight.apply(validation, output="strategy_prediction"),
            ),
            StrategyOutput(
                "score_context",
                validation_context.with_columns(pl.col("score_context_prediction").alias("strategy_prediction")),
            ),
            StrategyOutput(
                "score_context_time_weight",
                context_time_weight.apply(
                    validation_context,
                    prediction="score_context_prediction",
                    output="strategy_prediction",
                ),
            ),
        ]
        for output in outputs:
            rows.append(
                {
                    **_fold_metadata(fold),
                    "strategy": output.name,
                    "alpha": args.alpha,
                    "clip_abs": clip_abs if output.name != "raw" else None,
                    "score_market_loo_coef": float(context_model.coefficients[0])
                    if output.name.startswith("score_context")
                    else None,
                    "score_deviation_coef": float(context_model.coefficients[1])
                    if output.name.startswith("score_context")
                    else None,
                    **_score_frame(output.frame),
                }
            )
            for slice_name, by in {
                "time_bucket": ["time_bucket"],
                "weight_bucket": ["weight_bucket"],
                "missing_bucket": ["missing_bucket"],
            }.items():
                slice_partials[slice_name].append(
                    aggregate_weighted_r2_by_slice(output.frame, by, prediction="strategy_prediction")
                    .with_columns(pl.lit(output.name).alias("strategy"))
                )

    results = pl.DataFrame(rows)
    summary = _summary_by_strategy(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(args.output_dir / "score_context_by_fold.csv")
    summary.write_csv(args.output_dir / "score_context_summary.csv")
    pl.DataFrame(parameter_rows).write_csv(args.output_dir / "score_context_parameters.csv")
    _write_slice_outputs(slice_partials, args.output_dir)
    report = {
        "experiment": "ridge_score_context",
        "hypothesis": "Leave-one-out timestamp score context can improve Ridge calibration without feature-level random projections.",
        "n_folds": args.n_folds,
        "train_window": args.train_window,
        "valid_window": args.valid_window,
        "calibration_window": args.calibration_window,
        "alpha": args.alpha,
        "context_alpha": args.context_alpha,
        "best_strategy": summary.row(0, named=True),
    }
    (args.output_dir / "score_context_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(results)
    print(summary)
    print(f"Wrote {args.output_dir}")


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


def _calibration_fold(fold: DateFold, calibration_window: int) -> DateFold:
    calibration_start = fold.train_end - calibration_window + 1
    return DateFold(
        name=f"{fold.name}_inner",
        train_start=fold.train_start,
        train_end=calibration_start - 1,
        valid_start=calibration_start,
        valid_end=fold.train_end,
    )


def _collect_prediction_frame(
    data: pl.LazyFrame,
    features: tuple[str, ...],
    model,
    start: int,
    end: int,
    *,
    chunk_days: int,
    time_bucket_size: int,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for chunk_start, chunk_end in _date_chunks(start, end, chunk_days):
        frame = (
            data.filter(pl.col("date_id").is_between(chunk_start, chunk_end))
            .select(
                [
                    pl.col("date_id").cast(pl.Int32),
                    pl.col("time_id").cast(pl.Int32),
                    pl.col("symbol_id").cast(pl.Int16),
                    pl.col("weight").cast(pl.Float64),
                    pl.col("responder_6").cast(pl.Float64),
                    pl.sum_horizontal(
                        *[pl.col(name).is_null().cast(pl.UInt16) for name in features]
                    ).alias("missing_count"),
                ]
                + [pl.col(name).fill_null(0.0).cast(pl.Float64).alias(name) for name in features]
            )
            .collect()
        )
        predictions = model.predict_array(frame.select(list(features)).to_numpy())
        frames.append(
            frame.select(["date_id", "time_id", "symbol_id", "weight", "responder_6", "missing_count"])
            .with_columns(
                [
                    pl.Series("prediction", predictions),
                    (pl.col("time_id") // time_bucket_size).cast(pl.Int16).alias("time_bucket"),
                ]
            )
        )
    return pl.concat(frames)


def _add_regime_columns(
    frame: pl.DataFrame,
    weight_thresholds: dict[str, float],
    pred_abs_thresholds: dict[str, float],
) -> pl.DataFrame:
    return frame.with_columns(
        pl.when(pl.col("weight") <= weight_thresholds["q50"])
        .then(pl.lit("q00_q50"))
        .when(pl.col("weight") <= weight_thresholds["q90"])
        .then(pl.lit("q50_q90"))
        .when(pl.col("weight") <= weight_thresholds["q99"])
        .then(pl.lit("q90_q99"))
        .otherwise(pl.lit("q99_q100"))
        .alias("weight_bucket"),
        pl.when(pl.col("missing_count") == 0)
        .then(pl.lit("m00"))
        .when(pl.col("missing_count") <= 5)
        .then(pl.lit("m01_m05"))
        .when(pl.col("missing_count") <= 20)
        .then(pl.lit("m06_m20"))
        .otherwise(pl.lit("m21_plus"))
        .alias("missing_bucket"),
        pl.when(pl.col("prediction").abs() <= pred_abs_thresholds["q50"])
        .then(pl.lit("p00_p50"))
        .when(pl.col("prediction").abs() <= pred_abs_thresholds["q90"])
        .then(pl.lit("p50_p90"))
        .when(pl.col("prediction").abs() <= pred_abs_thresholds["q99"])
        .then(pl.lit("p90_p99"))
        .otherwise(pl.lit("p99_p100"))
        .alias("prediction_abs_bucket"),
    )


def _score_frame(frame: pl.DataFrame) -> dict[str, float | int]:
    row = frame.select(
        [
            pl.len().alias("rows"),
            pl.col("weight").sum().alias("weight_sum"),
            (pl.col("weight") * (pl.col("responder_6") - pl.col("strategy_prediction")).pow(2)).sum().alias(
                "numerator"
            ),
            (pl.col("weight") * pl.col("responder_6").pow(2)).sum().alias("denominator"),
            pl.col("strategy_prediction").mean().alias("prediction_mean"),
            pl.col("strategy_prediction").std().alias("prediction_std"),
        ]
    ).row(0, named=True)
    return {
        "rows": int(row["rows"]),
        "weight_sum": float(row["weight_sum"]),
        "numerator": float(row["numerator"]),
        "denominator": float(row["denominator"]),
        "weighted_zero_mean_r2": 1.0 - float(row["numerator"]) / float(row["denominator"]),
        "prediction_mean": float(row["prediction_mean"]),
        "prediction_std": float(row["prediction_std"]),
    }


def _summary_by_strategy(results: pl.DataFrame) -> pl.DataFrame:
    return (
        results.group_by("strategy")
        .agg(
            pl.len().alias("folds"),
            pl.col("weighted_zero_mean_r2").mean().alias("mean_r2"),
            pl.col("weighted_zero_mean_r2").std().alias("std_r2"),
            pl.col("weighted_zero_mean_r2").min().alias("min_r2"),
            pl.col("weighted_zero_mean_r2").max().alias("max_r2"),
            (1.0 - pl.col("numerator").sum() / pl.col("denominator").sum()).alias("global_r2"),
            pl.col("rows").sum().alias("validation_rows"),
        )
        .sort(["global_r2", "min_r2"], descending=[True, True])
    )


def _write_slice_outputs(slice_partials: dict[str, list[pl.DataFrame]], output_dir: Path) -> None:
    group_columns = {
        "time_bucket": ["strategy", "time_bucket"],
        "weight_bucket": ["strategy", "weight_bucket"],
        "missing_bucket": ["strategy", "missing_bucket"],
    }
    for name, frames in slice_partials.items():
        combine_slice_aggregates(frames, group_columns[name]).write_csv(output_dir / f"{name}.csv")


def _target_abs_quantile(frame: pl.DataFrame, quantile: float) -> float:
    value = frame.select(pl.col("responder_6").abs().quantile(quantile)).item()
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("invalid clip threshold")
    return result


def _weight_thresholds(frame: pl.DataFrame) -> dict[str, float]:
    row = frame.select(
        pl.col("weight").quantile(0.50).alias("q50"),
        pl.col("weight").quantile(0.90).alias("q90"),
        pl.col("weight").quantile(0.99).alias("q99"),
    ).row(0, named=True)
    return {name: float(row[name]) for name in ("q50", "q90", "q99")}


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


if __name__ == "__main__":
    main()
