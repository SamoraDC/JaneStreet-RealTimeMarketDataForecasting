"""Run causal clipping/shrinkage experiments for Ridge predictions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from janestreet.calibration import (
    ShrinkageCalibrator,
    add_abs_prediction_bucket,
    fit_abs_prediction_thresholds,
    fit_shrinkage_calibrator,
)
from janestreet.cross_sectional import (
    make_random_projection_spec,
    with_cross_sectional_random_projections,
)
from janestreet.diagnostics import aggregate_weighted_r2_by_slice, combine_slice_aggregates
from janestreet.folds import DateFold, make_rolling_folds
from janestreet.linear import (
    build_weighted_ridge_fit_data,
    feature_columns_from_schema,
    solve_weighted_ridge,
)
from janestreet.paths import TRAIN_PARQUET_DIR
from janestreet.paths import FEATURES_CSV
from janestreet.tag_features import load_feature_tag_spec, with_feature_tag_market_state


@dataclass(frozen=True)
class Strategy:
    name: str
    calibrator: ShrinkageCalibrator | None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--valid-window", type=int, default=60)
    parser.add_argument("--calibration-window", type=int, default=30)
    parser.add_argument("--inner-oof-folds", type=int, default=0)
    parser.add_argument("--inner-valid-window", type=int, default=20)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1000.0)
    parser.add_argument("--chunk-days", type=int, default=10)
    parser.add_argument("--time-bucket-size", type=int, default=100)
    parser.add_argument("--min-group-rows", type=int, default=2_000)
    parser.add_argument("--clip-target-abs-quantile", type=float, default=0.999)
    parser.add_argument("--xs-random-projections", type=int, default=0)
    parser.add_argument("--xs-seed", type=int, default=17)
    parser.add_argument("--tag-factors", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/ridge_calibration"))
    args = parser.parse_args()

    if args.calibration_window <= 0:
        raise ValueError("--calibration-window must be positive")
    if args.calibration_window >= args.train_window:
        raise ValueError("--calibration-window must be smaller than --train-window")
    if args.inner_oof_folds < 0:
        raise ValueError("--inner-oof-folds must be non-negative")
    if args.inner_valid_window <= 0:
        raise ValueError("--inner-valid-window must be positive")

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    schema = train.collect_schema()
    features = feature_columns_from_schema(schema)
    xs_feature_count = 0
    if args.xs_random_projections > 0:
        xs_spec = make_random_projection_spec(
            features,
            n_projections=args.xs_random_projections,
            seed=args.xs_seed,
        )
        train = with_cross_sectional_random_projections(train, xs_spec)
        features = features + xs_spec.output_columns
        xs_feature_count = len(xs_spec.output_columns)
    tag_feature_count = 0
    if args.tag_factors:
        tag_spec = load_feature_tag_spec(FEATURES_CSV, features)
        train = with_feature_tag_market_state(train, tag_spec)
        features = features + tag_spec.output_columns
        tag_feature_count = len(tag_spec.output_columns)
    folds = _make_folds(train, args)

    result_rows: list[dict[str, float | int | str]] = []
    alpha_rows: list[dict[str, float | int | str | None]] = []
    slice_partials = {
        "time_bucket": [],
        "weight_bucket": [],
        "missing_bucket": [],
        "date_id_symbol_id": [],
    }

    for fold in folds:
        if args.inner_oof_folds > 0:
            calibration_frame = _collect_inner_oof_predictions(train, features, fold, args)
        else:
            calibration_fold = _calibration_fold(fold, args.calibration_window)
            inner_fit = build_weighted_ridge_fit_data(
                train,
                calibration_fold,
                feature_columns=features,
                chunk_days=args.chunk_days,
            )
            inner_model = solve_weighted_ridge(inner_fit, alpha=args.alpha)
            calibration_frame = _collect_prediction_frame(
                train,
                features,
                inner_model,
                calibration_fold.valid_start,
                calibration_fold.valid_end,
                chunk_days=args.chunk_days,
                time_bucket_size=args.time_bucket_size,
            )
        clip_abs = _target_abs_quantile(calibration_frame, args.clip_target_abs_quantile)
        weight_thresholds = _weight_thresholds(calibration_frame)
        pred_abs_thresholds = fit_abs_prediction_thresholds(calibration_frame)
        calibration_frame = _add_regime_columns(calibration_frame, weight_thresholds, pred_abs_thresholds)
        strategies = _fit_strategies(
            calibration_frame,
            clip_abs=clip_abs,
            min_group_rows=args.min_group_rows,
        )
        alpha_rows.extend(_strategy_parameter_rows(fold, strategies, clip_abs))

        full_fit = build_weighted_ridge_fit_data(
            train,
            fold,
            feature_columns=features,
            chunk_days=args.chunk_days,
        )
        full_model = solve_weighted_ridge(full_fit, alpha=args.alpha)
        validation_frame = _collect_prediction_frame(
            train,
            features,
            full_model,
            fold.valid_start,
            fold.valid_end,
            chunk_days=args.chunk_days,
            time_bucket_size=args.time_bucket_size,
        )
        validation_frame = _add_regime_columns(validation_frame, weight_thresholds, pred_abs_thresholds)

        for strategy in strategies:
            scored = _apply_strategy(validation_frame, strategy)
            result_rows.append(
                {
                    **_fold_metadata(fold),
                    "strategy": strategy.name,
                    "alpha": args.alpha,
                    "clip_abs": clip_abs if strategy.name != "raw" else None,
                    **_score_frame(scored, prediction="strategy_prediction"),
                }
            )
            slice_partials["time_bucket"].append(
                _slice_with_metadata(scored, ["time_bucket"], fold, strategy.name)
            )
            slice_partials["weight_bucket"].append(
                _slice_with_metadata(scored, ["weight_bucket"], fold, strategy.name)
            )
            slice_partials["missing_bucket"].append(
                _slice_with_metadata(scored, ["missing_bucket"], fold, strategy.name)
            )
            slice_partials["date_id_symbol_id"].append(
                _slice_with_metadata(scored, ["date_id", "symbol_id"], fold, strategy.name).head(20)
            )

    results = pl.DataFrame(result_rows)
    summary = _summary_by_strategy(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(args.output_dir / "ridge_calibration_by_fold.csv")
    summary.write_csv(args.output_dir / "ridge_calibration_summary.csv")
    pl.DataFrame(alpha_rows).write_csv(args.output_dir / "ridge_calibration_parameters.csv")
    _write_slice_outputs(slice_partials, args.output_dir)

    best = summary.row(0, named=True)
    report = {
        "experiment": "ridge_calibration",
        "hypothesis": "Causal clipping and regime shrinkage can reduce Ridge amplitude failures without post-hoc slice removal.",
        "n_folds": args.n_folds,
        "train_window": args.train_window,
        "valid_window": args.valid_window,
        "calibration_window": args.calibration_window,
        "inner_oof_folds": args.inner_oof_folds,
        "inner_valid_window": args.inner_valid_window,
        "alpha": args.alpha,
        "chunk_days": args.chunk_days,
        "time_bucket_size": args.time_bucket_size,
        "min_group_rows": args.min_group_rows,
        "clip_target_abs_quantile": args.clip_target_abs_quantile,
        "xs_random_projections": args.xs_random_projections,
        "xs_seed": args.xs_seed,
        "tag_factors": args.tag_factors,
        "n_features": len(features),
        "xs_feature_count": xs_feature_count,
        "tag_feature_count": tag_feature_count,
        "best_strategy": best,
    }
    (args.output_dir / "ridge_calibration_report.json").write_text(
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
    inner_train_end = calibration_start - 1
    if inner_train_end < fold.train_start:
        raise ValueError(f"{fold.name}: calibration window leaves no inner training data")
    return DateFold(
        name=f"{fold.name}_inner",
        train_start=fold.train_start,
        train_end=inner_train_end,
        valid_start=calibration_start,
        valid_end=fold.train_end,
    )


def _collect_inner_oof_predictions(
    train: pl.LazyFrame,
    features: tuple[str, ...],
    fold: DateFold,
    args: argparse.Namespace,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for inner_fold in _inner_oof_folds(fold, args.inner_oof_folds, args.inner_valid_window):
        inner_fit = build_weighted_ridge_fit_data(
            train,
            inner_fold,
            feature_columns=features,
            chunk_days=args.chunk_days,
        )
        inner_model = solve_weighted_ridge(inner_fit, alpha=args.alpha)
        frames.append(
            _collect_prediction_frame(
                train,
                features,
                inner_model,
                inner_fold.valid_start,
                inner_fold.valid_end,
                chunk_days=args.chunk_days,
                time_bucket_size=args.time_bucket_size,
            )
        )
    return pl.concat(frames)


def _inner_oof_folds(fold: DateFold, n_folds: int, valid_window: int) -> list[DateFold]:
    first_valid_start = fold.train_end - n_folds * valid_window + 1
    if first_valid_start <= fold.train_start:
        raise ValueError(f"{fold.name}: not enough train days for requested inner OOF folds")
    folds: list[DateFold] = []
    for idx in range(n_folds):
        valid_start = first_valid_start + idx * valid_window
        valid_end = valid_start + valid_window - 1
        folds.append(
            DateFold(
                name=f"{fold.name}_oof_{idx + 1:02d}",
                train_start=fold.train_start,
                train_end=valid_start - 1,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )
    return folds


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
        missing_expr = pl.sum_horizontal(
            *[pl.col(name).is_null().cast(pl.UInt16) for name in features]
        ).alias("missing_count")
        frame = (
            data.filter(pl.col("date_id").is_between(chunk_start, chunk_end))
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
        if frame.height == 0:
            continue
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
    if not frames:
        raise ValueError(f"no rows found for date range {start}-{end}")
    return pl.concat(frames)


def _add_regime_columns(
    frame: pl.DataFrame,
    weight_thresholds: dict[str, float],
    pred_abs_thresholds: dict[str, float],
) -> pl.DataFrame:
    with_weight = frame.with_columns(
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
    )
    return add_abs_prediction_bucket(with_weight, pred_abs_thresholds)


def _fit_strategies(
    calibration_frame: pl.DataFrame,
    *,
    clip_abs: float,
    min_group_rows: int,
) -> list[Strategy]:
    specs: list[tuple[str, tuple[str, ...], int, float, float]] = [
        ("clip_only", (), 1, 1.0, 1.0),
        ("global", (), 1, 0.0, 1.0),
        ("time", ("time_bucket",), min_group_rows, 0.0, 1.0),
        ("time_weight", ("time_bucket", "weight_bucket"), min_group_rows, 0.0, 1.0),
        ("time_missing", ("time_bucket", "missing_bucket"), min_group_rows, 0.0, 1.0),
        ("time_predabs", ("time_bucket", "prediction_abs_bucket"), min_group_rows, 0.0, 1.0),
        ("weight_predabs", ("weight_bucket", "prediction_abs_bucket"), min_group_rows, 0.0, 1.0),
        ("weight_missing", ("weight_bucket", "missing_bucket"), min_group_rows, 0.0, 1.0),
        (
            "time_weight_predabs",
            ("time_bucket", "weight_bucket", "prediction_abs_bucket"),
            min_group_rows,
            0.0,
            1.0,
        ),
        (
            "time_weight_missing",
            ("time_bucket", "weight_bucket", "missing_bucket"),
            min_group_rows,
            0.0,
            1.0,
        ),
        ("symbol_time", ("symbol_id", "time_bucket"), min_group_rows, 0.0, 1.0),
    ]
    strategies = [Strategy("raw", None)]
    for name, groups, min_rows, alpha_min, alpha_max in specs:
        strategies.append(
            Strategy(
                name,
                fit_shrinkage_calibrator(
                    calibration_frame,
                    name=name,
                    group_columns=groups,
                    min_group_rows=min_rows,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max,
                    clip_abs=clip_abs,
                ),
            )
        )
    return strategies


def _apply_strategy(frame: pl.DataFrame, strategy: Strategy) -> pl.DataFrame:
    if strategy.calibrator is None:
        return frame.with_columns(pl.col("prediction").alias("strategy_prediction"))
    return strategy.calibrator.apply(frame, output="strategy_prediction")


def _score_frame(frame: pl.DataFrame, *, prediction: str) -> dict[str, float | int]:
    row = frame.select(
        [
            pl.len().alias("rows"),
            pl.col("weight").sum().alias("weight_sum"),
            (pl.col("weight") * (pl.col("responder_6") - pl.col(prediction)).pow(2)).sum().alias(
                "numerator"
            ),
            (pl.col("weight") * pl.col("responder_6").pow(2)).sum().alias("denominator"),
            pl.col(prediction).mean().alias("prediction_mean"),
            pl.col(prediction).std().alias("prediction_std"),
        ]
    ).row(0, named=True)
    denominator = float(row["denominator"])
    return {
        "rows": int(row["rows"]),
        "weight_sum": float(row["weight_sum"]),
        "numerator": float(row["numerator"]),
        "denominator": denominator,
        "weighted_zero_mean_r2": 1.0 - float(row["numerator"]) / denominator,
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
            pl.col("numerator").sum().alias("numerator_sum"),
            pl.col("denominator").sum().alias("denominator_sum"),
            pl.col("rows").sum().alias("validation_rows"),
        )
        .sort(["global_r2", "min_r2"], descending=[True, True])
    )


def _slice_with_metadata(
    frame: pl.DataFrame,
    by: Sequence[str],
    fold: DateFold,
    strategy: str,
) -> pl.DataFrame:
    return aggregate_weighted_r2_by_slice(
        frame,
        list(by),
        prediction="strategy_prediction",
    ).with_columns(
        [
            pl.lit(fold.name).alias("fold"),
            pl.lit(strategy).alias("strategy"),
        ]
    )


def _write_slice_outputs(slice_partials: dict[str, list[pl.DataFrame]], output_dir: Path) -> None:
    group_columns = {
        "time_bucket": ["strategy", "time_bucket"],
        "weight_bucket": ["strategy", "weight_bucket"],
        "missing_bucket": ["strategy", "missing_bucket"],
    }
    for name, frames in slice_partials.items():
        if name == "date_id_symbol_id":
            pl.concat(frames).write_csv(output_dir / f"{name}_worst.csv")
            continue
        combined = combine_slice_aggregates(frames, group_columns[name])
        combined.write_csv(output_dir / f"{name}.csv")


def _target_abs_quantile(frame: pl.DataFrame, quantile: float) -> float:
    if quantile <= 0.0 or quantile >= 1.0:
        raise ValueError("--clip-target-abs-quantile must be between 0 and 1")
    value = frame.select(pl.col("responder_6").abs().quantile(quantile)).item()
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError("invalid clip threshold from calibration window")
    return result


def _weight_thresholds(frame: pl.DataFrame) -> dict[str, float]:
    row = frame.select(
        pl.col("weight").quantile(0.50).alias("q50"),
        pl.col("weight").quantile(0.90).alias("q90"),
        pl.col("weight").quantile(0.99).alias("q99"),
    ).row(0, named=True)
    return {name: float(row[name]) for name in ("q50", "q90", "q99")}


def _strategy_parameter_rows(
    fold: DateFold,
    strategies: Sequence[Strategy],
    clip_abs: float,
) -> list[dict[str, float | int | str | None]]:
    rows: list[dict[str, float | int | str | None]] = []
    for strategy in strategies:
        if strategy.calibrator is None:
            rows.append(
                {
                    "fold": fold.name,
                    "strategy": strategy.name,
                    "groups": "",
                    "clip_abs": None,
                    "fallback_alpha": None,
                    "n_parameter_rows": 0,
                    "min_alpha": None,
                    "mean_alpha": None,
                    "max_alpha": None,
                    "zero_alpha_share": None,
                }
            )
            continue
        params = strategy.calibrator.parameters
        alpha = params["_calibration_alpha"]
        rows.append(
            {
                "fold": fold.name,
                "strategy": strategy.name,
                "groups": ",".join(strategy.calibrator.group_columns),
                "clip_abs": clip_abs,
                "fallback_alpha": strategy.calibrator.fallback_alpha,
                "n_parameter_rows": params.height,
                "min_alpha": float(alpha.min()),
                "mean_alpha": float(alpha.mean()),
                "max_alpha": float(alpha.max()),
                "zero_alpha_share": float((alpha == 0.0).mean()),
            }
        )
    return rows


def _fold_metadata(fold: DateFold) -> dict[str, int | str]:
    return {
        "fold": fold.name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "valid_start": fold.valid_start,
        "valid_end": fold.valid_end,
        "train_days": fold.train_days,
        "valid_days": fold.valid_days,
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
