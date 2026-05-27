"""Tournament for causal clock choices as ensemble auxiliary layers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import polars as pl

from janestreet.blending import add_simplex_blend_prediction, fit_simplex_blend_weights
from janestreet.calibration import add_abs_prediction_bucket, fit_abs_prediction_thresholds, fit_shrinkage_calibrator
from janestreet.folds import DateFold, make_rolling_folds
from janestreet.linear import feature_columns_from_schema
from janestreet.paths import PROJECT_ROOT, TRAIN_PARQUET_DIR
from janestreet.tail_control import add_tail_switch_prediction, fit_grouped_tail_advantage_policy
from janestreet.time_geometry import OperationalTimeSpec, with_operational_time_features


KEY_COLUMNS = ("date_id", "time_id", "symbol_id")
CLOCK_SPECS: dict[str, tuple[str, str]] = {
    "clock_time": ("time_id", "fixed_time"),
    "row_activity": ("ot_source_activity", "quantile"),
    "symbol_activity_ewm": ("ot_source_activity_ewm_64", "quantile"),
    "batch_activity": ("batch_activity_mean", "quantile"),
    "batch_weight": ("batch_weight_sum", "quantile"),
    "symbol_weight_cum": ("ot_symbol_weight_cum", "quantile"),
    "batch_missing": ("batch_missing_frac", "quantile"),
}


@dataclass(frozen=True)
class ClockBinner:
    name: str
    source_column: str
    mode: str
    bucket_count: int
    thresholds: tuple[float, ...] = ()
    max_time_id: int = 967


@dataclass(frozen=True)
class GroupedSimplexWeights:
    group_columns: tuple[str, ...]
    prediction_columns: tuple[str, ...]
    parameters: pl.DataFrame
    fallback_weights: dict[str, float]

    def apply(self, frame: pl.DataFrame, *, output: str) -> pl.DataFrame:
        if not self.group_columns or self.parameters.height == 0:
            expression = sum(
                float(self.fallback_weights[column]) * pl.col(column)
                for column in self.prediction_columns
            )
            return frame.with_columns(expression.alias(output))
        joined = frame.join(self.parameters, on=list(self.group_columns), how="left")
        expression = sum(
            pl.coalesce(pl.col(_simplex_weight_column(column)), pl.lit(float(self.fallback_weights[column])))
            * pl.col(column)
            for column in self.prediction_columns
        )
        drop_columns = [
            column
            for column in ["_simplex_rows", *(_simplex_weight_column(column) for column in self.prediction_columns)]
            if column in joined.columns
        ]
        return joined.with_columns(expression.alias(output)).drop(drop_columns)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--valid-window", type=int, default=20)
    parser.add_argument("--inner-oof-folds", type=int, default=2)
    parser.add_argument("--inner-valid-window", type=int, default=20)
    parser.add_argument("--engines", default="xgboost,lightgbm")
    parser.add_argument("--train-sample-frac", type=float, default=0.05)
    parser.add_argument("--gbdt-seeds", default="17")
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--lightgbm-min-child-samples", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--chunk-days", type=int, default=10)
    parser.add_argument("--time-bucket-size", type=int, default=100)
    parser.add_argument("--clock-bucket-count", type=int, default=20)
    parser.add_argument("--clock-candidates", default=",".join(CLOCK_SPECS))
    parser.add_argument("--n-operational-source-features", type=int, default=32)
    parser.add_argument("--operational-windows", default="16,64")
    parser.add_argument("--min-group-rows", type=int, default=2_000)
    parser.add_argument("--clip-target-abs-quantile", type=float, default=0.999)
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/time_geometry/reports/clock_tournament_smoke"))
    args = parser.parse_args()
    _validate_args(args)

    engines = _parse_engines(args.engines)
    clock_candidates = _parse_clock_candidates(args.clock_candidates)
    runner = _load_tree_ensemble_runner()

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    schema = train.collect_schema()
    base_features = feature_columns_from_schema(schema)
    source_features = base_features[: args.n_operational_source_features]
    model_features = tuple(dict.fromkeys([*base_features, "time_id", "symbol_id"]))
    bounds = train.select(
        pl.min("date_id").alias("min_date_id"),
        pl.max("date_id").alias("max_date_id"),
        pl.max("time_id").alias("max_time_id"),
    ).collect()
    max_time_id = int(bounds["max_time_id"][0])
    clock_data = _with_clock_inputs(
        train,
        source_features=source_features,
        windows=_parse_windows(args.operational_windows),
        max_time_id=max_time_id,
    )
    folds = make_rolling_folds(
        min_date_id=int(bounds["min_date_id"][0]),
        max_date_id=int(bounds["max_date_id"][0]),
        n_folds=args.n_folds,
        train_window=args.train_window,
        valid_window=args.valid_window,
        gap=0,
    )

    rows: list[dict[str, float | int | str | None]] = []
    parameter_rows: list[dict[str, float | int | str | None]] = []
    weight_bucket_rows: list[pl.DataFrame] = []
    prediction_columns = ("ridge_calibrated_prediction",) + tuple(f"{engine}_prediction" for engine in engines)
    for fold in folds:
        calibration = _collect_base_predictions(
            runner=runner,
            train=train,
            base_features=base_features,
            model_features=model_features,
            engines=engines,
            fold=fold,
            args=args,
            inner=True,
        )
        calibration = _add_base_ensemble(calibration, runner=runner, engines=engines, args=args, fold=fold)
        validation = _collect_base_predictions(
            runner=runner,
            train=train,
            base_features=base_features,
            model_features=model_features,
            engines=engines,
            fold=fold,
            args=args,
            inner=False,
        )
        validation = _add_base_ensemble(
            validation,
            runner=runner,
            engines=engines,
            args=args,
            fold=fold,
            calibration=calibration,
        )
        calibration = _join_clock_values(clock_data, calibration, fold.train_start, fold.train_end)
        validation = _join_clock_values(clock_data, validation, fold.valid_start, fold.valid_end)
        ensemble_abs_thresholds = fit_abs_prediction_thresholds(calibration, prediction="ensemble_prediction")
        calibration = add_abs_prediction_bucket(
            calibration,
            ensemble_abs_thresholds,
            prediction="ensemble_prediction",
            output="ensemble_abs_bucket",
        )
        validation = add_abs_prediction_bucket(
            validation,
            ensemble_abs_thresholds,
            prediction="ensemble_prediction",
            output="ensemble_abs_bucket",
        )

        _append_strategy_result(
            rows=rows,
            weight_bucket_rows=weight_bucket_rows,
            frame=validation,
            fold=fold,
            clock="none",
            strategy="base_ensemble",
            prediction="ensemble_prediction",
        )
        global_calibrator = fit_shrinkage_calibrator(
            calibration,
            name="global_ensemble_shrink",
            group_columns=(),
            prediction="ensemble_prediction",
            min_group_rows=args.min_group_rows,
            clip_abs=None,
        )
        validation_global = global_calibrator.apply(
            validation,
            prediction="ensemble_prediction",
            output="global_shrink_prediction",
        )
        _append_strategy_result(
            rows=rows,
            weight_bucket_rows=weight_bucket_rows,
            frame=validation_global,
            fold=fold,
            clock="none",
            strategy="global_shrink",
            prediction="global_shrink_prediction",
        )
        parameter_rows.append(
            _shrink_parameter_summary(
                fold=fold,
                clock="none",
                strategy="global_shrink",
                calibrator=global_calibrator,
            )
        )

        for clock_name in clock_candidates:
            binner = fit_clock_binner(
                calibration,
                clock_name=clock_name,
                bucket_count=args.clock_bucket_count,
                max_time_id=max_time_id,
            )
            calibration_clock = apply_clock_binner(calibration, binner)
            validation_clock = apply_clock_binner(validation, binner)
            _evaluate_clock_strategies(
                rows=rows,
                parameter_rows=parameter_rows,
                weight_bucket_rows=weight_bucket_rows,
                calibration=calibration_clock,
                validation=validation_clock,
                fold=fold,
                clock_name=clock_name,
                prediction_columns=prediction_columns,
                min_group_rows=args.min_group_rows,
            )

    results = pl.DataFrame(rows)
    summary = _summary(results)
    parameter_frame = pl.DataFrame(parameter_rows)
    weight_bucket_by_fold = pl.concat(weight_bucket_rows)
    weight_bucket_summary = _combine_weight_bucket_slices(weight_bucket_by_fold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(args.output_dir / "clock_tournament_by_fold.csv")
    summary.write_csv(args.output_dir / "clock_tournament_summary.csv")
    parameter_frame.write_csv(args.output_dir / "clock_tournament_parameters.csv")
    weight_bucket_by_fold.write_csv(args.output_dir / "clock_tournament_weight_bucket_by_fold.csv")
    weight_bucket_summary.write_csv(args.output_dir / "clock_tournament_weight_bucket_summary.csv")
    report = {
        "experiment": "clock_tournament",
        "hypothesis": "Different causal clocks should be selected by downstream auxiliary role, not by descriptive balance alone.",
        "n_folds": args.n_folds,
        "train_window": args.train_window,
        "valid_window": args.valid_window,
        "inner_oof_folds": args.inner_oof_folds,
        "inner_valid_window": args.inner_valid_window,
        "engines": engines,
        "train_sample_frac": args.train_sample_frac,
        "clock_candidates": clock_candidates,
        "clock_bucket_count": args.clock_bucket_count,
        "min_group_rows": args.min_group_rows,
        "best_strategy": summary.row(0, named=True),
    }
    (args.output_dir / "clock_tournament_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(results)
    print(summary)
    print(f"Wrote {args.output_dir}")


def _collect_base_predictions(
    *,
    runner: object,
    train: pl.LazyFrame,
    base_features: tuple[str, ...],
    model_features: tuple[str, ...],
    engines: tuple[str, ...],
    fold: DateFold,
    args: argparse.Namespace,
    inner: bool,
) -> pl.DataFrame:
    return runner._collect_engine_predictions(
        runner=runner._load_blend_runner(),
        train=train,
        ridge_features=base_features,
        model_features=model_features,
        engines=engines,
        fold=fold,
        args=args,
        inner=inner,
    )


def _add_base_ensemble(
    frame: pl.DataFrame,
    *,
    runner: object,
    engines: tuple[str, ...],
    args: argparse.Namespace,
    fold: DateFold,
    calibration: pl.DataFrame | None = None,
) -> pl.DataFrame:
    fit_frame = frame if calibration is None else calibration
    clip_abs = runner._load_blend_runner()._target_abs_quantile(fit_frame, args.clip_target_abs_quantile)
    weight_thresholds = runner._load_blend_runner()._weight_thresholds(fit_frame)
    pred_abs_thresholds = fit_abs_prediction_thresholds(fit_frame, prediction="ridge_prediction")
    fit_with_regime = runner._add_regime_columns(
        fit_frame,
        weight_thresholds,
        pred_abs_thresholds,
        time_bucket_size=args.time_bucket_size,
    )
    ridge_calibrator = fit_shrinkage_calibrator(
        fit_with_regime,
        name=f"{fold.name}_ridge_weight_predabs",
        group_columns=["weight_bucket", "prediction_abs_bucket"],
        prediction="ridge_prediction",
        min_group_rows=args.min_group_rows,
        clip_abs=clip_abs,
    )
    fit_with_regime = ridge_calibrator.apply(
        fit_with_regime,
        prediction="ridge_prediction",
        output="ridge_calibrated_prediction",
    )
    prediction_columns = ("ridge_calibrated_prediction",) + tuple(f"{engine}_prediction" for engine in engines)
    simplex_weights = fit_simplex_blend_weights(fit_with_regime, prediction_columns=prediction_columns)

    scored = runner._add_regime_columns(
        frame,
        weight_thresholds,
        pred_abs_thresholds,
        time_bucket_size=args.time_bucket_size,
    )
    scored = ridge_calibrator.apply(
        scored,
        prediction="ridge_prediction",
        output="ridge_calibrated_prediction",
    )
    return add_simplex_blend_prediction(scored, weights=simplex_weights, output="ensemble_prediction")


def _with_clock_inputs(
    data: pl.LazyFrame,
    *,
    source_features: tuple[str, ...],
    windows: tuple[int, ...],
    max_time_id: int,
) -> pl.LazyFrame:
    spec = OperationalTimeSpec(source_columns=source_features, windows=windows, max_time_id=max_time_id)
    with_ot = with_operational_time_features(data, spec)
    groups = ["date_id", "time_id"]
    return with_ot.with_columns(
        [
            pl.col("weight").sum().over(groups).cast(pl.Float32).alias("batch_weight_sum"),
            pl.col("ot_source_activity").mean().over(groups).cast(pl.Float32).alias("batch_activity_mean"),
            pl.col("ot_missing_frac").mean().over(groups).cast(pl.Float32).alias("batch_missing_frac"),
        ]
    )


def _join_clock_values(clock_data: pl.LazyFrame, frame: pl.DataFrame, start: int, end: int) -> pl.DataFrame:
    raw_columns = tuple(dict.fromkeys(source for source, _ in CLOCK_SPECS.values()))
    clock_values = (
        clock_data.filter(pl.col("date_id").is_between(start, end))
        .select(
            [pl.col(name).cast(pl.Int32 if name != "symbol_id" else pl.Int16) for name in KEY_COLUMNS]
            + [pl.col(name).fill_null(0.0).cast(pl.Float32).alias(name) for name in raw_columns if name not in KEY_COLUMNS]
        )
        .collect()
    )
    joined = frame.join(clock_values, on=list(KEY_COLUMNS), how="left")
    has_missing = joined.select(pl.any_horizontal([pl.col(name).is_null() for name in raw_columns]).any()).item()
    if has_missing:
        raise ValueError("clock join produced missing clock values")
    return joined


def fit_clock_binner(
    frame: pl.DataFrame,
    *,
    clock_name: str,
    bucket_count: int,
    max_time_id: int,
) -> ClockBinner:
    source_column, mode = CLOCK_SPECS[clock_name]
    if mode == "fixed_time":
        return ClockBinner(clock_name, source_column, mode, bucket_count=bucket_count, max_time_id=max_time_id)
    quantiles = tuple(idx / bucket_count for idx in range(1, bucket_count))
    row = frame.select(
        [
            pl.col(source_column).quantile(quantile, interpolation="linear").alias(f"q{idx}")
            for idx, quantile in enumerate(quantiles)
        ]
    ).row(0, named=True)
    thresholds = tuple(float(row[f"q{idx}"]) for idx in range(len(quantiles)))
    return ClockBinner(clock_name, source_column, mode, bucket_count=bucket_count, thresholds=thresholds, max_time_id=max_time_id)


def apply_clock_binner(frame: pl.DataFrame, binner: ClockBinner, *, output: str = "clock_bucket") -> pl.DataFrame:
    if binner.mode == "fixed_time":
        bucket_size = max((binner.max_time_id + 1) // binner.bucket_count, 1)
        expression = (pl.col(binner.source_column) // bucket_size).clip(0, binner.bucket_count - 1)
    else:
        expression = pl.lit(0, dtype=pl.Int16)
        for threshold in binner.thresholds:
            expression = expression + (pl.col(binner.source_column) > threshold).cast(pl.Int16)
    return frame.with_columns(expression.cast(pl.Int16).alias(output))


def _evaluate_clock_strategies(
    *,
    rows: list[dict[str, float | int | str | None]],
    parameter_rows: list[dict[str, float | int | str | None]],
    weight_bucket_rows: list[pl.DataFrame],
    calibration: pl.DataFrame,
    validation: pl.DataFrame,
    fold: DateFold,
    clock_name: str,
    prediction_columns: tuple[str, ...],
    min_group_rows: int,
) -> None:
    for strategy, group_columns in {
        "clock_shrink": ("clock_bucket",),
        "clock_weight_shrink": ("clock_bucket", "weight_bucket"),
        "clock_predabs_shrink": ("clock_bucket", "ensemble_abs_bucket"),
    }.items():
        calibrator = fit_shrinkage_calibrator(
            calibration,
            name=f"{clock_name}_{strategy}",
            group_columns=group_columns,
            prediction="ensemble_prediction",
            min_group_rows=min_group_rows,
            clip_abs=None,
        )
        scored = calibrator.apply(
            validation,
            prediction="ensemble_prediction",
            output=f"{strategy}_prediction",
        )
        _append_strategy_result(
            rows=rows,
            weight_bucket_rows=weight_bucket_rows,
            frame=scored,
            fold=fold,
            clock=clock_name,
            strategy=strategy,
            prediction=f"{strategy}_prediction",
        )
        parameter_rows.append(
            _shrink_parameter_summary(
                fold=fold,
                clock=clock_name,
                strategy=strategy,
                calibrator=calibrator,
            )
        )
    grouped_simplex = fit_grouped_simplex_weights(
        calibration,
        group_columns=("clock_bucket",),
        prediction_columns=prediction_columns,
        min_group_rows=min_group_rows,
    )
    simplex_calibration = grouped_simplex.apply(calibration, output="clock_simplex_prediction")
    simplex_scored = grouped_simplex.apply(validation, output="clock_simplex_prediction")
    _append_strategy_result(
        rows=rows,
        weight_bucket_rows=weight_bucket_rows,
        frame=simplex_scored,
        fold=fold,
        clock=clock_name,
        strategy="clock_simplex",
        prediction="clock_simplex_prediction",
    )
    parameter_rows.append(
        {
            **_fold_metadata(fold),
            "clock": clock_name,
            "strategy": "clock_simplex",
            "group_columns": "clock_bucket",
            "fallback_alpha": None,
            "parameter_rows": grouped_simplex.parameters.height,
            "fallback_weights": json.dumps(grouped_simplex.fallback_weights, sort_keys=True),
        }
    )
    for strategy, tail_buckets in {
        "clock_simplex_tail_q99_q100": ("q99_q100",),
        "clock_simplex_tail_q90_q100": ("q90_q99", "q99_q100"),
    }.items():
        output = f"{strategy}_prediction"
        tail_scored = add_tail_switch_prediction(
            simplex_scored,
            base_prediction="ensemble_prediction",
            candidate_prediction="clock_simplex_prediction",
            tail_buckets=tail_buckets,
            output=output,
        )
        _append_strategy_result(
            rows=rows,
            weight_bucket_rows=weight_bucket_rows,
            frame=tail_scored,
            fold=fold,
            clock=clock_name,
            strategy=strategy,
            prediction=output,
        )
        parameter_rows.append(
            {
                **_fold_metadata(fold),
                "clock": clock_name,
                "strategy": strategy,
                "group_columns": "clock_bucket",
                "fallback_alpha": None,
                "parameter_rows": grouped_simplex.parameters.height,
                "fallback_weights": json.dumps(grouped_simplex.fallback_weights, sort_keys=True),
                "tail_buckets": ",".join(tail_buckets),
            }
        )
    tail_advantage_policy = fit_grouped_tail_advantage_policy(
        simplex_calibration,
        group_columns=("clock_bucket",),
        base_prediction="ensemble_prediction",
        candidate_prediction="clock_simplex_prediction",
        tail_buckets=("q90_q99", "q99_q100"),
        min_group_rows=min_group_rows,
        output="clock_simplex_tail_advantage_prediction",
    )
    tail_advantage_scored = tail_advantage_policy.apply(simplex_scored)
    _append_strategy_result(
        rows=rows,
        weight_bucket_rows=weight_bucket_rows,
        frame=tail_advantage_scored,
        fold=fold,
        clock=clock_name,
        strategy="clock_simplex_tail_advantage_q90_q100",
        prediction="clock_simplex_tail_advantage_prediction",
    )
    parameter_rows.append(
        {
            **_fold_metadata(fold),
            "clock": clock_name,
            "strategy": "clock_simplex_tail_advantage_q90_q100",
            "group_columns": ",".join(tail_advantage_policy.group_columns),
            "fallback_alpha": None,
            "parameter_rows": tail_advantage_policy.parameters.height,
            "fallback_weights": None,
            "tail_buckets": ",".join(tail_advantage_policy.tail_buckets),
            "fallback_use_candidate": tail_advantage_policy.fallback_use_candidate,
        }
    )


def fit_grouped_simplex_weights(
    frame: pl.DataFrame,
    *,
    group_columns: Sequence[str],
    prediction_columns: Sequence[str],
    min_group_rows: int,
) -> GroupedSimplexWeights:
    if min_group_rows <= 0:
        raise ValueError("min_group_rows must be positive")
    groups = tuple(group_columns)
    columns = tuple(prediction_columns)
    fallback_weights = fit_simplex_blend_weights(frame, prediction_columns=columns)
    if not groups:
        return GroupedSimplexWeights(groups, columns, pl.DataFrame(), fallback_weights)
    counts = frame.group_by(list(groups)).agg(pl.len().alias("_simplex_rows"))
    rows: list[dict[str, float | int | str]] = []
    for group_row in counts.iter_rows(named=True):
        row_count = int(group_row["_simplex_rows"])
        if row_count < min_group_rows:
            continue
        group_filter = pl.all_horizontal([pl.col(column) == group_row[column] for column in groups])
        group_frame = frame.filter(group_filter)
        weights = fit_simplex_blend_weights(group_frame, prediction_columns=columns)
        rows.append(
            {
                **{column: group_row[column] for column in groups},
                "_simplex_rows": row_count,
                **{_simplex_weight_column(column): weight for column, weight in weights.items()},
            }
        )
    return GroupedSimplexWeights(groups, columns, pl.DataFrame(rows), fallback_weights)


def _append_strategy_result(
    *,
    rows: list[dict[str, float | int | str | None]],
    weight_bucket_rows: list[pl.DataFrame],
    frame: pl.DataFrame,
    fold: DateFold,
    clock: str,
    strategy: str,
    prediction: str,
) -> None:
    rows.append(
        {
            **_fold_metadata(fold),
            "clock": clock,
            "strategy": strategy,
            **_score(calibration=frame, prediction=prediction),
        }
    )
    weight_bucket_rows.append(
        _weight_bucket_slice(frame, prediction=prediction)
        .with_columns(
            [
                pl.lit(fold.name).alias("fold"),
                pl.lit(clock).alias("clock"),
                pl.lit(strategy).alias("strategy"),
            ]
        )
        .select(
            [
                "fold",
                "clock",
                "strategy",
                "weight_bucket",
                "rows",
                "weight_sum",
                "numerator",
                "denominator",
                "weighted_zero_mean_r2",
            ]
        )
    )


def _score(*, calibration: pl.DataFrame, prediction: str) -> dict[str, float | int]:
    row = calibration.select(
        [
            pl.len().alias("rows"),
            (pl.col("weight") * (pl.col("responder_6") - pl.col(prediction)).pow(2)).sum().alias("numerator"),
            (pl.col("weight") * pl.col("responder_6").pow(2)).sum().alias("denominator"),
            pl.col(prediction).mean().alias("prediction_mean"),
            pl.col(prediction).std().alias("prediction_std"),
        ]
    ).row(0, named=True)
    denominator = float(row["denominator"])
    if denominator <= 0.0:
        raise ValueError("non-positive denominator")
    return {
        "rows": int(row["rows"]),
        "numerator": float(row["numerator"]),
        "denominator": denominator,
        "weighted_zero_mean_r2": 1.0 - float(row["numerator"]) / denominator,
        "prediction_mean": float(row["prediction_mean"]),
        "prediction_std": float(row["prediction_std"]),
    }


def _weight_bucket_slice(frame: pl.DataFrame, *, prediction: str) -> pl.DataFrame:
    return (
        frame.group_by("weight_bucket")
        .agg(
            pl.len().alias("rows"),
            pl.col("weight").sum().alias("weight_sum"),
            (pl.col("weight") * (pl.col("responder_6") - pl.col(prediction)).pow(2)).sum().alias("numerator"),
            (pl.col("weight") * pl.col("responder_6").pow(2)).sum().alias("denominator"),
        )
        .with_columns((1.0 - pl.col("numerator") / pl.col("denominator")).alias("weighted_zero_mean_r2"))
        .sort("weight_bucket")
    )


def _combine_weight_bucket_slices(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.group_by(["clock", "strategy", "weight_bucket"])
        .agg(
            pl.col("rows").sum(),
            pl.col("weight_sum").sum(),
            pl.col("numerator").sum(),
            pl.col("denominator").sum(),
        )
        .with_columns((1.0 - pl.col("numerator") / pl.col("denominator")).alias("weighted_zero_mean_r2"))
        .sort(["clock", "strategy", "weight_bucket"])
    )


def _summary(results: pl.DataFrame) -> pl.DataFrame:
    return (
        results.group_by(["clock", "strategy"])
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


def _shrink_parameter_summary(
    *,
    fold: DateFold,
    clock: str,
    strategy: str,
    calibrator: object,
) -> dict[str, float | int | str | None]:
    return {
        **_fold_metadata(fold),
        "clock": clock,
        "strategy": strategy,
        "group_columns": ",".join(calibrator.group_columns),
        "fallback_alpha": float(calibrator.fallback_alpha),
        "parameter_rows": int(calibrator.parameters.height),
        "fallback_weights": None,
    }


def _fold_metadata(fold: DateFold) -> dict[str, int | str]:
    return {
        "fold": fold.name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "valid_start": fold.valid_start,
        "valid_end": fold.valid_end,
    }


def _parse_engines(raw: str) -> tuple[str, ...]:
    engines = tuple(part.strip() for part in raw.split(",") if part.strip())
    allowed = {"xgboost", "lightgbm"}
    unknown = set(engines) - allowed
    if unknown:
        raise ValueError(f"unknown engines for clock tournament: {', '.join(sorted(unknown))}")
    if not engines:
        raise ValueError("--engines must not be empty")
    return engines


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not seeds:
        raise ValueError("--gbdt-seeds must contain at least one integer")
    return seeds


def _parse_clock_candidates(raw: str) -> tuple[str, ...]:
    clocks = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    unknown = set(clocks) - set(CLOCK_SPECS)
    if unknown:
        raise ValueError(f"unknown clock candidates: {', '.join(sorted(unknown))}")
    if not clocks:
        raise ValueError("--clock-candidates must not be empty")
    return clocks


def _parse_windows(raw: str) -> tuple[int, ...]:
    windows = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not windows or any(window <= 0 for window in windows):
        raise ValueError("--operational-windows must contain positive integers")
    return windows


def _simplex_weight_column(prediction_column: str) -> str:
    return f"_simplex_weight_{prediction_column}"


def _load_tree_ensemble_runner() -> object:
    path = PROJECT_ROOT / "scripts" / "run_tree_engine_ensemble.py"
    spec = importlib.util.spec_from_file_location("run_tree_engine_ensemble", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_folds <= 0 or args.train_window <= 0 or args.valid_window <= 0:
        raise ValueError("fold counts and windows must be positive")
    if args.inner_oof_folds <= 0 or args.inner_valid_window <= 0:
        raise ValueError("inner OOF settings must be positive")
    if not 0.0 < args.train_sample_frac <= 1.0:
        raise ValueError("--train-sample-frac must be in (0, 1]")
    if args.max_iter <= 0 or args.max_leaf_nodes <= 0:
        raise ValueError("tree size settings must be positive")
    if args.n_jobs <= 0 or args.chunk_days <= 0:
        raise ValueError("resource settings must be positive")
    if args.clock_bucket_count <= 1:
        raise ValueError("--clock-bucket-count must be greater than 1")
    if args.n_operational_source_features <= 0:
        raise ValueError("--n-operational-source-features must be positive")
    if args.min_group_rows <= 0:
        raise ValueError("--min-group-rows must be positive")


if __name__ == "__main__":
    main()
