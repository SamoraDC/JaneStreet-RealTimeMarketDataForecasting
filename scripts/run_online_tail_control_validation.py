"""Nested walk-forward validation for gateway-observable tail-control."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from janestreet.folds import DateFold
from janestreet.linear import feature_columns_from_schema
from janestreet.paths import PROJECT_ROOT, TRAIN_PARQUET_DIR
from janestreet.tail_control import add_tail_switch_prediction


@dataclass(frozen=True)
class OnlineTailFold:
    """Train/selection/validation split for nested tail-control validation."""

    name: str
    train_start: int
    train_end: int
    selection_start: int
    selection_end: int
    valid_start: int
    valid_end: int

    @property
    def train_days(self) -> int:
        return self.train_end - self.train_start + 1

    @property
    def selection_days(self) -> int:
        return self.selection_end - self.selection_start + 1

    @property
    def valid_days(self) -> int:
        return self.valid_end - self.valid_start + 1

    def selection_model_fold(self) -> DateFold:
        return DateFold(
            name=f"{self.name}_selection",
            train_start=self.train_start,
            train_end=self.train_end,
            valid_start=self.selection_start,
            valid_end=self.selection_end,
        )

    def validation_model_fold(self) -> DateFold:
        return DateFold(
            name=f"{self.name}_validation",
            train_start=self.train_start,
            train_end=self.train_end,
            valid_start=self.valid_start,
            valid_end=self.valid_end,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=2)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--selection-window", type=int, default=20)
    parser.add_argument("--valid-window", type=int, default=20)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--min-date-id", type=int, default=None)
    parser.add_argument("--max-date-id", type=int, default=None)
    parser.add_argument("--inner-oof-folds", type=int, default=3)
    parser.add_argument("--inner-valid-window", type=int, default=20)
    parser.add_argument("--engines", default="xgboost,lightgbm")
    parser.add_argument("--train-sample-frac", type=float, default=0.10)
    parser.add_argument("--gbdt-seeds", default="17,23,37")
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--lightgbm-min-child-samples", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--chunk-days", type=int, default=10)
    parser.add_argument("--time-bucket-size", type=int, default=100)
    parser.add_argument("--clock-name", default="batch_missing")
    parser.add_argument("--clock-bucket-count", type=int, default=20)
    parser.add_argument("--tail-buckets", default="q90_q99,q99_q100")
    parser.add_argument("--selection-min-delta", type=float, default=0.0)
    parser.add_argument("--n-operational-source-features", type=int, default=32)
    parser.add_argument("--operational-windows", default="16,64")
    parser.add_argument("--min-group-rows", type=int, default=2_000)
    parser.add_argument("--clip-target-abs-quantile", type=float, default=0.999)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/time_geometry/reports/online_tail_control_walk_forward"),
    )
    args = parser.parse_args()
    _validate_args(args)

    clock = _load_clock_tournament_module()
    runner = clock._load_tree_ensemble_runner()
    engines = clock._parse_engines(args.engines)
    tail_buckets = _parse_tail_buckets(args.tail_buckets)
    windows = clock._parse_windows(args.operational_windows)

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    schema = train.collect_schema()
    base_features = feature_columns_from_schema(schema)
    source_features = base_features[: args.n_operational_source_features]
    model_features = tuple(dict.fromkeys([*base_features, "time_id", "symbol_id"]))
    bounds = train.select(
        [
            pl.min("date_id").alias("min_date_id"),
            pl.max("date_id").alias("max_date_id"),
            pl.max("time_id").alias("max_time_id"),
        ]
    ).collect()
    dataset_min_date_id = int(bounds["min_date_id"][0])
    dataset_max_date_id = int(bounds["max_date_id"][0])
    fold_min_date_id = dataset_min_date_id if args.min_date_id is None else args.min_date_id
    fold_max_date_id = dataset_max_date_id if args.max_date_id is None else args.max_date_id
    _validate_date_bounds(
        dataset_min_date_id=dataset_min_date_id,
        dataset_max_date_id=dataset_max_date_id,
        fold_min_date_id=fold_min_date_id,
        fold_max_date_id=fold_max_date_id,
    )
    max_time_id = int(bounds["max_time_id"][0])
    folds = make_online_tail_folds(
        min_date_id=fold_min_date_id,
        max_date_id=fold_max_date_id,
        n_folds=args.n_folds,
        train_window=args.train_window,
        selection_window=args.selection_window,
        valid_window=args.valid_window,
        gap=args.gap,
    )
    clock_data = clock._with_clock_inputs(
        train,
        source_features=source_features,
        windows=windows,
        max_time_id=max_time_id,
    )
    prediction_columns = ("ridge_calibrated_prediction",) + tuple(f"{engine}_prediction" for engine in engines)

    score_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    observability_rows: list[dict[str, Any]] = []
    weight_bucket_rows: list[pl.DataFrame] = []
    simplex_parameter_rows: list[pl.DataFrame] = []

    for fold in folds:
        calibration = clock._collect_base_predictions(
            runner=runner,
            train=train,
            base_features=base_features,
            model_features=model_features,
            engines=engines,
            fold=fold.selection_model_fold(),
            args=args,
            inner=True,
        )
        calibration = clock._add_base_ensemble(
            calibration,
            runner=runner,
            engines=engines,
            args=args,
            fold=fold.selection_model_fold(),
        )
        selection = _collect_scored_period(
            clock=clock,
            runner=runner,
            train=train,
            base_features=base_features,
            model_features=model_features,
            engines=engines,
            args=args,
            model_fold=fold.selection_model_fold(),
            calibration=calibration,
        )
        validation = _collect_scored_period(
            clock=clock,
            runner=runner,
            train=train,
            base_features=base_features,
            model_features=model_features,
            engines=engines,
            args=args,
            model_fold=fold.validation_model_fold(),
            calibration=calibration,
        )

        calibration = clock._join_clock_values(clock_data, calibration, fold.train_start, fold.train_end)
        selection = clock._join_clock_values(clock_data, selection, fold.selection_start, fold.selection_end)
        validation = clock._join_clock_values(clock_data, validation, fold.valid_start, fold.valid_end)

        for phase, frame, start, end in (
            ("selection", selection, fold.selection_start, fold.selection_end),
            ("validation", validation, fold.valid_start, fold.valid_end),
        ):
            observability_rows.append(
                {
                    **_fold_metadata(fold),
                    "phase": phase,
                    **audit_batch_missing_observability(
                        train,
                        frame,
                        source_features=source_features,
                        start=start,
                        end=end,
                    ),
                }
            )

        binner = clock.fit_clock_binner(
            calibration,
            clock_name=args.clock_name,
            bucket_count=args.clock_bucket_count,
            max_time_id=max_time_id,
        )
        calibration = clock.apply_clock_binner(calibration, binner)
        selection = clock.apply_clock_binner(selection, binner)
        validation = clock.apply_clock_binner(validation, binner)
        grouped_simplex = clock.fit_grouped_simplex_weights(
            calibration,
            group_columns=("clock_bucket",),
            prediction_columns=prediction_columns,
            min_group_rows=args.min_group_rows,
        )
        simplex_parameter_rows.append(
            grouped_simplex.parameters.with_columns(
                [
                    pl.lit(fold.name).alias("fold"),
                    pl.lit(args.clock_name).alias("clock"),
                ]
            )
        )
        selection = grouped_simplex.apply(selection, output="clock_simplex_prediction")
        validation = grouped_simplex.apply(validation, output="clock_simplex_prediction")
        selection = add_tail_switch_prediction(
            selection,
            base_prediction="ensemble_prediction",
            candidate_prediction="clock_simplex_prediction",
            tail_buckets=tail_buckets,
            output="tail_frozen_prediction",
        )
        validation = add_tail_switch_prediction(
            validation,
            base_prediction="ensemble_prediction",
            candidate_prediction="clock_simplex_prediction",
            tail_buckets=tail_buckets,
            output="tail_frozen_prediction",
        )

        selection_base = _score(clock, selection, "ensemble_prediction")
        selection_tail = _score(clock, selection, "tail_frozen_prediction")
        selection_delta = selection_tail["weighted_zero_mean_r2"] - selection_base["weighted_zero_mean_r2"]
        selection_passed = selection_delta > args.selection_min_delta
        validation = add_selection_gated_prediction(
            validation,
            selection_passed=selection_passed,
            base_prediction="ensemble_prediction",
            tail_prediction="tail_frozen_prediction",
            output="tail_selection_gated_prediction",
        )

        for phase, frame in (("selection", selection), ("validation", validation)):
            strategy_predictions = {
                "base_ensemble": "ensemble_prediction",
                f"{args.clock_name}_clock_simplex": "clock_simplex_prediction",
                f"{args.clock_name}_tail_frozen_{_tail_suffix(tail_buckets)}": "tail_frozen_prediction",
            }
            if phase == "validation":
                strategy_predictions[
                    f"{args.clock_name}_tail_selection_gated_{_tail_suffix(tail_buckets)}"
                ] = "tail_selection_gated_prediction"
            for strategy, prediction in strategy_predictions.items():
                score_rows.append(
                    {
                        **_fold_metadata(fold),
                        "phase": phase,
                        "strategy": strategy,
                        **_score(clock, frame, prediction),
                    }
                )
                if phase == "validation":
                    weight_bucket_rows.append(
                        clock._weight_bucket_slice(frame, prediction=prediction)
                        .with_columns(
                            [
                                pl.lit(fold.name).alias("fold"),
                                pl.lit(strategy).alias("strategy"),
                            ]
                        )
                        .select(
                            [
                                "fold",
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

        policy_rows.append(
            {
                **_fold_metadata(fold),
                "clock": args.clock_name,
                "clock_bucket_count": args.clock_bucket_count,
                "tail_buckets": ",".join(tail_buckets),
                "selection_base_r2": selection_base["weighted_zero_mean_r2"],
                "selection_tail_r2": selection_tail["weighted_zero_mean_r2"],
                "selection_delta_r2": selection_delta,
                "selection_min_delta": args.selection_min_delta,
                "selection_passed": selection_passed,
                "simplex_parameter_rows": grouped_simplex.parameters.height,
                "simplex_fallback_weights": json.dumps(grouped_simplex.fallback_weights, sort_keys=True),
                **_tail_activation_audit(validation, tail_buckets=tail_buckets),
            }
        )
        del calibration, selection, validation
        gc.collect()

    results = pl.DataFrame(score_rows)
    validation_summary = _summary(results.filter(pl.col("phase") == "validation"))
    selection_summary = _summary(results.filter(pl.col("phase") == "selection"))
    weight_bucket_by_fold = pl.concat(weight_bucket_rows) if weight_bucket_rows else pl.DataFrame()
    weight_bucket_summary = _combine_weight_bucket_slices(weight_bucket_by_fold) if weight_bucket_rows else pl.DataFrame()
    simplex_parameters = pl.concat(simplex_parameter_rows, how="diagonal") if simplex_parameter_rows else pl.DataFrame()
    policy_frame = pl.DataFrame(policy_rows)
    observability_frame = pl.DataFrame(observability_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(args.output_dir / "online_tail_control_by_fold.csv")
    validation_summary.write_csv(args.output_dir / "online_tail_control_validation_summary.csv")
    selection_summary.write_csv(args.output_dir / "online_tail_control_selection_summary.csv")
    policy_frame.write_csv(args.output_dir / "online_tail_control_policy_by_fold.csv")
    observability_frame.write_csv(args.output_dir / "gateway_observability_by_fold.csv")
    simplex_parameters.write_csv(args.output_dir / "clock_simplex_parameters.csv")
    if not weight_bucket_by_fold.is_empty():
        weight_bucket_by_fold.write_csv(args.output_dir / "online_tail_control_weight_bucket_by_fold.csv")
        weight_bucket_summary.write_csv(args.output_dir / "online_tail_control_weight_bucket_summary.csv")

    report = {
        "experiment": "online_tail_control_walk_forward",
        "hypothesis": "A frozen high-weight tail-control rule remains useful when chosen before the external validation window.",
        "folds": [_fold_metadata(fold) for fold in folds],
        "train_window": args.train_window,
        "selection_window": args.selection_window,
        "valid_window": args.valid_window,
        "dataset_date_bounds": {
            "min_date_id": dataset_min_date_id,
            "max_date_id": dataset_max_date_id,
        },
        "fold_date_bounds": {
            "min_date_id": fold_min_date_id,
            "max_date_id": fold_max_date_id,
        },
        "engines": engines,
        "train_sample_frac": args.train_sample_frac,
        "gbdt_seeds": args.gbdt_seeds,
        "max_iter": args.max_iter,
        "clock": args.clock_name,
        "tail_buckets": tail_buckets,
        "causality_contract": {
            "model_training_dates": "strictly before selection and validation",
            "tail_rule_family": "frozen before this script; selection window gates only the nested strategy",
            "clock_binner_and_simplex": "fit on inner OOF predictions from the train window",
            "batch_missing_frac": "computed from the current (date_id,time_id) gateway batch only",
            "validation_targets_used_for_fitting": False,
            "official_lags_used": False,
            "official_lags_note": "The active baseline excludes responder lag features after prior negative validation.",
        },
        "anti_leakage_audit": {
            "train_selection_validation_disjoint": all(_fold_is_ordered(fold) for fold in folds),
            "selection_used_to_score_validation": False,
            "selection_used_to_gate_nested_strategy": True,
            "validation_used_to_choose_tail_rule": False,
            "gateway_observability_max_abs_diff": (
                float(observability_frame["max_abs_diff"].max()) if not observability_frame.is_empty() else None
            ),
        },
        "best_validation_strategy": validation_summary.row(0, named=True) if not validation_summary.is_empty() else None,
    }
    (args.output_dir / "online_tail_control_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(validation_summary)
    print(policy_frame)
    print(f"Wrote {args.output_dir}")


def make_online_tail_folds(
    *,
    min_date_id: int,
    max_date_id: int,
    n_folds: int,
    train_window: int,
    selection_window: int,
    valid_window: int,
    gap: int = 0,
) -> list[OnlineTailFold]:
    _require_positive("n_folds", n_folds)
    _require_positive("train_window", train_window)
    _require_positive("selection_window", selection_window)
    _require_positive("valid_window", valid_window)
    _require_non_negative("gap", gap)
    if min_date_id > max_date_id:
        raise ValueError("min_date_id must be <= max_date_id")
    first_valid_start = max_date_id - n_folds * valid_window + 1
    first_train_start = first_valid_start - gap - selection_window - gap - train_window
    if first_train_start < min_date_id:
        raise ValueError("not enough dates for requested online tail folds")
    folds: list[OnlineTailFold] = []
    for idx in range(n_folds):
        valid_start = first_valid_start + idx * valid_window
        valid_end = valid_start + valid_window - 1
        selection_end = valid_start - gap - 1
        selection_start = selection_end - selection_window + 1
        train_end = selection_start - gap - 1
        train_start = train_end - train_window + 1
        folds.append(
            OnlineTailFold(
                name=f"otf_{idx + 1:02d}",
                train_start=train_start,
                train_end=train_end,
                selection_start=selection_start,
                selection_end=selection_end,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )
    return folds


def add_selection_gated_prediction(
    frame: pl.DataFrame,
    *,
    selection_passed: bool,
    base_prediction: str,
    tail_prediction: str,
    output: str,
) -> pl.DataFrame:
    selected = tail_prediction if selection_passed else base_prediction
    return frame.with_columns(pl.col(selected).alias(output))


def audit_batch_missing_observability(
    train: pl.LazyFrame,
    validation: pl.DataFrame,
    *,
    source_features: tuple[str, ...],
    start: int,
    end: int,
) -> dict[str, float | int]:
    if not source_features:
        raise ValueError("source_features must not be empty")
    source_count = float(len(source_features))
    current_missing = sum(pl.col(name).is_null().cast(pl.Float32) for name in source_features) / source_count
    gateway_batches = (
        train.filter(pl.col("date_id").is_between(start, end))
        .select(
            [
                pl.col("date_id").cast(pl.Int32),
                pl.col("time_id").cast(pl.Int32),
                current_missing.alias("_row_missing_frac"),
            ]
        )
        .group_by(["date_id", "time_id"])
        .agg(
            pl.col("_row_missing_frac").mean().cast(pl.Float32).alias("gateway_batch_missing_frac"),
            pl.len().alias("batch_rows"),
        )
        .collect()
    )
    tournament_batches = validation.select(["date_id", "time_id", "batch_missing_frac"]).unique()
    comparison = gateway_batches.join(tournament_batches, on=["date_id", "time_id"], how="inner").with_columns(
        (pl.col("gateway_batch_missing_frac") - pl.col("batch_missing_frac")).abs().alias("abs_diff")
    )
    row = comparison.select(
        [
            pl.len().alias("batches"),
            pl.col("batch_rows").min().alias("batch_rows_min"),
            pl.col("batch_rows").max().alias("batch_rows_max"),
            pl.col("abs_diff").max().alias("max_abs_diff"),
            pl.col("abs_diff").mean().alias("mean_abs_diff"),
        ]
    ).row(0, named=True)
    return {
        "batches": int(row["batches"]),
        "batch_rows_min": int(row["batch_rows_min"]),
        "batch_rows_max": int(row["batch_rows_max"]),
        "max_abs_diff": float(row["max_abs_diff"]),
        "mean_abs_diff": float(row["mean_abs_diff"]),
    }


def _collect_scored_period(
    *,
    clock: Any,
    runner: Any,
    train: pl.LazyFrame,
    base_features: tuple[str, ...],
    model_features: tuple[str, ...],
    engines: tuple[str, ...],
    args: argparse.Namespace,
    model_fold: DateFold,
    calibration: pl.DataFrame,
) -> pl.DataFrame:
    frame = clock._collect_base_predictions(
        runner=runner,
        train=train,
        base_features=base_features,
        model_features=model_features,
        engines=engines,
        fold=model_fold,
        args=args,
        inner=False,
    )
    return clock._add_base_ensemble(
        frame,
        runner=runner,
        engines=engines,
        args=args,
        fold=model_fold,
        calibration=calibration,
    )


def _score(clock: Any, frame: pl.DataFrame, prediction: str) -> dict[str, float | int]:
    return clock._score(calibration=frame, prediction=prediction)


def _summary(results: pl.DataFrame) -> pl.DataFrame:
    if results.is_empty():
        return pl.DataFrame()
    return (
        results.group_by("strategy")
        .agg(
            [
                pl.len().alias("folds"),
                pl.col("weighted_zero_mean_r2").mean().alias("mean_r2"),
                pl.col("weighted_zero_mean_r2").std().alias("std_r2"),
                pl.col("weighted_zero_mean_r2").min().alias("min_r2"),
                pl.col("weighted_zero_mean_r2").max().alias("max_r2"),
                (1.0 - pl.col("numerator").sum() / pl.col("denominator").sum()).alias("global_r2"),
                pl.col("rows").sum().alias("validation_rows"),
            ]
        )
        .sort(["global_r2", "min_r2"], descending=[True, True])
    )


def _combine_weight_bucket_slices(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame()
    return (
        frame.group_by(["strategy", "weight_bucket"])
        .agg(
            [
                pl.col("rows").sum(),
                pl.col("weight_sum").sum(),
                pl.col("numerator").sum(),
                pl.col("denominator").sum(),
            ]
        )
        .with_columns((1.0 - pl.col("numerator") / pl.col("denominator")).alias("weighted_zero_mean_r2"))
        .sort(["strategy", "weight_bucket"])
    )


def _tail_activation_audit(frame: pl.DataFrame, *, tail_buckets: tuple[str, ...]) -> dict[str, int]:
    tail = frame.filter(pl.col("weight_bucket").is_in(list(tail_buckets)))
    return {
        "validation_rows": frame.height,
        "validation_tail_rows": tail.height,
        "validation_non_tail_rows": frame.height - tail.height,
        "validation_tail_batches": tail.select(["date_id", "time_id"]).unique().height,
    }


def _fold_metadata(fold: OnlineTailFold) -> dict[str, int | str]:
    return {
        "fold": fold.name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "selection_start": fold.selection_start,
        "selection_end": fold.selection_end,
        "valid_start": fold.valid_start,
        "valid_end": fold.valid_end,
    }


def _fold_is_ordered(fold: OnlineTailFold) -> bool:
    return fold.train_start <= fold.train_end < fold.selection_start <= fold.selection_end < fold.valid_start <= fold.valid_end


def _tail_suffix(tail_buckets: tuple[str, ...]) -> str:
    return "_".join(bucket.replace("q", "q") for bucket in tail_buckets)


def _parse_tail_buckets(raw: str) -> tuple[str, ...]:
    buckets = tuple(part.strip() for part in raw.split(",") if part.strip())
    allowed = {"q00_q50", "q50_q90", "q90_q99", "q99_q100"}
    unknown = set(buckets) - allowed
    if unknown:
        raise ValueError(f"unknown tail buckets: {', '.join(sorted(unknown))}")
    if not buckets:
        raise ValueError("--tail-buckets must not be empty")
    return buckets


def _load_clock_tournament_module() -> Any:
    path = PROJECT_ROOT / "scripts" / "run_clock_tournament.py"
    spec = importlib.util.spec_from_file_location("run_clock_tournament", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("n_folds", "train_window", "selection_window", "valid_window", "inner_oof_folds", "inner_valid_window"):
        _require_positive(name, getattr(args, name))
    _require_non_negative("gap", args.gap)
    if not 0.0 < args.train_sample_frac <= 1.0:
        raise ValueError("--train-sample-frac must be in (0, 1]")
    for name in ("max_iter", "max_leaf_nodes", "n_jobs", "chunk_days", "clock_bucket_count", "min_group_rows"):
        _require_positive(name, getattr(args, name))
    if args.clock_bucket_count <= 1:
        raise ValueError("--clock-bucket-count must be greater than 1")
    if args.n_operational_source_features <= 0:
        raise ValueError("--n-operational-source-features must be positive")


def _validate_date_bounds(
    *,
    dataset_min_date_id: int,
    dataset_max_date_id: int,
    fold_min_date_id: int,
    fold_max_date_id: int,
) -> None:
    if fold_min_date_id < dataset_min_date_id:
        raise ValueError("--min-date-id is before the dataset start")
    if fold_max_date_id > dataset_max_date_id:
        raise ValueError("--max-date-id is after the dataset end")
    if fold_min_date_id > fold_max_date_id:
        raise ValueError("--min-date-id must be <= --max-date-id")


def _require_positive(name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


if __name__ == "__main__":
    main()
