"""Validate the frozen high-weight tail-control policy on an explicit holdout."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import polars as pl

from janestreet.folds import DateFold
from janestreet.linear import feature_columns_from_schema
from janestreet.paths import PROJECT_ROOT, TRAIN_PARQUET_DIR
from janestreet.tail_control import add_tail_switch_prediction, fit_grouped_tail_advantage_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold-name", default="pre_tail_holdout")
    parser.add_argument("--train-start", type=int, default=1219)
    parser.add_argument("--train-end", type=int, default=1338)
    parser.add_argument("--valid-start", type=int, default=1339)
    parser.add_argument("--valid-end", type=int, default=1398)
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
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--chunk-days", type=int, default=10)
    parser.add_argument("--time-bucket-size", type=int, default=100)
    parser.add_argument("--clock-bucket-count", type=int, default=20)
    parser.add_argument("--n-operational-source-features", type=int, default=32)
    parser.add_argument("--operational-windows", default="16,64")
    parser.add_argument("--min-group-rows", type=int, default=2_000)
    parser.add_argument("--clip-target-abs-quantile", type=float, default=0.999)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/time_geometry/reports/frozen_tail_holdout_pre_stage3"),
    )
    args = parser.parse_args()
    _validate_args(args)

    clock = _load_clock_tournament_module()
    runner = clock._load_tree_ensemble_runner()
    engines = clock._parse_engines(args.engines)
    windows = clock._parse_windows(args.operational_windows)
    fold = DateFold(
        name=args.fold_name,
        train_start=args.train_start,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
    )
    args.n_folds = 1
    args.train_window = fold.train_days
    args.valid_window = fold.valid_days

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    schema = train.collect_schema()
    base_features = feature_columns_from_schema(schema)
    source_features = base_features[: args.n_operational_source_features]
    model_features = tuple(dict.fromkeys([*base_features, "time_id", "symbol_id"]))
    max_time_id = int(train.select(pl.max("time_id")).collect().item())
    clock_data = clock._with_clock_inputs(
        train,
        source_features=source_features,
        windows=windows,
        max_time_id=max_time_id,
    )
    prediction_columns = ("ridge_calibrated_prediction",) + tuple(f"{engine}_prediction" for engine in engines)

    calibration = clock._collect_base_predictions(
        runner=runner,
        train=train,
        base_features=base_features,
        model_features=model_features,
        engines=engines,
        fold=fold,
        args=args,
        inner=True,
    )
    calibration = clock._add_base_ensemble(calibration, runner=runner, engines=engines, args=args, fold=fold)
    calibration = clock._join_clock_values(clock_data, calibration, fold.train_start, fold.train_end)
    binner = clock.fit_clock_binner(
        calibration,
        clock_name="batch_missing",
        bucket_count=args.clock_bucket_count,
        max_time_id=max_time_id,
    )
    calibration = clock.apply_clock_binner(calibration, binner)
    grouped_simplex = clock.fit_grouped_simplex_weights(
        calibration,
        group_columns=("clock_bucket",),
        prediction_columns=prediction_columns,
        min_group_rows=args.min_group_rows,
    )
    calibration = grouped_simplex.apply(calibration, output="clock_simplex_prediction")
    tail_advantage_policy = fit_grouped_tail_advantage_policy(
        calibration,
        group_columns=("clock_bucket",),
        base_prediction="ensemble_prediction",
        candidate_prediction="clock_simplex_prediction",
        tail_buckets=("q90_q99", "q99_q100"),
        min_group_rows=args.min_group_rows,
        output="tail_advantage_prediction",
    )

    validation = clock._collect_base_predictions(
        runner=runner,
        train=train,
        base_features=base_features,
        model_features=model_features,
        engines=engines,
        fold=fold,
        args=args,
        inner=False,
    )
    validation = clock._add_base_ensemble(
        validation,
        runner=runner,
        engines=engines,
        args=args,
        fold=fold,
        calibration=calibration,
    )
    del calibration
    gc.collect()
    validation = clock._join_clock_values(clock_data, validation, fold.valid_start, fold.valid_end)
    observability = audit_batch_missing_observability(
        train,
        validation,
        source_features=source_features,
        start=fold.valid_start,
        end=fold.valid_end,
    )
    validation = clock.apply_clock_binner(validation, binner)
    validation = grouped_simplex.apply(validation, output="clock_simplex_prediction")
    validation = add_tail_switch_prediction(
        validation,
        base_prediction="ensemble_prediction",
        candidate_prediction="clock_simplex_prediction",
        tail_buckets=("q99_q100",),
        output="tail_q99_prediction",
    )
    validation = add_tail_switch_prediction(
        validation,
        base_prediction="ensemble_prediction",
        candidate_prediction="clock_simplex_prediction",
        tail_buckets=("q90_q99", "q99_q100"),
        output="tail_q90_q100_prediction",
    )
    validation = tail_advantage_policy.apply(validation)
    tail_advantage_equivalence = audit_tail_advantage_equivalence(
        validation,
        policy=tail_advantage_policy,
        fixed_prediction="tail_q90_q100_prediction",
        advantage_prediction="tail_advantage_prediction",
    )

    strategy_predictions = {
        "base_ensemble": "ensemble_prediction",
        "batch_missing_clock_simplex": "clock_simplex_prediction",
        "batch_missing_tail_q99_q100": "tail_q99_prediction",
        "batch_missing_tail_q90_q100": "tail_q90_q100_prediction",
        "batch_missing_tail_advantage_q90_q100": "tail_advantage_prediction",
    }
    summary = pl.DataFrame(
        [
            {
                **_fold_metadata(fold),
                "strategy": strategy,
                **clock._score(calibration=validation, prediction=prediction),
            }
            for strategy, prediction in strategy_predictions.items()
        ]
    ).sort(["weighted_zero_mean_r2", "strategy"], descending=[True, False])
    weight_slices = pl.concat(
        [
            clock._weight_bucket_slice(validation, prediction=prediction).with_columns(
                pl.lit(strategy).alias("strategy")
            )
            for strategy, prediction in strategy_predictions.items()
        ]
    ).select(["strategy", "weight_bucket", "rows", "weight_sum", "numerator", "denominator", "weighted_zero_mean_r2"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.output_dir / "frozen_tail_holdout_summary.csv")
    weight_slices.write_csv(args.output_dir / "frozen_tail_holdout_weight_bucket.csv")
    tail_advantage_policy.parameters.write_csv(args.output_dir / "frozen_tail_advantage_parameters.csv")
    (args.output_dir / "gateway_observability.json").write_text(
        json.dumps(observability, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "experiment": "frozen_tail_holdout",
        "fold": _fold_metadata(fold),
        "engines": engines,
        "train_sample_frac": args.train_sample_frac,
        "gbdt_seeds": args.gbdt_seeds,
        "max_iter": args.max_iter,
        "frozen_policy": {
            "clock": "batch_missing",
            "candidate_strategy": "clock_simplex",
            "tail_buckets": ["q90_q99", "q99_q100"],
        },
        "tail_advantage_policy": summarize_tail_advantage_policy(tail_advantage_policy),
        "tail_advantage_equivalence": tail_advantage_equivalence,
        "observability": observability,
        "best_strategy": summary.row(0, named=True),
    }
    (args.output_dir / "frozen_tail_holdout_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary)
    print(observability)
    print(f"Wrote {args.output_dir}")


def audit_batch_missing_observability(
    train: pl.LazyFrame,
    validation: pl.DataFrame,
    *,
    source_features: tuple[str, ...],
    start: int,
    end: int,
) -> dict[str, float | int]:
    """Check that `batch_missing_frac` is computable from each gateway batch.

    The Kaggle API serves all rows for one `(date_id, time_id)` together, so the
    batch mean of current-row feature missingness is observable without peeking
    at future batches or targets.
    """

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
        pl.len().alias("batches"),
        pl.col("batch_rows").min().alias("batch_rows_min"),
        pl.col("batch_rows").max().alias("batch_rows_max"),
        pl.col("abs_diff").max().alias("max_abs_diff"),
        pl.col("abs_diff").mean().alias("mean_abs_diff"),
    ).row(0, named=True)
    return {
        "batches": int(row["batches"]),
        "batch_rows_min": int(row["batch_rows_min"]),
        "batch_rows_max": int(row["batch_rows_max"]),
        "max_abs_diff": float(row["max_abs_diff"]),
        "mean_abs_diff": float(row["mean_abs_diff"]),
    }


def summarize_tail_advantage_policy(policy: Any) -> dict[str, Any]:
    parameters = policy.parameters
    summary: dict[str, Any] = {
        "group_columns": list(policy.group_columns),
        "fallback_use_candidate": bool(policy.fallback_use_candidate),
        "parameter_rows": int(parameters.height),
        "use_candidate_parameter_rows": 0,
        "use_base_parameter_rows": 0,
        "calibration_tail_rows": 0,
        "all_parameter_rows_use_candidate": False,
    }
    if parameters.height == 0:
        return summary
    row = parameters.select(
        [
            pl.col("_tail_use_candidate").cast(pl.Int64).sum().alias("use_candidate_parameter_rows"),
            (1 - pl.col("_tail_use_candidate").cast(pl.Int64)).sum().alias("use_base_parameter_rows"),
            pl.col("_tail_rows").sum().alias("calibration_tail_rows"),
        ]
    ).row(0, named=True)
    summary.update(
        {
            "use_candidate_parameter_rows": int(row["use_candidate_parameter_rows"]),
            "use_base_parameter_rows": int(row["use_base_parameter_rows"]),
            "calibration_tail_rows": int(row["calibration_tail_rows"]),
            "all_parameter_rows_use_candidate": int(row["use_base_parameter_rows"]) == 0,
        }
    )
    return summary


def audit_tail_advantage_equivalence(
    validation: pl.DataFrame,
    *,
    policy: Any,
    fixed_prediction: str,
    advantage_prediction: str,
) -> dict[str, Any]:
    compared = validation.select(
        [
            "weight_bucket",
            (pl.col(advantage_prediction) - pl.col(fixed_prediction)).abs().alias("_abs_diff"),
        ]
    )
    row = compared.select(
        [
            pl.len().alias("rows"),
            (pl.col("_abs_diff") > 0.0).cast(pl.Int64).sum().alias("differing_rows"),
            pl.col("_abs_diff").max().alias("max_abs_diff"),
        ]
    ).row(0, named=True)
    tail = validation.filter(pl.col("weight_bucket").is_in(list(policy.tail_buckets)))
    result: dict[str, Any] = {
        "rows": int(row["rows"]),
        "tail_rows": int(tail.height),
        "differing_rows": int(row["differing_rows"]),
        "max_abs_diff": float(row["max_abs_diff"] or 0.0),
        "fixed_and_advantage_identical": int(row["differing_rows"]) == 0,
    }
    if not policy.group_columns:
        result.update(
            {
                "validation_tail_groups": 1 if tail.height else 0,
                "validation_tail_groups_using_candidate": int(bool(policy.fallback_use_candidate and tail.height)),
                "validation_tail_groups_using_base": int(bool((not policy.fallback_use_candidate) and tail.height)),
                "validation_tail_groups_using_fallback": int(bool(tail.height)),
            }
        )
        return result
    group_columns = list(policy.group_columns)
    groups = tail.select(group_columns).unique()
    if groups.height == 0:
        result.update(
            {
                "validation_tail_groups": 0,
                "validation_tail_groups_using_candidate": 0,
                "validation_tail_groups_using_base": 0,
                "validation_tail_groups_using_fallback": 0,
            }
        )
        return result
    joined = groups.join(policy.parameters.select(group_columns + ["_tail_use_candidate"]), on=group_columns, how="left")
    joined = joined.with_columns(
        [
            pl.col("_tail_use_candidate").is_null().alias("_uses_fallback"),
            pl.coalesce(pl.col("_tail_use_candidate"), pl.lit(policy.fallback_use_candidate)).alias(
                "_effective_use_candidate"
            ),
        ]
    )
    group_row = joined.select(
        [
            pl.len().alias("validation_tail_groups"),
            pl.col("_effective_use_candidate").cast(pl.Int64).sum().alias("validation_tail_groups_using_candidate"),
            (1 - pl.col("_effective_use_candidate").cast(pl.Int64)).sum().alias("validation_tail_groups_using_base"),
            pl.col("_uses_fallback").cast(pl.Int64).sum().alias("validation_tail_groups_using_fallback"),
        ]
    ).row(0, named=True)
    result.update({key: int(value) for key, value in group_row.items()})
    return result


def _load_clock_tournament_module() -> Any:
    path = PROJECT_ROOT / "scripts" / "run_clock_tournament.py"
    spec = importlib.util.spec_from_file_location("run_clock_tournament", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fold_metadata(fold: DateFold) -> dict[str, int | str]:
    return {
        "fold": fold.name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "valid_start": fold.valid_start,
        "valid_end": fold.valid_end,
    }


def _validate_args(args: argparse.Namespace) -> None:
    if args.train_start > args.train_end:
        raise ValueError("--train-start must be <= --train-end")
    if args.valid_start > args.valid_end:
        raise ValueError("--valid-start must be <= --valid-end")
    if args.train_end >= args.valid_start:
        raise ValueError("train and validation windows must be ordered and disjoint")
    if args.inner_oof_folds <= 0:
        raise ValueError("--inner-oof-folds must be positive")
    if args.inner_valid_window <= 0:
        raise ValueError("--inner-valid-window must be positive")
    if args.n_operational_source_features <= 0:
        raise ValueError("--n-operational-source-features must be positive")


if __name__ == "__main__":
    main()
