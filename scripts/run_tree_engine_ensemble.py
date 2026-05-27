"""OOF-trained simplex ensemble across tree engines."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import polars as pl

from janestreet.blending import add_simplex_blend_prediction, fit_simplex_blend_weights
from janestreet.calibration import add_abs_prediction_bucket, fit_abs_prediction_thresholds, fit_shrinkage_calibrator
from janestreet.diagnostics import aggregate_weighted_r2_by_slice, combine_slice_aggregates
from janestreet.folds import DateFold
from janestreet.linear import feature_columns_from_schema
from janestreet.official_lags import DAILY_LAG_JOIN_KEYS, daily_last_responder_lag_columns, responder_lag_columns
from janestreet.paths import DAILY_RESPONDER_LAGS_LAST_PARQUET, PROJECT_ROOT, TRAIN_PARQUET_DIR, TRAIN_WITH_RESPONDER_LAGS_PARQUET


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--valid-window", type=int, default=60)
    parser.add_argument("--max-date-id", type=int, default=-1)
    parser.add_argument("--fold-start-index", type=int, default=0)
    parser.add_argument("--fold-limit", type=int, default=0)
    parser.add_argument("--inner-oof-folds", type=int, default=3)
    parser.add_argument("--inner-valid-window", type=int, default=20)
    parser.add_argument("--engines", default="xgboost,lightgbm,catboost")
    parser.add_argument("--train-sample-frac", type=float, default=0.05)
    parser.add_argument("--gbdt-seeds", default="17")
    parser.add_argument("--max-iter", type=int, default=80)
    parser.add_argument("--catboost-max-iter", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--lightgbm-min-child-samples", type=int, default=200)
    parser.add_argument("--lightgbm-subsample", type=float, default=1.0)
    parser.add_argument("--lightgbm-colsample-bytree", type=float, default=1.0)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--chunk-days", type=int, default=10)
    parser.add_argument("--time-bucket-size", type=int, default=100)
    parser.add_argument("--use-responder-lags", action="store_true")
    parser.add_argument("--responder-lag-mode", choices=["same_time", "daily_last"], default="same_time")
    parser.add_argument("--responder-lag-target", choices=["both", "gbdt"], default="both")
    parser.add_argument("--gbdt-target-mode", choices=["target", "residual_raw_ridge"], default="target")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/tree_engine_ensemble"))
    args = parser.parse_args()
    engines = _parse_engines(args.engines)
    runner = _load_blend_runner()

    _validate_args(args)
    train, lag_features = _scan_training_frame(args)
    if args.max_date_id >= 0:
        train = train.filter(pl.col("date_id") <= args.max_date_id)
    base_features = feature_columns_from_schema(train.collect_schema())
    feature_sets = _make_feature_sets(
        base_features=base_features,
        lag_features=lag_features,
        id_values=("time_id", "symbol_id"),
        responder_lag_target=args.responder_lag_target,
    )
    folds = _select_requested_folds(runner._make_folds(train, _runner_args(args, engines[0])), args)
    rows: list[dict[str, float | int | str | None]] = []
    parameter_rows: list[dict[str, float | int | str | None]] = []
    slice_partials = {"weight_bucket": [], "time_bucket": [], "date_id_symbol_id": []}
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        calibration = _collect_engine_predictions(
            runner=runner,
            train=train,
            ridge_features=feature_sets["ridge_features"],
            gbdt_raw_features=feature_sets["gbdt_raw_features"],
            model_features=feature_sets["model_features"],
            engines=engines,
            fold=fold,
            args=args,
            inner=True,
        )
        clip_abs = runner._target_abs_quantile(calibration, 0.999)
        weight_thresholds = runner._weight_thresholds(calibration)
        pred_abs_thresholds = fit_abs_prediction_thresholds(calibration, prediction="ridge_prediction")
        calibration = _add_regime_columns(
            calibration,
            weight_thresholds,
            pred_abs_thresholds,
            time_bucket_size=args.time_bucket_size,
        )
        ridge_calibrator = fit_shrinkage_calibrator(
            calibration,
            name="ridge_weight_predabs",
            group_columns=["weight_bucket", "prediction_abs_bucket"],
            prediction="ridge_prediction",
            min_group_rows=2_000,
            clip_abs=clip_abs,
        )
        calibration = ridge_calibrator.apply(
            calibration,
            prediction="ridge_prediction",
            output="ridge_calibrated_prediction",
        )
        ensemble_columns = ("ridge_calibrated_prediction",) + tuple(f"{engine}_prediction" for engine in engines)
        simplex_weights = fit_simplex_blend_weights(calibration, prediction_columns=ensemble_columns)
        parameter_rows.extend(
            {
                **_fold_metadata(fold),
                "prediction": prediction,
                "weight": weight,
            }
            for prediction, weight in simplex_weights.items()
        )

        validation = _collect_engine_predictions(
            runner=runner,
            train=train,
            ridge_features=feature_sets["ridge_features"],
            gbdt_raw_features=feature_sets["gbdt_raw_features"],
            model_features=feature_sets["model_features"],
            engines=engines,
            fold=fold,
            args=args,
            inner=False,
        )
        validation = _add_regime_columns(
            validation,
            weight_thresholds,
            pred_abs_thresholds,
            time_bucket_size=args.time_bucket_size,
        )
        validation = ridge_calibrator.apply(
            validation,
            prediction="ridge_prediction",
            output="ridge_calibrated_prediction",
        )
        validation = add_simplex_blend_prediction(
            validation,
            weights=simplex_weights,
            output="ensemble_prediction",
        )
        if args.save_predictions:
            _write_prediction_frame(validation, fold, args.output_dir / "validation_predictions")

        strategy_columns = {"ensemble": "ensemble_prediction", "ridge_calibrated": "ridge_calibrated_prediction"}
        strategy_columns.update({engine: f"{engine}_prediction" for engine in engines})
        for strategy, prediction in strategy_columns.items():
            scored = validation.with_columns(pl.col(prediction).alias("strategy_prediction"))
            rows.append(
                {
                    **_fold_metadata(fold),
                    "strategy": strategy,
                    **_score_frame(scored),
                }
            )
            for name, by in {
                "weight_bucket": ["weight_bucket"],
                "time_bucket": ["time_bucket"],
                "date_id_symbol_id": ["date_id", "symbol_id"],
            }.items():
                slice_partials[name].append(
                    aggregate_weighted_r2_by_slice(scored, by, prediction="strategy_prediction").with_columns(
                        pl.lit(strategy).alias("strategy")
                    )
                )

    results = pl.DataFrame(rows)
    summary = _summary_by_strategy(results)
    results.write_csv(args.output_dir / "tree_engine_ensemble_by_fold.csv")
    summary.write_csv(args.output_dir / "tree_engine_ensemble_summary.csv")
    pl.DataFrame(parameter_rows).write_csv(args.output_dir / "tree_engine_ensemble_weights.csv")
    _write_slice_outputs(slice_partials, args.output_dir)
    report = {
        "experiment": "tree_engine_ensemble",
        "engines": engines,
        "max_date_id": args.max_date_id,
        "fold_start_index": args.fold_start_index,
        "fold_limit": args.fold_limit,
        "train_sample_frac": args.train_sample_frac,
        "gbdt_seeds": args.gbdt_seeds,
        "max_iter": args.max_iter,
        "catboost_max_iter": args.catboost_max_iter,
        "gbdt_target_mode": args.gbdt_target_mode,
        "lightgbm_subsample": args.lightgbm_subsample,
        "lightgbm_colsample_bytree": args.lightgbm_colsample_bytree,
        "id_columns": ["time_id", "symbol_id"],
        "use_responder_lags": args.use_responder_lags,
        "responder_lag_mode": args.responder_lag_mode,
        "responder_lag_target": args.responder_lag_target,
        "responder_lag_features": lag_features,
        "responder_lag_feature_count": len(lag_features),
        "n_ridge_features": len(feature_sets["ridge_features"]),
        "n_gbdt_features": len(feature_sets["model_features"]),
        "save_predictions": args.save_predictions,
        "best_strategy": summary.row(0, named=True),
    }
    (args.output_dir / "tree_engine_ensemble_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(results)
    print(summary)
    print(f"Wrote {args.output_dir}")


def _validate_args(args: argparse.Namespace) -> None:
    for name in ["n_folds", "train_window", "valid_window", "inner_oof_folds", "inner_valid_window", "max_iter", "catboost_max_iter", "max_leaf_nodes", "n_jobs", "chunk_days", "time_bucket_size"]:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_date_id < -1:
        raise ValueError("--max-date-id must be -1 or a non-negative date_id")
    if args.fold_start_index < 0 or args.fold_limit < 0:
        raise ValueError("fold slicing settings must be non-negative")
    if args.train_sample_frac <= 0.0 or args.train_sample_frac > 1.0:
        raise ValueError("--train-sample-frac must be in (0, 1]")
    if args.learning_rate <= 0.0 or args.l2_regularization < 0.0:
        raise ValueError("learning rate and regularization settings are invalid")
    if args.lightgbm_subsample <= 0.0 or args.lightgbm_subsample > 1.0:
        raise ValueError("--lightgbm-subsample must be in (0, 1]")
    if args.lightgbm_colsample_bytree <= 0.0 or args.lightgbm_colsample_bytree > 1.0:
        raise ValueError("--lightgbm-colsample-bytree must be in (0, 1]")


def _select_requested_folds(folds: list[DateFold], args: argparse.Namespace) -> list[DateFold]:
    selected = folds[args.fold_start_index :]
    if args.fold_limit > 0:
        selected = selected[: args.fold_limit]
    if not selected:
        raise ValueError("fold selection is empty")
    return selected


def _collect_engine_predictions(
    *,
    runner: object,
    train: pl.LazyFrame,
    ridge_features: tuple[str, ...],
    gbdt_raw_features: tuple[str, ...],
    model_features: tuple[str, ...],
    engines: tuple[str, ...],
    fold: DateFold,
    args: argparse.Namespace,
    inner: bool,
) -> pl.DataFrame:
    merged: pl.DataFrame | None = None
    keys = ["date_id", "time_id", "symbol_id"]
    for engine in engines:
        engine_args = _runner_args(args, engine)
        if inner:
            frame = runner._collect_inner_oof_predictions(
                train,
                ridge_features,
                gbdt_raw_features,
                model_features,
                (),
                fold,
                engine_args,
            )
        else:
            models = runner._fit_models(
                train,
                ridge_features,
                ridge_features,
                model_features,
                (),
                fold,
                engine_args,
            )
            frame = runner._collect_prediction_frame(
                train,
                ridge_features,
                gbdt_raw_features,
                model_features,
                models,
                fold.valid_start,
                fold.valid_end,
                chunk_days=args.chunk_days,
            )
        frame = frame.rename({"gbdt_prediction": f"{engine}_prediction"})
        if merged is None:
            merged = frame
        else:
            merged = merged.join(frame.select(keys + [f"{engine}_prediction"]), on=keys, how="inner")
    if merged is None:
        raise ValueError("no engines configured")
    return merged


def _write_prediction_frame(frame: pl.DataFrame, fold: DateFold, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_columns = [
        column
        for column in [
            "ridge_prediction",
            "ridge_calibrated_prediction",
            "xgboost_prediction",
            "lightgbm_prediction",
            "catboost_prediction",
            "ensemble_prediction",
        ]
        if column in frame.columns
    ]
    frame.select(["date_id", "time_id", "symbol_id", "weight", "responder_6", *prediction_columns]).with_columns(
        pl.lit(fold.name).alias("fold")
    ).write_parquet(output_dir / f"{fold.name}.parquet", compression="zstd")


def _runner_args(args: argparse.Namespace, engine: str) -> argparse.Namespace:
    max_iter = args.catboost_max_iter if engine == "catboost" else args.max_iter
    return argparse.Namespace(
        n_folds=args.n_folds,
        train_window=args.train_window,
        valid_window=args.valid_window,
        inner_oof_folds=args.inner_oof_folds,
        inner_valid_window=args.inner_valid_window,
        gap=0,
        ridge_alpha=1000.0,
        train_sample_frac=args.train_sample_frac,
        chunk_days=args.chunk_days,
        time_bucket_size=args.time_bucket_size,
        symbol_graph_neighbors=5,
        reservoir_states=8,
        koopman_span=16,
        lightgbm_min_child_samples=args.lightgbm_min_child_samples,
        lightgbm_subsample=args.lightgbm_subsample,
        lightgbm_colsample_bytree=args.lightgbm_colsample_bytree,
        lightgbm_n_jobs=args.n_jobs,
        blend_group_values=(),
        min_group_rows=2_000,
        min_blend_group_rows=2_000,
        gbdt_seed_values=_parse_seeds(args.gbdt_seeds),
        gbdt_engine=engine,
        gbdt_target_mode=args.gbdt_target_mode,
        gbdt_id_values=("time_id", "symbol_id"),
        catboost_categorical_id_columns=engine == "catboost",
        max_iter=max_iter,
        learning_rate=args.learning_rate,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
    )


def _scan_training_frame(args: argparse.Namespace) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    if not args.use_responder_lags:
        return pl.scan_parquet(str(TRAIN_PARQUET_DIR)), ()
    if args.responder_lag_mode == "same_time":
        if not TRAIN_WITH_RESPONDER_LAGS_PARQUET.exists():
            raise FileNotFoundError(
                f"{TRAIN_WITH_RESPONDER_LAGS_PARQUET} not found. Run: uv run python scripts/build_responder_lag_cache.py"
            )
        return pl.scan_parquet(str(TRAIN_WITH_RESPONDER_LAGS_PARQUET)), responder_lag_columns()
    if not DAILY_RESPONDER_LAGS_LAST_PARQUET.exists():
        raise FileNotFoundError(
            f"{DAILY_RESPONDER_LAGS_LAST_PARQUET} not found. Run: uv run python scripts/build_daily_responder_lag_cache.py"
        )
    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR)).join(
        pl.scan_parquet(str(DAILY_RESPONDER_LAGS_LAST_PARQUET / "*.parquet")),
        on=list(DAILY_LAG_JOIN_KEYS),
        how="left",
    )
    return train, daily_last_responder_lag_columns()


def _make_feature_sets(
    *,
    base_features: tuple[str, ...],
    lag_features: tuple[str, ...],
    id_values: tuple[str, ...],
    responder_lag_target: str,
) -> dict[str, tuple[str, ...]]:
    if responder_lag_target not in {"both", "gbdt"}:
        raise ValueError("responder_lag_target must be 'both' or 'gbdt'")
    ridge_features = tuple(dict.fromkeys([*base_features, *lag_features])) if responder_lag_target == "both" else base_features
    gbdt_raw_features = tuple(dict.fromkeys([*base_features, *lag_features]))
    model_features = tuple(dict.fromkeys([*gbdt_raw_features, *id_values]))
    return {
        "ridge_features": ridge_features,
        "gbdt_raw_features": gbdt_raw_features,
        "model_features": model_features,
    }


def _add_regime_columns(
    frame: pl.DataFrame,
    weight_thresholds: dict[str, float],
    pred_abs_thresholds: dict[str, float],
    *,
    time_bucket_size: int,
) -> pl.DataFrame:
    with_regimes = frame.with_columns(
        [
            (pl.col("time_id") // time_bucket_size).cast(pl.Int16).alias("time_bucket"),
            (
                pl.when(pl.col("weight") <= weight_thresholds["q50"])
                .then(pl.lit("q00_q50"))
                .when(pl.col("weight") <= weight_thresholds["q90"])
                .then(pl.lit("q50_q90"))
                .when(pl.col("weight") <= weight_thresholds["q99"])
                .then(pl.lit("q90_q99"))
                .otherwise(pl.lit("q99_q100"))
                .alias("weight_bucket")
            ),
        ]
    )
    return add_abs_prediction_bucket(
        with_regimes,
        pred_abs_thresholds,
        prediction="ridge_prediction",
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
    for name, group_cols in {
        "weight_bucket": ["strategy", "weight_bucket"],
        "time_bucket": ["strategy", "time_bucket"],
        "date_id_symbol_id": ["strategy", "date_id", "symbol_id"],
    }.items():
        combine_slice_aggregates(slice_partials[name], group_cols).write_csv(output_dir / f"{name}.csv")


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
    allowed = {"xgboost", "lightgbm", "catboost"}
    unknown = set(engines) - allowed
    if unknown:
        raise ValueError(f"unknown engines: {', '.join(sorted(unknown))}")
    if not engines:
        raise ValueError("--engines must contain at least one engine")
    return engines


def _parse_seeds(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not seeds:
        raise ValueError("--gbdt-seeds must contain at least one integer")
    return seeds


def _load_blend_runner() -> object:
    path = PROJECT_ROOT / "scripts" / "run_ridge_gbdt_blend.py"
    spec = importlib.util.spec_from_file_location("run_ridge_gbdt_blend", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
