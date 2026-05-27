"""Train a causal residual NN stacker on OOF tree-ensemble predictions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

from janestreet.blending import add_simplex_blend_prediction, fit_simplex_blend_weights
from janestreet.calibration import add_abs_prediction_bucket, fit_abs_prediction_thresholds, fit_shrinkage_calibrator
from janestreet.linear import feature_columns_from_schema
from janestreet.paths import PROJECT_ROOT, TRAIN_PARQUET_DIR


STACKER_CONTINUOUS_COLUMNS = (
    "ridge_calibrated_prediction",
    "xgboost_prediction",
    "lightgbm_prediction",
    "ensemble_prediction",
    "xgb_minus_lgb",
    "ensemble_abs",
    "xgb_lgb_abs_gap",
    "weight_feature",
    "time_sin_967",
    "time_cos_967",
    "time_sin_483",
    "time_cos_483",
)
STACKER_CATEGORICAL_COLUMNS = ("symbol_id", "time_id", "weight_bucket_code", "prediction_abs_bucket_code")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=1)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--valid-window", type=int, default=5)
    parser.add_argument("--inner-oof-folds", type=int, default=3)
    parser.add_argument("--inner-valid-window", type=int, default=10)
    parser.add_argument("--train-sample-frac", type=float, default=0.05)
    parser.add_argument("--gbdt-seeds", default="17")
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--learning-rate-tree", type=float, default=0.03)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--lightgbm-min-child-samples", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--chunk-days", type=int, default=10)
    parser.add_argument("--time-bucket-size", type=int, default=100)
    parser.add_argument("--stacker-early-days", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=8e-4)
    parser.add_argument("--aux-loss-weight", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-5)
    parser.add_argument("--max-calibration-scale", type=float, default=1.0)
    parser.add_argument("--sophia-beta1", type=float, default=0.965)
    parser.add_argument("--sophia-beta2", type=float, default=0.99)
    parser.add_argument("--sophia-rho", type=float, default=0.04)
    parser.add_argument("--sophia-clip", type=float, default=1.0)
    parser.add_argument("--sophia-update-period", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/tree_residual_stacker_smoke"))
    args = parser.parse_args()
    _validate_args(args)

    tree = _load_script("run_tree_engine_ensemble")
    nnmod = _load_script("run_competitive_tabular_nn")
    runner = tree._load_blend_runner()
    nnmod._set_reproducibility(args.seed, args.torch_threads)
    device = nnmod._resolve_device(args.device)

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    base_features = feature_columns_from_schema(train.collect_schema())
    model_features = tuple(dict.fromkeys([*base_features, "time_id", "symbol_id"]))
    folds = runner._make_folds(train, _tree_args(args, "xgboost"))

    rows: list[dict[str, float | int | str]] = []
    parameter_rows: list[dict[str, float | int | str]] = []
    for fold in folds:
        calibration = _collect_engine_predictions(
            runner=runner,
            train=train,
            ridge_features=base_features,
            model_features=model_features,
            engines=("xgboost", "lightgbm"),
            fold=fold,
            args=args,
            inner=True,
        )
        clip_abs = runner._target_abs_quantile(calibration, 0.999)
        weight_thresholds = runner._weight_thresholds(calibration)
        pred_abs_thresholds = fit_abs_prediction_thresholds(calibration, prediction="ridge_prediction")
        calibration = tree._add_regime_columns(
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
        ensemble_columns = ("ridge_calibrated_prediction", "xgboost_prediction", "lightgbm_prediction")
        simplex_weights = fit_simplex_blend_weights(calibration, prediction_columns=ensemble_columns)
        calibration = add_simplex_blend_prediction(calibration, weights=simplex_weights, output="ensemble_prediction")
        calibration = _add_stacker_columns(calibration)

        validation = _collect_engine_predictions(
            runner=runner,
            train=train,
            ridge_features=base_features,
            model_features=model_features,
            engines=("xgboost", "lightgbm"),
            fold=fold,
            args=args,
            inner=False,
        )
        validation = tree._add_regime_columns(
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
        validation = add_simplex_blend_prediction(validation, weights=simplex_weights, output="ensemble_prediction")
        validation = _add_stacker_columns(validation)

        fit_frame, early_frame = _split_stacker_fit_early(calibration, args.stacker_early_days)
        fit_frame = _with_residual_target(fit_frame)
        early_frame = _with_residual_target(early_frame)

        standardization = nnmod._fit_standardization(
            fit_frame,
            STACKER_CONTINUOUS_COLUMNS,
            STACKER_CATEGORICAL_COLUMNS,
            ("responder_6",),
            center_target=False,
        )
        model = nnmod.TabularMLP(
            n_continuous=len(STACKER_CONTINUOUS_COLUMNS),
            categorical_cardinalities=[spec.num_classes for spec in standardization.categorical_specs],
            hidden_size=args.hidden_size,
            depth=args.depth,
            dropout=args.dropout,
            output_dim=1,
            ensemble_size=args.ensemble_size,
        )
        fit_result = nnmod._fit_model(
            model,
            fit_frame,
            early_frame,
            standardization,
            STACKER_CONTINUOUS_COLUMNS,
            STACKER_CATEGORICAL_COLUMNS,
            ("responder_6",),
            args,
            device,
        )
        residual_scale = _fit_residual_scale(
            nnmod,
            model,
            early_frame,
            standardization,
            args,
            device,
        )
        base_metrics = _score(validation, "ensemble_prediction")
        validation_for_pred = _with_residual_target(validation)
        residual_pred, _, _ = nnmod._predict_target_vectors(
            model,
            validation_for_pred,
            standardization,
            STACKER_CONTINUOUS_COLUMNS,
            STACKER_CATEGORICAL_COLUMNS,
            ("responder_6",),
            args,
            device,
        )
        validation = validation.with_columns(
            pl.Series("stacker_residual_prediction", residual_pred * residual_scale),
        ).with_columns(
            (pl.col("ensemble_prediction") + pl.col("stacker_residual_prediction")).alias("stacked_prediction")
        )
        stacked_metrics = _score(validation, "stacked_prediction")
        rows.extend(
            [
                {**_fold_metadata(fold), "strategy": "base_ensemble", **base_metrics},
                {**_fold_metadata(fold), "strategy": "nn_residual_stacker", **stacked_metrics},
            ]
        )
        parameter_rows.append(
            {
                **_fold_metadata(fold),
                "residual_scale": residual_scale,
                "fit_rows": fit_frame.height,
                "early_rows": early_frame.height,
                "trained_epochs": fit_result["trained_epochs"],
                "best_epoch": fit_result["best_epoch"],
                **{f"simplex_{key}": value for key, value in simplex_weights.items()},
            }
        )

    results = pl.DataFrame(rows)
    summary = _summary_by_strategy(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(args.output_dir / "tree_residual_stacker_by_fold.csv")
    summary.write_csv(args.output_dir / "tree_residual_stacker_summary.csv")
    pl.DataFrame(parameter_rows).write_csv(args.output_dir / "tree_residual_stacker_parameters.csv")
    report = {
        "experiment": "tree_residual_stacker",
        "hypothesis": "A NN trained only on OOF tree residuals can correct the active tree ensemble without in-sample leakage.",
        "baseline": "tree_engine_ensemble ridge/xgboost/lightgbm simplex",
        "stacker_continuous_columns": STACKER_CONTINUOUS_COLUMNS,
        "stacker_categorical_columns": STACKER_CATEGORICAL_COLUMNS,
        "params": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "summary": summary.to_dicts(),
    }
    (args.output_dir / "tree_residual_stacker_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(results)
    print(summary)
    print(f"Wrote {args.output_dir}")


def _validate_args(args: argparse.Namespace) -> None:
    for name in (
        "n_folds",
        "train_window",
        "valid_window",
        "inner_oof_folds",
        "inner_valid_window",
        "chunk_days",
        "stacker_early_days",
        "epochs",
        "batch_size",
        "hidden_size",
        "depth",
        "ensemble_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 < args.train_sample_frac <= 1.0:
        raise ValueError("--train-sample-frac must be in (0, 1]")
    if args.max_calibration_scale <= 0.0:
        raise ValueError("--max-calibration-scale must be positive")


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tree_args(args: argparse.Namespace, engine: str) -> argparse.Namespace:
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
        lightgbm_subsample=1.0,
        lightgbm_colsample_bytree=1.0,
        lightgbm_n_jobs=args.n_jobs,
        blend_group_values=(),
        min_group_rows=2_000,
        min_blend_group_rows=2_000,
        gbdt_seed_values=tuple(int(part) for part in args.gbdt_seeds.split(",") if part.strip()),
        gbdt_engine=engine,
        gbdt_target_mode="target",
        gbdt_id_values=("time_id", "symbol_id"),
        catboost_categorical_id_columns=False,
        max_iter=args.max_iter,
        learning_rate=args.learning_rate_tree,
        max_leaf_nodes=args.max_leaf_nodes,
        l2_regularization=args.l2_regularization,
    )


def _collect_engine_predictions(
    *,
    runner: object,
    train: pl.LazyFrame,
    ridge_features: tuple[str, ...],
    model_features: tuple[str, ...],
    engines: tuple[str, ...],
    fold,
    args: argparse.Namespace,
    inner: bool,
) -> pl.DataFrame:
    merged: pl.DataFrame | None = None
    keys = ["date_id", "time_id", "symbol_id"]
    for engine in engines:
        engine_args = _tree_args(args, engine)
        if inner:
            frame = runner._collect_inner_oof_predictions(
                train,
                ridge_features,
                ridge_features,
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
                ridge_features,
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


def _add_stacker_columns(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        [
            (pl.col("xgboost_prediction") - pl.col("lightgbm_prediction")).alias("xgb_minus_lgb"),
            pl.col("ensemble_prediction").abs().alias("ensemble_abs"),
            (pl.col("xgboost_prediction") - pl.col("lightgbm_prediction")).abs().alias("xgb_lgb_abs_gap"),
            pl.col("weight").alias("weight_feature"),
            (pl.col("time_id") * (2.0 * math.pi / 967.0)).sin().alias("time_sin_967"),
            (pl.col("time_id") * (2.0 * math.pi / 967.0)).cos().alias("time_cos_967"),
            (pl.col("time_id") * (2.0 * math.pi / 483.0)).sin().alias("time_sin_483"),
            (pl.col("time_id") * (2.0 * math.pi / 483.0)).cos().alias("time_cos_483"),
            _bucket_code_expr("weight_bucket").alias("weight_bucket_code"),
            _bucket_code_expr("prediction_abs_bucket").alias("prediction_abs_bucket_code"),
        ]
    ).with_columns([pl.col(name).fill_null(0.0).cast(pl.Float32).alias(name) for name in STACKER_CONTINUOUS_COLUMNS])


def _bucket_code_expr(column: str) -> pl.Expr:
    return (
        pl.when(pl.col(column) == "q00_q50")
        .then(pl.lit(0))
        .when(pl.col(column) == "q50_q90")
        .then(pl.lit(1))
        .when(pl.col(column) == "q90_q99")
        .then(pl.lit(2))
        .otherwise(pl.lit(3))
        .cast(pl.Int32)
    )


def _split_stacker_fit_early(frame: pl.DataFrame, early_days: int) -> tuple[pl.DataFrame, pl.DataFrame]:
    max_date = int(frame["date_id"].max())
    early_start = max_date - early_days + 1
    fit = frame.filter(pl.col("date_id") < early_start)
    early = frame.filter(pl.col("date_id") >= early_start)
    if fit.height == 0 or early.height == 0:
        raise ValueError("stacker fit/early split produced an empty frame")
    return fit, early


def _with_residual_target(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns((pl.col("responder_6") - pl.col("ensemble_prediction")).alias("responder_6"))


def _fit_residual_scale(nnmod, model, frame: pl.DataFrame, standardization, args: argparse.Namespace, device) -> float:
    residual_pred, residual_target, weight = nnmod._predict_target_vectors(
        model,
        frame,
        standardization,
        STACKER_CONTINUOUS_COLUMNS,
        STACKER_CATEGORICAL_COLUMNS,
        ("responder_6",),
        args,
        device,
    )
    denom = float(np.sum(weight * residual_pred * residual_pred))
    if denom <= 1e-12:
        return 0.0
    scale = float(np.sum(weight * residual_target * residual_pred) / denom)
    if not math.isfinite(scale):
        return 0.0
    return max(-args.max_calibration_scale, min(args.max_calibration_scale, scale))


def _score(frame: pl.DataFrame, prediction: str) -> dict[str, float | int]:
    row = frame.select(
        [
            pl.len().alias("rows"),
            pl.col("weight").sum().alias("weight_sum"),
            (pl.col("weight") * (pl.col("responder_6") - pl.col(prediction)).pow(2)).sum().alias("numerator"),
            (pl.col("weight") * pl.col("responder_6").pow(2)).sum().alias("denominator"),
            pl.col(prediction).mean().alias("prediction_mean"),
            pl.col(prediction).std().alias("prediction_std"),
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


def _fold_metadata(fold) -> dict[str, int | str]:
    return {
        "fold": fold.name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "valid_start": fold.valid_start,
        "valid_end": fold.valid_end,
    }


if __name__ == "__main__":
    main()
