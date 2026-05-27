"""Train and export base model artifacts for the Kaggle submission runtime.

This script prepares the heavy artifacts that `submission/submission.py` will
eventually load: a final TabM model and final XGBoost/LightGBM tree components.
It is intentionally separate from the notebook entrypoint so expensive training
can be done locally and attached to Kaggle as a dataset for late submission.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl
import torch

from janestreet.calibration import add_abs_prediction_bucket, fit_abs_prediction_thresholds, fit_shrinkage_calibrator
from janestreet.folds import DateFold
from janestreet.linear import feature_columns_from_schema
from janestreet.paths import TRAIN_PARQUET_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/jane_street_submission/base_models"))
    parser.add_argument("--max-date-id", type=int, default=-1)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--tabm-train-window", type=int, default=700)
    parser.add_argument("--tabm-max-train-rows", type=int, default=4_000_000)
    parser.add_argument("--tabm-epochs", type=int, default=4)
    parser.add_argument("--tabm-batch-size", type=int, default=8192)
    parser.add_argument("--tabm-hidden-size", type=int, default=512)
    parser.add_argument("--tabm-depth", type=int, default=3)
    parser.add_argument("--tabm-dropout", type=float, default=0.25)
    parser.add_argument("--tabm-ensemble-size", type=int, default=16)
    parser.add_argument("--tabm-learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--tabm-weight-decay", type=float, default=8e-4)
    parser.add_argument("--aux-targets", default="responder_0,responder_1,responder_2,responder_3,responder_4,responder_5,responder_7,responder_8")
    parser.add_argument("--aux-loss-weight", type=float, default=0.25)
    parser.add_argument("--early-stopping-valid-days", type=int, default=5)
    parser.add_argument("--tree-train-window", type=int, default=120)
    parser.add_argument("--tree-inner-oof-folds", type=int, default=3)
    parser.add_argument("--tree-inner-valid-window", type=int, default=20)
    parser.add_argument("--tree-engines", default="xgboost,lightgbm")
    parser.add_argument("--tree-sample-frac", type=float, default=0.10)
    parser.add_argument("--tree-seeds", default="17,23,37")
    parser.add_argument("--tree-max-iter", type=int, default=80)
    parser.add_argument("--tree-learning-rate", type=float, default=0.03)
    parser.add_argument("--tree-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--tree-l2-regularization", type=float, default=1.0)
    parser.add_argument("--tree-n-jobs", type=int, default=4)
    parser.add_argument("--chunk-days", type=int, default=10)
    args = parser.parse_args()

    _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    nn_module = _load_script_module("run_competitive_tabular_nn_for_artifacts", PROJECT_ROOT / "scripts" / "run_competitive_tabular_nn.py")
    tree_module = _load_script_module("run_tree_engine_ensemble_for_artifacts", PROJECT_ROOT / "scripts" / "run_tree_engine_ensemble.py")
    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    if args.max_date_id >= 0:
        train = train.filter(pl.col("date_id") <= args.max_date_id)
    min_date, train_end = _date_bounds(train)

    tabm_report = _train_tabm_artifact(nn_module, train, min_date, train_end, args)
    tree_report = _train_tree_artifact(tree_module, train, min_date, train_end, args)
    manifest = {
        "artifact_type": "jane_street_submission_base_models",
        "max_date_id": train_end,
        "tabm": tabm_report,
        "tree": tree_report,
        "causality_note": (
            "Final base models are trained only on public train dates up to max_date_id. "
            "Submission online updates must use only gateway lags."
        ),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print(f"Wrote {args.output_dir}")


def _validate_args(args: argparse.Namespace) -> None:
    positive = [
        "tabm_train_window",
        "tabm_max_train_rows",
        "tabm_epochs",
        "tabm_batch_size",
        "tabm_hidden_size",
        "tabm_depth",
        "tabm_ensemble_size",
        "early_stopping_valid_days",
        "tree_train_window",
        "tree_inner_oof_folds",
        "tree_inner_valid_window",
        "tree_max_iter",
        "tree_max_leaf_nodes",
        "tree_n_jobs",
        "chunk_days",
    ]
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.max_date_id < -1:
        raise ValueError("--max-date-id must be -1 or a non-negative date_id")
    if args.tree_sample_frac <= 0.0 or args.tree_sample_frac > 1.0:
        raise ValueError("--tree-sample-frac must be in (0, 1]")


def _train_tabm_artifact(nn, train: pl.LazyFrame, min_date: int, train_end: int, args: argparse.Namespace) -> dict[str, object]:
    nn._set_reproducibility(args.seed, args.torch_threads)
    device = nn._resolve_device(args.device)
    schema = train.collect_schema()
    continuous_columns, categorical_columns = nn._select_model_columns(
        schema,
        n_features=79,
        use_official_lags=True,
        include_time_features=True,
        include_weight_feature=True,
    )
    target_columns = (nn.TARGET_COLUMN, *nn._parse_aux_targets(args.aux_targets))
    fit_start = max(min_date, train_end - args.tabm_train_window + 1)
    fit_end = max(fit_start, train_end - args.early_stopping_valid_days)
    train_frame = nn._collect_model_frame(
        train,
        fit_start,
        fit_end,
        continuous_columns,
        categorical_columns,
        target_columns,
        sample_max_rows=args.tabm_max_train_rows,
        seed=args.seed,
    )
    early_frame = None
    if fit_end < train_end:
        early_frame = nn._collect_model_frame(
            train,
            fit_end + 1,
            train_end,
            continuous_columns,
            categorical_columns,
            target_columns,
            sample_max_rows=max(args.tabm_batch_size * 4, args.tabm_max_train_rows // 8),
            seed=args.seed + 3,
        )
    standardization = nn._fit_standardization(
        train_frame,
        continuous_columns,
        categorical_columns,
        target_columns,
        center_target=False,
    )
    model = nn.make_tabular_model(
        model_type="tabm",
        n_continuous=len(continuous_columns),
        categorical_cardinalities=[spec.num_classes for spec in standardization.categorical_specs],
        hidden_size=args.tabm_hidden_size,
        depth=args.tabm_depth,
        dropout=args.tabm_dropout,
        output_dim=len(target_columns),
        ensemble_size=args.tabm_ensemble_size,
    )
    nn_args = argparse.Namespace(
        learning_rate=args.tabm_learning_rate,
        weight_decay=args.tabm_weight_decay,
        aux_loss_weight=args.aux_loss_weight,
        batch_size=args.tabm_batch_size,
        epochs=args.tabm_epochs,
        early_stopping_min_delta=1e-5,
        early_stopping_patience=2,
        sophia_beta1=0.965,
        sophia_beta2=0.99,
        sophia_rho=0.04,
        sophia_clip=1.0,
        sophia_update_period=10,
        max_calibration_scale=2.0,
    )
    fit_result = nn._fit_model(
        model,
        train_frame,
        early_frame,
        standardization,
        continuous_columns,
        categorical_columns,
        target_columns,
        nn_args,
        device,
    )
    calibration = nn.PredictionCalibration()
    if early_frame is not None and early_frame.height > 0:
        calibration = nn._fit_prediction_calibration(
            model,
            early_frame,
            standardization,
            continuous_columns,
            categorical_columns,
            target_columns,
            nn_args,
            device,
        )
    tabm_dir = args.output_dir / "tabm"
    tabm_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_config": {
                "model_type": "tabm",
                "n_continuous": len(continuous_columns),
                "categorical_cardinalities": [spec.num_classes for spec in standardization.categorical_specs],
                "hidden_size": args.tabm_hidden_size,
                "depth": args.tabm_depth,
                "dropout": args.tabm_dropout,
                "output_dim": len(target_columns),
                "ensemble_size": args.tabm_ensemble_size,
            },
        },
        tabm_dir / "model.pt",
    )
    np.savez_compressed(
        tabm_dir / "standardization.npz",
        continuous_mean=standardization.continuous_mean,
        continuous_scale=standardization.continuous_scale,
        target_mean=standardization.target_mean,
        target_scale=standardization.target_scale,
    )
    tabm_config = {
        "fit_start": fit_start,
        "fit_end": fit_end,
        "train_end": train_end,
        "continuous_columns": continuous_columns,
        "categorical_columns": categorical_columns,
        "target_columns": target_columns,
        "categorical_specs": [spec.__dict__ for spec in standardization.categorical_specs],
        "prediction_scale": calibration.scale,
        "fit_result": fit_result,
        "train_rows": train_frame.height,
        "early_rows": 0 if early_frame is None else early_frame.height,
        "online_learning_rate": 1e-4,
        "online_epochs": 1,
        "online_max_update_rows_per_date": 20000,
    }
    (tabm_dir / "config.json").write_text(json.dumps(tabm_config, indent=2) + "\n", encoding="utf-8")
    return tabm_config


def _train_tree_artifact(tree, train: pl.LazyFrame, min_date: int, train_end: int, args: argparse.Namespace) -> dict[str, object]:
    runner = tree._load_blend_runner()
    engines = tree._parse_engines(args.tree_engines)
    base_features = feature_columns_from_schema(train.collect_schema())
    model_features = tuple(dict.fromkeys([*base_features, "time_id", "symbol_id"]))
    tree_start = max(min_date, train_end - args.tree_train_window + 1)
    fold = DateFold("submission_tree", tree_start, train_end, train_end + 1, train_end + 1)
    tree_args = argparse.Namespace(
        n_folds=1,
        train_window=args.tree_train_window,
        valid_window=1,
        inner_oof_folds=args.tree_inner_oof_folds,
        inner_valid_window=args.tree_inner_valid_window,
        engines=args.tree_engines,
        train_sample_frac=args.tree_sample_frac,
        gbdt_seeds=args.tree_seeds,
        max_iter=args.tree_max_iter,
        catboost_max_iter=args.tree_max_iter,
        learning_rate=args.tree_learning_rate,
        max_leaf_nodes=args.tree_max_leaf_nodes,
        l2_regularization=args.tree_l2_regularization,
        lightgbm_min_child_samples=200,
        n_jobs=args.tree_n_jobs,
        chunk_days=args.chunk_days,
        time_bucket_size=100,
        max_date_id=args.max_date_id,
        fold_start_index=0,
        fold_limit=0,
    )
    calibration = tree._collect_engine_predictions(
        runner=runner,
        train=train,
        ridge_features=base_features,
        model_features=model_features,
        engines=engines,
        fold=fold,
        args=tree_args,
        inner=True,
    )
    clip_abs = runner._target_abs_quantile(calibration, 0.999)
    weight_thresholds = runner._weight_thresholds(calibration)
    pred_abs_thresholds = fit_abs_prediction_thresholds(calibration, prediction="ridge_prediction")
    calibration = tree._add_regime_columns(
        calibration,
        weight_thresholds,
        pred_abs_thresholds,
        time_bucket_size=tree_args.time_bucket_size,
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
    simplex_weights = tree.fit_simplex_blend_weights(calibration, prediction_columns=ensemble_columns)
    models_by_engine = {}
    for engine in engines:
        fitted = runner._fit_models(
            train,
            base_features,
            base_features,
            model_features,
            (),
            fold,
            tree._runner_args(tree_args, engine),
        )
        models_by_engine[engine] = {
            "ridge_model": fitted.ridge_model,
            "gbdt_models": fitted.gbdt_models,
            "gbdt_target_mode": fitted.gbdt_target_mode,
            "cat_feature_indices": fitted.cat_feature_indices,
        }
    tree_dir = args.output_dir / "tree"
    tree_dir.mkdir(parents=True, exist_ok=True)
    with (tree_dir / "tree_artifact.pkl").open("wb") as handle:
        pickle.dump(
            {
                "engines": engines,
                "base_features": base_features,
                "model_features": model_features,
                "models_by_engine": models_by_engine,
                "ridge_calibrator": ridge_calibrator,
                "weight_thresholds": weight_thresholds,
                "pred_abs_thresholds": pred_abs_thresholds,
                "simplex_weights": simplex_weights,
                "tree_args": vars(tree_args),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return {
        "fit_start": tree_start,
        "fit_end": train_end,
        "engines": engines,
        "base_feature_count": len(base_features),
        "model_feature_count": len(model_features),
        "simplex_weights": simplex_weights,
        "calibration_rows": calibration.height,
    }


def _date_bounds(train: pl.LazyFrame) -> tuple[int, int]:
    row = train.select(pl.min("date_id").alias("min_date_id"), pl.max("date_id").alias("max_date_id")).collect().row(0, named=True)
    return int(row["min_date_id"]), int(row["max_date_id"])


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
