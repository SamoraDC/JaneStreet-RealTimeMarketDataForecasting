from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_tree_engine_ensemble.py"
    spec = importlib.util.spec_from_file_location("run_tree_engine_ensemble", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_write_prediction_frame_keeps_available_prediction_columns(tmp_path: Path) -> None:
    module = _load_script_module()
    fold = module.DateFold("rw_01", 0, 1, 2, 3)
    frame = pl.DataFrame(
        {
            "date_id": [2],
            "time_id": [0],
            "symbol_id": [1],
            "weight": [1.0],
            "responder_6": [0.5],
            "ridge_calibrated_prediction": [0.2],
            "xgboost_prediction": [0.3],
            "ensemble_prediction": [0.4],
        }
    )

    module._write_prediction_frame(frame, fold, tmp_path)

    written = pl.read_parquet(tmp_path / "rw_01.parquet")
    assert written.columns == [
        "date_id",
        "time_id",
        "symbol_id",
        "weight",
        "responder_6",
        "ridge_calibrated_prediction",
        "xgboost_prediction",
        "ensemble_prediction",
        "fold",
    ]
    assert written["fold"].to_list() == ["rw_01"]


def test_select_requested_folds_slices_without_renaming() -> None:
    module = _load_script_module()
    folds = [
        module.DateFold("rw_01", 0, 9, 10, 19),
        module.DateFold("rw_02", 10, 19, 20, 29),
        module.DateFold("rw_03", 20, 29, 30, 39),
    ]
    args = type("Args", (), {"fold_start_index": 1, "fold_limit": 1})()

    selected = module._select_requested_folds(folds, args)

    assert [fold.name for fold in selected] == ["rw_02"]


def test_validate_args_accepts_historical_max_date_and_rejects_bad_slice() -> None:
    module = _load_script_module()
    args = type(
        "Args",
        (),
        {
            "n_folds": 5,
            "train_window": 700,
            "valid_window": 60,
            "inner_oof_folds": 3,
            "inner_valid_window": 20,
            "max_iter": 80,
            "catboost_max_iter": 160,
            "max_leaf_nodes": 31,
            "n_jobs": 4,
            "chunk_days": 10,
            "time_bucket_size": 100,
            "max_date_id": 1398,
            "fold_start_index": 0,
            "fold_limit": 0,
            "train_sample_frac": 0.10,
            "learning_rate": 0.03,
            "l2_regularization": 1.0,
            "lightgbm_subsample": 1.0,
            "lightgbm_colsample_bytree": 1.0,
        },
    )()

    module._validate_args(args)
    args.fold_start_index = -1
    with pytest.raises(ValueError, match="fold slicing"):
        module._validate_args(args)


def test_make_feature_sets_routes_lags_to_requested_models() -> None:
    module = _load_script_module()

    both = module._make_feature_sets(
        base_features=("feature_00", "feature_01"),
        lag_features=("responder_6_lag_1",),
        id_values=("time_id", "symbol_id"),
        responder_lag_target="both",
    )
    gbdt_only = module._make_feature_sets(
        base_features=("feature_00", "feature_01"),
        lag_features=("responder_6_lag_1",),
        id_values=("time_id", "symbol_id"),
        responder_lag_target="gbdt",
    )

    assert both["ridge_features"] == ("feature_00", "feature_01", "responder_6_lag_1")
    assert both["model_features"] == ("feature_00", "feature_01", "responder_6_lag_1", "time_id", "symbol_id")
    assert gbdt_only["ridge_features"] == ("feature_00", "feature_01")
    assert gbdt_only["gbdt_raw_features"] == ("feature_00", "feature_01", "responder_6_lag_1")


def test_runner_args_preserves_primary_alpha_options() -> None:
    module = _load_script_module()
    args = type(
        "Args",
        (),
        {
            "n_folds": 5,
            "train_window": 700,
            "valid_window": 60,
            "inner_oof_folds": 3,
            "inner_valid_window": 20,
            "train_sample_frac": 0.10,
            "chunk_days": 10,
            "time_bucket_size": 100,
            "lightgbm_min_child_samples": 200,
            "lightgbm_subsample": 0.8,
            "lightgbm_colsample_bytree": 0.7,
            "n_jobs": 4,
            "gbdt_seeds": "17,23",
            "gbdt_target_mode": "residual_raw_ridge",
            "catboost_max_iter": 160,
            "max_iter": 80,
            "learning_rate": 0.03,
            "max_leaf_nodes": 31,
            "l2_regularization": 1.0,
        },
    )()

    runner_args = module._runner_args(args, "lightgbm")

    assert runner_args.gbdt_target_mode == "residual_raw_ridge"
    assert runner_args.lightgbm_subsample == pytest.approx(0.8)
    assert runner_args.lightgbm_colsample_bytree == pytest.approx(0.7)
