from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def _load_script_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gateway_online_ridge_matches_after_day_update_simulation() -> None:
    gateway = _load_script_module("run_bayesian_gateway_meta_simulation", "scripts/run_bayesian_gateway_meta_simulation.py")
    meta = _load_script_module("run_bayesian_meta_experiments", "scripts/run_bayesian_meta_experiments.py")
    frame = pl.DataFrame(
        {
            "date_id": [1, 2, 3],
            "time_id": [0, 0, 0],
            "symbol_id": [0, 0, 0],
            "x": [1.0, 1.0, 1.0],
            "responder_6": [1.0, 1.0, 1.0],
            "weight": [1.0, 1.0, 1.0],
        }
    )
    precision = np.array([[1.0]], dtype=np.float64)
    rhs = np.array([0.0], dtype=np.float64)

    gateway_metrics, gateway_beta, audit = gateway._simulate_gateway_online_ridge(
        frame,
        features=("x",),
        precision=precision.copy(),
        rhs=rhs.copy(),
    )
    meta_metrics, meta_beta = meta._stream_online_ridge(
        frame,
        features=("x",),
        precision=precision.copy(),
        rhs=rhs.copy(),
    )

    assert gateway_metrics["numerator"] == pytest.approx(meta_metrics["numerator"])
    assert gateway_beta == pytest.approx(meta_beta)
    assert audit["update_source_date_id"].to_list() == [None, 1, 2]
    assert audit["update_is_strictly_past"].to_list() == [True, True, True]


def test_gateway_lag_delivery_rejects_current_day_update() -> None:
    gateway = _load_script_module("run_bayesian_gateway_meta_simulation_reject", "scripts/run_bayesian_gateway_meta_simulation.py")
    pending = pl.DataFrame(
        {
            "date_id": [5],
            "time_id": [0],
            "symbol_id": [0],
            "responder_6": [1.0],
            "weight": [1.0],
            "x": [1.0],
        }
    )

    with pytest.raises(ValueError, match="strictly earlier"):
        gateway._deliver_previous_day_lags(pending, current_date=5)


def test_gateway_feature_sets_remove_duplicate_presets() -> None:
    gateway = _load_script_module("run_bayesian_gateway_meta_simulation_feature_sets", "scripts/run_bayesian_gateway_meta_simulation.py")

    sets = gateway._gateway_feature_sets(
        (
            "tabm_prediction",
            "tree_prediction",
            "xgboost_prediction",
            "lightgbm_prediction",
            "ridge_calibrated_prediction",
        )
    )

    assert sets["experts"] == (
        "tabm_prediction",
        "tree_prediction",
        "xgboost_prediction",
        "lightgbm_prediction",
        "ridge_calibrated_prediction",
    )
    assert sets["components_no_tree_ensemble"] == (
        "tabm_prediction",
        "xgboost_prediction",
        "lightgbm_prediction",
        "ridge_calibrated_prediction",
    )
    assert len(set(sets.values())) == len(sets)
