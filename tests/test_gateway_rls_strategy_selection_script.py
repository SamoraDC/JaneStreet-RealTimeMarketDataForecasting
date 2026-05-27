from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_gateway_rls_strategy_selection.py"
    spec = importlib.util.spec_from_file_location("run_gateway_rls_strategy_selection_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_online_loss_selector_updates_from_previous_day_only() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01"],
            "date_id": [1, 2],
            "time_id": [0, 0],
            "symbol_id": [0, 0],
            "weight": [1.0, 1.0],
            "responder_6": [1.0, 0.0],
            "conservative": [0.0, 0.0],
            "aggressive": [1.0, 10.0],
            "baseline": [0.5, 0.0],
        }
    )

    selected, audit = module.apply_online_loss_selector(
        frame,
        strategies=(("conservative", "conservative"), ("aggressive", "aggressive"), ("baseline", "baseline")),
        default_strategy="conservative",
        ewma_decay=0.5,
        output="selected_prediction",
    )

    assert selected.select(["date_id", "_selected_strategy"]).to_dicts() == [
        {"date_id": 1, "_selected_strategy": "conservative"},
        {"date_id": 2, "_selected_strategy": "aggressive"},
    ]
    assert audit.select(["date_id", "update_source_date_id", "uses_current_day_target", "update_is_strictly_past"]).to_dicts() == [
        {"date_id": 1, "update_source_date_id": None, "uses_current_day_target": False, "update_is_strictly_past": True},
        {"date_id": 2, "update_source_date_id": 1, "uses_current_day_target": False, "update_is_strictly_past": True},
    ]


def test_static_previous_fold_selector_uses_only_prior_folds() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "fold": ["rw_01", "rw_02"],
            "date_id": [1, 2],
            "time_id": [0, 0],
            "symbol_id": [0, 0],
            "weight": [1.0, 1.0],
            "responder_6": [1.0, 0.0],
            "conservative": [0.0, 0.0],
            "aggressive": [1.0, 10.0],
            "baseline": [0.5, 0.0],
        }
    )

    selected, audit = module.apply_static_previous_fold_selector(
        frame,
        strategies=(("conservative", "conservative"), ("aggressive", "aggressive"), ("baseline", "baseline")),
        default_strategy="conservative",
        output="selected_prediction",
    )

    assert selected.select(["fold", "_selected_strategy"]).to_dicts() == [
        {"fold": "rw_01", "_selected_strategy": "conservative"},
        {"fold": "rw_02", "_selected_strategy": "aggressive"},
    ]
    assert audit.select(["fold", "source_folds", "uses_current_day_target"]).to_dicts() == [
        {"fold": "rw_01", "source_folds": "", "uses_current_day_target": False},
        {"fold": "rw_02", "source_folds": "rw_01", "uses_current_day_target": False},
    ]


def test_dynamic_scale_calibrator_updates_scale_from_previous_day_only() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01"],
            "date_id": [1, 2],
            "time_id": [0, 0],
            "symbol_id": [0, 0],
            "weight": [1.0, 1.0],
            "responder_6": [2.0, 0.0],
            "prediction": [1.0, 1.0],
        }
    )

    calibrated, audit = module.apply_dynamic_scale_calibrator(
        frame,
        prediction="prediction",
        alpha=1.0,
        forgetting_factor=1.0,
        output="calibrated_prediction",
    )

    assert calibrated.select(["date_id", "calibrated_prediction"]).to_dicts() == [
        {"date_id": 1, "calibrated_prediction": 1.0},
        {"date_id": 2, "calibrated_prediction": 1.5},
    ]
    assert audit.select(["date_id", "update_source_date_id", "uses_current_day_target", "update_is_strictly_past"]).to_dicts() == [
        {"date_id": 1, "update_source_date_id": None, "uses_current_day_target": False, "update_is_strictly_past": True},
        {"date_id": 2, "update_source_date_id": 1, "uses_current_day_target": False, "update_is_strictly_past": True},
    ]


def test_posterior_shrinkage_uses_rls_leverage_before_current_target() -> None:
    module = _load_script_module()
    gateway = module._load_gateway_module()
    dynamic = module._load_dynamic_module()
    frame = pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01"],
            "date_id": [1, 2],
            "time_id": [0, 0],
            "symbol_id": [0, 0],
            "x": [1.0, 1.0],
            "responder_6": [1.0, 1.0],
            "weight": [1.0, 1.0],
        }
    )

    predictions, audit = module._simulate_dynamic_rls_posterior_shrinkage(
        gateway,
        dynamic,
        frame,
        features=("x",),
        precision=np.array([[1.0]], dtype=np.float64),
        rhs=np.array([1.0], dtype=np.float64),
        forgetting_factor=1.0,
        strengths=(3.0,),
        output_prefix="posterior",
    )

    values = predictions.select("posterior_s3_prediction").to_series().to_list()
    assert values == pytest.approx([0.5, 1.0 / np.sqrt(2.5)])
    assert audit.select(["date_id", "update_source_date_id", "update_is_strictly_past"]).to_dicts() == [
        {"date_id": 1, "update_source_date_id": None, "update_is_strictly_past": True},
        {"date_id": 2, "update_source_date_id": 1, "update_is_strictly_past": True},
    ]


def test_risk_modulated_shrinkage_uses_observable_row_risk() -> None:
    module = _load_script_module()
    gateway = module._load_gateway_module()
    dynamic = module._load_dynamic_module()
    frame = pl.DataFrame(
        {
            "fold": ["rw_01"],
            "date_id": [1],
            "time_id": [0],
            "symbol_id": [0],
            "x": [1.0],
            "responder_6": [1.0],
            "weight": [np.e - 1.0],
        }
    )

    predictions, audit = module._simulate_dynamic_rls_risk_modulated_shrinkage(
        gateway,
        dynamic,
        frame,
        features=("x",),
        precision=np.array([[1.0]], dtype=np.float64),
        rhs=np.array([1.0], dtype=np.float64),
        forgetting_factor=1.0,
        strengths=(1.0,),
        profiles=("weight",),
        output_prefix="risk",
    )

    value = predictions.select("risk_weight_s1_prediction").item()
    assert value == pytest.approx(1.0 / np.sqrt(3.0))
    assert audit.select(["date_id", "update_source_date_id", "update_is_strictly_past"]).to_dicts() == [
        {"date_id": 1, "update_source_date_id": None, "update_is_strictly_past": True},
    ]


def test_daily_oracle_selector_is_same_day_upper_bound() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01"],
            "date_id": [1, 2],
            "time_id": [0, 0],
            "symbol_id": [0, 0],
            "weight": [1.0, 1.0],
            "responder_6": [1.0, 0.0],
            "conservative": [0.0, 0.0],
            "aggressive": [1.0, 10.0],
            "baseline": [0.5, 0.0],
        }
    )

    oracle = module.apply_daily_oracle_selector(
        frame,
        strategies=(("conservative", "conservative"), ("aggressive", "aggressive"), ("baseline", "baseline")),
        output="oracle_prediction",
    )

    assert oracle.select(["date_id", "_selected_strategy"]).to_dicts() == [
        {"date_id": 1, "_selected_strategy": "aggressive"},
        {"date_id": 2, "_selected_strategy": "conservative"},
    ]
