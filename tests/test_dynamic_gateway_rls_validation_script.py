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


def test_forgetting_one_matches_existing_gateway_online_ridge() -> None:
    dynamic = _load_script_module("run_dynamic_gateway_rls_validation", "scripts/run_dynamic_gateway_rls_validation.py")
    gateway = _load_script_module("run_bayesian_gateway_meta_simulation_dynamic_test", "scripts/run_bayesian_gateway_meta_simulation.py")
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

    dynamic_metrics, dynamic_beta, audit = dynamic._simulate_dynamic_gateway_rls(
        gateway,
        frame,
        features=("x",),
        precision=precision.copy(),
        rhs=rhs.copy(),
        forgetting_factor=1.0,
    )
    gateway_metrics, gateway_beta, _audit = gateway._simulate_gateway_online_ridge(
        frame,
        features=("x",),
        precision=precision.copy(),
        rhs=rhs.copy(),
    )

    assert dynamic_metrics["numerator"] == pytest.approx(gateway_metrics["numerator"])
    assert dynamic_beta == pytest.approx(gateway_beta)
    assert audit["update_is_strictly_past"].to_list() == [True, True, True]


def test_forgetting_update_decays_previous_precision_and_rhs() -> None:
    dynamic = _load_script_module("run_dynamic_gateway_rls_validation_decay", "scripts/run_dynamic_gateway_rls_validation.py")
    precision = np.array([[10.0]], dtype=np.float64)
    rhs = np.array([5.0], dtype=np.float64)
    frame = pl.DataFrame({"x": [2.0], "responder_6": [3.0], "weight": [4.0]})

    updated_precision, updated_rhs = dynamic._forgetting_update(
        precision,
        rhs,
        frame,
        features=("x",),
        forgetting_factor=0.5,
    )

    assert updated_precision[0, 0] == pytest.approx(0.5 * 10.0 + 4.0 * 2.0 * 2.0)
    assert updated_rhs[0] == pytest.approx(0.5 * 5.0 + 4.0 * 2.0 * 3.0)


def test_forgetting_factors_are_bounded() -> None:
    dynamic = _load_script_module("run_dynamic_gateway_rls_validation_bounds", "scripts/run_dynamic_gateway_rls_validation.py")

    assert dynamic._parse_forgetting_factors("1.0,0.999") == [1.0, 0.999]
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        dynamic._parse_forgetting_factors("1.001")
    with pytest.raises(ValueError, match="in \\(0, 1\\]"):
        dynamic._parse_forgetting_factors("0")
