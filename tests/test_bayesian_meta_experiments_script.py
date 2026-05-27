from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_bayesian_meta_experiments.py"
    spec = importlib.util.spec_from_file_location("run_bayesian_meta_experiments", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_stream_online_ridge_updates_after_predicting_day() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "date_id": [1, 2],
            "time_id": [0, 0],
            "symbol_id": [0, 0],
            "x": [1.0, 1.0],
            "responder_6": [1.0, 1.0],
            "weight": [1.0, 1.0],
        }
    )

    metrics, beta = module._stream_online_ridge(
        frame,
        features=("x",),
        precision=np.array([[1.0]], dtype=np.float64),
        rhs=np.array([0.0], dtype=np.float64),
    )

    assert metrics["numerator"] == pytest.approx(1.25)
    assert metrics["weighted_zero_mean_r2"] == pytest.approx(1.0 - 1.25 / 2.0)
    assert beta[0] == pytest.approx(0.5)


def test_stream_expert_averaging_updates_after_predicting_day() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "date_id": [1, 2],
            "time_id": [0, 0],
            "symbol_id": [0, 0],
            "good": [1.0, 1.0],
            "bad": [0.0, 0.0],
            "responder_6": [1.0, 1.0],
            "weight": [1.0, 1.0],
        }
    )

    metrics = module._stream_expert_averaging(
        frame,
        experts=("good", "bad"),
        log_weights=np.zeros(2, dtype=np.float64),
        eta=10.0,
    )

    assert metrics["numerator"] > 0.25
    assert metrics["numerator"] < 0.26
