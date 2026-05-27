from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_clock_property_comparison.py"
    spec = importlib.util.spec_from_file_location("run_clock_property_comparison", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cv_returns_zero_for_constant_positive_values() -> None:
    module = _load_script_module()

    assert module._cv(np.array([2.0, 2.0, 2.0])) == pytest.approx(0.0)


def test_clock_columns_include_cross_sectional_batch_clocks() -> None:
    module = _load_script_module()

    assert "batch_activity_rank_bucket" in module._clock_columns()
    assert "batch_weight_rank_bucket" in module._clock_columns()
