from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_operational_time_feature_baseline.py"
    spec = importlib.util.spec_from_file_location("run_operational_time_feature_baseline", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summary_uses_additive_global_r2() -> None:
    module = _load_script_module()
    results = pl.DataFrame(
        {
            "weighted_zero_mean_r2": [0.0, 0.5],
            "numerator": [2.0, 1.0],
            "denominator": [2.0, 2.0],
            "train_sample_rows": [10, 20],
            "rows": [100, 200],
        }
    )

    summary = module._summary(results)

    assert summary["global_r2"][0] == pytest.approx(0.25)
