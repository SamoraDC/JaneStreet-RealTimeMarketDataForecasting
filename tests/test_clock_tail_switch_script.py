from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_clock_tail_switch.py"
    spec = importlib.util.spec_from_file_location("analyze_clock_tail_switch", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_tail_switch_uses_candidate_only_on_requested_bucket() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01", "rw_01", "rw_01"],
            "clock": ["none", "none", "row_activity", "row_activity"],
            "strategy": ["base_ensemble", "base_ensemble", "clock_simplex", "clock_simplex"],
            "weight_bucket": ["q00_q50", "q99_q100", "q00_q50", "q99_q100"],
            "rows": [10, 2, 10, 2],
            "weight_sum": [10.0, 2.0, 10.0, 2.0],
            "numerator": [90.0, 30.0, 99.0, 10.0],
            "denominator": [100.0, 50.0, 100.0, 50.0],
            "weighted_zero_mean_r2": [0.1, 0.4, 0.01, 0.8],
        }
    )

    result = module.evaluate_tail_switch(
        frame,
        base_clock="none",
        base_strategy="base_ensemble",
        candidate_clock="row_activity",
        candidate_strategy="clock_simplex",
        tail_buckets=("q99_q100",),
    )

    assert result["global_r2"] == 1.0 - 100.0 / 150.0
    assert result["min_fold_r2"] == result["global_r2"]
