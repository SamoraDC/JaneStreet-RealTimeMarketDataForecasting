from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_online_tail_variants.py"
    spec = importlib.util.spec_from_file_location("analyze_online_tail_variants", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluate_tail_variant_switches_only_requested_buckets() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "fold": ["f1", "f1", "f1", "f1"],
            "strategy": ["base_ensemble", "base_ensemble", "clock_simplex", "clock_simplex"],
            "weight_bucket": ["q50_q90", "q99_q100", "q50_q90", "q99_q100"],
            "rows": [10, 2, 10, 2],
            "weight_sum": [10.0, 2.0, 10.0, 2.0],
            "numerator": [90.0, 30.0, 99.0, 10.0],
            "denominator": [100.0, 50.0, 100.0, 50.0],
            "weighted_zero_mean_r2": [0.1, 0.4, 0.01, 0.8],
        }
    )

    result = module.evaluate_tail_variant(
        frame,
        base_strategy="base_ensemble",
        candidate_strategy="clock_simplex",
        variant_name="q99_only",
        tail_buckets=("q99_q100",),
    )

    assert result["global_r2"] == pytest.approx(1.0 - 100.0 / 150.0)
    assert result["numerator_improvement"] == pytest.approx(20.0)
    assert result["min_delta_r2"] == pytest.approx(1.0 - 100.0 / 150.0 - (1.0 - 120.0 / 150.0))


def test_parse_variants_requires_named_bucket_sets() -> None:
    module = _load_script_module()

    assert module._parse_variants("q99=q99_q100,q90=q90_q99+q99_q100") == (
        ("q99", ("q99_q100",)),
        ("q90", ("q90_q99", "q99_q100")),
    )
    with pytest.raises(ValueError, match="format"):
        module._parse_variants("q99_q100")
