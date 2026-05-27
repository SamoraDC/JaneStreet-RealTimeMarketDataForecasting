from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_frozen_risk_shrink_slice_audit.py"
    spec = importlib.util.spec_from_file_location("run_frozen_risk_shrink_slice_audit", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_compare_prediction_pair_by_slice_uses_weighted_error_components() -> None:
    script = _load_script_module()
    frame = pl.DataFrame(
        {
            "bucket": ["a", "a", "b", "b"],
            "weight": [1.0, 3.0, 1.0, 1.0],
            "responder_6": [1.0, 2.0, 1.0, 2.0],
            "base_prediction": [0.0, 0.0, 0.0, 0.0],
            "candidate_prediction": [1.0, 2.0, 3.0, 0.0],
        }
    )

    result = script.compare_prediction_pair_by_slice(
        frame,
        group_columns=("bucket",),
        baseline_prediction="base_prediction",
        candidate_prediction="candidate_prediction",
    )
    rows = {row["bucket"]: row for row in result.iter_rows(named=True)}

    assert rows["a"]["baseline_numerator"] == pytest.approx(13.0)
    assert rows["a"]["candidate_numerator"] == pytest.approx(0.0)
    assert rows["a"]["delta_r2"] == pytest.approx(1.0)
    assert rows["b"]["baseline_numerator"] == pytest.approx(5.0)
    assert rows["b"]["candidate_numerator"] == pytest.approx(8.0)
    assert rows["b"]["delta_r2"] == pytest.approx(-0.6)


def test_add_diagnostic_buckets_adds_target_free_slice_columns() -> None:
    script = _load_script_module()
    frame = pl.DataFrame(
        {
            "date_id": [0, 0, 0, 0],
            "time_id": [0, 99, 100, 250],
            "symbol_id": [1, 1, 2, 2],
            "weight": [1.0, 2.0, 4.0, 8.0],
            "responder_6": [10.0, -10.0, 5.0, -5.0],
            "base_prediction": [0.1, -0.2, 0.4, -0.8],
            "candidate_prediction": [0.1, -0.1, 0.2, -0.4],
        }
    )

    result = script.add_diagnostic_buckets(
        frame,
        reference_prediction="base_prediction",
        candidate_prediction="candidate_prediction",
        time_bucket_size=100,
    )

    assert result["time_bucket"].to_list() == [0, 0, 1, 2]
    assert set(result["weight_bucket"].to_list()) <= {"p00_p50", "p50_p90", "p90_p99", "p99_p100"}
    assert set(result["reference_abs_pred_bucket"].to_list()) <= {"p00_p50", "p50_p90", "p90_p99", "p99_p100"}
    assert set(result["candidate_delta_abs_bucket"].to_list()) <= {"p00_p50", "p50_p90", "p90_p99", "p99_p100"}
