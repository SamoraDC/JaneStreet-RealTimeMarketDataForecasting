import importlib.util
import sys
from pathlib import Path

import polars as pl


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_ridge_gbdt_blend.py"
    spec = importlib.util.spec_from_file_location("run_ridge_gbdt_blend", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_add_regime_columns_adds_all_slice_columns():
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "time_id": [0, 99, 100, 250],
            "weight": [0.1, 0.7, 2.0, 5.0],
            "ridge_prediction": [0.01, 0.2, 1.2, -3.0],
        }
    )

    result = module._add_regime_columns(
        frame,
        {"q50": 0.5, "q90": 2.5, "q99": 4.0},
        {"q50": 0.1, "q90": 1.0, "q99": 2.0},
        time_bucket_size=100,
    )

    assert result["time_bucket"].to_list() == [0, 0, 1, 2]
    assert set(result["weight_bucket"]) == {"q00_q50", "q50_q90", "q99_q100"}
    assert set(result["prediction_abs_bucket"]) == {"p00_p50", "p50_p90", "p90_p99", "p99_p100"}


def test_parse_gbdt_seeds_uses_random_state_fallback():
    module = _load_script_module()

    assert module._parse_gbdt_seeds("", 17) == (17,)
    assert module._parse_gbdt_seeds("17, 23,37", 17) == (17, 23, 37)


def test_parse_blend_group_columns_trims_empty_values():
    module = _load_script_module()

    assert module._parse_blend_group_columns("weight_bucket, prediction_abs_bucket,") == (
        "weight_bucket",
        "prediction_abs_bucket",
    )


def test_parse_gbdt_id_columns_deduplicates_values():
    module = _load_script_module()

    assert module._parse_gbdt_id_columns("time_id, symbol_id, time_id") == ("time_id", "symbol_id")
