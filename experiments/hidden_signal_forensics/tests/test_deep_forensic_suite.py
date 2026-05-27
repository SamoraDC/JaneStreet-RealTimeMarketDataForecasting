from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative: str):
    path = EXPERIMENT_DIR / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_iaaft_preserves_sample_distribution() -> None:
    suite = _load_module("hidden_signal_deep_suite_iaaft", "run_deep_forensic_suite.py")
    rng = np.random.default_rng(7)
    values = np.array([3.0, -1.0, 2.0, 2.0, 9.0, -4.0, 0.5])

    surrogate = suite.iaaft(values, iterations=5, rng=rng)

    assert np.sort(surrogate).tolist() == pytest.approx(np.sort(values).tolist())


def test_null_target_preserves_values_for_all_null_kinds() -> None:
    suite = _load_module("hidden_signal_deep_suite_nulls", "run_deep_forensic_suite.py")
    rng = np.random.default_rng(11)
    frame = pl.DataFrame(
        {
            "date_id": [0, 0, 1, 1, 2, 2],
            "responder_6": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    for kind in ("iid", "block", "date", "circular"):
        null = suite.null_target(frame, kind=kind, rng=rng, block_size=2)
        assert sorted(null.tolist()) == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_interaction_views_include_xor_like_sign_product() -> None:
    suite = _load_module("hidden_signal_deep_suite_interactions", "run_deep_forensic_suite.py")
    left = {
        "train_sign": np.array([1.0, -1.0]),
        "valid_sign": np.array([1.0, -1.0]),
        "train_rank": np.array([0.5, -0.5]),
        "valid_rank": np.array([0.5, -0.5]),
        "train_z": np.array([2.0, -1.0]),
        "valid_z": np.array([1.0, -2.0]),
    }
    right = {
        "train_sign": np.array([-1.0, -1.0]),
        "valid_sign": np.array([1.0, -1.0]),
        "train_rank": np.array([-0.5, -0.5]),
        "valid_rank": np.array([0.5, -0.5]),
        "train_z": np.array([-3.0, -1.0]),
        "valid_z": np.array([4.0, -2.0]),
    }

    views = {name: (train_view, valid_view) for name, train_view, valid_view in suite.interaction_views(left, right)}

    assert views["sign_product"][0].tolist() == pytest.approx([-1.0, 1.0])
    assert "rank_diff" in views
    assert "z_product" in views


def test_cross_sectional_columns_are_group_local_and_target_free() -> None:
    suite = _load_module("hidden_signal_deep_suite_cross_sectional", "run_deep_forensic_suite.py")
    frame = pl.DataFrame(
        {
            "date_id": [0, 0, 0, 0],
            "time_id": [0, 0, 1, 1],
            "symbol_id": [0, 1, 0, 1],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "responder_6": [10.0, -10.0, 5.0, -5.0],
            "feature_00": [1.0, 3.0, 10.0, 14.0],
        }
    )

    result = suite.add_cross_sectional_columns(frame, ["feature_00"], ["date_id", "time_id"])

    assert "feature_00_cs_z_date_time" in result.columns
    assert "feature_00_cs_rank_date_time" in result.columns
    assert result["feature_00_cs_rank_date_time"].to_list() == pytest.approx([-0.5, 0.5, -0.5, 0.5])


def test_residual_mining_uses_matching_train_and_valid_baseline_predictions() -> None:
    suite = _load_module("hidden_signal_deep_suite_residual", "run_deep_forensic_suite.py")
    train = pl.DataFrame(
        {
            "feature_00": [-1.0, 0.0, 1.0, 2.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "responder_6": [-1.0, 0.0, 1.0, 2.0],
        }
    )
    valid = pl.DataFrame(
        {
            "feature_00": [-2.0, 3.0],
            "weight": [1.0, 1.0],
            "responder_6": [-2.0, 3.0],
        }
    )
    baseline_predictions = {
        "bad": (np.zeros(4), np.zeros(2)),
        "good": (train["responder_6"].to_numpy(), valid["responder_6"].to_numpy()),
    }

    result = suite.residual_mining(train, valid, ["feature_00"], baseline_predictions)

    assert result.row(0, named=True)["baseline_model"] == "good"
    assert result.row(0, named=True)["baseline_valid_r2"] == pytest.approx(1.0)
