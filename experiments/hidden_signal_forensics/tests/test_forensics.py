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


def test_marchenko_pastur_bounds_match_formula() -> None:
    forensics = _load_module("hidden_signal_forensics_forensics", "forensics.py")

    bounds = forensics.marchenko_pastur_bounds(n_samples=100, n_features=25)

    assert bounds.aspect_ratio == pytest.approx(0.25)
    assert bounds.lambda_minus == pytest.approx(0.25)
    assert bounds.lambda_plus == pytest.approx(2.25)


def test_weighted_univariate_fit_recovers_simple_projection() -> None:
    forensics = _load_module("hidden_signal_forensics_forensics_fit", "forensics.py")
    phi = np.array([1.0, 2.0, 3.0])
    y = 2.0 * phi
    weight = np.array([1.0, 2.0, 3.0])

    alpha = forensics.optimal_univariate_fit(phi, y, weight)
    r2 = forensics.weighted_zero_mean_r2_arrays(y, alpha * phi, weight)

    assert alpha == pytest.approx(2.0)
    assert r2 == pytest.approx(1.0)


def test_average_rank_preserves_ties() -> None:
    forensics = _load_module("hidden_signal_forensics_forensics_rank", "forensics.py")

    ranks = forensics.average_rank(np.array([2.0, 1.0, 2.0, 4.0]))

    assert ranks.tolist() == pytest.approx([1.5, 0.0, 1.5, 3.0])


def test_modulo_screen_uses_train_means_only() -> None:
    runner = _load_module("hidden_signal_forensics_runner", "run_forensic_screen.py")
    train = pl.DataFrame(
        {
            "row_index": [0, 1, 2, 3],
            "sample_index": [0, 1, 2, 3],
            "date_id": [0, 0, 1, 1],
            "time_id": [0, 1, 0, 1],
            "symbol_id": [0, 0, 0, 0],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "responder_6": [1.0, -1.0, 1.0, -1.0],
        }
    )
    valid = pl.DataFrame(
        {
            "row_index": [4, 5],
            "sample_index": [4, 5],
            "date_id": [2, 2],
            "time_id": [0, 1],
            "symbol_id": [0, 0],
            "weight": [1.0, 1.0],
            "responder_6": [1.0, -1.0],
        }
    )

    result = runner.modulo_periodicity_screen(train, valid, modulo_values=(2,))

    assert result["valid_r2"].item() == pytest.approx(1.0)


def test_transformed_feature_views_include_signed_tail_without_bool_subtraction() -> None:
    runner = _load_module("hidden_signal_forensics_runner_transforms", "run_forensic_screen.py")

    views = {name: (train_view, valid_view) for name, train_view, valid_view in runner.transformed_feature_views(np.arange(20.0), np.array([-1.0, 0.0, 19.0, 20.0]))}

    assert "signed_tail05" in views
    train_tail, valid_tail = views["signed_tail05"]
    assert set(train_tail.tolist()) <= {-1.0, 0.0, 1.0}
    assert set(valid_tail.tolist()) <= {-1.0, 0.0, 1.0}
