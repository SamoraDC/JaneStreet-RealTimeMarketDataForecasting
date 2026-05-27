from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_frozen_gateway_candidate_validation.py"
    spec = importlib.util.spec_from_file_location("run_frozen_gateway_candidate_validation", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_candidates_are_exactly_the_two_selected_before_validation() -> None:
    script = _load_script_module()

    candidates = script._frozen_candidates()

    assert [candidate.name for candidate in candidates] == [
        "components_no_tree_ensemble_alpha1000",
        "experts_alpha10000",
    ]
    assert [(candidate.feature_set, candidate.ridge_alpha) for candidate in candidates] == [
        ("components_no_tree_ensemble", 1000.0),
        ("experts", 10000.0),
    ]


def test_validate_candidates_rejects_implicit_candidate_search() -> None:
    script = _load_script_module()
    candidates = script._frozen_candidates()

    with pytest.raises(ValueError, match="exactly two candidates"):
        script._validate_candidates(candidates[:1], {"components_no_tree_ensemble": ("x",)})

    with pytest.raises(ValueError, match="missing feature sets"):
        script._validate_candidates(candidates, {"components_no_tree_ensemble": ("x",)})


def test_subset_summary_uses_metric_numerators_not_mean_of_fold_r2() -> None:
    script = _load_script_module()
    by_fold = pl.DataFrame(
        {
            "strategy": ["a", "a", "b", "b"],
            "method_family": ["m", "m", "m", "m"],
            "fold": ["rw_01", "rw_02", "rw_01", "rw_02"],
            "rows": [1, 1, 1, 1],
            "weight_sum": [1.0, 1.0, 1.0, 1.0],
            "numerator": [1.0, 10.0, 2.0, 4.0],
            "denominator": [10.0, 100.0, 10.0, 100.0],
            "weighted_zero_mean_r2": [0.9, 0.9, 0.8, 0.96],
        }
    )

    summary = script.summarize_candidate_subsets(by_fold, {"full": ("rw_01", "rw_02")})
    rows = {row["strategy"]: row for row in summary.iter_rows(named=True)}

    assert rows["a"]["global_r2"] == pytest.approx(1.0 - 11.0 / 110.0)
    assert rows["b"]["global_r2"] == pytest.approx(1.0 - 6.0 / 110.0)
    assert rows["b"]["global_r2"] > rows["a"]["global_r2"]
