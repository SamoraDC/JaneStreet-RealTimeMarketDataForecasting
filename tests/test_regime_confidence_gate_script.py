from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_regime_confidence_gate_validation.py"
    spec = importlib.util.spec_from_file_location("run_regime_confidence_gate_validation", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_gate_strategy_uses_only_eligible_prevalidation_scores() -> None:
    module = _load_script_module()
    scores = {
        "base_ensemble": {"weighted_zero_mean_r2": 0.10},
        "batch_missing_clock_simplex": {"weighted_zero_mean_r2": 0.50},
        "clock_weight_shrink": {"weighted_zero_mean_r2": 0.11},
        "clock_weight_abs_shrink": {"weighted_zero_mean_r2": 0.12},
    }

    selected = module.select_gate_strategy(
        selection_scores=scores,
        base_strategy="base_ensemble",
        eligible_strategies=("clock_weight_shrink", "clock_weight_abs_shrink"),
        min_delta=0.0,
    )

    assert selected == "clock_weight_abs_shrink"


def test_select_gate_strategy_falls_back_to_base_when_delta_is_too_small() -> None:
    module = _load_script_module()
    scores = {
        "base_ensemble": {"weighted_zero_mean_r2": 0.10},
        "clock_weight_shrink": {"weighted_zero_mean_r2": 0.101},
    }

    selected = module.select_gate_strategy(
        selection_scores=scores,
        base_strategy="base_ensemble",
        eligible_strategies=("clock_weight_shrink",),
        min_delta=0.002,
    )

    assert selected == "base_ensemble"


def test_add_selected_gate_prediction_reuses_preselected_column() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "base_prediction": [1.0, 2.0],
            "gate_prediction": [10.0, 20.0],
        }
    )

    scored = module.add_selected_gate_prediction(
        frame,
        selected_strategy="gate",
        strategy_predictions={"base": "base_prediction", "gate": "gate_prediction"},
        output="prediction",
    )

    assert scored["prediction"].to_list() == [10.0, 20.0]


def test_model_disagreement_is_cross_model_range() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "p1": [1.0, 2.0, -1.0],
            "p2": [3.0, -1.0, -4.0],
            "p3": [2.0, 4.0, 0.0],
        }
    )

    scored = module.add_model_disagreement(frame, prediction_columns=("p1", "p2", "p3"))

    assert scored["model_disagreement"].to_list() == [2.0, 5.0, 4.0]


def test_parse_gate_candidates_rejects_unknown_and_deduplicates() -> None:
    module = _load_script_module()

    assert module._parse_gate_candidates("clock_blend,clock_blend,clock_weight_shrink") == (
        "clock_blend",
        "clock_weight_shrink",
    )
    with pytest.raises(ValueError, match="unknown gate candidates"):
        module._parse_gate_candidates("clock_blend,not_a_gate")


def test_fit_and_apply_gate_policies_produce_candidate_outputs() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "responder_6": [1.0, 0.5, -1.0, -0.5],
            "weight": [1.0, 2.0, 1.0, 2.0],
            "ensemble_prediction": [0.8, 0.4, -0.8, -0.4],
            "clock_simplex_prediction": [1.0, 0.6, -1.0, -0.6],
            "clock_bucket": [0, 0, 1, 1],
            "weight_bucket": ["q00_q50", "q50_q90", "q90_q99", "q99_q100"],
            "ensemble_abs_bucket": ["p00_p50", "p00_p50", "p50_p90", "p50_p90"],
            "disagreement_bucket": ["p00_p50", "p00_p50", "p50_p90", "p50_p90"],
            "candidate_delta_bucket": ["p00_p50", "p00_p50", "p50_p90", "p50_p90"],
        }
    )

    policies = module.fit_gate_policies(
        frame,
        candidates=("global_blend", "clock_weight_shrink"),
        min_group_rows=1,
    )
    scored = module.apply_gate_policies(frame, policies)

    assert {policy.name for policy in policies} == {"global_blend", "clock_weight_shrink"}
    assert "global_blend_prediction" in scored.columns
    assert "clock_weight_shrink_prediction" in scored.columns


def test_group_selection_policy_activates_only_improving_groups() -> None:
    module = _load_script_module()
    selection = pl.DataFrame(
        {
            "group": ["a", "a", "b", "b"],
            "responder_6": [1.0, 1.0, 1.0, 1.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "base_prediction": [0.0, 0.0, 1.0, 1.0],
            "candidate_prediction": [1.0, 1.0, 0.0, 0.0],
        }
    )
    validation = pl.DataFrame(
        {
            "group": ["a", "b", "c"],
            "base_prediction": [10.0, 20.0, 30.0],
            "candidate_prediction": [100.0, 200.0, 300.0],
        }
    )

    policy = module.fit_group_selection_policy(
        selection,
        name="group_select_candidate",
        group_columns=("group",),
        base_prediction="base_prediction",
        candidate_prediction="candidate_prediction",
        min_group_rows=1,
    )
    scored = policy.apply(validation)

    assert scored["group_select_candidate_prediction"].to_list() == [100.0, 20.0, 30.0]
    active = policy.parameters.filter(pl.col("_group_gate_use_candidate"))
    assert active["group"].to_list() == ["a"]


def test_group_selection_policy_respects_min_delta() -> None:
    module = _load_script_module()
    selection = pl.DataFrame(
        {
            "group": ["a", "a"],
            "responder_6": [1.0, 1.0],
            "weight": [1.0, 1.0],
            "base_prediction": [0.9, 0.9],
            "candidate_prediction": [1.0, 1.0],
        }
    )

    policy = module.fit_group_selection_policy(
        selection,
        name="group_select_candidate",
        group_columns=("group",),
        base_prediction="base_prediction",
        candidate_prediction="candidate_prediction",
        min_group_rows=1,
        min_delta=0.02,
    )

    assert policy.parameters["_group_gate_use_candidate"].to_list() == [False]
