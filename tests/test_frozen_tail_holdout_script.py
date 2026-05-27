from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import polars as pl


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_frozen_tail_holdout.py"
    spec = importlib.util.spec_from_file_location("run_frozen_tail_holdout", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_batch_missing_observability_matches_gateway_batch_mean() -> None:
    module = _load_script_module()
    train = pl.DataFrame(
        {
            "date_id": [1, 1, 1, 1, 1],
            "time_id": [0, 0, 0, 1, 1],
            "symbol_id": [0, 1, 2, 0, 1],
            "feature_00": [None, 1.0, 2.0, None, 5.0],
            "feature_01": [None, None, 2.0, 4.0, 5.0],
        }
    )
    validation = pl.DataFrame(
        {
            "date_id": [1, 1, 1, 1, 1],
            "time_id": [0, 0, 0, 1, 1],
            "batch_missing_frac": [0.5, 0.5, 0.5, 0.25, 0.25],
        }
    )

    result = module.audit_batch_missing_observability(
        train.lazy(),
        validation,
        source_features=("feature_00", "feature_01"),
        start=1,
        end=1,
    )

    assert result["batches"] == 2
    assert result["batch_rows_min"] == 2
    assert result["batch_rows_max"] == 3
    assert result["max_abs_diff"] == 0.0
    assert result["mean_abs_diff"] == 0.0


def test_tail_advantage_audit_reports_equivalence_and_fallback_usage() -> None:
    module = _load_script_module()
    policy = SimpleNamespace(
        group_columns=("clock_bucket",),
        tail_buckets=("q90_q99", "q99_q100"),
        fallback_use_candidate=True,
        parameters=pl.DataFrame(
            {
                "clock_bucket": [0],
                "_tail_rows": [2],
                "_tail_base_numerator": [10.0],
                "_tail_candidate_numerator": [1.0],
                "_tail_use_candidate": [True],
            }
        ),
    )
    validation = pl.DataFrame(
        {
            "clock_bucket": [0, 1, 0],
            "weight_bucket": ["q90_q99", "q99_q100", "q00_q50"],
            "fixed_prediction": [10.0, 20.0, 3.0],
            "advantage_prediction": [10.0, 20.0, 3.0],
        }
    )

    policy_summary = module.summarize_tail_advantage_policy(policy)
    equivalence = module.audit_tail_advantage_equivalence(
        validation,
        policy=policy,
        fixed_prediction="fixed_prediction",
        advantage_prediction="advantage_prediction",
    )

    assert policy_summary["fallback_use_candidate"] is True
    assert policy_summary["use_candidate_parameter_rows"] == 1
    assert policy_summary["use_base_parameter_rows"] == 0
    assert policy_summary["all_parameter_rows_use_candidate"] is True
    assert equivalence["fixed_and_advantage_identical"] is True
    assert equivalence["differing_rows"] == 0
    assert equivalence["validation_tail_groups"] == 2
    assert equivalence["validation_tail_groups_using_candidate"] == 2
    assert equivalence["validation_tail_groups_using_fallback"] == 1
