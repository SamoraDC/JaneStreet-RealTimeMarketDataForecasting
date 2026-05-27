from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_online_tail_control_validation.py"
    spec = importlib.util.spec_from_file_location("run_online_tail_control_validation", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_make_online_tail_folds_are_ordered_and_disjoint() -> None:
    module = _load_script_module()

    folds = module.make_online_tail_folds(
        min_date_id=0,
        max_date_id=99,
        n_folds=2,
        train_window=20,
        selection_window=5,
        valid_window=10,
        gap=1,
    )

    assert [fold.name for fold in folds] == ["otf_01", "otf_02"]
    assert folds[0].train_start == 53
    assert folds[0].train_end == 72
    assert folds[0].selection_start == 74
    assert folds[0].selection_end == 78
    assert folds[0].valid_start == 80
    assert folds[0].valid_end == 89
    assert all(module._fold_is_ordered(fold) for fold in folds)


def test_make_online_tail_folds_rejects_impossible_window() -> None:
    module = _load_script_module()

    with pytest.raises(ValueError, match="not enough dates"):
        module.make_online_tail_folds(
            min_date_id=0,
            max_date_id=10,
            n_folds=2,
            train_window=20,
            selection_window=5,
            valid_window=10,
        )


def test_validate_date_bounds_rejects_out_of_dataset_range() -> None:
    module = _load_script_module()

    with pytest.raises(ValueError, match="before the dataset start"):
        module._validate_date_bounds(
            dataset_min_date_id=10,
            dataset_max_date_id=20,
            fold_min_date_id=9,
            fold_max_date_id=20,
        )
    with pytest.raises(ValueError, match="after the dataset end"):
        module._validate_date_bounds(
            dataset_min_date_id=10,
            dataset_max_date_id=20,
            fold_min_date_id=10,
            fold_max_date_id=21,
        )
    with pytest.raises(ValueError, match="must be <="):
        module._validate_date_bounds(
            dataset_min_date_id=10,
            dataset_max_date_id=20,
            fold_min_date_id=15,
            fold_max_date_id=14,
        )


def test_selection_gated_prediction_uses_only_pre_validation_decision() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "base_prediction": [1.0, 2.0],
            "tail_prediction": [10.0, 20.0],
        }
    )

    passed = module.add_selection_gated_prediction(
        frame,
        selection_passed=True,
        base_prediction="base_prediction",
        tail_prediction="tail_prediction",
        output="prediction",
    )
    failed = module.add_selection_gated_prediction(
        frame,
        selection_passed=False,
        base_prediction="base_prediction",
        tail_prediction="tail_prediction",
        output="prediction",
    )

    assert passed["prediction"].to_list() == [10.0, 20.0]
    assert failed["prediction"].to_list() == [1.0, 2.0]


def test_batch_missing_observability_uses_current_gateway_batch_only() -> None:
    module = _load_script_module()
    train = pl.DataFrame(
        {
            "date_id": [7, 7, 7, 7],
            "time_id": [0, 0, 1, 1],
            "symbol_id": [1, 2, 1, 2],
            "feature_00": [None, 1.0, None, None],
            "feature_01": [2.0, None, None, 3.0],
        }
    )
    validation = pl.DataFrame(
        {
            "date_id": [7, 7, 7, 7],
            "time_id": [0, 0, 1, 1],
            "batch_missing_frac": [0.5, 0.5, 0.75, 0.75],
        }
    )

    result = module.audit_batch_missing_observability(
        train.lazy(),
        validation,
        source_features=("feature_00", "feature_01"),
        start=7,
        end=7,
    )

    assert result["batches"] == 2
    assert result["batch_rows_min"] == 2
    assert result["batch_rows_max"] == 2
    assert result["max_abs_diff"] == 0.0
