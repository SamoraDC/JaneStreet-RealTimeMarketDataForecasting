from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_historical_gateway_validation.py"
    spec = importlib.util.spec_from_file_location("prepare_historical_gateway_validation", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_plan_reports_missing_artifacts_and_freezes_historical_commands(tmp_path: Path) -> None:
    module = _load_script_module()

    plan = module.build_plan(
        max_date_id=1398,
        tabm_output_dir=tmp_path / "tabm",
        tree_output_dir=tmp_path / "tree",
        validation_output_dir=tmp_path / "validation",
    )

    assert plan["status"]["ready_for_validation"] is False
    assert plan["status"]["missing_artifacts"] == ["tabm_predictions", "tree_predictions"]
    assert "--max-date-id 1398" in plan["commands"]["tabm_oof"]
    assert "--max-date-id 1398" in plan["commands"]["tree_oof"]
    assert "--aux-targets responder_0,responder_1,responder_2,responder_3,responder_4,responder_5,responder_7,responder_8" in plan["commands"]["tabm_oof"]
    assert "--engines xgboost,lightgbm" in plan["commands"]["tree_oof"]
    assert "--experiment-name frozen_gateway_candidate_validation_hist_max1398_stage3_protocol" in plan["commands"]["frozen_gateway_validation"]
    assert "run_frozen_gateway_candidate_validation.py" in plan["commands"]["frozen_gateway_validation"]


def test_build_plan_becomes_ready_when_both_prediction_dirs_have_five_folds(tmp_path: Path) -> None:
    module = _load_script_module()
    for family in ["tabm", "tree"]:
        prediction_dir = tmp_path / family / "validation_predictions"
        prediction_dir.mkdir(parents=True)
        for idx in range(1, 6):
            (prediction_dir / f"rw_{idx:02d}.parquet").write_text("placeholder", encoding="utf-8")

    plan = module.build_plan(
        max_date_id=1398,
        tabm_output_dir=tmp_path / "tabm",
        tree_output_dir=tmp_path / "tree",
        validation_output_dir=tmp_path / "validation",
    )

    assert plan["status"]["ready_for_validation"] is True
    assert plan["status"]["missing_artifacts"] == []
