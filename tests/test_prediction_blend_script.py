from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_prediction_blend.py"
    spec = importlib.util.spec_from_file_location("run_prediction_blend", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_joined_predictions_renames_tree_ensemble(tmp_path: Path) -> None:
    module = _load_script_module()
    tabm_dir = tmp_path / "tabm"
    tree_dir = tmp_path / "tree"
    tabm_dir.mkdir()
    tree_dir.mkdir()
    pl.DataFrame(
        {
            "fold": ["rw_01"],
            "date_id": [1],
            "time_id": [2],
            "symbol_id": [3],
            "weight": [1.0],
            "responder_6": [0.5],
            "tabm_prediction": [0.4],
        }
    ).write_parquet(tabm_dir / "rw_01.parquet")
    pl.DataFrame(
        {
            "fold": ["rw_01"],
            "date_id": [1],
            "time_id": [2],
            "symbol_id": [3],
            "ensemble_prediction": [0.45],
            "xgboost_prediction": [0.46],
        }
    ).write_parquet(tree_dir / "rw_01.parquet")

    joined = module._load_joined_predictions(tabm_dir, tree_dir)

    assert joined.select(["tabm_prediction", "tree_prediction", "xgboost_prediction"]).row(0) == (0.4, 0.45, 0.46)


def test_walk_forward_global_uses_previous_folds_only() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01", "rw_02", "rw_02"],
            "date_id": [1, 1, 2, 2],
            "time_id": [0, 1, 0, 1],
            "symbol_id": [0, 0, 0, 0],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "responder_6": [1.0, 2.0, 3.0, 4.0],
            "tabm_prediction": [1.0, 2.0, 0.0, 0.0],
            "tree_prediction": [0.0, 0.0, 3.0, 4.0],
        }
    )

    blended, rows = module._walk_forward_global(frame, initial_tabm_weight=1.0)

    assert rows[0]["rows"] == 0
    assert rows[1]["rows"] == 2
    assert rows[1]["tabm_weight"] == pytest.approx(1.0)
    assert blended.filter(pl.col("fold") == "rw_02")["walk_forward_global_prediction"].to_list() == pytest.approx([0.0, 0.0])
