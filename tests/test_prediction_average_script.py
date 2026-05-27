from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_prediction_average.py"
    spec = importlib.util.spec_from_file_location("build_prediction_average", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_prediction(path: Path, predictions: list[float]) -> None:
    pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01"],
            "date_id": [1, 1],
            "time_id": [0, 1],
            "symbol_id": [7, 7],
            "weight": [1.0, 2.0],
            "responder_6": [1.0, -1.0],
            "tabm_prediction": predictions,
        }
    ).write_parquet(path)


def test_build_fold_average_preserves_keys_and_averages_predictions(tmp_path: Path) -> None:
    module = _load_script_module()
    seed_a = tmp_path / "seed_a"
    seed_b = tmp_path / "seed_b"
    seed_a.mkdir()
    seed_b.mkdir()
    _write_prediction(seed_a / "rw_01.parquet", [0.0, 2.0])
    _write_prediction(seed_b / "rw_01.parquet", [2.0, 0.0])

    averaged = module._build_fold_average(
        [seed_a / "rw_01.parquet", seed_b / "rw_01.parquet"],
        prediction_column="tabm_prediction",
        output_column="tabm_prediction",
    )

    assert averaged.select(["fold", "date_id", "time_id", "symbol_id"]).rows() == [
        ("rw_01", 1, 0, 7),
        ("rw_01", 1, 1, 7),
    ]
    assert averaged["tabm_prediction"].to_list() == pytest.approx([1.0, 1.0])


def test_fold_files_rejects_mismatched_fold_sets(tmp_path: Path) -> None:
    module = _load_script_module()
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _write_prediction(first / "rw_01.parquet", [0.0, 0.0])
    _write_prediction(second / "rw_02.parquet", [0.0, 0.0])

    with pytest.raises(ValueError, match="do not share fold"):
        module._fold_files([first, second])
