from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import polars as pl


def _load_submission_module():
    path = Path(__file__).resolve().parents[1] / "submission" / "submission.py"
    spec = importlib.util.spec_from_file_location("jane_street_submission_entrypoint", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_submission_predict_smoke_fallback_contract(monkeypatch) -> None:
    monkeypatch.setenv("JANE_STREET_ALLOW_SMOKE_FALLBACK", "1")
    module = _load_submission_module()
    module._PREDICTOR = None
    test = pl.DataFrame(
        {
            "row_id": [0, 1],
            "date_id": [0, 0],
            "time_id": [0, 0],
            "symbol_id": [0, 1],
            "weight": [1.0, 2.0],
            "feature_00": [0.5, -0.25],
            "feature_01": [1.0, 1.0],
        }
    )

    predictions = module.predict(test, None)

    assert predictions.columns == ["row_id", "responder_6"]
    assert predictions["row_id"].to_list() == [0, 1]
    assert predictions["responder_6"].to_list() == [0.5, -0.25]
