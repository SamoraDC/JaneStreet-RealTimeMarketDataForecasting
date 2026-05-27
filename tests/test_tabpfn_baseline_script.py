from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_tabpfn_baseline.py"
    spec = importlib.util.spec_from_file_location("run_tabpfn_baseline", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_dotenv_returns_keys_without_values(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    env_path = tmp_path / ".env"
    env_path.write_text("TABPFN_TOKEN=secret-value\nOTHER='quoted'\n# ignored\n", encoding="utf-8")
    monkeypatch.delenv("TABPFN_TOKEN", raising=False)
    monkeypatch.delenv("OTHER", raising=False)

    loaded = module._load_dotenv(env_path)

    assert loaded == ("TABPFN_TOKEN", "OTHER")
    assert "secret-value" not in repr(loaded)


def test_evaluate_prediction_matches_weighted_zero_mean_r2() -> None:
    module = _load_script_module()
    y = np.array([1.0, -2.0], dtype=np.float32)
    pred = np.array([0.5, -1.5], dtype=np.float64)
    weight = np.array([2.0, 1.0], dtype=np.float32)

    result = module._evaluate_prediction(y, pred, weight)

    expected = 1.0 - (2.0 * 0.5**2 + 1.0 * (-0.5) ** 2) / (2.0 * 1.0**2 + 1.0 * (-2.0) ** 2)
    assert result["weighted_zero_mean_r2"] == pytest.approx(expected)


def test_predict_in_batches_concatenates_predictions() -> None:
    module = _load_script_module()

    class Model:
        def predict(self, x):
            return x[:, 0] + 1.0

    x = np.arange(10, dtype=np.float32).reshape(5, 2)

    pred = module._predict_in_batches(Model(), x, batch_size=2)

    assert pred.tolist() == pytest.approx([1.0, 3.0, 5.0, 7.0, 9.0])
