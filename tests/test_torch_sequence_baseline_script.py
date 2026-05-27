from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_torch_sequence_baseline.py"
    spec = importlib.util.spec_from_file_location("run_torch_sequence_baseline", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_make_sequence_arrays_left_pads_within_symbol_only() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "date_id": [1, 1, 1, 1],
            "time_id": [0, 1, 0, 1],
            "symbol_id": [0, 0, 1, 1],
            "feature_00": [1.0, 2.0, 10.0, 20.0],
            "responder_6": [0.1, 0.2, 1.0, 2.0],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    standardization = module.Standardization(
        feature_mean=np.array([0.0], dtype=np.float32),
        feature_scale=np.array([1.0], dtype=np.float32),
        target_mean=0.0,
        target_scale=1.0,
    )

    seq, target, weight = module._make_sequence_arrays(
        frame,
        ["feature_00"],
        sequence_length=3,
        target_start=1,
        target_end=1,
        standardization=standardization,
    )

    assert seq.shape == (4, 3, 1)
    assert seq[:, :, 0].tolist() == [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 2.0],
        [0.0, 0.0, 10.0],
        [0.0, 10.0, 20.0],
    ]
    assert target.tolist() == pytest.approx([0.1, 0.2, 1.0, 2.0])
    assert weight.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])


def test_collect_frame_reconstructs_previous_date_lag_and_weight_feature() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "date_id": [0, 1],
            "time_id": [3, 3],
            "symbol_id": [2, 2],
            "feature_00": [10.0, 20.0],
            "responder_0": [0.7, 9.0],
            "responder_6": [1.5, -0.5],
            "weight": [1.0, 2.0],
        }
    )

    result = module._collect_frame(
        frame.lazy(),
        1,
        1,
        ("feature_00", "weight_feature", "responder_0_lag_1", "time_id", "symbol_id"),
    )

    assert result["feature_00"].to_list() == pytest.approx([20.0])
    assert result["weight_feature"].to_list() == pytest.approx([2.0])
    assert result["responder_0_lag_1"].to_list() == pytest.approx([0.7])


def test_parse_models_rejects_unknown_names() -> None:
    module = _load_script_module()
    with pytest.raises(ValueError, match="unknown models"):
        module._parse_models("lstm,not_a_model")


def test_sophia_g_updates_parameter_without_nan() -> None:
    module = _load_script_module()
    import torch

    param = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = module.SophiaG([param], lr=0.01, update_period=1)
    loss = (param - 0.0).pow(2).sum()
    loss.backward()

    optimizer.step()

    assert torch.isfinite(param).all()
    assert float(param.detach()[0]) < 1.0
