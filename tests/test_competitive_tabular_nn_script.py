from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_competitive_tabular_nn.py"
    spec = importlib.util.spec_from_file_location("run_competitive_tabular_nn", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_competition_folds_include_recent_and_gap_controls() -> None:
    module = _load_script_module()
    frame = pl.DataFrame({"date_id": list(range(1000)), "time_id": [0] * 1000, "symbol_id": [0] * 1000})
    args = type(
        "Args",
        (),
        {
            "fold_protocol": "competition",
            "competition_valid_window": 100,
            "competition_train_start": 200,
            "competition_gap": 50,
            "max_date_id": -1,
        },
    )()

    folds = module._make_folds(frame.lazy(), args)

    assert [(fold.name, fold.train_start, fold.train_end, fold.valid_start, fold.valid_end) for fold in folds] == [
        ("competition_recent", 200, 899, 900, 999),
        ("competition_gap", 200, 849, 900, 999),
    ]


def test_select_requested_folds_slices_without_renaming() -> None:
    module = _load_script_module()
    folds = [
        module.DateFold("rw_01", 0, 9, 10, 19),
        module.DateFold("rw_02", 10, 19, 20, 29),
        module.DateFold("rw_03", 20, 29, 30, 39),
    ]
    args = type("Args", (), {"fold_start_index": 1, "fold_limit": 1})()

    selected = module._select_requested_folds(folds, args)

    assert [fold.name for fold in selected] == ["rw_02"]


def test_model_lazy_frame_reconstructs_previous_date_official_lag() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "date_id": [0, 1],
            "time_id": [3, 3],
            "symbol_id": [2, 2],
            "weight": [1.0, 2.0],
            "feature_00": [10.0, 20.0],
            "feature_09": [2, 3],
            "responder_0": [0.7, 9.0],
            "responder_3": [0.1, 0.2],
            "responder_6": [1.5, -0.5],
            "responder_7": [0.3, 0.4],
            "responder_8": [0.5, 0.6],
        }
    )

    result = module._model_lazy_frame(
        frame.lazy(),
        1,
        1,
        ("feature_00", "responder_0_lag_1"),
        ("symbol_id", "time_id", "feature_09"),
        ("responder_6", "responder_3", "responder_7", "responder_8"),
    ).collect()

    assert result["responder_0_lag_1"].to_list() == pytest.approx([0.7])
    assert result["feature_00"].to_list() == [20.0]
    assert result["feature_09"].to_list() == [3]


def test_model_lazy_frame_builds_causal_derived_aux_target() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "date_id": [0, 1, 2, 3],
            "time_id": [3, 3, 3, 3],
            "symbol_id": [2, 2, 2, 2],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "feature_00": [10.0, 20.0, 30.0, 40.0],
            "responder_6": [0.1, 0.2, 0.3, 0.4],
            "responder_8": [1.0, 3.0, 5.0, 100.0],
        }
    )

    result = module._model_lazy_frame(
        frame.lazy(),
        2,
        2,
        ("feature_00",),
        ("symbol_id", "time_id"),
        ("responder_6", "responder_8_roll8"),
    ).collect()

    assert result["date_id"].to_list() == [2]
    assert result["responder_8_roll8"].to_list() == pytest.approx([3.0])


def test_model_lazy_frame_builds_target_transform_auxiliaries() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "date_id": [0, 1, 2],
            "time_id": [3, 3, 3],
            "symbol_id": [2, 2, 2],
            "weight": [1.0, 1.0, 1.0],
            "feature_00": [10.0, 20.0, 30.0],
            "responder_6": [-4.0, 0.0, 2.5],
        }
    )

    result = module._model_lazy_frame(
        frame.lazy(),
        0,
        2,
        ("feature_00",),
        ("symbol_id", "time_id"),
        ("responder_6", "responder_6_abs", "responder_6_clip3", "responder_6_sign"),
    ).collect()

    assert result["responder_6_abs"].to_list() == pytest.approx([4.0, 0.0, 2.5])
    assert result["responder_6_clip3"].to_list() == pytest.approx([-3.0, 0.0, 2.5])
    assert result["responder_6_sign"].to_list() == pytest.approx([-1.0, 0.0, 1.0])


def test_select_model_columns_adds_batch_feature_names() -> None:
    module = _load_script_module()
    schema = pl.Schema(
        {
            "feature_00": pl.Float32,
            "feature_01": pl.Float32,
            "feature_09": pl.Int8,
            "responder_6": pl.Float32,
        }
    )

    continuous, categorical = module._select_model_columns(
        schema,
        n_features=3,
        use_official_lags=False,
        include_time_features=False,
        include_weight_feature=False,
        batch_feature_sources=("feature_00",),
        batch_feature_modes=("batch_rank", "batch_demean", "row_missing_frac", "batch_missing_frac"),
    )

    assert "feature_00__batch_rank" in continuous
    assert "feature_00__batch_demean" in continuous
    assert "row_missing_frac" in continuous
    assert "batch_missing_frac" in continuous
    assert "feature_09" in categorical


def test_select_model_columns_rejects_responder_batch_source() -> None:
    module = _load_script_module()
    schema = pl.Schema({"feature_00": pl.Float32, "responder_6": pl.Float32})

    with pytest.raises(ValueError, match="invalid batch feature source"):
        module._select_model_columns(
            schema,
            n_features=1,
            use_official_lags=False,
            include_time_features=False,
            include_weight_feature=False,
            batch_feature_sources=("responder_6",),
            batch_feature_modes=("batch_rank",),
        )


def test_resolve_batch_feature_sources_prefers_requested_sources() -> None:
    module = _load_script_module()

    sources = module._resolve_batch_feature_sources(
        ("feature_00", "feature_01", "feature_00__batch_rank"),
        ("feature_01",),
        ("batch_rank", "batch_missing_frac"),
    )

    assert sources == ("feature_01",)


def test_model_lazy_frame_builds_same_batch_features_without_future_batches() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "date_id": [1, 1, 1, 2],
            "time_id": [5, 5, 6, 5],
            "symbol_id": [1, 2, 1, 1],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "feature_00": [1.0, 3.0, 100.0, 10.0],
            "feature_01": [None, 8.0, 9.0, None],
            "responder_6": [0.1, 0.2, 0.3, 0.4],
        }
    )

    result = module._model_lazy_frame(
        frame.lazy(),
        1,
        1,
        (
            "feature_00",
            "feature_01",
            "feature_00__batch_rank",
            "feature_00__batch_mean",
            "feature_00__batch_demean",
            "feature_00__batch_zscore",
            "row_missing_count",
            "row_missing_frac",
            "batch_missing_frac",
        ),
        ("symbol_id", "time_id"),
        ("responder_6",),
    ).collect()

    same_batch = result.filter(pl.col("time_id") == 5).sort("symbol_id")
    other_batch = result.filter(pl.col("time_id") == 6)

    assert same_batch["feature_00__batch_rank"].to_list() == pytest.approx([-0.5, 0.5])
    assert same_batch["feature_00__batch_mean"].to_list() == pytest.approx([2.0, 2.0])
    assert same_batch["feature_00__batch_demean"].to_list() == pytest.approx([-1.0, 1.0])
    assert same_batch["row_missing_count"].to_list() == pytest.approx([1.0, 0.0])
    assert same_batch["row_missing_frac"].to_list() == pytest.approx([0.5, 0.0])
    assert same_batch["batch_missing_frac"].to_list() == pytest.approx([0.25, 0.25])
    assert other_batch["feature_00__batch_mean"].to_list() == pytest.approx([100.0])
    assert other_batch["feature_00__batch_rank"].to_list() == pytest.approx([-0.5])


def test_model_lazy_frame_computes_batch_features_before_sampling() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "date_id": [1, 1, 1],
            "time_id": [5, 5, 6],
            "symbol_id": [1, 2, 1],
            "weight": [10.0, 1.0, 1.0],
            "feature_00": [1.0, 3.0, 100.0],
            "responder_6": [0.1, 0.2, 0.3],
        }
    )
    plan = module.SamplePlan(
        policy="weight_tail",
        row_count=3,
        sample_max_rows=1,
        high_weight_threshold=10.0,
        high_weight_bps=10_000,
        rest_bps=0,
    )

    result = module._model_lazy_frame(
        frame.lazy(),
        1,
        1,
        ("feature_00", "feature_00__batch_mean", "feature_00__batch_rank"),
        ("symbol_id", "time_id"),
        ("responder_6",),
        sample_plan=plan,
    ).collect()

    assert result["symbol_id"].to_list() == [1]
    assert result["feature_00__batch_mean"].to_list() == pytest.approx([2.0])
    assert result["feature_00__batch_rank"].to_list() == pytest.approx([-0.5])


def test_weight_tail_sample_plan_preserves_high_weight_rows() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "date_id": [0] * 10,
            "time_id": list(range(10)),
            "symbol_id": [0] * 10,
            "weight": [float(i) for i in range(1, 11)],
            "feature_00": [float(i) for i in range(10)],
            "responder_6": [0.1] * 10,
        }
    )

    plan = module._make_sample_plan(
        frame.lazy(),
        0,
        0,
        sample_max_rows=3,
        sample_policy="weight_tail",
        weight_tail_quantile=0.80,
    )
    result = module._model_lazy_frame(
        frame.lazy(),
        0,
        0,
        ("feature_00",),
        ("symbol_id", "time_id"),
        ("responder_6",),
        sample_plan=plan,
        seed=17,
    ).collect()

    assert plan.policy == "weight_tail"
    assert plan.high_weight_threshold is not None
    selected_weights = set(result["weight"].to_list())
    required_tail = set(frame.filter(pl.col("weight") >= plan.high_weight_threshold)["weight"].to_list())
    assert required_tail
    assert required_tail.issubset(selected_weights)
    assert result.height <= 4


def test_tabular_mlp_outputs_one_column_per_target() -> None:
    module = _load_script_module()
    import torch

    model = module.TabularMLP(
        n_continuous=5,
        categorical_cardinalities=[4, 10],
        hidden_size=16,
        depth=2,
        dropout=0.0,
        output_dim=3,
        ensemble_size=2,
    )
    continuous = torch.from_numpy(np.zeros((7, 5), dtype=np.float32))
    categorical = torch.zeros((7, 2), dtype=torch.long)

    output = model(continuous, categorical)

    assert output.shape == (7, 3)
    assert torch.isfinite(output).all()


def test_make_tabm_outputs_independent_ensemble_axis() -> None:
    module = _load_script_module()
    import torch

    model = module.make_tabular_model(
        model_type="tabm",
        n_continuous=5,
        categorical_cardinalities=[4, 10],
        hidden_size=16,
        depth=2,
        dropout=0.0,
        output_dim=3,
        ensemble_size=4,
    )
    continuous = torch.from_numpy(np.zeros((7, 5), dtype=np.float32))
    categorical = torch.zeros((7, 2), dtype=torch.long)

    output = model(continuous, categorical)

    assert output.shape == (7, 4, 3)
    assert torch.isfinite(output).all()


def test_frame_to_batch_ids_groups_date_time_batches() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "date_id": [1, 1, 1, 2, 2],
            "time_id": [0, 0, 1, 0, 0],
            "symbol_id": [3, 4, 3, 3, 4],
        }
    )

    batch_ids = module._frame_to_batch_ids(frame)

    assert batch_ids.tolist() == [0, 0, 1, 2, 2]


def test_make_batch_deepset_requires_and_uses_batch_ids() -> None:
    module = _load_script_module()
    import torch

    model = module.make_tabular_model(
        model_type="batch_deepset",
        n_continuous=5,
        categorical_cardinalities=[4, 10],
        hidden_size=16,
        depth=2,
        dropout=0.0,
        output_dim=3,
        ensemble_size=2,
    )
    continuous = torch.from_numpy(np.zeros((7, 5), dtype=np.float32))
    categorical = torch.zeros((7, 2), dtype=torch.long)
    batch_id = torch.tensor([0, 0, 1, 1, 1, 2, 2], dtype=torch.long)

    with pytest.raises(ValueError, match="requires batch_id"):
        model(continuous, categorical)
    output = model(continuous, categorical, batch_id)

    assert output.shape == (7, 3)
    assert torch.isfinite(output).all()


def test_iter_set_batches_preserves_groups_without_shuffle() -> None:
    module = _load_script_module()
    import torch

    continuous = torch.arange(10, dtype=torch.float32).reshape(10, 1)
    categorical = torch.zeros((10, 0), dtype=torch.long)
    target = torch.zeros((10, 1), dtype=torch.float32)
    weight = torch.ones(10, dtype=torch.float32)
    batch_id = torch.tensor([0, 0, 1, 1, 1, 2, 3, 3, 4, 4], dtype=torch.long)

    batches = list(module._iter_set_batches((continuous, categorical, target, weight, batch_id), 4, shuffle_groups=False))

    assert [batch[-1].tolist() for batch in batches] == [[0, 0], [1, 1, 1, 2], [3, 3, 4, 4]]
    assert torch.cat([batch[0].flatten() for batch in batches]).tolist() == list(range(10))


def test_fit_standardization_uses_zero_mean_target_by_default() -> None:
    module = _load_script_module()
    frame = pl.DataFrame(
        {
            "feature_00": [0.0, 1.0],
            "symbol_id": [0, 1],
            "responder_6": [1.0, 3.0],
            "weight": [1.0, 1.0],
        }
    )

    standardization = module._fit_standardization(
        frame,
        ("feature_00",),
        ("symbol_id",),
        ("responder_6",),
        center_target=False,
    )

    assert standardization.target_mean.tolist() == pytest.approx([0.0])
    assert standardization.target_scale.tolist() == pytest.approx([(5.0) ** 0.5])


def test_weighted_multi_loss_huber_matches_mse_inside_delta() -> None:
    module = _load_script_module()
    import torch

    pred = torch.tensor([[0.1, -0.2], [0.3, 0.4]], dtype=torch.float32)
    target = torch.zeros_like(pred)
    weight = torch.tensor([1.0, 2.0], dtype=torch.float32)

    mse = module._weighted_multi_loss(pred, target, weight, aux_weight=0.5, loss_type="mse")
    huber = module._weighted_multi_loss(pred, target, weight, aux_weight=0.5, loss_type="huber", huber_delta=1.0)

    assert huber.item() == pytest.approx(mse.item())


def test_pointwise_huber_caps_large_error_growth() -> None:
    module = _load_script_module()
    import torch

    pred = torch.tensor([0.0, 3.0], dtype=torch.float32)
    target = torch.zeros_like(pred)

    loss = module._pointwise_regression_loss(pred, target, loss_type="huber", huber_delta=1.0)

    assert loss.tolist() == pytest.approx([0.0, 5.0])


def test_parse_mem_available_gb_reads_linux_meminfo() -> None:
    module = _load_script_module()

    available = module._parse_mem_available_gb("MemTotal:  1000 kB\nMemAvailable:  1048576 kB\n")

    assert available == pytest.approx(1.0)


def test_assert_min_mem_available_allows_disabled_guard() -> None:
    module = _load_script_module()

    module._assert_min_mem_available(0.0, "disabled")


def test_write_prediction_frames_writes_fold_parquet(tmp_path: Path) -> None:
    module = _load_script_module()
    fold = module.DateFold("rw_01", 0, 1, 2, 3)
    frame = pl.DataFrame(
        {
            "date_id": [2],
            "time_id": [0],
            "symbol_id": [1],
            "weight": [1.0],
            "responder_6": [0.5],
            "tabm_prediction": [0.4],
        }
    )

    module._write_prediction_frames([frame], tmp_path, fold)

    written = pl.read_parquet(tmp_path / "rw_01.parquet")
    assert written.select(["fold", "date_id", "time_id", "symbol_id", "tabm_prediction"]).row(0) == (
        "rw_01",
        2,
        0,
        1,
        0.4,
    )
