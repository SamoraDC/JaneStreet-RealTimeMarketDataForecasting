from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from multimodels.strong_family_bridge import (
    add_walk_forward_alpha_stack,
    add_walk_forward_family_regime_scale,
    add_walk_forward_family_residual_bridge,
    add_walk_forward_family_residual_gate,
    add_walk_forward_family_residual_tail_masks,
    add_walk_forward_family_risk_shrinkage,
)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01", "rw_02", "rw_02", "rw_03", "rw_03"],
            "date_id": [1, 1, 2, 2, 3, 3],
            "time_id": [0, 100, 0, 100, 0, 100],
            "symbol_id": [0, 1, 0, 1, 0, 1],
            "weight": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            "responder_6": [1.0, -1.0, 1.2, -1.2, 1.4, -1.4],
            "strong": [0.8, -0.8, 0.8, -0.8, 0.8, -0.8],
            "weak_alpha": [0.1, -0.1, 0.2, -0.2, 0.3, -0.3],
            "risk_score": [0.5, 2.0, 0.5, 2.0, 0.5, 2.0],
        }
    )


def test_alpha_stack_first_fold_is_identity_and_later_uses_history() -> None:
    preds, params = add_walk_forward_alpha_stack(_frame(), base_prediction="strong", alpha_columns=("weak_alpha",), alpha=1.0)

    first = preds.filter(pl.col("fold") == "rw_01")["strong_family_alpha_stack_prediction"].to_numpy()

    assert first.tolist() == pytest.approx([0.8, -0.8])
    assert any(row["fold"] == "rw_02" and row["fit_rows"] == 2 for row in params)


def test_family_residual_bridge_first_fold_is_identity() -> None:
    preds, params = add_walk_forward_family_residual_bridge(_frame(), base_prediction="strong", feature_columns=("weak_alpha",), alpha=1.0)

    col = "strong_family_residual_prediction"
    first = preds.filter(pl.col("fold") == "rw_01")[col].to_numpy()
    second = preds.filter(pl.col("fold") == "rw_02")[col].to_numpy()

    assert first.tolist() == pytest.approx([0.8, -0.8])
    assert not np.allclose(second, np.array([0.8, -0.8], dtype=np.float64))
    assert any(row["component"] == "family_residual" for row in params)


def test_family_risk_shrinkage_shrinks_after_first_fold_only() -> None:
    preds, _ = add_walk_forward_family_risk_shrinkage(_frame(), base_prediction="strong", risk_columns=("risk_score",), strengths=(0.5,))
    col = "strong__risk_score_s0p5_family_risk_shrink_prediction"

    first = preds.filter(pl.col("fold") == "rw_01")[col].to_numpy()
    second = preds.filter(pl.col("fold") == "rw_02")[col].to_numpy()

    assert first.tolist() == pytest.approx([0.8, -0.8])
    assert np.all(np.abs(second) < np.array([0.8, 0.8]))


def test_family_regime_scale_first_fold_is_identity() -> None:
    preds, params = add_walk_forward_family_regime_scale(
        _frame(),
        base_prediction="strong",
        risk_columns=("risk_score",),
        strengths=(0.1,),
        time_bucket_size=100,
        symbol_mod=2,
        min_rows=1,
        prior_strength=1.0,
    )
    col = "strong__risk_score_s0p1_family_regime_scaled_prediction"
    first = preds.filter(pl.col("fold") == "rw_01")[col].to_numpy()

    assert first.tolist() == pytest.approx([0.8, -0.8])
    assert any(row["component"] == "family_regime_scale" and row["fold"] == "rw_02" for row in params)


def test_family_residual_gate_uses_previous_fold_group_history() -> None:
    frame = pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01", "rw_02", "rw_02"],
            "date_id": [1, 1, 2, 2],
            "time_id": [0, 0, 0, 0],
            "symbol_id": [0, 1, 0, 1],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "responder_6": [1.0, -1.0, 1.0, -1.0],
            "strong": [0.0, -0.8, 0.0, -0.8],
            "strong_family_residual_prediction": [1.0, 0.8, 1.0, 0.8],
            "risk_score": [1.0, 1.0, 1.0, 1.0],
        }
    )

    preds, params = add_walk_forward_family_residual_gate(
        frame,
        base_prediction="strong",
        residual_prediction="strong_family_residual_prediction",
        risk_column="risk_score",
        time_bucket_size=100,
        symbol_mod=2,
        min_rows=1,
        prior_strength=0.0,
        min_delta=0.0,
    )

    col = "strong_family_residual_gate_closed_prediction"
    first = preds.filter(pl.col("fold") == "rw_01")[col].to_numpy()
    second = preds.filter(pl.col("fold") == "rw_02").sort("symbol_id")[col].to_numpy()

    assert first.tolist() == pytest.approx([0.0, -0.8])
    assert second.tolist() == pytest.approx([1.0, -0.8])
    assert any(row["component"] == "family_residual_gate" and row["n_open_groups"] == 1 for row in params)


def test_family_residual_gate_open_defaults_to_residual_after_first_fold() -> None:
    frame = _frame().with_columns(
        (pl.col("strong") + 0.1 * pl.col("weak_alpha")).alias("strong_family_residual_prediction")
    )

    preds, _ = add_walk_forward_family_residual_gate(
        frame,
        base_prediction="strong",
        residual_prediction="strong_family_residual_prediction",
        risk_column="risk_score",
        time_bucket_size=100,
        symbol_mod=2,
        min_rows=10,
        prior_strength=0.0,
        min_delta=0.0,
    )

    col = "strong_family_residual_gate_open_prediction"
    second = preds.filter(pl.col("fold") == "rw_02")[col].to_numpy()
    expected = frame.filter(pl.col("fold") == "rw_02")["strong_family_residual_prediction"].to_numpy()

    assert second.tolist() == pytest.approx(expected.tolist())


def test_family_residual_tail_mask_uses_previous_fold_thresholds() -> None:
    frame = pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01", "rw_02", "rw_02"],
            "date_id": [1, 1, 2, 2],
            "time_id": [0, 1, 0, 1],
            "symbol_id": [0, 1, 0, 1],
            "weight": [1.0, 10.0, 1.0, 10.0],
            "responder_6": [0.0, 0.0, 0.0, 0.0],
            "strong": [0.0, 0.0, 0.0, 0.0],
            "strong_family_residual_prediction": [1.0, 2.0, 1.0, 2.0],
            "risk_score": [0.0, 0.0, 0.0, 0.0],
        }
    )

    preds, params = add_walk_forward_family_residual_tail_masks(
        frame,
        base_prediction="strong",
        residual_prediction="strong_family_residual_prediction",
        risk_column="risk_score",
        quantiles=(0.5,),
    )

    col = "strong_family_residual_weight_q0p5_family_residual_tail_prediction"
    first = preds.filter(pl.col("fold") == "rw_01").sort("time_id")[col].to_numpy()
    second = preds.filter(pl.col("fold") == "rw_02").sort("time_id")[col].to_numpy()

    assert first.tolist() == pytest.approx([0.0, 0.0])
    assert second.tolist() == pytest.approx([0.0, 2.0])
    assert any(row["component"] == "family_residual_tail" and row["selected_rows"] == 1 for row in params)
