from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from multimodels.strong_oof import (
    StrongOOFConfig,
    add_fixed_candidate_blends,
    add_online_daily_affine,
    add_prediction_context,
    add_prediction_risk_shrinkage,
    add_raw_preprocessing_features,
    _add_prediction_expert_expansions,
    add_online_daily_scales,
    _additional_prediction_experts,
    add_walk_forward_candidate_blends,
    add_walk_forward_contextual_candidate_blends,
    add_walk_forward_ridge_stack_candidates,
    add_walk_forward_ridge_stack,
    add_walk_forward_regime_scales,
    add_walk_forward_residual_tail_masks,
    load_joined_predictions,
    _assert_resource_floor,
    _parse_meminfo_gb,
    score_walk_forward_contextual_candidate_blends,
)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01", "rw_02", "rw_02"],
            "date_id": [1, 1, 2, 2],
            "time_id": [0, 1, 0, 1],
            "symbol_id": [0, 0, 0, 0],
            "weight": [1.0, 2.0, 1.0, 2.0],
            "responder_6": [1.0, -1.0, 2.0, -2.0],
            "tabm_prediction": [0.8, -0.8, 1.8, -1.8],
            "tree_prediction": [0.6, -0.4, 1.2, -1.0],
        }
    )


def test_prediction_context_adds_disagreement_without_target() -> None:
    frame = add_prediction_context(_frame())

    assert "prediction_disagreement" in frame.columns
    assert "tabm_tree_diff" in frame.columns
    assert "responder_6" not in {"prediction_disagreement", "tabm_tree_diff"}
    assert frame["prediction_disagreement"].min() >= 0.0


def test_load_joined_predictions_adds_prefixed_extra_predictions(tmp_path) -> None:
    tabm_dir = tmp_path / "tabm"
    tree_dir = tmp_path / "tree"
    extra_dir = tmp_path / "extra"
    tabm_dir.mkdir()
    tree_dir.mkdir()
    extra_dir.mkdir()
    keys = {"fold": ["rw_01"], "date_id": [1], "time_id": [0], "symbol_id": [7]}
    pl.DataFrame({**keys, "weight": [1.0], "responder_6": [0.5], "tabm_prediction": [0.1]}).write_parquet(tabm_dir / "rw_01.parquet")
    pl.DataFrame({**keys, "ensemble_prediction": [0.2], "xgboost_prediction": [0.3]}).write_parquet(tree_dir / "rw_01.parquet")
    pl.DataFrame(
        {
            **keys,
            "tabm_prediction": [0.35],
            "ensemble_prediction": [0.4],
            "catboost_prediction": [0.5],
            "latent_alpha_linear_stack": [0.6],
            "ridge_rank_alpha10000": [0.7],
            "pls_rank_k8": [0.8],
        }
    ).write_parquet(extra_dir / "rw_01.parquet")

    result = load_joined_predictions(
        tabm_dir,
        tree_dir,
        extra_prediction_dirs=(extra_dir,),
        extra_prediction_prefixes=("residual_tree",),
    )

    assert result["tree_prediction"].to_list() == pytest.approx([0.2])
    assert result["residual_tree_tabm_prediction"].to_list() == pytest.approx([0.35])
    assert result["residual_tree_ensemble_prediction"].to_list() == pytest.approx([0.4])
    assert result["residual_tree_catboost_prediction"].to_list() == pytest.approx([0.5])
    assert result["residual_tree_latent_alpha_linear_stack"].to_list() == pytest.approx([0.6])
    assert result["residual_tree_ridge_rank_alpha10000"].to_list() == pytest.approx([0.7])
    assert result["residual_tree_pls_rank_k8"].to_list() == pytest.approx([0.8])


def test_additional_prediction_experts_include_custom_prefixed_oof_predictions() -> None:
    frame = pl.DataFrame(
        {
            "tabm_prediction": [0.1],
            "tree_prediction": [0.2],
            "tabm_s23_aux8_tabm_prediction": [0.3],
            "tabm_s17_tabm_prediction": [0.4],
            "gateway_risk_conservative_rls_prediction": [0.5],
            "fixed_blend_0_w0p5_fixed_blend_prediction": [0.6],
            "prediction_disagreement": [0.7],
        }
    )

    experts = _additional_prediction_experts(frame, ("tabm_prediction", "tree_prediction"))

    assert experts == ("tabm_s23_aux8_tabm_prediction", "tabm_s17_tabm_prediction")


def test_prediction_expert_expansions_are_target_free_transforms() -> None:
    frame = pl.DataFrame({"tabm_prediction": [2.0, -3.0], "responder_6": [10.0, -10.0]})

    expanded, names = _add_prediction_expert_expansions(
        frame,
        ("tabm_prediction",),
        ("signed_square", "abs", "signed_sqrt", "signed_log1p", "cube", "sign"),
    )

    assert names == (
        "tabm_prediction__signed_square",
        "tabm_prediction__abs",
        "tabm_prediction__signed_sqrt",
        "tabm_prediction__signed_log1p",
        "tabm_prediction__cube",
        "tabm_prediction__sign",
    )
    assert expanded["tabm_prediction__signed_square"].to_list() == pytest.approx([4.0, -9.0])
    assert expanded["tabm_prediction__abs"].to_list() == pytest.approx([2.0, 3.0])
    assert expanded["tabm_prediction__signed_sqrt"].to_list() == pytest.approx([np.sqrt(2.0), -np.sqrt(3.0)])
    assert expanded["tabm_prediction__signed_log1p"].to_list() == pytest.approx([np.log1p(2.0), -np.log1p(3.0)])
    assert expanded["tabm_prediction__cube"].to_list() == pytest.approx([8.0, -27.0])
    assert expanded["tabm_prediction__sign"].to_list() == pytest.approx([1.0, -1.0])


def test_prediction_pair_product_expansion_uses_only_saved_predictions() -> None:
    frame = pl.DataFrame(
        {
            "tabm_prediction": [2.0, -3.0],
            "tree_prediction": [0.5, 4.0],
            "responder_6": [10.0, -10.0],
        }
    )

    expanded, names = _add_prediction_expert_expansions(frame, ("tabm_prediction", "tree_prediction"), ("pair_product",))

    assert names == ("tabm_prediction__x__tree_prediction",)
    assert expanded["tabm_prediction__x__tree_prediction"].to_list() == pytest.approx([1.0, -12.0])


def test_prediction_batch_expansions_use_only_same_batch_predictions() -> None:
    frame = pl.DataFrame(
        {
            "date_id": [1, 1, 1, 1],
            "time_id": [0, 0, 1, 1],
            "symbol_id": [0, 1, 0, 1],
            "tabm_prediction": [1.0, 3.0, 2.0, 2.0],
            "responder_6": [10.0, -10.0, 5.0, -5.0],
        }
    )

    expanded, names = _add_prediction_expert_expansions(
        frame,
        ("tabm_prediction",),
        ("batch_rank", "batch_mean", "batch_demean", "batch_std", "batch_zscore"),
    )

    assert names == (
        "tabm_prediction__batch_rank",
        "tabm_prediction__batch_mean",
        "tabm_prediction__batch_demean",
        "tabm_prediction__batch_std",
        "tabm_prediction__batch_zscore",
    )
    assert expanded["tabm_prediction__batch_rank"].to_list() == pytest.approx([-0.5, 0.5, 0.0, 0.0])
    assert expanded["tabm_prediction__batch_mean"].to_list() == pytest.approx([2.0, 2.0, 2.0, 2.0])
    assert expanded["tabm_prediction__batch_demean"].to_list() == pytest.approx([-1.0, 1.0, 0.0, 0.0])
    assert expanded["tabm_prediction__batch_std"].to_list() == pytest.approx([np.sqrt(2.0), np.sqrt(2.0), 0.0, 0.0])
    assert expanded["tabm_prediction__batch_zscore"].to_list() == pytest.approx([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0, 0.0])


def test_raw_preprocessing_features_are_batch_observable_and_target_safe() -> None:
    frame = pl.DataFrame(
        {
            "date_id": [1, 1, 1, 1],
            "time_id": [0, 0, 1, 1],
            "symbol_id": [0, 1, 0, 1],
            "feature_47": [1.0, 3.0, None, 2.0],
            "feature_59": [2.0, 4.0, 6.0, 8.0],
            "responder_6": [10.0, -10.0, 5.0, -5.0],
        }
    )

    expanded, names = add_raw_preprocessing_features(
        frame,
        raw_feature_columns=("feature_47", "feature_59"),
        modes=(
            "batch_rank",
            "batch_demean",
            "batch_zscore",
            "batch_abs_zscore",
            "batch_top_bottom",
            "row_missing_count",
            "row_abs_mean",
            "row_l2_energy",
        ),
    )

    assert "responder_6" not in names
    assert "feature_47__raw_batch_rank" in names
    assert "raw_row_missing_count" in names
    assert expanded["feature_47__raw_batch_rank"].to_list() == pytest.approx([-0.5, 0.5, 0.0, 0.0])
    assert expanded["feature_47__raw_batch_demean"].to_list() == pytest.approx([-1.0, 1.0, 0.0, 0.0])
    assert expanded["feature_47__raw_batch_zscore"].to_list() == pytest.approx([-1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0, 0.0])
    assert expanded["feature_47__raw_batch_abs_zscore"].to_list() == pytest.approx([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0, 0.0])
    assert expanded["feature_47__raw_batch_top10"].to_list() == pytest.approx([0.0, 1.0, 0.0, 0.0])
    assert expanded["feature_47__raw_batch_bottom10"].to_list() == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert expanded["raw_row_missing_count"].to_list() == pytest.approx([0.0, 0.0, 1.0, 0.0])
    assert expanded["raw_row_abs_mean"].to_list() == pytest.approx([1.5, 3.5, 3.0, 5.0])
    assert expanded["raw_row_l2_energy"].to_list() == pytest.approx([np.sqrt(2.5), np.sqrt(12.5), np.sqrt(18.0), np.sqrt(34.0)])


def test_raw_preprocessing_rejects_responder_columns() -> None:
    frame = pl.DataFrame({"date_id": [1], "time_id": [0], "symbol_id": [0], "responder_6": [1.0]})

    with pytest.raises(ValueError, match="target/responder"):
        add_raw_preprocessing_features(
            frame,
            raw_feature_columns=("responder_6",),
            modes=("batch_rank",),
        )


def test_parse_meminfo_gb_extracts_available_memory_and_swap() -> None:
    values = _parse_meminfo_gb("MemAvailable: 2097152 kB\nSwapFree: 1048576 kB\n")

    assert values["MemAvailable"] == pytest.approx(2.0)
    assert values["SwapFree"] == pytest.approx(1.0)


def test_resource_floor_can_be_disabled() -> None:
    _assert_resource_floor(StrongOOFConfig(min_mem_available_gb=0.0, min_swap_free_gb=0.0), "disabled")


def test_walk_forward_stack_uses_default_first_fold_and_fits_second_fold() -> None:
    frame = add_prediction_context(_frame())

    predictions, params = add_walk_forward_ridge_stack(frame, prediction_columns=("tabm_prediction", "tree_prediction"), alpha=1000.0)

    first = predictions.filter(pl.col("fold") == "rw_01")["strong_oof_ridge_stack_prediction"].to_numpy()
    assert first.tolist() == pytest.approx([0.8, -0.8])
    assert any(row["fold"] == "rw_02" and row["fit_rows"] == 2 for row in params)


def test_walk_forward_stack_candidates_names_multiple_alphas() -> None:
    frame = add_prediction_context(_frame())

    predictions, params = add_walk_forward_ridge_stack_candidates(
        frame,
        prediction_columns=("tabm_prediction", "tree_prediction"),
        alphas=(1.0, 1000.0),
    )

    assert "strong_oof_ridge_stack_alpha1_ridge_stack_prediction" in predictions.columns
    assert "strong_oof_ridge_stack_prediction" in predictions.columns
    assert {row["alpha"] for row in params} == {1.0, 1000.0}


def test_risk_shrinkage_creates_fixed_strength_candidates() -> None:
    frame = add_prediction_context(_frame())

    predictions, params = add_prediction_risk_shrinkage(frame, base_predictions=("tabm_prediction",), strengths=(0.0, 0.1))

    assert "tabm_prediction_s0_risk_shrink_prediction" in predictions.columns
    assert "tabm_prediction_s0p1_risk_shrink_prediction" in predictions.columns
    assert len(params) == 2
    base = predictions["tabm_prediction_s0_risk_shrink_prediction"].to_numpy()
    shrunk = predictions["tabm_prediction_s0p1_risk_shrink_prediction"].to_numpy()
    assert np.all(np.abs(shrunk) <= np.abs(base) + 1e-12)


def test_regime_scale_first_fold_is_identity() -> None:
    frame = add_prediction_context(_frame())

    predictions, params = add_walk_forward_regime_scales(
        frame,
        base_predictions=("tabm_prediction",),
        time_bucket_sizes=(100,),
        min_group_rows_values=(1,),
        prior_strengths=(1.0,),
    )

    column = "tabm_prediction_tb100_min1_p1_regime_scaled_prediction"
    first = predictions.filter(pl.col("fold") == "rw_01")[column].to_numpy()
    assert first.tolist() == pytest.approx([0.8, -0.8])
    assert any(row["component"] == "regime_scale" for row in params)


def test_residual_tail_uses_previous_fold_thresholds() -> None:
    frame = add_prediction_context(_frame()).with_columns(
        pl.Series("tabm_prediction_residual_prediction", [1.0, -1.0, 2.5, -2.5])
    )

    predictions, params = add_walk_forward_residual_tail_masks(
        frame,
        residual_predictions=("tabm_prediction_residual_prediction",),
        quantiles=(0.5,),
        modes=("weight",),
    )

    column = "tabm_prediction_residual_weight_q0p5_residual_tail_prediction"
    first = predictions.filter(pl.col("fold") == "rw_01").sort("time_id")[column].to_numpy()
    second = predictions.filter(pl.col("fold") == "rw_02").sort("time_id")[column].to_numpy()

    assert first.tolist() == pytest.approx([0.8, -0.8])
    assert second.tolist() == pytest.approx([1.8, -2.5])
    assert any(row["component"] == "residual_tail" and row["selected_rows"] == 1 for row in params)


def test_fixed_candidate_blend_is_weighted_average() -> None:
    frame = add_prediction_context(_frame())

    predictions, params = add_fixed_candidate_blends(
        frame,
        candidates=("tabm_prediction", "tree_prediction"),
        weights=(0.25,),
    )

    column = "fixed_blend_0_w0p25_fixed_blend_prediction"
    expected = 0.25 * frame["tabm_prediction"].to_numpy() + 0.75 * frame["tree_prediction"].to_numpy()

    assert predictions[column].to_numpy().tolist() == pytest.approx(expected.tolist())
    assert params[0]["left"] == "tabm_prediction"
    assert params[0]["right"] == "tree_prediction"


def test_walk_forward_candidate_blend_fits_from_previous_fold() -> None:
    frame = add_prediction_context(_frame())

    predictions, params = add_walk_forward_candidate_blends(
        frame,
        candidates=("tabm_prediction", "tree_prediction"),
    )

    column = "wf_blend_0_wf_blend_prediction"
    first = predictions.filter(pl.col("fold") == "rw_01").sort("time_id")[column].to_numpy()
    second = predictions.filter(pl.col("fold") == "rw_02").sort("time_id")[column].to_numpy()

    assert first.tolist() == pytest.approx([0.7, -0.6])
    assert not np.allclose(second, frame.filter(pl.col("fold") == "rw_02").sort("time_id")["tree_prediction"].to_numpy())
    assert any(row["component"] == "walk_forward_blend" and row["fit_rows"] == 2 for row in params)


def test_contextual_candidate_blend_uses_previous_fold_groups() -> None:
    frame = add_prediction_context(_frame())

    predictions, params = add_walk_forward_contextual_candidate_blends(
        frame,
        candidates=("tabm_prediction", "tree_prediction"),
        group_specs=("weight",),
        time_bucket_sizes=(100,),
        min_group_rows_values=(1,),
        prior_strengths=(0.0,),
    )

    column = "ctx_blend_0_weight_tb100_min1_p0_contextual_blend_prediction"
    first = predictions.filter(pl.col("fold") == "rw_01").sort("time_id")[column].to_numpy()
    second = predictions.filter(pl.col("fold") == "rw_02").sort("time_id")[column].to_numpy()

    assert first.tolist() == pytest.approx([0.7, -0.6])
    assert second.tolist() == pytest.approx([1.8, -1.8])
    assert any(row["component"] == "contextual_blend" and row["n_groups"] == 2 for row in params)


def test_contextual_candidate_blend_streaming_scores_match_materialized_predictions() -> None:
    frame = add_prediction_context(_frame())

    score_rows, params = score_walk_forward_contextual_candidate_blends(
        frame,
        candidates=("tabm_prediction", "tree_prediction"),
        group_specs=("weight",),
        time_bucket_sizes=(100,),
        min_group_rows_values=(1,),
        prior_strengths=(0.0,),
    )

    by_fold = {row["fold"]: row for row in score_rows}
    assert by_fold["rw_01"]["weighted_zero_mean_r2"] == pytest.approx(1.0 - 0.41 / 3.0)
    assert by_fold["rw_02"]["weighted_zero_mean_r2"] == pytest.approx(0.99)
    assert any(row["scoring_mode"] == "array_streaming" and row["n_groups"] == 2 for row in params)


def test_online_daily_scale_updates_only_after_date() -> None:
    frame = pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01", "rw_01", "rw_01"],
            "date_id": [1, 1, 2, 2],
            "time_id": [0, 1, 0, 1],
            "symbol_id": [0, 0, 0, 0],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "responder_6": [2.0, -2.0, 3.0, -3.0],
            "tabm_prediction": [1.0, -1.0, 1.0, -1.0],
        }
    )

    predictions, params = add_online_daily_scales(
        frame,
        base_predictions=("tabm_prediction",),
        prior_strengths=(0.0,),
        forgetting_factors=(1.0,),
        min_scale=0.0,
        max_scale=10.0,
    )

    column = "tabm_online_scale_f1_p0_online_scale_prediction"
    first = predictions.filter(pl.col("date_id") == 1).sort("time_id")[column].to_numpy()
    second = predictions.filter(pl.col("date_id") == 2).sort("time_id")[column].to_numpy()

    assert first.tolist() == pytest.approx([1.0, -1.0])
    assert second.tolist() == pytest.approx([2.0, -2.0])
    assert params[0]["fit_rows"] == 0
    assert any(row["date_id"] == 2 and row["fit_rows"] == 2 for row in params)


def test_online_daily_affine_updates_only_after_date() -> None:
    frame = pl.DataFrame(
        {
            "fold": ["rw_01", "rw_01", "rw_01", "rw_01"],
            "date_id": [1, 1, 2, 2],
            "time_id": [0, 1, 0, 1],
            "symbol_id": [0, 0, 0, 0],
            "weight": [1.0, 1.0, 1.0, 1.0],
            "responder_6": [3.0, 1.0, 4.0, 2.0],
            "tabm_prediction": [1.0, -1.0, 1.0, -1.0],
        }
    )

    predictions, params = add_online_daily_affine(
        frame,
        base_predictions=("tabm_prediction",),
        prior_strengths=(0.0,),
        forgetting_factors=(1.0,),
        min_scale=-10.0,
        max_scale=10.0,
    )

    column = "tabm_online_affine_f1_p0_online_affine_prediction"
    first = predictions.filter(pl.col("date_id") == 1).sort("time_id")[column].to_numpy()
    second = predictions.filter(pl.col("date_id") == 2).sort("time_id")[column].to_numpy()

    assert first.tolist() == pytest.approx([1.0, -1.0])
    assert second.tolist() == pytest.approx([3.0, 1.0])
    assert params[0]["fit_rows"] == 0
    assert any(row["date_id"] == 2 and row["fit_rows"] == 2 for row in params)
