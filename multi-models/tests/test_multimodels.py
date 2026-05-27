from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from multimodels.metrics import weighted_zero_mean_r2_arrays
from multimodels.features import add_context_features, add_raw_preprocessing_features
from multimodels.models import fit_grouped_scale_calibrator, fit_residual_rule, MicrostructureRegimeBinner, risk_shrink
from multimodels.pipeline import artifact_manifest, ExperimentConfig, _validate_config
from multimodels.transforms import RankQuantileEncoder


def test_rank_quantile_encoder_uses_train_edges_and_bounds_valid_values() -> None:
    train = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float64)
    valid = np.array([[-100.0], [100.0], [np.nan]], dtype=np.float64)

    encoder = RankQuantileEncoder.fit(train, n_bins=8)
    transformed = encoder.transform(valid)

    assert transformed.shape == valid.shape
    assert np.all(np.isfinite(transformed))
    assert float(transformed.min()) >= -1.0
    assert float(transformed.max()) <= 1.0
    assert transformed[0, 0] < transformed[1, 0]


def test_residual_rule_recovers_simple_train_only_residual_signal() -> None:
    frame = pl.DataFrame({"feature_47": [-2.0, -1.0, 1.0, 2.0]})
    residual = np.array([-0.2, -0.1, 0.1, 0.2], dtype=np.float64)
    weight = np.ones(4, dtype=np.float64)

    rule = fit_residual_rule(train=frame, feature="feature_47", residual=residual, weight=weight)
    pred = rule.predict(frame)

    assert weighted_zero_mean_r2_arrays(residual, pred, weight) > 0.99


def test_grouped_scale_calibrator_falls_back_for_sparse_groups() -> None:
    groups = np.array([0, 0, 0, 1], dtype=np.int64)
    pred = np.array([1.0, 2.0, 3.0, 10.0], dtype=np.float64)
    target = 2.0 * pred
    weight = np.ones(4, dtype=np.float64)

    calibrator = fit_grouped_scale_calibrator(
        group_codes=groups,
        prediction=pred,
        target=target,
        weight=weight,
        min_rows=2,
        prior_strength=0.0,
    )
    out = calibrator.apply(np.array([0, 1, 2], dtype=np.int64), np.array([1.0, 1.0, 1.0], dtype=np.float64))

    assert calibrator.group_scales[0] == pytest.approx(2.0)
    assert out[0] == pytest.approx(2.0)
    assert out[1] == pytest.approx(calibrator.default_scale)
    assert out[2] == pytest.approx(calibrator.default_scale)


def test_grouped_scale_prior_shrinks_toward_global_scale() -> None:
    groups = np.array([0, 0], dtype=np.int64)
    pred = np.array([1.0, 1.0], dtype=np.float64)
    target = np.array([4.0, 4.0], dtype=np.float64)
    weight = np.ones(2, dtype=np.float64)

    calibrator = fit_grouped_scale_calibrator(
        group_codes=groups,
        prediction=pred,
        target=target,
        weight=weight,
        min_rows=1,
        prior_strength=10.0,
    )

    assert calibrator.default_scale < calibrator.group_scales[0] < 4.0


def test_risk_shrink_reduces_high_risk_predictions_more() -> None:
    pred = np.array([1.0, 1.0], dtype=np.float64)
    risk = np.array([0.5, 2.0], dtype=np.float64)

    shrunk = risk_shrink(pred, risk, train_risk_mean=1.0, strength=0.5)

    assert shrunk[1] < shrunk[0] < 1.0


def test_context_features_include_observable_microstructure_inputs() -> None:
    lazy = pl.DataFrame(
        {
            "date_id": [1, 1],
            "time_id": [10, 10],
            "symbol_id": [3, 4],
            "weight": [1.0, 3.0],
            "feature_47": [1.0, None],
            "responder_0_lag_1": [0.5, -0.25],
        }
    ).lazy()

    enriched, context = add_context_features(
        lazy,
        base_features=("feature_47",),
        lag_features=("responder_0_lag_1",),
        cross_sectional_features=("feature_47",),
        time_bucket_size=10,
    )
    frame = enriched.collect()

    assert "ctx_log1p_weight" in context
    assert "ctx_missing_count" in context
    assert "ctx_lag_energy" in context
    assert frame["ctx_lag_energy"].to_list() == pytest.approx([0.5, 0.25])


def test_raw_preprocessing_features_are_batch_observable_and_target_safe() -> None:
    lazy = pl.DataFrame(
        {
            "date_id": [1, 1, 1],
            "time_id": [10, 10, 20],
            "symbol_id": [1, 2, 1],
            "feature_47": [1.0, 3.0, None],
            "feature_59": [2.0, 4.0, 8.0],
            "responder_6": [9.0, -9.0, 4.0],
        }
    ).lazy()

    enriched, names = add_raw_preprocessing_features(
        lazy,
        raw_feature_columns=("feature_47", "feature_59"),
        modes=("batch_rank", "batch_demean", "batch_zscore", "batch_top_bottom", "row_missing_count", "row_abs_mean", "row_l2_energy"),
    )
    frame = enriched.collect()

    assert "feature_47__raw_batch_rank" in names
    assert "feature_47__raw_batch_zscore" in names
    assert "raw_row_missing_count" in names
    assert "responder_6" not in names
    assert frame["feature_47__raw_batch_rank"].to_list()[:2] == pytest.approx([-0.5, 0.5])
    assert frame["feature_47__raw_batch_demean"].to_list()[:2] == pytest.approx([-1.0, 1.0])
    assert frame["feature_47__raw_batch_zscore"].to_list()[:2] == pytest.approx([-1.0, 1.0])
    assert frame["feature_47__raw_batch_bottom10"].to_list()[:2] == pytest.approx([1.0, 0.0])
    assert frame["feature_47__raw_batch_top10"].to_list()[:2] == pytest.approx([0.0, 1.0])
    assert frame["raw_row_missing_count"].to_list() == pytest.approx([0.0, 0.0, 1.0])
    assert all(np.isfinite(frame[name].to_numpy()).all() for name in names)


def test_raw_preprocessing_rejects_responder_columns_in_config_and_builder() -> None:
    with pytest.raises(ValueError, match="target/responder"):
        add_raw_preprocessing_features(
            pl.DataFrame({"date_id": [1], "time_id": [1], "responder_6": [0.0]}).lazy(),
            raw_feature_columns=("responder_6",),
            modes=("batch_rank",),
        )
    with pytest.raises(ValueError, match="target/responder"):
        _validate_config(ExperimentConfig(raw_preprocess_features=("responder_0_lag_1",), raw_preprocess_modes=("batch_rank",)))


def test_microstructure_regime_binner_uses_target_free_context() -> None:
    frame = pl.DataFrame(
        {
            "time_id": [0, 50, 100, 150],
            "symbol_id": [1, 2, 3, 4],
            "weight": [1.0, 2.0, 3.0, 4.0],
            "ctx_missing_count": [0.0, 1.0, 0.0, 2.0],
            "ctx_lag_energy": [0.1, 0.2, 0.3, 0.4],
        }
    )
    base_pred = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    risk = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)

    binner = MicrostructureRegimeBinner.fit(frame=frame, base_pred=base_pred, risk=risk, time_bucket_size=100, symbol_mod=4)
    codes = binner.transform(frame, base_pred=base_pred, risk=risk)

    assert codes.shape == (4,)
    assert codes.dtype == np.int64
    assert len(set(codes.tolist())) > 1


def test_artifact_manifest_names_primary_family_artifacts() -> None:
    manifest = artifact_manifest(ExperimentConfig())
    names = {row["artifact"] for row in manifest}

    assert len(manifest) == 9
    assert "ridge_rank_alpha10000" in names
    assert "pls_rank_k8" in names
    assert "gateway_risk_conservative_rls_abs_pred_s100_prediction" in names
    assert "ridge_rank_alpha10000__feature_47_z_residual" in names
    assert "pls_rank_k8__feature_59_z_residual" in names
    assert "ridge_rank_alpha10000__risk_abs_responder6_ridge_rank_score" in names
    assert "ridge_rank_alpha10000__risk_sq_responder6_ridge_rank_score" in names
    assert "ridge_rank_alpha10000__risk_abs_error_ridge_rank_score" in names
    assert "ridge_rank_alpha10000__risk_abs_error_s0p05_micro_regime_scaled" in names
