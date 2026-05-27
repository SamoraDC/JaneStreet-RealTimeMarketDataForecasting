from __future__ import annotations

import polars as pl
import pytest
import numpy as np

from janestreet.submission_artifacts import (
    fit_initial_rls_state_from_oof,
    load_rls_state_artifact,
    save_rls_state_artifact,
)


def test_fit_initial_rls_state_from_oof_adds_weighted_normal_stats() -> None:
    frame = pl.DataFrame(
        {
            "tabm_prediction": [2.0],
            "responder_6": [3.0],
            "weight": [4.0],
        }
    )

    state = fit_initial_rls_state_from_oof(
        frame,
        feature_columns=("tabm_prediction",),
        ridge_alpha=10.0,
        forgetting_factor=0.995,
    )

    assert state.precision[0, 0] == pytest.approx(10.0 + 4.0 * 2.0 * 2.0)
    assert state.rhs[0] == pytest.approx(10.0 + 4.0 * 2.0 * 3.0)
    assert state.forgetting_factor == pytest.approx(0.995)


def test_rls_state_artifact_roundtrip(tmp_path) -> None:
    frame = pl.DataFrame(
        {
            "tabm_prediction": [1.0, 0.5],
            "xgboost_prediction": [0.0, 1.0],
            "responder_6": [1.5, -0.5],
            "weight": [2.0, 3.0],
        }
    )
    state = fit_initial_rls_state_from_oof(
        frame,
        feature_columns=("tabm_prediction", "xgboost_prediction"),
        ridge_alpha=1000.0,
        forgetting_factor=0.995,
    )

    save_rls_state_artifact(tmp_path, state, metadata={"candidate": "smoke"})
    loaded, metadata = load_rls_state_artifact(tmp_path)

    assert loaded.feature_columns == state.feature_columns
    assert np.allclose(loaded.precision, state.precision)
    assert loaded.rhs.tolist() == pytest.approx(state.rhs.tolist())
    assert loaded.forgetting_factor == pytest.approx(0.995)
    assert metadata == {"candidate": "smoke"}
