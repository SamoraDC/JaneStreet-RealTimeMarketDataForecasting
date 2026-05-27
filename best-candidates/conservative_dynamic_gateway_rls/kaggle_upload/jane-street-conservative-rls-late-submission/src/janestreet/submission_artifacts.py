"""Artifact helpers for Jane Street submission runtime."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from janestreet.bayesian_meta import weighted_normal_stats
from janestreet.submission_inference import DynamicRLSMetaState, make_prior_rls_state


RLS_STATE_FILE = "rls_state.npz"


def fit_initial_rls_state_from_oof(
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    ridge_alpha: float,
    forgetting_factor: float,
    prior_feature: str = "tabm_prediction",
) -> DynamicRLSMetaState:
    """Fit the initial RLS posterior from out-of-fold prediction rows."""

    if frame.is_empty():
        raise ValueError("OOF frame must not be empty")
    state = make_prior_rls_state(
        feature_columns,
        ridge_alpha=ridge_alpha,
        forgetting_factor=forgetting_factor,
        prior_feature=prior_feature,
    )
    gram, rhs = weighted_normal_stats(frame, feature_columns=state.feature_columns)
    state.precision = state.precision + gram
    state.rhs = state.rhs + rhs
    return state


def save_rls_state_artifact(
    artifact_dir: Path,
    state: DynamicRLSMetaState,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metadata_json = json.dumps(metadata or {}, sort_keys=True)
    np.savez_compressed(
        artifact_dir / RLS_STATE_FILE,
        feature_columns=np.asarray(state.feature_columns, dtype=object),
        precision=state.precision,
        rhs=state.rhs,
        forgetting_factor=np.asarray([state.forgetting_factor], dtype=np.float64),
        metadata_json=np.asarray([metadata_json], dtype=object),
    )


def load_rls_state_artifact(artifact_dir: Path) -> tuple[DynamicRLSMetaState, dict[str, Any]]:
    path = artifact_dir / RLS_STATE_FILE
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        feature_columns = tuple(str(value) for value in data["feature_columns"].tolist())
        state = DynamicRLSMetaState(
            feature_columns=feature_columns,
            precision=data["precision"].astype(np.float64, copy=True),
            rhs=data["rhs"].astype(np.float64, copy=True),
            forgetting_factor=float(data["forgetting_factor"][0]),
        )
        metadata = json.loads(str(data["metadata_json"][0]))
    return state, metadata
