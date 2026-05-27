"""Kaggle submission entrypoint.

This file is intentionally conservative: without a trained artifact directory it
will refuse to run, unless `JANE_STREET_ALLOW_SMOKE_FALLBACK=1` is set for local
gateway smoke tests. The smoke fallback is not a competitive model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SRC = PROJECT_ROOT / "src"
LOCAL_VENDOR = PROJECT_ROOT / "vendor"
LOCAL_COMPETITION_DIR = PROJECT_ROOT / "data" / "raw" / "jane-street-real-time-market-data-forecasting"
KAGGLE_COMPETITION_DIR = Path(
    os.getenv(
        "JANE_STREET_COMPETITION_DIR",
        "/kaggle/input/jane-street-real-time-market-data-forecasting",
    )
)

for candidate in (LOCAL_VENDOR, LOCAL_SRC, LOCAL_COMPETITION_DIR, KAGGLE_COMPETITION_DIR):
    if candidate.exists():
        sys.path.insert(0, str(candidate))

from janestreet.submission_artifacts import load_rls_state_artifact
from janestreet.submission_inference import KaggleRLSSubmissionPredictor, make_prior_rls_state
from janestreet.submission_models import ArtifactFeaturePredictor


FEATURE_COLUMNS = ("tabm_prediction", "xgboost_prediction")
_PREDICTOR: KaggleRLSSubmissionPredictor | None = None


class SmokeFeaturePredictor:
    """Deterministic fallback used only to test the gateway contract."""

    def update_from_lags(self, lags: pl.DataFrame | None) -> None:
        return None

    def predict_features(self, test_with_lags: pl.DataFrame) -> pl.DataFrame:
        return test_with_lags.select(
            "date_id",
            "time_id",
            "symbol_id",
            "weight",
            pl.col("feature_00").fill_null(0.0).cast(pl.Float64).alias("tabm_prediction"),
            pl.col("feature_01").fill_null(0.0).cast(pl.Float64).alias("xgboost_prediction"),
        )


def _load_predictor() -> KaggleRLSSubmissionPredictor:
    if os.getenv("JANE_STREET_ALLOW_SMOKE_FALLBACK") == "1":
        state = make_prior_rls_state(
            FEATURE_COLUMNS,
            ridge_alpha=1000.0,
            forgetting_factor=0.995,
            prior_feature="tabm_prediction",
        )
        return KaggleRLSSubmissionPredictor(SmokeFeaturePredictor(), state)
    default_base_artifact_dir = PROJECT_ROOT / "artifacts" / "jane_street_submission" / "base_models"
    base_artifact_dir = Path(os.getenv("JANE_STREET_BASE_ARTIFACT_DIR", str(default_base_artifact_dir)))
    meta_artifact_dir = Path(
        os.getenv(
            "JANE_STREET_META_ARTIFACT_DIR",
            str(PROJECT_ROOT / "artifacts" / "jane_street_submission" / "meta_rls_experts_alpha10000_f0p995"),
        )
    )
    if base_artifact_dir.exists() and meta_artifact_dir.exists():
        state, _metadata = load_rls_state_artifact(meta_artifact_dir)
        base = ArtifactFeaturePredictor(
            base_artifact_dir,
            device=os.getenv("JANE_STREET_DEVICE", "auto"),
        )
        return KaggleRLSSubmissionPredictor(base, state)
    raise FileNotFoundError(
        f"Missing trained artifact directories: base={base_artifact_dir}, meta={meta_artifact_dir}. "
        "Set JANE_STREET_ALLOW_SMOKE_FALLBACK=1 only for local gateway smoke tests."
    )


def _get_predictor() -> KaggleRLSSubmissionPredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = _load_predictor()
    return _PREDICTOR


def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
    predictions = _get_predictor().predict(test, lags)
    if predictions.columns != ["row_id", "responder_6"]:
        raise ValueError("predictions must contain exactly row_id,responder_6")
    if len(predictions) != len(test):
        raise ValueError("prediction row count must match test row count")
    return predictions


if __name__ == "__main__":
    import kaggle_evaluation.jane_street_inference_server
    import jane_street_gateway

    class LocalCompatibleJSInferenceServer(kaggle_evaluation.jane_street_inference_server.JSInferenceServer):
        def _get_gateway_for_test(self, data_paths=None, file_share_dir=None, *args, **kwargs):
            return jane_street_gateway.JSGateway(data_paths)

    server_cls = (
        kaggle_evaluation.jane_street_inference_server.JSInferenceServer
        if os.getenv("KAGGLE_IS_COMPETITION_RERUN")
        else LocalCompatibleJSInferenceServer
    )
    inference_server = server_cls(predict)
    if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
        inference_server.serve()
    elif os.getenv("JANE_STREET_RUN_LOCAL_GATEWAY") == "1":
        competition_dir = KAGGLE_COMPETITION_DIR if KAGGLE_COMPETITION_DIR.exists() else LOCAL_COMPETITION_DIR
        inference_server.run_local_gateway(
            (
                str(competition_dir / "test.parquet"),
                str(competition_dir / "lags.parquet"),
            )
        )
