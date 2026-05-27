"""Runtime helpers for Kaggle-style Jane Street inference.

The classes here intentionally avoid training logic. They implement the online
contract that must be shared by the local gateway test and by a Kaggle
submission notebook: update only when the gateway serves previous-day lags,
predict the current batch, then cache the batch features for the next day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np
import polars as pl

from janestreet.official_lags import GatewayResponderLagCache


KEY_COLUMNS: tuple[str, ...] = ("date_id", "time_id", "symbol_id")
TARGET_COLUMN = "responder_6"
TARGET_LAG_COLUMN = "responder_6_lag_1"
WEIGHT_COLUMN = "weight"


class BatchFeaturePredictor(Protocol):
    """Protocol for base predictors used by the RLS meta layer."""

    def update_from_lags(self, lags: pl.DataFrame | None) -> None:
        """Update internal online state from gateway-provided previous-day lags."""

    def predict_features(self, test_with_lags: pl.DataFrame) -> pl.DataFrame:
        """Return key columns, weight, and base prediction feature columns."""


@dataclass
class DynamicRLSMetaState:
    """Small dense RLS/Kalman-like meta state."""

    feature_columns: tuple[str, ...]
    precision: np.ndarray
    rhs: np.ndarray
    forgetting_factor: float = 1.0

    def __post_init__(self) -> None:
        if not self.feature_columns:
            raise ValueError("feature_columns must not be empty")
        n_features = len(self.feature_columns)
        if self.precision.shape != (n_features, n_features):
            raise ValueError("precision shape does not match feature_columns")
        if self.rhs.shape != (n_features,):
            raise ValueError("rhs shape does not match feature_columns")
        if self.forgetting_factor <= 0.0 or self.forgetting_factor > 1.0:
            raise ValueError("forgetting_factor must be in (0, 1]")

    @property
    def beta(self) -> np.ndarray:
        return np.linalg.solve(self.precision, self.rhs)

    def predict(self, features: pl.DataFrame) -> np.ndarray:
        x = features.select(list(self.feature_columns)).to_numpy().astype(np.float64, copy=False)
        return x @ self.beta

    def update(self, frame: pl.DataFrame, *, target_column: str = TARGET_COLUMN) -> None:
        if frame.is_empty():
            return
        missing = set((*self.feature_columns, target_column, WEIGHT_COLUMN)) - set(frame.columns)
        if missing:
            raise ValueError(f"missing RLS update columns: {', '.join(sorted(missing))}")
        arrays = frame.select(list(self.feature_columns) + [target_column, WEIGHT_COLUMN]).to_numpy()
        x = arrays[:, : len(self.feature_columns)].astype(np.float64, copy=False)
        y = arrays[:, len(self.feature_columns)].astype(np.float64, copy=False)
        sample_weight = arrays[:, len(self.feature_columns) + 1].astype(np.float64, copy=False)
        self.precision = self.forgetting_factor * self.precision + x.T @ (x * sample_weight[:, None])
        self.rhs = self.forgetting_factor * self.rhs + x.T @ (sample_weight * y)


@dataclass
class PredictionFeatureCache:
    """Cache base prediction features for the latest date."""

    feature_columns: tuple[str, ...]
    cached_date_id: int | None = None
    _frames: list[pl.DataFrame] = field(default_factory=list)

    def cache_batch(self, frame: pl.DataFrame) -> None:
        if frame.is_empty():
            return
        missing = set((*KEY_COLUMNS, WEIGHT_COLUMN, *self.feature_columns)) - set(frame.columns)
        if missing:
            raise ValueError(f"missing cache columns: {', '.join(sorted(missing))}")
        date_ids = frame.select("date_id").unique()["date_id"].to_list()
        if len(date_ids) != 1:
            raise ValueError("one prediction batch must contain exactly one date_id")
        date_id = int(date_ids[0])
        selected = frame.select(list(KEY_COLUMNS) + [WEIGHT_COLUMN, *self.feature_columns])
        if self.cached_date_id is None or self.cached_date_id != date_id:
            self.cached_date_id = date_id
            self._frames = [selected]
        else:
            self._frames.append(selected)

    def build_lag_update_frame(self, lags: pl.DataFrame | None) -> pl.DataFrame:
        if lags is None or lags.is_empty() or self.cached_date_id is None or not self._frames:
            return pl.DataFrame()
        if TARGET_LAG_COLUMN not in lags.columns:
            raise ValueError(f"lags must contain {TARGET_LAG_COLUMN}")
        lag_date_ids = lags.select("date_id").unique()["date_id"].to_list()
        if len(lag_date_ids) != 1:
            raise ValueError("lags batch must contain exactly one date_id")
        lag_date_id = int(lag_date_ids[0])
        shifted_cache = (
            pl.concat(self._frames, how="vertical")
            .with_columns((pl.col("date_id") + 1).cast(lags.schema["date_id"]).alias("date_id"))
            .filter(pl.col("date_id") == lag_date_id)
        )
        if shifted_cache.is_empty():
            return pl.DataFrame()
        joined = shifted_cache.join(
            lags.select(list(KEY_COLUMNS) + [TARGET_LAG_COLUMN]),
            on=list(KEY_COLUMNS),
            how="inner",
        )
        if joined.is_empty():
            return pl.DataFrame()
        return joined.rename({TARGET_LAG_COLUMN: TARGET_COLUMN})


@dataclass
class KaggleRLSSubmissionPredictor:
    """Composable predictor implementing the competition `predict(test, lags)` contract."""

    base_predictor: BatchFeaturePredictor
    meta_state: DynamicRLSMetaState
    lag_cache: GatewayResponderLagCache = field(default_factory=GatewayResponderLagCache)
    feature_cache: PredictionFeatureCache | None = None
    last_meta_update_date_id: int | None = None

    def __post_init__(self) -> None:
        if self.feature_cache is None:
            self.feature_cache = PredictionFeatureCache(self.meta_state.feature_columns)

    def predict(self, test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
        if "row_id" not in test.columns:
            raise ValueError("test batch must contain row_id")
        test_polars = _ensure_polars(test)
        lags_polars = None if lags is None else _ensure_polars(lags)
        if lags_polars is not None:
            self.base_predictor.update_from_lags(lags_polars)
            self._update_meta_from_lags(lags_polars)
        test_with_lags = self.lag_cache.add_to_batch(test_polars, lags_polars)
        features = self.base_predictor.predict_features(test_with_lags)
        predictions = self.meta_state.predict(features).astype(np.float64, copy=False)
        self.feature_cache.cache_batch(features)
        return test_polars.select("row_id").with_columns(
            pl.Series(TARGET_COLUMN, predictions.astype(np.float64, copy=False))
        )

    def _update_meta_from_lags(self, lags: pl.DataFrame) -> None:
        lag_date_id = _single_date_id(lags)
        if self.last_meta_update_date_id == lag_date_id:
            return
        update_frame = self.feature_cache.build_lag_update_frame(lags)
        self.meta_state.update(update_frame)
        self.last_meta_update_date_id = lag_date_id


def make_prior_rls_state(
    feature_columns: Sequence[str],
    *,
    ridge_alpha: float,
    forgetting_factor: float,
    prior_feature: str = "tabm_prediction",
) -> DynamicRLSMetaState:
    features = tuple(feature_columns)
    if ridge_alpha <= 0.0:
        raise ValueError("ridge_alpha must be positive")
    precision = ridge_alpha * np.eye(len(features), dtype=np.float64)
    rhs = np.zeros(len(features), dtype=np.float64)
    if prior_feature in features:
        rhs[features.index(prior_feature)] = ridge_alpha
    return DynamicRLSMetaState(features, precision, rhs, forgetting_factor=forgetting_factor)


def _single_date_id(frame: pl.DataFrame) -> int:
    date_ids = frame.select("date_id").unique()["date_id"].to_list()
    if len(date_ids) != 1:
        raise ValueError("frame must contain exactly one date_id")
    return int(date_ids[0])


def _ensure_polars(frame: pl.DataFrame) -> pl.DataFrame:
    if isinstance(frame, pl.DataFrame):
        return frame
    return pl.from_pandas(frame)
