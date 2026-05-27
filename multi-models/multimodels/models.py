"""Small, auditable model components for modular experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge

from multimodels.metrics import weighted_scale
from multimodels.transforms import RankQuantileEncoder, ZScoreEncoder, fit_encoder, frame_to_matrix


@dataclass
class SupervisedArrayModel:
    """A fitted sklearn-style regressor plus its causal encoder."""

    name: str
    family: str
    feature_columns: tuple[str, ...]
    encoder: ZScoreEncoder | RankQuantileEncoder
    estimator: Ridge | PLSRegression

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        x = frame_to_matrix(frame, self.feature_columns)
        encoded = self.encoder.transform(x)
        pred = self.estimator.predict(encoded)
        return np.asarray(pred, dtype=np.float64).reshape(-1)


def fit_ridge_model(
    *,
    name: str,
    family: str,
    train: pl.DataFrame,
    feature_columns: tuple[str, ...],
    target: np.ndarray,
    weight: np.ndarray,
    alpha: float,
    encoder_kind: str,
    rank_bins: int,
) -> SupervisedArrayModel:
    """Fit a Ridge model on encoded features."""

    x = frame_to_matrix(train, feature_columns)
    encoder = fit_encoder(encoder_kind, x, rank_bins=rank_bins)
    encoded = encoder.transform(x)
    estimator = Ridge(alpha=alpha, fit_intercept=True)
    estimator.fit(encoded, np.asarray(target, dtype=np.float64), sample_weight=np.asarray(weight, dtype=np.float64))
    return SupervisedArrayModel(name=name, family=family, feature_columns=feature_columns, encoder=encoder, estimator=estimator)


def fit_pls_model(
    *,
    name: str,
    family: str,
    train: pl.DataFrame,
    feature_columns: tuple[str, ...],
    target: np.ndarray,
    components: int,
    rank_bins: int,
) -> SupervisedArrayModel:
    """Fit PLS on rank-normalized features.

    `PLSRegression` has no native sample weights in sklearn, so this component
    is intentionally used as a latent representation baseline and scored with
    weighted validation metrics.
    """

    x = frame_to_matrix(train, feature_columns)
    encoder = RankQuantileEncoder.fit(x, n_bins=rank_bins)
    encoded = encoder.transform(x)
    n_components = max(1, min(int(components), encoded.shape[1], max(1, encoded.shape[0] - 1)))
    estimator = PLSRegression(n_components=n_components, scale=False)
    estimator.fit(encoded, np.asarray(target, dtype=np.float64).reshape(-1, 1))
    return SupervisedArrayModel(name=name, family=family, feature_columns=feature_columns, encoder=encoder, estimator=estimator)


@dataclass(frozen=True)
class LinearStacker:
    """Low-dimensional Ridge stacker over model predictions."""

    name: str
    model_names: tuple[str, ...]
    coefficients: np.ndarray

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        x = np.asarray(predictions, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.coefficients.size:
            raise ValueError("prediction matrix has incompatible shape")
        return x @ self.coefficients


def fit_linear_stacker(
    *,
    name: str,
    model_names: tuple[str, ...],
    predictions: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    alpha: float,
) -> LinearStacker:
    """Fit a small Ridge stacker without intercept to preserve zero baseline."""

    x = np.asarray(predictions, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    xtw = x.T * w
    lhs = xtw @ x + np.eye(x.shape[1], dtype=np.float64) * float(alpha)
    rhs = xtw @ y
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return LinearStacker(name=name, model_names=model_names, coefficients=coef.astype(np.float64, copy=False))


@dataclass(frozen=True)
class ResidualRule:
    """Frozen univariate residual correction y_resid ~= alpha * z(feature)."""

    name: str
    feature: str
    mean: float
    scale: float
    alpha: float

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        values = frame[self.feature].to_numpy().astype(np.float64, copy=False)
        filled = np.where(np.isfinite(values), values, self.mean)
        return self.alpha * ((filled - self.mean) / self.scale)


def fit_residual_rule(
    *,
    train: pl.DataFrame,
    feature: str,
    residual: np.ndarray,
    weight: np.ndarray,
    name: str | None = None,
) -> ResidualRule:
    """Fit a single residual rule from train-only residuals."""

    values = train[feature].to_numpy().astype(np.float64, copy=False)
    finite = np.isfinite(values)
    mean = float(np.mean(values[finite])) if np.any(finite) else 0.0
    centered = np.where(finite, values, mean) - mean
    scale = float(np.std(centered))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    phi = centered / scale
    alpha = weighted_scale(np.asarray(residual, dtype=np.float64), phi, np.asarray(weight, dtype=np.float64))
    return ResidualRule(name=name or f"residual_{feature}_z", feature=feature, mean=mean, scale=scale, alpha=alpha)


@dataclass(frozen=True)
class RegimeBinner:
    """Train-fitted bins for observable regime features."""

    weight_edges: tuple[float, float]
    abs_pred_edges: tuple[float, float]
    risk_edges: tuple[float, float]
    time_bucket_size: int

    @classmethod
    def fit(cls, *, weight: np.ndarray, base_pred: np.ndarray, risk: np.ndarray, time_bucket_size: int) -> "RegimeBinner":
        return cls(
            weight_edges=_safe_quantile_edges(weight),
            abs_pred_edges=_safe_quantile_edges(np.abs(base_pred)),
            risk_edges=_safe_quantile_edges(risk),
            time_bucket_size=time_bucket_size,
        )

    def transform(self, frame: pl.DataFrame, *, base_pred: np.ndarray, risk: np.ndarray) -> np.ndarray:
        time_bucket = (frame["time_id"].to_numpy().astype(np.int64, copy=False) // self.time_bucket_size).astype(np.int64)
        weight_bucket = _bucketize(frame["weight"].to_numpy().astype(np.float64, copy=False), self.weight_edges)
        abs_bucket = _bucketize(np.abs(base_pred), self.abs_pred_edges)
        risk_bucket = _bucketize(risk, self.risk_edges)
        return time_bucket * 27 + weight_bucket * 9 + abs_bucket * 3 + risk_bucket


@dataclass(frozen=True)
class MicrostructureRegimeBinner:
    """Observable regime bins for time, symbol, weight, risk and batch context."""

    weight_edges: tuple[float, float]
    abs_pred_edges: tuple[float, float]
    risk_edges: tuple[float, float]
    missing_edges: tuple[float, float]
    lag_energy_edges: tuple[float, float]
    time_bucket_size: int
    symbol_mod: int

    @classmethod
    def fit(
        cls,
        *,
        frame: pl.DataFrame,
        base_pred: np.ndarray,
        risk: np.ndarray,
        time_bucket_size: int,
        symbol_mod: int,
    ) -> "MicrostructureRegimeBinner":
        if symbol_mod <= 0:
            raise ValueError("symbol_mod must be positive")
        return cls(
            weight_edges=_safe_quantile_edges(_frame_or_zero(frame, "weight")),
            abs_pred_edges=_safe_quantile_edges(np.abs(base_pred)),
            risk_edges=_safe_quantile_edges(risk),
            missing_edges=_safe_quantile_edges(_frame_or_zero(frame, "ctx_missing_count")),
            lag_energy_edges=_safe_quantile_edges(_frame_or_zero(frame, "ctx_lag_energy")),
            time_bucket_size=time_bucket_size,
            symbol_mod=symbol_mod,
        )

    def transform(self, frame: pl.DataFrame, *, base_pred: np.ndarray, risk: np.ndarray) -> np.ndarray:
        if self.time_bucket_size <= 0:
            raise ValueError("time_bucket_size must be positive")
        time_bucket = frame["time_id"].to_numpy().astype(np.int64, copy=False) // self.time_bucket_size
        symbol_bucket = np.mod(frame["symbol_id"].to_numpy().astype(np.int64, copy=False), self.symbol_mod)
        weight_bucket = _bucketize(_frame_or_zero(frame, "weight"), self.weight_edges)
        abs_bucket = _bucketize(np.abs(base_pred), self.abs_pred_edges)
        risk_bucket = _bucketize(risk, self.risk_edges)
        missing_bucket = _bucketize(_frame_or_zero(frame, "ctx_missing_count"), self.missing_edges)
        lag_bucket = _bucketize(_frame_or_zero(frame, "ctx_lag_energy"), self.lag_energy_edges)
        code = time_bucket
        for multiplier, bucket in (
            (self.symbol_mod, symbol_bucket),
            (3, weight_bucket),
            (3, abs_bucket),
            (3, risk_bucket),
            (3, missing_bucket),
            (3, lag_bucket),
        ):
            code = code * multiplier + bucket
        return code.astype(np.int64, copy=False)


@dataclass(frozen=True)
class GroupedScaleCalibrator:
    """Per-regime scale with global fallback and ridge-like prior."""

    default_scale: float
    group_scales: dict[int, float]

    def apply(self, group_codes: np.ndarray, prediction: np.ndarray) -> np.ndarray:
        codes = np.asarray(group_codes, dtype=np.int64)
        scale = np.full(codes.shape, self.default_scale, dtype=np.float64)
        for code, value in self.group_scales.items():
            scale[codes == code] = value
        return scale * np.asarray(prediction, dtype=np.float64)


def fit_grouped_scale_calibrator(
    *,
    group_codes: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    min_rows: int,
    prior_strength: float,
) -> GroupedScaleCalibrator:
    """Fit per-group scales on train only."""

    default = weighted_scale(target, prediction, weight, prior=prior_strength)
    scales: dict[int, float] = {}
    codes = np.asarray(group_codes, dtype=np.int64)
    for code in np.unique(codes):
        mask = codes == code
        if int(mask.sum()) < min_rows:
            continue
        rhs = float(np.sum(weight[mask] * prediction[mask] * target[mask]) + prior_strength * default)
        lhs = float(np.sum(weight[mask] * prediction[mask] * prediction[mask]) + prior_strength)
        scales[int(code)] = rhs / lhs if lhs > 0.0 else default
    return GroupedScaleCalibrator(default_scale=default, group_scales=scales)


def risk_shrink(prediction: np.ndarray, risk: np.ndarray, *, train_risk_mean: float, strength: float) -> np.ndarray:
    """Shrink predictions more aggressively in high predicted-risk regions."""

    pred = np.asarray(prediction, dtype=np.float64)
    risk_values = np.maximum(np.asarray(risk, dtype=np.float64), 0.0)
    denom = max(float(train_risk_mean), 1e-12)
    normalized = np.clip(risk_values / denom, 0.0, 10.0)
    return pred / (1.0 + float(strength) * normalized)


def _safe_quantile_edges(values: np.ndarray) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (0.0, 1.0)
    low, high = np.quantile(x, [1.0 / 3.0, 2.0 / 3.0])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        low = float(np.min(x))
        high = float(np.max(x))
    if low >= high:
        high = low + 1.0
    return (float(low), float(high))


def _bucketize(values: np.ndarray, edges: tuple[float, float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = np.where(np.isfinite(x), x, edges[0])
    return np.searchsorted(np.asarray(edges, dtype=np.float64), x, side="right").astype(np.int64)


def _frame_or_zero(frame: pl.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.zeros(frame.height, dtype=np.float64)
    return frame[column].to_numpy().astype(np.float64, copy=False)
