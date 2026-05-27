"""Load trained base artifacts for Kaggle submission inference."""

from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
import tabm
import torch
from torch.utils.data import DataLoader, TensorDataset

from janestreet.calibration import add_abs_prediction_bucket
from janestreet.official_lags import responder_lag_columns
from janestreet.submission_inference import KEY_COLUMNS, WEIGHT_COLUMN


class SophiaG(torch.optim.Optimizer):
    """Sophia-G style optimizer matching the training script's online updates."""

    def __init__(
        self,
        params,
        *,
        lr: float,
        betas: tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.0,
        eps: float = 1e-12,
        clip: float = 1.0,
        update_period: int = 10,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "rho": rho,
            "weight_decay": weight_decay,
            "eps": eps,
            "clip": clip,
            "update_period": update_period,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["hessian"] = torch.zeros_like(param)
                exp_avg = state["exp_avg"]
                hessian = state["hessian"]
                state["step"] += 1
                if group["weight_decay"] != 0.0:
                    param.mul_(1.0 - group["lr"] * group["weight_decay"])
                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                if state["step"] % group["update_period"] == 1:
                    hessian.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = torch.clamp(group["rho"] * hessian + group["eps"], min=group["eps"])
                update = torch.clamp(exp_avg / denom, min=-group["clip"], max=group["clip"])
                param.add_(update, alpha=-group["lr"])
        return loss


class TabMArtifactPredictor:
    """Inference-only TabM artifact wrapper."""

    def __init__(self, artifact_dir: Path, *, device: str = "auto", batch_size: int = 8192) -> None:
        self.artifact_dir = artifact_dir
        self.config = json.loads((artifact_dir / "config.json").read_text(encoding="utf-8"))
        standardization = np.load(artifact_dir / "standardization.npz")
        self.continuous_mean = standardization["continuous_mean"].astype(np.float32, copy=False)
        self.continuous_scale = standardization["continuous_scale"].astype(np.float32, copy=False)
        self.target_mean = standardization["target_mean"].astype(np.float32, copy=False)
        self.target_scale = standardization["target_scale"].astype(np.float32, copy=False)
        checkpoint = torch.load(artifact_dir / "model.pt", map_location="cpu")
        model_config = checkpoint["model_config"]
        self.model = tabm.TabM.make(
            n_num_features=int(model_config["n_continuous"]),
            cat_cardinalities=list(model_config["categorical_cardinalities"]),
            d_out=int(model_config["output_dim"]),
            n_blocks=int(model_config["depth"]),
            d_block=int(model_config["hidden_size"]),
            dropout=float(model_config["dropout"]),
            k=int(model_config["ensemble_size"]),
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.device = _resolve_device(device)
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.continuous_columns = tuple(self.config["continuous_columns"])
        self.categorical_columns = tuple(self.config["categorical_columns"])
        self.target_columns = tuple(self.config["target_columns"])
        self.categorical_specs = tuple(self.config["categorical_specs"])
        self.prediction_scale = float(self.config.get("prediction_scale", 1.0))
        self.online_epochs = int(self.config.get("online_epochs", 1))
        self.online_max_update_rows_per_date = int(self.config.get("online_max_update_rows_per_date", 20000))
        self.online_seed = int(self.config.get("seed", 37))
        self.optimizer = SophiaG(
            self.model.parameters(),
            lr=float(self.config.get("online_learning_rate", 1e-4)),
            weight_decay=8e-4,
        )
        self.cached_date_id: int | None = None
        self._cached_frames: list[pl.DataFrame] = []

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        continuous = _continuous_matrix(frame, self.continuous_columns)
        continuous = (continuous - self.continuous_mean) / self.continuous_scale
        categorical = _categorical_matrix(frame, self.categorical_columns, self.categorical_specs)
        predictions: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, frame.height, self.batch_size):
                end = min(frame.height, start + self.batch_size)
                x_num = torch.from_numpy(continuous[start:end]).to(self.device)
                x_cat = torch.from_numpy(categorical[start:end]).to(self.device)
                pred_scaled = self.model(x_num, x_cat)
                if pred_scaled.ndim == 3:
                    pred_scaled = pred_scaled.mean(dim=1)
                pred = pred_scaled[:, 0].detach().cpu().numpy().astype(np.float64, copy=False)
                predictions.append(pred)
        output = np.concatenate(predictions)
        output = output * float(self.target_scale[0]) + float(self.target_mean[0])
        return output * self.prediction_scale

    def update_from_lags(self, lags: pl.DataFrame | None) -> None:
        if lags is None or lags.is_empty() or self.cached_date_id is None or not self._cached_frames:
            return
        lag_date_id = _single_date_id(lags)
        update_frame = build_lagged_tabm_update_frame(self._cached_frames, lags, target_columns=self.target_columns)
        if update_frame.is_empty():
            return
        if self.online_max_update_rows_per_date > 0 and update_frame.height > self.online_max_update_rows_per_date:
            update_frame = update_frame.sample(
                n=self.online_max_update_rows_per_date,
                seed=self.online_seed + lag_date_id,
                shuffle=True,
            )
        self._train_online(update_frame)

    def cache_batch(self, frame: pl.DataFrame) -> None:
        if frame.is_empty():
            return
        date_ids = frame.select("date_id").unique()["date_id"].to_list()
        if len(date_ids) != 1:
            raise ValueError("one TabM cache batch must contain exactly one date_id")
        date_id = int(date_ids[0])
        columns = list(dict.fromkeys([*KEY_COLUMNS, WEIGHT_COLUMN, *self.continuous_columns, *self.categorical_columns]))
        cached = frame.select([_feature_expr(name).alias(name) if name not in set(KEY_COLUMNS) | {WEIGHT_COLUMN} else pl.col(name) for name in columns])
        if self.cached_date_id is None or self.cached_date_id != date_id:
            self.cached_date_id = date_id
            self._cached_frames = [cached]
        else:
            self._cached_frames.append(cached)

    def _train_online(self, frame: pl.DataFrame) -> None:
        continuous = _continuous_matrix(frame, self.continuous_columns)
        continuous = (continuous - self.continuous_mean) / self.continuous_scale
        categorical = _categorical_matrix(frame, self.categorical_columns, self.categorical_specs)
        target = frame.select(list(self.target_columns)).to_numpy().astype(np.float32, copy=True)
        target = (target - self.target_mean) / self.target_scale
        weight = frame[WEIGHT_COLUMN].to_numpy().astype(np.float32, copy=True)
        dataset = TensorDataset(
            torch.from_numpy(continuous.astype(np.float32, copy=False)),
            torch.from_numpy(categorical),
            torch.from_numpy(target.astype(np.float32, copy=False)),
            torch.from_numpy(weight),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, pin_memory=self.device.type == "cuda")
        self.model.train()
        for _ in range(self.online_epochs):
            for continuous_batch, categorical_batch, target_batch, weight_batch in loader:
                continuous_batch = continuous_batch.to(self.device, non_blocking=True)
                categorical_batch = categorical_batch.to(self.device, non_blocking=True)
                target_batch = target_batch.to(self.device, non_blocking=True)
                weight_batch = weight_batch.to(self.device, non_blocking=True)
                self.optimizer.zero_grad(set_to_none=True)
                pred = self.model(continuous_batch, categorical_batch)
                loss = _weighted_multi_loss(pred, target_batch, weight_batch, aux_weight=0.25)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
        self.model.eval()


class TreeArtifactPredictor:
    """Inference wrapper for the final XGBoost/LightGBM/Ridge tree artifact."""

    def __init__(self, artifact_dir: Path) -> None:
        with (artifact_dir / "tree_artifact.pkl").open("rb") as handle:
            artifact = pickle.load(handle)
        self.engines = tuple(artifact["engines"])
        self.base_features = tuple(artifact["base_features"])
        self.model_features = tuple(artifact["model_features"])
        self.models_by_engine = artifact["models_by_engine"]
        self.ridge_calibrator = artifact["ridge_calibrator"]
        self.weight_thresholds = artifact["weight_thresholds"]
        self.pred_abs_thresholds = artifact["pred_abs_thresholds"]
        self.simplex_weights = dict(artifact["simplex_weights"])

    def predict_components(self, frame: pl.DataFrame) -> pl.DataFrame:
        feature_columns = [
            name
            for name in dict.fromkeys([*self.base_features, *self.model_features])
            if name not in set(KEY_COLUMNS) and name != WEIGHT_COLUMN
        ]
        working = frame.select(list(KEY_COLUMNS) + [WEIGHT_COLUMN] + feature_columns)
        ridge_model = self.models_by_engine[self.engines[0]]["ridge_model"]
        ridge_x = working.select([pl.col(name).fill_null(0.0).cast(pl.Float64).alias(name) for name in self.base_features]).to_numpy()
        ridge_prediction = ridge_model.predict_array(ridge_x)
        output = working.select(list(KEY_COLUMNS) + [WEIGHT_COLUMN]).with_columns(
            pl.Series("ridge_prediction", ridge_prediction)
        )
        gbdt_x = working.select([pl.col(name).fill_null(0.0).cast(pl.Float32).alias(name) for name in self.model_features]).to_numpy()
        for engine in self.engines:
            models = self.models_by_engine[engine]["gbdt_models"]
            prediction = np.zeros(gbdt_x.shape[0], dtype=np.float64)
            for model in models:
                prediction += model.predict(gbdt_x).astype(np.float64, copy=False)
            prediction /= max(len(models), 1)
            output = output.with_columns(pl.Series(f"{engine}_prediction", prediction))
        output = _add_tree_regime_columns(output, self.weight_thresholds, self.pred_abs_thresholds)
        output = self.ridge_calibrator.apply(
            output,
            prediction="ridge_prediction",
            output="ridge_calibrated_prediction",
        )
        ensemble = np.zeros(output.height, dtype=np.float64)
        for column, weight in self.simplex_weights.items():
            if column in output.columns:
                ensemble += float(weight) * output[column].to_numpy().astype(np.float64, copy=False)
        return output.with_columns(pl.Series("tree_prediction", ensemble))


class ArtifactFeaturePredictor:
    """Base feature provider consumed by `KaggleRLSSubmissionPredictor`."""

    def __init__(self, base_artifact_dir: Path, *, device: str = "auto") -> None:
        self.tabm = TabMArtifactPredictor(base_artifact_dir / "tabm", device=device)
        self.tree = TreeArtifactPredictor(base_artifact_dir / "tree")

    def update_from_lags(self, lags: pl.DataFrame | None) -> None:
        self.tabm.update_from_lags(lags)

    def predict_features(self, test_with_lags: pl.DataFrame) -> pl.DataFrame:
        tabm_prediction = self.tabm.predict(test_with_lags)
        self.tabm.cache_batch(test_with_lags)
        tree_components = self.tree.predict_components(test_with_lags)
        return tree_components.with_columns(pl.Series("tabm_prediction", tabm_prediction)).select(
            [
                *KEY_COLUMNS,
                WEIGHT_COLUMN,
                "tabm_prediction",
                "tree_prediction",
                "xgboost_prediction",
                "lightgbm_prediction",
                "ridge_calibrated_prediction",
            ]
        )


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(raw)


def build_lagged_tabm_update_frame(
    cached_frames: Sequence[pl.DataFrame],
    lags: pl.DataFrame,
    *,
    target_columns: Sequence[str],
) -> pl.DataFrame:
    """Join previous-day cached model features to gateway lagged responders."""

    if not cached_frames or lags.is_empty():
        return pl.DataFrame()
    lag_date_id = _single_date_id(lags)
    shifted_cache = (
        pl.concat(cached_frames, how="vertical")
        .with_columns((pl.col("date_id") + 1).cast(lags.schema["date_id"]).alias("date_id"))
        .filter(pl.col("date_id") == lag_date_id)
    )
    if shifted_cache.is_empty():
        return pl.DataFrame()
    lag_columns = responder_lag_columns(target_columns)
    missing = set(lag_columns) - set(lags.columns)
    if missing:
        raise ValueError(f"lags missing target columns for TabM online update: {', '.join(sorted(missing))}")
    renamed_lags = lags.select(
        list(KEY_COLUMNS)
        + [
            pl.col(lag_column).alias(target_column)
            for lag_column, target_column in zip(lag_columns, target_columns, strict=True)
        ]
    )
    return shifted_cache.join(renamed_lags, on=list(KEY_COLUMNS), how="inner")


def _single_date_id(frame: pl.DataFrame) -> int:
    date_ids = frame.select("date_id").unique()["date_id"].to_list()
    if len(date_ids) != 1:
        raise ValueError("frame must contain exactly one date_id")
    return int(date_ids[0])


def _continuous_matrix(frame: pl.DataFrame, columns: Sequence[str]) -> np.ndarray:
    expressions = [_feature_expr(name).alias(name) for name in columns]
    return frame.select(expressions).to_numpy().astype(np.float32, copy=False)


def _feature_expr(name: str) -> pl.Expr:
    if name == "weight_feature":
        return pl.col("weight").fill_null(0.0).cast(pl.Float32)
    if name == "time_sin_967":
        return (pl.col("time_id") * (2.0 * math.pi / 967.0)).sin().cast(pl.Float32)
    if name == "time_cos_967":
        return (pl.col("time_id") * (2.0 * math.pi / 967.0)).cos().cast(pl.Float32)
    if name == "time_sin_483":
        return (pl.col("time_id") * (2.0 * math.pi / 483.0)).sin().cast(pl.Float32)
    if name == "time_cos_483":
        return (pl.col("time_id") * (2.0 * math.pi / 483.0)).cos().cast(pl.Float32)
    if name == "date_sin_20":
        return (pl.col("date_id") * (2.0 * math.pi / 20.0)).sin().cast(pl.Float32)
    if name == "date_cos_20":
        return (pl.col("date_id") * (2.0 * math.pi / 20.0)).cos().cast(pl.Float32)
    return pl.col(name).fill_null(0.0).cast(pl.Float32)


def _categorical_matrix(frame: pl.DataFrame, columns: Sequence[str], specs: Sequence[dict[str, int]]) -> np.ndarray:
    if not columns:
        return np.zeros((frame.height, 0), dtype=np.int64)
    encoded_columns: list[np.ndarray] = []
    for name, spec in zip(columns, specs, strict=True):
        values = frame[name].fill_null(-1).to_numpy().astype(np.int64, copy=False)
        encoded = values - int(spec["min_value"])
        invalid = (values < int(spec["min_value"])) | (values > int(spec["max_value"]))
        encoded[invalid] = int(spec["max_value"]) - int(spec["min_value"]) + 1
        encoded_columns.append(encoded)
    return np.stack(encoded_columns, axis=1).astype(np.int64, copy=False)


def _weighted_multi_loss(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, *, aux_weight: float) -> torch.Tensor:
    if pred.ndim == 3:
        target_for_loss = target[:, None, :]
        weight_for_loss = weight[:, None, None]
        per_target = torch.sum(weight_for_loss * (pred - target_for_loss).square(), dim=(0, 1)) / torch.clamp(
            torch.sum(weight) * pred.shape[1],
            min=1e-12,
        )
    else:
        per_target = torch.sum(weight[:, None] * (pred - target).square(), dim=0) / torch.clamp(torch.sum(weight), min=1e-12)
    if per_target.numel() == 1:
        return per_target[0]
    return per_target[0] + aux_weight * per_target[1:].mean()


def _add_tree_regime_columns(
    frame: pl.DataFrame,
    weight_thresholds: dict[str, float],
    pred_abs_thresholds: dict[str, float],
) -> pl.DataFrame:
    with_regimes = frame.with_columns(
        [
            (pl.col("time_id") // 100).cast(pl.Int16).alias("time_bucket"),
            (
                pl.when(pl.col("weight") <= weight_thresholds["q50"])
                .then(pl.lit("q00_q50"))
                .when(pl.col("weight") <= weight_thresholds["q90"])
                .then(pl.lit("q50_q90"))
                .when(pl.col("weight") <= weight_thresholds["q99"])
                .then(pl.lit("q90_q99"))
                .otherwise(pl.lit("q99_q100"))
                .alias("weight_bucket")
            ),
        ]
    )
    return add_abs_prediction_bucket(with_regimes, pred_abs_thresholds, prediction="ridge_prediction")
