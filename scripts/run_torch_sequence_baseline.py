"""Evaluate small Torch neural baselines on rolling Jane Street folds."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import numba as nb
except ModuleNotFoundError:  # pragma: no cover - depends on optional local environment
    nb = None

from janestreet.folds import DateFold, make_rolling_folds
from janestreet.linear import feature_columns_from_schema
from janestreet.official_lags import RESPONDER_COLUMNS, responder_lag_columns
from janestreet.paths import TRAIN_PARQUET_DIR


_NUMBA_AVAILABLE = nb is not None


@dataclass(frozen=True)
class Standardization:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    target_mean: float
    target_scale: float


class SophiaG(torch.optim.Optimizer):
    """Small Sophia-G style optimizer with diagonal gradient-square curvature."""

    def __init__(
        self,
        params,
        *,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.0,
        eps: float = 1e-12,
        clip: float = 1.0,
        update_period: int = 10,
    ) -> None:
        if lr <= 0.0:
            raise ValueError("lr must be positive")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("betas must be in [0, 1)")
        if rho <= 0.0 or eps <= 0.0 or clip <= 0.0 or update_period <= 0:
            raise ValueError("rho, eps, clip, and update_period must be positive")
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
                if grad.is_sparse:
                    raise RuntimeError("SophiaG does not support sparse gradients")
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


class MLPRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x[:, -1, :]).squeeze(-1)


class RecurrentRegressor(nn.Module):
    def __init__(self, cell: str, n_features: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1, :]).squeeze(-1)


class TCNRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, hidden_size, kernel_size=3, padding=2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=2),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.net(x.transpose(1, 2))[:, :, : x.shape[1]]
        return self.head(hidden[:, :, -1]).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Linear(n_features, hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=2,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(self.input(x))
        return self.head(hidden[:, -1, :]).squeeze(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="mlp,lstm,gru,tcn,transformer")
    parser.add_argument("--n-folds", type=int, default=1)
    parser.add_argument("--train-window", type=int, default=10)
    parser.add_argument("--valid-window", type=int, default=5)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--n-features", type=int, default=32)
    parser.add_argument("--id-columns", default="time_id,symbol_id")
    parser.add_argument("--use-official-lags", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--include-weight-feature", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sophia-beta1", type=float, default=0.965)
    parser.add_argument("--sophia-beta2", type=float, default=0.99)
    parser.add_argument("--sophia-rho", type=float, default=0.04)
    parser.add_argument("--sophia-clip", type=float, default=1.0)
    parser.add_argument("--sophia-update-period", type=int, default=10)
    parser.add_argument("--early-stopping-valid-days", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--max-train-rows", type=int, default=200_000)
    parser.add_argument("--max-valid-rows", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/torch_sequence_smoke"))
    args = parser.parse_args()

    _validate_args(args)
    _set_reproducibility(args.seed, args.torch_threads)

    models = _parse_models(args.models)
    id_columns = _parse_id_columns(args.id_columns)
    device = _resolve_device(args.device)
    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    schema = train.collect_schema()
    feature_columns = _select_feature_columns(
        schema,
        args.n_features,
        id_columns,
        use_official_lags=args.use_official_lags,
        include_weight_feature=args.include_weight_feature,
    )
    folds = _make_folds(train, args)

    rows: list[dict[str, float | int | str]] = []
    for fold in folds:
        train_frame = _collect_frame(train, fold.train_start, fold.train_end, feature_columns)
        early_days = min(args.early_stopping_valid_days, max(fold.train_days - 1, 0))
        fit_train_end = fold.train_end - early_days
        fit_frame = train_frame.filter(pl.col("date_id") <= fit_train_end)
        valid_context_start = max(fold.train_start, fold.valid_start - 1)
        valid_frame = _collect_frame(train, valid_context_start, fold.valid_end, feature_columns)
        standardization = _fit_standardization(fit_frame, feature_columns)
        train_arrays = _make_sequence_arrays(
            train_frame,
            feature_columns,
            sequence_length=args.sequence_length,
            target_start=fold.train_start,
            target_end=fit_train_end,
            standardization=standardization,
            max_rows=args.max_train_rows,
            seed=args.seed,
        )
        early_arrays = None
        if early_days > 0:
            early_arrays = _make_sequence_arrays(
                train_frame,
                feature_columns,
                sequence_length=args.sequence_length,
                target_start=fit_train_end + 1,
                target_end=fold.train_end,
                standardization=standardization,
                max_rows=max(args.batch_size * 4, args.max_train_rows // 10),
                seed=args.seed + 2,
            )
        valid_arrays = _make_sequence_arrays(
            valid_frame,
            feature_columns,
            sequence_length=args.sequence_length,
            target_start=fold.valid_start,
            target_end=fold.valid_end,
            standardization=standardization,
            max_rows=args.max_valid_rows,
            seed=args.seed + 1,
        )
        for model_name in models:
            model = _make_model(
                model_name,
                n_features=len(feature_columns),
                hidden_size=args.hidden_size,
                dropout=args.dropout,
            )
            fit_result = _fit_model(model, train_arrays, early_arrays, standardization, args, device)
            evaluation = _evaluate_model(model, valid_arrays, standardization, args, device)
            rows.append(
                {
                    **_fold_metadata(fold),
                    "model": model_name,
                    "n_features": len(feature_columns),
                    "sequence_length": args.sequence_length,
                    "train_rows": int(train_arrays[0].shape[0]),
                    "valid_rows": int(valid_arrays[0].shape[0]),
                    "early_rows": 0 if early_arrays is None else int(early_arrays[0].shape[0]),
                    **fit_result,
                    **evaluation,
                }
            )

    results = pl.DataFrame(rows)
    summary = _summary_by_model(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(args.output_dir / "torch_sequence_by_fold.csv")
    summary.write_csv(args.output_dir / "torch_sequence_summary.csv")
    report = {
        "experiment": "torch_sequence_baseline",
        "models": models,
        "n_folds": args.n_folds,
        "train_window": args.train_window,
        "valid_window": args.valid_window,
        "n_features": args.n_features,
        "id_columns": id_columns,
        "use_official_lags": args.use_official_lags,
        "include_weight_feature": args.include_weight_feature,
        "sequence_length": args.sequence_length,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "optimizer": "sophia_g",
        "numba_sequence_builder": _NUMBA_AVAILABLE,
        "early_stopping_valid_days": args.early_stopping_valid_days,
        "early_stopping_patience": args.early_stopping_patience,
        "device": str(device),
        "summary": summary.to_dicts(),
    }
    (args.output_dir / "torch_sequence_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(results)
    print(summary)
    print(f"Wrote {args.output_dir}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.n_folds <= 0:
        raise ValueError("--n-folds must be positive")
    if args.train_window <= 0 or args.valid_window <= 0:
        raise ValueError("--train-window and --valid-window must be positive")
    if args.n_features <= 0:
        raise ValueError("--n-features must be positive")
    if args.sequence_length <= 0:
        raise ValueError("--sequence-length must be positive")
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.hidden_size <= 0:
        raise ValueError("--hidden-size must be positive")
    if args.max_train_rows < 0 or args.max_valid_rows < 0:
        raise ValueError("row limits must be non-negative")
    if args.early_stopping_valid_days < 0:
        raise ValueError("--early-stopping-valid-days must be non-negative")
    if args.early_stopping_patience < 0:
        raise ValueError("--early-stopping-patience must be non-negative")
    if args.sophia_update_period <= 0:
        raise ValueError("--sophia-update-period must be positive")


def _set_reproducibility(seed: int, torch_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)


def _parse_models(raw: str) -> tuple[str, ...]:
    models = tuple(dict.fromkeys(part.strip().lower() for part in raw.split(",") if part.strip()))
    allowed = {"mlp", "lstm", "gru", "tcn", "transformer"}
    unknown = sorted(set(models) - allowed)
    if unknown:
        raise ValueError(f"unknown models: {', '.join(unknown)}")
    if not models:
        raise ValueError("at least one model is required")
    return models


def _parse_id_columns(raw: str) -> tuple[str, ...]:
    allowed = {"time_id", "symbol_id"}
    columns = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    unknown = sorted(set(columns) - allowed)
    if unknown:
        raise ValueError(f"unknown id columns: {', '.join(unknown)}")
    return columns


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(raw)


def _select_feature_columns(
    schema: pl.Schema,
    n_features: int,
    id_columns: tuple[str, ...],
    *,
    use_official_lags: bool = False,
    include_weight_feature: bool = False,
) -> tuple[str, ...]:
    base = feature_columns_from_schema(schema)[:n_features]
    if len(base) < n_features:
        raise ValueError(f"requested {n_features} features, only {len(base)} available")
    extras: list[str] = []
    if include_weight_feature:
        extras.append("weight_feature")
    if use_official_lags:
        extras.extend(responder_lag_columns(RESPONDER_COLUMNS))
    return tuple(dict.fromkeys([*base, *extras, *id_columns]))


def _collect_frame(
    data: pl.LazyFrame,
    start: int,
    end: int,
    feature_columns: Sequence[str],
) -> pl.DataFrame:
    metadata_columns = {"date_id", "time_id", "symbol_id"}
    model_feature_columns = [name for name in feature_columns if name not in metadata_columns]
    lag_columns = tuple(name for name in model_feature_columns if name.endswith("_lag_1"))
    base_feature_columns = [name for name in model_feature_columns if name not in lag_columns and name != "weight_feature"]
    base = data.filter(pl.col("date_id").is_between(start, end))
    if lag_columns:
        lag_sources = tuple(name.removesuffix("_lag_1") for name in lag_columns)
        lags = data.filter(pl.col("date_id").is_between(start - 1, end - 1)).select(
            [
                (pl.col("date_id") + 1).cast(pl.Int32).alias("date_id"),
                pl.col("time_id").cast(pl.Int32),
                pl.col("symbol_id").cast(pl.Int32),
            ]
            + [pl.col(source).cast(pl.Float32).alias(target) for source, target in zip(lag_sources, lag_columns, strict=True)]
        )
        base = base.join(lags, on=["date_id", "time_id", "symbol_id"], how="left")
    select_exprs = [
        pl.col("date_id").cast(pl.Int32),
        pl.col("time_id").cast(pl.Int32),
        pl.col("symbol_id").cast(pl.Int32),
    ]
    select_exprs.extend(pl.col(name).fill_null(0.0).cast(pl.Float32).alias(name) for name in base_feature_columns)
    if "weight_feature" in model_feature_columns:
        select_exprs.append(pl.col("weight").cast(pl.Float32).alias("weight_feature"))
    select_exprs.extend(pl.col(name).fill_null(0.0).cast(pl.Float32).alias(name) for name in lag_columns)
    select_exprs.extend([pl.col("responder_6").cast(pl.Float32), pl.col("weight").cast(pl.Float32)])
    return (
        base.select(select_exprs)
        .sort(["symbol_id", "date_id", "time_id"])
        .collect()
    )


def _fit_standardization(frame: pl.DataFrame, feature_columns: Sequence[str]) -> Standardization:
    x = frame.select(list(feature_columns)).to_numpy().astype(np.float32, copy=False)
    y = frame["responder_6"].to_numpy().astype(np.float32, copy=False)
    w = frame["weight"].to_numpy().astype(np.float32, copy=False)
    feature_mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale = x.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_scale[~np.isfinite(feature_scale)] = 1.0
    feature_scale[feature_scale <= 1e-6] = 1.0
    weight_sum = float(np.sum(w))
    if weight_sum <= 0.0:
        raise ValueError("training weights must have positive sum")
    target_mean = float(np.sum(w * y) / weight_sum)
    target_var = float(np.sum(w * (y - target_mean) ** 2) / weight_sum)
    target_scale = float(np.sqrt(max(target_var, 1e-6)))
    return Standardization(feature_mean, feature_scale, target_mean, target_scale)


def _make_sequence_arrays(
    frame: pl.DataFrame,
    feature_columns: Sequence[str],
    *,
    sequence_length: int,
    target_start: int,
    target_end: int,
    standardization: Standardization,
    max_rows: int = 0,
    seed: int = 17,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if frame.height == 0:
        raise ValueError("cannot create sequences from an empty frame")

    sorted_frame = frame.sort(["symbol_id", "date_id", "time_id"])
    features = sorted_frame.select(list(feature_columns)).to_numpy().astype(np.float32, copy=False)
    features = (features - standardization.feature_mean) / standardization.feature_scale
    date_id = sorted_frame["date_id"].to_numpy()
    symbol_id = sorted_frame["symbol_id"].to_numpy()
    target = sorted_frame["responder_6"].to_numpy().astype(np.float32, copy=False)
    weight = sorted_frame["weight"].to_numpy().astype(np.float32, copy=False)
    group_start_by_row = _symbol_group_starts(symbol_id)

    target_mask = (date_id >= target_start) & (date_id <= target_end)
    row_indices = np.flatnonzero(target_mask)
    if row_indices.size == 0:
        raise ValueError(f"no target rows for date_id {target_start}-{target_end}")
    if max_rows > 0 and row_indices.size > max_rows:
        rng = np.random.default_rng(seed)
        row_indices = np.sort(rng.choice(row_indices, size=max_rows, replace=False))

    sequences = np.zeros((row_indices.size, sequence_length, features.shape[1]), dtype=np.float32)
    _fill_sequence_windows(features, row_indices, group_start_by_row, sequences)
    return sequences, target[row_indices].astype(np.float32), weight[row_indices].astype(np.float32)


def _fill_sequence_windows(
    features: np.ndarray,
    row_indices: np.ndarray,
    group_start_by_row: np.ndarray,
    sequences: np.ndarray,
) -> None:
    if _fill_sequence_windows_numba is not None:
        _fill_sequence_windows_numba(features, row_indices, group_start_by_row, sequences)
        return
    _fill_sequence_windows_python(features, row_indices, group_start_by_row, sequences)


def _fill_sequence_windows_python(
    features: np.ndarray,
    row_indices: np.ndarray,
    group_start_by_row: np.ndarray,
    sequences: np.ndarray,
) -> None:
    sequence_length = sequences.shape[1]
    for out_idx, row_idx in enumerate(row_indices):
        start = max(int(group_start_by_row[row_idx]), row_idx - sequence_length + 1)
        window = features[start : row_idx + 1]
        sequences[out_idx, -window.shape[0] :, :] = window


if nb is not None:

    @nb.njit(cache=True)
    def _fill_sequence_windows_numba(  # pragma: no cover - exercised only when numba is installed
        features: np.ndarray,
        row_indices: np.ndarray,
        group_start_by_row: np.ndarray,
        sequences: np.ndarray,
    ) -> None:
        sequence_length = sequences.shape[1]
        n_features = features.shape[1]
        for out_idx in range(row_indices.shape[0]):
            row_idx = int(row_indices[out_idx])
            start = int(group_start_by_row[row_idx])
            lower = row_idx - sequence_length + 1
            if lower > start:
                start = lower
            window_length = row_idx - start + 1
            offset = sequence_length - window_length
            for src_idx in range(start, row_idx + 1):
                dst_idx = offset + src_idx - start
                for feature_idx in range(n_features):
                    sequences[out_idx, dst_idx, feature_idx] = features[src_idx, feature_idx]

else:
    _fill_sequence_windows_numba = None


def _symbol_group_starts(symbol_id: np.ndarray) -> np.ndarray:
    starts = np.zeros(symbol_id.shape[0], dtype=np.int64)
    if symbol_id.size == 0:
        return starts
    boundaries = np.flatnonzero(symbol_id[1:] != symbol_id[:-1]) + 1
    group_starts = np.concatenate([np.array([0]), boundaries])
    group_ends = np.concatenate([boundaries, np.array([symbol_id.size])])
    for start, end in zip(group_starts, group_ends, strict=True):
        starts[start:end] = start
    return starts

def _make_model(model_name: str, *, n_features: int, hidden_size: int, dropout: float) -> nn.Module:
    if model_name == "mlp":
        return MLPRegressor(n_features, hidden_size, dropout)
    if model_name in {"lstm", "gru"}:
        return RecurrentRegressor(model_name, n_features, hidden_size, dropout)
    if model_name == "tcn":
        return TCNRegressor(n_features, hidden_size, dropout)
    if model_name == "transformer":
        return TransformerRegressor(n_features, hidden_size, dropout)
    raise ValueError(f"unknown model: {model_name}")


def _fit_model(
    model: nn.Module,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    early_arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    standardization: Standardization,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | int]:
    x, y, weight = arrays
    target = ((y - standardization.target_mean) / standardization.target_scale).astype(np.float32, copy=False)
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(target.astype(np.float32, copy=False)),
        torch.from_numpy(weight.astype(np.float32, copy=False)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model.to(device)
    optimizer = SophiaG(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.sophia_beta1, args.sophia_beta2),
        rho=args.sophia_rho,
        weight_decay=args.weight_decay,
        clip=args.sophia_clip,
        update_period=args.sophia_update_period,
    )
    history: list[float] = []
    best_epoch = 0
    best_valid_loss = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    stale_epochs = 0
    for _ in range(args.epochs):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for batch_x, batch_y, batch_w in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_w = batch_w.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_x)
            loss = _weighted_mse(pred, batch_y, batch_w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_weight = float(batch_w.sum().detach().cpu())
            loss_sum += float(loss.detach().cpu()) * batch_weight
            weight_sum += batch_weight
        train_loss = loss_sum / max(weight_sum, 1.0)
        history.append(train_loss)
        valid_loss = train_loss
        if early_arrays is not None:
            valid_loss = _scaled_validation_loss(model, early_arrays, standardization, args, device)
        if valid_loss < best_valid_loss - args.early_stopping_min_delta:
            best_valid_loss = valid_loss
            best_epoch = len(history)
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if args.early_stopping_patience > 0 and stale_epochs >= args.early_stopping_patience:
                break
    model.load_state_dict(best_state)
    return {
        "trained_epochs": len(history),
        "best_epoch": best_epoch,
        "last_train_loss": float(history[-1]),
        "best_internal_valid_loss": float(best_valid_loss),
    }


def _weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.sum(weight * (pred - target).square()) / torch.clamp(torch.sum(weight), min=1e-12)


def _scaled_validation_loss(
    model: nn.Module,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    standardization: Standardization,
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    x, y, weight = arrays
    target = ((y - standardization.target_mean) / standardization.target_scale).astype(np.float32, copy=False)
    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(target),
        torch.from_numpy(weight.astype(np.float32, copy=False)),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    model.eval()
    loss_sum = 0.0
    weight_sum = 0.0
    with torch.no_grad():
        for batch_x, batch_y, batch_w in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            batch_w = batch_w.to(device, non_blocking=True)
            loss = _weighted_mse(model(batch_x), batch_y, batch_w)
            batch_weight = float(batch_w.sum().detach().cpu())
            loss_sum += float(loss.detach().cpu()) * batch_weight
            weight_sum += batch_weight
    return loss_sum / max(weight_sum, 1.0)


def _evaluate_model(
    model: nn.Module,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    standardization: Standardization,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    x, y, weight = arrays
    dataset = TensorDataset(torch.from_numpy(x))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for (batch_x,) in loader:
            pred_scaled = model(batch_x.to(device, non_blocking=True)).detach().cpu().numpy()
            preds.append(pred_scaled.astype(np.float64, copy=False))
    pred = np.concatenate(preds)
    pred = pred * standardization.target_scale + standardization.target_mean
    y64 = y.astype(np.float64, copy=False)
    w64 = weight.astype(np.float64, copy=False)
    err = y64 - pred
    numerator = float(np.sum(w64 * err * err))
    denominator = float(np.sum(w64 * y64 * y64))
    if denominator <= 0.0:
        raise ValueError("validation target energy must be positive")
    prediction_mean = float(np.mean(pred))
    prediction_std = float(np.std(pred))
    return {
        "numerator": numerator,
        "denominator": denominator,
        "weighted_zero_mean_r2": 1.0 - numerator / denominator,
        "weight_sum": float(np.sum(w64)),
        "prediction_mean": prediction_mean,
        "prediction_std": prediction_std,
    }


def _make_folds(train: pl.LazyFrame, args: argparse.Namespace) -> list[DateFold]:
    bounds = train.select(
        pl.min("date_id").alias("min_date_id"),
        pl.max("date_id").alias("max_date_id"),
    ).collect()
    return make_rolling_folds(
        min_date_id=int(bounds["min_date_id"][0]),
        max_date_id=int(bounds["max_date_id"][0]),
        n_folds=args.n_folds,
        train_window=args.train_window,
        valid_window=args.valid_window,
        gap=args.gap,
    )


def _summary_by_model(results: pl.DataFrame) -> pl.DataFrame:
    return (
        results.group_by("model")
        .agg(
            pl.len().alias("folds"),
            pl.col("weighted_zero_mean_r2").mean().alias("mean_r2"),
            pl.col("weighted_zero_mean_r2").std().alias("std_r2"),
            pl.col("weighted_zero_mean_r2").min().alias("min_r2"),
            pl.col("weighted_zero_mean_r2").max().alias("max_r2"),
            (1.0 - pl.col("numerator").sum() / pl.col("denominator").sum()).alias("global_r2"),
            pl.col("train_rows").sum().alias("train_rows"),
            pl.col("valid_rows").sum().alias("validation_rows"),
        )
        .sort(["global_r2", "min_r2"], descending=[True, True])
    )


def _fold_metadata(fold: DateFold) -> dict[str, int | str]:
    return {
        "fold": fold.name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "valid_start": fold.valid_start,
        "valid_end": fold.valid_end,
    }


if __name__ == "__main__":
    main()
