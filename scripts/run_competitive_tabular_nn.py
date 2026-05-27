"""Train competitive causal tabular neural baselines for Jane Street folds."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import tabm

from janestreet.folds import DateFold, make_rolling_folds
from janestreet.linear import feature_columns_from_schema
from janestreet.official_lags import RESPONDER_COLUMNS, responder_lag_columns
from janestreet.paths import TRAIN_PARQUET_DIR


TARGET_COLUMN = "responder_6"
DEFAULT_AUX_TARGETS = ("responder_3", "responder_7", "responder_8")
DEFAULT_CATEGORICAL_FEATURES = ("feature_09", "feature_10", "feature_11")
BATCH_FEATURE_MODES = (
    "batch_rank",
    "batch_mean",
    "batch_demean",
    "batch_std",
    "batch_zscore",
    "row_missing_count",
    "row_missing_frac",
    "batch_missing_frac",
)
BATCH_FEATURE_SUFFIXES = {
    "batch_rank": "__batch_rank",
    "batch_mean": "__batch_mean",
    "batch_demean": "__batch_demean",
    "batch_std": "__batch_std",
    "batch_zscore": "__batch_zscore",
}
DERIVED_TARGET_SPECS = {
    "responder_6_roll8": {"kind": "rolling_mean", "source": "responder_6", "window": 8},
    "responder_6_roll60": {"kind": "rolling_mean", "source": "responder_6", "window": 60},
    "responder_8_roll8": {"kind": "rolling_mean", "source": "responder_8", "window": 8},
    "responder_6_abs": {"kind": "abs", "source": "responder_6"},
    "responder_6_clip3": {"kind": "clip", "source": "responder_6", "limit": 3.0},
    "responder_6_sign": {"kind": "sign", "source": "responder_6"},
}


@dataclass(frozen=True)
class CategoricalSpec:
    name: str
    min_value: int
    max_value: int

    @property
    def unknown_index(self) -> int:
        return self.max_value - self.min_value + 1

    @property
    def num_classes(self) -> int:
        return self.unknown_index + 1


@dataclass(frozen=True)
class Standardization:
    continuous_mean: np.ndarray
    continuous_scale: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    categorical_specs: tuple[CategoricalSpec, ...]


@dataclass(frozen=True)
class PredictionCalibration:
    scale: float = 1.0


@dataclass(frozen=True)
class SamplePlan:
    policy: str
    row_count: int
    sample_max_rows: int
    uniform_bps: int = 10_000
    high_weight_threshold: float | None = None
    high_weight_bps: int = 10_000
    rest_bps: int = 10_000


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


class TabularMLP(nn.Module):
    """TabM-lite tabular MLP with categorical embeddings and multi-head output."""

    def __init__(
        self,
        *,
        n_continuous: int,
        categorical_cardinalities: Sequence[int],
        hidden_size: int,
        depth: int,
        dropout: float,
        output_dim: int,
        ensemble_size: int,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        if ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive")
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, _embedding_dim(cardinality)) for cardinality in categorical_cardinalities]
        )
        embedded_size = sum(embedding.embedding_dim for embedding in self.embeddings)
        input_size = n_continuous + embedded_size
        layers: list[nn.Module] = [nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.SiLU()]
        for _ in range(depth - 1):
            layers.extend(
                [
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size, hidden_size),
                    nn.LayerNorm(hidden_size),
                    nn.SiLU(),
                ]
            )
        self.backbone = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, output_dim * ensemble_size)
        self.output_dim = output_dim
        self.ensemble_size = ensemble_size

    def forward(self, continuous: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        if self.embeddings:
            embedded = [embedding(categorical[:, idx]) for idx, embedding in enumerate(self.embeddings)]
            x = torch.cat([continuous, *embedded], dim=1)
        else:
            x = continuous
        output = self.head(self.dropout(self.backbone(x)))
        output = output.view(output.shape[0], self.ensemble_size, self.output_dim)
        return output.mean(dim=1)


class BatchDeepSetMLP(nn.Module):
    """Set-aware tabular MLP with learned context per observed (date_id,time_id) batch."""

    requires_batch_ids = True

    def __init__(
        self,
        *,
        n_continuous: int,
        categorical_cardinalities: Sequence[int],
        hidden_size: int,
        depth: int,
        dropout: float,
        output_dim: int,
        ensemble_size: int,
    ) -> None:
        super().__init__()
        if depth <= 0:
            raise ValueError("depth must be positive")
        if ensemble_size <= 0:
            raise ValueError("ensemble_size must be positive")
        self.embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, _embedding_dim(cardinality)) for cardinality in categorical_cardinalities]
        )
        embedded_size = sum(embedding.embedding_dim for embedding in self.embeddings)
        input_size = n_continuous + embedded_size
        self.row_in = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size), nn.SiLU())
        row_layers: list[nn.Module] = []
        for _ in range(depth - 1):
            row_layers.extend([nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size), nn.LayerNorm(hidden_size), nn.SiLU()])
        self.row_backbone = nn.Sequential(*row_layers)
        self.context = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_dim * ensemble_size),
        )
        self.output_dim = output_dim
        self.ensemble_size = ensemble_size

    def forward(self, continuous: torch.Tensor, categorical: torch.Tensor, batch_id: torch.Tensor | None = None) -> torch.Tensor:
        if batch_id is None:
            raise ValueError("BatchDeepSetMLP requires batch_id")
        if self.embeddings:
            embedded = [embedding(categorical[:, idx]) for idx, embedding in enumerate(self.embeddings)]
            x = torch.cat([continuous, *embedded], dim=1)
        else:
            x = continuous
        row = self.row_backbone(self.row_in(x))
        _unique, inverse = torch.unique(batch_id.to(torch.long), sorted=True, return_inverse=True)
        n_groups = int(inverse.max().item()) + 1 if inverse.numel() else 0
        if n_groups <= 0:
            raise ValueError("BatchDeepSetMLP received an empty batch")
        sums = row.new_zeros((n_groups, row.shape[1]))
        sums.index_add_(0, inverse, row)
        counts = torch.bincount(inverse, minlength=n_groups).to(dtype=row.dtype, device=row.device).clamp_min_(1.0)
        means = sums / counts[:, None]
        context = means[inverse]
        output = self.context(torch.cat([row, context, row - context], dim=1))
        output = output.view(output.shape[0], self.ensemble_size, self.output_dim)
        return output.mean(dim=1)


def make_tabular_model(
    *,
    model_type: str,
    n_continuous: int,
    categorical_cardinalities: Sequence[int],
    hidden_size: int,
    depth: int,
    dropout: float,
    output_dim: int,
    ensemble_size: int,
) -> nn.Module:
    if model_type == "mlp_ensemble":
        return TabularMLP(
            n_continuous=n_continuous,
            categorical_cardinalities=categorical_cardinalities,
            hidden_size=hidden_size,
            depth=depth,
            dropout=dropout,
            output_dim=output_dim,
            ensemble_size=ensemble_size,
        )
    if model_type == "tabm":
        return tabm.TabM.make(
            n_num_features=n_continuous,
            cat_cardinalities=list(categorical_cardinalities),
            d_out=output_dim,
            n_blocks=depth,
            d_block=hidden_size,
            dropout=dropout,
            k=ensemble_size,
        )
    if model_type == "batch_deepset":
        return BatchDeepSetMLP(
            n_continuous=n_continuous,
            categorical_cardinalities=categorical_cardinalities,
            hidden_size=hidden_size,
            depth=depth,
            dropout=dropout,
            output_dim=output_dim,
            ensemble_size=ensemble_size,
        )
    raise ValueError(f"unknown model type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", choices=["mlp_ensemble", "tabm", "batch_deepset"], default="tabm")
    parser.add_argument("--fold-protocol", choices=["rolling", "competition"], default="rolling")
    parser.add_argument("--n-folds", type=int, default=1)
    parser.add_argument("--train-window", type=int, default=40)
    parser.add_argument("--valid-window", type=int, default=10)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--competition-train-start", type=int, default=700)
    parser.add_argument("--competition-valid-window", type=int, default=200)
    parser.add_argument("--competition-gap", type=int, default=200)
    parser.add_argument("--max-date-id", type=int, default=-1)
    parser.add_argument("--fold-start-index", type=int, default=0)
    parser.add_argument("--fold-limit", type=int, default=0)
    parser.add_argument("--n-features", type=int, default=79)
    parser.add_argument("--use-official-lags", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-time-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-weight-feature", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-feature-columns", default="")
    parser.add_argument("--batch-feature-modes", default="")
    parser.add_argument("--aux-targets", default=",".join(DEFAULT_AUX_TARGETS))
    parser.add_argument("--derived-aux-targets", default="")
    parser.add_argument("--center-target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-train-rows", type=int, default=600_000)
    parser.add_argument("--max-valid-rows", type=int, default=0)
    parser.add_argument("--train-sample-policy", choices=["uniform", "weight_tail"], default="uniform")
    parser.add_argument("--train-weight-tail-quantile", type=float, default=0.80)
    parser.add_argument("--eval-chunk-days", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=8e-4)
    parser.add_argument("--aux-loss-weight", type=float, default=0.25)
    parser.add_argument("--loss-type", choices=["mse", "huber"], default="mse")
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--calibrate-prediction-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-calibration-scale", type=float, default=2.0)
    parser.add_argument("--early-stopping-valid-days", type=int, default=5)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-5)
    parser.add_argument("--online-update", action="store_true")
    parser.add_argument("--online-learning-rate", type=float, default=1e-4)
    parser.add_argument("--online-epochs", type=int, default=1)
    parser.add_argument("--online-max-update-rows-per-date", type=int, default=20000)
    parser.add_argument("--sophia-beta1", type=float, default=0.965)
    parser.add_argument("--sophia-beta2", type=float, default=0.99)
    parser.add_argument("--sophia-rho", type=float, default=0.04)
    parser.add_argument("--sophia-clip", type=float, default=1.0)
    parser.add_argument("--sophia-update-period", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--min-mem-available-gb", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/competitive_tabular_nn_smoke"))
    args = parser.parse_args()

    _validate_args(args)
    _set_reproducibility(args.seed, args.torch_threads)
    device = _resolve_device(args.device)

    train = pl.scan_parquet(str(TRAIN_PARQUET_DIR))
    if args.max_date_id >= 0:
        train = train.filter(pl.col("date_id") <= args.max_date_id)
    schema = train.collect_schema()
    batch_feature_modes = _parse_batch_feature_modes(args.batch_feature_modes)
    batch_feature_sources = _parse_feature_list(args.batch_feature_columns)
    continuous_columns, categorical_columns = _select_model_columns(
        schema,
        n_features=args.n_features,
        use_official_lags=args.use_official_lags,
        include_time_features=args.include_time_features,
        include_weight_feature=args.include_weight_feature,
        batch_feature_sources=batch_feature_sources,
        batch_feature_modes=batch_feature_modes,
    )
    resolved_batch_feature_sources = _resolve_batch_feature_sources(
        continuous_columns,
        batch_feature_sources,
        batch_feature_modes,
    )
    args.resolved_batch_feature_sources = resolved_batch_feature_sources
    aux_targets = _parse_aux_targets(args.aux_targets)
    derived_aux_targets = _parse_aux_targets(args.derived_aux_targets)
    target_columns = (TARGET_COLUMN, *aux_targets)
    if derived_aux_targets:
        target_columns = tuple(dict.fromkeys([*target_columns, *derived_aux_targets]))
    folds = _select_requested_folds(_make_folds(train, args), args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, float | int | str | bool]] = []
    for fold in folds:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _assert_min_mem_available(args.min_mem_available_gb, f"before fold {fold.name}")
        fit_end = fold.train_end - min(args.early_stopping_valid_days, max(fold.train_days - 1, 0))
        if fit_end < fold.train_start:
            fit_end = fold.train_end
        train_frame = _collect_model_frame(
            train,
            fold.train_start,
            fit_end,
            continuous_columns,
            categorical_columns,
            target_columns,
            sample_max_rows=args.max_train_rows,
            seed=args.seed,
            sample_policy=args.train_sample_policy,
            weight_tail_quantile=args.train_weight_tail_quantile,
            batch_feature_sources=resolved_batch_feature_sources,
        )
        _assert_min_mem_available(args.min_mem_available_gb, f"after collecting train frame for {fold.name}")
        early_frame = None
        if fit_end < fold.train_end:
            early_frame = _collect_model_frame(
                train,
                fit_end + 1,
                fold.train_end,
                continuous_columns,
                categorical_columns,
                target_columns,
                sample_max_rows=max(args.batch_size * 4, args.max_train_rows // 8),
                seed=args.seed + 3,
                batch_feature_sources=resolved_batch_feature_sources,
            )
            _assert_min_mem_available(args.min_mem_available_gb, f"after collecting early frame for {fold.name}")
        standardization = _fit_standardization(
            train_frame,
            continuous_columns,
            categorical_columns,
            target_columns,
            center_target=args.center_target,
        )
        model = make_tabular_model(
            model_type=args.model_type,
            n_continuous=len(continuous_columns),
            categorical_cardinalities=[spec.num_classes for spec in standardization.categorical_specs],
            hidden_size=args.hidden_size,
            depth=args.depth,
            dropout=args.dropout,
            output_dim=len(target_columns),
            ensemble_size=args.ensemble_size,
        )
        fit_result = _fit_model(model, train_frame, early_frame, standardization, continuous_columns, categorical_columns, target_columns, args, device)
        calibration = PredictionCalibration()
        if args.calibrate_prediction_scale and early_frame is not None and early_frame.height > 0:
            calibration = _fit_prediction_calibration(
                model,
                early_frame,
                standardization,
                continuous_columns,
                categorical_columns,
                target_columns,
                args,
                device,
            )
        if args.online_update:
            evaluation = _evaluate_online(
                model,
                train,
                fold,
                standardization,
                continuous_columns,
                categorical_columns,
                target_columns,
                calibration,
                args,
                device,
                prediction_output_dir=args.output_dir / "validation_predictions" if args.save_predictions else None,
            )
        else:
            evaluation = _evaluate_offline(
                model,
                train,
                fold,
                standardization,
                continuous_columns,
                categorical_columns,
                target_columns,
                calibration,
                args,
                device,
                prediction_output_dir=args.output_dir / "validation_predictions" if args.save_predictions else None,
            )
        fold_rows.append(
            {
                **_fold_metadata(fold),
                "model": args.model_type,
                "use_official_lags": args.use_official_lags,
                "online_update": args.online_update,
                "n_continuous": len(continuous_columns),
                "n_categorical": len(categorical_columns),
                "train_rows": train_frame.height,
                "early_rows": 0 if early_frame is None else early_frame.height,
                "prediction_scale": calibration.scale,
                **fit_result,
                **evaluation,
            }
        )
        del train_frame, early_frame, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = pl.DataFrame(fold_rows)
    summary = _summary(results)
    results.write_csv(args.output_dir / "competitive_tabular_nn_by_fold.csv")
    summary.write_csv(args.output_dir / "competitive_tabular_nn_summary.csv")
    report = {
        "experiment": "competitive_tabular_nn",
        "model_type": args.model_type,
        "fold_protocol": args.fold_protocol,
        "use_official_lags": args.use_official_lags,
        "online_update": args.online_update,
        "continuous_columns": continuous_columns,
        "categorical_columns": categorical_columns,
        "target_columns": target_columns,
        "derived_aux_targets": derived_aux_targets,
        "batch_feature_sources": resolved_batch_feature_sources,
        "batch_feature_modes": batch_feature_modes,
        "device": str(device),
        "params": {
            "center_target": args.center_target,
            "max_train_rows": args.max_train_rows,
            "max_valid_rows": args.max_valid_rows,
            "train_sample_policy": args.train_sample_policy,
            "train_weight_tail_quantile": args.train_weight_tail_quantile,
            "batch_feature_columns": args.batch_feature_columns,
            "batch_feature_modes": args.batch_feature_modes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "depth": args.depth,
            "dropout": args.dropout,
            "ensemble_size": args.ensemble_size,
            "learning_rate": args.learning_rate,
            "loss_type": args.loss_type,
            "huber_delta": args.huber_delta,
            "online_learning_rate": args.online_learning_rate,
            "online_epochs": args.online_epochs,
            "save_predictions": args.save_predictions,
            "calibrate_prediction_scale": args.calibrate_prediction_scale,
            "max_calibration_scale": args.max_calibration_scale,
            "min_mem_available_gb": args.min_mem_available_gb,
        },
        "summary": summary.to_dicts(),
    }
    (args.output_dir / "competitive_tabular_nn_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(results)
    print(summary)
    print(f"Wrote {args.output_dir}")


def _validate_args(args: argparse.Namespace) -> None:
    for name in ["n_folds", "train_window", "valid_window", "n_features", "epochs", "batch_size", "hidden_size", "depth", "ensemble_size", "eval_chunk_days"]:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.gap < 0 or args.competition_gap < 0:
        raise ValueError("gaps must be non-negative")
    if args.fold_start_index < 0 or args.fold_limit < 0:
        raise ValueError("fold slicing settings must be non-negative")
    if args.max_train_rows < 0 or args.max_valid_rows < 0:
        raise ValueError("row limits must be non-negative")
    if args.min_mem_available_gb < 0.0:
        raise ValueError("--min-mem-available-gb must be non-negative")
    if not 0.0 < args.train_weight_tail_quantile < 1.0:
        raise ValueError("--train-weight-tail-quantile must be in (0, 1)")
    if args.aux_loss_weight < 0.0:
        raise ValueError("--aux-loss-weight must be non-negative")
    if args.huber_delta <= 0.0:
        raise ValueError("--huber-delta must be positive")
    if args.max_calibration_scale <= 0.0:
        raise ValueError("--max-calibration-scale must be positive")
    if args.online_epochs <= 0 or args.online_max_update_rows_per_date < 0:
        raise ValueError("online update settings must be positive/non-negative")
    _parse_batch_feature_modes(args.batch_feature_modes)
    batch_feature_sources = _parse_feature_list(args.batch_feature_columns)
    invalid_sources = [name for name in batch_feature_sources if not name.startswith("feature_")]
    if invalid_sources:
        raise ValueError(f"batch feature sources must be feature_* columns: {', '.join(invalid_sources)}")


def _parse_mem_available_gb(meminfo: str) -> float | None:
    for line in meminfo.splitlines():
        if not line.startswith("MemAvailable:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        return float(parts[1]) / (1024.0 * 1024.0)
    return None


def _mem_available_gb() -> float | None:
    try:
        return _parse_mem_available_gb(Path("/proc/meminfo").read_text(encoding="utf-8"))
    except OSError:
        return None


def _assert_min_mem_available(min_gb: float, stage: str) -> None:
    if min_gb <= 0.0:
        return
    available = _mem_available_gb()
    if available is None:
        return
    if available < min_gb:
        raise RuntimeError(
            f"available RAM below safety threshold at {stage}: "
            f"{available:.2f} GiB < {min_gb:.2f} GiB"
        )


def _set_reproducibility(seed: int, torch_threads: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch_threads > 0:
        torch.set_num_threads(torch_threads)


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(raw)


def _parse_aux_targets(raw: str) -> tuple[str, ...]:
    targets = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    invalid = [
        name
        for name in targets
        if (not name.startswith("responder_") or name == TARGET_COLUMN)
        and name not in DERIVED_TARGET_SPECS
    ]
    if invalid:
        raise ValueError(f"invalid auxiliary target(s): {', '.join(invalid)}")
    return targets


def _parse_feature_list(raw: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))


def _parse_batch_feature_modes(raw: str) -> tuple[str, ...]:
    aliases = {
        "rank": "batch_rank",
        "mean": "batch_mean",
        "demean": "batch_demean",
        "std": "batch_std",
        "zscore": "batch_zscore",
        "missing_count": "row_missing_count",
        "missing_frac": "row_missing_frac",
    }
    modes = tuple(dict.fromkeys(aliases.get(part.strip(), part.strip()) for part in raw.split(",") if part.strip()))
    invalid = [mode for mode in modes if mode not in BATCH_FEATURE_MODES]
    if invalid:
        raise ValueError(f"invalid batch feature mode(s): {', '.join(invalid)}")
    return modes


def _select_model_columns(
    schema: pl.Schema,
    *,
    n_features: int,
    use_official_lags: bool,
    include_time_features: bool,
    include_weight_feature: bool,
    batch_feature_sources: Sequence[str] = (),
    batch_feature_modes: Sequence[str] = (),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    feature_columns = feature_columns_from_schema(schema)[:n_features]
    if len(feature_columns) < n_features:
        raise ValueError(f"requested {n_features} features, only {len(feature_columns)} available")
    categorical = ["symbol_id", "time_id"]
    categorical.extend(name for name in DEFAULT_CATEGORICAL_FEATURES if name in feature_columns)
    continuous = [name for name in feature_columns if name not in categorical]
    if include_weight_feature:
        continuous.append("weight_feature")
    if include_time_features:
        continuous.extend(("time_sin_967", "time_cos_967", "time_sin_483", "time_cos_483", "date_sin_20", "date_cos_20"))
    if batch_feature_modes:
        sources = tuple(batch_feature_sources) if batch_feature_sources else tuple(name for name in continuous if name.startswith("feature_"))
        _validate_batch_feature_sources(sources, schema)
        continuous.extend(_batch_feature_names(sources, batch_feature_modes))
    if use_official_lags:
        continuous.extend(responder_lag_columns(RESPONDER_COLUMNS))
    return tuple(dict.fromkeys(continuous)), tuple(dict.fromkeys(categorical))


def _resolve_batch_feature_sources(
    continuous_columns: Sequence[str],
    requested_sources: Sequence[str],
    batch_feature_modes: Sequence[str],
) -> tuple[str, ...]:
    if not batch_feature_modes:
        return ()
    if requested_sources:
        return tuple(dict.fromkeys(requested_sources))
    return tuple(name for name in continuous_columns if name.startswith("feature_") and "__batch_" not in name)


def _validate_batch_feature_sources(sources: Sequence[str], schema: pl.Schema) -> None:
    if not sources:
        raise ValueError("batch feature modes require at least one source feature")
    invalid = [name for name in sources if not name.startswith("feature_") or name not in schema]
    if invalid:
        raise ValueError(f"invalid batch feature source(s): {', '.join(invalid)}")


def _batch_feature_names(sources: Sequence[str], modes: Sequence[str]) -> tuple[str, ...]:
    names: list[str] = []
    for source in sources:
        for mode in modes:
            suffix = BATCH_FEATURE_SUFFIXES.get(mode)
            if suffix is not None:
                names.append(f"{source}{suffix}")
    if "row_missing_count" in modes:
        names.append("row_missing_count")
    if "row_missing_frac" in modes:
        names.append("row_missing_frac")
    if "batch_missing_frac" in modes:
        names.append("batch_missing_frac")
    return tuple(names)


def _make_folds(train: pl.LazyFrame, args: argparse.Namespace) -> list[DateFold]:
    bounds = train.select(
        pl.min("date_id").alias("min_date_id"),
        pl.max("date_id").alias("max_date_id"),
    ).collect()
    min_date = int(bounds["min_date_id"][0])
    max_date = int(bounds["max_date_id"][0])
    if args.fold_protocol == "rolling":
        return make_rolling_folds(
            min_date_id=min_date,
            max_date_id=max_date,
            n_folds=args.n_folds,
            train_window=args.train_window,
            valid_window=args.valid_window,
            gap=args.gap,
        )
    valid_start = max_date - args.competition_valid_window + 1
    train_start = max(min_date, args.competition_train_start)
    recent_train_end = valid_start - 1
    gap_train_end = valid_start - args.competition_gap - 1
    folds = [
        DateFold("competition_recent", train_start, recent_train_end, valid_start, max_date),
    ]
    if gap_train_end >= train_start:
        folds.append(DateFold("competition_gap", train_start, gap_train_end, valid_start, max_date))
    return folds


def _select_requested_folds(folds: list[DateFold], args: argparse.Namespace) -> list[DateFold]:
    selected = folds[args.fold_start_index :]
    if args.fold_limit > 0:
        selected = selected[: args.fold_limit]
    if not selected:
        raise ValueError("fold selection is empty")
    return selected


def _collect_model_frame(
    data: pl.LazyFrame,
    start: int,
    end: int,
    continuous_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
    *,
    sample_max_rows: int = 0,
    seed: int = 17,
    sample_policy: str = "uniform",
    weight_tail_quantile: float = 0.80,
    batch_feature_sources: Sequence[str] = (),
) -> pl.DataFrame:
    if end < start:
        raise ValueError("end must be >= start")
    sample_plan = _make_sample_plan(
        data,
        start,
        end,
        sample_max_rows=sample_max_rows,
        sample_policy=sample_policy,
        weight_tail_quantile=weight_tail_quantile,
    )
    lazy = _model_lazy_frame(
        data,
        start,
        end,
        continuous_columns,
        categorical_columns,
        target_columns,
        sample_plan=sample_plan,
        seed=seed,
        batch_feature_sources=batch_feature_sources,
    )
    if sample_max_rows > 0:
        lazy = lazy.head(int(math.ceil(sample_max_rows * 1.10)))
    return lazy.collect()


def _model_lazy_frame(
    data: pl.LazyFrame,
    start: int,
    end: int,
    continuous_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
    *,
    sample_plan: SamplePlan | None = None,
    seed: int = 17,
    batch_feature_sources: Sequence[str] = (),
) -> pl.LazyFrame:
    lag_columns = tuple(name for name in continuous_columns if name.endswith("_lag_1"))
    derived_targets = tuple(name for name in target_columns if name in DERIVED_TARGET_SPECS)
    lookback_days = max((_derived_target_lookback(name) for name in derived_targets), default=1)
    history_start = start - lookback_days + 1 if derived_targets else start
    base = data.filter(pl.col("date_id").is_between(history_start, end))
    if derived_targets:
        base = _with_derived_aux_targets(base, derived_targets).filter(pl.col("date_id").is_between(start, end))

    batch_feature_sources = tuple(batch_feature_sources) if batch_feature_sources else _infer_batch_feature_sources(continuous_columns)
    batch_feature_names = tuple(name for name in continuous_columns if _is_batch_feature_name(name))
    batch_stat_frame = _batch_stat_lazy_frame(base, batch_feature_names, batch_feature_sources)
    batch_rank_frame = _batch_rank_lazy_frame(base, batch_feature_names)

    plan = sample_plan or SamplePlan(policy="uniform", row_count=0, sample_max_rows=0)
    sample_filter = _sample_filter_expr(plan, seed=seed)
    if sample_filter is not None:
        base = base.filter(sample_filter)

    if lag_columns:
        lag_sources = tuple(name.removesuffix("_lag_1") for name in lag_columns)
        lags = (
            data.filter(pl.col("date_id").is_between(start - 1, end - 1))
            .select(
                [
                    (pl.col("date_id") + 1).cast(pl.Int16).alias("date_id"),
                    pl.col("time_id"),
                    pl.col("symbol_id"),
                ]
                + [pl.col(source).alias(target) for source, target in zip(lag_sources, lag_columns, strict=True)]
            )
        )
        base = base.join(lags, on=["date_id", "time_id", "symbol_id"], how="left")

    if batch_stat_frame is not None:
        base = base.join(batch_stat_frame, on=["date_id", "time_id"], how="left")
    if batch_rank_frame is not None:
        base = base.join(batch_rank_frame, on=["date_id", "time_id", "symbol_id"], how="left")
    if batch_feature_names:
        base = base.with_columns([_sampled_batch_feature_expr(name, batch_feature_sources) for name in batch_feature_names])

    select_exprs: list[pl.Expr] = [
        pl.col("date_id").cast(pl.Int32),
        pl.col("time_id").cast(pl.Int32),
        pl.col("symbol_id").cast(pl.Int32),
        pl.col("weight").cast(pl.Float32),
    ]
    metadata = {"date_id", "time_id", "symbol_id", "weight"}
    for name in continuous_columns:
        if name == "weight_feature":
            select_exprs.append(pl.col("weight").cast(pl.Float32).alias(name))
        elif name == "time_sin_967":
            select_exprs.append((pl.col("time_id") * (2.0 * math.pi / 967.0)).sin().cast(pl.Float32).alias(name))
        elif name == "time_cos_967":
            select_exprs.append((pl.col("time_id") * (2.0 * math.pi / 967.0)).cos().cast(pl.Float32).alias(name))
        elif name == "time_sin_483":
            select_exprs.append((pl.col("time_id") * (2.0 * math.pi / 483.0)).sin().cast(pl.Float32).alias(name))
        elif name == "time_cos_483":
            select_exprs.append((pl.col("time_id") * (2.0 * math.pi / 483.0)).cos().cast(pl.Float32).alias(name))
        elif name == "date_sin_20":
            select_exprs.append((pl.col("date_id") * (2.0 * math.pi / 20.0)).sin().cast(pl.Float32).alias(name))
        elif name == "date_cos_20":
            select_exprs.append((pl.col("date_id") * (2.0 * math.pi / 20.0)).cos().cast(pl.Float32).alias(name))
        elif _is_batch_feature_name(name):
            select_exprs.append(pl.col(name).fill_null(0.0).cast(pl.Float32).alias(name))
        elif name not in metadata:
            select_exprs.append(pl.col(name).fill_null(0.0).cast(pl.Float32).alias(name))
    for name in categorical_columns:
        if name not in metadata:
            select_exprs.append(pl.col(name).fill_null(-1).cast(pl.Int32).alias(name))
    for name in target_columns:
        if name in DERIVED_TARGET_SPECS:
            select_exprs.append(pl.col(name).fill_null(0.0).cast(pl.Float32).alias(name))
        else:
            select_exprs.append(pl.col(name).cast(pl.Float32).alias(name))
    return base.select(select_exprs).sort(["date_id", "time_id", "symbol_id"])


def _is_batch_feature_name(name: str) -> bool:
    return (
        any(name.endswith(suffix) for suffix in BATCH_FEATURE_SUFFIXES.values())
        or name in {"row_missing_count", "row_missing_frac", "batch_missing_frac"}
    )


def _infer_batch_feature_sources(continuous_columns: Sequence[str]) -> tuple[str, ...]:
    sources: list[str] = []
    for name in continuous_columns:
        if name.startswith("feature_") and "__batch_" not in name:
            sources.append(name)
            continue
        parsed = _parse_batch_feature_name(name)
        if parsed is not None:
            source, _mode = parsed
            sources.append(source)
    return tuple(dict.fromkeys(sources))


def _parse_batch_feature_name(name: str) -> tuple[str, str] | None:
    for mode, suffix in BATCH_FEATURE_SUFFIXES.items():
        if name.endswith(suffix):
            source = name[: -len(suffix)]
            if source:
                return source, mode
    return None


def _batch_feature_expr(name: str, sources: Sequence[str]) -> pl.Expr:
    parsed = _parse_batch_feature_name(name)
    if parsed is not None:
        source, mode = parsed
        value = pl.col(source).fill_null(0.0).cast(pl.Float32)
        mean = value.mean().over(["date_id", "time_id"])
        if mode == "batch_rank":
            count = pl.len().over(["date_id", "time_id"]).cast(pl.Float32)
            denom = pl.when(count > 1.0).then(count - 1.0).otherwise(pl.lit(1.0))
            rank = value.rank(method="average").over(["date_id", "time_id"]).cast(pl.Float32)
            return (((rank - 1.0) / denom) - 0.5).fill_null(0.0).cast(pl.Float32).alias(name)
        if mode == "batch_mean":
            return mean.fill_null(0.0).cast(pl.Float32).alias(name)
        if mode == "batch_demean":
            return (value - mean).fill_null(0.0).cast(pl.Float32).alias(name)
        if mode == "batch_std":
            return value.std().over(["date_id", "time_id"]).fill_null(0.0).cast(pl.Float32).alias(name)
        if mode == "batch_zscore":
            std = value.std().over(["date_id", "time_id"]).fill_null(0.0)
            denom = pl.when(std.abs() > 1e-6).then(std).otherwise(pl.lit(1.0))
            return ((value - mean) / denom).fill_null(0.0).cast(pl.Float32).alias(name)
        raise ValueError(f"unsupported batch feature mode: {mode}")
    if not sources:
        raise ValueError(f"{name} requires at least one batch feature source")
    row_missing_count = sum((pl.col(source).is_null()).cast(pl.Float32) for source in sources)
    if name == "row_missing_count":
        return row_missing_count.cast(pl.Float32).alias(name)
    row_missing_frac = row_missing_count / float(len(sources))
    if name == "row_missing_frac":
        return row_missing_frac.cast(pl.Float32).alias(name)
    if name == "batch_missing_frac":
        return row_missing_frac.mean().over(["date_id", "time_id"]).fill_null(0.0).cast(pl.Float32).alias(name)
    raise ValueError(f"unknown batch feature name: {name}")


def _batch_stat_name(source: str, stat: str) -> str:
    return f"__batch_stat__{source}__{stat}"


def _batch_missing_frac_stat_name() -> str:
    return "__batch_stat__row_missing_frac__mean"


def _batch_stat_lazy_frame(
    base: pl.LazyFrame,
    batch_feature_names: Sequence[str],
    sources: Sequence[str],
) -> pl.LazyFrame | None:
    mean_sources: set[str] = set()
    std_sources: set[str] = set()
    needs_batch_missing = "batch_missing_frac" in batch_feature_names
    for name in batch_feature_names:
        parsed = _parse_batch_feature_name(name)
        if parsed is None:
            continue
        source, mode = parsed
        if mode in {"batch_mean", "batch_demean", "batch_zscore"}:
            mean_sources.add(source)
        if mode in {"batch_std", "batch_zscore"}:
            std_sources.add(source)

    agg_exprs: list[pl.Expr] = []
    for source in sorted(mean_sources):
        agg_exprs.append(pl.col(source).fill_null(0.0).cast(pl.Float32).mean().alias(_batch_stat_name(source, "mean")))
    for source in sorted(std_sources):
        agg_exprs.append(pl.col(source).fill_null(0.0).cast(pl.Float32).std().alias(_batch_stat_name(source, "std")))
    if needs_batch_missing:
        if not sources:
            raise ValueError("batch_missing_frac requires at least one batch feature source")
        row_missing_count = sum((pl.col(source).is_null()).cast(pl.Float32) for source in sources)
        agg_exprs.append((row_missing_count / float(len(sources))).mean().alias(_batch_missing_frac_stat_name()))

    if not agg_exprs:
        return None
    return base.group_by(["date_id", "time_id"]).agg(agg_exprs)


def _batch_rank_lazy_frame(base: pl.LazyFrame, batch_feature_names: Sequence[str]) -> pl.LazyFrame | None:
    rank_sources: list[str] = []
    for name in batch_feature_names:
        parsed = _parse_batch_feature_name(name)
        if parsed is None:
            continue
        source, mode = parsed
        if mode == "batch_rank":
            rank_sources.append(source)
    rank_sources = list(dict.fromkeys(rank_sources))
    if not rank_sources:
        return None

    count = pl.len().over(["date_id", "time_id"]).cast(pl.Float32)
    denom = pl.when(count > 1.0).then(count - 1.0).otherwise(pl.lit(1.0))
    exprs: list[pl.Expr] = [
        pl.col("date_id"),
        pl.col("time_id"),
        pl.col("symbol_id"),
    ]
    for source in rank_sources:
        value = pl.col(source).fill_null(0.0).cast(pl.Float32)
        rank = value.rank(method="average").over(["date_id", "time_id"]).cast(pl.Float32)
        exprs.append((((rank - 1.0) / denom) - 0.5).fill_null(0.0).cast(pl.Float32).alias(f"{source}__batch_rank"))
    return base.select(exprs)


def _sampled_batch_feature_expr(name: str, sources: Sequence[str]) -> pl.Expr:
    parsed = _parse_batch_feature_name(name)
    if parsed is not None:
        source, mode = parsed
        value = pl.col(source).fill_null(0.0).cast(pl.Float32)
        if mode == "batch_rank":
            return pl.col(name).fill_null(0.0).cast(pl.Float32).alias(name)
        if mode == "batch_mean":
            return pl.col(_batch_stat_name(source, "mean")).fill_null(0.0).cast(pl.Float32).alias(name)
        if mode == "batch_std":
            return pl.col(_batch_stat_name(source, "std")).fill_null(0.0).cast(pl.Float32).alias(name)
        if mode == "batch_demean":
            mean = pl.col(_batch_stat_name(source, "mean")).fill_null(0.0).cast(pl.Float32)
            return (value - mean).fill_null(0.0).cast(pl.Float32).alias(name)
        if mode == "batch_zscore":
            mean = pl.col(_batch_stat_name(source, "mean")).fill_null(0.0).cast(pl.Float32)
            std = pl.col(_batch_stat_name(source, "std")).fill_null(0.0).cast(pl.Float32)
            denom = pl.when(std.abs() > 1e-6).then(std).otherwise(pl.lit(1.0))
            return ((value - mean) / denom).fill_null(0.0).cast(pl.Float32).alias(name)
        raise ValueError(f"unsupported batch feature mode: {mode}")

    if not sources:
        raise ValueError(f"{name} requires at least one batch feature source")
    row_missing_count = sum((pl.col(source).is_null()).cast(pl.Float32) for source in sources)
    if name == "row_missing_count":
        return row_missing_count.cast(pl.Float32).alias(name)
    row_missing_frac = row_missing_count / float(len(sources))
    if name == "row_missing_frac":
        return row_missing_frac.cast(pl.Float32).alias(name)
    if name == "batch_missing_frac":
        return pl.col(_batch_missing_frac_stat_name()).fill_null(0.0).cast(pl.Float32).alias(name)
    raise ValueError(f"unknown batch feature name: {name}")


def _with_derived_aux_targets(data: pl.LazyFrame, target_columns: Sequence[str]) -> pl.LazyFrame:
    return data.with_columns([_derived_target_expr(target) for target in target_columns])


def _derived_target_lookback(name: str) -> int:
    spec = DERIVED_TARGET_SPECS[name]
    if spec["kind"] == "rolling_mean":
        return int(spec["window"])
    return 1


def _derived_target_expr(name: str) -> pl.Expr:
    spec = DERIVED_TARGET_SPECS[name]
    source = str(spec["source"])
    base = pl.col(source).cast(pl.Float32)
    kind = spec["kind"]
    if kind == "rolling_mean":
        return (
            base.rolling_mean(window_size=int(spec["window"]), min_samples=1)
            .over(["time_id", "symbol_id"], order_by="date_id")
            .cast(pl.Float32)
            .alias(name)
        )
    if kind == "abs":
        return base.abs().cast(pl.Float32).alias(name)
    if kind == "clip":
        limit = float(spec["limit"])
        return (
            pl.when(base > limit)
            .then(pl.lit(limit))
            .when(base < -limit)
            .then(pl.lit(-limit))
            .otherwise(base)
            .cast(pl.Float32)
            .alias(name)
        )
    if kind == "sign":
        return (
            pl.when(base > 0.0)
            .then(pl.lit(1.0))
            .when(base < 0.0)
            .then(pl.lit(-1.0))
            .otherwise(pl.lit(0.0))
            .cast(pl.Float32)
            .alias(name)
        )
    raise ValueError(f"unknown derived target kind for {name}: {kind}")


def _make_sample_plan(
    data: pl.LazyFrame,
    start: int,
    end: int,
    *,
    sample_max_rows: int,
    sample_policy: str,
    weight_tail_quantile: float,
) -> SamplePlan:
    row_count = _count_rows(data, start, end)
    if sample_max_rows <= 0 or row_count <= sample_max_rows:
        return SamplePlan(policy=sample_policy, row_count=row_count, sample_max_rows=sample_max_rows)
    if sample_policy == "uniform":
        return SamplePlan(
            policy="uniform",
            row_count=row_count,
            sample_max_rows=sample_max_rows,
            uniform_bps=_sample_basis_points(row_count, sample_max_rows),
        )
    if sample_policy != "weight_tail":
        raise ValueError(f"unknown train sample policy: {sample_policy}")
    threshold = _weight_quantile(data, start, end, weight_tail_quantile)
    high_count = _count_weight_tail_rows(data, start, end, threshold)
    if high_count <= 0:
        return SamplePlan(
            policy="uniform",
            row_count=row_count,
            sample_max_rows=sample_max_rows,
            uniform_bps=_sample_basis_points(row_count, sample_max_rows),
        )
    if high_count >= sample_max_rows:
        high_bps = _sample_basis_points(high_count, sample_max_rows)
        rest_bps = 0
    else:
        high_bps = 10_000
        rest_count = max(row_count - high_count, 0)
        rest_budget = max(sample_max_rows - high_count, 0)
        rest_bps = _sample_basis_points(rest_count, rest_budget) if rest_count > 0 and rest_budget > 0 else 0
    return SamplePlan(
        policy="weight_tail",
        row_count=row_count,
        sample_max_rows=sample_max_rows,
        high_weight_threshold=threshold,
        high_weight_bps=high_bps,
        rest_bps=rest_bps,
    )


def _sample_filter_expr(plan: SamplePlan, *, seed: int) -> pl.Expr | None:
    if plan.sample_max_rows <= 0 or plan.row_count <= plan.sample_max_rows:
        return None
    hashed = pl.struct(["date_id", "time_id", "symbol_id"]).hash(seed=seed) % 10_000
    if plan.policy == "uniform":
        return hashed < plan.uniform_bps
    if plan.policy == "weight_tail":
        if plan.high_weight_threshold is None:
            return hashed < plan.uniform_bps
        high_weight = pl.col("weight") >= plan.high_weight_threshold
        keep_high = high_weight & (hashed < plan.high_weight_bps)
        keep_rest = (~high_weight) & (hashed < plan.rest_bps)
        return keep_high | keep_rest
    raise ValueError(f"unknown train sample policy: {plan.policy}")


def _count_rows(data: pl.LazyFrame, start: int, end: int) -> int:
    return int(data.filter(pl.col("date_id").is_between(start, end)).select(pl.len()).collect().item())


def _weight_quantile(data: pl.LazyFrame, start: int, end: int, quantile: float) -> float:
    value = (
        data.filter(pl.col("date_id").is_between(start, end))
        .select(pl.col("weight").quantile(float(quantile)).alias("weight_quantile"))
        .collect()
        .item()
    )
    if value is None or not math.isfinite(float(value)):
        raise ValueError("could not compute finite weight quantile for sample policy")
    return float(value)


def _count_weight_tail_rows(data: pl.LazyFrame, start: int, end: int, threshold: float) -> int:
    return int(
        data.filter(pl.col("date_id").is_between(start, end) & (pl.col("weight") >= float(threshold)))
        .select(pl.len())
        .collect()
        .item()
    )


def _sample_basis_points(row_count: int, sample_max_rows: int) -> int:
    if sample_max_rows <= 0 or row_count <= sample_max_rows:
        return 10_000
    return max(1, min(10_000, int(math.ceil(sample_max_rows / row_count * 10_000.0))))


def _fit_standardization(
    frame: pl.DataFrame,
    continuous_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
    *,
    center_target: bool,
) -> Standardization:
    continuous = frame.select(list(continuous_columns)).to_numpy().astype(np.float32, copy=True)
    mean = continuous.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = continuous.std(axis=0, dtype=np.float64).astype(np.float32)
    scale[~np.isfinite(scale)] = 1.0
    scale[scale <= 1e-6] = 1.0
    targets = frame.select(list(target_columns)).to_numpy().astype(np.float32, copy=False)
    weights = frame["weight"].to_numpy().astype(np.float64, copy=False)
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        raise ValueError("training weights must have positive sum")
    targets64 = targets.astype(np.float64)
    if center_target:
        target_mean = ((targets64 * weights[:, None]).sum(axis=0) / weight_sum).astype(np.float32)
        target_energy = ((targets64 - target_mean.astype(np.float64)) ** 2 * weights[:, None]).sum(axis=0) / weight_sum
    else:
        target_mean = np.zeros(targets.shape[1], dtype=np.float32)
        target_energy = (targets64**2 * weights[:, None]).sum(axis=0) / weight_sum
    target_scale = np.sqrt(np.maximum(target_energy, 1e-6)).astype(np.float32)
    specs = []
    for name in categorical_columns:
        values = frame[name].drop_nulls()
        specs.append(CategoricalSpec(name=name, min_value=int(values.min()), max_value=int(values.max())))
    return Standardization(mean, scale, target_mean, target_scale, tuple(specs))


def _frame_to_tensors(
    frame: pl.DataFrame,
    standardization: Standardization,
    continuous_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    continuous = frame.select(list(continuous_columns)).to_numpy().astype(np.float32, copy=False)
    continuous = (continuous - standardization.continuous_mean) / standardization.continuous_scale
    categorical = _categorical_matrix(frame, categorical_columns, standardization.categorical_specs)
    target = frame.select(list(target_columns)).to_numpy().astype(np.float32, copy=True)
    target = (target - standardization.target_mean) / standardization.target_scale
    weight = frame["weight"].to_numpy().astype(np.float32, copy=True)
    return (
        torch.from_numpy(continuous.astype(np.float32, copy=False)),
        torch.from_numpy(categorical),
        torch.from_numpy(target.astype(np.float32, copy=False)),
        torch.from_numpy(weight),
    )


def _frame_to_batch_ids(frame: pl.DataFrame) -> torch.Tensor:
    if frame.height == 0:
        return torch.empty(0, dtype=torch.long)
    date_id = frame["date_id"].to_numpy().astype(np.int64, copy=False)
    time_id = frame["time_id"].to_numpy().astype(np.int64, copy=False)
    is_new = np.ones(frame.height, dtype=bool)
    if frame.height > 1:
        is_new[1:] = (date_id[1:] != date_id[:-1]) | (time_id[1:] != time_id[:-1])
    batch_ids = np.cumsum(is_new, dtype=np.int64) - 1
    return torch.from_numpy(batch_ids)


def _model_requires_batch_ids(model: nn.Module) -> bool:
    return bool(getattr(model, "requires_batch_ids", False))


def _categorical_matrix(
    frame: pl.DataFrame,
    categorical_columns: Sequence[str],
    specs: Sequence[CategoricalSpec],
) -> np.ndarray:
    if not categorical_columns:
        return np.zeros((frame.height, 0), dtype=np.int64)
    columns = []
    for name, spec in zip(categorical_columns, specs, strict=True):
        values = frame[name].to_numpy().astype(np.int64, copy=False)
        encoded = values - spec.min_value
        invalid = (values < spec.min_value) | (values > spec.max_value)
        encoded[invalid] = spec.unknown_index
        columns.append(encoded)
    return np.stack(columns, axis=1).astype(np.int64, copy=False)


def _fit_model(
    model: nn.Module,
    train_frame: pl.DataFrame,
    early_frame: pl.DataFrame | None,
    standardization: Standardization,
    continuous_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | int]:
    train_tensors = _frame_to_tensors(train_frame, standardization, continuous_columns, categorical_columns, target_columns)
    if _model_requires_batch_ids(model):
        train_tensors = (*train_tensors, _frame_to_batch_ids(train_frame))
    early_tensors = None if early_frame is None or early_frame.height == 0 else _frame_to_tensors(early_frame, standardization, continuous_columns, categorical_columns, target_columns)
    if early_tensors is not None and _model_requires_batch_ids(model):
        early_tensors = (*early_tensors, _frame_to_batch_ids(early_frame))
    model.to(device)
    optimizer = _make_optimizer(model, args.learning_rate, args)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[float] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = _train_epochs(model, train_tensors, optimizer, args, device, epochs=1)
        history.append(train_loss)
        valid_loss = train_loss if early_tensors is None else _evaluate_scaled_loss(model, early_tensors, args, device)
        if valid_loss < best_loss - args.early_stopping_min_delta:
            best_loss = valid_loss
            best_epoch = epoch
            stale = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if args.early_stopping_patience > 0 and stale >= args.early_stopping_patience:
                break
    model.load_state_dict(best_state)
    return {
        "trained_epochs": len(history),
        "best_epoch": best_epoch,
        "last_train_loss": float(history[-1]),
        "best_internal_valid_loss": float(best_loss),
    }


def _make_optimizer(model: nn.Module, lr: float, args: argparse.Namespace) -> SophiaG:
    return SophiaG(
        model.parameters(),
        lr=lr,
        betas=(args.sophia_beta1, args.sophia_beta2),
        rho=args.sophia_rho,
        weight_decay=args.weight_decay,
        clip=args.sophia_clip,
        update_period=args.sophia_update_period,
    )


def _train_epochs(
    model: nn.Module,
    tensors: tuple[torch.Tensor, ...],
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    *,
    epochs: int,
) -> float:
    if _model_requires_batch_ids(model):
        return _train_set_epochs(model, tensors, optimizer, args, device, epochs=epochs)
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, pin_memory=device.type == "cuda")
    last_loss = 0.0
    for _ in range(epochs):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for continuous, categorical, target, weight in loader:
            continuous = continuous.to(device, non_blocking=True)
            categorical = categorical.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(continuous, categorical)
            loss = _weighted_multi_loss(
                pred,
                target,
                weight,
                aux_weight=args.aux_loss_weight,
                loss_type=args.loss_type,
                huber_delta=args.huber_delta,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_weight = float(weight.sum().detach().cpu())
            loss_sum += float(loss.detach().cpu()) * batch_weight
            weight_sum += batch_weight
        last_loss = loss_sum / max(weight_sum, 1.0)
    return last_loss


def _train_set_epochs(
    model: nn.Module,
    tensors: tuple[torch.Tensor, ...],
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    device: torch.device,
    *,
    epochs: int,
) -> float:
    last_loss = 0.0
    for _ in range(epochs):
        model.train()
        loss_sum = 0.0
        weight_sum = 0.0
        for continuous, categorical, target, weight, batch_id in _iter_set_batches(tensors, args.batch_size, shuffle_groups=True):
            continuous = continuous.to(device, non_blocking=True)
            categorical = categorical.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            batch_id = batch_id.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(continuous, categorical, batch_id)
            loss = _weighted_multi_loss(
                pred,
                target,
                weight,
                aux_weight=args.aux_loss_weight,
                loss_type=args.loss_type,
                huber_delta=args.huber_delta,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_weight = float(weight.sum().detach().cpu())
            loss_sum += float(loss.detach().cpu()) * batch_weight
            weight_sum += batch_weight
        last_loss = loss_sum / max(weight_sum, 1.0)
    return last_loss


def _iter_set_batches(
    tensors: tuple[torch.Tensor, ...],
    batch_size: int,
    *,
    shuffle_groups: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if len(tensors) != 5:
        raise ValueError("set-aware batches require continuous, categorical, target, weight and batch_id tensors")
    continuous, categorical, target, weight, batch_id = tensors
    if batch_id.numel() == 0:
        return
    change = torch.ones(batch_id.numel(), dtype=torch.bool)
    if batch_id.numel() > 1:
        change[1:] = batch_id[1:] != batch_id[:-1]
    starts = torch.nonzero(change, as_tuple=False).flatten()
    ends = torch.cat([starts[1:], torch.tensor([batch_id.numel()], dtype=torch.long)])
    group_order = torch.randperm(starts.numel()) if shuffle_groups else torch.arange(starts.numel())
    selected: list[torch.Tensor] = []
    selected_rows = 0
    for group_idx in group_order.tolist():
        start = int(starts[group_idx])
        end = int(ends[group_idx])
        group_indices = torch.arange(start, end, dtype=torch.long)
        group_rows = end - start
        if selected and selected_rows + group_rows > batch_size:
            indices = torch.cat(selected)
            yield continuous[indices], categorical[indices], target[indices], weight[indices], batch_id[indices]
            selected = []
            selected_rows = 0
        selected.append(group_indices)
        selected_rows += group_rows
    if selected:
        indices = torch.cat(selected)
        yield continuous[indices], categorical[indices], target[indices], weight[indices], batch_id[indices]


def _weighted_multi_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
    *,
    aux_weight: float,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    if pred.ndim == 3:
        target_for_loss = target[:, None, :]
        weight_for_loss = weight[:, None, None]
        per_row_target = _pointwise_regression_loss(pred, target_for_loss, loss_type=loss_type, huber_delta=huber_delta)
        per_target = torch.sum(weight_for_loss * per_row_target, dim=(0, 1)) / torch.clamp(
            torch.sum(weight) * pred.shape[1], min=1e-12
        )
    else:
        per_row_target = _pointwise_regression_loss(pred, target, loss_type=loss_type, huber_delta=huber_delta)
        per_target = torch.sum(weight[:, None] * per_row_target, dim=0) / torch.clamp(torch.sum(weight), min=1e-12)
    if per_target.numel() == 1:
        return per_target[0]
    return per_target[0] + aux_weight * per_target[1:].mean()


def _pointwise_regression_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_type: str,
    huber_delta: float,
) -> torch.Tensor:
    error = pred - target
    if loss_type == "mse":
        return error.square()
    if loss_type == "huber":
        abs_error = error.abs()
        delta = float(huber_delta)
        return torch.where(abs_error <= delta, error.square(), 2.0 * delta * (abs_error - 0.5 * delta))
    raise ValueError(f"unknown loss type: {loss_type}")


def _evaluate_scaled_loss(
    model: nn.Module,
    tensors: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    if _model_requires_batch_ids(model):
        return _evaluate_set_scaled_loss(model, tensors, args, device)
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=device.type == "cuda")
    model.eval()
    loss_sum = 0.0
    weight_sum = 0.0
    with torch.no_grad():
        for continuous, categorical, target, weight in loader:
            continuous = continuous.to(device, non_blocking=True)
            categorical = categorical.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            loss = _weighted_multi_loss(
                model(continuous, categorical),
                target,
                weight,
                aux_weight=args.aux_loss_weight,
                loss_type=args.loss_type,
                huber_delta=args.huber_delta,
            )
            batch_weight = float(weight.sum().detach().cpu())
            loss_sum += float(loss.detach().cpu()) * batch_weight
            weight_sum += batch_weight
    return loss_sum / max(weight_sum, 1.0)


def _evaluate_set_scaled_loss(
    model: nn.Module,
    tensors: tuple[torch.Tensor, ...],
    args: argparse.Namespace,
    device: torch.device,
) -> float:
    model.eval()
    loss_sum = 0.0
    weight_sum = 0.0
    with torch.no_grad():
        for continuous, categorical, target, weight, batch_id in _iter_set_batches(tensors, args.batch_size, shuffle_groups=False):
            continuous = continuous.to(device, non_blocking=True)
            categorical = categorical.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            weight = weight.to(device, non_blocking=True)
            batch_id = batch_id.to(device, non_blocking=True)
            loss = _weighted_multi_loss(
                model(continuous, categorical, batch_id),
                target,
                weight,
                aux_weight=args.aux_loss_weight,
                loss_type=args.loss_type,
                huber_delta=args.huber_delta,
            )
            batch_weight = float(weight.sum().detach().cpu())
            loss_sum += float(loss.detach().cpu()) * batch_weight
            weight_sum += batch_weight
    return loss_sum / max(weight_sum, 1.0)


def _evaluate_offline(
    model: nn.Module,
    data: pl.LazyFrame,
    fold: DateFold,
    standardization: Standardization,
    continuous_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
    calibration: PredictionCalibration,
    args: argparse.Namespace,
    device: torch.device,
    prediction_output_dir: Path | None = None,
) -> dict[str, float | int]:
    numerator = 0.0
    denominator = 0.0
    rows = 0
    weight_sum = 0.0
    pred_sum = 0.0
    pred_sumsq = 0.0
    prediction_frames: list[pl.DataFrame] = []
    for chunk_start in range(fold.valid_start, fold.valid_end + 1, args.eval_chunk_days):
        _assert_min_mem_available(args.min_mem_available_gb, f"before offline evaluation chunk {fold.name}:{chunk_start}")
        chunk_end = min(fold.valid_end, chunk_start + args.eval_chunk_days - 1)
        frame = _collect_model_frame(
            data,
            chunk_start,
            chunk_end,
            continuous_columns,
            categorical_columns,
            target_columns,
            sample_max_rows=args.max_valid_rows,
            seed=args.seed + 101 + chunk_start,
            batch_feature_sources=getattr(args, "resolved_batch_feature_sources", ()),
        )
        metrics, predictions = _evaluate_frame_with_predictions(
            model,
            frame,
            standardization,
            continuous_columns,
            categorical_columns,
            target_columns,
            calibration,
            args,
            device,
            save_predictions=prediction_output_dir is not None,
        )
        numerator += metrics["numerator"]
        denominator += metrics["denominator"]
        rows += int(metrics["valid_rows"])
        weight_sum += metrics["weight_sum"]
        pred_sum += metrics["prediction_sum"]
        pred_sumsq += metrics["prediction_sumsq"]
        if predictions is not None:
            prediction_frames.append(predictions)
        del frame
        gc.collect()
    _write_prediction_frames(prediction_frames, prediction_output_dir, fold)
    return _metric_dict(numerator, denominator, rows, weight_sum, pred_sum=pred_sum, pred_sumsq=pred_sumsq)


def _evaluate_online(
    model: nn.Module,
    data: pl.LazyFrame,
    fold: DateFold,
    standardization: Standardization,
    continuous_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
    calibration: PredictionCalibration,
    args: argparse.Namespace,
    device: torch.device,
    prediction_output_dir: Path | None = None,
) -> dict[str, float | int]:
    numerator = 0.0
    denominator = 0.0
    rows = 0
    weight_sum = 0.0
    pred_sum = 0.0
    pred_sumsq = 0.0
    previous_frame: pl.DataFrame | None = None
    prediction_frames: list[pl.DataFrame] = []
    optimizer = _make_optimizer(model, args.online_learning_rate, args)
    for date_id in range(fold.valid_start, fold.valid_end + 1):
        _assert_min_mem_available(args.min_mem_available_gb, f"before online evaluation date {fold.name}:{date_id}")
        if previous_frame is not None and previous_frame.height > 0:
            update_frame = previous_frame
            if args.online_max_update_rows_per_date > 0 and update_frame.height > args.online_max_update_rows_per_date:
                update_frame = update_frame.sample(n=args.online_max_update_rows_per_date, seed=args.seed + date_id, shuffle=True)
            if _model_requires_batch_ids(model):
                update_frame = update_frame.sort(["date_id", "time_id", "symbol_id"])
            update_tensors = _frame_to_tensors(update_frame, standardization, continuous_columns, categorical_columns, target_columns)
            if _model_requires_batch_ids(model):
                update_tensors = (*update_tensors, _frame_to_batch_ids(update_frame))
            _train_epochs(model, update_tensors, optimizer, args, device, epochs=args.online_epochs)
            del update_tensors
        frame = _collect_model_frame(
            data,
            date_id,
            date_id,
            continuous_columns,
            categorical_columns,
            target_columns,
            sample_max_rows=args.max_valid_rows,
            seed=args.seed + 211 + date_id,
            batch_feature_sources=getattr(args, "resolved_batch_feature_sources", ()),
        )
        metrics, predictions = _evaluate_frame_with_predictions(
            model,
            frame,
            standardization,
            continuous_columns,
            categorical_columns,
            target_columns,
            calibration,
            args,
            device,
            save_predictions=prediction_output_dir is not None,
        )
        numerator += metrics["numerator"]
        denominator += metrics["denominator"]
        rows += int(metrics["valid_rows"])
        weight_sum += metrics["weight_sum"]
        pred_sum += metrics["prediction_sum"]
        pred_sumsq += metrics["prediction_sumsq"]
        if predictions is not None:
            prediction_frames.append(predictions)
        previous_frame = frame
        gc.collect()
    _write_prediction_frames(prediction_frames, prediction_output_dir, fold)
    return _metric_dict(numerator, denominator, rows, weight_sum, pred_sum=pred_sum, pred_sumsq=pred_sumsq)


def _evaluate_frame(
    model: nn.Module,
    frame: pl.DataFrame,
    standardization: Standardization,
    continuous_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
    calibration: PredictionCalibration,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float | int]:
    metrics, _predictions = _evaluate_frame_with_predictions(
        model,
        frame,
        standardization,
        continuous_columns,
        categorical_columns,
        target_columns,
        calibration,
        args,
        device,
        save_predictions=False,
    )
    return metrics


def _evaluate_frame_with_predictions(
    model: nn.Module,
    frame: pl.DataFrame,
    standardization: Standardization,
    continuous_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
    calibration: PredictionCalibration,
    args: argparse.Namespace,
    device: torch.device,
    *,
    save_predictions: bool,
) -> tuple[dict[str, float | int], pl.DataFrame | None]:
    prediction, y, weight = _predict_target_vectors(
        model,
        frame,
        standardization,
        continuous_columns,
        categorical_columns,
        target_columns,
        args,
        device,
    )
    prediction = prediction * calibration.scale
    err = y - prediction
    numerator = float(np.sum(weight * err * err))
    denominator = float(np.sum(weight * y * y))
    metrics: dict[str, float | int] = {
        "numerator": numerator,
        "denominator": denominator,
        "valid_rows": frame.height,
        "weight_sum": float(weight.sum()),
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction)),
        "prediction_sum": float(np.sum(prediction)),
        "prediction_sumsq": float(np.sum(prediction * prediction)),
    }
    predictions = None
    if save_predictions:
        predictions = frame.select(["date_id", "time_id", "symbol_id", "weight", TARGET_COLUMN]).with_columns(
            pl.Series("tabm_prediction", prediction.astype(np.float32, copy=False))
        )
    return metrics, predictions


def _write_prediction_frames(
    frames: Sequence[pl.DataFrame],
    output_dir: Path | None,
    fold: DateFold,
) -> None:
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError(f"{fold.name}: no prediction frames to write")
    (
        pl.concat(frames)
        .with_columns(pl.lit(fold.name).alias("fold"))
        .write_parquet(output_dir / f"{fold.name}.parquet", compression="zstd")
    )


def _fit_prediction_calibration(
    model: nn.Module,
    frame: pl.DataFrame,
    standardization: Standardization,
    continuous_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> PredictionCalibration:
    prediction, target, weight = _predict_target_vectors(
        model,
        frame,
        standardization,
        continuous_columns,
        categorical_columns,
        target_columns,
        args,
        device,
    )
    denom = float(np.sum(weight * prediction * prediction))
    if denom <= 1e-12:
        return PredictionCalibration()
    scale = float(np.sum(weight * target * prediction) / denom)
    if not math.isfinite(scale):
        return PredictionCalibration()
    scale = max(-args.max_calibration_scale, min(args.max_calibration_scale, scale))
    return PredictionCalibration(scale=scale)


def _predict_target_vectors(
    model: nn.Module,
    frame: pl.DataFrame,
    standardization: Standardization,
    continuous_columns: Sequence[str],
    categorical_columns: Sequence[str],
    target_columns: Sequence[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensors = _frame_to_tensors(
        frame,
        standardization,
        continuous_columns,
        categorical_columns,
        target_columns,
    )
    preds = []
    model.eval()
    with torch.no_grad():
        if _model_requires_batch_ids(model):
            set_tensors = (*tensors, _frame_to_batch_ids(frame))
            for continuous_batch, categorical_batch, _target, _weight, batch_id in _iter_set_batches(
                set_tensors, args.batch_size, shuffle_groups=False
            ):
                pred_scaled = model(
                    continuous_batch.to(device, non_blocking=True),
                    categorical_batch.to(device, non_blocking=True),
                    batch_id.to(device, non_blocking=True),
                )
                if pred_scaled.ndim == 3:
                    pred_scaled = pred_scaled.mean(dim=1)
                pred = pred_scaled[:, 0].detach().cpu().numpy().astype(np.float64, copy=False)
                preds.append(pred)
        else:
            continuous, categorical, _target_scaled, _weight_tensor = tensors
            dataset = TensorDataset(continuous, categorical)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, pin_memory=device.type == "cuda")
            for continuous_batch, categorical_batch in loader:
                pred_scaled = model(continuous_batch.to(device, non_blocking=True), categorical_batch.to(device, non_blocking=True))
                if pred_scaled.ndim == 3:
                    pred_scaled = pred_scaled.mean(dim=1)
                pred = pred_scaled[:, 0].detach().cpu().numpy().astype(np.float64, copy=False)
                preds.append(pred)
    prediction = np.concatenate(preds) * float(standardization.target_scale[0]) + float(standardization.target_mean[0])
    target = frame[TARGET_COLUMN].to_numpy().astype(np.float64, copy=False)
    weight = tensors[3].numpy().astype(np.float64, copy=False)
    return prediction, target, weight


def _metric_dict(
    numerator: float,
    denominator: float,
    rows: int,
    weight_sum: float,
    *,
    pred_sum: float = 0.0,
    pred_sumsq: float = 0.0,
) -> dict[str, float | int]:
    if denominator <= 0.0:
        raise ValueError("validation target energy must be positive")
    pred_mean = pred_sum / rows if rows > 0 else 0.0
    pred_var = max(pred_sumsq / rows - pred_mean * pred_mean, 0.0) if rows > 0 else 0.0
    return {
        "numerator": numerator,
        "denominator": denominator,
        "valid_rows": rows,
        "weight_sum": weight_sum,
        "weighted_zero_mean_r2": 1.0 - numerator / denominator,
        "prediction_mean": pred_mean,
        "prediction_std": math.sqrt(pred_var),
    }


def _summary(results: pl.DataFrame) -> pl.DataFrame:
    return (
        results.group_by(["model", "use_official_lags", "online_update"])
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
        "train_days": fold.train_days,
        "valid_days": fold.valid_days,
    }


def _embedding_dim(cardinality: int) -> int:
    return max(2, min(32, int(math.ceil(math.sqrt(cardinality) * 2.0))))


if __name__ == "__main__":
    main()
