"""Blend calibrated Ridge and conservative sklearn GBDT."""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
from catboost import CatBoostRegressor
from catboost import Pool
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor

from janestreet.blending import (
    GroupedBlendWeights,
    add_convex_blend_prediction,
    fit_convex_blend_weight,
    fit_grouped_convex_blend_weights,
)
from janestreet.calibration import (
    add_abs_prediction_bucket,
    fit_abs_prediction_thresholds,
    fit_shrinkage_calibrator,
)
from janestreet.diagnostics import aggregate_weighted_r2_by_slice, combine_slice_aggregates
from janestreet.folds import DateFold, make_rolling_folds
from janestreet.koopman_features import (
    KoopmanSpec,
    parse_koopman_columns,
    require_koopman_columns,
    with_koopman_features,
)
from janestreet.linear import build_weighted_ridge_fit_data, feature_columns_from_schema, solve_weighted_ridge
from janestreet.multiscale_features import (
    MultiscaleSpec,
    parse_multiscale_columns,
    parse_multiscale_spans,
    require_multiscale_columns,
    with_multiscale_features,
)
from janestreet.official_lags import DAILY_LAG_JOIN_KEYS, daily_last_responder_lag_columns, responder_lag_columns
from janestreet.paths import (
    DAILY_RESPONDER_LAGS_LAST_PARQUET,
    TRAIN_PARQUET_DIR,
    TRAIN_WITH_RESPONDER_LAGS_PARQUET,
)
from janestreet.reservoir_features import (
    make_reservoir_spec,
    parse_reservoir_columns,
    parse_reservoir_spans,
    require_reservoir_columns,
    with_reservoir_features,
)
from janestreet.symbol_graph import (
    SymbolGraphSpec,
    add_symbol_graph_features,
    fit_symbol_graph_spec,
    parse_symbol_graph_columns,
    require_symbol_graph_columns,
)
from janestreet.temporal_geometry import (
    TemporalGeometrySpec,
    parse_temporal_geometry_columns,
    parse_temporal_geometry_windows,
    require_temporal_geometry_columns,
    with_temporal_geometry_features,
)


warnings.filterwarnings("ignore", message="X does not have valid feature names")


@dataclass(frozen=True)
class PredictionModels:
    ridge_model: object
    gbdt_models: tuple[HistGradientBoostingRegressor | LGBMRegressor | XGBRegressor | CatBoostRegressor, ...]
    gbdt_target_mode: str = "target"
    cat_feature_indices: tuple[int, ...] = ()
    symbol_graph_spec: SymbolGraphSpec | None = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--valid-window", type=int, default=60)
    parser.add_argument("--inner-oof-folds", type=int, default=3)
    parser.add_argument("--inner-valid-window", type=int, default=20)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1000.0)
    parser.add_argument("--train-sample-frac", type=float, default=0.05)
    parser.add_argument("--chunk-days", type=int, default=10)
    parser.add_argument("--time-bucket-size", type=int, default=100)
    parser.add_argument("--use-responder-lags", action="store_true")
    parser.add_argument("--responder-lag-mode", choices=["same_time", "daily_last"], default="same_time")
    parser.add_argument("--responder-lag-target", choices=["both", "gbdt"], default="both")
    parser.add_argument("--temporal-geometry-columns", default="")
    parser.add_argument("--temporal-geometry-windows", default="5,20")
    parser.add_argument("--temporal-geometry-target", choices=["both", "gbdt"], default="both")
    parser.add_argument("--multiscale-columns", default="")
    parser.add_argument("--multiscale-spans", default="4,16,64")
    parser.add_argument("--multiscale-target", choices=["both", "gbdt"], default="gbdt")
    parser.add_argument("--koopman-columns", default="")
    parser.add_argument("--koopman-span", type=int, default=16)
    parser.add_argument("--koopman-target", choices=["both", "gbdt"], default="gbdt")
    parser.add_argument("--reservoir-columns", default="")
    parser.add_argument("--reservoir-states", type=int, default=8)
    parser.add_argument("--reservoir-spans", default="5,20")
    parser.add_argument("--reservoir-seed", type=int, default=17)
    parser.add_argument("--reservoir-target", choices=["both", "gbdt"], default="gbdt")
    parser.add_argument("--symbol-graph-columns", default="")
    parser.add_argument("--symbol-graph-neighbors", type=int, default=5)
    parser.add_argument("--gbdt-engine", choices=["sklearn", "lightgbm", "xgboost", "catboost"], default="sklearn")
    parser.add_argument("--gbdt-target-mode", choices=["target", "residual_raw_ridge"], default="target")
    parser.add_argument("--gbdt-id-columns", default="")
    parser.add_argument("--catboost-categorical-id-columns", action="store_true")
    parser.add_argument("--max-iter", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-leaf-nodes", type=int, default=15)
    parser.add_argument("--l2-regularization", type=float, default=1.0)
    parser.add_argument("--lightgbm-min-child-samples", type=int, default=200)
    parser.add_argument("--lightgbm-subsample", type=float, default=1.0)
    parser.add_argument("--lightgbm-colsample-bytree", type=float, default=1.0)
    parser.add_argument("--lightgbm-n-jobs", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=17)
    parser.add_argument("--gbdt-seeds", default="")
    parser.add_argument("--blend-group-columns", default="")
    parser.add_argument("--min-blend-group-rows", type=int, default=2_000)
    parser.add_argument("--min-group-rows", type=int, default=2_000)
    parser.add_argument("--clip-target-abs-quantile", type=float, default=0.999)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/ridge_gbdt_blend"))
    args = parser.parse_args()
    args.gbdt_seed_values = _parse_gbdt_seeds(args.gbdt_seeds, args.random_state)
    args.blend_group_values = _parse_blend_group_columns(args.blend_group_columns)
    args.gbdt_id_values = _parse_gbdt_id_columns(args.gbdt_id_columns)
    _validate_args(args)

    train_path = TRAIN_WITH_RESPONDER_LAGS_PARQUET if _uses_same_time_lags(args) else TRAIN_PARQUET_DIR
    if _uses_same_time_lags(args) and not train_path.exists():
        raise FileNotFoundError(
            f"{train_path} not found. Run: uv run python scripts/build_responder_lag_cache.py"
        )
    train = _scan_train_path(train_path)
    base_features = feature_columns_from_schema(train.collect_schema())
    ridge_features = base_features
    gbdt_raw_features = base_features
    gbdt_model_features = base_features
    lag_features: tuple[str, ...] = ()
    if args.use_responder_lags:
        if args.responder_lag_mode == "same_time":
            lag_features = responder_lag_columns()
        else:
            if not DAILY_RESPONDER_LAGS_LAST_PARQUET.exists():
                raise FileNotFoundError(
                    f"{DAILY_RESPONDER_LAGS_LAST_PARQUET} not found. "
                    "Run: uv run python scripts/build_daily_responder_lag_cache.py"
                )
            lag_features = daily_last_responder_lag_columns()
            train = train.join(
                pl.scan_parquet(str(DAILY_RESPONDER_LAGS_LAST_PARQUET / "*.parquet")),
                on=list(DAILY_LAG_JOIN_KEYS),
                how="left",
            )
        if args.responder_lag_target == "both":
            ridge_features = ridge_features + lag_features
        gbdt_raw_features = gbdt_raw_features + lag_features
        gbdt_model_features = gbdt_raw_features
    temporal_geometry_columns: tuple[str, ...] = ()
    temporal_geometry_windows: tuple[int, ...] = ()
    temporal_geometry_feature_count = 0
    if args.temporal_geometry_columns:
        temporal_geometry_columns = parse_temporal_geometry_columns(args.temporal_geometry_columns)
        temporal_geometry_windows = parse_temporal_geometry_windows(args.temporal_geometry_windows)
        require_temporal_geometry_columns(temporal_geometry_columns, base_features)
        temporal_spec = TemporalGeometrySpec(
            columns=temporal_geometry_columns,
            windows=temporal_geometry_windows,
        )
        train = with_temporal_geometry_features(train, temporal_spec)
        if args.temporal_geometry_target == "both":
            ridge_features = base_features + temporal_spec.output_columns
        gbdt_raw_features = base_features + temporal_spec.output_columns
        gbdt_model_features = gbdt_raw_features
        temporal_geometry_feature_count = len(temporal_spec.output_columns)
    multiscale_columns = parse_multiscale_columns(args.multiscale_columns)
    multiscale_spans: tuple[int, ...] = ()
    multiscale_feature_count = 0
    if multiscale_columns:
        multiscale_spans = parse_multiscale_spans(args.multiscale_spans)
        require_multiscale_columns(multiscale_columns, gbdt_raw_features)
        multiscale_spec = MultiscaleSpec(columns=multiscale_columns, spans=multiscale_spans)
        train = with_multiscale_features(train, multiscale_spec)
        if args.multiscale_target == "both":
            ridge_features = ridge_features + multiscale_spec.output_columns
        gbdt_raw_features = gbdt_raw_features + multiscale_spec.output_columns
        gbdt_model_features = gbdt_raw_features
        multiscale_feature_count = len(multiscale_spec.output_columns)
    koopman_columns = parse_koopman_columns(args.koopman_columns)
    koopman_feature_count = 0
    if koopman_columns:
        require_koopman_columns(koopman_columns, gbdt_raw_features)
        koopman_spec = KoopmanSpec(columns=koopman_columns, span=args.koopman_span)
        train = with_koopman_features(train, koopman_spec)
        if args.koopman_target == "both":
            ridge_features = ridge_features + koopman_spec.output_columns
        gbdt_raw_features = gbdt_raw_features + koopman_spec.output_columns
        gbdt_model_features = gbdt_raw_features
        koopman_feature_count = len(koopman_spec.output_columns)
    reservoir_columns = parse_reservoir_columns(args.reservoir_columns)
    reservoir_spans: tuple[int, ...] = ()
    reservoir_feature_count = 0
    if reservoir_columns:
        reservoir_spans = parse_reservoir_spans(args.reservoir_spans)
        require_reservoir_columns(reservoir_columns, gbdt_raw_features)
        reservoir_spec = make_reservoir_spec(
            reservoir_columns,
            n_states=args.reservoir_states,
            spans=reservoir_spans,
            seed=args.reservoir_seed,
        )
        train = with_reservoir_features(train, reservoir_spec)
        if args.reservoir_target == "both":
            ridge_features = ridge_features + reservoir_spec.output_columns
        gbdt_raw_features = gbdt_raw_features + reservoir_spec.output_columns
        gbdt_model_features = gbdt_raw_features
        reservoir_feature_count = len(reservoir_spec.output_columns)
    symbol_graph_columns = parse_symbol_graph_columns(args.symbol_graph_columns)
    symbol_graph_feature_count = 0
    if symbol_graph_columns:
        require_symbol_graph_columns(symbol_graph_columns, gbdt_raw_features)
        symbol_graph_feature_count = 1 + 2 * len(symbol_graph_columns)
        graph_output_columns = ["symbol_graph_neighbor_count"]
        for column in symbol_graph_columns:
            graph_output_columns.append(f"{column}_sg_neighbor_mean")
            graph_output_columns.append(f"{column}_sg_deviation")
        gbdt_model_features = gbdt_raw_features + tuple(graph_output_columns)
    if args.gbdt_id_values:
        gbdt_model_features = tuple(dict.fromkeys([*gbdt_model_features, *args.gbdt_id_values]))
    folds = _make_folds(train, args)
    rows: list[dict[str, float | int | str | None]] = []
    parameter_rows: list[dict[str, float | int | str | None]] = []
    slice_partials = {"weight_bucket": [], "time_bucket": [], "date_id_symbol_id": []}

    for fold in folds:
        calibration = _collect_inner_oof_predictions(
            train,
            ridge_features,
            gbdt_raw_features,
            gbdt_model_features,
            symbol_graph_columns,
            fold,
            args,
        )
        clip_abs = _target_abs_quantile(calibration, args.clip_target_abs_quantile)
        weight_thresholds = _weight_thresholds(calibration)
        pred_abs_thresholds = fit_abs_prediction_thresholds(calibration, prediction="ridge_prediction")
        calibration = _add_regime_columns(
            calibration,
            weight_thresholds,
            pred_abs_thresholds,
            time_bucket_size=args.time_bucket_size,
        )
        ridge_calibrator = fit_shrinkage_calibrator(
            calibration,
            name="ridge_weight_predabs",
            group_columns=["weight_bucket", "prediction_abs_bucket"],
            prediction="ridge_prediction",
            min_group_rows=args.min_group_rows,
            clip_abs=clip_abs,
        )
        calibration = ridge_calibrator.apply(
            calibration,
            prediction="ridge_prediction",
            output="ridge_calibrated_prediction",
        )
        blend_weight = fit_convex_blend_weight(
            calibration,
            left_prediction="ridge_calibrated_prediction",
            right_prediction="gbdt_prediction",
        )
        grouped_blend = _fit_grouped_blend_if_requested(calibration, args)
        parameter_rows.append(
            _global_parameter_row(
                fold=fold,
                clip_abs=clip_abs,
                ridge_fallback_alpha=ridge_calibrator.fallback_alpha,
                blend_weight=blend_weight,
            )
        )
        if grouped_blend is not None:
            parameter_rows.extend(
                _grouped_blend_parameter_rows(
                    fold=fold,
                    clip_abs=clip_abs,
                    ridge_fallback_alpha=ridge_calibrator.fallback_alpha,
                    grouped_blend=grouped_blend,
                )
            )

        models = _fit_models(
            train,
            ridge_features,
            gbdt_raw_features,
            gbdt_model_features,
            symbol_graph_columns,
            fold,
            args,
        )
        validation = _collect_prediction_frame(
            train,
            ridge_features,
            gbdt_raw_features,
            gbdt_model_features,
            models,
            fold.valid_start,
            fold.valid_end,
            chunk_days=args.chunk_days,
        )
        validation = _add_regime_columns(
            validation,
            weight_thresholds,
            pred_abs_thresholds,
            time_bucket_size=args.time_bucket_size,
        )
        validation = ridge_calibrator.apply(
            validation,
            prediction="ridge_prediction",
            output="ridge_calibrated_prediction",
        )
        validation = add_convex_blend_prediction(
            validation,
            blend_weight=blend_weight,
            left_prediction="ridge_calibrated_prediction",
            right_prediction="gbdt_prediction",
            output="blend_prediction",
        )
        if grouped_blend is not None:
            validation = grouped_blend.apply(
                validation,
                left_prediction="ridge_calibrated_prediction",
                right_prediction="gbdt_prediction",
                output="blend_regime_prediction",
            )

        strategy_columns = {
            "ridge_calibrated": "ridge_calibrated_prediction",
            "gbdt": "gbdt_prediction",
            "blend": "blend_prediction",
        }
        if grouped_blend is not None:
            strategy_columns["blend_regime"] = "blend_regime_prediction"
        for strategy, prediction in strategy_columns.items():
            scored = validation.with_columns(pl.col(prediction).alias("strategy_prediction"))
            rows.append(
                {
                    **_fold_metadata(fold),
                    "strategy": strategy,
                    "blend_weight_ridge": blend_weight if strategy == "blend" else None,
                    **_score_frame(scored),
                }
            )
            for name, by in {
                "weight_bucket": ["weight_bucket"],
                "time_bucket": ["time_bucket"],
                "date_id_symbol_id": ["date_id", "symbol_id"],
            }.items():
                slice_partials[name].append(
                    aggregate_weighted_r2_by_slice(scored, by, prediction="strategy_prediction")
                    .with_columns(pl.lit(strategy).alias("strategy"))
                )

    results = pl.DataFrame(rows)
    summary = _summary_by_strategy(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(args.output_dir / "ridge_gbdt_blend_by_fold.csv")
    summary.write_csv(args.output_dir / "ridge_gbdt_blend_summary.csv")
    pl.DataFrame(parameter_rows).write_csv(args.output_dir / "ridge_gbdt_blend_parameters.csv")
    _write_slice_outputs(slice_partials, args.output_dir)
    report = {
        "experiment": "ridge_gbdt_blend",
        "hypothesis": "A convex OOF-trained blend can combine Ridge stability with GBDT global signal.",
        "best_strategy": summary.row(0, named=True),
        "n_folds": args.n_folds,
        "inner_oof_folds": args.inner_oof_folds,
        "inner_valid_window": args.inner_valid_window,
        "train_sample_frac": args.train_sample_frac,
        "gbdt_engine": args.gbdt_engine,
        "gbdt_target_mode": args.gbdt_target_mode,
        "gbdt_id_columns": args.gbdt_id_values,
        "gbdt_seeds": args.gbdt_seed_values,
        "blend_group_columns": args.blend_group_values,
        "min_blend_group_rows": args.min_blend_group_rows,
        "time_bucket_size": args.time_bucket_size,
        "n_ridge_features": len(ridge_features),
        "n_gbdt_features": len(gbdt_model_features),
        "use_responder_lags": args.use_responder_lags,
        "responder_lag_mode": args.responder_lag_mode,
        "responder_lag_target": args.responder_lag_target,
        "responder_lag_features": lag_features,
        "responder_lag_feature_count": len(lag_features),
        "responder_lag_join_keys": ("date_id", "time_id", "symbol_id"),
        "temporal_geometry_columns": temporal_geometry_columns,
        "temporal_geometry_windows": temporal_geometry_windows,
        "temporal_geometry_target": args.temporal_geometry_target,
        "temporal_geometry_feature_count": temporal_geometry_feature_count,
        "multiscale_columns": multiscale_columns,
        "multiscale_spans": multiscale_spans,
        "multiscale_target": args.multiscale_target,
        "multiscale_feature_count": multiscale_feature_count,
        "koopman_columns": koopman_columns,
        "koopman_span": args.koopman_span,
        "koopman_target": args.koopman_target,
        "koopman_feature_count": koopman_feature_count,
        "reservoir_columns": reservoir_columns,
        "reservoir_states": args.reservoir_states,
        "reservoir_spans": reservoir_spans,
        "reservoir_seed": args.reservoir_seed,
        "reservoir_target": args.reservoir_target,
        "reservoir_feature_count": reservoir_feature_count,
        "symbol_graph_columns": symbol_graph_columns,
        "symbol_graph_neighbors": args.symbol_graph_neighbors,
        "symbol_graph_feature_count": symbol_graph_feature_count,
    }
    (args.output_dir / "ridge_gbdt_blend_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(results)
    print(summary)
    print(f"Wrote {args.output_dir}")


def _validate_args(args: argparse.Namespace) -> None:
    if args.train_window <= 0:
        raise ValueError("--train-window must be positive")
    if args.valid_window <= 0:
        raise ValueError("--valid-window must be positive")
    if args.inner_oof_folds <= 0:
        raise ValueError("--inner-oof-folds must be positive")
    if args.inner_valid_window <= 0:
        raise ValueError("--inner-valid-window must be positive")
    if args.chunk_days <= 0:
        raise ValueError("--chunk-days must be positive")
    if args.time_bucket_size <= 0:
        raise ValueError("--time-bucket-size must be positive")
    if args.train_sample_frac <= 0.0 or args.train_sample_frac > 1.0:
        raise ValueError("--train-sample-frac must be in (0, 1]")
    if args.min_group_rows <= 0:
        raise ValueError("--min-group-rows must be positive")
    if args.min_blend_group_rows <= 0:
        raise ValueError("--min-blend-group-rows must be positive")
    if args.symbol_graph_neighbors <= 0:
        raise ValueError("--symbol-graph-neighbors must be positive")
    if args.reservoir_states <= 0:
        raise ValueError("--reservoir-states must be positive")
    if args.koopman_span <= 0:
        raise ValueError("--koopman-span must be positive")
    if args.lightgbm_min_child_samples <= 0:
        raise ValueError("--lightgbm-min-child-samples must be positive")
    if args.lightgbm_subsample <= 0.0 or args.lightgbm_subsample > 1.0:
        raise ValueError("--lightgbm-subsample must be in (0, 1]")
    if args.lightgbm_colsample_bytree <= 0.0 or args.lightgbm_colsample_bytree > 1.0:
        raise ValueError("--lightgbm-colsample-bytree must be in (0, 1]")
    if args.lightgbm_n_jobs <= 0:
        raise ValueError("--lightgbm-n-jobs must be positive")
    allowed_blend_groups = {"time_bucket", "weight_bucket", "prediction_abs_bucket"}
    unknown = set(args.blend_group_values) - allowed_blend_groups
    if unknown:
        raise ValueError(f"unknown --blend-group-columns: {', '.join(sorted(unknown))}")
    allowed_id_columns = {"time_id", "symbol_id"}
    unknown_ids = set(args.gbdt_id_values) - allowed_id_columns
    if unknown_ids:
        raise ValueError(f"unknown --gbdt-id-columns: {', '.join(sorted(unknown_ids))}")


def _scan_train_path(path: Path) -> pl.LazyFrame:
    if path.is_dir() and any(path.glob("*.parquet")):
        return pl.scan_parquet(str(path / "*.parquet"))
    return pl.scan_parquet(str(path))


def _uses_same_time_lags(args: argparse.Namespace) -> bool:
    return bool(args.use_responder_lags and args.responder_lag_mode == "same_time")


def _parse_gbdt_seeds(raw: str, fallback: int) -> tuple[int, ...]:
    if not raw.strip():
        return (fallback,)
    seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not seeds:
        raise ValueError("--gbdt-seeds must contain at least one integer")
    return seeds


def _parse_blend_group_columns(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_gbdt_id_columns(raw: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))


def _fit_grouped_blend_if_requested(
    calibration: pl.DataFrame,
    args: argparse.Namespace,
) -> GroupedBlendWeights | None:
    if not args.blend_group_values:
        return None
    return fit_grouped_convex_blend_weights(
        calibration,
        group_columns=args.blend_group_values,
        left_prediction="ridge_calibrated_prediction",
        right_prediction="gbdt_prediction",
        min_group_rows=args.min_blend_group_rows,
    )


def _global_parameter_row(
    *,
    fold: DateFold,
    clip_abs: float,
    ridge_fallback_alpha: float,
    blend_weight: float,
) -> dict[str, float | int | str | None]:
    return {
        "fold": fold.name,
        "parameter_type": "global_blend",
        "blend_group_columns": "",
        "blend_group_key": "",
        "blend_group_rows": None,
        "clip_abs": clip_abs,
        "ridge_fallback_alpha": ridge_fallback_alpha,
        "blend_weight_ridge": blend_weight,
        "blend_weight_gbdt": 1.0 - blend_weight,
    }


def _grouped_blend_parameter_rows(
    *,
    fold: DateFold,
    clip_abs: float,
    ridge_fallback_alpha: float,
    grouped_blend: GroupedBlendWeights,
) -> list[dict[str, float | int | str | None]]:
    rows: list[dict[str, float | int | str | None]] = [
        {
            "fold": fold.name,
            "parameter_type": "grouped_blend_fallback",
            "blend_group_columns": ",".join(grouped_blend.group_columns),
            "blend_group_key": "__fallback__",
            "blend_group_rows": None,
            "clip_abs": clip_abs,
            "ridge_fallback_alpha": ridge_fallback_alpha,
            "blend_weight_ridge": grouped_blend.fallback_weight,
            "blend_weight_gbdt": 1.0 - grouped_blend.fallback_weight,
        }
    ]
    for row in grouped_blend.parameters.iter_rows(named=True):
        key = "|".join(str(row[column]) for column in grouped_blend.group_columns)
        blend_weight = float(row["_blend_weight"])
        rows.append(
            {
                "fold": fold.name,
                "parameter_type": "grouped_blend",
                "blend_group_columns": ",".join(grouped_blend.group_columns),
                "blend_group_key": key,
                "blend_group_rows": int(row["_blend_rows"]),
                "clip_abs": clip_abs,
                "ridge_fallback_alpha": ridge_fallback_alpha,
                "blend_weight_ridge": blend_weight,
                "blend_weight_gbdt": 1.0 - blend_weight,
            }
        )
    return rows


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


def _collect_inner_oof_predictions(
    train: pl.LazyFrame,
    ridge_features: tuple[str, ...],
    gbdt_raw_features: tuple[str, ...],
    gbdt_model_features: tuple[str, ...],
    symbol_graph_columns: tuple[str, ...],
    fold: DateFold,
    args: argparse.Namespace,
) -> pl.DataFrame:
    return pl.concat(
        [
            _collect_prediction_frame(
                train,
                ridge_features,
                gbdt_raw_features,
                gbdt_model_features,
                _fit_models(
                    train,
                    ridge_features,
                    gbdt_raw_features,
                    gbdt_model_features,
                    symbol_graph_columns,
                    inner_fold,
                    args,
                ),
                inner_fold.valid_start,
                inner_fold.valid_end,
                chunk_days=args.chunk_days,
            )
            for inner_fold in _inner_oof_folds(fold, args.inner_oof_folds, args.inner_valid_window)
        ]
    )


def _inner_oof_folds(fold: DateFold, n_folds: int, valid_window: int) -> list[DateFold]:
    first_valid_start = fold.train_end - n_folds * valid_window + 1
    if first_valid_start <= fold.train_start:
        raise ValueError(f"{fold.name}: not enough train days for requested inner OOF folds")
    return [
        DateFold(
            name=f"{fold.name}_oof_{idx + 1:02d}",
            train_start=fold.train_start,
            train_end=first_valid_start + idx * valid_window - 1,
            valid_start=first_valid_start + idx * valid_window,
            valid_end=first_valid_start + (idx + 1) * valid_window - 1,
        )
        for idx in range(n_folds)
    ]


def _fit_models(
    train: pl.LazyFrame,
    ridge_features: tuple[str, ...],
    gbdt_raw_features: tuple[str, ...],
    gbdt_model_features: tuple[str, ...],
    symbol_graph_columns: tuple[str, ...],
    fold: DateFold,
    args: argparse.Namespace,
) -> PredictionModels:
    ridge_model = solve_weighted_ridge(
        build_weighted_ridge_fit_data(
            train,
            fold,
            feature_columns=ridge_features,
            chunk_days=args.chunk_days,
        ),
        alpha=args.ridge_alpha,
    )
    symbol_graph_spec = None
    if symbol_graph_columns:
        symbol_graph_spec = fit_symbol_graph_spec(
            train,
            start=fold.train_start,
            end=fold.train_end,
            columns=symbol_graph_columns,
            n_neighbors=args.symbol_graph_neighbors,
        )
    gbdt_models: list[HistGradientBoostingRegressor | LGBMRegressor | XGBRegressor | CatBoostRegressor] = []
    for seed in args.gbdt_seed_values:
        x_train, y_train, w_train = _collect_gbdt_train_sample(
            train,
            ridge_features,
            gbdt_raw_features,
            gbdt_model_features,
            ridge_model,
            fold.train_start,
            fold.train_end,
            sample_frac=args.train_sample_frac,
            seed=seed,
            chunk_days=args.chunk_days,
            symbol_graph_spec=symbol_graph_spec,
            target_mode=args.gbdt_target_mode,
        )
        gbdt_model = _make_gbdt_model(args, seed)
        cat_feature_indices = _cat_feature_indices(args, gbdt_model_features)
        if args.gbdt_engine == "catboost":
            gbdt_model.fit(
                Pool(
                    _catboost_matrix(x_train, cat_feature_indices),
                    y_train,
                    weight=w_train,
                    cat_features=list(cat_feature_indices),
                )
            )
        else:
            gbdt_model.fit(x_train, y_train, sample_weight=w_train)
        gbdt_models.append(gbdt_model)
    return PredictionModels(
        ridge_model=ridge_model,
        gbdt_models=tuple(gbdt_models),
        gbdt_target_mode=args.gbdt_target_mode,
        cat_feature_indices=_cat_feature_indices(args, gbdt_model_features),
        symbol_graph_spec=symbol_graph_spec,
    )


def _cat_feature_indices(args: argparse.Namespace, model_features: tuple[str, ...]) -> tuple[int, ...]:
    if args.gbdt_engine != "catboost" or not args.catboost_categorical_id_columns:
        return ()
    cat_names = {"time_id", "symbol_id"}
    return tuple(idx for idx, name in enumerate(model_features) if name in cat_names)


def _make_gbdt_model(
    args: argparse.Namespace,
    seed: int,
) -> HistGradientBoostingRegressor | LGBMRegressor | XGBRegressor | CatBoostRegressor:
    if args.gbdt_engine == "sklearn":
        return HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=args.learning_rate,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaf_nodes,
            l2_regularization=args.l2_regularization,
            early_stopping=False,
            random_state=seed,
        )
    if args.gbdt_engine == "lightgbm":
        return LGBMRegressor(
            objective="regression_l2",
            n_estimators=args.max_iter,
            learning_rate=args.learning_rate,
            num_leaves=args.max_leaf_nodes,
            reg_lambda=args.l2_regularization,
            min_child_samples=args.lightgbm_min_child_samples,
            subsample=args.lightgbm_subsample,
            subsample_freq=1 if args.lightgbm_subsample < 1.0 else 0,
            colsample_bytree=args.lightgbm_colsample_bytree,
            random_state=seed,
            n_jobs=args.lightgbm_n_jobs,
            force_col_wise=True,
            verbosity=-1,
        )
    if args.gbdt_engine == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=args.max_iter,
            learning_rate=args.learning_rate,
            max_leaves=args.max_leaf_nodes,
            grow_policy="lossguide",
            tree_method="hist",
            device="cpu",
            reg_lambda=args.l2_regularization,
            subsample=args.lightgbm_subsample,
            colsample_bytree=args.lightgbm_colsample_bytree,
            random_state=seed,
            n_jobs=args.lightgbm_n_jobs,
            verbosity=0,
        )
    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=args.max_iter,
        learning_rate=args.learning_rate,
        depth=max(1, int(np.ceil(np.log2(args.max_leaf_nodes)))),
        l2_leaf_reg=args.l2_regularization,
        random_seed=seed,
        thread_count=args.lightgbm_n_jobs,
        task_type="CPU",
        verbose=False,
        allow_writing_files=False,
    )


def _collect_gbdt_train_sample(
    data: pl.LazyFrame,
    ridge_features: tuple[str, ...],
    raw_features: tuple[str, ...],
    model_features: tuple[str, ...],
    ridge_model: object,
    start: int,
    end: int,
    *,
    sample_frac: float,
    seed: int,
    chunk_days: int,
    symbol_graph_spec: SymbolGraphSpec | None,
    target_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    threshold = int(sample_frac * 1_000_000)
    frames: list[pl.DataFrame] = []
    selected_features = tuple(dict.fromkeys([*raw_features, *ridge_features]))
    for chunk_start, chunk_end in _date_chunks(start, end, chunk_days):
        sample_filter = (pl.struct(["date_id", "time_id", "symbol_id"]).hash(seed=seed) % 1_000_000) < threshold
        frame = (
            data.filter(pl.col("date_id").is_between(chunk_start, chunk_end))
            .select(
                [
                    pl.col("date_id").cast(pl.Int32),
                    pl.col("time_id").cast(pl.Int32),
                    pl.col("symbol_id").cast(pl.Int16),
                ]
                + [pl.col(name).cast(pl.Float32).alias(name) for name in selected_features]
                + [pl.col("responder_6").cast(pl.Float32), pl.col("weight").cast(pl.Float32)]
            )
            .collect()
        )
        if symbol_graph_spec is not None:
            frame = add_symbol_graph_features(frame, symbol_graph_spec)
        sampled = frame.filter(sample_filter)
        output_columns = tuple(dict.fromkeys([*model_features, *ridge_features, "responder_6", "weight"]))
        frames.append(sampled.select(list(output_columns)))
    sample = pl.concat(frames)
    y = sample["responder_6"].to_numpy()
    if target_mode == "residual_raw_ridge":
        ridge_x = sample.select(
            [pl.col(name).fill_null(0.0).cast(pl.Float64).alias(name) for name in ridge_features]
        ).to_numpy()
        y = y - ridge_model.predict_array(ridge_x).astype(y.dtype, copy=False)
    elif target_mode != "target":
        raise ValueError(f"unknown GBDT target mode: {target_mode}")
    return (
        sample.select(list(model_features)).to_numpy(),
        y,
        sample["weight"].to_numpy(),
    )


def _collect_prediction_frame(
    data: pl.LazyFrame,
    ridge_features: tuple[str, ...],
    gbdt_raw_features: tuple[str, ...],
    gbdt_model_features: tuple[str, ...],
    models: PredictionModels,
    start: int,
    end: int,
    *,
    chunk_days: int,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    selected_features = tuple(dict.fromkeys([*ridge_features, *gbdt_raw_features]))
    for chunk_start, chunk_end in _date_chunks(start, end, chunk_days):
        frame = (
            data.filter(pl.col("date_id").is_between(chunk_start, chunk_end))
            .select(
                [
                    pl.col("date_id").cast(pl.Int32),
                    pl.col("time_id").cast(pl.Int32),
                    pl.col("symbol_id").cast(pl.Int16),
                    pl.col("weight").cast(pl.Float64),
                    pl.col("responder_6").cast(pl.Float64),
                ]
                + [pl.col(name).cast(pl.Float32).alias(name) for name in selected_features]
            )
            .collect()
        )
        if models.symbol_graph_spec is not None:
            frame = add_symbol_graph_features(frame, models.symbol_graph_spec)
        ridge_x = frame.select(
            [pl.col(name).fill_null(0.0).cast(pl.Float64).alias(name) for name in ridge_features]
        ).to_numpy()
        gbdt_x = frame.select(list(gbdt_model_features)).to_numpy()
        ridge_prediction = models.ridge_model.predict_array(ridge_x)
        gbdt_prediction = _predict_gbdt_ensemble(models.gbdt_models, gbdt_x, models.cat_feature_indices)
        if models.gbdt_target_mode == "residual_raw_ridge":
            gbdt_prediction = ridge_prediction + gbdt_prediction
        elif models.gbdt_target_mode != "target":
            raise ValueError(f"unknown GBDT target mode: {models.gbdt_target_mode}")
        frames.append(
            frame.select(["date_id", "time_id", "symbol_id", "weight", "responder_6"])
            .with_columns(
                [
                    pl.Series("ridge_prediction", ridge_prediction),
                    pl.Series("gbdt_prediction", gbdt_prediction),
                ]
            )
        )
    return pl.concat(frames)


def _predict_gbdt_ensemble(
    models: tuple[HistGradientBoostingRegressor | LGBMRegressor | XGBRegressor | CatBoostRegressor, ...],
    features: np.ndarray,
    cat_feature_indices: tuple[int, ...] = (),
) -> np.ndarray:
    if not models:
        raise ValueError("gbdt model ensemble must not be empty")
    prediction = np.zeros(features.shape[0], dtype=np.float64)
    for model in models:
        prediction += _predict_single_gbdt(model, features, cat_feature_indices).astype(np.float64, copy=False)
    return prediction / len(models)


def _predict_single_gbdt(
    model: HistGradientBoostingRegressor | LGBMRegressor | XGBRegressor | CatBoostRegressor,
    features: np.ndarray,
    cat_feature_indices: tuple[int, ...] = (),
) -> np.ndarray:
    if isinstance(model, CatBoostRegressor) and cat_feature_indices:
        return model.predict(Pool(_catboost_matrix(features, cat_feature_indices), cat_features=list(cat_feature_indices)))
    return model.predict(features)


def _catboost_matrix(features: np.ndarray, cat_feature_indices: tuple[int, ...]) -> np.ndarray:
    if not cat_feature_indices:
        return features
    matrix = features.astype(object)
    for idx in cat_feature_indices:
        matrix[:, idx] = features[:, idx].astype(np.int64).astype(str)
    return matrix


def _add_regime_columns(
    frame: pl.DataFrame,
    weight_thresholds: dict[str, float],
    pred_abs_thresholds: dict[str, float],
    *,
    time_bucket_size: int,
) -> pl.DataFrame:
    with_regimes = frame.with_columns(
        [
            (pl.col("time_id") // time_bucket_size).cast(pl.Int16).alias("time_bucket"),
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
    return add_abs_prediction_bucket(
        with_regimes,
        pred_abs_thresholds,
        prediction="ridge_prediction",
    )


def _score_frame(frame: pl.DataFrame) -> dict[str, float | int]:
    row = frame.select(
        [
            pl.len().alias("rows"),
            pl.col("weight").sum().alias("weight_sum"),
            (pl.col("weight") * (pl.col("responder_6") - pl.col("strategy_prediction")).pow(2)).sum().alias(
                "numerator"
            ),
            (pl.col("weight") * pl.col("responder_6").pow(2)).sum().alias("denominator"),
            pl.col("strategy_prediction").mean().alias("prediction_mean"),
            pl.col("strategy_prediction").std().alias("prediction_std"),
        ]
    ).row(0, named=True)
    return {
        "rows": int(row["rows"]),
        "weight_sum": float(row["weight_sum"]),
        "numerator": float(row["numerator"]),
        "denominator": float(row["denominator"]),
        "weighted_zero_mean_r2": 1.0 - float(row["numerator"]) / float(row["denominator"]),
        "prediction_mean": float(row["prediction_mean"]),
        "prediction_std": float(row["prediction_std"]),
    }


def _summary_by_strategy(results: pl.DataFrame) -> pl.DataFrame:
    return (
        results.group_by("strategy")
        .agg(
            pl.len().alias("folds"),
            pl.col("weighted_zero_mean_r2").mean().alias("mean_r2"),
            pl.col("weighted_zero_mean_r2").std().alias("std_r2"),
            pl.col("weighted_zero_mean_r2").min().alias("min_r2"),
            pl.col("weighted_zero_mean_r2").max().alias("max_r2"),
            (1.0 - pl.col("numerator").sum() / pl.col("denominator").sum()).alias("global_r2"),
            pl.col("rows").sum().alias("validation_rows"),
        )
        .sort(["global_r2", "min_r2"], descending=[True, True])
    )


def _write_slice_outputs(slice_partials: dict[str, list[pl.DataFrame]], output_dir: Path) -> None:
    for name, group_cols in {
        "weight_bucket": ["strategy", "weight_bucket"],
        "time_bucket": ["strategy", "time_bucket"],
        "date_id_symbol_id": ["strategy", "date_id", "symbol_id"],
    }.items():
        combine_slice_aggregates(slice_partials[name], group_cols).write_csv(output_dir / f"{name}.csv")


def _target_abs_quantile(frame: pl.DataFrame, quantile: float) -> float:
    return float(frame.select(pl.col("responder_6").abs().quantile(quantile)).item())


def _weight_thresholds(frame: pl.DataFrame) -> dict[str, float]:
    row = frame.select(
        pl.col("weight").quantile(0.50).alias("q50"),
        pl.col("weight").quantile(0.90).alias("q90"),
        pl.col("weight").quantile(0.99).alias("q99"),
    ).row(0, named=True)
    return {name: float(row[name]) for name in ("q50", "q90", "q99")}


def _fold_metadata(fold: DateFold) -> dict[str, int | str]:
    return {
        "fold": fold.name,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "valid_start": fold.valid_start,
        "valid_end": fold.valid_end,
    }


def _date_chunks(start: int, end: int, chunk_days: int) -> list[tuple[int, int]]:
    chunks: list[tuple[int, int]] = []
    current = start
    while current <= end:
        chunk_end = min(current + chunk_days - 1, end)
        chunks.append((current, chunk_end))
        current = chunk_end + 1
    return chunks


if __name__ == "__main__":
    main()
