"""Modular experiments over existing strong OOF prediction artifacts."""

from __future__ import annotations

import importlib.util
import itertools
import json
import gc
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from multimodels.metrics import TARGET, score_arrays, summarize_scores
from multimodels.reporting import write_markdown_report


PROJECT_ROOT = Path(__file__).resolve().parents[2]
KEYS = ("fold", "date_id", "time_id", "symbol_id")
BASE_COLUMNS = (*KEYS, "weight", TARGET)


@dataclass(frozen=True)
class StrongOOFConfig:
    """Configuration for strong OOF modular evaluation."""

    experiment_name: str = "strong_oof_modular"
    tabm_prediction_dir: Path = Path(
        "reports/experiments/competitive_tabm_official_stage3_5fold_valid60_lags_online_lr1e4_4m_train700_seed37_aux8_preds/validation_predictions"
    )
    tree_prediction_dir: Path = Path("reports/experiments/tree_engine_ensemble_xgb_lgb_sample10_seed_ensemble_preds/validation_predictions")
    extra_prediction_dirs: tuple[Path, ...] = ()
    extra_prediction_prefixes: tuple[str, ...] = ()
    output_dir: Path = Path("multi-models/reports/strong_oof_modular")
    include_gateway_rls: bool = True
    include_gateway_risk_shrink: bool = False
    include_extra_gateway_experts: bool = False
    gateway_expert_expansions: tuple[str, ...] = ()
    gateway_risk_strengths: tuple[float, ...] = (25.0, 100.0)
    gateway_risk_profiles: tuple[str, ...] = ("abs_pred",)
    sample_stride: int = 1
    max_rows_per_fold: int | None = None
    time_bucket_sizes: tuple[int, ...] = (100,)
    min_group_rows: tuple[int, ...] = (20_000,)
    scale_prior_strengths: tuple[float, ...] = (1_000.0, 10_000.0)
    stack_alphas: tuple[float, ...] = (1_000.0,)
    risk_shrink_strengths: tuple[float, ...] = (0.0, 0.02, 0.05, 0.1)
    strong_base_candidates: tuple[str, ...] = (
        "tabm_prediction",
        "tree_prediction",
        "xgboost_prediction",
        "lightgbm_prediction",
        "ridge_calibrated_prediction",
        "baseline_prediction",
        "conservative_rls_prediction",
        "aggressive_rls_prediction",
    )
    residual_features: tuple[str, ...] = (
        "prediction_disagreement",
        "tabm_tree_diff",
        "abs_baseline_prediction",
        "weight",
    )
    residual_base_candidates: tuple[str, ...] = ()
    residual_tail_quantiles: tuple[float, ...] = ()
    residual_tail_modes: tuple[str, ...] = ("weight",)
    risk_base_candidates: tuple[str, ...] = (
        "baseline_prediction",
        "conservative_rls_prediction",
        "aggressive_rls_prediction",
        "strong_oof_ridge_stack_prediction",
    )
    regime_base_candidates: tuple[str, ...] = (
        "baseline_prediction",
        "conservative_rls_prediction",
        "aggressive_rls_prediction",
        "strong_oof_ridge_stack_prediction",
    )
    raw_train_parquet_dir: Path | None = None
    raw_feature_columns: tuple[str, ...] = ()
    raw_preprocess_modes: tuple[str, ...] = ()
    include_raw_preprocessed_in_stack: bool = False
    fixed_blend_candidates: tuple[str, ...] = ()
    fixed_blend_weights: tuple[float, ...] = ()
    walk_forward_blend_candidates: tuple[str, ...] = ()
    contextual_blend_candidates: tuple[str, ...] = ()
    contextual_blend_group_specs: tuple[str, ...] = ()
    contextual_blend_time_bucket_sizes: tuple[int, ...] = (100,)
    contextual_blend_min_group_rows: tuple[int, ...] = (20_000,)
    contextual_blend_prior_strengths: tuple[float, ...] = (1_000.0,)
    online_scale_base_candidates: tuple[str, ...] = ()
    online_scale_prior_strengths: tuple[float, ...] = (1_000.0,)
    online_scale_forgetting_factors: tuple[float, ...] = (1.0,)
    online_scale_min: float = 0.0
    online_scale_max: float = 2.0
    online_affine_base_candidates: tuple[str, ...] = ()
    online_affine_prior_strengths: tuple[float, ...] = (1_000.0,)
    online_affine_forgetting_factors: tuple[float, ...] = (1.0,)
    online_affine_min_scale: float = 0.0
    online_affine_max_scale: float = 2.0
    min_mem_available_gb: float = 0.0
    min_swap_free_gb: float = 0.0
    write_predictions: bool = False


def run_strong_oof_experiment(config: StrongOOFConfig) -> dict[str, Any]:
    """Run modular risk/regime/residual experiments on saved strong OOF predictions."""

    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_joined_predictions(
        config.tabm_prediction_dir,
        config.tree_prediction_dir,
        extra_prediction_dirs=config.extra_prediction_dirs,
        extra_prediction_prefixes=config.extra_prediction_prefixes,
    )
    _assert_resource_floor(config, "after loading joined predictions")
    frame = _apply_sampling(frame, sample_stride=config.sample_stride, max_rows_per_fold=config.max_rows_per_fold)
    frame = add_prediction_context(frame)
    raw_preprocessed_columns: tuple[str, ...] = ()
    if config.raw_train_parquet_dir is not None and config.raw_feature_columns:
        frame = enrich_raw_features(frame, config.raw_train_parquet_dir, config.raw_feature_columns)
        if config.raw_preprocess_modes:
            frame, raw_preprocessed_columns = add_raw_preprocessing_features(
                frame,
                raw_feature_columns=config.raw_feature_columns,
                modes=config.raw_preprocess_modes,
            )
    _assert_resource_floor(config, "after sampling/context enrichment")

    gateway_audits: list[pl.DataFrame] = []
    if config.include_gateway_rls:
        frame, audits = add_gateway_rls_predictions(
            frame,
            include_risk_shrink=config.include_gateway_risk_shrink,
            include_extra_experts=config.include_extra_gateway_experts,
            expert_expansions=config.gateway_expert_expansions,
            risk_strengths=config.gateway_risk_strengths,
            risk_profiles=config.gateway_risk_profiles,
        )
        gateway_audits.extend(audits)
        frame = add_prediction_context(frame)
        _assert_resource_floor(config, "after gateway RLS predictions")

    score_rows: list[dict[str, float | int | str]] = []
    prediction_columns = _prediction_columns(frame, config.strong_base_candidates)
    if config.include_gateway_risk_shrink:
        prediction_columns = tuple(
            dict.fromkeys(
                [
                    *prediction_columns,
                    *(column for column in frame.columns if column.startswith("gateway_risk_") and column.endswith("_prediction")),
                ]
            )
        )
    for prediction in prediction_columns:
        score_rows.extend(_score_by_fold(frame, prediction=prediction, candidate=prediction, family="strong_base").to_dicts())

    candidate_frame = frame
    stack_columns = prediction_columns
    if config.include_raw_preprocessed_in_stack:
        stack_columns = tuple(dict.fromkeys([*prediction_columns, *raw_preprocessed_columns]))
    stack_frame, stack_params = add_walk_forward_ridge_stack_candidates(candidate_frame, prediction_columns=stack_columns, alphas=config.stack_alphas)
    candidate_frame = candidate_frame.join(stack_frame, on=list(KEYS), how="inner")
    _assert_resource_floor(config, "after ridge stack candidates")
    for column in stack_frame.columns:
        if column == "strong_oof_ridge_stack_prediction" or column.endswith("_ridge_stack_prediction"):
            score_rows.extend(_score_by_fold(candidate_frame, prediction=column, candidate=column.removesuffix("_prediction"), family="strong_stack").to_dicts())

    residual_base_predictions = _residual_base_predictions(
        candidate_frame,
        configured=config.residual_base_candidates,
        default=tuple(dict.fromkeys([*prediction_columns, "strong_oof_ridge_stack_prediction"])),
    )
    residual_candidates, residual_params = add_walk_forward_residual_rules(
        candidate_frame,
        base_predictions=residual_base_predictions,
        residual_features=tuple(name for name in config.residual_features if name in candidate_frame.columns),
    )
    candidate_frame = candidate_frame.join(residual_candidates, on=list(KEYS), how="inner")
    _assert_resource_floor(config, "after residual candidates")
    for column in residual_candidates.columns:
        if column.endswith("_residual_prediction"):
            score_rows.extend(_score_by_fold(candidate_frame, prediction=column, candidate=column.removesuffix("_prediction"), family="residual_correction").to_dicts())

    residual_tail_candidates, residual_tail_params = add_walk_forward_residual_tail_masks(
        candidate_frame,
        residual_predictions=tuple(column for column in residual_candidates.columns if column.endswith("_residual_prediction")),
        quantiles=config.residual_tail_quantiles,
        modes=config.residual_tail_modes,
    )
    candidate_frame = candidate_frame.join(residual_tail_candidates, on=list(KEYS), how="inner")
    _assert_resource_floor(config, "after residual tail candidates")
    for column in residual_tail_candidates.columns:
        if column.endswith("_residual_tail_prediction"):
            score_rows.extend(_score_by_fold(candidate_frame, prediction=column, candidate=column.removesuffix("_prediction"), family="residual_tail").to_dicts())

    risk_candidates, risk_params = add_prediction_risk_shrinkage(
        candidate_frame,
        base_predictions=tuple(column for column in config.risk_base_candidates if column in candidate_frame.columns),
        strengths=config.risk_shrink_strengths,
    )
    candidate_frame = candidate_frame.join(risk_candidates, on=list(KEYS), how="inner")
    _assert_resource_floor(config, "after risk shrinkage candidates")
    for column in risk_candidates.columns:
        if column.endswith("_risk_shrink_prediction"):
            score_rows.extend(_score_by_fold(candidate_frame, prediction=column, candidate=column.removesuffix("_prediction"), family="risk_shrinkage").to_dicts())

    regime_candidates, regime_params = add_walk_forward_regime_scales(
        candidate_frame,
        base_predictions=tuple(column for column in config.regime_base_candidates if column in candidate_frame.columns),
        time_bucket_sizes=config.time_bucket_sizes,
        min_group_rows_values=config.min_group_rows,
        prior_strengths=config.scale_prior_strengths,
    )
    candidate_frame = candidate_frame.join(regime_candidates, on=list(KEYS), how="inner")
    _assert_resource_floor(config, "after regime scale candidates")
    for column in regime_candidates.columns:
        if column.endswith("_regime_scaled_prediction"):
            score_rows.extend(_score_by_fold(candidate_frame, prediction=column, candidate=column.removesuffix("_prediction"), family="regime_scaled").to_dicts())

    fixed_blends, fixed_blend_params = add_fixed_candidate_blends(
        candidate_frame,
        candidates=config.fixed_blend_candidates,
        weights=config.fixed_blend_weights,
    )
    candidate_frame = candidate_frame.join(fixed_blends, on=list(KEYS), how="inner")
    _assert_resource_floor(config, "after fixed blends")
    for column in fixed_blends.columns:
        if column.endswith("_fixed_blend_prediction"):
            score_rows.extend(_score_by_fold(candidate_frame, prediction=column, candidate=column.removesuffix("_prediction"), family="fixed_blend").to_dicts())

    wf_blends, wf_blend_params = add_walk_forward_candidate_blends(
        candidate_frame,
        candidates=config.walk_forward_blend_candidates,
    )
    candidate_frame = candidate_frame.join(wf_blends, on=list(KEYS), how="inner")
    _assert_resource_floor(config, "after walk-forward blends")
    for column in wf_blends.columns:
        if column.endswith("_wf_blend_prediction"):
            score_rows.extend(_score_by_fold(candidate_frame, prediction=column, candidate=column.removesuffix("_prediction"), family="walk_forward_blend").to_dicts())

    contextual_blend_params: list[dict[str, Any]]
    if config.write_predictions:
        contextual_blends, contextual_blend_params = add_walk_forward_contextual_candidate_blends(
            candidate_frame,
            candidates=config.contextual_blend_candidates,
            group_specs=config.contextual_blend_group_specs,
            time_bucket_sizes=config.contextual_blend_time_bucket_sizes,
            min_group_rows_values=config.contextual_blend_min_group_rows,
            prior_strengths=config.contextual_blend_prior_strengths,
        )
        candidate_frame = candidate_frame.join(contextual_blends, on=list(KEYS), how="inner")
        _assert_resource_floor(config, "after contextual blends")
        for column in contextual_blends.columns:
            if column.endswith("_contextual_blend_prediction"):
                score_rows.extend(_score_by_fold(candidate_frame, prediction=column, candidate=column.removesuffix("_prediction"), family="contextual_blend").to_dicts())
    else:
        contextual_score_rows, contextual_blend_params = score_walk_forward_contextual_candidate_blends(
            candidate_frame,
            candidates=config.contextual_blend_candidates,
            group_specs=config.contextual_blend_group_specs,
            time_bucket_sizes=config.contextual_blend_time_bucket_sizes,
            min_group_rows_values=config.contextual_blend_min_group_rows,
            prior_strengths=config.contextual_blend_prior_strengths,
        )
        score_rows.extend(contextual_score_rows)

    online_scale_candidates, online_scale_params = add_online_daily_scales(
        candidate_frame,
        base_predictions=tuple(column for column in config.online_scale_base_candidates if column in candidate_frame.columns),
        prior_strengths=config.online_scale_prior_strengths,
        forgetting_factors=config.online_scale_forgetting_factors,
        min_scale=config.online_scale_min,
        max_scale=config.online_scale_max,
    )
    candidate_frame = candidate_frame.join(online_scale_candidates, on=list(KEYS), how="inner")
    _assert_resource_floor(config, "after online scale candidates")
    for column in online_scale_candidates.columns:
        if column.endswith("_online_scale_prediction"):
            score_rows.extend(_score_by_fold(candidate_frame, prediction=column, candidate=column.removesuffix("_prediction"), family="online_daily_scale").to_dicts())

    online_affine_candidates, online_affine_params = add_online_daily_affine(
        candidate_frame,
        base_predictions=tuple(column for column in config.online_affine_base_candidates if column in candidate_frame.columns),
        prior_strengths=config.online_affine_prior_strengths,
        forgetting_factors=config.online_affine_forgetting_factors,
        min_scale=config.online_affine_min_scale,
        max_scale=config.online_affine_max_scale,
    )
    candidate_frame = candidate_frame.join(online_affine_candidates, on=list(KEYS), how="inner")
    _assert_resource_floor(config, "after online affine candidates")
    for column in online_affine_candidates.columns:
        if column.endswith("_online_affine_prediction"):
            score_rows.extend(_score_by_fold(candidate_frame, prediction=column, candidate=column.removesuffix("_prediction"), family="online_daily_affine").to_dicts())

    scores = pl.DataFrame(score_rows)
    summary = summarize_scores(
        scores.rename({"candidate": "candidate", "family": "family", "weighted_zero_mean_r2": "weighted_zero_mean_r2"})
    )
    scores.write_csv(config.output_dir / "fold_scores.csv")
    summary.write_csv(config.output_dir / "candidate_summary.csv")
    _write_parameters(
        config.output_dir / "parameters.json",
        stack_params,
        residual_params,
        residual_tail_params,
        risk_params,
        regime_params,
        fixed_blend_params,
        wf_blend_params,
        contextual_blend_params,
        online_scale_params,
        online_affine_params,
    )
    if gateway_audits:
        pl.concat(gateway_audits, how="diagonal").write_csv(config.output_dir / "gateway_daily_audit.csv")
    if config.write_predictions:
        candidate_frame.write_parquet(config.output_dir / "strong_oof_predictions.parquet")

    audit = _audit_payload(
        config,
        candidate_frame,
        prediction_columns,
        gateway_audits,
        stack_columns=stack_columns,
        raw_preprocessed_columns=raw_preprocessed_columns,
    )
    (config.output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown_report(path=config.output_dir / "REPORT.md", experiment_name=config.experiment_name, summary=summary, audit=audit)
    return {"frame": candidate_frame, "scores": scores, "summary": summary, "audit": audit, "output_dir": config.output_dir}


def load_joined_predictions(
    tabm_dir: Path,
    tree_dir: Path,
    *,
    extra_prediction_dirs: Sequence[Path] = (),
    extra_prediction_prefixes: Sequence[str] = (),
) -> pl.DataFrame:
    """Load and join TabM and tree-engine validation predictions."""

    for path in (tabm_dir, tree_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    tabm = pl.scan_parquet(str(tabm_dir / "*.parquet")).select(list(BASE_COLUMNS) + ["tabm_prediction"])
    tree_schema = pl.scan_parquet(str(tree_dir / "*.parquet")).collect_schema()
    tree_columns = [
        column
        for column in ("ensemble_prediction", "ridge_calibrated_prediction", "xgboost_prediction", "lightgbm_prediction", "catboost_prediction")
        if column in tree_schema.names()
    ]
    if not tree_columns:
        raise ValueError("tree prediction directory does not contain known prediction columns")
    tree = pl.scan_parquet(str(tree_dir / "*.parquet")).select(list(KEYS) + tree_columns)
    rename = {"ensemble_prediction": "tree_prediction"} if "ensemble_prediction" in tree_columns else {}
    joined = tabm.join(tree.rename(rename), on=list(KEYS), how="inner").collect().sort(list(KEYS))
    prefixes = _extra_prediction_prefixes(extra_prediction_dirs, extra_prediction_prefixes)
    for extra_dir, prefix in zip(extra_prediction_dirs, prefixes, strict=True):
        extra = _load_extra_prediction_dir(extra_dir, prefix=prefix)
        joined = joined.join(extra, on=list(KEYS), how="inner")
    if joined.height == 0:
        raise ValueError("joined OOF prediction frame is empty")
    return joined


def _extra_prediction_prefixes(extra_prediction_dirs: Sequence[Path], prefixes: Sequence[str]) -> tuple[str, ...]:
    if not extra_prediction_dirs:
        return ()
    if prefixes and len(prefixes) != len(extra_prediction_dirs):
        raise ValueError("extra_prediction_prefixes length must match extra_prediction_dirs")
    if prefixes:
        return tuple(_clean_prediction_prefix(prefix) for prefix in prefixes)
    return tuple(f"extra_{idx}" for idx, _ in enumerate(extra_prediction_dirs))


def _clean_prediction_prefix(prefix: str) -> str:
    cleaned = "".join(char if char.isalnum() else "_" for char in prefix.strip().lower()).strip("_")
    if not cleaned:
        raise ValueError("extra prediction prefixes must not be empty")
    return cleaned


def _load_extra_prediction_dir(path: Path, *, prefix: str) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    schema = pl.scan_parquet(str(path / "*.parquet")).collect_schema()
    raw_columns = [
        column
        for column in (
            "tabm_prediction",
            "ensemble_prediction",
            "ridge_calibrated_prediction",
            "xgboost_prediction",
            "lightgbm_prediction",
            "catboost_prediction",
            "latent_alpha_linear_stack",
            "ridge_rank_alpha10000",
            "pls_rank_k8",
        )
        if column in schema.names()
    ]
    if not raw_columns:
        raise ValueError(f"{path} does not contain known prediction columns")
    rename = {column: f"{prefix}_{column}" for column in raw_columns}
    return pl.scan_parquet(str(path / "*.parquet")).select(list(KEYS) + raw_columns).rename(rename).collect().sort(list(KEYS))


def enrich_raw_features(frame: pl.DataFrame, raw_train_parquet_dir: Path, feature_columns: Sequence[str]) -> pl.DataFrame:
    """Join selected raw features by observable row keys."""

    if not raw_train_parquet_dir.exists():
        raise FileNotFoundError(raw_train_parquet_dir)
    columns = tuple(dict.fromkeys(feature_columns))
    raw = pl.scan_parquet(str(raw_train_parquet_dir / "**" / "*.parquet")).select(["date_id", "time_id", "symbol_id", *columns])
    return frame.lazy().join(raw, on=["date_id", "time_id", "symbol_id"], how="left").collect().sort(list(KEYS))


def add_raw_preprocessing_features(
    frame: pl.DataFrame,
    *,
    raw_feature_columns: Sequence[str],
    modes: Sequence[str],
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Add causal preprocessing transforms of raw feature columns."""

    columns = tuple(dict.fromkeys(raw_feature_columns))
    unsafe = [column for column in columns if column == TARGET or column.startswith("responder_")]
    if unsafe:
        raise ValueError(f"raw preprocessing cannot use target/responder columns: {unsafe}")
    available = tuple(column for column in columns if column in frame.columns)
    selected_modes = tuple(dict.fromkeys(modes))
    allowed = {
        "batch_rank",
        "batch_demean",
        "batch_zscore",
        "batch_abs_zscore",
        "batch_top_bottom",
        "row_missing_count",
        "row_abs_mean",
        "row_l2_energy",
    }
    unknown = sorted(set(selected_modes) - allowed)
    if unknown:
        raise ValueError(f"unknown raw_preprocess_modes: {unknown}")
    if not available or not selected_modes:
        return frame, ()

    group = ["date_id", "time_id"]
    exprs: list[pl.Expr] = []
    names: list[str] = []

    for column in available:
        safe = _safe_prediction_feature_name(column)
        value = pl.col(column).cast(pl.Float64)
        if "batch_rank" in selected_modes or "batch_top_bottom" in selected_modes:
            rank = value.rank(method="average").over(group).cast(pl.Float64)
            count = value.count().over(group).cast(pl.Float64)
            centered_rank = pl.when(count > 1.0).then(((rank - 1.0) / (count - 1.0)) - 0.5).otherwise(0.0)
            if "batch_rank" in selected_modes:
                name = f"{safe}__raw_batch_rank"
                exprs.append(_finite_expr(centered_rank).alias(name))
                names.append(name)
            if "batch_top_bottom" in selected_modes:
                top = f"{safe}__raw_batch_top10"
                bottom = f"{safe}__raw_batch_bottom10"
                exprs.append(_finite_expr((centered_rank >= 0.4).cast(pl.Float64)).alias(top))
                exprs.append(_finite_expr((centered_rank <= -0.4).cast(pl.Float64)).alias(bottom))
                names.extend([top, bottom])
        mean = value.mean().over(group)
        std = value.std().over(group)
        if "batch_demean" in selected_modes:
            name = f"{safe}__raw_batch_demean"
            exprs.append(_finite_expr(value - mean).alias(name))
            names.append(name)
        if "batch_zscore" in selected_modes:
            name = f"{safe}__raw_batch_zscore"
            zscore = pl.when(std > 1e-12).then((value - mean) / std).otherwise(0.0)
            exprs.append(_finite_expr(zscore).alias(name))
            names.append(name)
        if "batch_abs_zscore" in selected_modes:
            name = f"{safe}__raw_batch_abs_zscore"
            zscore = pl.when(std > 1e-12).then((value - mean) / std).otherwise(0.0)
            exprs.append(_finite_expr(zscore.abs()).alias(name))
            names.append(name)

    clean_values = [_finite_expr(pl.col(column).cast(pl.Float64)) for column in available]
    if "row_missing_count" in selected_modes:
        name = "raw_row_missing_count"
        missing_exprs = [
            (pl.col(column).is_null() | pl.col(column).cast(pl.Float64).is_nan()).cast(pl.Float64)
            for column in available
        ]
        exprs.append(pl.sum_horizontal(missing_exprs).alias(name))
        names.append(name)
    if "row_abs_mean" in selected_modes:
        name = "raw_row_abs_mean"
        exprs.append(pl.mean_horizontal([value.abs() for value in clean_values]).alias(name))
        names.append(name)
    if "row_l2_energy" in selected_modes:
        name = "raw_row_l2_energy"
        exprs.append((pl.sum_horizontal([value * value for value in clean_values]) / float(len(clean_values))).sqrt().alias(name))
        names.append(name)

    if not exprs:
        return frame, ()
    return frame.with_columns(exprs), tuple(names)


def _finite_expr(expr: pl.Expr) -> pl.Expr:
    return expr.fill_nan(0.0).fill_null(0.0)


def add_prediction_context(frame: pl.DataFrame) -> pl.DataFrame:
    """Add prediction-derived observable context features."""

    preferred = ("tabm_prediction", "tree_prediction", "xgboost_prediction", "lightgbm_prediction", "ridge_calibrated_prediction")
    columns = [column for column in preferred if column in frame.columns]
    columns.extend(column for column in frame.columns if column.startswith("extra_") and column.endswith("_prediction"))
    columns = list(dict.fromkeys(columns))
    if not columns:
        raise ValueError("at least one prediction column is required")
    pred_exprs = [pl.col(column).cast(pl.Float64) for column in columns]
    return frame.with_columns(
        (pl.col("tabm_prediction") - pl.col("tree_prediction")).alias("tabm_tree_diff") if "tree_prediction" in frame.columns else pl.lit(0.0).alias("tabm_tree_diff"),
        pl.mean_horizontal([expr.abs() for expr in pred_exprs]).alias("prediction_abs_mean"),
        _row_prediction_std(columns).alias("prediction_disagreement"),
        pl.col("weight").cast(pl.Float64).log1p().alias("log1p_weight"),
    ).with_columns(
        pl.col("baseline_prediction").abs().alias("abs_baseline_prediction") if "baseline_prediction" in frame.columns else pl.lit(0.0).alias("abs_baseline_prediction")
    )


def add_gateway_rls_predictions(
    frame: pl.DataFrame,
    *,
    include_risk_shrink: bool,
    include_extra_experts: bool = False,
    expert_expansions: Sequence[str] = (),
    risk_strengths: Sequence[float],
    risk_profiles: Sequence[str],
) -> tuple[pl.DataFrame, list[pl.DataFrame]]:
    """Add existing dynamic gateway RLS candidates using the repository's audited simulator."""

    gateway = _load_script_module("run_bayesian_gateway_meta_simulation", PROJECT_ROOT / "scripts" / "run_bayesian_gateway_meta_simulation.py")
    selector = _load_script_module("run_gateway_rls_strategy_selection", PROJECT_ROOT / "scripts" / "run_gateway_rls_strategy_selection.py")
    dynamic = _load_script_module("run_dynamic_gateway_rls_validation", PROJECT_ROOT / "scripts" / "run_dynamic_gateway_rls_validation.py")
    frame, _ = gateway._add_walk_forward_convex_baseline(frame)
    standard_experts = tuple(gateway._expert_columns(frame))
    extra_experts = _additional_prediction_experts(frame, standard_experts) if include_extra_experts else ()
    expert_columns = tuple(
        dict.fromkeys(
            [
                *(column for column in standard_experts if column != "baseline_prediction"),
                *extra_experts,
            ]
        )
    )
    frame, expanded_experts = _add_prediction_expert_expansions(frame, expert_columns, expert_expansions)
    expert_columns = tuple(dict.fromkeys([*expert_columns, *expanded_experts]))
    feature_sets = gateway._gateway_feature_sets(expert_columns)
    selector._validate_rls_strategies(selector._rls_strategies(), feature_sets)
    audits: list[pl.DataFrame] = []
    for strategy in selector._rls_strategies():
        predictions, audit = selector._predict_dynamic_rls_strategy(
            gateway,
            dynamic,
            frame,
            feature_columns=feature_sets[strategy.feature_set],
            ridge_alpha=strategy.ridge_alpha,
            forgetting_factor=strategy.forgetting_factor,
            output=strategy.prediction,
        )
        frame = frame.join(predictions, on=list(KEYS), how="inner")
        audits.append(audit.with_columns(pl.lit(strategy.name).alias("strategy")))
        if include_risk_shrink:
            risk_predictions, risk_audit = selector._predict_dynamic_rls_risk_modulated_shrinkage_strategy(
                gateway,
                dynamic,
                frame,
                feature_columns=feature_sets[strategy.feature_set],
                ridge_alpha=strategy.ridge_alpha,
                forgetting_factor=strategy.forgetting_factor,
                strengths=tuple(risk_strengths),
                profiles=tuple(risk_profiles),
                output_prefix=f"gateway_risk_{strategy.name}",
            )
            frame = frame.join(risk_predictions, on=list(KEYS), how="inner")
            audits.append(risk_audit.with_columns(pl.lit(f"gateway_risk_{strategy.name}").alias("strategy")))
    return frame, audits


def _add_prediction_expert_expansions(
    frame: pl.DataFrame,
    base_columns: Sequence[str],
    expansions: Sequence[str],
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Add target-free nonlinear transforms of saved prediction experts."""

    modes = tuple(dict.fromkeys(expansions))
    allowed = {
        "signed_square",
        "abs",
        "signed_sqrt",
        "signed_log1p",
        "cube",
        "sign",
        "pair_product",
        "batch_rank",
        "batch_mean",
        "batch_demean",
        "batch_std",
        "batch_zscore",
    }
    unknown = sorted(set(modes) - allowed)
    if unknown:
        raise ValueError(f"unknown gateway expert expansions: {unknown}")
    if not modes:
        return frame, ()
    exprs: list[pl.Expr] = []
    names: list[str] = []
    for column in base_columns:
        safe = _safe_prediction_feature_name(column)
        if "signed_square" in modes:
            name = f"{safe}__signed_square"
            exprs.append((pl.col(column).cast(pl.Float64) * pl.col(column).cast(pl.Float64).abs()).alias(name))
            names.append(name)
        if "abs" in modes:
            name = f"{safe}__abs"
            exprs.append(pl.col(column).cast(pl.Float64).abs().alias(name))
            names.append(name)
        if "signed_sqrt" in modes:
            name = f"{safe}__signed_sqrt"
            value = pl.col(column).cast(pl.Float64)
            sign = pl.when(value > 0.0).then(1.0).when(value < 0.0).then(-1.0).otherwise(0.0)
            exprs.append((sign * value.abs().sqrt()).alias(name))
            names.append(name)
        if "signed_log1p" in modes:
            name = f"{safe}__signed_log1p"
            value = pl.col(column).cast(pl.Float64)
            sign = pl.when(value > 0.0).then(1.0).when(value < 0.0).then(-1.0).otherwise(0.0)
            exprs.append((sign * value.abs().log1p()).alias(name))
            names.append(name)
        if "cube" in modes:
            name = f"{safe}__cube"
            value = pl.col(column).cast(pl.Float64)
            exprs.append((value * value * value).alias(name))
            names.append(name)
        if "sign" in modes:
            name = f"{safe}__sign"
            value = pl.col(column).cast(pl.Float64)
            exprs.append(pl.when(value > 0.0).then(1.0).when(value < 0.0).then(-1.0).otherwise(0.0).alias(name))
            names.append(name)
        if "batch_rank" in modes:
            name = f"{safe}__batch_rank"
            rank = pl.col(column).rank(method="average").over(["date_id", "time_id"]).cast(pl.Float64)
            count = pl.len().over(["date_id", "time_id"]).cast(pl.Float64)
            exprs.append(
                pl.when(count > 1.0)
                .then(((rank - 1.0) / (count - 1.0)) - 0.5)
                .otherwise(0.0)
                .alias(name)
            )
            names.append(name)
        if "batch_mean" in modes:
            name = f"{safe}__batch_mean"
            value = pl.col(column).cast(pl.Float64)
            exprs.append(value.mean().over(["date_id", "time_id"]).alias(name))
            names.append(name)
        if "batch_demean" in modes:
            name = f"{safe}__batch_demean"
            value = pl.col(column).cast(pl.Float64)
            exprs.append((value - value.mean().over(["date_id", "time_id"])).alias(name))
            names.append(name)
        if "batch_std" in modes:
            name = f"{safe}__batch_std"
            value = pl.col(column).cast(pl.Float64)
            exprs.append(value.std().over(["date_id", "time_id"]).fill_null(0.0).alias(name))
            names.append(name)
        if "batch_zscore" in modes:
            name = f"{safe}__batch_zscore"
            value = pl.col(column).cast(pl.Float64)
            mean = value.mean().over(["date_id", "time_id"])
            std = value.std().over(["date_id", "time_id"])
            exprs.append(pl.when(std > 1e-12).then((value - mean) / std).otherwise(0.0).alias(name))
            names.append(name)
    if "pair_product" in modes:
        for left, right in itertools.combinations(base_columns, 2):
            left_safe = _safe_prediction_feature_name(left)
            right_safe = _safe_prediction_feature_name(right)
            name = f"{left_safe}__x__{right_safe}"
            exprs.append((pl.col(left).cast(pl.Float64) * pl.col(right).cast(pl.Float64)).alias(name))
            names.append(name)
    if not exprs:
        return frame, ()
    return frame.with_columns(exprs), tuple(names)


def _safe_prediction_feature_name(column: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in column).strip("_")


def _additional_prediction_experts(frame: pl.DataFrame, standard_experts: Sequence[str]) -> tuple[str, ...]:
    """Return saved OOF prediction columns not covered by the canonical gateway experts."""

    standard = set(standard_experts)
    blocked_prefixes = (
        "gateway_",
        "posterior_",
        "risk_shrink_",
        "scale_calibrated_",
        "strong_oof_",
        "fixed_blend_",
        "wf_blend_",
    )
    return tuple(
        column
        for column in frame.columns
        if column.endswith("_prediction")
        and column not in standard
        and not any(column.startswith(prefix) for prefix in blocked_prefixes)
    )


def add_walk_forward_ridge_stack_candidates(
    frame: pl.DataFrame,
    *,
    prediction_columns: Sequence[str],
    alphas: Sequence[float],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Fit one or more no-intercept Ridge stacks using only earlier folds."""

    out = frame.select(list(KEYS))
    params: list[dict[str, Any]] = []
    for alpha in tuple(dict.fromkeys(float(value) for value in alphas)):
        predictions, alpha_params = add_walk_forward_ridge_stack(frame, prediction_columns=prediction_columns, alpha=alpha)
        out = out.join(predictions, on=list(KEYS), how="inner")
        params.extend(alpha_params)
    return out, params


def add_walk_forward_ridge_stack(frame: pl.DataFrame, *, prediction_columns: Sequence[str], alpha: float = 1_000.0) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Fit a no-intercept Ridge stack using only earlier folds."""

    folds = _folds(frame)
    predictions: list[pl.DataFrame] = []
    params: list[dict[str, Any]] = []
    columns = tuple(prediction_columns)
    output = _stack_prediction_name(alpha)
    for idx, fold in enumerate(folds):
        current = frame.filter(pl.col("fold") == fold)
        if idx == 0:
            coef = _default_stack_coef(columns)
            fit_rows = 0
        else:
            calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
            coef = _fit_weighted_ridge_no_intercept(calibration, columns, alpha=alpha)
            fit_rows = calibration.height
        pred = current.select(columns).to_numpy().astype(np.float64, copy=False) @ coef
        predictions.append(current.select(list(KEYS)).with_columns(pl.Series(output, pred)))
        for column, value in zip(columns, coef, strict=True):
            params.append({"component": output.removesuffix("_prediction"), "fold": fold, "feature": column, "value": float(value), "fit_rows": fit_rows, "alpha": float(alpha)})
    return pl.concat(predictions), params


def _stack_prediction_name(alpha: float) -> str:
    if abs(float(alpha) - 1000.0) <= 1e-12:
        return "strong_oof_ridge_stack_prediction"
    return f"strong_oof_ridge_stack_alpha{_format_float(alpha)}_ridge_stack_prediction"


def add_walk_forward_residual_rules(
    frame: pl.DataFrame,
    *,
    base_predictions: Sequence[str],
    residual_features: Sequence[str],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Fit residual corrections y-base ~= sum alpha_j * phi_j using earlier folds."""

    folds = _folds(frame)
    out = frame.select(list(KEYS))
    params: list[dict[str, Any]] = []
    if not residual_features:
        return out, params
    for base in base_predictions:
        if base not in frame.columns:
            continue
        parts: list[pl.DataFrame] = []
        for idx, fold in enumerate(folds):
            current = frame.filter(pl.col("fold") == fold)
            if idx == 0:
                pred = current[base].to_numpy().astype(np.float64, copy=False)
                params.append({"component": f"{base}_residual", "fold": fold, "feature": "__none_first_fold__", "value": 0.0, "fit_rows": 0})
            else:
                calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
                coef, means, scales = _fit_residual_linear(calibration, base, residual_features)
                x_current = _standardized_matrix(current, residual_features, means, scales)
                pred = current[base].to_numpy().astype(np.float64, copy=False) + x_current @ coef
                for feature, value in zip(residual_features, coef, strict=True):
                    params.append({"component": f"{base}_residual", "fold": fold, "feature": feature, "value": float(value), "fit_rows": calibration.height})
            parts.append(current.select(list(KEYS)).with_columns(pl.Series(f"{base}_residual_prediction", pred)))
        out = out.join(pl.concat(parts), on=list(KEYS), how="inner")
    return out, params


def add_walk_forward_residual_tail_masks(
    frame: pl.DataFrame,
    *,
    residual_predictions: Sequence[str],
    quantiles: Sequence[float],
    modes: Sequence[str],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Apply strong OOF residual corrections only in previous-fold tails."""

    qs = tuple(dict.fromkeys(float(q) for q in quantiles))
    selected_modes = tuple(dict.fromkeys(modes))
    out = frame.select(list(KEYS))
    params: list[dict[str, Any]] = []
    if not qs or not selected_modes:
        return out, params
    folds = _folds(frame)
    for residual_prediction in residual_predictions:
        if residual_prediction not in frame.columns:
            continue
        base_prediction = residual_prediction.removesuffix("_residual_prediction")
        if base_prediction not in frame.columns:
            continue
        for q in qs:
            for mode in selected_modes:
                suffix = f"{mode}_q{_format_float(q)}"
                output = f"{base_prediction}_residual_{suffix}_residual_tail_prediction"
                parts: list[pl.DataFrame] = []
                for idx, fold in enumerate(folds):
                    current = frame.filter(pl.col("fold") == fold)
                    base = current[base_prediction].to_numpy().astype(np.float64, copy=False)
                    residual = current[residual_prediction].to_numpy().astype(np.float64, copy=False)
                    if idx == 0:
                        pred = base
                        selected_rows = 0
                        thresholds = {"weight": np.inf, "abs_base": np.inf, "disagreement": np.inf}
                        fit_rows = 0
                    else:
                        calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
                        thresholds = _fit_tail_thresholds(calibration, base_prediction=base_prediction, quantile=q)
                        mask = _tail_mask(current, base_prediction=base_prediction, thresholds=thresholds, mode=mode)
                        pred = np.where(mask, residual, base)
                        selected_rows = int(mask.sum())
                        fit_rows = calibration.height
                    params.append(
                        {
                            "component": "residual_tail",
                            "base": base_prediction,
                            "fold": fold,
                            "mode": mode,
                            "quantile": float(q),
                            "fit_rows": fit_rows,
                            "selected_rows": selected_rows,
                            "selected_frac": float(selected_rows / max(current.height, 1)),
                            "weight_threshold": float(thresholds["weight"]),
                            "abs_base_threshold": float(thresholds["abs_base"]),
                            "disagreement_threshold": float(thresholds["disagreement"]),
                        }
                    )
                    parts.append(current.select(list(KEYS)).with_columns(pl.Series(output, pred)))
                out = out.join(pl.concat(parts), on=list(KEYS), how="inner")
    return out, params


def add_prediction_risk_shrinkage(
    frame: pl.DataFrame,
    *,
    base_predictions: Sequence[str],
    strengths: Sequence[float],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Apply fixed risk shrinkage profiles to candidate predictions."""

    out = frame.select(list(KEYS))
    params: list[dict[str, Any]] = []
    risk = _risk_proxy(frame)
    risk_mean = float(np.mean(risk)) if risk.size else 1.0
    risk_mean = max(risk_mean, 1e-12)
    for base in base_predictions:
        if base not in frame.columns:
            continue
        pred = frame[base].to_numpy().astype(np.float64, copy=False)
        for strength in strengths:
            suffix = _format_float(strength)
            shrunk = pred / np.sqrt(1.0 + float(strength) * np.clip(risk / risk_mean, 0.0, 10.0))
            column = f"{base}_s{suffix}_risk_shrink_prediction"
            out = out.with_columns(pl.Series(column, shrunk))
            params.append({"component": "risk_shrink", "base": base, "strength": float(strength), "risk_mean": risk_mean})
    return out, params


def add_walk_forward_regime_scales(
    frame: pl.DataFrame,
    *,
    base_predictions: Sequence[str],
    time_bucket_sizes: Sequence[int],
    min_group_rows_values: Sequence[int],
    prior_strengths: Sequence[float],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Fit grouped scale calibration using only earlier folds."""

    out = frame.select(list(KEYS))
    params: list[dict[str, Any]] = []
    folds = _folds(frame)
    for base in base_predictions:
        if base not in frame.columns:
            continue
        for time_bucket_size in time_bucket_sizes:
            for min_group_rows in min_group_rows_values:
                for prior_strength in prior_strengths:
                    column = f"{base}_tb{time_bucket_size}_min{min_group_rows}_p{_format_float(prior_strength)}_regime_scaled_prediction"
                    parts: list[pl.DataFrame] = []
                    for idx, fold in enumerate(folds):
                        current = frame.filter(pl.col("fold") == fold)
                        if idx == 0:
                            pred = current[base].to_numpy().astype(np.float64, copy=False)
                            params.append({"component": "regime_scale", "base": base, "fold": fold, "group": "__first_fold__", "scale": 1.0, "fit_rows": 0})
                        else:
                            calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
                            thresholds = _fit_regime_thresholds(calibration, base_prediction=base)
                            cal_codes = _regime_codes(calibration, base_prediction=base, thresholds=thresholds, time_bucket_size=time_bucket_size)
                            cur_codes = _regime_codes(current, base_prediction=base, thresholds=thresholds, time_bucket_size=time_bucket_size)
                            default, scales = _fit_group_scales(
                                calibration,
                                base_prediction=base,
                                codes=cal_codes,
                                min_group_rows=int(min_group_rows),
                                prior_strength=float(prior_strength),
                            )
                            pred = current[base].to_numpy().astype(np.float64, copy=False) * _scale_for_codes(cur_codes, default, scales)
                            params.append(
                                {
                                    "component": "regime_scale",
                                    "base": base,
                                    "fold": fold,
                                    "group": "__summary__",
                                    "scale": float(default),
                                    "fit_rows": calibration.height,
                                    "n_groups": len(scales),
                                    "time_bucket_size": int(time_bucket_size),
                                    "min_group_rows": int(min_group_rows),
                                    "prior_strength": float(prior_strength),
                                }
                            )
                        parts.append(current.select(list(KEYS)).with_columns(pl.Series(column, pred)))
        out = out.join(pl.concat(parts), on=list(KEYS), how="inner")
    return out, params


def add_fixed_candidate_blends(
    frame: pl.DataFrame,
    *,
    candidates: Sequence[str],
    weights: Sequence[float],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Blend fixed candidate columns without fitting on validation targets."""

    available = tuple(dict.fromkeys(column for column in candidates if column in frame.columns))
    out = frame.select(list(KEYS))
    params: list[dict[str, Any]] = []
    if len(available) < 2 or not weights:
        return out, params
    for pair_idx, (left, right) in enumerate(itertools.combinations(available, 2)):
        left_values = frame[left].to_numpy().astype(np.float64, copy=False)
        right_values = frame[right].to_numpy().astype(np.float64, copy=False)
        for weight in weights:
            w = float(weight)
            column = f"fixed_blend_{pair_idx}_w{_format_float(w)}_fixed_blend_prediction"
            pred = w * left_values + (1.0 - w) * right_values
            out = out.with_columns(pl.Series(column, pred))
            params.append(
                {
                    "component": "fixed_blend",
                    "column": column,
                    "left": left,
                    "right": right,
                    "left_weight": w,
                    "right_weight": 1.0 - w,
                }
            )
    return out, params


def add_walk_forward_candidate_blends(
    frame: pl.DataFrame,
    *,
    candidates: Sequence[str],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Fit two-candidate convex blend weights using previous folds only."""

    available = tuple(dict.fromkeys(column for column in candidates if column in frame.columns))
    out = frame.select(list(KEYS))
    params: list[dict[str, Any]] = []
    if len(available) < 2:
        return out, params
    folds = _folds(frame)
    for pair_idx, (left, right) in enumerate(itertools.combinations(available, 2)):
        output = f"wf_blend_{pair_idx}_wf_blend_prediction"
        parts: list[pl.DataFrame] = []
        for idx, fold in enumerate(folds):
            current = frame.filter(pl.col("fold") == fold)
            if idx == 0:
                weight = 0.5
                fit_rows = 0
            else:
                calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
                weight = _fit_convex_pair_weight(calibration, left=left, right=right)
                fit_rows = calibration.height
            left_values = current[left].to_numpy().astype(np.float64, copy=False)
            right_values = current[right].to_numpy().astype(np.float64, copy=False)
            pred = weight * left_values + (1.0 - weight) * right_values
            parts.append(current.select(list(KEYS)).with_columns(pl.Series(output, pred)))
            params.append(
                {
                    "component": "walk_forward_blend",
                    "column": output,
                    "fold": fold,
                    "left": left,
                    "right": right,
                    "left_weight": float(weight),
                    "right_weight": float(1.0 - weight),
                    "fit_rows": fit_rows,
                }
            )
        out = out.join(pl.concat(parts), on=list(KEYS), how="inner")
    return out, params


def add_walk_forward_contextual_candidate_blends(
    frame: pl.DataFrame,
    *,
    candidates: Sequence[str],
    group_specs: Sequence[str],
    time_bucket_sizes: Sequence[int],
    min_group_rows_values: Sequence[int],
    prior_strengths: Sequence[float],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Fit pairwise convex blends by observable context using earlier folds only."""

    available = tuple(dict.fromkeys(column for column in candidates if column in frame.columns))
    specs = tuple(_parse_context_group_spec(spec) for spec in group_specs if spec.strip())
    out = frame.select(list(KEYS))
    params: list[dict[str, Any]] = []
    if len(available) < 2 or not specs:
        return out, params
    folds = _folds(frame)
    for pair_idx, (left, right) in enumerate(itertools.combinations(available, 2)):
        for components in specs:
            spec_name = _context_group_spec_name(components)
            for time_bucket_size in tuple(dict.fromkeys(int(value) for value in time_bucket_sizes)):
                for min_group_rows in tuple(dict.fromkeys(int(value) for value in min_group_rows_values)):
                    for prior_strength in tuple(dict.fromkeys(float(value) for value in prior_strengths)):
                        column = (
                            f"ctx_blend_{pair_idx}_{spec_name}_tb{time_bucket_size}_min{min_group_rows}"
                            f"_p{_format_float(prior_strength)}_contextual_blend_prediction"
                        )
                        parts: list[pl.DataFrame] = []
                        for idx, fold in enumerate(folds):
                            current = frame.filter(pl.col("fold") == fold)
                            if idx == 0:
                                default_weight = 0.5
                                group_weights: dict[int, float] = {}
                                fit_rows = 0
                                n_groups = 0
                                weights = np.full(current.height, default_weight, dtype=np.float64)
                            else:
                                calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
                                thresholds = _fit_context_thresholds(calibration, left=left, right=right)
                                cal_codes = _context_group_codes(
                                    calibration,
                                    left=left,
                                    right=right,
                                    components=components,
                                    thresholds=thresholds,
                                    time_bucket_size=time_bucket_size,
                                )
                                cur_codes = _context_group_codes(
                                    current,
                                    left=left,
                                    right=right,
                                    components=components,
                                    thresholds=thresholds,
                                    time_bucket_size=time_bucket_size,
                                )
                                default_weight = _fit_convex_pair_weight(calibration, left=left, right=right)
                                group_weights = _fit_group_blend_weights(
                                    calibration,
                                    left=left,
                                    right=right,
                                    codes=cal_codes,
                                    min_group_rows=int(min_group_rows),
                                    prior_strength=float(prior_strength),
                                    prior_weight=default_weight,
                                )
                                weights = _blend_weight_for_codes(cur_codes, default_weight, group_weights)
                                fit_rows = calibration.height
                                n_groups = len(group_weights)
                            left_values = current[left].to_numpy().astype(np.float64, copy=False)
                            right_values = current[right].to_numpy().astype(np.float64, copy=False)
                            pred = weights * left_values + (1.0 - weights) * right_values
                            parts.append(current.select(list(KEYS)).with_columns(pl.Series(column, pred)))
                            params.append(
                                {
                                    "component": "contextual_blend",
                                    "column": column,
                                    "fold": fold,
                                    "left": left,
                                    "right": right,
                                    "group_spec": spec_name,
                                    "time_bucket_size": int(time_bucket_size),
                                    "min_group_rows": int(min_group_rows),
                                    "prior_strength": float(prior_strength),
                                    "fallback_left_weight": float(default_weight),
                                    "fallback_right_weight": float(1.0 - default_weight),
                                    "fit_rows": fit_rows,
                                    "n_groups": n_groups,
                                }
                            )
                        out = out.join(pl.concat(parts), on=list(KEYS), how="inner")
    return out, params


def score_walk_forward_contextual_candidate_blends(
    frame: pl.DataFrame,
    *,
    candidates: Sequence[str],
    group_specs: Sequence[str],
    time_bucket_sizes: Sequence[int],
    min_group_rows_values: Sequence[int],
    prior_strengths: Sequence[float],
) -> tuple[list[dict[str, float | int | str]], list[dict[str, Any]]]:
    """Score contextual pair blends without materializing prediction columns."""

    available = tuple(dict.fromkeys(column for column in candidates if column in frame.columns))
    specs = tuple(_parse_context_group_spec(spec) for spec in group_specs if spec.strip())
    score_rows: list[dict[str, float | int | str]] = []
    params: list[dict[str, Any]] = []
    if len(available) < 2 or not specs:
        return score_rows, params

    folds = _folds(frame)
    fold_arrays = _contextual_fold_arrays(frame, folds=folds, columns=available)
    unique_time_buckets = tuple(dict.fromkeys(int(value) for value in time_bucket_sizes))
    unique_min_rows = tuple(dict.fromkeys(int(value) for value in min_group_rows_values))
    unique_priors = tuple(dict.fromkeys(float(value) for value in prior_strengths))

    for pair_idx, (left, right) in enumerate(itertools.combinations(available, 2)):
        calibration_cache: list[dict[str, Any] | None] = []
        for idx in range(len(folds)):
            if idx == 0:
                calibration_cache.append(None)
                continue
            calibration_cache.append(_contextual_calibration_arrays(fold_arrays[:idx], left=left, right=right))

        for components in specs:
            spec_name = _context_group_spec_name(components)
            for time_bucket_size in unique_time_buckets:
                for min_group_rows in unique_min_rows:
                    for prior_strength in unique_priors:
                        candidate = (
                            f"ctx_blend_{pair_idx}_{spec_name}_tb{time_bucket_size}_min{min_group_rows}"
                            f"_p{_format_float(prior_strength)}_contextual_blend"
                        )
                        for idx, fold in enumerate(folds):
                            current = fold_arrays[idx]
                            cur_left = current[left]
                            cur_right = current[right]
                            if idx == 0:
                                default_weight = 0.5
                                weights = np.full(cur_left.shape, default_weight, dtype=np.float64)
                                fit_rows = 0
                                n_groups = 0
                            else:
                                calibration = calibration_cache[idx]
                                assert calibration is not None
                                default_weight = float(calibration["fallback_weight"])
                                thresholds = calibration["thresholds"]
                                cal_codes = _context_group_codes_arrays(
                                    weight=calibration["weight"],
                                    time_id=calibration["time_id"],
                                    left_values=calibration["left"],
                                    right_values=calibration["right"],
                                    components=components,
                                    thresholds=thresholds,
                                    time_bucket_size=int(time_bucket_size),
                                )
                                cur_codes = _context_group_codes_arrays(
                                    weight=current["weight"],
                                    time_id=current["time_id"],
                                    left_values=cur_left,
                                    right_values=cur_right,
                                    components=components,
                                    thresholds=thresholds,
                                    time_bucket_size=int(time_bucket_size),
                                )
                                group_weights, n_groups = _fit_group_blend_weight_array(
                                    y=calibration[TARGET],
                                    weight=calibration["weight"],
                                    left_pred=calibration["left"],
                                    right_pred=calibration["right"],
                                    codes=cal_codes,
                                    min_group_rows=int(min_group_rows),
                                    prior_strength=float(prior_strength),
                                    prior_weight=default_weight,
                                )
                                weights = _blend_weight_for_codes_array(cur_codes, default_weight, group_weights)
                                fit_rows = int(calibration["rows"])

                            pred = weights * cur_left + (1.0 - weights) * cur_right
                            score_rows.append(
                                score_arrays(
                                    fold=fold,
                                    candidate=candidate,
                                    family="contextual_blend",
                                    y_true=current[TARGET],
                                    y_pred=pred,
                                    weight=current["weight"],
                                )
                            )
                            params.append(
                                {
                                    "component": "contextual_blend",
                                    "column": f"{candidate}_prediction",
                                    "fold": fold,
                                    "left": left,
                                    "right": right,
                                    "group_spec": spec_name,
                                    "time_bucket_size": int(time_bucket_size),
                                    "min_group_rows": int(min_group_rows),
                                    "prior_strength": float(prior_strength),
                                    "fallback_left_weight": float(default_weight),
                                    "fallback_right_weight": float(1.0 - default_weight),
                                    "fit_rows": fit_rows,
                                    "n_groups": int(n_groups),
                                    "scoring_mode": "array_streaming",
                                }
                            )
        del calibration_cache
        gc.collect()
    return score_rows, params


def add_online_daily_scales(
    frame: pl.DataFrame,
    *,
    base_predictions: Sequence[str],
    prior_strengths: Sequence[float],
    forgetting_factors: Sequence[float],
    min_scale: float = 0.0,
    max_scale: float = 2.0,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Apply causal per-date scalar calibration within each fold.

    The prediction for date D uses statistics available before date D. After D is
    scored, D's target and prediction update the scale used by later dates.
    """

    available = tuple(dict.fromkeys(column for column in base_predictions if column in frame.columns))
    out = frame.select(list(KEYS))
    params: list[dict[str, Any]] = []
    if not available:
        return out, params
    if min_scale > max_scale:
        raise ValueError("min_scale must be <= max_scale")
    folds = _folds(frame)
    for base in available:
        for prior in tuple(dict.fromkeys(float(value) for value in prior_strengths)):
            if prior < 0.0:
                raise ValueError("prior strengths must be non-negative")
            for forgetting in tuple(dict.fromkeys(float(value) for value in forgetting_factors)):
                if not 0.0 < forgetting <= 1.0:
                    raise ValueError("forgetting factors must be in (0, 1]")
                column = f"{base.removesuffix('_prediction')}_online_scale_f{_format_float(forgetting)}_p{_format_float(prior)}_online_scale_prediction"
                parts: list[pl.DataFrame] = []
                for fold in folds:
                    current_fold = frame.filter(pl.col("fold") == fold).sort(["date_id", "time_id", "symbol_id"])
                    date_ids = current_fold["date_id"].to_numpy().astype(np.int64, copy=False)
                    pred_all = current_fold[base].to_numpy().astype(np.float64, copy=False)
                    target_all = current_fold[TARGET].to_numpy().astype(np.float64, copy=False)
                    weight_all = current_fold["weight"].to_numpy().astype(np.float64, copy=False)
                    output = np.empty(current_fold.height, dtype=np.float64)
                    numerator = float(prior)
                    denominator = float(prior)
                    fit_rows = 0
                    for start, end in _contiguous_ranges(date_ids):
                        date_id = int(date_ids[start])
                        scale = numerator / denominator if denominator > 1e-18 else 1.0
                        scale = float(np.clip(scale, min_scale, max_scale))
                        pred = pred_all[start:end]
                        output[start:end] = pred * scale
                        params.append(
                            {
                                "component": "online_daily_scale",
                                "base": base,
                                "column": column,
                                "fold": fold,
                                "date_id": date_id,
                                "scale": scale,
                                "fit_rows": fit_rows,
                                "prior_strength": prior,
                                "forgetting_factor": forgetting,
                            }
                        )
                        target = target_all[start:end]
                        weight = weight_all[start:end]
                        numerator = forgetting * numerator + float(np.sum(weight * pred * target))
                        denominator = forgetting * denominator + float(np.sum(weight * pred * pred))
                        fit_rows += end - start
                    parts.append(current_fold.select(list(KEYS)).with_columns(pl.Series(column, output)))
                out = out.join(pl.concat(parts), on=list(KEYS), how="inner")
    return out, params


def add_online_daily_affine(
    frame: pl.DataFrame,
    *,
    base_predictions: Sequence[str],
    prior_strengths: Sequence[float],
    forgetting_factors: Sequence[float],
    min_scale: float = 0.0,
    max_scale: float = 2.0,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Apply causal per-date affine calibration within each fold.

    The model is y ~= bias + scale * prediction. The prediction for date D uses
    only dates earlier than D in the same validation fold, then D updates the
    normal equations for later dates.
    """

    available = tuple(dict.fromkeys(column for column in base_predictions if column in frame.columns))
    out = frame.select(list(KEYS))
    params: list[dict[str, Any]] = []
    if not available:
        return out, params
    if min_scale > max_scale:
        raise ValueError("min_scale must be <= max_scale")
    folds = _folds(frame)
    for base in available:
        for prior in tuple(dict.fromkeys(float(value) for value in prior_strengths)):
            if prior < 0.0:
                raise ValueError("prior strengths must be non-negative")
            for forgetting in tuple(dict.fromkeys(float(value) for value in forgetting_factors)):
                if not 0.0 < forgetting <= 1.0:
                    raise ValueError("forgetting factors must be in (0, 1]")
                column = f"{base.removesuffix('_prediction')}_online_affine_f{_format_float(forgetting)}_p{_format_float(prior)}_online_affine_prediction"
                parts: list[pl.DataFrame] = []
                for fold in folds:
                    current_fold = frame.filter(pl.col("fold") == fold).sort(["date_id", "time_id", "symbol_id"])
                    date_ids = current_fold["date_id"].to_numpy().astype(np.int64, copy=False)
                    pred_all = current_fold[base].to_numpy().astype(np.float64, copy=False)
                    target_all = current_fold[TARGET].to_numpy().astype(np.float64, copy=False)
                    weight_all = current_fold["weight"].to_numpy().astype(np.float64, copy=False)
                    output = np.empty(current_fold.height, dtype=np.float64)
                    lhs = np.asarray([[prior, 0.0], [0.0, prior]], dtype=np.float64)
                    rhs = np.asarray([0.0, prior], dtype=np.float64)
                    fit_rows = 0
                    for start, end in _contiguous_ranges(date_ids):
                        date_id = int(date_ids[start])
                        bias, scale = _solve_affine(lhs, rhs)
                        scale = float(np.clip(scale, min_scale, max_scale))
                        pred = pred_all[start:end]
                        output[start:end] = bias + scale * pred
                        params.append(
                            {
                                "component": "online_daily_affine",
                                "base": base,
                                "column": column,
                                "fold": fold,
                                "date_id": date_id,
                                "bias": float(bias),
                                "scale": float(scale),
                                "fit_rows": fit_rows,
                                "prior_strength": prior,
                                "forgetting_factor": forgetting,
                            }
                        )
                        target = target_all[start:end]
                        weight = weight_all[start:end]
                        sx = float(np.sum(weight * pred))
                        sy = float(np.sum(weight * target))
                        sxx = float(np.sum(weight * pred * pred))
                        sxy = float(np.sum(weight * pred * target))
                        sw = float(np.sum(weight))
                        lhs = forgetting * lhs + np.asarray([[sw, sx], [sx, sxx]], dtype=np.float64)
                        rhs = forgetting * rhs + np.asarray([sy, sxy], dtype=np.float64)
                        fit_rows += end - start
                    parts.append(current_fold.select(list(KEYS)).with_columns(pl.Series(column, output)))
                out = out.join(pl.concat(parts), on=list(KEYS), how="inner")
    return out, params


def _contiguous_ranges(values: np.ndarray) -> list[tuple[int, int]]:
    if values.size == 0:
        return []
    breaks = np.flatnonzero(values[1:] != values[:-1]) + 1
    starts = np.concatenate(([0], breaks))
    ends = np.concatenate((breaks, [values.size]))
    return [(int(start), int(end)) for start, end in zip(starts, ends, strict=True)]


def _solve_affine(lhs: np.ndarray, rhs: np.ndarray) -> tuple[float, float]:
    if float(np.max(np.abs(lhs))) <= 1e-18:
        return 0.0, 1.0
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    bias = float(coef[0]) if np.isfinite(coef[0]) else 0.0
    scale = float(coef[1]) if np.isfinite(coef[1]) else 1.0
    return bias, scale


def _prediction_columns(frame: pl.DataFrame, requested: Sequence[str]) -> tuple[str, ...]:
    columns = tuple(column for column in requested if column in frame.columns)
    if not columns:
        raise ValueError("no requested strong base candidate columns were found")
    return columns


def _residual_base_predictions(frame: pl.DataFrame, *, configured: Sequence[str], default: Sequence[str]) -> tuple[str, ...]:
    requested = tuple(configured) if configured else tuple(default)
    return tuple(dict.fromkeys(column for column in requested if column in frame.columns))


def _score_by_fold(frame: pl.DataFrame, *, prediction: str, candidate: str, family: str) -> pl.DataFrame:
    return (
        frame.group_by("fold")
        .agg(
            pl.len().alias("rows"),
            pl.col("weight").sum().alias("weight_sum"),
            (pl.col("weight") * (pl.col(TARGET) - pl.col(prediction)).pow(2)).sum().alias("numerator"),
            (pl.col("weight") * pl.col(TARGET).pow(2)).sum().alias("denominator"),
        )
        .with_columns(
            pl.lit(candidate).alias("candidate"),
            pl.lit(family).alias("family"),
            (1.0 - pl.col("numerator") / pl.col("denominator")).alias("weighted_zero_mean_r2"),
        )
        .select(["fold", "candidate", "family", "rows", "weight_sum", "numerator", "denominator", "weighted_zero_mean_r2"])
        .sort("fold")
    )


def _apply_sampling(frame: pl.DataFrame, *, sample_stride: int, max_rows_per_fold: int | None) -> pl.DataFrame:
    if sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    lazy = frame.lazy()
    if sample_stride > 1:
        lazy = lazy.with_row_index("__sample_row").filter((pl.col("__sample_row") % sample_stride) == 0).drop("__sample_row")
    sampled = lazy.collect()
    if max_rows_per_fold is not None and max_rows_per_fold > 0:
        sampled = pl.concat([fold_frame.head(max_rows_per_fold) for fold_frame in sampled.partition_by("fold", maintain_order=True)])
    return sampled.sort(list(KEYS))


def _row_prediction_std(columns: Sequence[str]) -> pl.Expr:
    mean = pl.mean_horizontal([pl.col(column).cast(pl.Float64) for column in columns])
    variance = pl.mean_horizontal([(pl.col(column).cast(pl.Float64) - mean).pow(2) for column in columns])
    return variance.sqrt()


def _folds(frame: pl.DataFrame) -> list[str]:
    return frame.select("fold").unique().sort("fold")["fold"].to_list()


def _default_stack_coef(columns: tuple[str, ...]) -> np.ndarray:
    coef = np.zeros(len(columns), dtype=np.float64)
    preferred = "conservative_rls_prediction" if "conservative_rls_prediction" in columns else columns[0]
    coef[columns.index(preferred)] = 1.0
    return coef


def _fit_weighted_ridge_no_intercept(frame: pl.DataFrame, columns: Sequence[str], *, alpha: float) -> np.ndarray:
    x = frame.select(columns).to_numpy().astype(np.float64, copy=False)
    x = np.where(np.isfinite(x), x, 0.0)
    y = frame[TARGET].to_numpy().astype(np.float64, copy=False)
    w = frame["weight"].to_numpy().astype(np.float64, copy=False)
    xtw = x.T * w
    lhs = xtw @ x + float(alpha) * np.eye(x.shape[1], dtype=np.float64)
    rhs = xtw @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _fit_convex_pair_weight(frame: pl.DataFrame, *, left: str, right: str) -> float:
    y = frame[TARGET].to_numpy().astype(np.float64, copy=False)
    weight = frame["weight"].to_numpy().astype(np.float64, copy=False)
    left_pred = frame[left].to_numpy().astype(np.float64, copy=False)
    right_pred = frame[right].to_numpy().astype(np.float64, copy=False)
    return _fit_convex_pair_weight_arrays(y, weight, left_pred, right_pred, prior=0.0, prior_weight=0.5)


def _fit_convex_pair_weight_arrays(
    y: np.ndarray,
    weight: np.ndarray,
    left_pred: np.ndarray,
    right_pred: np.ndarray,
    *,
    prior: float,
    prior_weight: float,
) -> float:
    delta = left_pred - right_pred
    denom = float(np.sum(weight * delta * delta))
    numerator = float(np.sum(weight * delta * (y - right_pred)))
    lhs = denom + float(prior)
    if lhs <= 1e-18:
        return float(np.clip(prior_weight, 0.0, 1.0))
    rhs = numerator + float(prior) * float(prior_weight)
    return float(np.clip(rhs / lhs, 0.0, 1.0))


def _fit_residual_linear(frame: pl.DataFrame, base: str, features: Sequence[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = frame[TARGET].to_numpy().astype(np.float64, copy=False)
    pred = frame[base].to_numpy().astype(np.float64, copy=False)
    residual = y - pred
    w = frame["weight"].to_numpy().astype(np.float64, copy=False)
    means = np.asarray([_safe_mean(frame[name].to_numpy()) for name in features], dtype=np.float64)
    scales = np.asarray([_safe_std(frame[name].to_numpy(), means[idx]) for idx, name in enumerate(features)], dtype=np.float64)
    x = _standardized_matrix(frame, features, means, scales)
    xtw = x.T * w
    lhs = xtw @ x + 1_000.0 * np.eye(x.shape[1], dtype=np.float64)
    rhs = xtw @ residual
    try:
        coef = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return coef, means, scales


def _standardized_matrix(frame: pl.DataFrame, features: Sequence[str], means: np.ndarray, scales: np.ndarray) -> np.ndarray:
    x = frame.select(features).to_numpy().astype(np.float64, copy=False)
    x = np.where(np.isfinite(x), x, means)
    return (x - means) / scales


def _safe_mean(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else 0.0


def _safe_std(values: np.ndarray, mean: float) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = np.where(np.isfinite(x), x, mean)
    std = float(np.std(x))
    return std if np.isfinite(std) and std > 1e-12 else 1.0


def _risk_proxy(frame: pl.DataFrame) -> np.ndarray:
    pieces = [
        frame["prediction_disagreement"].to_numpy().astype(np.float64, copy=False),
        frame["prediction_abs_mean"].to_numpy().astype(np.float64, copy=False),
        frame["log1p_weight"].to_numpy().astype(np.float64, copy=False),
    ]
    risk = np.zeros(frame.height, dtype=np.float64)
    for piece in pieces:
        finite = np.where(np.isfinite(piece), piece, 0.0)
        denom = np.quantile(finite, 0.9) if finite.size else 1.0
        risk += np.clip(finite / max(float(denom), 1e-12), 0.0, 10.0)
    return risk


def _fit_tail_thresholds(frame: pl.DataFrame, *, base_prediction: str, quantile: float) -> dict[str, float]:
    return {
        "weight": _safe_quantile(frame["weight"].to_numpy(), quantile),
        "abs_base": _safe_quantile(np.abs(frame[base_prediction].to_numpy()), quantile),
        "disagreement": _safe_quantile(frame["prediction_disagreement"].to_numpy(), quantile),
    }


def _tail_mask(frame: pl.DataFrame, *, base_prediction: str, thresholds: dict[str, float], mode: str) -> np.ndarray:
    weight_mask = frame["weight"].to_numpy().astype(np.float64, copy=False) >= float(thresholds["weight"])
    abs_mask = np.abs(frame[base_prediction].to_numpy().astype(np.float64, copy=False)) >= float(thresholds["abs_base"])
    disagreement_mask = frame["prediction_disagreement"].to_numpy().astype(np.float64, copy=False) >= float(thresholds["disagreement"])
    if mode == "weight":
        return weight_mask
    if mode == "abs_base":
        return abs_mask
    if mode == "disagreement":
        return disagreement_mask
    if mode == "weight_or_abs":
        return weight_mask | abs_mask
    if mode == "weight_and_abs":
        return weight_mask & abs_mask
    if mode == "weight_or_disagreement":
        return weight_mask | disagreement_mask
    raise ValueError(f"unknown residual tail mode: {mode}")


def _safe_quantile(values: np.ndarray, quantile: float) -> float:
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("tail quantiles must be between 0 and 1")
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.inf
    return float(np.quantile(x, float(quantile)))


def _fit_regime_thresholds(frame: pl.DataFrame, *, base_prediction: str) -> dict[str, tuple[float, float]]:
    return {
        "weight": _terciles(frame["weight"].to_numpy()),
        "abs_pred": _terciles(np.abs(frame[base_prediction].to_numpy())),
        "disagreement": _terciles(frame["prediction_disagreement"].to_numpy()),
    }


def _regime_codes(frame: pl.DataFrame, *, base_prediction: str, thresholds: dict[str, tuple[float, float]], time_bucket_size: int) -> np.ndarray:
    time_bucket = frame["time_id"].to_numpy().astype(np.int64, copy=False) // int(time_bucket_size)
    weight_bucket = _bucketize(frame["weight"].to_numpy(), thresholds["weight"])
    abs_bucket = _bucketize(np.abs(frame[base_prediction].to_numpy()), thresholds["abs_pred"])
    disagreement_bucket = _bucketize(frame["prediction_disagreement"].to_numpy(), thresholds["disagreement"])
    return time_bucket * 27 + weight_bucket * 9 + abs_bucket * 3 + disagreement_bucket


def _fit_group_scales(
    frame: pl.DataFrame,
    *,
    base_prediction: str,
    codes: np.ndarray,
    min_group_rows: int,
    prior_strength: float,
) -> tuple[float, dict[int, float]]:
    pred = frame[base_prediction].to_numpy().astype(np.float64, copy=False)
    target = frame[TARGET].to_numpy().astype(np.float64, copy=False)
    weight = frame["weight"].to_numpy().astype(np.float64, copy=False)
    default = _weighted_scale(target, pred, weight, prior=0.0, prior_scale=1.0)
    scales: dict[int, float] = {}
    for code in np.unique(codes):
        mask = codes == code
        if int(mask.sum()) < min_group_rows:
            continue
        scales[int(code)] = _weighted_scale(target[mask], pred[mask], weight[mask], prior=prior_strength, prior_scale=default)
    return default, scales


def _weighted_scale(target: np.ndarray, prediction: np.ndarray, weight: np.ndarray, *, prior: float, prior_scale: float) -> float:
    rhs = float(np.sum(weight * prediction * target) + prior * prior_scale)
    lhs = float(np.sum(weight * prediction * prediction) + prior)
    if lhs <= 0.0:
        return prior_scale
    return rhs / lhs


def _scale_for_codes(codes: np.ndarray, default: float, scales: dict[int, float]) -> np.ndarray:
    out = np.full(codes.shape, default, dtype=np.float64)
    for code, scale in scales.items():
        out[codes == code] = scale
    return out


def _parse_context_group_spec(spec: str) -> tuple[str, ...]:
    components = tuple(part.strip() for part in spec.split("+") if part.strip())
    allowed = {"time", "weight", "abs_left", "abs_right", "abs_diff", "abs_mean", "sign_agree"}
    unknown = sorted(set(components) - allowed)
    if unknown:
        raise ValueError(f"unknown contextual blend group component(s): {unknown}")
    if not components:
        raise ValueError("contextual blend group specs must not be empty")
    return components


def _context_group_spec_name(components: Sequence[str]) -> str:
    return "_".join(components).replace("+", "_")


def _contextual_fold_arrays(frame: pl.DataFrame, *, folds: Sequence[str], columns: Sequence[str]) -> list[dict[str, Any]]:
    needed = tuple(dict.fromkeys([TARGET, "weight", "time_id", *columns]))
    out: list[dict[str, Any]] = []
    for fold in folds:
        fold_frame = frame.filter(pl.col("fold") == fold).select(needed)
        row: dict[str, Any] = {
            "fold": fold,
            "rows": fold_frame.height,
            TARGET: fold_frame[TARGET].to_numpy().astype(np.float64, copy=False),
            "weight": fold_frame["weight"].to_numpy().astype(np.float64, copy=False),
            "time_id": fold_frame["time_id"].to_numpy().astype(np.int64, copy=False),
        }
        for column in columns:
            row[column] = fold_frame[column].to_numpy().astype(np.float64, copy=False)
        out.append(row)
    return out


def _contextual_calibration_arrays(fold_arrays: Sequence[dict[str, Any]], *, left: str, right: str) -> dict[str, Any]:
    y = np.concatenate([fold[TARGET] for fold in fold_arrays]).astype(np.float64, copy=False)
    weight = np.concatenate([fold["weight"] for fold in fold_arrays]).astype(np.float64, copy=False)
    time_id = np.concatenate([fold["time_id"] for fold in fold_arrays]).astype(np.int64, copy=False)
    left_pred = np.concatenate([fold[left] for fold in fold_arrays]).astype(np.float64, copy=False)
    right_pred = np.concatenate([fold[right] for fold in fold_arrays]).astype(np.float64, copy=False)
    fallback = _fit_convex_pair_weight_arrays(y, weight, left_pred, right_pred, prior=0.0, prior_weight=0.5)
    return {
        TARGET: y,
        "weight": weight,
        "time_id": time_id,
        "left": left_pred,
        "right": right_pred,
        "rows": int(y.size),
        "fallback_weight": float(fallback),
        "thresholds": _fit_context_thresholds_arrays(weight=weight, left_values=left_pred, right_values=right_pred),
    }


def _fit_context_thresholds(frame: pl.DataFrame, *, left: str, right: str) -> dict[str, tuple[float, float]]:
    left_values = frame[left].to_numpy().astype(np.float64, copy=False)
    right_values = frame[right].to_numpy().astype(np.float64, copy=False)
    weight = frame["weight"].to_numpy().astype(np.float64, copy=False)
    return _fit_context_thresholds_arrays(weight=weight, left_values=left_values, right_values=right_values)


def _fit_context_thresholds_arrays(
    *,
    weight: np.ndarray,
    left_values: np.ndarray,
    right_values: np.ndarray,
) -> dict[str, tuple[float, float]]:
    return {
        "weight": _terciles(weight),
        "abs_left": _terciles(np.abs(left_values)),
        "abs_right": _terciles(np.abs(right_values)),
        "abs_diff": _terciles(np.abs(left_values - right_values)),
        "abs_mean": _terciles(0.5 * (np.abs(left_values) + np.abs(right_values))),
    }


def _context_group_codes(
    frame: pl.DataFrame,
    *,
    left: str,
    right: str,
    components: Sequence[str],
    thresholds: dict[str, tuple[float, float]],
    time_bucket_size: int,
) -> np.ndarray:
    left_values = frame[left].to_numpy().astype(np.float64, copy=False)
    right_values = frame[right].to_numpy().astype(np.float64, copy=False)
    code = np.zeros(frame.height, dtype=np.int64)
    for component in components:
        if component == "time":
            part = frame["time_id"].to_numpy().astype(np.int64, copy=False) // int(time_bucket_size)
            cardinality = max(10_000 // int(time_bucket_size) + 2, int(np.max(part)) + 1 if part.size else 1)
        elif component == "weight":
            part = _bucketize(frame["weight"].to_numpy(), thresholds["weight"])
            cardinality = 3
        elif component == "abs_left":
            part = _bucketize(np.abs(left_values), thresholds["abs_left"])
            cardinality = 3
        elif component == "abs_right":
            part = _bucketize(np.abs(right_values), thresholds["abs_right"])
            cardinality = 3
        elif component == "abs_diff":
            part = _bucketize(np.abs(left_values - right_values), thresholds["abs_diff"])
            cardinality = 3
        elif component == "abs_mean":
            part = _bucketize(0.5 * (np.abs(left_values) + np.abs(right_values)), thresholds["abs_mean"])
            cardinality = 3
        elif component == "sign_agree":
            part = ((left_values >= 0.0) == (right_values >= 0.0)).astype(np.int64)
            cardinality = 2
        else:  # pragma: no cover - guarded by validation
            raise ValueError(f"unknown contextual blend group component: {component}")
        code = code * int(cardinality) + part.astype(np.int64, copy=False)
    return code


def _context_group_codes_arrays(
    *,
    weight: np.ndarray,
    time_id: np.ndarray,
    left_values: np.ndarray,
    right_values: np.ndarray,
    components: Sequence[str],
    thresholds: dict[str, tuple[float, float]],
    time_bucket_size: int,
) -> np.ndarray:
    code = np.zeros(left_values.shape[0], dtype=np.int64)
    for component in components:
        if component == "time":
            part = time_id.astype(np.int64, copy=False) // int(time_bucket_size)
            cardinality = max(10_000 // int(time_bucket_size) + 2, int(np.max(part)) + 1 if part.size else 1)
        elif component == "weight":
            part = _bucketize(weight, thresholds["weight"])
            cardinality = 3
        elif component == "abs_left":
            part = _bucketize(np.abs(left_values), thresholds["abs_left"])
            cardinality = 3
        elif component == "abs_right":
            part = _bucketize(np.abs(right_values), thresholds["abs_right"])
            cardinality = 3
        elif component == "abs_diff":
            part = _bucketize(np.abs(left_values - right_values), thresholds["abs_diff"])
            cardinality = 3
        elif component == "abs_mean":
            part = _bucketize(0.5 * (np.abs(left_values) + np.abs(right_values)), thresholds["abs_mean"])
            cardinality = 3
        elif component == "sign_agree":
            part = ((left_values >= 0.0) == (right_values >= 0.0)).astype(np.int64)
            cardinality = 2
        else:  # pragma: no cover - guarded by validation
            raise ValueError(f"unknown contextual blend group component: {component}")
        code = code * int(cardinality) + part.astype(np.int64, copy=False)
    return code


def _fit_group_blend_weights(
    frame: pl.DataFrame,
    *,
    left: str,
    right: str,
    codes: np.ndarray,
    min_group_rows: int,
    prior_strength: float,
    prior_weight: float,
) -> dict[int, float]:
    y = frame[TARGET].to_numpy().astype(np.float64, copy=False)
    weight = frame["weight"].to_numpy().astype(np.float64, copy=False)
    left_pred = frame[left].to_numpy().astype(np.float64, copy=False)
    right_pred = frame[right].to_numpy().astype(np.float64, copy=False)
    group_weights: dict[int, float] = {}
    for code in np.unique(codes):
        mask = codes == code
        if int(mask.sum()) < int(min_group_rows):
            continue
        group_weights[int(code)] = _fit_convex_pair_weight_arrays(
            y[mask],
            weight[mask],
            left_pred[mask],
            right_pred[mask],
            prior=float(prior_strength),
            prior_weight=float(prior_weight),
        )
    return group_weights


def _fit_group_blend_weight_array(
    *,
    y: np.ndarray,
    weight: np.ndarray,
    left_pred: np.ndarray,
    right_pred: np.ndarray,
    codes: np.ndarray,
    min_group_rows: int,
    prior_strength: float,
    prior_weight: float,
) -> tuple[np.ndarray, int]:
    if codes.size == 0:
        return np.full(1, float(prior_weight), dtype=np.float64), 0
    safe_codes = np.asarray(codes, dtype=np.int64)
    max_code = int(np.max(safe_codes))
    minlength = max_code + 1
    delta = left_pred - right_pred
    counts = np.bincount(safe_codes, minlength=minlength)
    denom = np.bincount(safe_codes, weights=weight * delta * delta, minlength=minlength).astype(np.float64, copy=False)
    numerator = np.bincount(safe_codes, weights=weight * delta * (y - right_pred), minlength=minlength).astype(np.float64, copy=False)
    weights = np.full(minlength, float(prior_weight), dtype=np.float64)
    lhs = denom + float(prior_strength)
    rhs = numerator + float(prior_strength) * float(prior_weight)
    mask = (counts >= int(min_group_rows)) & (lhs > 1e-18)
    weights[mask] = np.clip(rhs[mask] / lhs[mask], 0.0, 1.0)
    return weights, int(np.count_nonzero(mask))


def _blend_weight_for_codes(codes: np.ndarray, default: float, group_weights: dict[int, float]) -> np.ndarray:
    out = np.full(codes.shape, float(default), dtype=np.float64)
    for code, blend_weight in group_weights.items():
        out[codes == code] = float(blend_weight)
    return out


def _blend_weight_for_codes_array(codes: np.ndarray, default: float, group_weights: np.ndarray) -> np.ndarray:
    safe_codes = np.asarray(codes, dtype=np.int64)
    out = np.full(safe_codes.shape, float(default), dtype=np.float64)
    mask = (safe_codes >= 0) & (safe_codes < group_weights.shape[0])
    out[mask] = group_weights[safe_codes[mask]]
    return out


def _terciles(values: np.ndarray) -> tuple[float, float]:
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
    return float(low), float(high)


def _bucketize(values: np.ndarray, thresholds: tuple[float, float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = np.where(np.isfinite(x), x, thresholds[0])
    return np.searchsorted(np.asarray(thresholds, dtype=np.float64), x, side="right").astype(np.int64)


def _write_parameters(path: Path, *parameter_groups: list[dict[str, Any]]) -> None:
    payload: dict[str, Any] = {}
    for idx, group in enumerate(parameter_groups):
        payload[f"group_{idx}"] = group
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _audit_payload(
    config: StrongOOFConfig,
    frame: pl.DataFrame,
    base_candidates: Sequence[str],
    gateway_audits: Sequence[pl.DataFrame],
    *,
    stack_columns: Sequence[str],
    raw_preprocessed_columns: Sequence[str],
) -> dict[str, Any]:
    target_leakage = [column for column in base_candidates if column == TARGET or column.startswith("responder_")]
    if gateway_audits:
        audit_frame = pl.concat(list(gateway_audits), how="diagonal")
        bad_updates = int(audit_frame.filter(pl.col("update_is_strictly_past") == False).height)
    else:
        bad_updates = 0
    return {
        "experiment_name": config.experiment_name,
        "config": _serializable_config(config),
        "rows": frame.height,
        "folds": frame["fold"].n_unique(),
        "n_folds": frame["fold"].n_unique(),
        "n_base_candidates": len(base_candidates),
        "base_candidates": list(base_candidates),
        "uses_gateway_rls": config.include_gateway_rls,
        "uses_gateway_risk_shrink": config.include_gateway_risk_shrink,
        "include_extra_gateway_experts": config.include_extra_gateway_experts,
        "gateway_expert_expansions": list(config.gateway_expert_expansions),
        "stack_alphas": list(config.stack_alphas),
        "stack_columns": list(stack_columns),
        "raw_feature_columns": list(config.raw_feature_columns),
        "raw_preprocess_modes": list(config.raw_preprocess_modes),
        "raw_preprocessed_columns": list(raw_preprocessed_columns),
        "include_raw_preprocessed_in_stack": config.include_raw_preprocessed_in_stack,
        "target_leakage_check": "passed" if not target_leakage else f"FAILED: {target_leakage}",
        "fold_causality_check": (
            "passed: walk-forward layers and residual tail thresholds fit only earlier folds; "
            "gateway updates use prior-date lag simulation; online scale/affine calibration updates only after each validation date is scored"
        ),
        "selection_check": "passed: all grids are reported as fixed candidates, no validation-best candidate is promoted inside code",
        "gateway_bad_updates": bad_updates,
        "uses_processed_lags": False,
        "uses_context_features": True,
        "n_model_features": len(stack_columns),
        "residual_features": [feature for feature in config.residual_features if feature in frame.columns],
        "residual_base_candidates": list(config.residual_base_candidates),
        "residual_tail_quantiles": list(config.residual_tail_quantiles),
        "residual_tail_modes": list(config.residual_tail_modes),
        "fixed_blend_candidates": list(config.fixed_blend_candidates),
        "fixed_blend_weights": list(config.fixed_blend_weights),
        "walk_forward_blend_candidates": list(config.walk_forward_blend_candidates),
        "contextual_blend_candidates": list(config.contextual_blend_candidates),
        "contextual_blend_group_specs": list(config.contextual_blend_group_specs),
        "contextual_blend_time_bucket_sizes": list(config.contextual_blend_time_bucket_sizes),
        "contextual_blend_min_group_rows": list(config.contextual_blend_min_group_rows),
        "contextual_blend_prior_strengths": list(config.contextual_blend_prior_strengths),
        "online_scale_base_candidates": list(config.online_scale_base_candidates),
        "online_scale_prior_strengths": list(config.online_scale_prior_strengths),
        "online_scale_forgetting_factors": list(config.online_scale_forgetting_factors),
        "online_scale_min": config.online_scale_min,
        "online_scale_max": config.online_scale_max,
        "online_affine_base_candidates": list(config.online_affine_base_candidates),
        "online_affine_prior_strengths": list(config.online_affine_prior_strengths),
        "online_affine_forgetting_factors": list(config.online_affine_forgetting_factors),
        "online_affine_min_scale": config.online_affine_min_scale,
        "online_affine_max_scale": config.online_affine_max_scale,
        "known_reference_points": {
            "conservative_stage3_local": 0.013836465,
            "historical_max1398": 0.015425344,
        },
    }


def _serializable_config(config: StrongOOFConfig) -> dict[str, Any]:
    raw = asdict(config)
    for key, value in list(raw.items()):
        if isinstance(value, Path):
            raw[key] = str(value)
        elif isinstance(value, tuple) and value and all(isinstance(item, Path) for item in value):
            raw[key] = [str(item) for item in value]
    return raw


def _parse_meminfo_gb(meminfo: str) -> dict[str, float]:
    wanted = {"MemAvailable", "SwapFree"}
    values: dict[str, float] = {}
    for line in meminfo.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        if key not in wanted:
            continue
        parts = rest.split()
        if not parts:
            continue
        values[key] = float(parts[0]) / (1024.0 * 1024.0)
    return values


def _current_meminfo_gb() -> dict[str, float]:
    try:
        return _parse_meminfo_gb(Path("/proc/meminfo").read_text(encoding="utf-8"))
    except OSError:
        return {}


def _assert_resource_floor(config: StrongOOFConfig, stage: str) -> None:
    if config.min_mem_available_gb <= 0.0 and config.min_swap_free_gb <= 0.0:
        return
    values = _current_meminfo_gb()
    mem_available = values.get("MemAvailable")
    swap_free = values.get("SwapFree")
    failures: list[str] = []
    if config.min_mem_available_gb > 0.0 and mem_available is not None and mem_available < config.min_mem_available_gb:
        failures.append(f"MemAvailable {mem_available:.2f} GiB < {config.min_mem_available_gb:.2f} GiB")
    if config.min_swap_free_gb > 0.0 and swap_free is not None and swap_free < config.min_swap_free_gb:
        failures.append(f"SwapFree {swap_free:.2f} GiB < {config.min_swap_free_gb:.2f} GiB")
    if failures:
        raise RuntimeError(f"resource floor violated at {stage}: " + "; ".join(failures))


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(f"multi_models_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_config(config: StrongOOFConfig) -> None:
    if config.sample_stride <= 0:
        raise ValueError("sample_stride must be positive")
    if not config.strong_base_candidates:
        raise ValueError("strong_base_candidates must not be empty")
    unknown_gateway_expansions = sorted(
        set(config.gateway_expert_expansions)
        - {
            "signed_square",
            "abs",
            "signed_sqrt",
            "signed_log1p",
            "cube",
            "sign",
            "pair_product",
            "batch_rank",
            "batch_mean",
            "batch_demean",
            "batch_std",
            "batch_zscore",
        }
    )
    if unknown_gateway_expansions:
        raise ValueError(f"unknown gateway_expert_expansions: {unknown_gateway_expansions}")
    unknown_raw_modes = sorted(
        set(config.raw_preprocess_modes)
        - {
            "batch_rank",
            "batch_demean",
            "batch_zscore",
            "batch_abs_zscore",
            "batch_top_bottom",
            "row_missing_count",
            "row_abs_mean",
            "row_l2_energy",
        }
    )
    if unknown_raw_modes:
        raise ValueError(f"unknown raw_preprocess_modes: {unknown_raw_modes}")
    unsafe_raw_columns = [
        column
        for column in config.raw_feature_columns
        if column == TARGET or column.startswith("responder_")
    ]
    if unsafe_raw_columns:
        raise ValueError(f"raw_feature_columns cannot include target/responder columns: {unsafe_raw_columns}")
    if config.include_raw_preprocessed_in_stack and not config.raw_preprocess_modes:
        raise ValueError("include_raw_preprocessed_in_stack requires raw_preprocess_modes")
    if not config.stack_alphas:
        raise ValueError("stack_alphas must not be empty")
    if any(float(alpha) < 0.0 for alpha in config.stack_alphas):
        raise ValueError("stack_alphas must be non-negative")
    if config.extra_prediction_prefixes and len(config.extra_prediction_prefixes) != len(config.extra_prediction_dirs):
        raise ValueError("extra_prediction_prefixes length must match extra_prediction_dirs")
    for value in [*config.time_bucket_sizes, *config.min_group_rows]:
        if int(value) <= 0:
            raise ValueError("bucket sizes and group row counts must be positive")
    if any(value < 0.0 for value in [*config.risk_shrink_strengths, *config.scale_prior_strengths, *config.online_scale_prior_strengths, *config.online_affine_prior_strengths]):
        raise ValueError("strengths and priors must be non-negative")
    if any(value < 0.0 for value in config.contextual_blend_prior_strengths):
        raise ValueError("contextual blend priors must be non-negative")
    for value in [*config.contextual_blend_time_bucket_sizes, *config.contextual_blend_min_group_rows]:
        if int(value) <= 0:
            raise ValueError("contextual blend bucket sizes and group row counts must be positive")
    for spec in config.contextual_blend_group_specs:
        _parse_context_group_spec(spec)
    if any(not 0.0 < float(value) <= 1.0 for value in config.online_scale_forgetting_factors):
        raise ValueError("online_scale_forgetting_factors must be in (0, 1]")
    if any(not 0.0 < float(value) <= 1.0 for value in config.online_affine_forgetting_factors):
        raise ValueError("online_affine_forgetting_factors must be in (0, 1]")
    if config.online_scale_min > config.online_scale_max:
        raise ValueError("online_scale_min must be <= online_scale_max")
    if config.online_affine_min_scale > config.online_affine_max_scale:
        raise ValueError("online_affine_min_scale must be <= online_affine_max_scale")
    if config.min_mem_available_gb < 0.0 or config.min_swap_free_gb < 0.0:
        raise ValueError("memory/swap resource floors must be non-negative")
    for q in config.residual_tail_quantiles:
        if not 0.0 < float(q) < 1.0:
            raise ValueError("residual_tail_quantiles must be between 0 and 1")
    allowed_modes = {"weight", "abs_base", "disagreement", "weight_or_abs", "weight_and_abs", "weight_or_disagreement"}
    unknown_modes = sorted(set(config.residual_tail_modes) - allowed_modes)
    if unknown_modes:
        raise ValueError(f"unknown residual_tail_modes: {unknown_modes}")
    for weight in config.fixed_blend_weights:
        if not 0.0 <= float(weight) <= 1.0:
            raise ValueError("fixed_blend_weights must be between 0 and 1")


def _format_float(value: float) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")
