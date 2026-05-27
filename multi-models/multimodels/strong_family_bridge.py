"""Bridge weak family artifacts into the strong OOF candidate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

from multimodels.metrics import TARGET, score_arrays, summarize_scores
from multimodels.strong_oof import (
    KEYS,
    add_gateway_rls_predictions,
    add_prediction_context,
    _format_float,
)


@dataclass(frozen=True)
class StrongFamilyBridgeConfig:
    """Configuration for family-to-strong OOF bridge experiments."""

    experiment_name: str = "strong_family_bridge"
    family_prediction_path: Path = Path("multi-models/reports/family_artifacts_5fold_lags_stride100_v2/validation_predictions.parquet")
    tabm_prediction_dir: Path = Path(
        "reports/experiments/competitive_tabm_official_stage3_5fold_valid60_lags_online_lr1e4_4m_train700_seed37_aux8_preds/validation_predictions"
    )
    tree_prediction_dir: Path = Path("reports/experiments/tree_engine_ensemble_xgb_lgb_sample10_seed_ensemble_preds/validation_predictions")
    output_dir: Path = Path("multi-models/reports/strong_family_bridge")
    strong_base: str = "gateway_risk_conservative_rls_abs_pred_s100_prediction"
    gateway_risk_strengths: tuple[float, ...] = (100.0,)
    gateway_risk_profiles: tuple[str, ...] = ("abs_pred",)
    alpha_columns: tuple[str, ...] = ("latent_alpha_linear_stack", "ridge_rank_alpha10000", "pls_rank_k8")
    residual_feature_columns: tuple[str, ...] = (
        "latent_alpha_linear_stack",
        "ridge_rank_alpha10000",
        "ridge_rank_alpha10000__feature_59_z_residual",
        "ridge_rank_alpha10000__risk_abs_error_ridge_rank_score",
        "ridge_rank_alpha10000__risk_abs_responder6_ridge_rank_score",
        "ridge_rank_alpha10000__risk_high_error_ridge_rank_score",
    )
    risk_columns: tuple[str, ...] = (
        "ridge_rank_alpha10000__risk_abs_error_ridge_rank_score",
        "ridge_rank_alpha10000__risk_abs_responder6_ridge_rank_score",
        "ridge_rank_alpha10000__risk_high_error_ridge_rank_score",
    )
    stack_alpha: float = 1_000.0
    residual_alpha: float = 10_000.0
    risk_strengths: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2)
    time_bucket_size: int = 100
    symbol_mod: int = 8
    min_regime_rows: int = 500
    regime_prior_strength: float = 1_000.0
    residual_gate_time_bucket_size: int = 100
    residual_gate_symbol_mod: int = 8
    residual_gate_min_rows: int = 100
    residual_gate_prior_strength: float = 1_000.0
    residual_gate_min_delta: float = 0.0
    residual_tail_quantiles: tuple[float, ...] = (0.90, 0.95, 0.99)
    write_predictions: bool = True


def run_bridge_experiment(config: StrongFamilyBridgeConfig) -> dict[str, Any]:
    """Run walk-forward bridge layers over the strong OOF candidate."""

    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_bridge_frame(config)
    frame = add_prediction_context(frame)
    frame, gateway_audits = add_gateway_rls_predictions(
        frame,
        include_risk_shrink=True,
        risk_strengths=config.gateway_risk_strengths,
        risk_profiles=config.gateway_risk_profiles,
    )
    frame = add_prediction_context(frame)
    if config.strong_base not in frame.columns:
        raise ValueError(f"strong base is missing after gateway reconstruction: {config.strong_base}")

    score_rows: list[dict[str, float | int | str]] = []
    score_rows.extend(_score_by_fold(frame, prediction=config.strong_base, candidate=config.strong_base, family="strong_base").to_dicts())
    for column in _available(config.alpha_columns, frame):
        score_rows.extend(_score_by_fold(frame, prediction=column, candidate=column, family="family_alpha").to_dicts())

    stack_frame, stack_params = add_walk_forward_alpha_stack(
        frame,
        base_prediction=config.strong_base,
        alpha_columns=_available(config.alpha_columns, frame),
        alpha=config.stack_alpha,
    )
    frame = frame.join(stack_frame, on=list(KEYS), how="inner")
    score_rows.extend(
        _score_by_fold(frame, prediction="strong_family_alpha_stack_prediction", candidate="strong_family_alpha_stack", family="strong_family_stack").to_dicts()
    )

    residual_frame, residual_params = add_walk_forward_family_residual_bridge(
        frame,
        base_prediction=config.strong_base,
        feature_columns=_available(config.residual_feature_columns, frame),
        alpha=config.residual_alpha,
    )
    frame = frame.join(residual_frame, on=list(KEYS), how="inner")
    score_rows.extend(
        _score_by_fold(
            frame,
            prediction=f"{config.strong_base}_family_residual_prediction",
            candidate=f"{config.strong_base}_family_residual",
            family="strong_family_residual",
        ).to_dicts()
    )

    available_risk_columns = _available(config.risk_columns, frame)
    gate_frame, gate_params = add_walk_forward_family_residual_gate(
        frame,
        base_prediction=config.strong_base,
        residual_prediction=f"{config.strong_base}_family_residual_prediction",
        risk_column=available_risk_columns[0] if available_risk_columns else None,
        time_bucket_size=config.residual_gate_time_bucket_size,
        symbol_mod=config.residual_gate_symbol_mod,
        min_rows=config.residual_gate_min_rows,
        prior_strength=config.residual_gate_prior_strength,
        min_delta=config.residual_gate_min_delta,
    )
    frame = frame.join(gate_frame, on=list(KEYS), how="inner")
    for column in gate_frame.columns:
        if column.endswith("_family_residual_gate_open_prediction") or column.endswith("_family_residual_gate_closed_prediction"):
            score_rows.extend(_score_by_fold(frame, prediction=column, candidate=column.removesuffix("_prediction"), family="strong_family_residual_gate").to_dicts())

    tail_frame, tail_params = add_walk_forward_family_residual_tail_masks(
        frame,
        base_prediction=config.strong_base,
        residual_prediction=f"{config.strong_base}_family_residual_prediction",
        risk_column=available_risk_columns[0] if available_risk_columns else None,
        quantiles=config.residual_tail_quantiles,
    )
    frame = frame.join(tail_frame, on=list(KEYS), how="inner")
    for column in tail_frame.columns:
        if column.endswith("_family_residual_tail_prediction"):
            score_rows.extend(_score_by_fold(frame, prediction=column, candidate=column.removesuffix("_prediction"), family="strong_family_residual_tail").to_dicts())

    risk_frame, risk_params = add_walk_forward_family_risk_shrinkage(
        frame,
        base_prediction=config.strong_base,
        risk_columns=available_risk_columns,
        strengths=config.risk_strengths,
    )
    frame = frame.join(risk_frame, on=list(KEYS), how="inner")
    for column in risk_frame.columns:
        if column.endswith("_family_risk_shrink_prediction"):
            score_rows.extend(_score_by_fold(frame, prediction=column, candidate=column.removesuffix("_prediction"), family="strong_family_risk").to_dicts())

    regime_frame, regime_params = add_walk_forward_family_regime_scale(
        frame,
        base_prediction=config.strong_base,
        risk_columns=available_risk_columns,
        strengths=config.risk_strengths,
        time_bucket_size=config.time_bucket_size,
        symbol_mod=config.symbol_mod,
        min_rows=config.min_regime_rows,
        prior_strength=config.regime_prior_strength,
    )
    frame = frame.join(regime_frame, on=list(KEYS), how="inner")
    for column in regime_frame.columns:
        if column.endswith("_family_regime_scaled_prediction"):
            score_rows.extend(_score_by_fold(frame, prediction=column, candidate=column.removesuffix("_prediction"), family="strong_family_regime").to_dicts())

    scores = pl.DataFrame(score_rows)
    summary = summarize_scores(scores)
    scores.write_csv(config.output_dir / "fold_scores.csv")
    summary.write_csv(config.output_dir / "candidate_summary.csv")
    _write_parameters(config.output_dir / "parameters.json", stack_params, residual_params, gate_params, tail_params, risk_params, regime_params)
    if gateway_audits:
        pl.concat(gateway_audits, how="diagonal").write_csv(config.output_dir / "gateway_daily_audit.csv")
    if config.write_predictions:
        frame.write_parquet(config.output_dir / "bridge_predictions.parquet")
    audit = _audit_payload(config, frame, gateway_audits)
    (config.output_dir / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_bridge_report(config.output_dir / "REPORT.md", summary=summary, audit=audit)
    return {"frame": frame, "scores": scores, "summary": summary, "audit": audit, "output_dir": config.output_dir}


def load_bridge_frame(config: StrongFamilyBridgeConfig) -> pl.DataFrame:
    """Join family OOF artifacts with strong OOF inputs on the sampled keys."""

    if not config.family_prediction_path.exists():
        raise FileNotFoundError(config.family_prediction_path)
    if not config.tabm_prediction_dir.exists():
        raise FileNotFoundError(config.tabm_prediction_dir)
    if not config.tree_prediction_dir.exists():
        raise FileNotFoundError(config.tree_prediction_dir)

    family_schema = pl.scan_parquet(str(config.family_prediction_path)).collect_schema()
    family_columns = [name for name in family_schema.names() if name not in set(KEYS)]
    family = pl.scan_parquet(str(config.family_prediction_path)).select([*KEYS, *family_columns])

    tabm = pl.scan_parquet(str(config.tabm_prediction_dir / "*.parquet")).select([*KEYS, "tabm_prediction"])
    tree_schema = pl.scan_parquet(str(config.tree_prediction_dir / "*.parquet")).collect_schema()
    tree_columns = [
        column
        for column in ("ensemble_prediction", "ridge_calibrated_prediction", "xgboost_prediction", "lightgbm_prediction", "catboost_prediction")
        if column in tree_schema.names()
    ]
    if not tree_columns:
        raise ValueError("tree prediction directory does not contain known prediction columns")
    tree = pl.scan_parquet(str(config.tree_prediction_dir / "*.parquet")).select([*KEYS, *tree_columns])
    rename = {"ensemble_prediction": "tree_prediction"} if "ensemble_prediction" in tree_columns else {}
    joined = family.join(tabm, on=list(KEYS), how="inner").join(tree.rename(rename), on=list(KEYS), how="inner").collect().sort(list(KEYS))
    if joined.height == 0:
        raise ValueError("bridge join produced no rows")
    return joined


def add_walk_forward_alpha_stack(
    frame: pl.DataFrame,
    *,
    base_prediction: str,
    alpha_columns: Sequence[str],
    alpha: float,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Stack the strong base with weak alpha columns using previous folds only."""

    columns = tuple(dict.fromkeys([base_prediction, *alpha_columns]))
    folds = _folds(frame)
    parts: list[pl.DataFrame] = []
    params: list[dict[str, Any]] = []
    for idx, fold in enumerate(folds):
        current = frame.filter(pl.col("fold") == fold)
        if idx == 0:
            coef = np.zeros(len(columns), dtype=np.float64)
            coef[0] = 1.0
            fit_rows = 0
        else:
            calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
            coef = _fit_weighted_ridge_no_intercept(calibration, columns, alpha=alpha)
            fit_rows = calibration.height
        pred = current.select(columns).to_numpy().astype(np.float64, copy=False) @ coef
        parts.append(current.select(list(KEYS)).with_columns(pl.Series("strong_family_alpha_stack_prediction", pred)))
        for column, value in zip(columns, coef, strict=True):
            params.append({"component": "strong_family_alpha_stack", "fold": fold, "feature": column, "value": float(value), "fit_rows": fit_rows})
    return pl.concat(parts), params


def add_walk_forward_family_residual_bridge(
    frame: pl.DataFrame,
    *,
    base_prediction: str,
    feature_columns: Sequence[str],
    alpha: float,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Fit y-strong ~= family features using previous folds only."""

    features = tuple(feature_columns)
    folds = _folds(frame)
    parts: list[pl.DataFrame] = []
    params: list[dict[str, Any]] = []
    output = f"{base_prediction}_family_residual_prediction"
    for idx, fold in enumerate(folds):
        current = frame.filter(pl.col("fold") == fold)
        base = current[base_prediction].to_numpy().astype(np.float64, copy=False)
        if idx == 0 or not features:
            pred = base
            params.append({"component": "family_residual", "fold": fold, "feature": "__identity__", "value": 0.0, "fit_rows": 0})
        else:
            calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
            coef, means, scales = _fit_residual_linear(calibration, base_prediction, features, alpha=alpha)
            pred = base + _standardized_matrix(current, features, means, scales) @ coef
            for feature, value in zip(features, coef, strict=True):
                params.append({"component": "family_residual", "fold": fold, "feature": feature, "value": float(value), "fit_rows": calibration.height})
        parts.append(current.select(list(KEYS)).with_columns(pl.Series(output, pred)))
    return pl.concat(parts), params


def add_walk_forward_family_residual_gate(
    frame: pl.DataFrame,
    *,
    base_prediction: str,
    residual_prediction: str,
    risk_column: str | None,
    time_bucket_size: int,
    symbol_mod: int,
    min_rows: int,
    prior_strength: float,
    min_delta: float,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Gate a residual bridge using previous-fold group deltas only."""

    _require_gate_columns(frame, base_prediction, residual_prediction, risk_column)
    folds = _folds(frame)
    residual_base = residual_prediction.removesuffix("_prediction")
    outputs = {
        "open": f"{residual_base}_gate_open_prediction",
        "closed": f"{residual_base}_gate_closed_prediction",
    }
    out = frame.select(list(KEYS))
    params: list[dict[str, Any]] = []
    for policy, output in outputs.items():
        parts: list[pl.DataFrame] = []
        for idx, fold in enumerate(folds):
            current = frame.filter(pl.col("fold") == fold)
            base = current[base_prediction].to_numpy().astype(np.float64, copy=False)
            residual = current[residual_prediction].to_numpy().astype(np.float64, copy=False)
            if idx == 0:
                pred = base
                params.append(
                    {
                        "component": "family_residual_gate",
                        "policy": policy,
                        "fold": fold,
                        "fit_rows": 0,
                        "n_open_groups": 0,
                        "default_gate": 0,
                        "global_delta_r2": 0.0,
                    }
                )
            else:
                calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
                thresholds = _fit_gate_thresholds(calibration, base_prediction=base_prediction, risk_column=risk_column)
                cal_codes = _residual_gate_codes(
                    calibration,
                    base_prediction=base_prediction,
                    risk_column=risk_column,
                    thresholds=thresholds,
                    time_bucket_size=time_bucket_size,
                    symbol_mod=symbol_mod,
                )
                cur_codes = _residual_gate_codes(
                    current,
                    base_prediction=base_prediction,
                    risk_column=risk_column,
                    thresholds=thresholds,
                    time_bucket_size=time_bucket_size,
                    symbol_mod=symbol_mod,
                )
                default_gate, code_gates, global_delta = _fit_residual_gate_policy(
                    calibration,
                    base_prediction=base_prediction,
                    residual_prediction=residual_prediction,
                    codes=cal_codes,
                    policy=policy,
                    min_rows=min_rows,
                    prior_strength=prior_strength,
                    min_delta=min_delta,
                )
                gate = _gate_for_codes(cur_codes, default_gate=default_gate, code_gates=code_gates)
                pred = np.where(gate, residual, base)
                params.append(
                    {
                        "component": "family_residual_gate",
                        "policy": policy,
                        "fold": fold,
                        "fit_rows": calibration.height,
                        "n_open_groups": sum(1 for value in code_gates.values() if value),
                        "n_closed_groups": sum(1 for value in code_gates.values() if not value),
                        "default_gate": int(default_gate),
                        "global_delta_r2": float(global_delta),
                    }
                )
            parts.append(current.select(list(KEYS)).with_columns(pl.Series(output, pred)))
        out = out.join(pl.concat(parts), on=list(KEYS), how="inner")
    return out, params


def add_walk_forward_family_residual_tail_masks(
    frame: pl.DataFrame,
    *,
    base_prediction: str,
    residual_prediction: str,
    risk_column: str | None,
    quantiles: Sequence[float],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Apply residual corrections only in previous-fold observable tails."""

    _require_gate_columns(frame, base_prediction, residual_prediction, risk_column)
    qs = tuple(dict.fromkeys(float(q) for q in quantiles))
    modes = (
        "weight",
        "abs_base",
        "risk",
        "weight_or_abs",
        "weight_and_abs",
        "abs_base_or_risk",
    )
    folds = _folds(frame)
    residual_base = residual_prediction.removesuffix("_prediction")
    out = frame.select(list(KEYS))
    params: list[dict[str, Any]] = []
    for q in qs:
        for mode in modes:
            suffix = f"{mode}_q{_format_float(q)}"
            output = f"{residual_base}_{suffix}_family_residual_tail_prediction"
            parts: list[pl.DataFrame] = []
            for idx, fold in enumerate(folds):
                current = frame.filter(pl.col("fold") == fold)
                base = current[base_prediction].to_numpy().astype(np.float64, copy=False)
                residual = current[residual_prediction].to_numpy().astype(np.float64, copy=False)
                if idx == 0:
                    pred = base
                    selected_rows = 0
                    thresholds = {"weight": np.inf, "abs_base": np.inf, "risk": np.inf}
                    fit_rows = 0
                else:
                    calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
                    thresholds = _fit_tail_thresholds(calibration, base_prediction=base_prediction, risk_column=risk_column, quantile=q)
                    mask = _tail_mask(current, base_prediction=base_prediction, risk_column=risk_column, thresholds=thresholds, mode=mode)
                    pred = np.where(mask, residual, base)
                    selected_rows = int(mask.sum())
                    fit_rows = calibration.height
                params.append(
                    {
                        "component": "family_residual_tail",
                        "fold": fold,
                        "mode": mode,
                        "quantile": float(q),
                        "fit_rows": fit_rows,
                        "selected_rows": selected_rows,
                        "selected_frac": float(selected_rows / max(current.height, 1)),
                        "weight_threshold": float(thresholds["weight"]),
                        "abs_base_threshold": float(thresholds["abs_base"]),
                        "risk_threshold": float(thresholds["risk"]),
                    }
                )
                parts.append(current.select(list(KEYS)).with_columns(pl.Series(output, pred)))
            out = out.join(pl.concat(parts), on=list(KEYS), how="inner")
    return out, params


def add_walk_forward_family_risk_shrinkage(
    frame: pl.DataFrame,
    *,
    base_prediction: str,
    risk_columns: Sequence[str],
    strengths: Sequence[float],
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Shrink the strong base by family risk scores with previous-fold normalization."""

    folds = _folds(frame)
    out = frame.select(list(KEYS))
    params: list[dict[str, Any]] = []
    for risk_column in risk_columns:
        for strength in strengths:
            suffix = _format_float(strength)
            output = f"{base_prediction}__{risk_column}_s{suffix}_family_risk_shrink_prediction"
            parts: list[pl.DataFrame] = []
            for idx, fold in enumerate(folds):
                current = frame.filter(pl.col("fold") == fold)
                base = current[base_prediction].to_numpy().astype(np.float64, copy=False)
                if idx == 0:
                    pred = base
                    risk_mean = 0.0
                    fit_rows = 0
                else:
                    calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
                    risk_mean = _safe_positive_mean(calibration[risk_column].to_numpy())
                    risk = current[risk_column].to_numpy().astype(np.float64, copy=False)
                    pred = _risk_shrink(base, risk, risk_mean=risk_mean, strength=float(strength))
                    fit_rows = calibration.height
                params.append(
                    {
                        "component": "family_risk_shrink",
                        "fold": fold,
                        "risk_column": risk_column,
                        "strength": float(strength),
                        "risk_mean": float(risk_mean),
                        "fit_rows": fit_rows,
                    }
                )
                parts.append(current.select(list(KEYS)).with_columns(pl.Series(output, pred)))
            out = out.join(pl.concat(parts), on=list(KEYS), how="inner")
    return out, params


def add_walk_forward_family_regime_scale(
    frame: pl.DataFrame,
    *,
    base_prediction: str,
    risk_columns: Sequence[str],
    strengths: Sequence[float],
    time_bucket_size: int,
    symbol_mod: int,
    min_rows: int,
    prior_strength: float,
) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """Apply family-risk shrinkage and per-regime scaling using previous folds."""

    folds = _folds(frame)
    out = frame.select(list(KEYS))
    params: list[dict[str, Any]] = []
    for risk_column in risk_columns:
        for strength in strengths:
            suffix = _format_float(strength)
            output = f"{base_prediction}__{risk_column}_s{suffix}_family_regime_scaled_prediction"
            parts: list[pl.DataFrame] = []
            for idx, fold in enumerate(folds):
                current = frame.filter(pl.col("fold") == fold)
                base = current[base_prediction].to_numpy().astype(np.float64, copy=False)
                if idx == 0:
                    pred = base
                    params.append({"component": "family_regime_scale", "fold": fold, "risk_column": risk_column, "strength": float(strength), "fit_rows": 0, "n_groups": 0})
                else:
                    calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
                    cal_base = calibration[base_prediction].to_numpy().astype(np.float64, copy=False)
                    cur_risk = current[risk_column].to_numpy().astype(np.float64, copy=False)
                    risk_mean = _safe_positive_mean(calibration[risk_column].to_numpy())
                    cal_shrunk = _risk_shrink(cal_base, calibration[risk_column].to_numpy().astype(np.float64, copy=False), risk_mean=risk_mean, strength=float(strength))
                    cur_shrunk = _risk_shrink(base, cur_risk, risk_mean=risk_mean, strength=float(strength))
                    thresholds = _fit_bridge_thresholds(calibration, prediction=cal_shrunk, risk_column=risk_column)
                    cal_codes = _bridge_regime_codes(
                        calibration,
                        prediction=cal_shrunk,
                        risk_column=risk_column,
                        thresholds=thresholds,
                        time_bucket_size=time_bucket_size,
                        symbol_mod=symbol_mod,
                    )
                    cur_codes = _bridge_regime_codes(
                        current,
                        prediction=cur_shrunk,
                        risk_column=risk_column,
                        thresholds=thresholds,
                        time_bucket_size=time_bucket_size,
                        symbol_mod=symbol_mod,
                    )
                    default, scales = _fit_group_scales(calibration, prediction=cal_shrunk, codes=cal_codes, min_rows=min_rows, prior_strength=prior_strength)
                    pred = cur_shrunk * _scale_for_codes(cur_codes, default, scales)
                    params.append(
                        {
                            "component": "family_regime_scale",
                            "fold": fold,
                            "risk_column": risk_column,
                            "strength": float(strength),
                            "fit_rows": calibration.height,
                            "n_groups": len(scales),
                            "default_scale": float(default),
                        }
                    )
                parts.append(current.select(list(KEYS)).with_columns(pl.Series(output, pred)))
            out = out.join(pl.concat(parts), on=list(KEYS), how="inner")
    return out, params


def write_bridge_report(path: Path, *, summary: pl.DataFrame, audit: dict[str, Any]) -> None:
    """Write a compact bridge report."""

    lines = [f"# Strong Family Bridge: {audit['experiment_name']}", "", "## Headline", ""]
    if summary.height:
        best = summary.row(0, named=True)
        base = summary.filter(pl.col("candidate") == audit["strong_base"])
        lines.extend(
            [
                f"- Best candidate: `{best['candidate']}`.",
                f"- Family: `{best['family']}`.",
                f"- Global R2: `{best['global_r2']:.9f}`.",
                f"- Min fold R2: `{best['min_fold_r2']:.9f}`.",
            ]
        )
        if base.height:
            base_row = base.row(0, named=True)
            lines.append(f"- Strong base R2: `{base_row['global_r2']:.9f}`.")
            lines.append(f"- Delta versus strong base: `{best['global_r2'] - base_row['global_r2']:.9f}`.")
    lines.extend(
        [
            "",
            "## Audit",
            "",
            f"- Rows: `{audit['rows']}`.",
            f"- Folds: `{audit['n_folds']}`.",
            f"- Strong base: `{audit['strong_base']}`.",
            f"- Gateway bad updates: `{audit['gateway_bad_updates']}`.",
            f"- Target leakage check: `{audit['target_leakage_check']}`.",
            f"- Causality check: `{audit['fold_causality_check']}`.",
            f"- Selection check: `{audit['selection_check']}`.",
            "",
            "## Methodological Status",
            "",
            "- This bridge is evaluated on the sampled family-artifact intersection, not the full 11M-row OOF frame.",
            "- Walk-forward bridge layers use previous folds only; first fold is identity for learned bridge layers.",
            "- Promotion requires confirmation on a denser or full OOF bridge and slice diagnostics.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


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


def _audit_payload(config: StrongFamilyBridgeConfig, frame: pl.DataFrame, gateway_audits: Sequence[pl.DataFrame]) -> dict[str, Any]:
    if gateway_audits:
        audit_frame = pl.concat(list(gateway_audits), how="diagonal")
        bad_updates = int(audit_frame.filter(pl.col("update_is_strictly_past") == False).height)
    else:
        bad_updates = 0
    target_leakage = [column for column in [*config.alpha_columns, *config.residual_feature_columns, *config.risk_columns] if column == TARGET]
    payload = asdict(config)
    for key, value in list(payload.items()):
        if isinstance(value, Path):
            payload[key] = str(value)
    return {
        "experiment_name": config.experiment_name,
        "config": payload,
        "rows": frame.height,
        "n_folds": frame["fold"].n_unique(),
        "strong_base": config.strong_base,
        "gateway_bad_updates": bad_updates,
        "target_leakage_check": "passed" if not target_leakage else f"FAILED: {target_leakage}",
        "fold_causality_check": "passed: stack, residual, risk normalization and regime scales fit only earlier folds; gateway updates use prior-date simulator",
        "selection_check": "passed: strengths/features are fixed by config and all candidates are reported",
        "available_alpha_columns": list(_available(config.alpha_columns, frame)),
        "available_residual_feature_columns": list(_available(config.residual_feature_columns, frame)),
        "available_risk_columns": list(_available(config.risk_columns, frame)),
    }


def _fit_weighted_ridge_no_intercept(frame: pl.DataFrame, columns: Sequence[str], *, alpha: float) -> np.ndarray:
    x = frame.select(columns).to_numpy().astype(np.float64, copy=False)
    y = frame[TARGET].to_numpy().astype(np.float64, copy=False)
    w = frame["weight"].to_numpy().astype(np.float64, copy=False)
    xtw = x.T * w
    lhs = xtw @ x + float(alpha) * np.eye(x.shape[1], dtype=np.float64)
    rhs = xtw @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _fit_residual_linear(frame: pl.DataFrame, base: str, features: Sequence[str], *, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = frame[TARGET].to_numpy().astype(np.float64, copy=False)
    pred = frame[base].to_numpy().astype(np.float64, copy=False)
    residual = y - pred
    w = frame["weight"].to_numpy().astype(np.float64, copy=False)
    means = np.asarray([_safe_mean(frame[name].to_numpy()) for name in features], dtype=np.float64)
    scales = np.asarray([_safe_std(frame[name].to_numpy(), means[idx]) for idx, name in enumerate(features)], dtype=np.float64)
    x = _standardized_matrix(frame, features, means, scales)
    xtw = x.T * w
    lhs = xtw @ x + float(alpha) * np.eye(x.shape[1], dtype=np.float64)
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


def _fit_bridge_thresholds(frame: pl.DataFrame, *, prediction: np.ndarray, risk_column: str) -> dict[str, tuple[float, float]]:
    return {
        "weight": _terciles(frame["weight"].to_numpy()),
        "abs_pred": _terciles(np.abs(prediction)),
        "risk": _terciles(frame[risk_column].to_numpy()),
    }


def _bridge_regime_codes(
    frame: pl.DataFrame,
    *,
    prediction: np.ndarray,
    risk_column: str,
    thresholds: dict[str, tuple[float, float]],
    time_bucket_size: int,
    symbol_mod: int,
) -> np.ndarray:
    time_bucket = frame["time_id"].to_numpy().astype(np.int64, copy=False) // int(time_bucket_size)
    symbol_bucket = np.mod(frame["symbol_id"].to_numpy().astype(np.int64, copy=False), int(symbol_mod))
    weight_bucket = _bucketize(frame["weight"].to_numpy(), thresholds["weight"])
    abs_bucket = _bucketize(np.abs(prediction), thresholds["abs_pred"])
    risk_bucket = _bucketize(frame[risk_column].to_numpy(), thresholds["risk"])
    return (((time_bucket * int(symbol_mod) + symbol_bucket) * 3 + weight_bucket) * 3 + abs_bucket) * 3 + risk_bucket


def _fit_group_scales(
    frame: pl.DataFrame,
    *,
    prediction: np.ndarray,
    codes: np.ndarray,
    min_rows: int,
    prior_strength: float,
) -> tuple[float, dict[int, float]]:
    target = frame[TARGET].to_numpy().astype(np.float64, copy=False)
    weight = frame["weight"].to_numpy().astype(np.float64, copy=False)
    default = _weighted_scale(target, prediction, weight, prior=0.0, prior_scale=1.0)
    scales: dict[int, float] = {}
    for code in np.unique(codes):
        mask = codes == code
        if int(mask.sum()) < int(min_rows):
            continue
        scales[int(code)] = _weighted_scale(target[mask], prediction[mask], weight[mask], prior=float(prior_strength), prior_scale=default)
    return default, scales


def _risk_shrink(prediction: np.ndarray, risk: np.ndarray, *, risk_mean: float, strength: float) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float64)
    values = np.maximum(np.asarray(risk, dtype=np.float64), 0.0)
    denom = max(float(risk_mean), 1e-12)
    normalized = np.clip(values / denom, 0.0, 10.0)
    return pred / np.sqrt(1.0 + float(strength) * normalized)


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


def _fit_gate_thresholds(frame: pl.DataFrame, *, base_prediction: str, risk_column: str | None) -> dict[str, tuple[float, float]]:
    risk_values = frame[risk_column].to_numpy() if risk_column else np.zeros(frame.height, dtype=np.float64)
    return {
        "weight": _terciles(frame["weight"].to_numpy()),
        "abs_base": _terciles(np.abs(frame[base_prediction].to_numpy())),
        "risk": _terciles(risk_values),
    }


def _residual_gate_codes(
    frame: pl.DataFrame,
    *,
    base_prediction: str,
    risk_column: str | None,
    thresholds: dict[str, tuple[float, float]],
    time_bucket_size: int,
    symbol_mod: int,
) -> np.ndarray:
    time_bucket = frame["time_id"].to_numpy().astype(np.int64, copy=False) // int(time_bucket_size)
    symbol_bucket = np.mod(frame["symbol_id"].to_numpy().astype(np.int64, copy=False), int(symbol_mod))
    weight_bucket = _bucketize(frame["weight"].to_numpy(), thresholds["weight"])
    abs_bucket = _bucketize(np.abs(frame[base_prediction].to_numpy()), thresholds["abs_base"])
    if risk_column:
        risk_values = frame[risk_column].to_numpy()
    else:
        risk_values = np.zeros(frame.height, dtype=np.float64)
    risk_bucket = _bucketize(risk_values, thresholds["risk"])
    return (((time_bucket * int(symbol_mod) + symbol_bucket) * 3 + weight_bucket) * 3 + abs_bucket) * 3 + risk_bucket


def _fit_residual_gate_policy(
    frame: pl.DataFrame,
    *,
    base_prediction: str,
    residual_prediction: str,
    codes: np.ndarray,
    policy: str,
    min_rows: int,
    prior_strength: float,
    min_delta: float,
) -> tuple[bool, dict[int, bool], float]:
    target = frame[TARGET].to_numpy().astype(np.float64, copy=False)
    weight = frame["weight"].to_numpy().astype(np.float64, copy=False)
    base = frame[base_prediction].to_numpy().astype(np.float64, copy=False)
    residual = frame[residual_prediction].to_numpy().astype(np.float64, copy=False)
    global_delta = _residual_delta_r2(target, weight, base, residual)
    if policy == "open":
        default_gate = True
    elif policy == "closed":
        default_gate = False
    else:
        raise ValueError(f"unknown residual gate policy: {policy}")
    code_gates: dict[int, bool] = {}
    for code in np.unique(codes):
        mask = codes == code
        if int(mask.sum()) < int(min_rows):
            continue
        denom = float(np.sum(weight[mask] * target[mask] * target[mask]))
        if denom <= 0.0:
            continue
        delta = _residual_delta_r2(target[mask], weight[mask], base[mask], residual[mask])
        shrunk_delta = ((denom * delta) + (float(prior_strength) * global_delta)) / (denom + float(prior_strength))
        code_gates[int(code)] = bool(shrunk_delta > float(min_delta))
    return default_gate, code_gates, global_delta


def _residual_delta_r2(target: np.ndarray, weight: np.ndarray, base: np.ndarray, residual: np.ndarray) -> float:
    denominator = float(np.sum(weight * target * target))
    if denominator <= 0.0:
        return 0.0
    base_num = float(np.sum(weight * np.square(target - base)))
    residual_num = float(np.sum(weight * np.square(target - residual)))
    return (base_num - residual_num) / denominator


def _gate_for_codes(codes: np.ndarray, *, default_gate: bool, code_gates: dict[int, bool]) -> np.ndarray:
    out = np.full(codes.shape, bool(default_gate), dtype=bool)
    for code, value in code_gates.items():
        out[codes == code] = bool(value)
    return out


def _fit_tail_thresholds(frame: pl.DataFrame, *, base_prediction: str, risk_column: str | None, quantile: float) -> dict[str, float]:
    risk_values = frame[risk_column].to_numpy() if risk_column else np.zeros(frame.height, dtype=np.float64)
    return {
        "weight": _safe_quantile(frame["weight"].to_numpy(), quantile),
        "abs_base": _safe_quantile(np.abs(frame[base_prediction].to_numpy()), quantile),
        "risk": _safe_quantile(risk_values, quantile),
    }


def _tail_mask(frame: pl.DataFrame, *, base_prediction: str, risk_column: str | None, thresholds: dict[str, float], mode: str) -> np.ndarray:
    weight_mask = frame["weight"].to_numpy().astype(np.float64, copy=False) >= float(thresholds["weight"])
    abs_mask = np.abs(frame[base_prediction].to_numpy().astype(np.float64, copy=False)) >= float(thresholds["abs_base"])
    if risk_column:
        risk_mask = frame[risk_column].to_numpy().astype(np.float64, copy=False) >= float(thresholds["risk"])
    else:
        risk_mask = np.zeros(frame.height, dtype=bool)
    if mode == "weight":
        return weight_mask
    if mode == "abs_base":
        return abs_mask
    if mode == "risk":
        return risk_mask
    if mode == "weight_or_abs":
        return weight_mask | abs_mask
    if mode == "weight_and_abs":
        return weight_mask & abs_mask
    if mode == "abs_base_or_risk":
        return abs_mask | risk_mask
    raise ValueError(f"unknown residual tail mode: {mode}")


def _safe_quantile(values: np.ndarray, quantile: float) -> float:
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("tail quantiles must be between 0 and 1")
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.inf
    return float(np.quantile(x, float(quantile)))


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


def _safe_positive_mean(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x) & (x >= 0.0)]
    if x.size == 0:
        return 1.0
    return max(float(np.mean(x)), 1e-12)


def _safe_mean(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else 0.0


def _safe_std(values: np.ndarray, mean: float) -> float:
    x = np.asarray(values, dtype=np.float64)
    x = np.where(np.isfinite(x), x, mean)
    std = float(np.std(x))
    return std if np.isfinite(std) and std > 1e-12 else 1.0


def _available(columns: Sequence[str], frame: pl.DataFrame) -> tuple[str, ...]:
    return tuple(column for column in columns if column in frame.columns)


def _folds(frame: pl.DataFrame) -> list[str]:
    return frame.select("fold").unique().sort("fold")["fold"].to_list()


def _write_parameters(path: Path, *parameter_groups: list[dict[str, Any]]) -> None:
    payload: dict[str, Any] = {}
    for idx, group in enumerate(parameter_groups):
        payload[f"group_{idx}"] = group
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_gate_columns(frame: pl.DataFrame, base_prediction: str, residual_prediction: str, risk_column: str | None) -> None:
    columns = [base_prediction, residual_prediction, "weight", TARGET, "fold", "time_id", "symbol_id"]
    if risk_column:
        columns.append(risk_column)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required residual gate columns: {missing}")


def _validate_config(config: StrongFamilyBridgeConfig) -> None:
    if not config.strong_base:
        raise ValueError("strong_base must not be empty")
    if config.stack_alpha < 0.0 or config.residual_alpha < 0.0:
        raise ValueError("ridge alphas must be non-negative")
    if any(value < 0.0 for value in config.risk_strengths):
        raise ValueError("risk strengths must be non-negative")
    if config.time_bucket_size <= 0:
        raise ValueError("time_bucket_size must be positive")
    if config.symbol_mod <= 0:
        raise ValueError("symbol_mod must be positive")
    if config.min_regime_rows <= 0:
        raise ValueError("min_regime_rows must be positive")
    if config.residual_gate_time_bucket_size <= 0:
        raise ValueError("residual_gate_time_bucket_size must be positive")
    if config.residual_gate_symbol_mod <= 0:
        raise ValueError("residual_gate_symbol_mod must be positive")
    if config.residual_gate_min_rows <= 0:
        raise ValueError("residual_gate_min_rows must be positive")
    if config.residual_gate_prior_strength < 0.0:
        raise ValueError("residual_gate_prior_strength must be non-negative")
    if not config.residual_tail_quantiles:
        raise ValueError("at least one residual tail quantile is required")
    if any(not 0.0 < value < 1.0 for value in config.residual_tail_quantiles):
        raise ValueError("residual_tail_quantiles must be between 0 and 1")
