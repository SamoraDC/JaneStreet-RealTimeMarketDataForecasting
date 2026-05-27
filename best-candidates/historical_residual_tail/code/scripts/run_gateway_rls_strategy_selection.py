"""Causal strategy selection over existing gateway RLS meta candidates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from janestreet.bayesian_meta import score_prediction_by_fold, summarize_fold_scores, weighted_normal_stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "responder_6"
KEYS = ["fold", "date_id", "time_id", "symbol_id"]


@dataclass(frozen=True)
class RLSStrategy:
    name: str
    prediction: str
    feature_set: str
    ridge_alpha: float
    forgetting_factor: float
    role: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tabm-prediction-dir",
        type=Path,
        default=Path("reports/experiments/competitive_tabm_official_stage3_5fold_valid60_lags_online_lr1e4_4m_train700_seed37_aux8_preds/validation_predictions"),
    )
    parser.add_argument(
        "--tree-prediction-dir",
        type=Path,
        default=Path("reports/experiments/tree_engine_ensemble_xgb_lgb_sample10_seed_ensemble_preds/validation_predictions"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/gateway_rls_strategy_selection_stage3"))
    parser.add_argument("--experiment-name", default="gateway_rls_strategy_selection")
    parser.add_argument("--ewma-decays", default="0.90,0.95,0.99")
    parser.add_argument("--softmax-etas", default="100,500,1000")
    parser.add_argument("--scale-alphas", default="1000,10000")
    parser.add_argument("--scale-forgetting-factors", default="1.0,0.995")
    parser.add_argument("--posterior-shrink-strengths", default="100,1000,10000,100000")
    parser.add_argument("--risk-modulated-shrink-strengths", default="")
    parser.add_argument("--risk-modulation-profiles", default="disagreement,abs_pred,weight,all")
    args = parser.parse_args()

    gateway = _load_gateway_module()
    dynamic = _load_dynamic_module()
    frame = gateway._load_joined_predictions(args.tabm_prediction_dir, args.tree_prediction_dir)
    frame, convex_parameters = gateway._add_walk_forward_convex_baseline(frame)
    expert_columns = tuple(column for column in gateway._expert_columns(frame) if column != "baseline_prediction")
    feature_sets = gateway._gateway_feature_sets(expert_columns)
    rls_strategies = _rls_strategies()
    _validate_rls_strategies(rls_strategies, feature_sets)

    prediction_frame = frame.select(KEYS + ["weight", TARGET, "baseline_prediction"])
    audit_frames: list[pl.DataFrame] = []
    for strategy in rls_strategies:
        strategy_predictions, audit = _predict_dynamic_rls_strategy(
            gateway,
            dynamic,
            frame,
            feature_columns=feature_sets[strategy.feature_set],
            ridge_alpha=strategy.ridge_alpha,
            forgetting_factor=strategy.forgetting_factor,
            output=strategy.prediction,
        )
        prediction_frame = prediction_frame.join(strategy_predictions, on=KEYS, how="inner")
        audit_frames.append(audit.with_columns(pl.lit(strategy.name).alias("strategy")))

    selector_strategies = _selector_strategies()
    by_fold_frames: list[pl.DataFrame] = []
    summary_rows: list[dict[str, float | int | str | bool]] = []
    selector_audit_frames: list[pl.DataFrame] = []
    choice_frames: list[pl.DataFrame] = []

    for name, prediction in selector_strategies:
        scores = _score_strategy(prediction_frame, strategy=name, method_family="base_candidate", prediction=prediction)
        by_fold_frames.append(scores)
        summary_rows.append({"strategy": name, "method_family": "base_candidate", "is_promotable": True, **summarize_fold_scores(scores)})

    posterior_strengths = _parse_float_list(args.posterior_shrink_strengths)
    for strategy in rls_strategies:
        posterior_frame, posterior_audit = _predict_dynamic_rls_posterior_shrinkage_strategy(
            gateway,
            dynamic,
            frame,
            feature_columns=feature_sets[strategy.feature_set],
            ridge_alpha=strategy.ridge_alpha,
            forgetting_factor=strategy.forgetting_factor,
            strengths=posterior_strengths,
            output_prefix=f"posterior_shrink_{strategy.name}",
        )
        posterior_scoring_frame = prediction_frame.select(KEYS + ["weight", TARGET]).join(posterior_frame, on=KEYS, how="inner")
        for strength in posterior_strengths:
            strength_suffix = _format_float(strength)
            prediction = f"posterior_shrink_{strategy.name}_s{strength_suffix}_prediction"
            strategy_name = f"posterior_shrink_{strategy.name}_s{strength_suffix}"
            scores = _score_strategy(
                posterior_scoring_frame,
                strategy=strategy_name,
                method_family="posterior_uncertainty_shrinkage",
                prediction=prediction,
            )
            by_fold_frames.append(scores)
            summary_rows.append(
                {
                    "strategy": strategy_name,
                    "method_family": "posterior_uncertainty_shrinkage",
                    "is_promotable": True,
                    **summarize_fold_scores(scores),
                }
            )
        selector_audit_frames.append(posterior_audit.with_columns(pl.lit(f"posterior_shrink_{strategy.name}").alias("strategy")))

    risk_strengths = _parse_float_list(args.risk_modulated_shrink_strengths)
    risk_profiles = _parse_risk_profiles(args.risk_modulation_profiles)
    if risk_strengths:
        for strategy in rls_strategies:
            risk_frame, risk_audit = _predict_dynamic_rls_risk_modulated_shrinkage_strategy(
                gateway,
                dynamic,
                frame,
                feature_columns=feature_sets[strategy.feature_set],
                ridge_alpha=strategy.ridge_alpha,
                forgetting_factor=strategy.forgetting_factor,
                strengths=risk_strengths,
                profiles=risk_profiles,
                output_prefix=f"risk_shrink_{strategy.name}",
            )
            risk_scoring_frame = prediction_frame.select(KEYS + ["weight", TARGET]).join(risk_frame, on=KEYS, how="inner")
            for profile in risk_profiles:
                for strength in risk_strengths:
                    prediction = f"risk_shrink_{strategy.name}_{profile}_s{_format_float(strength)}_prediction"
                    strategy_name = f"risk_shrink_{strategy.name}_{profile}_s{_format_float(strength)}"
                    scores = _score_strategy(
                        risk_scoring_frame,
                        strategy=strategy_name,
                        method_family="risk_modulated_posterior_shrinkage",
                        prediction=prediction,
                    )
                    by_fold_frames.append(scores)
                    summary_rows.append(
                        {
                            "strategy": strategy_name,
                            "method_family": "risk_modulated_posterior_shrinkage",
                            "is_promotable": True,
                            **summarize_fold_scores(scores),
                        }
                    )
            selector_audit_frames.append(risk_audit.with_columns(pl.lit(f"risk_shrink_{strategy.name}").alias("strategy")))

    for base_name, prediction in selector_strategies:
        for scale_alpha in _parse_float_list(args.scale_alphas):
            for scale_forgetting in _parse_float_list(args.scale_forgetting_factors):
                output = f"scale_calibrated_{base_name}_a{_format_float(scale_alpha)}_f{_format_float(scale_forgetting)}_prediction"
                calibrated, audit = apply_dynamic_scale_calibrator(
                    prediction_frame,
                    prediction=prediction,
                    alpha=scale_alpha,
                    forgetting_factor=scale_forgetting,
                    output=output,
                )
                strategy_name = f"scale_calibrated_{base_name}_a{_format_float(scale_alpha)}_f{_format_float(scale_forgetting)}"
                scores = _score_strategy(calibrated, strategy=strategy_name, method_family="dynamic_scale_calibrator", prediction=output)
                by_fold_frames.append(scores)
                summary_rows.append(
                    {"strategy": strategy_name, "method_family": "dynamic_scale_calibrator", "is_promotable": True, **summarize_fold_scores(scores)}
                )
                selector_audit_frames.append(audit.with_columns(pl.lit(strategy_name).alias("strategy")))

    static_frame, static_audit = apply_static_previous_fold_selector(
        prediction_frame,
        strategies=selector_strategies,
        default_strategy="conservative_rls",
        output="static_previous_fold_best_prediction",
    )
    scores = _score_strategy(static_frame, strategy="static_previous_fold_best", method_family="causal_selector", prediction="static_previous_fold_best_prediction")
    by_fold_frames.append(scores)
    summary_rows.append({"strategy": "static_previous_fold_best", "method_family": "causal_selector", "is_promotable": True, **summarize_fold_scores(scores)})
    selector_audit_frames.append(static_audit.with_columns(pl.lit("static_previous_fold_best").alias("strategy")))
    choice_frames.append(_choice_counts(static_frame, strategy="static_previous_fold_best", choice_column="_selected_strategy"))

    for decay in _parse_float_list(args.ewma_decays):
        output = f"online_ewma_best_d{_format_float(decay)}_prediction"
        selected, audit = apply_online_loss_selector(
            prediction_frame,
            strategies=selector_strategies,
            default_strategy="conservative_rls",
            ewma_decay=decay,
            output=output,
        )
        strategy_name = f"online_ewma_best_d{_format_float(decay)}"
        scores = _score_strategy(selected, strategy=strategy_name, method_family="causal_selector", prediction=output)
        by_fold_frames.append(scores)
        summary_rows.append({"strategy": strategy_name, "method_family": "causal_selector", "is_promotable": True, **summarize_fold_scores(scores)})
        selector_audit_frames.append(audit.with_columns(pl.lit(strategy_name).alias("strategy")))
        choice_frames.append(_choice_counts(selected, strategy=strategy_name, choice_column="_selected_strategy"))

    for eta in _parse_float_list(args.softmax_etas):
        output = f"online_ewma_softmax_eta{_format_float(eta)}_prediction"
        selected, audit = apply_online_softmax_loss_blend(
            prediction_frame,
            strategies=selector_strategies,
            default_strategy="conservative_rls",
            ewma_decay=0.95,
            eta=eta,
            output=output,
        )
        strategy_name = f"online_ewma_softmax_eta{_format_float(eta)}"
        scores = _score_strategy(selected, strategy=strategy_name, method_family="causal_selector", prediction=output)
        by_fold_frames.append(scores)
        summary_rows.append({"strategy": strategy_name, "method_family": "causal_selector", "is_promotable": True, **summarize_fold_scores(scores)})
        selector_audit_frames.append(audit.with_columns(pl.lit(strategy_name).alias("strategy")))
        choice_frames.append(_choice_counts(selected, strategy=strategy_name, choice_column="_selected_strategy"))

    oracle = apply_daily_oracle_selector(
        prediction_frame,
        strategies=selector_strategies,
        output="daily_oracle_prediction",
    )
    scores = _score_strategy(oracle, strategy="daily_oracle_upper_bound", method_family="diagnostic_oracle", prediction="daily_oracle_prediction")
    by_fold_frames.append(scores)
    summary_rows.append({"strategy": "daily_oracle_upper_bound", "method_family": "diagnostic_oracle", "is_promotable": False, **summarize_fold_scores(scores)})
    choice_frames.append(_choice_counts(oracle, strategy="daily_oracle_upper_bound", choice_column="_selected_strategy"))

    by_fold = pl.concat(by_fold_frames, how="diagonal").select(
        ["strategy", "method_family", "fold", "rows", "weight_sum", "numerator", "denominator", "weighted_zero_mean_r2"]
    )
    summary = pl.DataFrame(summary_rows).sort(["is_promotable", "global_r2", "min_fold_r2"], descending=[True, True, True])
    candidate_audit = pl.concat(audit_frames, how="diagonal") if audit_frames else pl.DataFrame()
    selector_audit = pl.concat(selector_audit_frames, how="diagonal") if selector_audit_frames else pl.DataFrame()
    choices = pl.concat(choice_frames, how="diagonal") if choice_frames else pl.DataFrame()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.output_dir / "gateway_rls_strategy_selection_summary.csv")
    by_fold.write_csv(args.output_dir / "gateway_rls_strategy_selection_by_fold.csv")
    if not candidate_audit.is_empty():
        candidate_audit.write_csv(args.output_dir / "gateway_rls_candidate_daily_audit.csv")
    if not selector_audit.is_empty():
        selector_audit.write_csv(args.output_dir / "gateway_rls_selector_daily_audit.csv")
    if not choices.is_empty():
        choices.write_csv(args.output_dir / "gateway_rls_selector_choice_counts.csv")
    report = {
        "experiment": args.experiment_name,
        "rows": prediction_frame.height,
        "folds": prediction_frame["fold"].n_unique(),
        "base_candidate_scores": {
            row["strategy"]: row["global_r2"]
            for row in summary.filter(pl.col("method_family") == "base_candidate").iter_rows(named=True)
        },
        "best_promotable": summary.filter(pl.col("is_promotable")).row(0, named=True),
        "oracle_note": "daily_oracle_upper_bound uses same-day target to choose the strategy and is diagnostic only.",
        "rls_strategies": [asdict(strategy) for strategy in rls_strategies],
        "posterior_shrink_strengths": posterior_strengths,
        "risk_modulated_shrink_strengths": risk_strengths,
        "risk_modulation_profiles": risk_profiles,
        "convex_parameters": convex_parameters,
        "candidate_audit_status": _audit_status(candidate_audit),
        "selector_audit_status": _audit_status(selector_audit),
    }
    (args.output_dir / "gateway_rls_strategy_selection_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summary)
    print(f"Wrote {args.output_dir}")


def _load_gateway_module():
    path = PROJECT_ROOT / "scripts" / "run_bayesian_gateway_meta_simulation.py"
    spec = importlib.util.spec_from_file_location("run_bayesian_gateway_meta_simulation_for_strategy_selection", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_dynamic_module():
    path = PROJECT_ROOT / "scripts" / "run_dynamic_gateway_rls_validation.py"
    spec = importlib.util.spec_from_file_location("run_dynamic_gateway_rls_validation_for_strategy_selection", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rls_strategies() -> tuple[RLSStrategy, ...]:
    return (
        RLSStrategy(
            name="conservative_rls",
            prediction="conservative_rls_prediction",
            feature_set="experts",
            ridge_alpha=10000.0,
            forgetting_factor=0.995,
            role="preserved robustness reference",
        ),
        RLSStrategy(
            name="aggressive_rls",
            prediction="aggressive_rls_prediction",
            feature_set="components_no_tree_ensemble",
            ridge_alpha=1000.0,
            forgetting_factor=0.995,
            role="best Stage 3 local score",
        ),
    )


def _selector_strategies() -> tuple[tuple[str, str], ...]:
    return (
        ("conservative_rls", "conservative_rls_prediction"),
        ("aggressive_rls", "aggressive_rls_prediction"),
        ("tabm_tree_convex_walk_forward", "baseline_prediction"),
    )


def _validate_rls_strategies(strategies: Sequence[RLSStrategy], feature_sets: dict[str, tuple[str, ...]]) -> None:
    missing = [strategy.feature_set for strategy in strategies if strategy.feature_set not in feature_sets]
    if missing:
        raise ValueError(f"missing feature sets for RLS strategies: {missing}")


def _predict_dynamic_rls_strategy(
    gateway,
    dynamic,
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    ridge_alpha: float,
    forgetting_factor: float,
    output: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    folds = gateway._folds(frame)
    features = tuple(feature_columns)
    prior_beta = gateway._prior_beta(features)
    prediction_frames: list[pl.DataFrame] = []
    audit_frames: list[pl.DataFrame] = []
    for idx, fold in enumerate(folds):
        precision = ridge_alpha * np.eye(len(features), dtype=np.float64)
        rhs = ridge_alpha * prior_beta
        if idx > 0:
            calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
            gram, cal_rhs = weighted_normal_stats(calibration, feature_columns=features)
            precision += gram
            rhs += cal_rhs
        current = frame.filter(pl.col("fold") == fold)
        predictions, audit = _simulate_dynamic_rls_predictions(
            gateway,
            dynamic,
            current,
            features=features,
            precision=precision,
            rhs=rhs,
            forgetting_factor=forgetting_factor,
            output=output,
        )
        prediction_frames.append(predictions)
        audit_frames.append(audit.with_columns(pl.lit(fold).alias("fold")))
    return pl.concat(prediction_frames), pl.concat(audit_frames, how="diagonal")


def _simulate_dynamic_rls_predictions(
    gateway,
    dynamic,
    frame: pl.DataFrame,
    *,
    features: Sequence[str],
    precision: np.ndarray,
    rhs: np.ndarray,
    forgetting_factor: float,
    output: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    beta = np.linalg.solve(precision, rhs)
    pending_update: pl.DataFrame | None = None
    prediction_frames: list[pl.DataFrame] = []
    audit_rows: list[dict[str, int | float | bool | None]] = []
    for day in gateway._daily_frames(frame):
        current_date = int(day["date_id"][0])
        update_source_date: int | None = None
        update_rows = 0
        if pending_update is not None:
            lag_delivery = gateway._deliver_previous_day_lags(pending_update, current_date=current_date)
            update_source_date = int(lag_delivery["date_id"][0])
            update_rows = lag_delivery.height
            precision, rhs = dynamic._forgetting_update(
                precision,
                rhs,
                lag_delivery,
                features=features,
                forgetting_factor=forgetting_factor,
            )
        beta = np.linalg.solve(precision, rhs)
        x = day.select(list(features)).to_numpy()
        pred = x @ beta
        prediction_frames.append(day.select(KEYS).with_columns(pl.Series(output, pred)))
        pending_update = day.select(["date_id", "time_id", "symbol_id", TARGET, "weight"] + list(features))
        audit_rows.append(
            {
                "date_id": current_date,
                "predicted_rows": day.height,
                "update_source_date_id": update_source_date,
                "update_rows": update_rows,
                "forgetting_factor": forgetting_factor,
                "update_before_prediction": True,
                "update_is_strictly_past": update_source_date is None or update_source_date < current_date,
            }
        )
    return pl.concat(prediction_frames), pl.DataFrame(audit_rows)


def _predict_dynamic_rls_posterior_shrinkage_strategy(
    gateway,
    dynamic,
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    ridge_alpha: float,
    forgetting_factor: float,
    strengths: Sequence[float],
    output_prefix: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    _validate_posterior_strengths(strengths)
    folds = gateway._folds(frame)
    features = tuple(feature_columns)
    prior_beta = gateway._prior_beta(features)
    prediction_frames: list[pl.DataFrame] = []
    audit_frames: list[pl.DataFrame] = []
    for idx, fold in enumerate(folds):
        precision = ridge_alpha * np.eye(len(features), dtype=np.float64)
        rhs = ridge_alpha * prior_beta
        if idx > 0:
            calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
            gram, cal_rhs = weighted_normal_stats(calibration, feature_columns=features)
            precision += gram
            rhs += cal_rhs
        current = frame.filter(pl.col("fold") == fold)
        predictions, audit = _simulate_dynamic_rls_posterior_shrinkage(
            gateway,
            dynamic,
            current,
            features=features,
            precision=precision,
            rhs=rhs,
            forgetting_factor=forgetting_factor,
            strengths=strengths,
            output_prefix=output_prefix,
        )
        prediction_frames.append(predictions)
        audit_frames.append(audit.with_columns(pl.lit(fold).alias("fold")))
    return pl.concat(prediction_frames), pl.concat(audit_frames, how="diagonal")


def _predict_dynamic_rls_risk_modulated_shrinkage_strategy(
    gateway,
    dynamic,
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    ridge_alpha: float,
    forgetting_factor: float,
    strengths: Sequence[float],
    profiles: Sequence[str],
    output_prefix: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    _validate_posterior_strengths(strengths)
    _validate_risk_profiles(profiles)
    folds = gateway._folds(frame)
    features = tuple(feature_columns)
    prior_beta = gateway._prior_beta(features)
    prediction_frames: list[pl.DataFrame] = []
    audit_frames: list[pl.DataFrame] = []
    for idx, fold in enumerate(folds):
        precision = ridge_alpha * np.eye(len(features), dtype=np.float64)
        rhs = ridge_alpha * prior_beta
        if idx > 0:
            calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
            gram, cal_rhs = weighted_normal_stats(calibration, feature_columns=features)
            precision += gram
            rhs += cal_rhs
        current = frame.filter(pl.col("fold") == fold)
        predictions, audit = _simulate_dynamic_rls_risk_modulated_shrinkage(
            gateway,
            dynamic,
            current,
            features=features,
            precision=precision,
            rhs=rhs,
            forgetting_factor=forgetting_factor,
            strengths=strengths,
            profiles=profiles,
            output_prefix=output_prefix,
        )
        prediction_frames.append(predictions)
        audit_frames.append(audit.with_columns(pl.lit(fold).alias("fold")))
    return pl.concat(prediction_frames), pl.concat(audit_frames, how="diagonal")


def _simulate_dynamic_rls_posterior_shrinkage(
    gateway,
    dynamic,
    frame: pl.DataFrame,
    *,
    features: Sequence[str],
    precision: np.ndarray,
    rhs: np.ndarray,
    forgetting_factor: float,
    strengths: Sequence[float],
    output_prefix: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    pending_update: pl.DataFrame | None = None
    prediction_frames: list[pl.DataFrame] = []
    audit_rows: list[dict[str, int | float | bool | None]] = []
    for day in gateway._daily_frames(frame):
        current_date = int(day["date_id"][0])
        update_source_date: int | None = None
        update_rows = 0
        if pending_update is not None:
            lag_delivery = gateway._deliver_previous_day_lags(pending_update, current_date=current_date)
            update_source_date = int(lag_delivery["date_id"][0])
            update_rows = lag_delivery.height
            precision, rhs = dynamic._forgetting_update(
                precision,
                rhs,
                lag_delivery,
                features=features,
                forgetting_factor=forgetting_factor,
            )
        beta = np.linalg.solve(precision, rhs)
        covariance = np.linalg.inv(precision)
        x = day.select(list(features)).to_numpy()
        pred = x @ beta
        leverage = np.einsum("ij,jk,ik->i", x, covariance, x, optimize=True)
        leverage = np.maximum(leverage, 0.0)
        columns = []
        for strength in strengths:
            shrink = 1.0 / np.sqrt(1.0 + float(strength) * leverage)
            columns.append(pl.Series(f"{output_prefix}_s{_format_float(strength)}_prediction", pred * shrink))
        prediction_frames.append(day.select(KEYS).with_columns(columns))
        pending_update = day.select(["date_id", "time_id", "symbol_id", TARGET, "weight"] + list(features))
        audit_rows.append(
            {
                "date_id": current_date,
                "predicted_rows": day.height,
                "update_source_date_id": update_source_date,
                "update_rows": update_rows,
                "forgetting_factor": forgetting_factor,
                "posterior_leverage_mean": float(np.mean(leverage)) if leverage.size else 0.0,
                "posterior_leverage_max": float(np.max(leverage)) if leverage.size else 0.0,
                "update_before_prediction": True,
                "update_is_strictly_past": update_source_date is None or update_source_date < current_date,
            }
        )
    return pl.concat(prediction_frames), pl.DataFrame(audit_rows)


def _simulate_dynamic_rls_risk_modulated_shrinkage(
    gateway,
    dynamic,
    frame: pl.DataFrame,
    *,
    features: Sequence[str],
    precision: np.ndarray,
    rhs: np.ndarray,
    forgetting_factor: float,
    strengths: Sequence[float],
    profiles: Sequence[str],
    output_prefix: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    pending_update: pl.DataFrame | None = None
    prediction_frames: list[pl.DataFrame] = []
    audit_rows: list[dict[str, int | float | bool | None]] = []
    for day in gateway._daily_frames(frame):
        current_date = int(day["date_id"][0])
        update_source_date: int | None = None
        update_rows = 0
        if pending_update is not None:
            lag_delivery = gateway._deliver_previous_day_lags(pending_update, current_date=current_date)
            update_source_date = int(lag_delivery["date_id"][0])
            update_rows = lag_delivery.height
            precision, rhs = dynamic._forgetting_update(
                precision,
                rhs,
                lag_delivery,
                features=features,
                forgetting_factor=forgetting_factor,
            )
        beta = np.linalg.solve(precision, rhs)
        covariance = np.linalg.inv(precision)
        arrays = day.select(list(features) + ["weight"]).to_numpy()
        x = arrays[:, : len(features)]
        sample_weight = arrays[:, len(features)]
        pred = x @ beta
        leverage = np.einsum("ij,jk,ik->i", x, covariance, x, optimize=True)
        leverage = np.maximum(leverage, 0.0)
        multipliers = _risk_multipliers(x=x, pred=pred, sample_weight=sample_weight, profiles=profiles)
        columns = []
        for profile in profiles:
            risk = leverage * multipliers[profile]
            for strength in strengths:
                shrink = 1.0 / np.sqrt(1.0 + float(strength) * risk)
                columns.append(pl.Series(f"{output_prefix}_{profile}_s{_format_float(strength)}_prediction", pred * shrink))
        prediction_frames.append(day.select(KEYS).with_columns(columns))
        pending_update = day.select(["date_id", "time_id", "symbol_id", TARGET, "weight"] + list(features))
        audit_rows.append(
            {
                "date_id": current_date,
                "predicted_rows": day.height,
                "update_source_date_id": update_source_date,
                "update_rows": update_rows,
                "forgetting_factor": forgetting_factor,
                "posterior_leverage_mean": float(np.mean(leverage)) if leverage.size else 0.0,
                "posterior_leverage_max": float(np.max(leverage)) if leverage.size else 0.0,
                "risk_disagreement_mean": float(np.mean(multipliers.get("disagreement", np.array([1.0])) - 1.0)),
                "risk_abs_pred_mean": float(np.mean(multipliers.get("abs_pred", np.array([1.0])) - 1.0)),
                "risk_weight_mean": float(np.mean(multipliers.get("weight", np.array([1.0])) - 1.0)),
                "update_before_prediction": True,
                "update_is_strictly_past": update_source_date is None or update_source_date < current_date,
            }
        )
    return pl.concat(prediction_frames), pl.DataFrame(audit_rows)


def _risk_multipliers(
    *,
    x: np.ndarray,
    pred: np.ndarray,
    sample_weight: np.ndarray,
    profiles: Sequence[str],
) -> dict[str, np.ndarray]:
    if x.ndim != 2:
        raise ValueError("x must be two-dimensional")
    feature_abs_mean = np.mean(np.abs(x), axis=1)
    feature_std = np.std(x, axis=1)
    rel_disagreement = np.clip(feature_std / np.maximum(feature_abs_mean, 1e-8), 0.0, 5.0)
    abs_prediction = np.clip(np.abs(pred), 0.0, 5.0)
    weight_risk = np.clip(np.log1p(np.maximum(sample_weight, 0.0)), 0.0, 5.0)
    base: dict[str, np.ndarray] = {
        "disagreement": 1.0 + rel_disagreement,
        "abs_pred": 1.0 + abs_prediction,
        "weight": 1.0 + weight_risk,
        "disagreement_weight": 1.0 + rel_disagreement + weight_risk,
        "all": 1.0 + rel_disagreement + abs_prediction + weight_risk,
    }
    return {profile: base[profile] for profile in profiles}


def apply_static_previous_fold_selector(
    frame: pl.DataFrame,
    *,
    strategies: Sequence[tuple[str, str]],
    default_strategy: str,
    output: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    _validate_strategy_inputs(strategies, default_strategy)
    folds = _folds(frame)
    frames: list[pl.DataFrame] = []
    audit_rows: list[dict[str, str | int | bool | None]] = []
    for idx, fold in enumerate(folds):
        if idx == 0:
            selected = default_strategy
            source_folds: tuple[str, ...] = ()
        else:
            source_folds = tuple(folds[:idx])
            calibration = frame.filter(pl.col("fold").is_in(source_folds))
            selected = _best_strategy_by_loss(calibration, strategies)
        prediction = _prediction_for_strategy(strategies, selected)
        current = frame.filter(pl.col("fold") == fold)
        frames.append(
            current.with_columns(
                pl.col(prediction).alias(output),
                pl.lit(selected).alias("_selected_strategy"),
            )
        )
        audit_rows.append(
            {
                "fold": fold,
                "date_id": -1,
                "selected_strategy": selected,
                "source_folds": ",".join(source_folds),
                "selected_before_current_target": True,
                "uses_current_day_target": False,
            }
        )
    return pl.concat(frames), pl.DataFrame(audit_rows)


def apply_online_loss_selector(
    frame: pl.DataFrame,
    *,
    strategies: Sequence[tuple[str, str]],
    default_strategy: str,
    ewma_decay: float,
    output: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    _validate_strategy_inputs(strategies, default_strategy)
    _validate_decay(ewma_decay)
    frames: list[pl.DataFrame] = []
    audit_rows: list[dict[str, str | float | int | bool | None]] = []
    for fold_idx, fold in enumerate(_folds(frame)):
        current_fold = frame.filter(pl.col("fold") == fold)
        loss_state = _initial_loss_state(frame, strategies, current_fold=fold, default_strategy=default_strategy)
        pending_day: pl.DataFrame | None = None
        for day in _daily_frames(current_fold):
            current_date = int(day["date_id"][0])
            update_source_date: int | None = None
            if pending_day is not None:
                update_source_date = int(pending_day["date_id"][0])
                day_losses = _strategy_loss_ratios(pending_day, strategies)
                for name, value in day_losses.items():
                    loss_state[name] = ewma_decay * loss_state[name] + (1.0 - ewma_decay) * value
            selected = min(loss_state, key=lambda name: (loss_state[name], _strategy_order(strategies, name)))
            prediction = _prediction_for_strategy(strategies, selected)
            frames.append(
                day.with_columns(
                    pl.col(prediction).alias(output),
                    pl.lit(selected).alias("_selected_strategy"),
                )
            )
            pending_day = day
            audit_rows.append(
                {
                    "fold": fold,
                    "fold_index": fold_idx,
                    "date_id": current_date,
                    "selected_strategy": selected,
                    "update_source_date_id": update_source_date,
                    "ewma_decay": ewma_decay,
                    "selected_before_current_target": True,
                    "uses_current_day_target": False,
                    "update_is_strictly_past": update_source_date is None or update_source_date < current_date,
                }
            )
    return pl.concat(frames), pl.DataFrame(audit_rows)


def apply_online_softmax_loss_blend(
    frame: pl.DataFrame,
    *,
    strategies: Sequence[tuple[str, str]],
    default_strategy: str,
    ewma_decay: float,
    eta: float,
    output: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    _validate_strategy_inputs(strategies, default_strategy)
    _validate_decay(ewma_decay)
    if eta <= 0.0:
        raise ValueError("eta must be positive")
    frames: list[pl.DataFrame] = []
    audit_rows: list[dict[str, str | float | int | bool | None]] = []
    names = [name for name, _prediction in strategies]
    for fold_idx, fold in enumerate(_folds(frame)):
        current_fold = frame.filter(pl.col("fold") == fold)
        loss_state = _initial_loss_state(frame, strategies, current_fold=fold, default_strategy=default_strategy)
        pending_day: pl.DataFrame | None = None
        for day in _daily_frames(current_fold):
            current_date = int(day["date_id"][0])
            update_source_date: int | None = None
            if pending_day is not None:
                update_source_date = int(pending_day["date_id"][0])
                day_losses = _strategy_loss_ratios(pending_day, strategies)
                for name, value in day_losses.items():
                    loss_state[name] = ewma_decay * loss_state[name] + (1.0 - ewma_decay) * value
            weights = _softmax_negative_losses(np.array([loss_state[name] for name in names], dtype=np.float64), eta=eta)
            arrays = day.select([prediction for _name, prediction in strategies]).to_numpy()
            pred = arrays @ weights
            selected = names[int(np.argmax(weights))]
            frames.append(
                day.with_columns(
                    pl.Series(output, pred),
                    pl.lit(selected).alias("_selected_strategy"),
                )
            )
            pending_day = day
            audit_rows.append(
                {
                    "fold": fold,
                    "fold_index": fold_idx,
                    "date_id": current_date,
                    "selected_strategy": selected,
                    "update_source_date_id": update_source_date,
                    "ewma_decay": ewma_decay,
                    "eta": eta,
                    "selected_before_current_target": True,
                    "uses_current_day_target": False,
                    "update_is_strictly_past": update_source_date is None or update_source_date < current_date,
                }
            )
    return pl.concat(frames), pl.DataFrame(audit_rows)


def apply_dynamic_scale_calibrator(
    frame: pl.DataFrame,
    *,
    prediction: str,
    alpha: float,
    forgetting_factor: float,
    output: str,
    prior_scale: float = 1.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if forgetting_factor <= 0.0 or forgetting_factor > 1.0:
        raise ValueError("forgetting_factor must be in (0, 1]")
    frames: list[pl.DataFrame] = []
    audit_rows: list[dict[str, str | float | int | bool | None]] = []
    folds = _folds(frame)
    for fold_idx, fold in enumerate(folds):
        precision = float(alpha)
        rhs = float(alpha * prior_scale)
        fit_rows = 0
        if fold_idx > 0:
            calibration = frame.filter(pl.col("fold").is_in(folds[:fold_idx]))
            gram, cal_rhs = _scale_stats(calibration, prediction=prediction)
            precision += gram
            rhs += cal_rhs
            fit_rows = calibration.height
        pending_day: pl.DataFrame | None = None
        current_fold = frame.filter(pl.col("fold") == fold)
        for day in _daily_frames(current_fold):
            current_date = int(day["date_id"][0])
            update_source_date: int | None = None
            update_rows = 0
            if pending_day is not None:
                update_source_date = int(pending_day["date_id"][0])
                update_rows = pending_day.height
                gram, cal_rhs = _scale_stats(pending_day, prediction=prediction)
                precision = forgetting_factor * precision + gram
                rhs = forgetting_factor * rhs + cal_rhs
            scale = prior_scale if precision <= 1e-12 else rhs / precision
            frames.append(day.with_columns((scale * pl.col(prediction)).alias(output)))
            pending_day = day
            audit_rows.append(
                {
                    "fold": fold,
                    "fold_index": fold_idx,
                    "date_id": current_date,
                    "selected_strategy": prediction,
                    "update_source_date_id": update_source_date,
                    "update_rows": update_rows,
                    "fit_rows": fit_rows,
                    "scale": scale,
                    "alpha": alpha,
                    "forgetting_factor": forgetting_factor,
                    "selected_before_current_target": True,
                    "uses_current_day_target": False,
                    "update_is_strictly_past": update_source_date is None or update_source_date < current_date,
                }
            )
    return pl.concat(frames), pl.DataFrame(audit_rows)


def apply_daily_oracle_selector(
    frame: pl.DataFrame,
    *,
    strategies: Sequence[tuple[str, str]],
    output: str,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for day in _daily_frames(frame):
        losses = _strategy_loss_ratios(day, strategies)
        selected = min(losses, key=lambda name: (losses[name], _strategy_order(strategies, name)))
        prediction = _prediction_for_strategy(strategies, selected)
        frames.append(
            day.with_columns(
                pl.col(prediction).alias(output),
                pl.lit(selected).alias("_selected_strategy"),
            )
        )
    return pl.concat(frames)


def _scale_stats(frame: pl.DataFrame, *, prediction: str) -> tuple[float, float]:
    row = frame.select(
        (pl.col("weight") * pl.col(prediction).pow(2)).sum().alias("gram"),
        (pl.col("weight") * pl.col(prediction) * pl.col(TARGET)).sum().alias("rhs"),
    ).row(0, named=True)
    return float(row["gram"]), float(row["rhs"])


def _initial_loss_state(
    frame: pl.DataFrame,
    strategies: Sequence[tuple[str, str]],
    *,
    current_fold: str,
    default_strategy: str,
) -> dict[str, float]:
    folds = _folds(frame)
    idx = folds.index(current_fold)
    if idx == 0:
        return {name: (0.0 if name == default_strategy else 1e-9) for name, _prediction in strategies}
    calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
    losses = _strategy_loss_ratios(calibration, strategies)
    if not losses:
        return {name: (0.0 if name == default_strategy else 1e-9) for name, _prediction in strategies}
    return losses


def _strategy_loss_ratios(frame: pl.DataFrame, strategies: Sequence[tuple[str, str]]) -> dict[str, float]:
    denominator = float(frame.select((pl.col("weight") * pl.col(TARGET).pow(2)).sum()).item())
    if denominator <= 1e-12:
        return {name: 1.0 for name, _prediction in strategies}
    expressions = [
        (pl.col("weight") * (pl.col(TARGET) - pl.col(prediction)).pow(2)).sum().alias(name)
        for name, prediction in strategies
    ]
    row = frame.select(expressions).row(0, named=True)
    return {name: float(row[name]) / denominator for name, _prediction in strategies}


def _best_strategy_by_loss(frame: pl.DataFrame, strategies: Sequence[tuple[str, str]]) -> str:
    losses = _strategy_loss_ratios(frame, strategies)
    return min(losses, key=lambda name: (losses[name], _strategy_order(strategies, name)))


def _prediction_for_strategy(strategies: Sequence[tuple[str, str]], name: str) -> str:
    for strategy_name, prediction in strategies:
        if strategy_name == name:
            return prediction
    raise ValueError(f"unknown strategy: {name}")


def _strategy_order(strategies: Sequence[tuple[str, str]], name: str) -> int:
    names = [strategy_name for strategy_name, _prediction in strategies]
    try:
        return names.index(name)
    except ValueError as exc:
        raise ValueError(f"unknown strategy: {name}") from exc


def _validate_strategy_inputs(strategies: Sequence[tuple[str, str]], default_strategy: str) -> None:
    names = [name for name, _prediction in strategies]
    if not names:
        raise ValueError("strategies must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("strategy names must be unique")
    if default_strategy not in names:
        raise ValueError("default_strategy must be one of the strategies")


def _validate_decay(value: float) -> None:
    if value < 0.0 or value >= 1.0:
        raise ValueError("ewma_decay must be in [0, 1)")


def _validate_posterior_strengths(strengths: Sequence[float]) -> None:
    if not strengths:
        raise ValueError("posterior shrink strengths must not be empty")
    if any(strength < 0.0 for strength in strengths):
        raise ValueError("posterior shrink strengths must be non-negative")


def _validate_risk_profiles(profiles: Sequence[str]) -> None:
    valid = {"disagreement", "abs_pred", "weight", "disagreement_weight", "all"}
    if not profiles:
        raise ValueError("risk profiles must not be empty")
    unknown = sorted(set(profiles) - valid)
    if unknown:
        raise ValueError(f"unknown risk profiles: {unknown}")


def _score_strategy(frame: pl.DataFrame, *, strategy: str, method_family: str, prediction: str) -> pl.DataFrame:
    return score_prediction_by_fold(frame, prediction=prediction).with_columns(
        pl.lit(strategy).alias("strategy"),
        pl.lit(method_family).alias("method_family"),
    )


def _choice_counts(frame: pl.DataFrame, *, strategy: str, choice_column: str) -> pl.DataFrame:
    return (
        frame.group_by(["fold", choice_column])
        .agg(
            pl.len().alias("rows"),
            pl.col("weight").sum().alias("weight_sum"),
        )
        .rename({choice_column: "selected_strategy"})
        .with_columns(pl.lit(strategy).alias("strategy"))
        .select(["strategy", "fold", "selected_strategy", "rows", "weight_sum"])
        .sort(["strategy", "fold", "selected_strategy"])
    )


def _softmax_negative_losses(losses: np.ndarray, *, eta: float) -> np.ndarray:
    scaled = -eta * losses
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(losses.shape[0], 1.0 / losses.shape[0], dtype=np.float64)
    return weights / total


def _audit_status(audit_frame: pl.DataFrame) -> dict[str, bool | int]:
    if audit_frame.is_empty():
        return {"audit_rows": 0, "bad_updates": 0, "all_strictly_past": True}
    if "update_is_strictly_past" not in audit_frame.columns:
        return {"audit_rows": audit_frame.height, "bad_updates": 0, "all_strictly_past": True}
    bad_updates = audit_frame.filter(~pl.col("update_is_strictly_past")).height
    return {
        "audit_rows": audit_frame.height,
        "bad_updates": int(bad_updates),
        "all_strictly_past": bad_updates == 0,
    }


def _daily_frames(frame: pl.DataFrame) -> list[pl.DataFrame]:
    return frame.sort(["fold", "date_id", "time_id", "symbol_id"]).partition_by(["fold", "date_id"], maintain_order=True)


def _folds(frame: pl.DataFrame) -> list[str]:
    return frame.select("fold").unique().sort("fold")["fold"].to_list()


def _parse_float_list(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


def _parse_risk_profiles(raw: str) -> tuple[str, ...]:
    profiles = tuple(item.strip() for item in raw.split(",") if item.strip())
    _validate_risk_profiles(profiles)
    return profiles


def _format_float(value: float) -> str:
    return f"{value:.6g}".replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    main()
