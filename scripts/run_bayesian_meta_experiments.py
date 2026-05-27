"""Evaluate cheap causal Bayesian-style meta layers over saved OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from janestreet.bayesian_meta import (
    fit_empirical_bayes_scales,
    fit_hierarchical_means,
    score_prediction_by_fold,
    softmax_from_log_weights,
    summarize_fold_scores,
    weighted_normal_stats,
)
from janestreet.blending import add_convex_blend_prediction, fit_convex_blend_weight


TARGET = "responder_6"
KEYS = ["fold", "date_id", "time_id", "symbol_id"]


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
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/bayesian_meta_stage3_tabm_aux8_tree"))
    parser.add_argument("--time-bucket-size", type=int, default=100)
    parser.add_argument("--min-group-rows", type=int, default=20_000)
    parser.add_argument("--prior-strength", type=float, default=8.0)
    parser.add_argument("--ridge-alpha", type=float, default=10_000.0)
    parser.add_argument("--bma-etas", type=str, default="2.0,10.0")
    args = parser.parse_args()
    _validate_args(args)

    frame = _load_joined_predictions(args.tabm_prediction_dir, args.tree_prediction_dir)
    frame, convex_parameters = _add_walk_forward_convex_baseline(frame)
    expert_columns = _expert_columns(frame)

    summary_rows: list[dict[str, float | int | str]] = []
    fold_score_frames: list[pl.DataFrame] = []
    parameter_rows: list[dict[str, float | int | str | None]] = []
    parameter_rows.extend(convex_parameters)

    for strategy, prediction in {
        "tabm": "tabm_prediction",
        "tree_ensemble": "tree_prediction",
        "tabm_tree_convex_walk_forward": "baseline_prediction",
    }.items():
        scores = score_prediction_by_fold(frame, prediction=prediction).with_columns(
            pl.lit(strategy).alias("strategy"),
            pl.lit("baseline").alias("method_family"),
        )
        fold_score_frames.append(scores)
        summary_rows.append({"strategy": strategy, "method_family": "baseline", **summarize_fold_scores(scores)})

    eb_group_sets = [
        (),
        ("weight_bucket",),
        ("time_bucket",),
        ("weight_bucket", "time_bucket"),
        ("abs_pred_bucket",),
        ("weight_bucket", "abs_pred_bucket"),
        ("disagreement_bucket",),
    ]
    for groups in eb_group_sets:
        strategy = "eb_scale_" + ("global" if not groups else "_".join(groups))
        scores, params = _evaluate_empirical_bayes_scale(
            frame,
            group_columns=groups,
            min_group_rows=args.min_group_rows,
            prior_strength=args.prior_strength,
            time_bucket_size=args.time_bucket_size,
        )
        fold_score_frames.append(scores.with_columns(pl.lit(strategy).alias("strategy"), pl.lit("empirical_bayes_scale").alias("method_family")))
        summary_rows.append({"strategy": strategy, "method_family": "empirical_bayes_scale", **summarize_fold_scores(scores)})
        parameter_rows.extend({"strategy": strategy, **row} for row in params)

    for groups in [
        ("symbol_id",),
        ("time_bucket",),
        ("weight_bucket",),
        ("time_bucket", "weight_bucket"),
        ("symbol_id", "time_bucket"),
    ]:
        strategy = "hier_mean_" + "_".join(groups)
        scores, params = _evaluate_hierarchical_mean(
            frame,
            group_columns=groups,
            min_group_rows=args.min_group_rows,
            prior_strength=args.prior_strength,
            time_bucket_size=args.time_bucket_size,
        )
        fold_score_frames.append(scores.with_columns(pl.lit(strategy).alias("strategy"), pl.lit("hierarchical_mean").alias("method_family")))
        summary_rows.append({"strategy": strategy, "method_family": "hierarchical_mean", **summarize_fold_scores(scores)})
        parameter_rows.extend({"strategy": strategy, **row} for row in params)

    scores, params = _evaluate_market_maker_deadband(frame)
    fold_score_frames.append(scores.with_columns(pl.lit("market_maker_deadband").alias("strategy"), pl.lit("market_maker_proxy").alias("method_family")))
    summary_rows.append({"strategy": "market_maker_deadband", "method_family": "market_maker_proxy", **summarize_fold_scores(scores)})
    parameter_rows.extend({"strategy": "market_maker_deadband", **row} for row in params)

    for eta in _parse_float_list(args.bma_etas):
        strategy = f"bayesian_model_average_eta{eta:g}"
        scores, params = _evaluate_bayesian_model_average(frame, expert_columns=expert_columns, eta=eta)
        fold_score_frames.append(scores.with_columns(pl.lit(strategy).alias("strategy"), pl.lit("bayesian_model_average").alias("method_family")))
        summary_rows.append({"strategy": strategy, "method_family": "bayesian_model_average", **summarize_fold_scores(scores)})
        parameter_rows.extend({"strategy": strategy, **row} for row in params)

    for use_baseline in [True, False]:
        feature_columns = ("baseline_prediction",) + tuple(col for col in expert_columns if col != "baseline_prediction") if use_baseline else tuple(
            col for col in expert_columns if col != "baseline_prediction"
        )
        static_strategy = "bayesian_static_ridge_with_baseline" if use_baseline else "bayesian_static_ridge_experts"
        scores, params = _evaluate_bayesian_static_ridge(frame, feature_columns=feature_columns, ridge_alpha=args.ridge_alpha)
        fold_score_frames.append(scores.with_columns(pl.lit(static_strategy).alias("strategy"), pl.lit("bayesian_static_ridge").alias("method_family")))
        summary_rows.append({"strategy": static_strategy, "method_family": "bayesian_static_ridge", **summarize_fold_scores(scores)})
        parameter_rows.extend({"strategy": static_strategy, **row} for row in params)

        strategy = "bayesian_online_ridge_with_baseline" if use_baseline else "bayesian_online_ridge_experts"
        scores, params = _evaluate_bayesian_online_ridge(frame, feature_columns=feature_columns, ridge_alpha=args.ridge_alpha)
        fold_score_frames.append(scores.with_columns(pl.lit(strategy).alias("strategy"), pl.lit("bayesian_online_ridge").alias("method_family")))
        summary_rows.append({"strategy": strategy, "method_family": "bayesian_online_ridge", **summarize_fold_scores(scores)})
        parameter_rows.extend({"strategy": strategy, **row} for row in params)

    summary = pl.DataFrame(summary_rows).sort(["global_r2", "min_fold_r2"], descending=[True, True])
    by_fold = pl.concat(fold_score_frames, how="diagonal").select(
        ["strategy", "method_family", "fold", "rows", "weight_sum", "numerator", "denominator", "weighted_zero_mean_r2"]
    )
    parameters = pl.DataFrame(parameter_rows) if parameter_rows else pl.DataFrame()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.output_dir / "bayesian_meta_summary.csv")
    by_fold.write_csv(args.output_dir / "bayesian_meta_by_fold.csv")
    if not parameters.is_empty():
        parameters.write_csv(args.output_dir / "bayesian_meta_parameters.csv")
    report = {
        "experiment": "bayesian_meta_stage3",
        "tabm_prediction_dir": str(args.tabm_prediction_dir),
        "tree_prediction_dir": str(args.tree_prediction_dir),
        "rows": frame.height,
        "folds": frame["fold"].n_unique(),
        "expert_columns": expert_columns,
        "caveat": "All promoted rows use previous folds or previous days for fitting; first fold uses explicit priors.",
        "best": summary.row(0, named=True),
    }
    (args.output_dir / "bayesian_meta_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summary)
    print(f"Wrote {args.output_dir}")


def _validate_args(args: argparse.Namespace) -> None:
    if not args.tabm_prediction_dir.exists():
        raise FileNotFoundError(args.tabm_prediction_dir)
    if not args.tree_prediction_dir.exists():
        raise FileNotFoundError(args.tree_prediction_dir)
    if args.time_bucket_size <= 0:
        raise ValueError("--time-bucket-size must be positive")
    if args.min_group_rows <= 0:
        raise ValueError("--min-group-rows must be positive")
    if args.prior_strength < 0.0:
        raise ValueError("--prior-strength must be non-negative")
    if args.ridge_alpha < 0.0:
        raise ValueError("--ridge-alpha must be non-negative")


def _load_joined_predictions(tabm_dir: Path, tree_dir: Path) -> pl.DataFrame:
    tabm = pl.scan_parquet(str(tabm_dir / "*.parquet")).select(KEYS + ["weight", TARGET, "tabm_prediction"])
    tree_columns = ["ensemble_prediction", "ridge_calibrated_prediction", "xgboost_prediction", "lightgbm_prediction", "catboost_prediction"]
    tree_schema = pl.scan_parquet(str(tree_dir / "*.parquet")).collect_schema()
    selected_tree_columns = [column for column in tree_columns if column in tree_schema.names()]
    tree = (
        pl.scan_parquet(str(tree_dir / "*.parquet"))
        .select(KEYS + selected_tree_columns)
        .rename({"ensemble_prediction": "tree_prediction"})
    )
    frame = tabm.join(tree, on=KEYS, how="inner").collect().sort(KEYS)
    if frame.height == 0:
        raise ValueError("prediction join is empty")
    return frame


def _add_walk_forward_convex_baseline(frame: pl.DataFrame) -> tuple[pl.DataFrame, list[dict[str, float | int | str | None]]]:
    frames: list[pl.DataFrame] = []
    rows: list[dict[str, float | int | str | None]] = []
    folds = _folds(frame)
    for idx, fold in enumerate(folds):
        current = frame.filter(pl.col("fold") == fold)
        if idx == 0:
            blend_weight = 1.0
            fit_rows = 0
        else:
            calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
            blend_weight = fit_convex_blend_weight(calibration, left_prediction="tabm_prediction", right_prediction="tree_prediction")
            fit_rows = calibration.height
        rows.append(
            {
                "method_family": "baseline",
                "fit_mode": "walk_forward_previous_folds",
                "fold": fold,
                "group": "__global__",
                "value_name": "tabm_weight",
                "value": blend_weight,
                "rows": fit_rows,
            }
        )
        frames.append(
            add_convex_blend_prediction(
                current,
                blend_weight=blend_weight,
                left_prediction="tabm_prediction",
                right_prediction="tree_prediction",
                output="baseline_prediction",
            )
        )
    return pl.concat(frames), rows


def _evaluate_empirical_bayes_scale(
    frame: pl.DataFrame,
    *,
    group_columns: Sequence[str],
    min_group_rows: int,
    prior_strength: float,
    time_bucket_size: int,
) -> tuple[pl.DataFrame, list[dict[str, float | int | str | None]]]:
    score_rows: list[pl.DataFrame] = []
    parameter_rows: list[dict[str, float | int | str | None]] = []
    folds = _folds(frame)
    groups = tuple(group_columns)
    for idx, fold in enumerate(folds):
        current_raw = frame.filter(pl.col("fold") == fold)
        if idx == 0:
            current = current_raw.with_columns(pl.col("baseline_prediction").alias("eb_prediction"))
            parameter_rows.append(_parameter_row("empirical_bayes_scale", fold, "__prior__", "fallback_scale", 1.0, 0))
        else:
            calibration_raw = frame.filter(pl.col("fold").is_in(folds[:idx]))
            thresholds = _fit_thresholds(calibration_raw)
            calibration = _add_meta_buckets(calibration_raw, thresholds=thresholds, time_bucket_size=time_bucket_size)
            current_bucketed = _add_meta_buckets(current_raw, thresholds=thresholds, time_bucket_size=time_bucket_size)
            model = fit_empirical_bayes_scales(
                calibration,
                group_columns=groups,
                prediction="baseline_prediction",
                prior_strength=prior_strength,
                min_group_rows=min_group_rows,
            )
            current = model.apply(current_bucketed, prediction="baseline_prediction", output="eb_prediction")
            parameter_rows.append(
                _parameter_row("empirical_bayes_scale", fold, "__fallback__", "fallback_scale", model.fallback_scale, calibration.height)
            )
        score_rows.append(score_prediction_by_fold(current, prediction="eb_prediction"))
    return pl.concat(score_rows), parameter_rows


def _evaluate_hierarchical_mean(
    frame: pl.DataFrame,
    *,
    group_columns: Sequence[str],
    min_group_rows: int,
    prior_strength: float,
    time_bucket_size: int,
) -> tuple[pl.DataFrame, list[dict[str, float | int | str | None]]]:
    score_rows: list[pl.DataFrame] = []
    parameter_rows: list[dict[str, float | int | str | None]] = []
    folds = _folds(frame)
    groups = tuple(group_columns)
    for idx, fold in enumerate(folds):
        current_raw = frame.filter(pl.col("fold") == fold)
        if idx == 0:
            current = current_raw.with_columns(pl.lit(0.0).alias("hier_mean_prediction"))
            parameter_rows.append(_parameter_row("hierarchical_mean", fold, "__prior__", "fallback_mean", 0.0, 0))
        else:
            calibration_raw = frame.filter(pl.col("fold").is_in(folds[:idx]))
            thresholds = _fit_thresholds(calibration_raw)
            calibration = _add_meta_buckets(calibration_raw, thresholds=thresholds, time_bucket_size=time_bucket_size)
            current_bucketed = _add_meta_buckets(current_raw, thresholds=thresholds, time_bucket_size=time_bucket_size)
            model = fit_hierarchical_means(
                calibration,
                group_columns=groups,
                prior_strength=prior_strength,
                min_group_rows=min_group_rows,
            )
            current = model.apply(current_bucketed, output="hier_mean_prediction")
            parameter_rows.append(_parameter_row("hierarchical_mean", fold, "__fallback__", "fallback_mean", model.fallback_mean, calibration.height))
        score_rows.append(score_prediction_by_fold(current, prediction="hier_mean_prediction"))
    return pl.concat(score_rows), parameter_rows


def _evaluate_market_maker_deadband(frame: pl.DataFrame) -> tuple[pl.DataFrame, list[dict[str, float | int | str | None]]]:
    """Proxy market-maker rule: skip weak alpha when previous folds say so."""

    score_rows: list[pl.DataFrame] = []
    parameter_rows: list[dict[str, float | int | str | None]] = []
    folds = _folds(frame)
    for idx, fold in enumerate(folds):
        current_raw = frame.filter(pl.col("fold") == fold)
        if idx == 0:
            current = current_raw.with_columns(pl.col("baseline_prediction").alias("deadband_prediction"))
            parameter_rows.append(_parameter_row("market_maker_proxy", fold, "__prior__", "abs_threshold", 0.0, 0))
        else:
            calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
            abs_pred = calibration["baseline_prediction"].abs()
            thresholds = [0.0] + [float(abs_pred.quantile(q)) for q in (0.25, 0.50, 0.75, 0.90)]
            best_threshold = 0.0
            best_scale = 1.0
            best_loss = np.inf
            for threshold in thresholds:
                candidate = _apply_deadband(calibration, threshold=threshold, scale=1.0, output="candidate_prediction")
                model = fit_empirical_bayes_scales(candidate, group_columns=(), prediction="candidate_prediction", min_scale=0.0, max_scale=2.0)
                candidate = model.apply(candidate, prediction="candidate_prediction", output="scaled_candidate_prediction")
                score = score_prediction_by_fold(candidate, prediction="scaled_candidate_prediction")
                loss = float(score["numerator"].sum())
                if loss < best_loss:
                    best_loss = loss
                    best_threshold = threshold
                    best_scale = model.fallback_scale
            current = _apply_deadband(current_raw, threshold=best_threshold, scale=best_scale, output="deadband_prediction")
            parameter_rows.append(_parameter_row("market_maker_proxy", fold, "__global__", "abs_threshold", best_threshold, calibration.height))
            parameter_rows.append(_parameter_row("market_maker_proxy", fold, "__global__", "scale", best_scale, calibration.height))
        score_rows.append(score_prediction_by_fold(current, prediction="deadband_prediction"))
    return pl.concat(score_rows), parameter_rows


def _evaluate_bayesian_model_average(
    frame: pl.DataFrame,
    *,
    expert_columns: Sequence[str],
    eta: float,
) -> tuple[pl.DataFrame, list[dict[str, float | int | str | None]]]:
    if eta < 0.0:
        raise ValueError("eta must be non-negative")
    folds = _folds(frame)
    score_rows = []
    parameter_rows: list[dict[str, float | int | str | None]] = []
    experts = tuple(expert_columns)
    for idx, fold in enumerate(folds):
        current = frame.filter(pl.col("fold") == fold)
        log_weights = np.zeros(len(experts), dtype=np.float64)
        fit_rows = 0
        if idx > 0:
            calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
            losses = _expert_losses(calibration, experts)
            log_weights = -eta * losses
            fit_rows = calibration.height
        metrics = _stream_expert_averaging(current, experts=experts, log_weights=log_weights, eta=eta)
        score_rows.append(_metric_row_to_frame(fold, metrics))
        weights = softmax_from_log_weights(log_weights)
        for expert, value in zip(experts, weights, strict=True):
            parameter_rows.append(_parameter_row("bayesian_model_average", fold, expert, "initial_weight", float(value), fit_rows))
    return pl.concat(score_rows), parameter_rows


def _evaluate_bayesian_online_ridge(
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    ridge_alpha: float,
) -> tuple[pl.DataFrame, list[dict[str, float | int | str | None]]]:
    folds = _folds(frame)
    features = tuple(feature_columns)
    prior_beta = np.zeros(len(features), dtype=np.float64)
    if "baseline_prediction" in features:
        prior_beta[features.index("baseline_prediction")] = 1.0
    elif "tabm_prediction" in features:
        prior_beta[features.index("tabm_prediction")] = 1.0
    score_rows = []
    parameter_rows: list[dict[str, float | int | str | None]] = []
    for idx, fold in enumerate(folds):
        precision = ridge_alpha * np.eye(len(features), dtype=np.float64)
        rhs = ridge_alpha * prior_beta
        fit_rows = 0
        if idx > 0:
            calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
            gram, cal_rhs = weighted_normal_stats(calibration, feature_columns=features)
            precision += gram
            rhs += cal_rhs
            fit_rows = calibration.height
        current = frame.filter(pl.col("fold") == fold)
        metrics, last_beta = _stream_online_ridge(current, features=features, precision=precision, rhs=rhs)
        score_rows.append(_metric_row_to_frame(fold, metrics))
        for feature, value in zip(features, last_beta, strict=True):
            parameter_rows.append(_parameter_row("bayesian_online_ridge", fold, feature, "last_beta", float(value), fit_rows))
    return pl.concat(score_rows), parameter_rows


def _evaluate_bayesian_static_ridge(
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    ridge_alpha: float,
) -> tuple[pl.DataFrame, list[dict[str, float | int | str | None]]]:
    folds = _folds(frame)
    features = tuple(feature_columns)
    prior_beta = np.zeros(len(features), dtype=np.float64)
    if "baseline_prediction" in features:
        prior_beta[features.index("baseline_prediction")] = 1.0
    elif "tabm_prediction" in features:
        prior_beta[features.index("tabm_prediction")] = 1.0
    score_rows = []
    parameter_rows: list[dict[str, float | int | str | None]] = []
    for idx, fold in enumerate(folds):
        precision = ridge_alpha * np.eye(len(features), dtype=np.float64)
        rhs = ridge_alpha * prior_beta
        fit_rows = 0
        if idx > 0:
            calibration = frame.filter(pl.col("fold").is_in(folds[:idx]))
            gram, cal_rhs = weighted_normal_stats(calibration, feature_columns=features)
            precision += gram
            rhs += cal_rhs
            fit_rows = calibration.height
        beta = np.linalg.solve(precision, rhs)
        current = frame.filter(pl.col("fold") == fold)
        arrays = current.select(list(features) + [TARGET, "weight"]).to_numpy()
        x = arrays[:, : len(features)]
        y = arrays[:, len(features)]
        sample_weight = arrays[:, len(features) + 1]
        pred = x @ beta
        metrics = _metric_dict(
            float(np.sum(sample_weight * (y - pred) * (y - pred))),
            float(np.sum(sample_weight * y * y)),
            current.height,
            float(np.sum(sample_weight)),
        )
        score_rows.append(_metric_row_to_frame(fold, metrics))
        for feature, value in zip(features, beta, strict=True):
            parameter_rows.append(_parameter_row("bayesian_static_ridge", fold, feature, "beta", float(value), fit_rows))
    return pl.concat(score_rows), parameter_rows


def _stream_expert_averaging(
    frame: pl.DataFrame,
    *,
    experts: Sequence[str],
    log_weights: np.ndarray,
    eta: float,
) -> dict[str, float | int]:
    numerator = 0.0
    denominator = 0.0
    rows = 0
    weight_sum = 0.0
    for day in _daily_frames(frame):
        weights = softmax_from_log_weights(log_weights)
        arrays = day.select(list(experts) + [TARGET, "weight"]).to_numpy()
        predictions = arrays[:, : len(experts)]
        y = arrays[:, len(experts)]
        sample_weight = arrays[:, len(experts) + 1]
        pred = predictions @ weights
        err = y - pred
        numerator += float(np.sum(sample_weight * err * err))
        day_denominator = float(np.sum(sample_weight * y * y))
        denominator += day_denominator
        rows += day.height
        weight_sum += float(np.sum(sample_weight))
        if day_denominator > 1e-12:
            expert_errors = y[:, None] - predictions
            expert_losses = np.sum(sample_weight[:, None] * expert_errors * expert_errors, axis=0) / day_denominator
            log_weights -= eta * expert_losses
            log_weights -= np.max(log_weights)
    return _metric_dict(numerator, denominator, rows, weight_sum)


def _stream_online_ridge(
    frame: pl.DataFrame,
    *,
    features: Sequence[str],
    precision: np.ndarray,
    rhs: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray]:
    numerator = 0.0
    denominator = 0.0
    rows = 0
    weight_sum = 0.0
    beta = np.linalg.solve(precision, rhs)
    for day in _daily_frames(frame):
        arrays = day.select(list(features) + [TARGET, "weight"]).to_numpy()
        x = arrays[:, : len(features)]
        y = arrays[:, len(features)]
        sample_weight = arrays[:, len(features) + 1]
        beta = np.linalg.solve(precision, rhs)
        pred = x @ beta
        err = y - pred
        numerator += float(np.sum(sample_weight * err * err))
        denominator += float(np.sum(sample_weight * y * y))
        rows += day.height
        weight_sum += float(np.sum(sample_weight))
        weighted_x = x * sample_weight[:, None]
        precision += x.T @ weighted_x
        rhs += x.T @ (sample_weight * y)
    return _metric_dict(numerator, denominator, rows, weight_sum), beta


def _expert_losses(frame: pl.DataFrame, experts: Sequence[str]) -> np.ndarray:
    denominator = float(frame.select((pl.col("weight") * pl.col(TARGET).pow(2)).sum()).item())
    if denominator <= 1e-12:
        return np.zeros(len(experts), dtype=np.float64)
    losses = []
    for expert in experts:
        numerator = float(frame.select((pl.col("weight") * (pl.col(TARGET) - pl.col(expert)).pow(2)).sum()).item())
        losses.append(numerator / denominator)
    return np.asarray(losses, dtype=np.float64)


def _apply_deadband(frame: pl.DataFrame, *, threshold: float, scale: float, output: str) -> pl.DataFrame:
    return frame.with_columns(
        (
            pl.when(pl.col("baseline_prediction").abs() >= threshold)
            .then(scale * pl.col("baseline_prediction"))
            .otherwise(0.0)
        ).alias(output)
    )


def _fit_thresholds(frame: pl.DataFrame) -> dict[str, float]:
    disagreement = (frame["tabm_prediction"] - frame["tree_prediction"]).abs()
    abs_pred = frame["baseline_prediction"].abs()
    return {
        "weight_q50": float(frame["weight"].quantile(0.50)),
        "weight_q90": float(frame["weight"].quantile(0.90)),
        "weight_q99": float(frame["weight"].quantile(0.99)),
        "disagreement_q50": float(disagreement.quantile(0.50)),
        "disagreement_q90": float(disagreement.quantile(0.90)),
        "disagreement_q99": float(disagreement.quantile(0.99)),
        "abs_pred_q50": float(abs_pred.quantile(0.50)),
        "abs_pred_q90": float(abs_pred.quantile(0.90)),
        "abs_pred_q99": float(abs_pred.quantile(0.99)),
    }


def _add_meta_buckets(frame: pl.DataFrame, *, thresholds: dict[str, float], time_bucket_size: int) -> pl.DataFrame:
    disagreement = (pl.col("tabm_prediction") - pl.col("tree_prediction")).abs()
    abs_pred = pl.col("baseline_prediction").abs()
    return frame.with_columns(
        [
            (pl.col("time_id") // time_bucket_size).cast(pl.Int16).alias("time_bucket"),
            _quantile_bucket(pl.col("weight"), thresholds["weight_q50"], thresholds["weight_q90"], thresholds["weight_q99"]).alias("weight_bucket"),
            _quantile_bucket(disagreement, thresholds["disagreement_q50"], thresholds["disagreement_q90"], thresholds["disagreement_q99"]).alias("disagreement_bucket"),
            _quantile_bucket(abs_pred, thresholds["abs_pred_q50"], thresholds["abs_pred_q90"], thresholds["abs_pred_q99"]).alias("abs_pred_bucket"),
        ]
    )


def _quantile_bucket(expr: pl.Expr, q50: float, q90: float, q99: float) -> pl.Expr:
    return (
        pl.when(expr <= q50)
        .then(pl.lit("q00_q50"))
        .when(expr <= q90)
        .then(pl.lit("q50_q90"))
        .when(expr <= q99)
        .then(pl.lit("q90_q99"))
        .otherwise(pl.lit("q99_q100"))
    )


def _metric_row_to_frame(fold: str, metrics: dict[str, float | int]) -> pl.DataFrame:
    return pl.DataFrame({"fold": [fold], **{key: [value] for key, value in metrics.items()}})


def _metric_dict(numerator: float, denominator: float, rows: int, weight_sum: float) -> dict[str, float | int]:
    if denominator <= 0.0:
        raise ValueError("denominator must be positive")
    return {
        "rows": rows,
        "weight_sum": weight_sum,
        "numerator": numerator,
        "denominator": denominator,
        "weighted_zero_mean_r2": 1.0 - numerator / denominator,
    }


def _daily_frames(frame: pl.DataFrame) -> list[pl.DataFrame]:
    return frame.sort(["date_id", "time_id", "symbol_id"]).partition_by("date_id", maintain_order=True)


def _expert_columns(frame: pl.DataFrame) -> tuple[str, ...]:
    columns = ["baseline_prediction", "tabm_prediction", "tree_prediction"]
    columns.extend(column for column in ["xgboost_prediction", "lightgbm_prediction", "ridge_calibrated_prediction"] if column in frame.columns)
    return tuple(dict.fromkeys(columns))


def _folds(frame: pl.DataFrame) -> list[str]:
    return frame.select("fold").unique().sort("fold")["fold"].to_list()


def _parameter_row(
    method_family: str,
    fold: str,
    group: str,
    value_name: str,
    value: float,
    rows: int,
) -> dict[str, float | int | str]:
    return {
        "method_family": method_family,
        "fit_mode": "walk_forward_previous_folds",
        "fold": fold,
        "group": group,
        "value_name": value_name,
        "value": value,
        "rows": rows,
    }


def _parse_float_list(raw: str) -> list[float]:
    values = [float(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one eta")
    return values


if __name__ == "__main__":
    main()
