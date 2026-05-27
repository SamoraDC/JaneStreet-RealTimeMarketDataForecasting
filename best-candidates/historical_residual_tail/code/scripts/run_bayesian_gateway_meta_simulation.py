"""Strict gateway-style simulation for Bayesian meta prediction layers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from janestreet.bayesian_meta import score_prediction_by_fold, summarize_fold_scores, weighted_normal_stats
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
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/bayesian_gateway_meta_stage3_tabm_aux8_tree"))
    parser.add_argument("--ridge-alphas", type=str, default="1000,10000")
    args = parser.parse_args()
    _validate_args(args)

    frame = _load_joined_predictions(args.tabm_prediction_dir, args.tree_prediction_dir)
    frame, convex_parameters = _add_walk_forward_convex_baseline(frame)
    expert_columns = tuple(column for column in _expert_columns(frame) if column != "baseline_prediction")

    summary_rows: list[dict[str, float | int | str]] = []
    by_fold_frames: list[pl.DataFrame] = []
    parameter_rows: list[dict[str, float | int | str | None]] = []
    parameter_rows.extend(convex_parameters)
    audit_frames: list[pl.DataFrame] = []

    baseline_scores = score_prediction_by_fold(frame, prediction="baseline_prediction").with_columns(
        pl.lit("tabm_tree_convex_walk_forward").alias("strategy"),
        pl.lit("baseline").alias("method_family"),
    )
    by_fold_frames.append(baseline_scores)
    summary_rows.append({"strategy": "tabm_tree_convex_walk_forward", "method_family": "baseline", **summarize_fold_scores(baseline_scores)})

    feature_sets = _gateway_feature_sets(expert_columns)
    for alpha in _parse_float_list(args.ridge_alphas):
        for feature_set_name, feature_columns in feature_sets.items():
            strategy = f"gateway_online_ridge_{feature_set_name}_alpha{alpha:g}"
            scores, params, audit = _evaluate_gateway_online_ridge(
                frame,
                feature_columns=feature_columns,
                ridge_alpha=alpha,
            )
            scores = scores.with_columns(
                pl.lit(strategy).alias("strategy"),
                pl.lit("gateway_online_ridge").alias("method_family"),
            )
            by_fold_frames.append(scores)
            summary_rows.append({"strategy": strategy, "method_family": "gateway_online_ridge", **summarize_fold_scores(scores)})
            parameter_rows.extend({"strategy": strategy, **row} for row in params)
            audit_frames.append(audit.with_columns(pl.lit(strategy).alias("strategy")))

    summary = pl.DataFrame(summary_rows).sort(["global_r2", "min_fold_r2"], descending=[True, True])
    by_fold = pl.concat(by_fold_frames, how="diagonal").select(
        ["strategy", "method_family", "fold", "rows", "weight_sum", "numerator", "denominator", "weighted_zero_mean_r2"]
    )
    parameters = pl.DataFrame(parameter_rows) if parameter_rows else pl.DataFrame()
    audit_frame = pl.concat(audit_frames, how="diagonal") if audit_frames else pl.DataFrame()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.output_dir / "gateway_meta_summary.csv")
    by_fold.write_csv(args.output_dir / "gateway_meta_by_fold.csv")
    if not parameters.is_empty():
        parameters.write_csv(args.output_dir / "gateway_meta_parameters.csv")
    if not audit_frame.is_empty():
        audit_frame.write_csv(args.output_dir / "gateway_meta_daily_audit.csv")
    report = {
        "experiment": "bayesian_gateway_meta_simulation",
        "rows": frame.height,
        "folds": frame["fold"].n_unique(),
        "expert_columns": expert_columns,
        "ridge_alphas": _parse_float_list(args.ridge_alphas),
        "caveat": "Gateway simulation updates at the start of date D using cached features from D-1 joined to lag-delivered responder_6 from D-1.",
        "best": summary.row(0, named=True),
    }
    (args.output_dir / "gateway_meta_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summary)
    print(f"Wrote {args.output_dir}")


def _validate_args(args: argparse.Namespace) -> None:
    if not args.tabm_prediction_dir.exists():
        raise FileNotFoundError(args.tabm_prediction_dir)
    if not args.tree_prediction_dir.exists():
        raise FileNotFoundError(args.tree_prediction_dir)
    if not _parse_float_list(args.ridge_alphas):
        raise ValueError("--ridge-alphas must not be empty")


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


def _evaluate_gateway_online_ridge(
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    ridge_alpha: float,
) -> tuple[pl.DataFrame, list[dict[str, float | int | str | None]], pl.DataFrame]:
    folds = _folds(frame)
    features = tuple(feature_columns)
    prior_beta = _prior_beta(features)
    score_rows: list[pl.DataFrame] = []
    parameter_rows: list[dict[str, float | int | str | None]] = []
    audit_frames: list[pl.DataFrame] = []
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
        metrics, last_beta, audit = _simulate_gateway_online_ridge(current, features=features, precision=precision, rhs=rhs)
        score_rows.append(_metric_row_to_frame(fold, metrics))
        audit_frames.append(audit.with_columns(pl.lit(fold).alias("fold")))
        for feature, value in zip(features, last_beta, strict=True):
            parameter_rows.append(_parameter_row("gateway_online_ridge", fold, feature, "last_beta", float(value), fit_rows))
    return pl.concat(score_rows), parameter_rows, pl.concat(audit_frames, how="diagonal")


def _simulate_gateway_online_ridge(
    frame: pl.DataFrame,
    *,
    features: Sequence[str],
    precision: np.ndarray,
    rhs: np.ndarray,
) -> tuple[dict[str, float | int], np.ndarray, pl.DataFrame]:
    numerator = 0.0
    denominator = 0.0
    rows = 0
    weight_sum = 0.0
    beta = np.linalg.solve(precision, rhs)
    pending_update: pl.DataFrame | None = None
    audit_rows: list[dict[str, int | bool | None]] = []
    for day in _daily_frames(frame):
        current_date = int(day["date_id"][0])
        update_source_date: int | None = None
        update_rows = 0
        if pending_update is not None:
            lag_delivery = _deliver_previous_day_lags(pending_update, current_date=current_date)
            update_source_date = int(lag_delivery["date_id"][0])
            update_rows = lag_delivery.height
            precision, rhs = _update_ridge_state(precision, rhs, lag_delivery, features=features)
        beta = np.linalg.solve(precision, rhs)
        test_batch = day.select(list(features) + ["weight"])
        arrays = test_batch.to_numpy()
        x = arrays[:, : len(features)]
        sample_weight = arrays[:, len(features)]
        y = day[TARGET].to_numpy().astype(np.float64, copy=False)
        pred = x @ beta
        err = y - pred
        numerator += float(np.sum(sample_weight * err * err))
        denominator += float(np.sum(sample_weight * y * y))
        rows += day.height
        weight_sum += float(np.sum(sample_weight))
        pending_update = day.select(["date_id", "time_id", "symbol_id", TARGET, "weight"] + list(features))
        audit_rows.append(
            {
                "date_id": current_date,
                "predicted_rows": day.height,
                "update_source_date_id": update_source_date,
                "update_rows": update_rows,
                "update_before_prediction": True,
                "update_is_strictly_past": update_source_date is None or update_source_date < current_date,
            }
        )
    return _metric_dict(numerator, denominator, rows, weight_sum), beta, pl.DataFrame(audit_rows)


def _deliver_previous_day_lags(pending_update: pl.DataFrame, *, current_date: int) -> pl.DataFrame:
    previous_date = int(pending_update["date_id"][0])
    if previous_date >= current_date:
        raise ValueError("lag delivery must be strictly earlier than the prediction date")
    return pending_update


def _update_ridge_state(
    precision: np.ndarray,
    rhs: np.ndarray,
    frame: pl.DataFrame,
    *,
    features: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    arrays = frame.select(list(features) + [TARGET, "weight"]).to_numpy()
    x = arrays[:, : len(features)]
    y = arrays[:, len(features)]
    sample_weight = arrays[:, len(features) + 1]
    precision = precision + x.T @ (x * sample_weight[:, None])
    rhs = rhs + x.T @ (sample_weight * y)
    return precision, rhs


def _prior_beta(features: Sequence[str]) -> np.ndarray:
    beta = np.zeros(len(features), dtype=np.float64)
    if "baseline_prediction" in features:
        beta[features.index("baseline_prediction")] = 1.0
    elif "tabm_prediction" in features:
        beta[features.index("tabm_prediction")] = 1.0
    return beta


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


def _gateway_feature_sets(expert_columns: Sequence[str]) -> dict[str, tuple[str, ...]]:
    available = set(expert_columns)

    def keep(columns: Sequence[str]) -> tuple[str, ...]:
        return tuple(column for column in columns if column in available)

    candidates = {
        "experts": tuple(expert_columns),
        "with_baseline": ("baseline_prediction",) + tuple(expert_columns),
        "components_no_tree_ensemble": keep(("tabm_prediction", "xgboost_prediction", "lightgbm_prediction", "ridge_calibrated_prediction")),
        "tabm_xgb_lgb": keep(("tabm_prediction", "xgboost_prediction", "lightgbm_prediction")),
        "tabm_tree": keep(("tabm_prediction", "tree_prediction")),
        "tabm_ridge": keep(("tabm_prediction", "ridge_calibrated_prediction")),
    }
    result: dict[str, tuple[str, ...]] = {}
    seen: set[tuple[str, ...]] = set()
    for name, columns in candidates.items():
        if not columns or columns in seen:
            continue
        result[name] = columns
        seen.add(columns)
    return result


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
        "fit_mode": "gateway_previous_day_lags",
        "fold": fold,
        "group": group,
        "value_name": value_name,
        "value": value,
        "rows": rows,
    }


def _parse_float_list(raw: str) -> list[float]:
    return [float(item) for item in raw.split(",") if item.strip()]


if __name__ == "__main__":
    main()
