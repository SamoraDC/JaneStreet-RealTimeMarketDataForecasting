"""Dynamic gateway RLS/Kalman-style meta layer over saved OOF predictions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from janestreet.bayesian_meta import score_prediction_by_fold, summarize_fold_scores, weighted_normal_stats


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "responder_6"


@dataclass(frozen=True)
class DynamicRLSCandidate:
    name: str
    feature_set: str
    ridge_alpha: float


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
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/dynamic_gateway_rls_stage3"))
    parser.add_argument("--experiment-name", default="dynamic_gateway_rls")
    parser.add_argument("--forgetting-factors", default="1.0,0.999,0.995")
    args = parser.parse_args()

    gateway = _load_gateway_module()
    frame = gateway._load_joined_predictions(args.tabm_prediction_dir, args.tree_prediction_dir)
    frame, convex_parameters = gateway._add_walk_forward_convex_baseline(frame)
    expert_columns = tuple(column for column in gateway._expert_columns(frame) if column != "baseline_prediction")
    feature_sets = gateway._gateway_feature_sets(expert_columns)
    forgetting_factors = _parse_forgetting_factors(args.forgetting_factors)

    by_fold_frames: list[pl.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []
    parameter_rows: list[dict[str, float | int | str | None]] = list(convex_parameters)
    audit_frames: list[pl.DataFrame] = []

    baseline_scores = score_prediction_by_fold(frame, prediction="baseline_prediction").with_columns(
        pl.lit("tabm_tree_convex_walk_forward").alias("strategy"),
        pl.lit("baseline").alias("method_family"),
    )
    by_fold_frames.append(baseline_scores)
    summary_rows.append({"strategy": "tabm_tree_convex_walk_forward", "method_family": "baseline", **summarize_fold_scores(baseline_scores)})

    for candidate in _dynamic_candidates():
        if candidate.feature_set not in feature_sets:
            continue
        for forgetting in forgetting_factors:
            strategy = f"dynamic_gateway_rls_{candidate.name}_f{_format_forgetting(forgetting)}"
            scores, params, audit = _evaluate_dynamic_gateway_rls(
                gateway,
                frame,
                feature_columns=feature_sets[candidate.feature_set],
                ridge_alpha=candidate.ridge_alpha,
                forgetting_factor=forgetting,
            )
            scores = scores.with_columns(
                pl.lit(strategy).alias("strategy"),
                pl.lit("dynamic_gateway_rls").alias("method_family"),
            )
            by_fold_frames.append(scores)
            summary_rows.append({"strategy": strategy, "method_family": "dynamic_gateway_rls", **summarize_fold_scores(scores)})
            parameter_rows.extend({"strategy": strategy, **row} for row in params)
            audit_frames.append(audit.with_columns(pl.lit(strategy).alias("strategy")))

    by_fold = pl.concat(by_fold_frames, how="diagonal").select(
        ["strategy", "method_family", "fold", "rows", "weight_sum", "numerator", "denominator", "weighted_zero_mean_r2"]
    )
    summary = pl.DataFrame(summary_rows).sort(["global_r2", "min_fold_r2"], descending=[True, True])
    parameters = pl.DataFrame(parameter_rows) if parameter_rows else pl.DataFrame()
    audit_frame = pl.concat(audit_frames, how="diagonal") if audit_frames else pl.DataFrame()
    report = {
        "experiment": args.experiment_name,
        "rows": frame.height,
        "folds": frame["fold"].n_unique(),
        "expert_columns": expert_columns,
        "forgetting_factors": forgetting_factors,
        "candidates": [candidate.__dict__ for candidate in _dynamic_candidates()],
        "best": summary.row(0, named=True),
        "audit_status": _audit_status(audit_frame),
        "methodological_note": (
            "forgetting_factor=1.0 should reproduce the frozen gateway online ridge. "
            "Factors below 1.0 are pre-registered dynamic RLS/Kalman-style alternatives and must be confirmed across Stage 3 and historical OOF before promotion."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.output_dir / "dynamic_gateway_rls_summary.csv")
    by_fold.write_csv(args.output_dir / "dynamic_gateway_rls_by_fold.csv")
    if not parameters.is_empty():
        parameters.write_csv(args.output_dir / "dynamic_gateway_rls_parameters.csv")
    if not audit_frame.is_empty():
        audit_frame.write_csv(args.output_dir / "dynamic_gateway_rls_daily_audit.csv")
    (args.output_dir / "dynamic_gateway_rls_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summary)
    print(f"Wrote {args.output_dir}")


def _load_gateway_module():
    path = PROJECT_ROOT / "scripts" / "run_bayesian_gateway_meta_simulation.py"
    spec = importlib.util.spec_from_file_location("run_bayesian_gateway_meta_simulation_for_dynamic_rls", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _dynamic_candidates() -> tuple[DynamicRLSCandidate, ...]:
    return (
        DynamicRLSCandidate("components_no_tree_ensemble_alpha1000", "components_no_tree_ensemble", 1000.0),
        DynamicRLSCandidate("experts_alpha10000", "experts", 10000.0),
    )


def _evaluate_dynamic_gateway_rls(
    gateway,
    frame: pl.DataFrame,
    *,
    feature_columns: Sequence[str],
    ridge_alpha: float,
    forgetting_factor: float,
) -> tuple[pl.DataFrame, list[dict[str, float | int | str | None]], pl.DataFrame]:
    folds = gateway._folds(frame)
    features = tuple(feature_columns)
    prior_beta = gateway._prior_beta(features)
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
        metrics, beta, audit = _simulate_dynamic_gateway_rls(
            gateway,
            current,
            features=features,
            precision=precision,
            rhs=rhs,
            forgetting_factor=forgetting_factor,
        )
        score_rows.append(gateway._metric_row_to_frame(fold, metrics))
        audit_frames.append(audit.with_columns(pl.lit(fold).alias("fold")))
        for feature, value in zip(features, beta, strict=True):
            parameter_rows.append(
                {
                    "method_family": "dynamic_gateway_rls",
                    "fit_mode": "gateway_previous_day_lags_with_forgetting",
                    "fold": fold,
                    "group": feature,
                    "value_name": f"last_beta_f{_format_forgetting(forgetting_factor)}",
                    "value": float(value),
                    "rows": fit_rows,
                }
            )
    return pl.concat(score_rows), parameter_rows, pl.concat(audit_frames, how="diagonal")


def _simulate_dynamic_gateway_rls(
    gateway,
    frame: pl.DataFrame,
    *,
    features: Sequence[str],
    precision: np.ndarray,
    rhs: np.ndarray,
    forgetting_factor: float,
) -> tuple[dict[str, float | int], np.ndarray, pl.DataFrame]:
    numerator = 0.0
    denominator = 0.0
    rows = 0
    weight_sum = 0.0
    beta = np.linalg.solve(precision, rhs)
    pending_update: pl.DataFrame | None = None
    audit_rows: list[dict[str, int | float | bool | None]] = []
    for day in gateway._daily_frames(frame):
        current_date = int(day["date_id"][0])
        update_source_date: int | None = None
        update_rows = 0
        if pending_update is not None:
            lag_delivery = gateway._deliver_previous_day_lags(pending_update, current_date=current_date)
            update_source_date = int(lag_delivery["date_id"][0])
            update_rows = lag_delivery.height
            precision, rhs = _forgetting_update(precision, rhs, lag_delivery, features=features, forgetting_factor=forgetting_factor)
        beta = np.linalg.solve(precision, rhs)
        arrays = day.select(list(features) + [TARGET, "weight"]).to_numpy()
        x = arrays[:, : len(features)]
        y = arrays[:, len(features)]
        sample_weight = arrays[:, len(features) + 1]
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
                "forgetting_factor": forgetting_factor,
                "update_before_prediction": True,
                "update_is_strictly_past": update_source_date is None or update_source_date < current_date,
            }
        )
    return gateway._metric_dict(numerator, denominator, rows, weight_sum), beta, pl.DataFrame(audit_rows)


def _forgetting_update(
    precision: np.ndarray,
    rhs: np.ndarray,
    frame: pl.DataFrame,
    *,
    features: Sequence[str],
    forgetting_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = frame.select(list(features) + [TARGET, "weight"]).to_numpy()
    x = arrays[:, : len(features)]
    y = arrays[:, len(features)]
    sample_weight = arrays[:, len(features) + 1]
    precision = forgetting_factor * precision + x.T @ (x * sample_weight[:, None])
    rhs = forgetting_factor * rhs + x.T @ (sample_weight * y)
    return precision, rhs


def _parse_forgetting_factors(raw: str) -> list[float]:
    values = [float(item) for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("--forgetting-factors must not be empty")
    for value in values:
        if value <= 0.0 or value > 1.0:
            raise ValueError("forgetting factors must be in (0, 1]")
    return values


def _format_forgetting(value: float) -> str:
    return f"{value:.6g}".replace(".", "p")


def _audit_status(audit_frame: pl.DataFrame) -> dict[str, bool | int]:
    if audit_frame.is_empty():
        return {"audit_rows": 0, "bad_updates": 0, "all_strictly_past": True}
    bad_updates = audit_frame.filter(~pl.col("update_is_strictly_past")).height
    return {
        "audit_rows": audit_frame.height,
        "bad_updates": int(bad_updates),
        "all_strictly_past": bad_updates == 0,
    }


if __name__ == "__main__":
    main()
