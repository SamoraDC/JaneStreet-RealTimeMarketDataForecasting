"""Frozen slice audit for the selected risk-modulated gateway RLS candidate."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Sequence

import polars as pl

from janestreet.bayesian_meta import score_prediction_by_fold, summarize_fold_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = "responder_6"
KEYS = ["fold", "date_id", "time_id", "symbol_id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tabm-prediction-dir",
        type=Path,
        default=Path(
            "reports/experiments/competitive_tabm_official_stage3_5fold_valid60_lags_online_lr1e4_4m_train700_seed37_aux8_preds/validation_predictions"
        ),
    )
    parser.add_argument(
        "--tree-prediction-dir",
        type=Path,
        default=Path("reports/experiments/tree_engine_ensemble_xgb_lgb_sample10_seed_ensemble_preds/validation_predictions"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/frozen_risk_shrink_slice_audit_stage3"))
    parser.add_argument("--experiment-name", default="frozen_risk_shrink_slice_audit")
    parser.add_argument("--risk-profile", default="abs_pred")
    parser.add_argument("--risk-strength", type=float, default=100.0)
    parser.add_argument("--time-bucket-size", type=int, default=100)
    args = parser.parse_args()

    if args.time_bucket_size <= 0:
        raise ValueError("--time-bucket-size must be positive")

    strategy_module = _load_strategy_module()
    gateway = strategy_module._load_gateway_module()
    dynamic = strategy_module._load_dynamic_module()

    frame = gateway._load_joined_predictions(args.tabm_prediction_dir, args.tree_prediction_dir)
    frame, convex_parameters = gateway._add_walk_forward_convex_baseline(frame)
    expert_columns = tuple(column for column in gateway._expert_columns(frame) if column != "baseline_prediction")
    feature_sets = gateway._gateway_feature_sets(expert_columns)
    _validate_feature_sets(feature_sets)

    prediction_frame = frame.select(KEYS + ["weight", TARGET, "baseline_prediction"])
    conservative_predictions, conservative_audit = strategy_module._predict_dynamic_rls_strategy(
        gateway,
        dynamic,
        frame,
        feature_columns=feature_sets["experts"],
        ridge_alpha=10000.0,
        forgetting_factor=0.995,
        output="conservative_rls_prediction",
    )
    aggressive_predictions, aggressive_audit = strategy_module._predict_dynamic_rls_strategy(
        gateway,
        dynamic,
        frame,
        feature_columns=feature_sets["components_no_tree_ensemble"],
        ridge_alpha=1000.0,
        forgetting_factor=0.995,
        output="aggressive_rls_prediction",
    )
    risk_prefix = "risk_shrink_conservative_rls"
    risk_predictions, risk_audit = strategy_module._predict_dynamic_rls_risk_modulated_shrinkage_strategy(
        gateway,
        dynamic,
        frame,
        feature_columns=feature_sets["experts"],
        ridge_alpha=10000.0,
        forgetting_factor=0.995,
        strengths=(args.risk_strength,),
        profiles=(args.risk_profile,),
        output_prefix=risk_prefix,
    )
    strength_suffix = strategy_module._format_float(args.risk_strength)
    risk_prediction = f"{risk_prefix}_{args.risk_profile}_s{strength_suffix}_prediction"
    risk_strategy = f"{risk_prefix}_{args.risk_profile}_s{strength_suffix}"

    prediction_frame = (
        prediction_frame.join(conservative_predictions, on=KEYS, how="inner")
        .join(aggressive_predictions, on=KEYS, how="inner")
        .join(risk_predictions, on=KEYS, how="inner")
    )
    prediction_frame = add_diagnostic_buckets(
        prediction_frame,
        reference_prediction="conservative_rls_prediction",
        candidate_prediction=risk_prediction,
        time_bucket_size=args.time_bucket_size,
    )

    strategies = (
        ("tabm_tree_convex_walk_forward", "baseline", "baseline_prediction"),
        ("conservative_rls", "base_candidate", "conservative_rls_prediction"),
        ("aggressive_rls", "base_candidate", "aggressive_rls_prediction"),
        (risk_strategy, "frozen_risk_modulated_posterior_shrinkage", risk_prediction),
    )
    by_fold_frames = [
        score_prediction_by_fold(prediction_frame, prediction=prediction).with_columns(
            pl.lit(strategy).alias("strategy"),
            pl.lit(method_family).alias("method_family"),
        )
        for strategy, method_family, prediction in strategies
    ]
    by_fold = pl.concat(by_fold_frames, how="diagonal").select(
        ["strategy", "method_family", "fold", "rows", "weight_sum", "numerator", "denominator", "weighted_zero_mean_r2"]
    )
    summary = pl.DataFrame(
        [
            {"strategy": strategy, "method_family": method_family, **summarize_fold_scores(scores)}
            for (strategy, method_family, _), scores in zip(strategies, by_fold_frames, strict=True)
        ]
    ).sort(["global_r2", "min_fold_r2"], descending=[True, True])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.output_dir / "frozen_risk_shrink_summary.csv")
    by_fold.write_csv(args.output_dir / "frozen_risk_shrink_by_fold.csv")

    slice_specs = _slice_specs()
    for name, group_columns in slice_specs.items():
        write_strategy_slice_scores(
            prediction_frame,
            strategies=strategies,
            group_columns=group_columns,
            path=args.output_dir / f"slice_{name}.csv",
        )
        compare_prediction_pair_by_slice(
            prediction_frame,
            group_columns=group_columns,
            baseline_prediction="conservative_rls_prediction",
            candidate_prediction=risk_prediction,
        ).write_csv(args.output_dir / f"delta_vs_conservative_{name}.csv")

    candidate_audit = pl.concat(
        [
            conservative_audit.with_columns(pl.lit("conservative_rls").alias("strategy")),
            aggressive_audit.with_columns(pl.lit("aggressive_rls").alias("strategy")),
        ],
        how="diagonal",
    )
    selector_audit = risk_audit.with_columns(pl.lit(risk_strategy).alias("strategy"))
    candidate_audit.write_csv(args.output_dir / "candidate_daily_audit.csv")
    selector_audit.write_csv(args.output_dir / "risk_shrink_daily_audit.csv")

    report = {
        "experiment": args.experiment_name,
        "rows": prediction_frame.height,
        "risk_strategy": risk_strategy,
        "risk_profile": args.risk_profile,
        "risk_strength": args.risk_strength,
        "time_bucket_size": args.time_bucket_size,
        "bucket_thresholds": _bucket_thresholds(prediction_frame),
        "best_full": summary.row(0, named=True),
        "candidate_audit_status": _audit_status(candidate_audit),
        "risk_audit_status": _audit_status(selector_audit),
        "convex_parameters": convex_parameters,
        "validation_caveat": (
            "The candidate is frozen here, but it was selected after looking at this OOF family. "
            "Use this report as slice and causal audit, not as an independent holdout."
        ),
    }
    (args.output_dir / "frozen_risk_shrink_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summary)
    print(f"Wrote {args.output_dir}")


def _load_strategy_module():
    path = PROJECT_ROOT / "scripts" / "run_gateway_rls_strategy_selection.py"
    spec = importlib.util.spec_from_file_location("run_gateway_rls_strategy_selection_for_frozen_slice_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_feature_sets(feature_sets: dict[str, tuple[str, ...]]) -> None:
    missing = [name for name in ("experts", "components_no_tree_ensemble") if name not in feature_sets]
    if missing:
        raise ValueError(f"missing feature sets: {missing}")


def add_diagnostic_buckets(
    frame: pl.DataFrame,
    *,
    reference_prediction: str,
    candidate_prediction: str,
    time_bucket_size: int,
) -> pl.DataFrame:
    thresholds = _compute_bucket_thresholds(frame, reference_prediction=reference_prediction, candidate_prediction=candidate_prediction)
    candidate_delta = pl.col(candidate_prediction) - pl.col(reference_prediction)
    return frame.with_columns(
        (pl.col("time_id") // time_bucket_size).cast(pl.Int16).alias("time_bucket"),
        _quantile_bucket(pl.col("weight"), thresholds["weight_q50"], thresholds["weight_q90"], thresholds["weight_q99"]).alias("weight_bucket"),
        _quantile_bucket(
            pl.col(reference_prediction).abs(),
            thresholds["reference_abs_q50"],
            thresholds["reference_abs_q90"],
            thresholds["reference_abs_q99"],
        ).alias("reference_abs_pred_bucket"),
        _quantile_bucket(
            candidate_delta.abs(),
            thresholds["candidate_delta_abs_q50"],
            thresholds["candidate_delta_abs_q90"],
            thresholds["candidate_delta_abs_q99"],
        ).alias("candidate_delta_abs_bucket"),
    )


def _compute_bucket_thresholds(frame: pl.DataFrame, *, reference_prediction: str, candidate_prediction: str) -> dict[str, float]:
    candidate_delta = (pl.col(candidate_prediction) - pl.col(reference_prediction)).abs()
    row = frame.select(
        pl.col("weight").quantile(0.50).alias("weight_q50"),
        pl.col("weight").quantile(0.90).alias("weight_q90"),
        pl.col("weight").quantile(0.99).alias("weight_q99"),
        pl.col(reference_prediction).abs().quantile(0.50).alias("reference_abs_q50"),
        pl.col(reference_prediction).abs().quantile(0.90).alias("reference_abs_q90"),
        pl.col(reference_prediction).abs().quantile(0.99).alias("reference_abs_q99"),
        candidate_delta.quantile(0.50).alias("candidate_delta_abs_q50"),
        candidate_delta.quantile(0.90).alias("candidate_delta_abs_q90"),
        candidate_delta.quantile(0.99).alias("candidate_delta_abs_q99"),
    ).row(0, named=True)
    return {key: float(value) for key, value in row.items()}


def _quantile_bucket(expr: pl.Expr, q50: float, q90: float, q99: float) -> pl.Expr:
    return (
        pl.when(expr <= q50)
        .then(pl.lit("p00_p50"))
        .when(expr <= q90)
        .then(pl.lit("p50_p90"))
        .when(expr <= q99)
        .then(pl.lit("p90_p99"))
        .otherwise(pl.lit("p99_p100"))
    )


def _slice_specs() -> dict[str, tuple[str, ...]]:
    return {
        "fold": ("fold",),
        "weight_bucket": ("weight_bucket",),
        "time_bucket": ("time_bucket",),
        "symbol_id": ("symbol_id",),
        "reference_abs_pred_bucket": ("reference_abs_pred_bucket",),
        "candidate_delta_abs_bucket": ("candidate_delta_abs_bucket",),
        "fold_weight_bucket": ("fold", "weight_bucket"),
        "fold_time_bucket": ("fold", "time_bucket"),
    }


def write_strategy_slice_scores(
    frame: pl.DataFrame,
    *,
    strategies: Sequence[tuple[str, str, str]],
    group_columns: Sequence[str],
    path: Path,
) -> None:
    frames = []
    for strategy, method_family, prediction in strategies:
        frames.append(
            _aggregate_prediction_by_slice(frame, group_columns=group_columns, prediction=prediction).with_columns(
                pl.lit(strategy).alias("strategy"),
                pl.lit(method_family).alias("method_family"),
            )
        )
    pl.concat(frames, how="diagonal").select(
        ["strategy", "method_family", *group_columns, "rows", "weight_sum", "numerator", "denominator", "weighted_zero_mean_r2"]
    ).sort([*group_columns, "weighted_zero_mean_r2"], descending=[False] * len(group_columns) + [True]).write_csv(path)


def _aggregate_prediction_by_slice(frame: pl.DataFrame, *, group_columns: Sequence[str], prediction: str) -> pl.DataFrame:
    return (
        frame.lazy()
        .group_by(list(group_columns))
        .agg(
            pl.len().alias("rows"),
            pl.col("weight").sum().alias("weight_sum"),
            (pl.col("weight") * (pl.col(TARGET) - pl.col(prediction)).pow(2)).sum().alias("numerator"),
            (pl.col("weight") * pl.col(TARGET).pow(2)).sum().alias("denominator"),
        )
        .with_columns((1.0 - pl.col("numerator") / pl.col("denominator")).alias("weighted_zero_mean_r2"))
        .collect()
    )


def compare_prediction_pair_by_slice(
    frame: pl.DataFrame,
    *,
    group_columns: Sequence[str],
    baseline_prediction: str,
    candidate_prediction: str,
) -> pl.DataFrame:
    group_columns = tuple(group_columns)
    return (
        frame.lazy()
        .group_by(list(group_columns))
        .agg(
            pl.len().alias("rows"),
            pl.col("weight").sum().alias("weight_sum"),
            (pl.col("weight") * pl.col(TARGET).pow(2)).sum().alias("denominator"),
            (pl.col("weight") * (pl.col(TARGET) - pl.col(baseline_prediction)).pow(2)).sum().alias("baseline_numerator"),
            (pl.col("weight") * (pl.col(TARGET) - pl.col(candidate_prediction)).pow(2)).sum().alias("candidate_numerator"),
        )
        .with_columns(
            (1.0 - pl.col("baseline_numerator") / pl.col("denominator")).alias("baseline_r2"),
            (1.0 - pl.col("candidate_numerator") / pl.col("denominator")).alias("candidate_r2"),
            (pl.col("baseline_numerator") - pl.col("candidate_numerator")).alias("numerator_improvement"),
        )
        .with_columns((pl.col("candidate_r2") - pl.col("baseline_r2")).alias("delta_r2"))
        .sort("delta_r2")
        .collect()
    )


def _bucket_thresholds(frame: pl.DataFrame) -> dict[str, list[str]]:
    return {
        "weight_bucket": sorted(frame["weight_bucket"].unique().to_list()),
        "reference_abs_pred_bucket": sorted(frame["reference_abs_pred_bucket"].unique().to_list()),
        "candidate_delta_abs_bucket": sorted(frame["candidate_delta_abs_bucket"].unique().to_list()),
    }


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
