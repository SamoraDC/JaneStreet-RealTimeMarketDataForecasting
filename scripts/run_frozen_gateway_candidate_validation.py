"""Frozen validation for the two selected gateway Bayesian meta candidates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import polars as pl

from janestreet.bayesian_meta import score_prediction_by_fold, summarize_fold_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class FrozenGatewayCandidate:
    """A candidate selected before this validation script runs."""

    name: str
    feature_set: str
    ridge_alpha: float
    rationale: str


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
    parser.add_argument("--output-dir", type=Path, default=Path("reports/experiments/frozen_gateway_candidate_validation_stage3"))
    parser.add_argument("--experiment-name", default="frozen_gateway_candidate_validation")
    args = parser.parse_args()

    gateway = _load_gateway_module()
    frame = gateway._load_joined_predictions(args.tabm_prediction_dir, args.tree_prediction_dir)
    frame, convex_parameters = gateway._add_walk_forward_convex_baseline(frame)
    expert_columns = tuple(column for column in gateway._expert_columns(frame) if column != "baseline_prediction")
    feature_sets = gateway._gateway_feature_sets(expert_columns)
    folds = tuple(gateway._folds(frame))
    candidates = _frozen_candidates()
    _validate_candidates(candidates, feature_sets)

    by_fold_frames: list[pl.DataFrame] = []
    summary_rows: list[dict[str, float | int | str]] = []
    parameter_rows: list[dict[str, float | int | str | None]] = list(convex_parameters)
    audit_frames: list[pl.DataFrame] = []

    baseline_scores = _score_baseline(frame)
    by_fold_frames.append(baseline_scores)
    summary_rows.append({"strategy": "tabm_tree_convex_walk_forward", "method_family": "baseline", **summarize_fold_scores(baseline_scores)})

    for candidate in candidates:
        strategy = _strategy_name(candidate)
        scores, params, audit = gateway._evaluate_gateway_online_ridge(
            frame,
            feature_columns=feature_sets[candidate.feature_set],
            ridge_alpha=candidate.ridge_alpha,
        )
        scores = scores.with_columns(
            pl.lit(strategy).alias("strategy"),
            pl.lit("frozen_gateway_online_ridge").alias("method_family"),
        )
        by_fold_frames.append(scores)
        summary_rows.append({"strategy": strategy, "method_family": "frozen_gateway_online_ridge", **summarize_fold_scores(scores)})
        parameter_rows.extend({"strategy": strategy, **row} for row in params)
        audit_frames.append(audit.with_columns(pl.lit(strategy).alias("strategy")))

    by_fold = pl.concat(by_fold_frames, how="diagonal").select(
        ["strategy", "method_family", "fold", "rows", "weight_sum", "numerator", "denominator", "weighted_zero_mean_r2"]
    )
    summary = pl.DataFrame(summary_rows).sort(["global_r2", "min_fold_r2"], descending=[True, True])
    subset_summary = summarize_candidate_subsets(by_fold, _default_subset_specs(folds))
    parameters = pl.DataFrame(parameter_rows) if parameter_rows else pl.DataFrame()
    audit_frame = pl.concat(audit_frames, how="diagonal") if audit_frames else pl.DataFrame()
    audit_status = _audit_status(audit_frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.output_dir / "frozen_gateway_summary.csv")
    by_fold.write_csv(args.output_dir / "frozen_gateway_by_fold.csv")
    subset_summary.write_csv(args.output_dir / "frozen_gateway_subset_summary.csv")
    if not parameters.is_empty():
        parameters.write_csv(args.output_dir / "frozen_gateway_parameters.csv")
    if not audit_frame.is_empty():
        audit_frame.write_csv(args.output_dir / "frozen_gateway_daily_audit.csv")

    report = {
        "experiment": args.experiment_name,
        "rows": frame.height,
        "folds": folds,
        "expert_columns": expert_columns,
        "frozen_candidates": [asdict(candidate) for candidate in candidates],
        "subsets": {name: list(subset_folds) for name, subset_folds in _default_subset_specs(folds).items()},
        "best_full": summary.row(0, named=True),
        "audit_status": audit_status,
        "validation_caveat": (
            "This freezes two candidates selected in the previous gateway ablation and re-scores them on existing OOF folds. "
            "It is useful for stability and leakage audit, but it is not a clean historical holdout because the candidates were selected after seeing this OOF family."
        ),
    }
    (args.output_dir / "frozen_gateway_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summary)
    print(subset_summary)
    print(f"Wrote {args.output_dir}")


def _load_gateway_module():
    path = PROJECT_ROOT / "scripts" / "run_bayesian_gateway_meta_simulation.py"
    spec = importlib.util.spec_from_file_location("run_bayesian_gateway_meta_simulation_for_frozen_validation", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frozen_candidates() -> tuple[FrozenGatewayCandidate, ...]:
    return (
        FrozenGatewayCandidate(
            name="components_no_tree_ensemble_alpha1000",
            feature_set="components_no_tree_ensemble",
            ridge_alpha=1000.0,
            rationale="best global R2 in the prior strict gateway ablation",
        ),
        FrozenGatewayCandidate(
            name="experts_alpha10000",
            feature_set="experts",
            ridge_alpha=10000.0,
            rationale="best worst-fold stability among the leading prior gateway ablations",
        ),
    )


def _validate_candidates(candidates: Sequence[FrozenGatewayCandidate], feature_sets: dict[str, tuple[str, ...]]) -> None:
    if len(candidates) != 2:
        raise ValueError("this validation must remain frozen to exactly two candidates")
    missing = [candidate.feature_set for candidate in candidates if candidate.feature_set not in feature_sets]
    if missing:
        raise ValueError(f"missing feature sets for frozen candidates: {missing}")


def _strategy_name(candidate: FrozenGatewayCandidate) -> str:
    return f"frozen_gateway_online_ridge_{candidate.name}"


def _score_baseline(frame: pl.DataFrame) -> pl.DataFrame:
    return score_prediction_by_fold(frame, prediction="baseline_prediction").with_columns(
        pl.lit("tabm_tree_convex_walk_forward").alias("strategy"),
        pl.lit("baseline").alias("method_family"),
    )


def _default_subset_specs(folds: Sequence[str]) -> dict[str, tuple[str, ...]]:
    ordered = tuple(folds)
    if not ordered:
        raise ValueError("at least one fold is required")
    specs: dict[str, tuple[str, ...]] = {"full": ordered}
    if len(ordered) >= 3:
        specs[f"early_{ordered[0]}_{ordered[2]}"] = ordered[:3]
    if len(ordered) >= 2:
        specs[f"late_{ordered[-2]}_{ordered[-1]}"] = ordered[-2:]
    specs[f"last_{ordered[-1]}"] = (ordered[-1],)
    return specs


def summarize_candidate_subsets(by_fold: pl.DataFrame, subsets: dict[str, tuple[str, ...]]) -> pl.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    strategies = list(by_fold.select(["strategy", "method_family"]).unique().sort(["strategy", "method_family"]).iter_rows(named=True))
    for subset_name, subset_folds in subsets.items():
        subset = by_fold.filter(pl.col("fold").is_in(list(subset_folds)))
        if subset.is_empty():
            continue
        for strategy in strategies:
            scores = subset.filter((pl.col("strategy") == strategy["strategy"]) & (pl.col("method_family") == strategy["method_family"]))
            if scores.is_empty():
                continue
            rows.append(
                {
                    "subset": subset_name,
                    "subset_folds": ",".join(subset_folds),
                    "strategy": str(strategy["strategy"]),
                    "method_family": str(strategy["method_family"]),
                    **summarize_fold_scores(scores),
                }
            )
    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(["subset", "global_r2", "min_fold_r2"], descending=[False, True, True])


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
