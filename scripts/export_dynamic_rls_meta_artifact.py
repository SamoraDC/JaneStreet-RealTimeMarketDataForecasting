"""Export a submission RLS meta-state from saved OOF prediction artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import polars as pl

from janestreet.bayesian_meta import score_prediction_by_fold, summarize_fold_scores
from janestreet.submission_artifacts import fit_initial_rls_state_from_oof, save_rls_state_artifact


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    parser.add_argument("--feature-set", choices=["experts", "components_no_tree_ensemble"], default="experts")
    parser.add_argument("--ridge-alpha", type=float, default=10000.0)
    parser.add_argument("--forgetting-factor", type=float, default=0.995)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/jane_street_submission/meta_rls_experts_alpha10000_f0p995"))
    args = parser.parse_args()

    gateway = _load_gateway_module()
    frame = gateway._load_joined_predictions(args.tabm_prediction_dir, args.tree_prediction_dir)
    frame, convex_parameters = gateway._add_walk_forward_convex_baseline(frame)
    expert_columns = tuple(column for column in gateway._expert_columns(frame) if column != "baseline_prediction")
    feature_sets = gateway._gateway_feature_sets(expert_columns)
    feature_columns = feature_sets[args.feature_set]
    state = fit_initial_rls_state_from_oof(
        frame,
        feature_columns=feature_columns,
        ridge_alpha=args.ridge_alpha,
        forgetting_factor=args.forgetting_factor,
    )
    metadata = {
        "artifact_type": "dynamic_rls_meta_state",
        "feature_set": args.feature_set,
        "feature_columns": feature_columns,
        "ridge_alpha": args.ridge_alpha,
        "forgetting_factor": args.forgetting_factor,
        "tabm_prediction_dir": str(args.tabm_prediction_dir),
        "tree_prediction_dir": str(args.tree_prediction_dir),
        "oof_rows": frame.height,
        "oof_folds": frame["fold"].n_unique(),
        "baseline_summary": summarize_fold_scores(score_prediction_by_fold(frame, prediction="baseline_prediction")),
        "convex_parameter_rows": len(convex_parameters),
        "causality_note": (
            "This state is fitted from saved OOF prediction rows only. Online submission updates must still use "
            "previous-day lagged responders joined to cached previous-day base predictions."
        ),
    }
    save_rls_state_artifact(args.output_dir, state, metadata=metadata)
    (args.output_dir / "meta_rls_artifact_report.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    pl.DataFrame(
        {
            "feature": state.feature_columns,
            "beta": state.beta,
            "rhs": state.rhs,
        }
    ).write_csv(args.output_dir / "meta_rls_coefficients.csv")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {args.output_dir}")


def _load_gateway_module():
    path = PROJECT_ROOT / "scripts" / "run_bayesian_gateway_meta_simulation.py"
    spec = importlib.util.spec_from_file_location("run_bayesian_gateway_meta_simulation_for_export", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
