"""CLI for modular experiments over existing strong OOF prediction artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multimodels.strong_oof import StrongOOFConfig, run_strong_oof_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strong OOF modular risk/regime/residual experiments.")
    parser.add_argument("--experiment-name", default="strong_oof_modular")
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
    parser.add_argument("--extra-prediction-dirs", default="")
    parser.add_argument("--extra-prediction-prefixes", default="")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-gateway-rls", action="store_true")
    parser.add_argument("--include-gateway-risk-shrink", action="store_true")
    parser.add_argument("--include-extra-gateway-experts", action="store_true")
    parser.add_argument("--gateway-expert-expansions", default="")
    parser.add_argument("--gateway-risk-strengths", default="25,100")
    parser.add_argument("--gateway-risk-profiles", default="abs_pred")
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-rows-per-fold", type=int, default=None)
    parser.add_argument("--time-bucket-sizes", default="100")
    parser.add_argument("--min-group-rows", default="20000")
    parser.add_argument("--scale-prior-strengths", default="1000,10000")
    parser.add_argument("--stack-alphas", default="1000")
    parser.add_argument("--risk-shrink-strengths", default="0,0.02,0.05,0.1")
    parser.add_argument(
        "--strong-base-candidates",
        default=(
            "tabm_prediction,tree_prediction,xgboost_prediction,lightgbm_prediction,"
            "ridge_calibrated_prediction,baseline_prediction,conservative_rls_prediction,aggressive_rls_prediction"
        ),
    )
    parser.add_argument("--residual-features", default="prediction_disagreement,tabm_tree_diff,abs_baseline_prediction,weight")
    parser.add_argument("--residual-base-candidates", default="")
    parser.add_argument("--residual-tail-quantiles", default="")
    parser.add_argument("--residual-tail-modes", default="weight")
    parser.add_argument(
        "--risk-base-candidates",
        default="baseline_prediction,conservative_rls_prediction,aggressive_rls_prediction,strong_oof_ridge_stack_prediction",
    )
    parser.add_argument(
        "--regime-base-candidates",
        default="baseline_prediction,conservative_rls_prediction,aggressive_rls_prediction,strong_oof_ridge_stack_prediction",
    )
    parser.add_argument("--raw-train-parquet-dir", type=Path, default=None)
    parser.add_argument("--raw-feature-columns", default="")
    parser.add_argument("--raw-preprocess-modes", default="")
    parser.add_argument("--include-raw-preprocessed-in-stack", action="store_true")
    parser.add_argument("--fixed-blend-candidates", default="")
    parser.add_argument("--fixed-blend-weights", default="")
    parser.add_argument("--walk-forward-blend-candidates", default="")
    parser.add_argument("--contextual-blend-candidates", default="")
    parser.add_argument("--contextual-blend-group-specs", default="")
    parser.add_argument("--contextual-blend-time-bucket-sizes", default="100")
    parser.add_argument("--contextual-blend-min-group-rows", default="20000")
    parser.add_argument("--contextual-blend-prior-strengths", default="1000")
    parser.add_argument("--online-scale-base-candidates", default="")
    parser.add_argument("--online-scale-prior-strengths", default="1000")
    parser.add_argument("--online-scale-forgetting-factors", default="1.0")
    parser.add_argument("--online-scale-min", type=float, default=0.0)
    parser.add_argument("--online-scale-max", type=float, default=2.0)
    parser.add_argument("--online-affine-base-candidates", default="")
    parser.add_argument("--online-affine-prior-strengths", default="1000")
    parser.add_argument("--online-affine-forgetting-factors", default="1.0")
    parser.add_argument("--online-affine-min-scale", type=float, default=0.0)
    parser.add_argument("--online-affine-max-scale", type=float, default=2.0)
    parser.add_argument("--min-mem-available-gb", type=float, default=0.0)
    parser.add_argument("--min-swap-free-gb", type=float, default=0.0)
    parser.add_argument("--write-predictions", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or Path("multi-models") / "reports" / args.experiment_name
    config = StrongOOFConfig(
        experiment_name=args.experiment_name,
        tabm_prediction_dir=args.tabm_prediction_dir,
        tree_prediction_dir=args.tree_prediction_dir,
        extra_prediction_dirs=_parse_path_tuple(args.extra_prediction_dirs),
        extra_prediction_prefixes=_parse_str_tuple(args.extra_prediction_prefixes),
        output_dir=output_dir,
        include_gateway_rls=not args.no_gateway_rls,
        include_gateway_risk_shrink=args.include_gateway_risk_shrink,
        include_extra_gateway_experts=args.include_extra_gateway_experts,
        gateway_expert_expansions=_parse_str_tuple(args.gateway_expert_expansions),
        gateway_risk_strengths=_parse_float_tuple(args.gateway_risk_strengths),
        gateway_risk_profiles=_parse_str_tuple(args.gateway_risk_profiles),
        sample_stride=args.sample_stride,
        max_rows_per_fold=args.max_rows_per_fold,
        time_bucket_sizes=_parse_int_tuple(args.time_bucket_sizes),
        min_group_rows=_parse_int_tuple(args.min_group_rows),
        scale_prior_strengths=_parse_float_tuple(args.scale_prior_strengths),
        stack_alphas=_parse_float_tuple(args.stack_alphas),
        risk_shrink_strengths=_parse_float_tuple(args.risk_shrink_strengths),
        strong_base_candidates=_parse_str_tuple(args.strong_base_candidates),
        residual_features=_parse_str_tuple(args.residual_features),
        residual_base_candidates=_parse_str_tuple(args.residual_base_candidates),
        residual_tail_quantiles=_parse_optional_float_tuple(args.residual_tail_quantiles),
        residual_tail_modes=_parse_str_tuple(args.residual_tail_modes),
        risk_base_candidates=_parse_str_tuple(args.risk_base_candidates),
        regime_base_candidates=_parse_str_tuple(args.regime_base_candidates),
        raw_train_parquet_dir=args.raw_train_parquet_dir,
        raw_feature_columns=_parse_str_tuple(args.raw_feature_columns),
        raw_preprocess_modes=_parse_str_tuple(args.raw_preprocess_modes),
        include_raw_preprocessed_in_stack=args.include_raw_preprocessed_in_stack,
        fixed_blend_candidates=_parse_str_tuple(args.fixed_blend_candidates),
        fixed_blend_weights=_parse_optional_float_tuple(args.fixed_blend_weights),
        walk_forward_blend_candidates=_parse_str_tuple(args.walk_forward_blend_candidates),
        contextual_blend_candidates=_parse_str_tuple(args.contextual_blend_candidates),
        contextual_blend_group_specs=_parse_str_tuple(args.contextual_blend_group_specs),
        contextual_blend_time_bucket_sizes=_parse_int_tuple(args.contextual_blend_time_bucket_sizes),
        contextual_blend_min_group_rows=_parse_int_tuple(args.contextual_blend_min_group_rows),
        contextual_blend_prior_strengths=_parse_float_tuple(args.contextual_blend_prior_strengths),
        online_scale_base_candidates=_parse_str_tuple(args.online_scale_base_candidates),
        online_scale_prior_strengths=_parse_float_tuple(args.online_scale_prior_strengths),
        online_scale_forgetting_factors=_parse_float_tuple(args.online_scale_forgetting_factors),
        online_scale_min=args.online_scale_min,
        online_scale_max=args.online_scale_max,
        online_affine_base_candidates=_parse_str_tuple(args.online_affine_base_candidates),
        online_affine_prior_strengths=_parse_float_tuple(args.online_affine_prior_strengths),
        online_affine_forgetting_factors=_parse_float_tuple(args.online_affine_forgetting_factors),
        online_affine_min_scale=args.online_affine_min_scale,
        online_affine_max_scale=args.online_affine_max_scale,
        min_mem_available_gb=args.min_mem_available_gb,
        min_swap_free_gb=args.min_swap_free_gb,
        write_predictions=args.write_predictions,
    )
    result = run_strong_oof_experiment(config)
    print(result["summary"].head(20))
    print(f"Wrote {result['output_dir']}")


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_path_tuple(raw: str) -> tuple[Path, ...]:
    return tuple(Path(part.strip()) for part in raw.split(",") if part.strip())


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("at least one float value is required")
    return values


def _parse_optional_float_tuple(raw: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in raw.split(",") if part.strip())


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("at least one integer value is required")
    return values


if __name__ == "__main__":
    main()
