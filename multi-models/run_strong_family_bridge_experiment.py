"""CLI for bridging family artifacts into the strong OOF candidate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multimodels.strong_family_bridge import StrongFamilyBridgeConfig, run_bridge_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge weak family artifacts into a strong OOF candidate.")
    parser.add_argument("--experiment-name", default="strong_family_bridge")
    parser.add_argument("--family-prediction-path", type=Path, default=StrongFamilyBridgeConfig.family_prediction_path)
    parser.add_argument("--tabm-prediction-dir", type=Path, default=StrongFamilyBridgeConfig.tabm_prediction_dir)
    parser.add_argument("--tree-prediction-dir", type=Path, default=StrongFamilyBridgeConfig.tree_prediction_dir)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--strong-base", default=StrongFamilyBridgeConfig.strong_base)
    parser.add_argument("--gateway-risk-strengths", default="100")
    parser.add_argument("--gateway-risk-profiles", default="abs_pred")
    parser.add_argument("--alpha-columns", default="latent_alpha_linear_stack,ridge_rank_alpha10000,pls_rank_k8")
    parser.add_argument(
        "--residual-feature-columns",
        default=(
            "latent_alpha_linear_stack,ridge_rank_alpha10000,"
            "ridge_rank_alpha10000__feature_59_z_residual,"
            "ridge_rank_alpha10000__risk_abs_error_ridge_rank_score,"
            "ridge_rank_alpha10000__risk_abs_responder6_ridge_rank_score,"
            "ridge_rank_alpha10000__risk_high_error_ridge_rank_score"
        ),
    )
    parser.add_argument(
        "--risk-columns",
        default=(
            "ridge_rank_alpha10000__risk_abs_error_ridge_rank_score,"
            "ridge_rank_alpha10000__risk_abs_responder6_ridge_rank_score,"
            "ridge_rank_alpha10000__risk_high_error_ridge_rank_score"
        ),
    )
    parser.add_argument("--stack-alpha", type=float, default=1000.0)
    parser.add_argument("--residual-alpha", type=float, default=10000.0)
    parser.add_argument("--risk-strengths", default="0.02,0.05,0.1,0.2")
    parser.add_argument("--time-bucket-size", type=int, default=100)
    parser.add_argument("--symbol-mod", type=int, default=8)
    parser.add_argument("--min-regime-rows", type=int, default=500)
    parser.add_argument("--regime-prior-strength", type=float, default=1000.0)
    parser.add_argument("--residual-gate-time-bucket-size", type=int, default=100)
    parser.add_argument("--residual-gate-symbol-mod", type=int, default=8)
    parser.add_argument("--residual-gate-min-rows", type=int, default=100)
    parser.add_argument("--residual-gate-prior-strength", type=float, default=1000.0)
    parser.add_argument("--residual-gate-min-delta", type=float, default=0.0)
    parser.add_argument("--residual-tail-quantiles", default="0.90,0.95,0.99")
    parser.add_argument("--no-write-predictions", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir or Path("multi-models") / "reports" / args.experiment_name
    config = StrongFamilyBridgeConfig(
        experiment_name=args.experiment_name,
        family_prediction_path=args.family_prediction_path,
        tabm_prediction_dir=args.tabm_prediction_dir,
        tree_prediction_dir=args.tree_prediction_dir,
        output_dir=output_dir,
        strong_base=args.strong_base,
        gateway_risk_strengths=_parse_float_tuple(args.gateway_risk_strengths),
        gateway_risk_profiles=_parse_str_tuple(args.gateway_risk_profiles),
        alpha_columns=_parse_str_tuple(args.alpha_columns),
        residual_feature_columns=_parse_str_tuple(args.residual_feature_columns),
        risk_columns=_parse_str_tuple(args.risk_columns),
        stack_alpha=args.stack_alpha,
        residual_alpha=args.residual_alpha,
        risk_strengths=_parse_float_tuple(args.risk_strengths),
        time_bucket_size=args.time_bucket_size,
        symbol_mod=args.symbol_mod,
        min_regime_rows=args.min_regime_rows,
        regime_prior_strength=args.regime_prior_strength,
        residual_gate_time_bucket_size=args.residual_gate_time_bucket_size,
        residual_gate_symbol_mod=args.residual_gate_symbol_mod,
        residual_gate_min_rows=args.residual_gate_min_rows,
        residual_gate_prior_strength=args.residual_gate_prior_strength,
        residual_gate_min_delta=args.residual_gate_min_delta,
        residual_tail_quantiles=_parse_float_tuple(args.residual_tail_quantiles),
        write_predictions=not args.no_write_predictions,
    )
    result = run_bridge_experiment(config)
    print(result["summary"].head(20))
    print(f"Wrote {result['output_dir']}")


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("at least one float value is required")
    return values


if __name__ == "__main__":
    main()
