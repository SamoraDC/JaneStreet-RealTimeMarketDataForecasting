"""Slice diagnostics for strong OOF candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multimodels.strong_oof_diagnostics import DiagnosticConfig, run_diagnostics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose strong OOF candidate deltas by slice.")
    parser.add_argument("--experiment-name", default="strong_oof_diagnostics")
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
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--candidate", default="gateway_risk_conservative_rls_abs_pred_s100_prediction")
    parser.add_argument("--baseline", default="conservative_rls_prediction")
    parser.add_argument("--gateway-risk-strengths", default="25,100")
    parser.add_argument("--gateway-risk-profiles", default="abs_pred")
    parser.add_argument("--time-bucket-size", type=int, default=100)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-rows-per-fold", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or Path("multi-models") / "reports" / args.experiment_name
    config = DiagnosticConfig(
        experiment_name=args.experiment_name,
        tabm_prediction_dir=args.tabm_prediction_dir,
        tree_prediction_dir=args.tree_prediction_dir,
        output_dir=output_dir,
        candidate=args.candidate,
        baseline=args.baseline,
        gateway_risk_strengths=_parse_float_tuple(args.gateway_risk_strengths),
        gateway_risk_profiles=_parse_str_tuple(args.gateway_risk_profiles),
        time_bucket_size=args.time_bucket_size,
        sample_stride=args.sample_stride,
        max_rows_per_fold=args.max_rows_per_fold,
    )
    result = run_diagnostics(config)
    print(result["summary"])
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
