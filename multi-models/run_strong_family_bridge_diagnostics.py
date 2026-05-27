"""CLI for strong-family bridge slice diagnostics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from multimodels.strong_family_bridge_diagnostics import BridgeDiagnosticConfig, run_bridge_diagnostics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose a frozen strong-family bridge candidate by slice.")
    parser.add_argument("--experiment-name", default="strong_family_bridge_diagnostics")
    parser.add_argument("--bridge-prediction-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", default="gateway_risk_conservative_rls_abs_pred_s100_prediction")
    parser.add_argument("--time-bucket-size", type=int, default=100)
    args = parser.parse_args()

    output_dir = args.output_dir or args.bridge_prediction_path.parent
    result = run_bridge_diagnostics(
        BridgeDiagnosticConfig(
            experiment_name=args.experiment_name,
            bridge_prediction_path=args.bridge_prediction_path,
            output_dir=output_dir,
            candidate=args.candidate,
            baseline=args.baseline,
            time_bucket_size=args.time_bucket_size,
        )
    )
    print(result["summary"])
    print(f"Wrote {result['output_dir']}")


if __name__ == "__main__":
    main()
