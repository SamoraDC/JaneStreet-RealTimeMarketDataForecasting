"""CLI entrypoint for the isolated multi-model Jane Street lab."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(LOCAL_PACKAGE_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from janestreet.paths import TRAIN_PARQUET_DIR  # noqa: E402
from multimodels.pipeline import ExperimentConfig, run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run modular alpha/risk/regime validation experiments.")
    parser.add_argument("--experiment-name", default="multi_model_lab")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--train-parquet-dir", type=Path, default=TRAIN_PARQUET_DIR)
    parser.add_argument("--use-processed-lags", action="store_true")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--train-window", type=int, default=120)
    parser.add_argument("--valid-window", type=int, default=60)
    parser.add_argument("--gap", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-valid-rows", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--rank-bins", type=int, default=255)
    parser.add_argument("--ridge-alpha", type=float, default=10000.0)
    parser.add_argument("--pls-components", type=int, default=8)
    parser.add_argument("--stack-alpha", type=float, default=1000.0)
    parser.add_argument("--risk-alpha", type=float, default=1000.0)
    parser.add_argument("--risk-auxiliary-targets", default="abs_responder6,sq_responder6,abs_error,sq_error,high_error")
    parser.add_argument("--high-error-quantile", type=float, default=0.90)
    parser.add_argument("--risk-strengths", default="0,0.02,0.05,0.1,0.2")
    parser.add_argument("--residual-features", default="feature_47,feature_59,feature_04")
    parser.add_argument("--auxiliary-base-candidates", default="ridge_rank_alpha10000,pls_rank_k8")
    parser.add_argument("--no-context-features", action="store_true")
    parser.add_argument("--cross-sectional-features", default="feature_04,feature_47,feature_59")
    parser.add_argument("--time-bucket-size", type=int, default=100)
    parser.add_argument("--raw-preprocess-features", default="")
    parser.add_argument("--raw-preprocess-modes", default="")
    parser.add_argument("--min-regime-rows", type=int, default=2000)
    parser.add_argument("--regime-prior-strength", type=float, default=1000.0)
    parser.add_argument("--regime-symbol-mod", type=int, default=8)
    parser.add_argument("--write-predictions", action="store_true")
    parser.add_argument("--seed", type=int, default=37)
    args = parser.parse_args()

    output_dir = args.output_dir or Path("multi-models") / "reports" / args.experiment_name
    config = ExperimentConfig(
        experiment_name=args.experiment_name,
        output_dir=output_dir,
        train_parquet_dir=args.train_parquet_dir,
        use_processed_lags=args.use_processed_lags,
        n_folds=args.n_folds,
        train_window=args.train_window,
        valid_window=args.valid_window,
        gap=args.gap,
        sample_stride=args.sample_stride,
        max_train_rows=args.max_train_rows,
        max_valid_rows=args.max_valid_rows,
        max_features=args.max_features,
        rank_bins=args.rank_bins,
        ridge_alpha=args.ridge_alpha,
        pls_components=args.pls_components,
        stack_alpha=args.stack_alpha,
        risk_alpha=args.risk_alpha,
        risk_auxiliary_targets=_parse_str_tuple(args.risk_auxiliary_targets),
        high_error_quantile=args.high_error_quantile,
        risk_strengths=_parse_float_tuple(args.risk_strengths),
        residual_features=_parse_str_tuple(args.residual_features),
        auxiliary_base_candidates=_parse_str_tuple(args.auxiliary_base_candidates),
        include_context_features=not args.no_context_features,
        cross_sectional_features=_parse_str_tuple(args.cross_sectional_features),
        time_bucket_size=args.time_bucket_size,
        raw_preprocess_features=_parse_str_tuple(args.raw_preprocess_features),
        raw_preprocess_modes=_parse_str_tuple(args.raw_preprocess_modes),
        min_regime_rows=args.min_regime_rows,
        regime_prior_strength=args.regime_prior_strength,
        regime_symbol_mod=args.regime_symbol_mod,
        write_predictions=args.write_predictions,
        seed=args.seed,
    )
    result = run_experiment(config)
    summary = result["summary"]
    print(summary.head(12))
    print(f"Wrote {result['output_dir']}")


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("at least one float value is required")
    return values


def _parse_str_tuple(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


if __name__ == "__main__":
    main()
