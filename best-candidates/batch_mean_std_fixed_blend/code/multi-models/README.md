# Multi-Model Research Lab

This directory contains an isolated experimental stack for modular Jane Street
modeling. It does not modify the main `src/`, `scripts/`, `submission/`, or
notebook code paths.

The stack tests a single competition-compatible prediction built from internal
components:

- rank-latent alpha models: Ridge and PLS on causal rank encodings;
- residual correction: frozen univariate residual rules;
- risk/volatility model: predicts model error magnitude for shrinkage;
- regime gate: grouped scale calibration by observable context;
- microstructure/context features: current-row and current-batch features only.

The concrete artifact manifest is in `MODEL_FAMILY_ARTIFACTS.md`.

Auxiliary layers are evaluated per fixed base candidate. The default bases are
`ridge_rank_alpha10000` and `pls_rank_k8`, which prevents a weak stack from
hiding a stronger individual alpha model.

## Quick Smoke

```bash
uv run python multi-models/run_multimodel_experiment.py \
  --experiment-name smoke_stride500 \
  --sample-stride 500 \
  --n-folds 2 \
  --train-window 30 \
  --valid-window 10 \
  --max-train-rows 30000 \
  --max-valid-rows 15000 \
  --max-features 60 \
  --risk-auxiliary-targets abs_responder6,sq_responder6,abs_error,sq_error,high_error \
  --risk-strengths 0,0.05
```

Outputs are written under `multi-models/reports/<experiment-name>/`.

## Strong OOF Adapter

Use this path before training new models. It applies residual/risk/regime
layers to saved TabM/tree/gateway OOF predictions.

```bash
uv run python multi-models/run_strong_oof_experiment.py \
  --experiment-name strong_oof_stage3_smoke \
  --sample-stride 200 \
  --max-rows-per-fold 10000 \
  --time-bucket-sizes 100 \
  --min-group-rows 500 \
  --scale-prior-strengths 1000 \
  --risk-shrink-strengths 0,0.02,0.05
```

Full Stage 3:

```bash
uv run python multi-models/run_strong_oof_experiment.py \
  --experiment-name strong_oof_stage3_full \
  --time-bucket-sizes 50,100,200 \
  --min-group-rows 2000,10000 \
  --scale-prior-strengths 1000,10000 \
  --risk-shrink-strengths 0,0.02,0.05,0.1 \
  --risk-base-candidates baseline_prediction,conservative_rls_prediction,aggressive_rls_prediction,strong_oof_ridge_stack_prediction \
  --regime-base-candidates baseline_prediction,conservative_rls_prediction,aggressive_rls_prediction,strong_oof_ridge_stack_prediction \
  --write-predictions
```

Focused diagnostics for the current robust candidate:

```bash
uv run python multi-models/run_strong_oof_diagnostics.py \
  --experiment-name strong_oof_stage3_conservative_s100_diagnostics \
  --candidate gateway_risk_conservative_rls_abs_pred_s100_prediction \
  --baseline conservative_rls_prediction
```

Focused tail/blend validation over full strong OOF frames:

```bash
uv run python multi-models/run_strong_oof_experiment.py \
  --experiment-name strong_oof_stage3_gateway_tail_fixed_blend_v1 \
  --include-gateway-risk-shrink \
  --gateway-risk-strengths 25,100 \
  --gateway-risk-profiles abs_pred \
  --strong-base-candidates conservative_rls_prediction,aggressive_rls_prediction \
  --residual-base-candidates gateway_risk_conservative_rls_abs_pred_s100_prediction \
  --residual-features prediction_disagreement,tabm_tree_diff,abs_baseline_prediction,weight \
  --residual-tail-quantiles 0.95,0.99 \
  --residual-tail-modes weight_and_abs,abs_base \
  --fixed-blend-candidates gateway_risk_aggressive_rls_abs_pred_s25_prediction,gateway_risk_conservative_rls_abs_pred_s100_prediction,gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail_prediction,gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_abs_base_q0p99_residual_tail_prediction \
  --fixed-blend-weights 0.25,0.5,0.75 \
  --risk-base-candidates '' \
  --regime-base-candidates ''
```

The current Stage 3 exploratory leader from this path is
`fixed_blend_2_w0p5_fixed_blend` at `global_r2=0.013895762`; see
`reports/STRONG_OOF_TAIL_BLEND_SYNTHESIS.md` for the audit and limitation.

## Strong Family Bridge

Use this after generating `family_artifacts_5fold_lags_stride100_v2`. It tests
whether weak family alphas and risk models improve the strong gateway candidate
as residual/gating signals. The current surviving bridge hypothesis is the
tail-only residual correction, especially `weight q0.95`, which applies the
residual bridge only on thresholds learned from earlier folds.

```bash
uv run python multi-models/run_strong_family_bridge_experiment.py \
  --experiment-name strong_family_bridge_stride100_residual_minimal_v1 \
  --family-prediction-path multi-models/reports/family_artifacts_5fold_lags_stride100_v2/validation_predictions.parquet \
  --strong-base gateway_risk_conservative_rls_abs_pred_s100_prediction \
  --gateway-risk-strengths 100 \
  --gateway-risk-profiles abs_pred \
  --alpha-columns latent_alpha_linear_stack,ridge_rank_alpha10000 \
  --residual-feature-columns latent_alpha_linear_stack,ridge_rank_alpha10000,ridge_rank_alpha10000__risk_high_error_ridge_rank_score \
  --risk-columns ridge_rank_alpha10000__risk_high_error_ridge_rank_score \
  --risk-strengths 0.02,0.05,0.1 \
  --min-regime-rows 500 \
  --regime-prior-strength 1000 \
  --time-bucket-size 100 \
  --symbol-mod 8 \
  --residual-gate-time-bucket-size 100 \
  --residual-gate-symbol-mod 8 \
  --residual-gate-min-rows 100 \
  --residual-gate-prior-strength 1000 \
  --residual-tail-quantiles 0.90,0.95,0.99
```

Bridge diagnostics compare a frozen bridge candidate against the frozen strong
base by fold, weight, prediction magnitude, time bucket, and symbol:

```bash
uv run python multi-models/run_strong_family_bridge_diagnostics.py \
  --experiment-name strong_family_bridge_stride100_residual_minimal_v1_diagnostics \
  --bridge-prediction-path multi-models/reports/strong_family_bridge_stride100_residual_minimal_v1/bridge_predictions.parquet \
  --candidate gateway_risk_conservative_rls_abs_pred_s100_prediction_family_residual_prediction \
  --baseline gateway_risk_conservative_rls_abs_pred_s100_prediction
```

Tail-only diagnostic for the current sampled bridge leader:

```bash
uv run python multi-models/run_strong_family_bridge_diagnostics.py \
  --experiment-name strong_family_bridge_stride50_tail_weight_q0p95_diagnostics \
  --bridge-prediction-path multi-models/reports/strong_family_bridge_stride50_tail_v1/bridge_predictions.parquet \
  --candidate gateway_risk_conservative_rls_abs_pred_s100_prediction_family_residual_weight_q0p95_family_residual_tail_prediction \
  --baseline gateway_risk_conservative_rls_abs_pred_s100_prediction
```

## Full Protocol

After smoke passes, increase data gradually:

```bash
uv run python multi-models/run_multimodel_experiment.py \
  --experiment-name stage3_rank_risk_regime_5fold \
  --sample-stride 1 \
  --n-folds 5 \
  --train-window 120 \
  --valid-window 60 \
  --max-features 79 \
  --use-processed-lags \
  --write-predictions
```

Promotion requires improvement versus the preserved conservative checkpoint and
no evidence of leakage in the generated audit report.
