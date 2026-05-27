# Multi-Model Experiment: strong_oof_hist_max1398_gateway_residual_tail_v1

## Headline

- Best candidate: `gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_q0p95_residual_tail`.
- Family: `residual_tail`.
- Global weighted zero-mean R2: `0.015607877`.
- Mean fold R2: `0.015403261`.
- Min fold R2: `0.012894355`.

## Audit

- Folds: `5`.
- Model features: `6`.
- Uses processed lags: `False`.
- Uses context features: `True`.
- Target leakage check: `passed`.
- Fold causality check: `passed: walk-forward layers and residual tail thresholds fit only earlier folds; gateway updates use prior-date lag simulation`.
- Selection check: `passed: all grids are reported as fixed candidates, no validation-best candidate is promoted inside code`.
- Residual rules: `prediction_disagreement, tabm_tree_diff, abs_baseline_prediction, weight`.
- Risk auxiliary targets: `none`.

## Artifact Manifest


## Methodological Status

- This is an experimental OOF validation stack, not yet a Kaggle runtime artifact.
- Risk and regime models are auxiliary: they modulate the alpha prediction; they do not change the required final output schema.
- Promotion requires comparison against the preserved conservative and historical checkpoints under the same fold protocol.
