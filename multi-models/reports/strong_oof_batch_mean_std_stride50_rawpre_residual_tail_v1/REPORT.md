# Multi-Model Experiment: strong_oof_batch_mean_std_stride50_rawpre_residual_tail_v1

## Headline

- Best candidate: `strong_oof_ridge_stack_prediction_residual_weight_q0p95_residual_tail`.
- Family: `residual_tail`.
- Global weighted zero-mean R2: `0.014214327`.
- Mean fold R2: `0.013973502`.
- Min fold R2: `0.008345827`.

## Audit

- Folds: `5`.
- Model features: `14`.
- Uses processed lags: `False`.
- Uses context features: `True`.
- Raw preprocessed features: `0`.
- Raw preprocessing modes: `none`.
- Target leakage check: `passed`.
- Fold causality check: `passed: walk-forward layers and residual tail thresholds fit only earlier folds; gateway updates use prior-date lag simulation; online scale/affine calibration updates only after each validation date is scored`.
- Selection check: `passed: all grids are reported as fixed candidates, no validation-best candidate is promoted inside code`.
- Residual rules: `rawpre_latent_alpha_linear_stack, rawpre_ridge_rank_alpha10000, rawpre_pls_rank_k8`.
- Risk auxiliary targets: `none`.

## Artifact Manifest


## Methodological Status

- This is an experimental OOF validation stack, not yet a Kaggle runtime artifact.
- Risk and regime models are auxiliary: they modulate the alpha prediction; they do not change the required final output schema.
- Promotion requires comparison against the preserved conservative and historical checkpoints under the same fold protocol.
