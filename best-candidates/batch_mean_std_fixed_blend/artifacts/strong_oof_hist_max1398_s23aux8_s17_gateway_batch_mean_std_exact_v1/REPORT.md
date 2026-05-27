# Multi-Model Experiment: strong_oof_hist_max1398_s23aux8_s17_gateway_batch_mean_std_exact_v1

## Headline

- Best candidate: `fixed_blend_0_w0p75_fixed_blend`.
- Family: `fixed_blend`.
- Global weighted zero-mean R2: `0.015621628`.
- Mean fold R2: `0.015428592`.
- Min fold R2: `0.011794008`.

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
- Residual rules: `prediction_disagreement, tabm_tree_diff, abs_baseline_prediction, weight`.
- Risk auxiliary targets: `none`.

## Artifact Manifest


## Methodological Status

- This is an experimental OOF validation stack, not yet a Kaggle runtime artifact.
- Risk and regime models are auxiliary: they modulate the alpha prediction; they do not change the required final output schema.
- Promotion requires comparison against the preserved conservative and historical checkpoints under the same fold protocol.
