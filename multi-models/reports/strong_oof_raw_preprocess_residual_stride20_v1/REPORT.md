# Multi-Model Experiment: strong_oof_raw_preprocess_residual_stride20_v1

## Headline

- Best candidate: `strong_oof_ridge_stack`.
- Family: `strong_stack`.
- Global weighted zero-mean R2: `0.014023341`.
- Mean fold R2: `0.014015552`.
- Min fold R2: `0.008019192`.

## Audit

- Folds: `5`.
- Model features: `14`.
- Uses processed lags: `False`.
- Uses context features: `True`.
- Target leakage check: `passed`.
- Fold causality check: `passed: walk-forward layers and residual tail thresholds fit only earlier folds; gateway updates use prior-date lag simulation; online scale/affine calibration updates only after each validation date is scored`.
- Selection check: `passed: all grids are reported as fixed candidates, no validation-best candidate is promoted inside code`.
- Residual rules: `feature_47__raw_batch_zscore, feature_60__raw_batch_zscore, feature_06__raw_batch_zscore, feature_59__raw_batch_zscore, feature_58__raw_batch_zscore, feature_49__raw_batch_zscore, feature_04__raw_batch_zscore, feature_48__raw_batch_zscore, feature_16__raw_batch_zscore, raw_row_missing_count, raw_row_abs_mean, raw_row_l2_energy`.
- Risk auxiliary targets: `none`.

## Artifact Manifest


## Methodological Status

- This is an experimental OOF validation stack, not yet a Kaggle runtime artifact.
- Risk and regime models are auxiliary: they modulate the alpha prediction; they do not change the required final output schema.
- Promotion requires comparison against the preserved conservative and historical checkpoints under the same fold protocol.
