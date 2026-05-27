# Multi-Model Experiment: strong_oof_subset_s23aux8_s17_gateway_batch_rank_smoke_stride5_v1

## Headline

- Best candidate: `strong_oof_ridge_stack_prediction_s0_risk_shrink`.
- Family: `risk_shrinkage`.
- Global weighted zero-mean R2: `0.014158651`.
- Mean fold R2: `0.013827625`.
- Min fold R2: `0.008014008`.

## Audit

- Folds: `5`.
- Model features: `14`.
- Uses processed lags: `False`.
- Uses context features: `True`.
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
