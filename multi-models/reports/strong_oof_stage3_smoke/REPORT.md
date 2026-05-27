# Multi-Model Experiment: strong_oof_stage3_smoke

## Headline

- Best candidate: `tabm_prediction`.
- Family: `strong_base`.
- Global weighted zero-mean R2: `0.017038958`.
- Mean fold R2: `0.017285466`.
- Min fold R2: `0.004220128`.

## Audit

- Folds: `5`.
- Model features: `8`.
- Uses processed lags: `False`.
- Uses context features: `True`.
- Target leakage check: `passed`.
- Fold causality check: `passed: walk-forward layers fit only earlier folds; gateway updates use prior-date lag simulation`.
- Selection check: `passed: all grids are reported as fixed candidates, no validation-best candidate is promoted inside code`.
- Residual rules: `prediction_disagreement, tabm_tree_diff, abs_baseline_prediction, weight`.

## Methodological Status

- This is an experimental OOF validation stack, not yet a Kaggle runtime artifact.
- Risk and regime models are auxiliary: they modulate the alpha prediction; they do not change the required final output schema.
- Promotion requires comparison against the preserved conservative and historical checkpoints under the same fold protocol.
