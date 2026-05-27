# Multi-Model Experiment: strong_oof_hist_max1398_stack_confirm

## Headline

- Best candidate: `conservative_rls_prediction`.
- Family: `strong_base`.
- Global weighted zero-mean R2: `0.015425373`.
- Mean fold R2: `0.015235126`.
- Min fold R2: `0.012924771`.

## Audit

- Folds: `5`.
- Model features: `5`.
- Uses processed lags: `False`.
- Uses context features: `True`.
- Target leakage check: `passed`.
- Fold causality check: `passed: walk-forward layers fit only earlier folds; gateway updates use prior-date lag simulation`.
- Selection check: `passed: all grids are reported as fixed candidates, no validation-best candidate is promoted inside code`.
- Residual rules: `none`.

## Methodological Status

- This is an experimental OOF validation stack, not yet a Kaggle runtime artifact.
- Risk and regime models are auxiliary: they modulate the alpha prediction; they do not change the required final output schema.
- Promotion requires comparison against the preserved conservative and historical checkpoints under the same fold protocol.
