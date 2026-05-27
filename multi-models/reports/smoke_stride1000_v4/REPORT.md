# Multi-Model Experiment: smoke_stride1000_v4

## Headline

- Best candidate: `alpha_ridge_rank__risk_shrink_s0p05`.
- Family: `risk_shrinkage`.
- Global weighted zero-mean R2: `0.004673321`.
- Mean fold R2: `0.004652261`.
- Min fold R2: `0.003625977`.

## Audit

- Folds: `2`.
- Model features: `30`.
- Uses processed lags: `False`.
- Uses context features: `False`.
- Target leakage check: `passed`.
- Fold causality check: `passed`.
- Selection check: `passed: risk strengths and auxiliary bases are fixed by config, not selected inside validation`.
- Residual rules: `feature_04`.

## Methodological Status

- This is an experimental OOF validation stack, not yet a Kaggle runtime artifact.
- Risk and regime models are auxiliary: they modulate the alpha prediction; they do not change the required final output schema.
- Promotion requires comparison against the preserved conservative and historical checkpoints under the same fold protocol.
