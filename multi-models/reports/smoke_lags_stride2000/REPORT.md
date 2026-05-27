# Multi-Model Experiment: smoke_lags_stride2000

## Headline

- Best candidate: `alpha_ridge_rank`.
- Family: `alpha_rank_latent`.
- Global weighted zero-mean R2: `0.002642559`.
- Mean fold R2: `0.001505554`.
- Min fold R2: `-0.004517745`.

## Audit

- Folds: `2`.
- Model features: `34`.
- Uses processed lags: `True`.
- Uses context features: `False`.
- Target leakage check: `passed`.
- Fold causality check: `passed`.
- Selection check: `passed: risk strengths and auxiliary bases are fixed by config, not selected inside validation`.
- Residual rules: `feature_04`.

## Methodological Status

- This is an experimental OOF validation stack, not yet a Kaggle runtime artifact.
- Risk and regime models are auxiliary: they modulate the alpha prediction; they do not change the required final output schema.
- Promotion requires comparison against the preserved conservative and historical checkpoints under the same fold protocol.
