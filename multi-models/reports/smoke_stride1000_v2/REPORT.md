# Multi-Model Experiment: smoke_stride1000_v2

## Headline

- Best candidate: `alpha_ridge_rank`.
- Family: `alpha_rank_latent`.
- Global weighted zero-mean R2: `0.003766118`.
- Mean fold R2: `0.003738419`.
- Min fold R2: `0.002388617`.

## Audit

- Folds: `2`.
- Model features: `30`.
- Uses processed lags: `False`.
- Uses context features: `False`.
- Target leakage check: `passed`.
- Residual rules: `feature_04`.

## Methodological Status

- This is an experimental OOF validation stack, not yet a Kaggle runtime artifact.
- Risk and regime models are auxiliary: they modulate the alpha prediction; they do not change the required final output schema.
- Promotion requires comparison against the preserved conservative and historical checkpoints under the same fold protocol.
