# Multi-Model Experiment: family_artifacts_smoke_stride2000

## Headline

- Best candidate: `latent_alpha_linear_stack`.
- Family: `alpha_latent_stack`.
- Global weighted zero-mean R2: `-0.000746156`.
- Mean fold R2: `-0.000821920`.
- Min fold R2: `-0.003751418`.

## Audit

- Folds: `2`.
- Model features: `38`.
- Uses processed lags: `False`.
- Uses context features: `True`.
- Target leakage check: `passed`.
- Fold causality check: `passed`.
- Selection check: `passed: risk strengths and auxiliary bases are fixed by config, not selected inside validation`.
- Residual rules: `feature_04`.

## Methodological Status

- This is an experimental OOF validation stack, not yet a Kaggle runtime artifact.
- Risk and regime models are auxiliary: they modulate the alpha prediction; they do not change the required final output schema.
- Promotion requires comparison against the preserved conservative and historical checkpoints under the same fold protocol.
