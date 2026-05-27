# Multi-Model Experiment: smoke_context_stride1500

## Headline

- Best candidate: `alpha_rank_linear_stack`.
- Family: `alpha_stack`.
- Global weighted zero-mean R2: `-0.000548828`.
- Mean fold R2: `-0.000577217`.
- Min fold R2: `-0.001553100`.

## Audit

- Folds: `2`.
- Model features: `28`.
- Uses processed lags: `False`.
- Uses context features: `True`.
- Target leakage check: `passed`.
- Residual rules: `feature_04`.

## Methodological Status

- This is an experimental OOF validation stack, not yet a Kaggle runtime artifact.
- Risk and regime models are auxiliary: they modulate the alpha prediction; they do not change the required final output schema.
- Promotion requires comparison against the preserved conservative and historical checkpoints under the same fold protocol.
