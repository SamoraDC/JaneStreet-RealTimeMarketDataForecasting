# Multi-Model Experiment: family_artifacts_smoke_stride2500_features60_v2

## Headline

- Best candidate: `ridge_rank_alpha10000__feature_59_z_residual`.
- Family: `residual_correction`.
- Global weighted zero-mean R2: `0.020785348`.
- Mean fold R2: `0.020709795`.
- Min fold R2: `0.020000397`.

## Audit

- Folds: `2`.
- Model features: `72`.
- Uses processed lags: `False`.
- Uses context features: `True`.
- Target leakage check: `passed`.
- Fold causality check: `passed`.
- Selection check: `passed: risk strengths and auxiliary bases are fixed by config, not selected inside validation`.
- Residual rules: `feature_47, feature_59, feature_04`.

## Methodological Status

- This is an experimental OOF validation stack, not yet a Kaggle runtime artifact.
- Risk and regime models are auxiliary: they modulate the alpha prediction; they do not change the required final output schema.
- Promotion requires comparison against the preserved conservative and historical checkpoints under the same fold protocol.
