# Multi-Model Experiment: primary_preprocess_raw9_stride100_v1

## Headline

- Best candidate: `latent_alpha_linear_stack`.
- Family: `alpha_latent_stack`.
- Global weighted zero-mean R2: `0.003501597`.
- Mean fold R2: `0.003460232`.
- Min fold R2: `0.000952744`.

## Audit

- Folds: `5`.
- Model features: `148`.
- Uses processed lags: `False`.
- Uses context features: `True`.
- Raw preprocessed features: `57`.
- Raw preprocessing modes: `batch_rank, batch_demean, batch_zscore, batch_abs_zscore, batch_top_bottom, row_missing_count, row_abs_mean, row_l2_energy`.
- Target leakage check: `passed`.
- Fold causality check: `passed`.
- Selection check: `passed: risk strengths and auxiliary bases are fixed by config, not selected inside validation`.
- Residual rules: `feature_47, feature_59, feature_04`.
- Risk auxiliary targets: `abs_responder6, abs_error`.

## Artifact Manifest

- `ridge_rank_alpha10000`: alpha_latent_global - Ridge on train-fitted rank encodings.
- `pls_rank_k8`: alpha_latent_recent - PLS rank compression.
- `gateway_risk_conservative_rls_abs_pred_s100_prediction`: alpha_strong_existing - best confirmed strong OOF candidate from gateway RLS adapter.
- `ridge_rank_alpha10000__feature_47_z_residual`: residual_correction - global residual bridge.
- `pls_rank_k8__feature_59_z_residual`: residual_correction - recent residual bridge.
- `ridge_rank_alpha10000__risk_abs_responder6_ridge_rank_score`: risk_volatility_uncertainty - absolute responder risk target.
- `ridge_rank_alpha10000__risk_sq_responder6_ridge_rank_score`: risk_volatility_uncertainty - squared responder risk target.
- `ridge_rank_alpha10000__risk_abs_error_ridge_rank_score`: risk_volatility_uncertainty - absolute model error target.
- `ridge_rank_alpha10000__risk_abs_error_s0p05_micro_regime_scaled`: regime_microstructure - observable microstructure scale gate.

## Methodological Status

- This is an experimental OOF validation stack, not yet a Kaggle runtime artifact.
- Risk and regime models are auxiliary: they modulate the alpha prediction; they do not change the required final output schema.
- Promotion requires comparison against the preserved conservative and historical checkpoints under the same fold protocol.
