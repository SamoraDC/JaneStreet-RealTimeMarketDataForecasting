# Multi-Model Family Artifacts

This manifest names the concrete artifacts implemented in `multi-models`.
They are fixed candidates or fixed auxiliary roles; promotion still requires
walk-forward validation against the active robust reference.

## Primary Artifacts


| #   | Artifact                                                          | Family                | Role                                                                |
| --- | ----------------------------------------------------------------- | --------------------- | ------------------------------------------------------------------- |
| 1   | `ridge_rank_alpha10000`                                           | Alpha Latente Global  | Ridge on train-fitted rank-normalized features.                     |
| 2   | `pls_rank_k8`                                                     | Alpha Latente Recente | PLS rank compression with `k=8`.                                    |
| 3   | `gateway_risk_conservative_rls_abs_pred_s100_prediction`          | Alpha Forte Existente | Current confirmed strong OOF gateway RLS candidate.                 |
| 4   | `ridge_rank_alpha10000__feature_47_z_residual`                    | Correção Residual     | Train-only univariate residual correction for the global alpha.     |
| 5   | `pls_rank_k8__feature_59_z_residual`                              | Correção Residual     | Train-only univariate residual correction for the recent alpha.     |
| 6   | `ridge_rank_alpha10000__risk_abs_responder6_ridge_rank_score`     | Risco/Volatilidade    | Auxiliary model for `abs(responder_6)`.                             |
| 7   | `ridge_rank_alpha10000__risk_sq_responder6_ridge_rank_score`      | Risco/Volatilidade    | Auxiliary model for `responder_6^2`.                                |
| 8   | `ridge_rank_alpha10000__risk_abs_error_ridge_rank_score`          | Incerteza             | Auxiliary model for absolute model error.                           |
| 9   | `ridge_rank_alpha10000__risk_abs_error_s0p05_micro_regime_scaled` | Regime/Microestrutura | Risk-shrunk alpha with observable microstructure scale calibration. |


## Secondary Implemented Risk Targets

The pipeline also implements `sq_error` and `high_error` risk targets for every
configured auxiliary base. These are scored through downstream shrinkage
candidates rather than promoted as direct alpha predictors.

## Causality Rules

- Rank/z encoders are fit on the training slice only.
- Residual rules use training residuals only.
- Risk models use training targets/errors only and are applied to validation
rows through target-free features.
- Microstructure gates use observable fields: `date_id`, `time_id`,
`symbol_id`, `weight`, missingness, lag energy, prediction magnitude and
predicted risk.
- Slice diagnostics are post-hoc and cannot create a promoted rule without a
fresh validation pass.

