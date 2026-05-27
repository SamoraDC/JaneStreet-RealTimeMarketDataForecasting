# Strong Family Bridge: strong_family_bridge_stride50_tail_v1

## Headline

- Best candidate: `gateway_risk_conservative_rls_abs_pred_s100_prediction_family_residual_weight_q0p95_family_residual_tail`.
- Family: `strong_family_residual_tail`.
- Global R2: `0.013361703`.
- Min fold R2: `0.007839666`.
- Strong base R2: `0.013235539`.
- Delta versus strong base: `0.000126163`.

## Audit

- Rows: `220570`.
- Folds: `5`.
- Strong base: `gateway_risk_conservative_rls_abs_pred_s100_prediction`.
- Gateway bad updates: `0`.
- Target leakage check: `passed`.
- Causality check: `passed: stack, residual, risk normalization and regime scales fit only earlier folds; gateway updates use prior-date simulator`.
- Selection check: `passed: strengths/features are fixed by config and all candidates are reported`.

## Methodological Status

- This bridge is evaluated on the sampled family-artifact intersection, not the full 11M-row OOF frame.
- Walk-forward bridge layers use previous folds only; first fold is identity for learned bridge layers.
- Promotion requires confirmation on a denser or full OOF bridge and slice diagnostics.
