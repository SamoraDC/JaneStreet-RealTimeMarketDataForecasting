# Strong OOF Diagnostics: strong_oof_stage3_conservative_s100_diagnostics

## Overall

- Candidate: `gateway_risk_conservative_rls_abs_pred_s100_prediction`.
- Baseline: `conservative_rls_prediction`.
- Candidate R2: `0.013875822`.
- Baseline R2: `0.013836443`.
- Delta R2: `0.000039380`.
- Rows: `11028424`.

## Audit

- Gateway bad updates: `0`.
- Selection status: `diagnostic only; candidate and baseline are fixed before slicing`.
- Slice files: `delta_vs_baseline_*.csv`.

## Interpretation

- Positive `candidate_delta_r2` means the candidate reduces weighted squared error versus the baseline in that slice.
- Buckets are diagnostic only; they are not used to fit the candidate.
- This report explains where the frozen candidate wins or loses; it is not a new model search by itself.
