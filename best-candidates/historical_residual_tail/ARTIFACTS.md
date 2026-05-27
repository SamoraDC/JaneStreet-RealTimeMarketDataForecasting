# Artifacts

This directory contains the GitHub-facing package for the preserved full
historical reference:

```text
strong_oof_hist_max1398_gateway_residual_tail_modes_v1

candidate:
gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail
```

## Layout

```text
historical_residual_tail/
  README.md
  ARTIFACTS.md
  CODE.md
  code/
  artifacts/
    strong_oof_hist_max1398_gateway_residual_tail_modes_v1/
      REPORT.md
      audit.json
      candidate_summary.csv
      fold_scores.csv
      gateway_daily_audit.csv
      parameters.json
```

## What Is Included

- A code snapshot for the strong OOF residual-tail pipeline.
- The experiment report.
- Candidate summary and fold-level scores.
- Gateway daily audit.
- Parameters for the residual and residual-tail rules.
- The full audit payload from the original report directory.

## Runtime Status

This is a validated OOF/reference artifact, not yet a Kaggle runtime package.
To make it submissible, the residual correction coefficients and tail thresholds
must be exported into the runtime and implemented in `submission.py`.
