# Artifacts

This directory contains the GitHub-facing package for the preserved best local
Stage 3 reference:

```text
strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1

candidate:
fixed_blend_0_w0p75_fixed_blend
```

## Layout

```text
batch_mean_std_fixed_blend/
  README.md
  ARTIFACTS.md
  CODE.md
  code/
  artifacts/
    strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1/
      REPORT.md
      audit.json
      candidate_summary.csv
      fold_scores.csv
      gateway_daily_audit.csv
      parameters.json
    strong_oof_hist_max1398_s23aux8_s17_gateway_batch_mean_std_exact_v1/
      REPORT.md
      audit.json
      candidate_summary.csv
      fold_scores.csv
      gateway_daily_audit.csv
      parameters.json
```

## What Is Included

- A code snapshot for the strong OOF batch mean/std and fixed-blend pipeline.
- The best local Stage 3 report and all generated report artifacts.
- The exact historical confirmation report for the same frozen rule.
- Candidate summaries, fold scores, gateway audits, and parameters.

## Runtime Status

This is a validated OOF/reference artifact, not yet a Kaggle runtime package.
To make it submissible, the Ridge stack coefficients, batch prediction features,
RLS features, and final fixed blend must be exported and reproduced inside the
gateway runtime.
