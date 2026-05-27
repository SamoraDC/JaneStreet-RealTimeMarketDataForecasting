# Artifacts

This directory contains the GitHub-facing package for the preserved conservative
operational reference:

```text
dynamic_gateway_rls_experts_alpha10000_f0p995
```

## Layout

```text
conservative_dynamic_gateway_rls/
  README.md
  ARTIFACTS.md
  CODE.md
  code/
  artifacts/
    meta_rls_experts_alpha10000_f0p995/
      rls_state.npz
      meta_rls_coefficients.csv
      meta_rls_artifact_report.json
  validation/
    dynamic_gateway_rls_stage3/
      dynamic_gateway_rls_summary.csv
      dynamic_gateway_rls_by_fold.csv
      dynamic_gateway_rls_daily_audit.csv
      dynamic_gateway_rls_parameters.csv
      dynamic_gateway_rls_report.json
    dynamic_gateway_rls_hist_max1398/
      dynamic_gateway_rls_summary.csv
      dynamic_gateway_rls_by_fold.csv
      dynamic_gateway_rls_daily_audit.csv
      dynamic_gateway_rls_parameters.csv
      dynamic_gateway_rls_report.json
  kaggle_upload/
    jane-street-conservative-rls-late-submission/
  kaggle_kernel/
    jane-street-conservative-rls-late-submission/
```

## What Is Included

- A code snapshot for validation, artifact export, and Kaggle runtime packaging.
- The exported RLS meta-state used by the conservative candidate.
- Stage 3 and historical validation reports.
- The Kaggle dataset package and notebook/kernel folder that were prepared for
  the late-submission workflow.

## What Is Not Included

- Raw Kaggle data.
- Any claim that the official competition submission succeeded. The local and
  Kaggle notebook runtime produced `submission.parquet`, but the official final
  submission was blocked because submissions were disabled for the competition.
