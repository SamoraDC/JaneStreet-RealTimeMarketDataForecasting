# CLI Submission Folder

This folder is prepared for `kaggle kernels push`.

Expected order:

```bash
kaggle datasets create -p kaggle_upload/jane-street-conservative-rls-late-submission
kaggle kernels push -p kaggle_kernel/jane-street-conservative-rls-late-submission --accelerator NvidiaTeslaT4
kaggle kernels status samoradc/jane-street-conservative-rls-late-submission-nb
kaggle kernels files samoradc/jane-street-conservative-rls-late-submission-nb
kaggle competitions submit jane-street-real-time-market-data-forecasting \
  -k samoradc/jane-street-conservative-rls-late-submission-nb \
  -f submission.parquet \
  -m "Conservative causal RLS meta-ensemble over TabM, XGBoost, LightGBM, calibrated Ridge, and tree ensemble predictions."
```

If the Dataset already exists, use:

```bash
kaggle datasets version -p kaggle_upload/jane-street-conservative-rls-late-submission \
  -m "Update conservative RLS late submission package"
```

Do not submit before the notebook run completes and exposes `submission.parquet`.

Current status:

- Kernel version 4 completed on Kaggle and exposed `submission.parquet`.
- `kaggle competitions submit ... -v 4 ...` reached `CreateCodeSubmission` but returned:

```text
400 FAILED_PRECONDITION: Submission not allowed: Submissions have been disabled for this competition.
```

This is a platform/competition-policy block, not a notebook execution failure.
