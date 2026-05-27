# Jane Street Conservative RLS Late Submission Package

This package is prepared for a manual Kaggle late submission to the Jane Street Real-Time Market Data Forecasting competition.

## Candidate

- Strategy: `dynamic_gateway_rls_experts_alpha10000_f0p995`
- Local Stage 3 score: `global_r2=0.013836465`, `min_fold_r2=0.007030887`
- Historical confirmation: `global_r2=0.015425344`
- Intended first submission: conservative RLS meta-layer over TabM, XGBoost, LightGBM, Ridge-calibrated predictions, and tree ensemble prediction.

## Files

- `submission/submission.py`: Kaggle inference server entrypoint.
- `src/janestreet/`: local package required by the entrypoint.
- `artifacts/jane_street_submission/base_models/`: final TabM and tree artifacts.
- `artifacts/jane_street_submission/meta_rls_experts_alpha10000_f0p995/`: conservative RLS meta-state.
- `vendor/`: small offline modules required by the TabM artifact.
- `kaggle_notebook_launcher.py`: one-cell Kaggle launcher.
- `jane_street_conservative_late_submission.ipynb`: importable Kaggle Notebook.

## Manual Kaggle Upload

1. Create a new private Kaggle Dataset.
2. Upload the contents of this package directory. If you use the `.zip` archive for transfer, make sure the Kaggle Dataset exposes the extracted files rather than only one zip file.
3. Create a new notebook for the competition, or import `jane_street_conservative_late_submission.ipynb`.
4. Attach the competition data and this package Dataset.
5. Set Accelerator to GPU if available.
6. Disable internet.
7. Paste the contents of `kaggle_notebook_launcher.py` into the first notebook cell.
8. Save a notebook version.
9. Submit that notebook version as the late submission.

## Audit Notes

The online update is causal by construction: at `date_id=D`, it updates from cached `D-1` features joined to gateway-provided `responder_*_lag_1`, then predicts the current batch. Local gateway smoke passed, but the official local mock has only one `date_id`; the Kaggle rerun remains the real packaging test.
