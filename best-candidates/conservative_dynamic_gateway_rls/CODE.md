# Code Snapshot

This directory includes the code directly related to generating, validating, and
packaging the preserved conservative candidate:

```text
dynamic_gateway_rls_experts_alpha10000_f0p995
```

## Layout

```text
code/
  submission.py
  scripts/
    run_bayesian_gateway_meta_simulation.py
    run_dynamic_gateway_rls_validation.py
    run_gateway_rls_strategy_selection.py
    export_dynamic_rls_meta_artifact.py
    train_submission_artifacts.py
    build_kaggle_late_submission_package.py
    run_competitive_tabular_nn.py
    run_tree_engine_ensemble.py
  src/janestreet/
    bayesian_meta.py
    blending.py
    calibration.py
    folds.py
    linear.py
    metrics.py
    official_lags.py
    paths.py
    submission_artifacts.py
    submission_inference.py
    submission_models.py
  tests/
    test_bayesian_gateway_meta_simulation_script.py
    test_dynamic_gateway_rls_validation_script.py
    test_gateway_rls_strategy_selection_script.py
    test_submission_artifacts.py
    test_submission_entrypoint.py
    test_submission_inference.py
    test_submission_models.py
```

## What Each Part Does

- `run_bayesian_gateway_meta_simulation.py`: builds the strict gateway-style
  baseline and expert feature sets.
- `run_dynamic_gateway_rls_validation.py`: validates dynamic RLS candidates with
  forgetting factors, including `experts_alpha10000_f0p995`.
- `run_gateway_rls_strategy_selection.py`: defines the preserved conservative and
  aggressive RLS strategies and tests causal selectors/shrinkage alternatives.
- `export_dynamic_rls_meta_artifact.py`: exports the RLS state used by the
  submission runtime.
- `train_submission_artifacts.py`: trains/exports base model artifacts for the
  Kaggle runtime package.
- `run_competitive_tabular_nn.py` and `run_tree_engine_ensemble.py`: training
  code loaded by `train_submission_artifacts.py` to produce the TabM and
  tree-family base artifacts.
- `build_kaggle_late_submission_package.py`: builds the Kaggle dataset/notebook
  package.
- `submission.py` plus `src/janestreet/submission_*`: implement the
  `predict(test, lags)` gateway contract.
- `calibration.py`, `folds.py`, `linear.py`, `official_lags.py`, and `paths.py`:
  direct support modules used by the base artifact training and inference code.

## Reproduction Commands

Stage 3 validation:

```bash
uv run python scripts/run_dynamic_gateway_rls_validation.py \
  --output-dir reports/experiments/dynamic_gateway_rls_stage3 \
  --experiment-name dynamic_gateway_rls_stage3
```

Historical validation:

```bash
uv run python scripts/run_dynamic_gateway_rls_validation.py \
  --tabm-prediction-dir reports/experiments/competitive_tabm_official_stage3_hist_max1398_5fold_valid60_lags_online_lr1e4_4m_train700_seed37_aux8_preds/validation_predictions \
  --tree-prediction-dir reports/experiments/tree_engine_ensemble_hist_max1398_xgb_lgb_sample10_seed_ensemble_preds/validation_predictions \
  --output-dir reports/experiments/dynamic_gateway_rls_hist_max1398 \
  --experiment-name dynamic_gateway_rls_hist_max1398
```

Export the conservative RLS state:

```bash
uv run python scripts/export_dynamic_rls_meta_artifact.py \
  --feature-set experts \
  --ridge-alpha 10000 \
  --forgetting-factor 0.995 \
  --output-dir artifacts/jane_street_submission/meta_rls_experts_alpha10000_f0p995
```

Build the Kaggle package:

```bash
uv run python scripts/build_kaggle_late_submission_package.py
```

## Notes

This code snapshot is meant for GitHub review and reproducibility. It does not
include raw Kaggle data. Full reproduction requires the original local data and
OOF prediction artifacts referenced by the commands above.
