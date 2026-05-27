# Multi-Model Experiment: strong_oof_tabm_w010_complete_stage3_v1

## Headline

- Best candidate: `fixed_blend_2_w0p5_fixed_blend`.
- Family: `fixed_blend`.
- Global weighted zero-mean R2: `0.013628744`.
- Mean fold R2: `0.013372873`.
- Min fold R2: `0.007148818`.

## Audit

- Folds: `5`.
- Model features: `12`.
- Uses processed lags: `False`.
- Uses context features: `True`.
- Target leakage check: `passed`.
- Fold causality check: `passed: walk-forward layers and residual tail thresholds fit only earlier folds; gateway updates use prior-date lag simulation`.
- Selection check: `passed: all grids are reported as fixed candidates, no validation-best candidate is promoted inside code`.
- Residual rules: `prediction_disagreement, tabm_tree_diff, abs_baseline_prediction, weight`.
- Risk auxiliary targets: `none`.

## Artifact Manifest


## Methodological Status

- This is an experimental OOF validation stack, not yet a Kaggle runtime artifact.
- Risk and regime models are auxiliary: they modulate the alpha prediction; they do not change the required final output schema.
- Promotion requires comparison against the preserved conservative and historical checkpoints under the same fold protocol.
