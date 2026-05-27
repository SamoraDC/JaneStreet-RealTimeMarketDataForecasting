# Fold-Score Nested Selection

## Scope

This diagnostic selects among completed OOF candidates using only fold-level sufficient statistics from earlier folds.
It is cheap and anti-leakage by construction, but it is not a deployable row-level policy unless reimplemented with an observable time rule.

## Configuration

- Selection metric: `mean_fold_r2`
- Min history folds: `1`
- First fold candidate: `tail/gateway_risk_conservative_rls_abs_pred_s100_prediction`

## Nested Summary

```text
[
  {
    "candidate": "nested_previous_fold_selector",
    "family": "nested_selector",
    "rows": 11151360,
    "weight_sum": 21105826.5,
    "numerator": 16170948.617828202,
    "denominator": 16427583.5,
    "mean_fold_r2": 0.015408277482383447,
    "min_fold_r2": 0.012890489078113632,
    "std_fold_r2": 0.00330204551138907,
    "global_r2": 0.015622193134601892
  }
]
```

## Selected Folds

```text
[
  {
    "fold": "rw_01",
    "selected_candidate_id": "tail/gateway_risk_conservative_rls_abs_pred_s100_prediction",
    "weighted_zero_mean_r2": 0.016288663434381956,
    "selection_score": null,
    "history_folds": 0
  },
  {
    "fold": "rw_02",
    "selected_candidate_id": "nonlinear/fixed_blend_5_w0p75_fixed_blend",
    "weighted_zero_mean_r2": 0.012890489078113632,
    "selection_score": 0.016328639609601403,
    "history_folds": 1
  },
  {
    "fold": "rw_03",
    "selected_candidate_id": "nonlinear/fixed_blend_5_w0p75_fixed_blend",
    "weighted_zero_mean_r2": 0.013577267223620293,
    "selection_score": 0.014609564343857517,
    "history_folds": 2
  },
  {
    "fold": "rw_04",
    "selected_candidate_id": "tail/gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail",
    "weighted_zero_mean_r2": 0.013459269315960753,
    "selection_score": 0.01426577743744839,
    "history_folds": 3
  },
  {
    "fold": "rw_05",
    "selected_candidate_id": "tail/gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail",
    "weighted_zero_mean_r2": 0.020825698359840605,
    "selection_score": 0.014064150407076481,
    "history_folds": 4
  }
]
```

## Candidate Pool Top 10

```text
[
  {
    "candidate_id": "tail/gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail",
    "global_r2": 0.015630171202296594,
    "min_fold_r2": 0.012909969694714585
  },
  {
    "candidate_id": "nonlinear/fixed_blend_5_w0p5_fixed_blend",
    "global_r2": 0.015618963849932666,
    "min_fold_r2": 0.012831522318604183
  },
  {
    "candidate_id": "tail/gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p99_residual_tail",
    "global_r2": 0.015590931202279279,
    "min_fold_r2": 0.012912331898926044
  },
  {
    "candidate_id": "nonlinear/fixed_blend_5_w0p75_fixed_blend",
    "global_r2": 0.01558326863689874,
    "min_fold_r2": 0.012890489078113632
  },
  {
    "candidate_id": "tail/gateway_risk_conservative_rls_abs_pred_s100_prediction",
    "global_r2": 0.015535311185594258,
    "min_fold_r2": 0.012927335822554564
  }
]
```

## Audit

- Uses only prior folds for selection after the predeclared first-fold candidate.
- Does not inspect row-level validation targets beyond the already-written fold sufficient statistics.
- Diagnostic only until converted into a deployable causal time policy.
