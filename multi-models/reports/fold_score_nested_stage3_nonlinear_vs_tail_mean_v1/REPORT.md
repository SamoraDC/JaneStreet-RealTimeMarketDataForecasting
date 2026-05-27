# Fold-Score Nested Selection

## Scope

This diagnostic selects among completed OOF candidates using only fold-level sufficient statistics from earlier folds.
It is cheap and anti-leakage by construction, but it is not a deployable row-level policy unless reimplemented with an observable time rule.

## Configuration

- Selection metric: `mean_fold_r2`
- Min history folds: `1`
- First fold candidate: `prior/fixed_blend_1_w0p5_fixed_blend`

## Nested Summary

```text
[
  {
    "candidate": "nested_previous_fold_selector",
    "family": "nested_selector",
    "rows": 11028424,
    "weight_sum": 28034941.5,
    "numerator": 17148999.12310873,
    "denominator": 17394888.5,
    "mean_fold_r2": 0.013810960733996768,
    "min_fold_r2": 0.0077634663745760335,
    "std_fold_r2": 0.00781426160682317,
    "global_r2": 0.014135725957155176
  }
]
```

## Selected Folds

```text
[
  {
    "fold": "rw_01",
    "selected_candidate_id": "prior/fixed_blend_1_w0p5_fixed_blend",
    "weighted_zero_mean_r2": 0.01397140715871259,
    "selection_score": null,
    "history_folds": 0
  },
  {
    "fold": "rw_02",
    "selected_candidate_id": "nonlinear/fixed_blend_1_w0p75_fixed_blend",
    "weighted_zero_mean_r2": 0.027142255244450908,
    "selection_score": 0.014164533689698167,
    "history_folds": 1
  },
  {
    "fold": "rw_03",
    "selected_candidate_id": "nonlinear/fixed_blend_1_w0p75_fixed_blend",
    "weighted_zero_mean_r2": 0.00903503720596488,
    "selection_score": 0.020653394467074537,
    "history_folds": 2
  },
  {
    "fold": "rw_04",
    "selected_candidate_id": "nonlinear/fixed_blend_1_w0p75_fixed_blend",
    "weighted_zero_mean_r2": 0.011142637686279433,
    "selection_score": 0.01678060871337132,
    "history_folds": 3
  },
  {
    "fold": "rw_05",
    "selected_candidate_id": "nonlinear/fixed_blend_1_w0p75_fixed_blend",
    "weighted_zero_mean_r2": 0.0077634663745760335,
    "selection_score": 0.015371115956598347,
    "history_folds": 4
  }
]
```

## Candidate Pool Top 10

```text
[
  {
    "candidate_id": "nonlinear/fixed_blend_1_w0p75_fixed_blend",
    "global_r2": 0.014182394685754218,
    "min_fold_r2": 0.0077634663745760335
  },
  {
    "candidate_id": "prior/fixed_blend_1_w0p5_fixed_blend",
    "global_r2": 0.014033829902132089,
    "min_fold_r2": 0.007406491677363181
  },
  {
    "candidate_id": "tail/gateway_risk_aggressive_rls_abs_pred_s25_prediction_residual_weight_and_abs_q0p99_residual_tail",
    "global_r2": 0.013880932670747637,
    "min_fold_r2": 0.0069636285976644174
  },
  {
    "candidate_id": "tail/gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p99_residual_tail",
    "global_r2": 0.013875150272695258,
    "min_fold_r2": 0.00701220579037487
  }
]
```

## Audit

- Uses only prior folds for selection after the predeclared first-fold candidate.
- Does not inspect row-level validation targets beyond the already-written fold sufficient statistics.
- Diagnostic only until converted into a deployable causal time policy.
