# Fold-Score Nested Selection

## Scope

This diagnostic selects among completed OOF candidates using only fold-level sufficient statistics from earlier folds.
It is cheap and anti-leakage by construction, but it is not a deployable row-level policy unless reimplemented with an observable time rule.

## Configuration

- Selection metric: `global_r2`
- Min history folds: `2`
- First fold candidate: `prior/fixed_blend_1_w0p5_fixed_blend`

## Nested Summary

```text
[
  {
    "candidate": "nested_previous_fold_selector",
    "family": "nested_selector",
    "rows": 11028424,
    "weight_sum": 28034941.5,
    "numerator": 17149554.110690854,
    "denominator": 17394888.5,
    "mean_fold_r2": 0.013778492474938498,
    "min_fold_r2": 0.0077634663745760335,
    "std_fold_r2": 0.007745052975269643,
    "global_r2": 0.014103820746487972
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
    "selected_candidate_id": "prior/fixed_blend_1_w0p5_fixed_blend",
    "weighted_zero_mean_r2": 0.02697991394915955,
    "selection_score": null,
    "history_folds": 0
  },
  {
    "fold": "rw_03",
    "selected_candidate_id": "nonlinear/fixed_blend_1_w0p75_fixed_blend",
    "weighted_zero_mean_r2": 0.00903503720596488,
    "selection_score": 0.019985275737650965,
    "history_folds": 2
  },
  {
    "fold": "rw_04",
    "selected_candidate_id": "nonlinear/fixed_blend_1_w0p75_fixed_blend",
    "weighted_zero_mean_r2": 0.011142637686279433,
    "selection_score": 0.016964460569811934,
    "history_folds": 3
  },
  {
    "fold": "rw_05",
    "selected_candidate_id": "nonlinear/fixed_blend_1_w0p75_fixed_blend",
    "weighted_zero_mean_r2": 0.0077634663745760335,
    "selection_score": 0.015253121982787943,
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
