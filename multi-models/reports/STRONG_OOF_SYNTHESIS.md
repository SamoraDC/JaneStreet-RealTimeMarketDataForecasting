# Strong OOF Modular Synthesis

## Protocol

The `multi-models` lab was extended with a strong OOF adapter that evaluates
saved TabM/tree/gateway predictions rather than retraining weak alphas from raw
features. The adapter joins:

- TabM OOF predictions;
- tree-engine OOF predictions;
- dynamic gateway RLS predictions reconstructed with the existing audited
  gateway simulator;
- optional gateway risk shrinkage candidates.

All modular layers are reported as fixed candidates. The code does not promote a
validation-best candidate inside the run. Gateway RLS updates are simulated with
strict previous-date updates.

## Stage 3 Results

Main Stage 3 focused run:
`multi-models/reports/strong_oof_stage3_gateway_abs_pred_shrink/`.

| Candidate | Global R2 | Mean Fold R2 | Min Fold R2 | Status |
| --- | ---: | ---: | ---: | --- |
| `gateway_risk_aggressive_rls_abs_pred_s25_prediction` | `0.013881135` | `0.013493024` | `0.006963629` | Best Stage 3 experimental |
| `gateway_risk_conservative_rls_abs_pred_s100_prediction` | `0.013875851` | `0.013489946` | `0.007012206` | Best robust Stage 3 candidate |
| `aggressive_rls_prediction` | `0.013866049` | `0.013480113` | `0.006970559` | Preserved aggressive baseline |
| `conservative_rls_prediction` | `0.013836471` | `0.013455631` | `0.007030935` | Preserved conservative baseline |

The broader Stage 3 run with stack/residual/risk/regime was
`multi-models/reports/strong_oof_stage3_full_v1/`. Its best new stack was
`strong_oof_ridge_stack`, `global_r2=0.013871582`, but this did not survive the
historical confirmation.

## Historical Confirmation

Historical max1398 focused run:
`multi-models/reports/strong_oof_hist_max1398_gateway_abs_pred_shrink/`.

| Candidate | Global R2 | Mean Fold R2 | Min Fold R2 | Status |
| --- | ---: | ---: | ---: | --- |
| `gateway_risk_conservative_rls_abs_pred_s100_prediction` | `0.015535311` | `0.015319532` | `0.012927336` | Best confirmed robust candidate |
| `gateway_risk_aggressive_rls_abs_pred_s100_prediction` | `0.015474168` | `0.015260896` | `0.012811188` | Good historical, weaker Stage 3 |
| `gateway_risk_conservative_rls_abs_pred_s25_prediction` | `0.015463534` | `0.015264651` | `0.012929243` | Stable but below s100 |
| `conservative_rls_prediction` | `0.015425373` | `0.015235126` | `0.012924771` | Previous robust reference |

The stack-only historical confirmation
`multi-models/reports/strong_oof_hist_max1398_stack_confirm/` showed that
`strong_oof_ridge_stack` drops to `global_r2=0.015180605`, below the preserved
conservative baseline. Therefore the stack is not promoted.

## Diagnostic Localization

Focused diagnostics were run for the promoted candidate against
`conservative_rls_prediction`:

- Stage 3:
  `multi-models/reports/strong_oof_stage3_conservative_s100_diagnostics/`.
- Historical max1398:
  `multi-models/reports/strong_oof_hist_max1398_conservative_s100_diagnostics/`.

Both runs use a frozen candidate and a frozen baseline before slicing; buckets
are diagnostic only and are not used to fit or select a new rule.

| Window | Candidate R2 | Baseline R2 | Delta R2 | Gateway Bad Updates |
| --- | ---: | ---: | ---: | ---: |
| Stage 3 | `0.013875822` | `0.013836443` | `0.000039380` | `0` |
| Historical max1398 | `0.015535281` | `0.015425343` | `0.000109939` | `0` |

The improvement is highly localized:

- by `abs(baseline_prediction)`, the top 1% bucket explains about all observed
  weighted SSE improvement in both windows;
- by candidate-baseline disagreement, the top 1% bucket also explains about all
  observed weighted SSE improvement;
- Stage 3 fold `rw_05` is mildly negative (`delta_r2=-0.000018729`), while the
  other folds are positive;
- historical fold `rw_05` carries most of the improvement, while `rw_01` and
  `rw_03` are mildly negative;
- high sample-weight tail behavior is mixed: Stage 3 is near flat positive in
  `q99_q100`, while historical `q99_q100` is slightly negative.

This supports a narrow interpretation: the promoted candidate is a causal tail
shrink/risk-control correction on top of conservative RLS, not a broad new alpha
source.

## Negative Results

- Residual correction over prediction-derived features worsened performance.
- Regime scale calibration did not beat the best RLS shrink candidate.
- The broad Stage 3 stack result looked positive locally, but failed historical
  confirmation.
- Prediction-risk shrinkage built from generic disagreement/weight proxies did
  not beat the existing gateway posterior/risk shrink formulation.

## Decision

Promote `gateway_risk_conservative_rls_abs_pred_s100_prediction` as the current
robust experimental candidate:

- Stage 3: `0.013875851`, above conservative `0.013836471`;
- Historical max1398: `0.015535311`, above conservative `0.015425373`;
- all gateway update audits report strict past updates;
- the candidate was evaluated by the same frozen rule in both windows.

Keep `gateway_risk_aggressive_rls_abs_pred_s25_prediction` as a Stage 3 score
candidate only. It is not the robust operational reference because it is weaker
on the historical confirmation.

## Technical Audit

- Target leakage: no target/responder columns are used as prediction features.
- Gateway causality: update rows are delivered from previous dates before
  current-date prediction.
- Diagnostic causality: all diagnostic slices are computed after fixed candidate
  selection and are not fed back into model fitting.
- Selection risk: Stage 3 and historical comparisons are still over known OOF
  families, not a fresh leaderboard-equivalent holdout.
- Robustness risk: the promoted gain is concentrated in extreme prediction and
  disagreement buckets. This is mechanically plausible for shrinkage, but it is
  more fragile than a gain spread uniformly across folds and regimes.
- RAM note: the broad Stage 3 run materialized too many candidate columns and
  peaked near 18GB RSS. Future full runs should use focused candidate lists or a
  streaming scorer before expanding grids.

## Ethical Audit

This does not prove live leaderboard superiority. It does show that the strongest
current path is not new raw-feature alphas, residual rules, or generic regimes.
The evidence favors a small, causal risk-shrink layer on top of the already
audited dynamic gateway RLS family.
