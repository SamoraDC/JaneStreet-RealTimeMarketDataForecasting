# Strong OOF Tail Blend Synthesis

## Objective

This stage moved the tail-only residual idea from sampled family artifacts to
full strong OOF frames. The goal was to test whether observable residual tails
can improve the preserved gateway candidates without training new heavy models.

All tail thresholds and walk-forward blend weights are causal: `rw_01` is
identity/default, and later folds use only earlier folds. Fixed blends are not
learned from targets, but their reported selection is still exploratory because
the best blend is chosen after validation.

## Implemented Layers

- `residual_tail`: applies a residual correction only when the current row is in
  a previous-fold tail of `weight`, `abs_base`, `prediction_disagreement`, or
  simple combinations.
- `fixed_blend`: averages frozen candidate columns with fixed weights.
- `walk_forward_blend`: estimates a two-candidate convex weight from earlier
  folds only.
- `residual_base_candidates`: restricts residual/tail generation to selected
  bases, reducing memory and avoiding unnecessary candidate churn.

## Full Results

Stage 3 full best:

| Candidate | Global R2 | Reference | Delta |
| --- | ---: | ---: | ---: |
| `fixed_blend_2_w0p5_fixed_blend` | `0.013895762` | `0.013881135` | `+0.000014627` |
| `wf_blend_4_wf_blend` | `0.013887358` | `0.013881135` | `+0.000006223` |
| `gateway_risk_aggressive_rls_abs_pred_s25_prediction` | `0.013881135` | baseline | `0.000000000` |

`fixed_blend_2_w0p5` is a 50/50 blend of:

- `gateway_risk_aggressive_rls_abs_pred_s25_prediction`;
- `gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_abs_base_q0p99_residual_tail_prediction`.

Historical `max1398` best:

| Candidate | Global R2 | Reference | Delta |
| --- | ---: | ---: | ---: |
| `gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail` | `0.015630171` | `0.015535311` | `+0.000094860` |
| `wf_blend_5_wf_blend` | `0.015623090` | `0.015535311` | `+0.000087779` |
| `gateway_risk_conservative_rls_abs_pred_s100_prediction` | `0.015535311` | baseline | `0.000000000` |

The Stage 3 winner does not maximize historical score. The historical winner
does not maximize Stage 3 score. This is a regime-dependence warning, not a
failure of causality.

## Audit

- Full test suite after implementation: `180 passed`.
- `gateway_bad_updates=0` in generated audits.
- Tail thresholds use previous folds only.
- Walk-forward blend weights use previous folds only.
- No target/responder columns are configured as residual features.
- Fixed-blend selection is exploratory; do not promote it without independent
  validation or a pre-declared criterion.

## Decision

The best empirical Stage 3 candidate is now
`fixed_blend_2_w0p5_fixed_blend` with `global_r2=0.013895762`, but it is not yet
the new submission candidate because its weight is selected post-hoc.

The best historical robustness candidate is
`gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail`
with `global_r2=0.015630171`, but it underperforms the Stage 3 fixed blend.

Next step: if this line continues, freeze one pre-declared selection rule
before running any further validation. The most defensible candidates are the
walk-forward blend `wf_blend_4_wf_blend` for Stage 3 robustness and
`wf_blend_5_wf_blend` for historical robustness.
