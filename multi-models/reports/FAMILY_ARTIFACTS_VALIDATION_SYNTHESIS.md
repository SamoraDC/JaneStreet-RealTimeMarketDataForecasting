# Multi-Model Family Validation Synthesis

## Protocol

Two real-data validations were run for the implemented six-family, nine-artifact
pipeline:

- `family_artifacts_5fold_lags_stride250`
- `family_artifacts_5fold_lags_stride100_v2`

Both use 5 temporal folds, `train_window=120`, `valid_window=60`, all 79
features, official processed responder lags, context features, and fixed grids.
No candidate is selected inside the validation loop.

## Audit Result

The implementation audit passed after one fix. The first `stride100` run wrote
prediction artifacts without the `fold` column, which blocked post-hoc fold
diagnostics. This was an auditability bug, not a score-computation bug. The
pipeline now writes `fold` into `validation_predictions.parquet`, and the
`stride100_v2` run regenerated complete artifacts.

Latest checks:

- `target_leakage_check`: passed.
- `fold_causality_check`: passed.
- `selection_check`: passed.
- full test suite after the fix: `169 passed`.

## Main Results

| Run | Rows | Best Candidate | Family | Global R2 | Min Fold R2 |
| --- | ---: | --- | --- | ---: | ---: |
| `stride250` | `44117` | `latent_alpha_linear_stack` | `alpha_latent_stack` | `0.002006218` | `-0.000026850` |
| `stride100_v2` | `110286` | `latent_alpha_linear_stack` | `alpha_latent_stack` | `0.003324014` | `0.000957903` |

Best family scores in `stride100_v2`:

- `alpha_latent_stack`: `0.003324014`
- `alpha_latent_global`: `0.003044`
- `residual_correction`: `0.003020`
- `risk_shrinkage`: `0.002889`
- `regime_microstructure`: `0.002479`
- `alpha_latent_recent`: `0.001498`

These results are below the preserved robust checkpoint
`dynamic_gateway_rls_experts_alpha10000_f0p995` at `0.013836465`, and below the
current robust strong OOF candidate
`gateway_risk_conservative_rls_abs_pred_s100_prediction` at about `0.013875851`
on Stage 3.

## Auxiliary Risk Models

Risk/uncertainty targets are learnable, but their direct shrinkage over weak
alphas is not enough to make the raw-family pipeline competitive.

Mean auxiliary R2 in `stride100_v2`:

- `abs_error`: `0.488699`
- `abs_responder6`: `0.487294`
- `high_error`: `0.152195`
- `sq_error`: `0.147559`
- `sq_responder6`: `0.147060`

Interpretation: volatility/error magnitude is much easier to predict than alpha
direction/magnitude. This supports using risk models as gates or shrinkage
inputs for the strong RLS/OOF candidate, not as standalone alpha predictors.

## Slice Diagnostics

For `latent_alpha_linear_stack` in `stride100_v2`:

- fold R2 is positive across all five folds, but weak: best `rw_02=0.008194`,
  worst `rw_05=0.000958`;
- by sample weight, signal is stronger in high-weight rows:
  `q90_q99=0.007090`, `q99_q100=0.004639`;
- by prediction magnitude, signal is concentrated in the tails:
  `q99_q100=0.021173`, `q90_q99=0.010692`, while `q00_q50=0.000118`;
- several symbols are negative, including `symbol_id=1` at `-0.002110`.

This is not a broad robust alpha. It is a weak rank-latent alpha whose usable
information is concentrated in high-confidence/high-magnitude regions.

## Decision

Do not promote the raw multi-model family pipeline as a direct predictor.

Promote only these lessons:

1. Keep `latent_alpha_linear_stack` as a weak auxiliary alpha feature candidate.
2. Keep the risk auxiliary models as potentially valuable uncertainty features.
3. Test the next bridge by applying learned risk/regime signals to the strong
   gateway RLS/OOF candidate, not to weak Ridge/PLS alphas.
4. Do not spend full-data compute on direct PLS/Ridge/residual replacement until
   a nested or stronger OOF integration shows incremental value.

## Socratic Audit

- Implementation failure: one auditability issue was found and fixed
  (`fold` missing from saved predictions).
- Mathematical failure: no evidence that formulas were implemented incorrectly;
  tests cover encoders, residual rules, shrinkage, regime bins and artifact
  manifest.
- Data-regime failure: direct latent alphas are too weak for this competition
  regime.
- Validation failure risk: this is still sampled validation, not full-data
  leaderboard-equivalent validation. The conclusion is conservative: do not
  promote direct raw-family predictors.
