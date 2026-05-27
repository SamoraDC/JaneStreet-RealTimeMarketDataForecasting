# Strong Family Bridge Synthesis

## Objective

This stage tested whether the six model families add value when connected to the
current strong alpha instead of used as standalone predictors. The strong base is
`gateway_risk_conservative_rls_abs_pred_s100_prediction`; family artifacts were
tested on `family_artifacts_5fold_lags_stride100_v2` and denser
`family_artifacts_5fold_lags_stride50_v1` intersections.

The bridge is evaluated on the sampled family-artifact intersection
(`110286` rows), not on the full 11M-row OOF frame. Absolute R2 values therefore
must not be compared directly to full Stage 3 or historical max1398 scores.
Only deltas versus the reconstructed strong base inside the same frame are
valid.

## Implemented Connections

Implemented bridge layers:

- alpha stack: Ridge stack of strong base plus weak family alphas;
- residual bridge: Ridge on previous-fold residuals using weak alpha and risk
  features;
- tail-only residual bridge: apply the residual bridge only on observable
  high-tail masks learned from previous folds;
- risk shrinkage: shrink strong base by predicted error/volatility scores;
- regime scale: risk-shrunk strong base with observable context groups.

All learned bridge layers are walk-forward. Fold `rw_01` is identity, and folds
`rw_02` onward fit only on earlier folds. Gateway updates still use the
prior-date simulator.

## Results

Main broad bridge:
`strong_family_bridge_stride100_v1`.

Minimal bridge:
`strong_family_bridge_stride100_residual_minimal_v1`.

Dense confirmation:
`strong_family_bridge_stride50_residual_gate_v1`.

Tail-only confirmation:
`strong_family_bridge_stride100_tail_v1` and
`strong_family_bridge_stride50_tail_v1`.

| Candidate | Family | Global R2 | Delta vs Strong Base | Min Fold R2 | Status |
| --- | --- | ---: | ---: | ---: | --- |
| `...family_residual_weight_q0p95_family_residual_tail` on stride50 | tail residual | `0.013361703` | `+0.000126163` | `0.007839666` | Best sampled bridge |
| `...family_residual_weight_q0p95_family_residual_tail` on stride100 | tail residual | `0.011152894` | `+0.000111899` | `0.005231074` | Confirmed on independent sample density |
| `gateway_risk_conservative_rls_abs_pred_s100_prediction_family_residual` | residual bridge | `0.011144201` | `+0.000103206` | `0.005304399` | Rejected globally after stride50 |
| best coarse regime scale | regime/microstructure | `0.011098450` | `+0.000057455` | `0.005281938` | Positive but weaker |
| strong base on same frame | strong base | `0.011040995` | `0.000000000` | `0.005235980` | Reference |
| best risk shrink | risk/volatility | `0.011028489` | `-0.000012506` | `0.005165859` | Not promoted |
| alpha stack | alpha bridge | `0.008545801` | `-0.002495194` | `0.004058311` | Rejected |
| `latent_alpha_linear_stack` | raw family alpha | `0.003324032` | `-0.007716963` | `0.000957932` | Auxiliary only |

The minimal residual bridge slightly beat the broader residual bridge on
`stride100`, but the denser `stride50` run showed that the global residual
bridge was unstable. The tail-only version fixed the main failure mode by
leaving mid-weight rows untouched.

The denser `stride50` confirmation weakened the result materially:

| Candidate | Global R2 | Delta vs Strong Base | Min Fold R2 | Status |
| --- | ---: | ---: | ---: | --- |
| residual bridge | `0.013236138` | `+0.000000599` | `0.007486397` | Not promotable |
| strong base on same frame | `0.013235539` | `0.000000000` | `0.007787099` | Reference |
| residual gate open | `0.013234097` | `-0.000001443` | `0.007636543` | Rejected |
| residual gate closed | `0.013039380` | `-0.000196159` | `0.007790112` | Rejected |

This means the global `stride100` residual gain should be treated as unstable
sampled signal. The tail-only residual correction is now the only surviving
bridge hypothesis, because the same `weight q0.95` policy was best on both
`stride100` and `stride50`.

Tail-only `weight q0.95` behavior on the denser `stride50` frame:

- global delta: `+0.000126163`;
- `rw_02`: `+0.000404188`;
- `rw_03`: `+0.000064550`;
- `rw_04`: `+0.000099579`;
- `rw_05`: `+0.000052567`;
- `rw_01`: identity by construction.

Tail-only gains are concentrated exactly where intended:

- weight `q99_q100`: `+0.002092589`;
- weight `q90_q99`: `+0.000578418`;
- weight `q50_q90`: `0.000000000`;
- weight `q00_q50`: `0.000000000`.

The same candidate on `stride100` had global delta `+0.000111899`.
Fold deltas were positive except `rw_05`, which was essentially flat
(`-0.000004906`). This is supportive but still not full OOF evidence.

## Fold And Slice Behavior

Best residual bridge fold deltas versus the strong base:

- `rw_02`: `+0.000912874`
- `rw_03`: `+0.000187383`
- `rw_05`: `+0.000068419`
- `rw_01`: `0.000000000`
- `rw_04`: `-0.000424276`

The gain is concentrated where a risk/residual correction is plausible:

- weight `q99_q100`: `+0.001039066`
- weight `q90_q99`: `+0.000518011`
- baseline absolute prediction `q99_q100`: `+0.001051975`
- baseline absolute prediction `q90_q99`: `-0.000456188`

This is not a broad new alpha. It is a localized residual/risk correction on top
of the strong gateway candidate.

On `stride50`, the same candidate was nearly flat overall:

- global delta: `+0.000000599`;
- `rw_02`: positive versus base by about `+0.001628494`;
- `rw_03`: negative by about `-0.000767429`;
- `rw_04`: negative by about `-0.000920677`;
- `rw_05`: positive by about `+0.000177125`.

The high-weight and high-absolute-prediction tails remained positive, but losses
in the middle buckets erased almost all global improvement:

- weight `q99_q100`: `+0.002092589`;
- weight `q90_q99`: `+0.000302008`;
- weight `q50_q90`: `-0.000383359`;
- baseline absolute prediction `q99_q100`: `+0.001199972`;
- baseline absolute prediction `q50_q90`: `-0.000274411`.

## Regime And Microstructure

The first bridge configuration used `min_regime_rows=500`, which produced
`n_groups=0`; in that run the regime layer did not learn local regimes. It only
applied a previous-fold global scale after risk shrinkage.

A coarser run,
`strong_family_bridge_stride100_regime_coarse_v1`, used `time_bucket_size=200`,
`symbol_mod=4`, and `min_regime_rows=100`. This learned real regime groups:

- `rw_02`: `15-16` groups;
- `rw_03`: `141-143` groups;
- `rw_04`: `314-315` groups;
- `rw_05`: `402-406` groups.

Even with real groups, the best regime candidate reached only `0.011098450`,
below the residual bridge. Conclusion: the current regime definition is valid
and causal, but it is not yet the best connector. It may be useful later as a
gate over the residual correction, not as the main correction itself.

## Volatility And Risk Interpretation

Risk/volatility auxiliary targets are strongly learnable in the raw family run:

- `abs_error`: mean auxiliary R2 about `0.488699`;
- `abs_responder6`: about `0.487294`;
- `high_error`: about `0.152195`;
- squared targets: about `0.147`.

However, direct risk shrinkage of the strong base was negative on this sampled
bridge. The useful risk signal entered through the residual bridge, especially
`ridge_rank_alpha10000__risk_high_error_ridge_rank_score`. So the current
evidence says: use risk as a residual/gating feature, not as a standalone
monotone shrink rule.

In the denser `stride50` raw-family run, direct weak-family risk shrinkage became
the best raw-family candidate (`ridge_rank_alpha10000__risk_high_error_shrink_s0p2`,
`global_r2=0.003867448`), but this still remains far below the strong gateway
candidate. It supports risk as an auxiliary representation, not as a replacement
alpha.

## Residual Gate

A causal residual gate was implemented after the first bridge pass. It estimates
previous-fold group deltas between the strong base and residual bridge, then
applies or disables the residual correction in the current fold using only
observable groups: time bucket, symbol bucket, weight bucket, base prediction
magnitude and risk score bucket.

Two fixed policies are reported:

- `gate_open`: apply residual by default, close groups with negative historical
  evidence;
- `gate_closed`: keep base by default, open groups with positive historical
  evidence.

`gate_open` preserved most of the residual signal but did not beat the pure
residual bridge. `gate_closed` reduced some drawdown in bad folds but sacrificed
too much positive signal, especially in early folds. Both are rejected as current
promotion candidates.

## Audit

- Tests after bridge diagnostics, residual gate, and tail-only residual masks:
  targeted bridge/diagnostic tests passed.
- Full suite after tail-only implementation: `177 passed`.
- Target leakage: no target/responder column is used as a bridge feature.
- Lookahead: bridge coefficients, residual gates, tail thresholds, risk
  normalization, and regime scales use earlier folds only.
- Selection: all configured candidates are reported; diagnostics are post-hoc
  and do not refit candidates.
- Validation limitation: this is sampled OOF intersection evidence. Promotion
  requires dense or full OOF confirmation.

## Decision

Do not replace the strong candidate with raw alpha, raw stack, direct risk
shrinkage, current regime scale, global residual bridge, or the first residual
gate.

Keep the tail-only residual correction as the leading weak integration
hypothesis, not yet a promoted candidate:
`gateway_risk_conservative_rls_abs_pred_s100_prediction_family_residual_weight_q0p95_family_residual_tail`.

Next scientifically justified step: confirm this exact frozen policy on a full
or more independent OOF frame before any new tuning. It improves sampled bridge
R2, but it does not yet prove a full-score improvement above the historical
`0.015535283` robust checkpoint.
