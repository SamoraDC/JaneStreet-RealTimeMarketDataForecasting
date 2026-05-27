# Cheap OOF Search Synthesis - Stage 3

## Scope

This round used only saved OOF predictions and CPU-side meta-combination. No GPU training was used after starting this phase. The goal was to test cheap options until marginal returns saturated, then stop and synthesize before any new primary-model training. Large grids were stopped after IDE instability/GPU-availability symptoms; only completed reports with written `candidate_summary.csv` and passing audits are treated as evidence.

## Tested Families

- Subset Ridge stacks over saved TabM variants, tree-engine predictions, RLS predictions, and gateway risk-shrink predictions.
- Fixed predeclared two-way blends with weights `0.25`, `0.5`, and `0.75`.
- Walk-forward convex two-way blends fit only on earlier folds.
- Stack Ridge alpha checks around the best subset: `100`, `1000`, `10000`.
- Exploratory fine fixed-weight grid around `0.5`: `0.35` to `0.65`.
- Online daily scale calibration, where each validation date uses only earlier dates in the same fold to update a scalar multiplier.
- Online daily affine calibration, where each validation date uses only earlier dates in the same fold to update `bias + scale * prediction`.
- Gateway RLS prediction-space expansions, using target-free transforms of saved expert predictions inside the existing causal RLS update.
- Contextual row-level pair blends, fitted only on earlier folds and grouped by observable `weight`, prediction disagreement, and coarse time buckets.
- Cross-sectional batch prediction expansions, using target-free `batch_rank`, `batch_demean`, and `batch_zscore` inside each `date_id,time_id`.

## Key Results

| Status | Experiment | Candidate | Global R2 | Min Fold R2 | Notes |
|---|---|---:|---:|---:|---|
| New best local | `strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1` | `fixed_blend_0_w0p75_fixed_blend` | `0.014424968604` | `0.008046148603` | 75% subset stack + 25% conservative gateway risk S100 after signed-square, cube, pair-product, and target-free batch rank/mean/demean/std/zscore expansions. |
| Exact historical confirmation | `strong_oof_hist_max1398_s23aux8_s17_gateway_batch_mean_std_exact_v1` | `fixed_blend_0_w0p75_fixed_blend` | `0.015621628372` | `0.011794008455` | Same rule as the Stage 3 winner, with newly generated historical `seed23_aux8` and `seed17` OOF artifacts. Close, but below historical residual-tail best. |
| Sampled primary-preprocess residual | `strong_oof_batch_mean_std_stride50_rawpre_residual_tail_v1` | `strong_oof_ridge_stack_prediction_residual_weight_q0p95_residual_tail` | `0.014214327030` | `0.008345826769` | Paired `stride=50` rawpre intersection. Beats the same-subset stack control `0.014119788235`, but is not a full Stage 3 promotion. |
| Prior batch local | `strong_oof_subset_s23aux8_s17_gateway_batch_rank_stage3_narrow_v1` | `fixed_blend_0_w0p75_fixed_blend` | `0.014382557448` | `0.008044916993` | Same protocol without batch mean/std context. |
| Prior nonlinear local | `strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_pair_stage3_v1` | `fixed_blend_1_w0p75_fixed_blend` | `0.014182394686` | `0.007763466375` | 75% subset stack + 25% conservative gateway risk S100 after signed-square, cube, and pair-product RLS expansion. |
| Historical confirmation, not top | `strong_oof_hist_max1398_gateway_signed_square_cube_pair_v1` | `fixed_blend_5_w0p5_fixed_blend` | `0.015618963850` | `0.012831522319` | Confirms transfer direction, but remains below the historical residual-tail best. |
| Historical best retained | `strong_oof_hist_max1398_gateway_residual_tail_modes_v1` | `gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail` | `0.015630171202` | `0.012909969695` | Still the best historical max1398 reference. |
| Nested selector rejected | `fold_score_nested_stage3_nonlinear_vs_tail_v1` | `nested_previous_fold_selector` | `0.014135725957` | `0.007763466375` | Uses only previous fold statistics, but underperforms fixed nonlinear. |
| Hist nested selector rejected | `fold_score_nested_hist_max1398_nonlinear_vs_tail_mean_v1` | `nested_previous_fold_selector` | `0.015622193135` | `0.012890489078` | Best cheap selector variant, still below fixed historical residual-tail. |
| Contextual row gate rejected | `strong_oof_contextual_blend_stage3_stream_narrow_v1` | `ctx_blend_0_weight_tb100_min20000_p1000_contextual_blend` | `0.014176624576` | `0.007776857195` | Causal and memory-safe, but below the fixed `0.75/0.25` control at `0.014182394686`. |
| Batch expansion sampled confirmation | `strong_oof_subset_s23aux8_s17_gateway_batch_rank_smoke_stride5_v1` | `strong_oof_ridge_stack` | `0.014158650595` | `0.008014007810` | Sampled `stride=5` improved the paired nonlinear control `0.014071661350`, preceding the memory-guarded full narrow promotion. |
| Prior nonlinear local | `strong_oof_subset_s23aux8_s17_gateway_signed_square_stage3_v1` | `fixed_blend_0_w0p5_fixed_blend` | `0.014101721030` | `0.007437087406` | 50% subset stack + 50% aggressive gateway risk S25 after signed-square RLS expansion. |
| Prior best defensible | `strong_oof_subset_s23aux8_s17_stack_blend_stage3_v1` | `fixed_blend_1_w0p5_fixed_blend` | `0.014033829902` | `0.007406491677` | 50% subset stack + 50% conservative gateway risk S100. |
| Best exploratory ceiling | `strong_oof_cheap_subset_search_s23aux8_s17_weight_grid_stage3_v1` | `s03_11_fixed_risk_cons_s100_w0p6` | `0.014038604934` | `0.007471407693` | Weight chosen from validation grid; do not promote without confirmation. |
| Best raw subset stack | `strong_oof_subset_s23aux8_s17_stack_blend_stage3_v1` | `strong_oof_ridge_stack` | `0.013968302392` | `0.007684601218` | Uses default aux8 primary plus `seed23_aux8` and `seed17` extras. |
| Prior multi-TabM fixed | `strong_oof_multi_tabm_seeds_stack_stage3_v1` | `fixed_blend_1_w0p5_fixed_blend` | `0.014007491921` | `0.007375982593` | More extras did not help enough to justify all-column collinearity. |
| Best online scale | `strong_oof_subset_s23aux8_s17_online_scale_stage3_v1` | `fixed_blend_1_w0p5_fixed_blend_online_scale_f1_p10000_online_scale` | `0.013935812475` | `0.007464940028` | Causal, but worse than the unscaled fixed blend. |
| Best online affine | `strong_oof_subset_s23aux8_s17_online_affine_stage3_v1` | `fixed_blend_1_w0p5_fixed_blend_online_affine_f1_p10000_online_affine` | `0.013804602090` | `0.007421191875` | Causal, but worse than scale-only and worse than the uncalibrated blend. |
| Extra experts in gateway RLS | `strong_oof_subset_s23aux8_s17_gateway_extra_rls_stage3_v1` | `fixed_blend_1_w0p5_fixed_blend` | `0.014023668999` | `0.007554350092` | Better min fold, worse global. |
| Cube only | `strong_oof_subset_s23aux8_s17_gateway_cube_stage3_v1` | `fixed_blend_0_w0p5_fixed_blend` | `0.014105674101` | `0.007419391839` | Slightly above signed-square alone, but not enough without combination. |
| Pair product only | `strong_oof_subset_s23aux8_s17_gateway_pair_product_stage3_v1` | `fixed_blend_1_w0p75_fixed_blend` | `0.014104546166` | `0.007640547405` | Similar global to cube and better min fold. |
| Signed-square + cube | `strong_oof_subset_s23aux8_s17_gateway_signed_square_cube_stage3_v1` | `fixed_blend_1_w0p75_fixed_blend` | `0.014181106622` | `0.007756035917` | Most of the combined nonlinear gain. |
| Signed-square + abs | `strong_oof_subset_s23aux8_s17_gateway_signed_square_abs_stage3_v1` | `fixed_blend_0_w0p5_fixed_blend` | `0.014086922521` | `0.007428086104` | More complex than signed-square alone and worse global. |
| Extra experts + signed-square | `strong_oof_subset_s23aux8_s17_gateway_extra_signed_square_stage3_v1` | `fixed_blend_0_w0p5_fixed_blend` | `0.014038265767` | `0.007451090050` | Extra seeds inside RLS did not help once nonlinear terms were present. |
| Prior conservative checkpoint | memory/reference | `dynamic_gateway_rls_experts_alpha10000_f0p995` | `0.013836465` | not re-evaluated here | Preserved robustness checkpoint. |
| Historical max reference | memory/reference | historical max1398 | `0.015425344` to `0.015630171` | not re-evaluated here | Still above this cheap-search ceiling. |

## What Worked

The best subset was `seed23_aux8 + seed17`, not the full set of all saved TabM variants. The useful pattern is stable: a Ridge stack captures cross-model alpha, then the conservative gateway-risk prediction reduces fold risk when blended in.

The new best local blend is:

```text
0.75 * strong_oof_ridge_stack_prediction
+ 0.25 * gateway_risk_conservative_rls_abs_pred_s100_prediction
```

Here, the stack is built after adding target-free nonlinear prediction features
and target-free batch features inside the causal gateway RLS family:

- `prediction * abs(prediction)` (`signed_square`)
- `prediction ** 3` (`cube`)
- pairwise products between saved expert predictions (`pair_product`)
- `batch_rank`, `batch_demean`, and `batch_zscore` computed only inside each observable `date_id,time_id` batch
- `batch_mean` and `batch_std` computed inside the same observable batch as prediction-context features

The final Stage 3 blend shifts back toward the conservative risk-shrunk RLS
branch. The gain is not from a larger primary model; it is from a richer
prediction-space basis inside the same strict past-update RLS simulator.

Fold R2 for this candidate:

```text
rw_01: 0.014362370575
rw_02: 0.027209033134
rw_03: 0.009559255052
rw_04: 0.011355883054
rw_05: 0.008046148603
```

Historical max1398 confirmation for the same nonlinear family reached
`0.015618963850`, with fold R2:

```text
rw_01: 0.016230140813
rw_02: 0.012831522319
rw_03: 0.013570511060
rw_04: 0.013145290197
rw_05: 0.021155052284
```

This is close to, but still below, the pre-existing historical residual-tail
best `0.015630171202`. Therefore the nonlinear basis is useful evidence, but
not a promoted operational replacement.

The exact historical confirmation of the later batch mean/std Stage 3 winner
used the same fixed blend and the same prediction-space expansions:

```text
strong_oof_hist_max1398_s23aux8_s17_gateway_batch_mean_std_exact_v1
fixed_blend_0_w0p75_fixed_blend
global_r2=0.015621628372
min_fold_r2=0.011794008455
```

Fold R2:

```text
rw_01: 0.016511689222
rw_02: 0.013221960466
rw_03: 0.014442493048
rw_04: 0.011794008455
rw_05: 0.021172809364
```

This beats the earlier nonlinear historical confirmation (`0.015618963850`) but
still trails the residual-tail historical reference (`0.015630171202`) by
`0.000008543`. The practical conclusion is unchanged: batch context transfers,
but it does not yet justify more cheap meta-search.

## What Did Not Work

- Loading and testing all five extra TabM variants in one cheap-grid process reached about 15 GB RSS, so that path was stopped for memory discipline.
- The all-extra stack lowered raw stack quality (`0.013898097704`) despite a slightly useful fixed blend.
- `w010 + seed23_aux8` topped out near `0.013991`.
- `seed23_aux8 + seed23 + seed37` topped out near `0.013992`.
- Stack alpha `100` and `10000` did not improve over `1000`.
- The fine fixed-weight grid improved only from `0.014033829902` to `0.014038604934`, a tiny gain with clear validation-selection risk.
- Online scale calibration lowered the best defensible candidate from `0.014033829902` to `0.013935812475`.
- Online affine calibration lowered it further to `0.013804602090`. Adding a free intercept did not reveal a cheap missing bias; it increased adaptation variance.
- Including saved extra TabM seeds directly inside gateway RLS worsened global score despite improving the minimum fold.
- Adding `abs(prediction)` on top of `prediction * abs(prediction)` worsened the signed-square result, so the smaller expansion is preferred.
- Combining extra gateway experts with signed-square also worsened global score.
- `signed_sqrt`, `signed_log1p`, and `sign` compressions lost against the combined tail/interactions basis.
- Large residual-tail and risk-profile grids were aborted or lost before writing summaries after machine instability. They are explicitly excluded from evidence.
- A cheap nested selector over saved `fold_scores.csv` was implemented in `multi-models/run_fold_score_selection.py`. It uses only earlier fold sufficient statistics, but did not improve. Stage 3 variants stayed at `0.014135725957` or worse versus fixed nonlinear `0.014182394686`. Historical variants topped at `0.015622193135` versus fixed residual-tail `0.015630171202`.
- The leaky fold oracle over the same small candidate pool is only `0.014200766924` on Stage 3, so even perfect whole-fold switching has tiny local headroom. Historical oracle is `0.015724775324`, which suggests some regime-specific upside exists, but the causal fold selector failed to recover it robustly.
- Contextual row-level pair blends were implemented and then rewritten in streaming array form to avoid RAM blow-ups from materializing many OOF columns. The memory-safe full narrow run still lost: best contextual `0.014176624576` versus fixed blend `0.014182394686`. This kills the current cheap row-gate formulation.
- The first broad full batch-expansion run was aborted after swap pressure. It is excluded from evidence. A narrow full run with residual/risk/regime side families disabled completed under explicit RAM/swap guards and improved the paired narrow control from `0.014182394686` to `0.014382557448`.
- The first ablation smokes for `batch_rank`, `batch_demean+batch_zscore`, `batch_rank+zscore`, and `batch_rank+demean` were later found non-comparable because they used `12` stack features instead of the `14` features in the promoted protocol. They are not evidence. Corrected smokes with explicit `--strong-base-candidates` showed `batch_rank=0.014145`, `batch_demean+batch_zscore=0.014162`, prior all-batch `0.014159`, and `+batch_mean+batch_std=0.014201`.
- Raw feature preprocessing failed when appended directly to the full strong OOF meta-stack, but survived as a primary weak alpha in `multi-models`. On `stride=50`, direct inclusion in the strong stack was effectively neutral (`0.014118022160` vs control stack `0.014119788235`), while a residual-tail correction using rawpre alpha features improved the same-subset stack to `0.014214327030`. The frozen denser `stride=20` confirmation then failed (`0.014064611992` vs control `0.014070753306`), so this bridge is rejected as a promotion path.

## Audit

- `target_leakage_check`: passed in the principal report.
- `fold_causality_check`: passed; walk-forward layers use earlier folds only, gateway updates use prior-date lag simulation, and online scale/affine calibration updates only after each validation date is scored.
- `gateway_bad_updates`: `0`.
- Gateway nonlinear features are target-free transforms of predictions available before scoring; they do not use responders, current-date targets, or validation labels.
- Online scale and affine implementations are unit-tested for same-date non-observability: date `D` is predicted before date `D` updates the calibration state.
- Fixed blends are deterministic once weights are chosen, but picking the best weight from the same validation report is exploratory and should not be treated as robust.
- The new nonlinear Stage 3 and historical reports both passed `target_leakage_check`, `fold_causality_check`, and had `gateway_bad_updates=0`.
- Fold-score nested selection audit passed for leakage/causality, but is diagnostic only because it selects whole-fold candidates and is not yet a deployable row-level inference policy.
- Contextual blend scoring now has a streaming implementation, `score_walk_forward_contextual_candidate_blends`, that does not join prediction columns into the full OOF frame unless `--write-predictions` is requested. A unit test checks that streaming fold scores match the materialized prediction path.
- Batch prediction expansions are target-free and computed only within the observable `date_id,time_id` batch. Unit tests cover rank/demean/zscore behavior and ensure no target dependency.
- A broad full Stage 3 run of the batch expansion family was started and then aborted after swap pressure closed the IDE. The output directory was empty and the run is explicitly excluded from evidence.
- The promoted batch-expansion result is the later narrow full run, which disabled nonessential residual/risk/regime side families and used resource guards. Its paired narrow control used the same setup without `batch_rank,batch_demean,batch_zscore`.
- The new `batch_mean/std` full run passed `target_leakage_check`, `fold_causality_check`, and had `gateway_bad_updates=0`. It did use swap transiently, so further full runs in this family should require tighter streaming or a higher swap floor.
- The primary-preprocessing loader now accepts weak family prediction columns as extra OOF predictions. Focused strong OOF tests and the full suite passed after this change; final full-suite result in this round was `212 passed in 8.60s`.
- `multi-models/run_strong_oof_experiment.py` now exposes `--min-mem-available-gb` and `--min-swap-free-gb` to stop future runs after resource-floor violations.
- Validation commands passed:

```bash
uv run python -m py_compile multi-models/run_cheap_oof_subset_search.py multi-models/run_strong_oof_experiment.py multi-models/multimodels/strong_oof.py
uv run pytest -q multi-models/tests/test_strong_oof.py tests/test_tree_engine_ensemble_script.py
uv run pytest -q
```

Focused fold-score selection and strong OOF tests: `17 passed in 0.70s`.

After the contextual streaming patch: `uv run pytest -q multi-models/tests/test_strong_oof.py` produced `16 passed`; `uv run python -m py_compile multi-models/multimodels/strong_oof.py multi-models/run_strong_oof_experiment.py` passed.

Final full-suite check after the streaming contextual scorer and documentation update: `uv run pytest -q` produced `200 passed in 6.55s`.

Final full-suite result after the nonlinear-expansion patch, before the fold-score selector module: `192 passed in 6.00s`.

Final full-suite result after batch expansion resource guards and tests:
`uv run pytest -q` produced `208 passed in 7.42s`.

After adding `batch_mean` and `batch_std`, focused strong OOF tests produced
`19 passed`, and the full suite produced `208 passed in 3.39s`.

## Conclusion

The cheap OOF/meta layer moved from `0.01404` to `0.014182394686` after allowing a nonlinear prediction basis inside the causal gateway RLS, then to `0.014382557448` after adding observable batch rank/demean/zscore expansions, and now to `0.014424968604` after adding batch mean/std context. This is a real local improvement over the preserved conservative checkpoint, but it is still below the historical max1398 best `0.015630171202` and far from `0.02`.

The current CPU-side candidate to freeze for follow-up is `fixed_blend_0_w0p75_fixed_blend` from `strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1`. Because the comparable historical confirmation has not yet been rerun for this exact batch-expanded family, it should be treated as the best local Stage 3 ablation and possible ensemble ingredient, not a promoted submission candidate. Cheap fold-level selection and the current contextual row gate remain rejected. The next material step is either a historical/nested confirmation of this exact batch-expanded rule or a stronger primary alpha; GPU should be used only for a predeclared primary-model run with resource limits.
