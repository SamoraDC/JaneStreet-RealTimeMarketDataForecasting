# Research Evolution And Experiment History

This document summarizes the evolution of the Jane Street research project from
the first local baselines to the preserved final candidates. It is written as a
public-facing technical history: what was tested, why it was tested, what the
empirical result was, and what decision followed.

The detailed personal audit archive lives in `path/docs/`. The preserved
GitHub-facing candidate packages live in `best-candidates/`.

## Final State

The research line ended with three preserved references:

| Role | Candidate | Validation | Score |
| --- | --- | --- | --- |
| Best local Stage 3 result | `strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1/fixed_blend_0_w0p75_fixed_blend` | Stage 3 full OOF | `global_r2=0.014424968604`, `min_fold_r2=0.008046148603` |
| Best full historical result | `strong_oof_hist_max1398_gateway_residual_tail_modes_v1/gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail` | Historical `max_date_id=1398` OOF | `global_r2=0.015630171202`, `min_fold_r2=0.012909969695` |
| Conservative operational reference | `dynamic_gateway_rls_experts_alpha10000_f0p995` | Stage 3 and historical gateway/RLS validation | Stage 3 `global_r2=0.013836465`; historical `global_r2=0.015425344` |

The strongest technical lesson is that the best validated gains came from
causal meta-learning over saved OOF predictions and target-free cross-sectional
prediction context inside the observable `date_id,time_id` batch. Purely local
tuning of blends, selectors, shrinkage, and residual masks saturated below the
desired `0.02` level.

## How To Read This Document

This file is a chronological and methodological map, not only a scoreboard. Each
experiment family is summarized through the same lens:

- the modeling question being tested;
- the baseline or control it had to beat;
- the validation protocol used;
- the strongest empirical result;
- the reason the family was promoted, preserved, or killed.

Several reported numbers are intentionally negative or below the best known
candidate. They are included because they constrain future work. A failed
experiment is useful when it says which hypothesis, data regime, or validation
assumption did not survive temporal testing.

For the scientific protocol behind the tests, read
`docs/scientific_methodology.md`.

## Score Ladder

The approximate local score progression looked like this:

| Stage | Reference | Approximate global R2 | Main lesson |
| --- | --- | ---: | --- |
| Zero baseline | constant zero prediction | `0.000000` | Metric sanity check. |
| Raw Ridge | weighted Ridge, rolling folds | `0.000654` | Linear signal exists but is unstable. |
| Calibrated Ridge | causal amplitude control | `0.004805` | Calibration fixes large extrapolation failures. |
| Ridge + GBDT | low-cost nonlinear blend | `0.006443` to `0.006537` | Nonlinear interactions matter. |
| Tree engine ensemble | Ridge + XGBoost + LightGBM OOF simplex | `0.008947` | Stronger tabular engines and OOF blending improve both global and worst fold. |
| TabM / tree / Ridge meta layer | Bayesian/gateway online Ridge | `0.013824` to `0.013862` | The large gain comes from causal meta-learning over strong OOF experts. |
| Dynamic RLS | forgetting-factor gateway meta layer | `0.013836` to `0.013866` | Continuous causal adaptation gives small but real improvements. |
| Strong OOF with batch prediction context | fixed blend with batch mean/std features | `0.014425` | Same-batch prediction geometry is the strongest local Stage 3 insight. |
| Historical residual-tail | conservative RLS residual tail | `0.015630` | Tail-specific residual correction is the strongest historical reference. |

The desired `0.02` level was not reached by a full validated candidate. Values
above `0.02` appeared only as folds, slices, probes, or diagnostic upper bounds,
and were not treated as promotable evidence.

## Methodological Rules

### Metric

All promoted local results use the weighted zero-mean R2:

```text
R2 = 1 - sum_i w_i * (y_i - p_i)^2 / sum_i w_i * y_i^2
```

The zero predictor scores exactly `0.0`, which makes it a useful metric sanity
check.

### Temporal Validation

The main protocol is five rolling temporal folds over `date_id`, commonly with
`valid_window=60`. Every fold satisfies:

```text
train_end < valid_start
```

This is an offline causal approximation of the competition's streaming setup,
not a literal private-leaderboard simulator.

### Causality Rules

Promoted candidates must avoid:

- target leakage;
- responder leakage;
- look-ahead bias;
- validation-set hyperparameter selection disguised as a fixed rule;
- hidden fallbacks that make a score non-reproducible.

Gateway and online candidates update only from data that would have been known
before the current prediction batch. The strongest gateway artifacts record
checks such as `bad_updates=0`, `all_strictly_past=true`,
`target_leakage_check=passed`, and `fold_causality_check=passed`.

### Evidence Tiers

The project uses evidence tiers:

| Tier | Meaning |
| --- | --- |
| Smoke | Confirms that a pipeline runs and has nontrivial direction. Not promotable. |
| Probe | Tests an idea cheaply. Useful for learning, not for final claims. |
| Partial validation | Screens candidates before expensive full validation. |
| Full Stage 3 validation | Main local promotion protocol. |
| Historical `max_date_id=1398` validation | Additional regime check against an earlier historical cutoff. |
| Runtime package | Operational packaging evidence. It does not prove official leaderboard score. |

Plain-language reading:

- `Stage 3` is the recent local stress test. It was used to ask whether a
  candidate still worked near the end of the available training timeline.
- `Historical` validation is the broader temporal sanity check. It was used to
  ask whether a candidate survived more than one local regime view.
- `Runtime package` is an engineering-readiness check. It was used to ask
  whether the method could be exported into the Kaggle-style online
  `predict(test, lags)` loop.
- `Official leaderboard` would require an accepted Kaggle submission. Local
  validation and runtime packaging are not the same as official leaderboard
  evidence.

This is why the preserved candidates have different roles. The batch mean/std
fixed blend is the strongest local Stage 3 result. The historical residual-tail
candidate is the strongest wider historical result. The conservative dynamic
RLS candidate is the most operationally ready because it has the closest
Kaggle-style runtime package, even though it is not the highest local score.

## Experiment Record Format

Most experiments in the private audit archive followed this structure:

```text
hypothesis -> implementation -> validation command -> fold metrics
           -> leakage audit -> decision -> next action or kill rule
```

That format matters because the project had many small possible improvements.
Without explicit hypotheses and kill criteria, it would be easy to turn the
research into validation-set tuning. The public summary preserves the decision
trace so that a future contributor can tell the difference between:

- an implementation failure;
- a mathematical failure;
- a data-regime failure;
- a validation failure;
- a promising but not-yet-operational idea.

## Evolution Timeline

### 1. Zero Baseline And Metric Foundation

The first layer established the local metric, deterministic temporal folds, and
canonical paths. The zero predictor was used as a metric sanity check because it
must score exactly `0.0`.

Key modules:

- `src/janestreet/metrics.py`
- `src/janestreet/folds.py`
- `src/janestreet/paths.py`
- `scripts/run_zero_baseline.py`

Decision: keep the zero baseline as a permanent health check, not as a modeling
candidate.

### 2. Weighted Ridge Baseline

The first real model family was a weighted Ridge regression fitted over temporal
rolling folds. The early five-fold rolling sweep with `train_window=120`,
`valid_window=60`, and `alpha=1000` reached only:

```text
global_r2=0.000654028
mean_r2=0.000589161
min_r2=-0.008888301
```

The negative `rw_02` fold became a major diagnostic target. Row-level and slice
analysis identified a dominant failure around:

```text
date_id=1489
symbol_id=25
slice_r2=-8.612948
```

Decision: Ridge alone was a useful baseline and diagnostic tool, but not a
candidate strategy.

### 3. Ridge Calibration And Amplitude Control

The next hypothesis was that Ridge failed partly by amplitude extrapolation, not
only by weak signal. Causal clipping, scaling, and group calibration were added.

Important results:

```text
time_bucket x weight_bucket shrinkage:
global_r2=0.004186872951
min_r2=0.002842881287

high-weight update:
global_r2=0.004600982867

weight_bucket x prediction_abs_bucket with internal OOF 3x20:
global_r2=0.004805336851
min_r2=0.002904930235
```

Decision: promote calibrated Ridge as a stronger baseline, but continue toward
nonlinear tabular models because the score ceiling remained low.

### 4. Early Cross-Sectional Feature Attempts

The project tested whether same-batch cross-sectional context could help Ridge.
Early RP8 and score-context formulations were causal because they used only the
observable `date_id,time_id` batch, but they did not beat the calibrated Ridge
baseline.

Representative results:

```text
RP8 best:        global_r2=0.003709
score context:  global_r2=0.003741, min_r2=0.001527
baseline:       global_r2=0.004186873
```

Decision: reject the early cross-sectional formulation, but keep the broader
idea alive because the batch is causally observable.

### 5. Small GBDT And Ridge/GBDT Blends

A small sklearn GBDT was introduced to capture nonlinear interactions that Ridge
could not represent. The conservative GBDT improved global score but was less
stable than calibrated Ridge:

```text
conservative GBDT:
global_r2=0.005734
min_r2=0.001360
```

Blending calibrated Ridge with GBDT became the first stronger low-cost ensemble:

```text
Ridge + conservative GBDT, seed 17:
global_r2=0.006442732
min_r2=0.003383691

10% sample, seeds 17/23/37:
global_r2=0.006537188
mean_r2=0.006492084
min_r2=0.003205592
```

Decision: promote the Ridge/GBDT blend as a historical stepping stone, while
tracking the tradeoff between better global score and worse worst-fold score.

### 6. Rejected Feature Geometry Bridges

Several feature families were tested as deliberate bridges away from the default
linear/GBDT setup:

- F3 temporal geometry;
- F5 reservoir-lite;
- F6 multiscale/wavelet-lite;
- F7 graph-symbol features;
- F8 Koopman/EDMD-lite features.

Representative result:

```text
temporal geometry with GBDT:
global_r2=0.006336228
min_r2=0.003323327

same baseline without F3:
global_r2=0.006442732
min_r2=0.003383691
```

The graph/reservoir, multiscale, and Koopman families were implemented causally
but failed to improve the active baseline.

Decision: record the negative evidence and stop expanding these feature
families in their tested forms.

### 7. Official Lag Reconstruction

Official responder lags were added and audited. The key causality rule is that
only previous-day responders are available at the start of a new `date_id`.

Findings:

- `same_time` lag formulation failed smoke tests.
- `daily_last` lags produced a small positive smoke in GBDT.
- Five-fold validation still failed to beat the active no-lag baseline.

Representative result:

```text
10% seed ensemble with daily_last lags:
global_r2=0.006399649

no-lag blend baseline:
global_r2=0.006537188
```

Decision: keep official lag infrastructure because it matters for gateway
runtime, but do not promote the initial lag-only formulation.

### 8. Stronger Tree Engines

The next improvement came from stronger tree engines and seed ensembles:

```text
XGBoost + IDs, 10% sample, seeds 17/23/37:
global_r2=0.008841346
mean_r2=0.008709197
min_r2=0.004342244

LightGBM + IDs:
global_r2=0.008698894
min_r2=0.004361963

CatBoost with categorical IDs:
global_r2=0.008386916
min_r2=0.004189512
```

A per-fold OOF simplex ensemble of calibrated Ridge, XGBoost, and LightGBM then
became the active comparison baseline:

```text
tree_engine_ensemble_xgb_lgb_sample10_seed_ensemble:
global_r2=0.008946915
mean_r2=0.008799602
min_r2=0.004403783
rows=11,028,424
```

Decision: use the tree engine ensemble as the active baseline for later work,
not the older `0.006537188` sklearn blend.

### 9. Online Linear Learning

The project tested SGD, Huber-averaged SGD, Passive-Aggressive, and online Ridge
variants to see whether causal adaptation alone could beat static tabular
models.

Best online linear result:

```text
expanding online Ridge without IDs:
global_r2=0.000653664
```

Decision: reject online linear learning as a primary path. Adaptation matters,
but the representation was too weak.

### 10. Neural And TabPFN Probes

Recurrent baselines and TabPFN were tested as exploratory alternatives.

Findings:

- GRU/LSTM smokes showed some signal but remained below Ridge calibration and
  tree baselines in stronger validation.
- One sequence primary-alpha probe was negative:

```text
sequence primary alpha probe:
global_r2=-0.004801
```

- TabPFN v2 cached runs were negative.
- TabPFN v3 was blocked by an unaccepted `tabpfn-3-license-v1.0` license.

Decision: keep direct recurrent and TabPFN paths exploratory unless a new
representation or stacker role beats the active temporal-fold baseline.

### 11. Competitive TabM Path

The neural path became credible only after switching to the official `tabm`
package and preserving the ensemble axis correctly during training. The TabM
line used:

- official lag reconstruction;
- online update simulation;
- auxiliary targets such as `aux8`;
- larger temporal training windows;
- saved validation predictions for later OOF stacking.

The TabM family eventually became a strong component, especially in historical
validation, but did not by itself solve the full score gap.

Representative later results:

```text
TabM seed37 aux8, Stage 3 component in consolidated blend:
global_r2 around 0.0127

historical TabM seed37 aux8:
global_r2=0.015231527
min_r2=0.012535378
```

Decision: use TabM as a strong primary component and OOF source, not as the
entire final system.

### 12. Bayesian Meta Layers And Gateway Simulation

The project then moved from primary models to causal meta-learning over saved
OOF predictions.

Initial Bayesian/meta results included:

```text
bayesian_online_ridge_experts:
global_r2=0.013824562
mean_fold_r2=0.013445262
min_fold_r2=0.007028562

alpha=1000 sensitivity:
global_r2=0.013854891
min_fold_r2=0.006981445
```

A stricter gateway simulation updated the meta-layer at the start of `date_id=D`
using only previous-day lagged responders and cached previous-day predictions.

Key gateway result:

```text
gateway_online_ridge_components_no_tree_ensemble_alpha1000:
global_r2=0.013862361
min_fold_r2=0.006967319
bad_updates=0
```

The conservative alternative had slightly lower global score but stronger
operational appeal:

```text
experts_alpha10000:
global_r2=0.013824562
min_fold_r2=0.007028562
```

Decision: promote gateway meta-learning as the main structural improvement over
tree/TabM components.

### 13. Historical Gateway Confirmation

A clean historical confirmation was performed with `max_date_id=1398`, requiring
historical OOF predictions for both TabM and tree models.

Important historical results:

```text
historical TabM aux8:
global_r2=0.015231527
min_r2=0.012535378

historical tree ensemble:
global_r2=0.005956396

frozen gateway experts_alpha10000:
global_r2=0.015407033

historical baseline tabm_tree_convex_walk_forward:
global_r2=0.015224315
```

Decision: the gateway family generalized to the historical protocol, with the
conservative expert set becoming the operational reference direction.

### 14. Dynamic RLS / Kalman-like Forgetting

The frozen gateway ridge was extended into a dynamic RLS update with forgetting
factors. This made the meta-layer adapt more continuously while preserving the
same causal ordering.

Preserved conservative candidate:

```text
dynamic_gateway_rls_experts_alpha10000_f0p995

Stage 3:
global_r2=0.013836465
min_fold_r2=0.007030887

Historical max1398:
global_r2=0.015425344
min_fold_r2=0.012924724
```

A more aggressive Stage 3 variant reached:

```text
dynamic_gateway_rls_components_no_tree_ensemble_alpha1000_f0p995:
global_r2=0.013866043
```

Decision: keep two interpretations separate:

- aggressive RLS for highest local Stage 3 score in that family;
- conservative RLS for operational robustness and historical stability.

### 15. RLS Strategy Selection And Shrinkage

The project tested whether selectors, calibrators, posterior shrinkage, and
softmax EWMA routing could improve over the preserved RLS lines.

Findings:

- No selector dominated the aggressive Stage 3 RLS line.
- Softmax EWMA helped historically but worsened Stage 3.
- Posterior shrinkage produced micro-gains but not a new dominant thesis.
- Daily oracle variants produced higher numbers but were explicitly leaky and
  used only as diagnostic upper bounds.

Representative diagnostics:

```text
daily oracle Stage 3:
global_r2=0.014550499

daily oracle historical:
global_r2=0.016147181
```

Decision: do not promote oracle or selector lines. Preserve the conservative
dynamic RLS candidate.

### 16. Strong OOF Stack And Cheap Meta-search

A stronger OOF experiment framework was built under `multi-models/`. It combined
saved OOF predictions, gateway RLS features, risk shrinkage, nonlinear
prediction transforms, fixed blends, and residual-tail candidates.

Early cheap search result:

```text
0.5 * strong_oof_ridge_stack_prediction
+ 0.5 * gateway_risk_conservative_rls_abs_pred_s100_prediction

global_r2=0.014033829902
min_fold_r2=0.007406491677
```

Target-free nonlinear prediction expansions helped:

```text
signed_square + cube + pair_product:
global_r2=0.014182394686
min_fold_r2=0.007763466375
```

Decision: prediction-space transformations were useful, but the large gain came
from adding batch prediction context.

### 17. Batch Prediction Expansions

The strongest local Stage 3 result came from target-free batch statistics over
expert predictions inside each observable `date_id,time_id` batch:

- `batch_rank`
- `batch_mean`
- `batch_demean`
- `batch_std`
- `batch_zscore`

Best Stage 3 result:

```text
strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1
fixed_blend_0_w0p75_fixed_blend

global_r2=0.014424968604
min_fold_r2=0.008046148603
rows=11,028,424
```

Formula:

```text
prediction =
  0.75 * strong_oof_ridge_stack_prediction
+ 0.25 * gateway_risk_conservative_rls_abs_pred_s100_prediction
```

Historical confirmation of the same frozen rule:

```text
strong_oof_hist_max1398_s23aux8_s17_gateway_batch_mean_std_exact_v1
fixed_blend_0_w0p75_fixed_blend

global_r2=0.015621628372
min_fold_r2=0.011794008455
rows=11,151,360
```

Audit status:

```text
target_leakage_check=passed
fold_causality_check=passed
selection_check=passed
gateway_bad_updates=0
```

Decision: preserve this as the best local Stage 3 result and strongest insight
about prediction-space batch context. Do not treat it as the operational runtime
candidate until the logic is exported into `predict(test, lags)`.

### 18. Historical Residual-tail Candidate

The best full historical candidate used a conservative RLS base, an `abs_pred`
risk shrinkage profile, and a residual correction applied only in a high-weight
and high-absolute-prediction tail.

Best historical result:

```text
strong_oof_hist_max1398_gateway_residual_tail_modes_v1
gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail

global_r2=0.015630171202
min_fold_r2=0.012909969695
rows=11,151,360
```

Decision: preserve as the best historical full result. Treat it as more
experimental than conservative dynamic RLS because residual-tail rules are more
sensitive to regime and require additional runtime export work.

### 19. Raw Feature Preprocessing

Raw `feature_*` preprocessing was tested in two ways:

1. Directly appended into the strong OOF meta-stack.
2. Used as a weak primary-model bridge.

Direct append failed. The stronger primary raw preprocessing idea showed signal
but did not improve the full stack.

Representative results:

```text
raw9 rank-only 1.2M primary:
global_r2=0.011454561206148
min_fold=0.006670550392639

saved OOF recomputation:
raw9rank_tabm_prediction global_r2=0.011444166486
min_fold_r2=0.006373763084
```

Post-integration ablations showed that the gain was not explained by pure rank,
pure missingness, or relative-only preprocessing alone.

Decision: raw batch preprocessing contains weak signal, but the tested
integration paths were not strong enough to displace the preserved OOF/RLS
family.

### 20. Tail Control And Regime Gates

Clock and missingness features were tested as causal regime controls over the
tree ensemble and tail regions.

Useful but non-final result:

```text
batch_missing + clock_simplex_tail_q90_q100:
global_r2=0.009010271
min_fold_r2=0.004536404
```

This beat the active tree ensemble in that context, but later online tail-control
validation did not produce a robust promoted rule. Regime/confidence gates showed
small recurring benefits in some windows, but not enough worst-fold dominance.

Decision: keep the infrastructure and evidence, but do not promote tail-control
or regime gates above the preserved RLS/strong OOF candidates.

### 21. Primary Batch-aware TabM And Set-aware Final Probe

The final structural question was whether batch-aware primary models could
produce a new source of alpha, rather than only improving OOF post-processing.

TabM with batch raw preprocessing had valid causal implementation and some weak
signal, but it did not improve the strong stack.

The last structural family was a `batch_deepset` model that consumed observable
`date_id,time_id` batches and learned a set-aware latent context. It was
implemented and tested in a five-fold smoke:

```text
primary_batch_deepset_smoke_400k_seed37_v1
global_r2=0.003538596379
min_fold_r2=0.000237110623
rows=11,028,424
```

Decision: kill the current set-aware primary family. It was a valid
implementation of the hypothesis, but empirically far below strong baselines and
not worth scaling.

### 22. Submission Packaging

The conservative dynamic RLS candidate was exported into runtime artifacts and
packaged for Kaggle.

Selected candidate:

```text
dynamic_gateway_rls_experts_alpha10000_f0p995
```

Package artifacts were prepared under:

```text
kaggle_upload/
kaggle_kernel/
best-candidates/conservative_dynamic_gateway_rls/kaggle_upload/
best-candidates/conservative_dynamic_gateway_rls/kaggle_kernel/
```

The local and Kaggle notebook workflows produced `submission.parquet`. Official
submission was blocked because submissions were disabled for the competition.

Decision: preserve the package as operational evidence, not as an official
leaderboard claim.

## Expanded Experiment Ledger

This section gives a more detailed view of the important experiment families.
The goal is to make clear what each family actually tested and what it changed
relative to the previous baseline.

### Ridge And Slice Diagnostics

The raw Ridge work established three durable principles. First, metric and fold
math had to be trusted before model complexity increased. Second, high global
variance in a fold could be explained by a small number of heavy slices. Third,
diagnostics could guide future model design but could not be used to remove bad
rows from reported metrics.

The `rw_02` failure is the canonical example. Removing
`date_id=1489,symbol_id=25` would have made the fold look much better, but doing
so would be post-hoc target-conditioned pruning. The valid conclusion was only
that the model was vulnerable to a localized high-weight regime, especially
early in the day and under partial missingness.

### Calibration As A Bridge

The calibrated Ridge stage changed the role of the model. Ridge stopped being
treated as a final predictor and became a raw signal that needed causal
amplitude control. This was an important conceptual move: the project did not
only tune `alpha`; it introduced clipping, shrinkage, and group-specific scale
rules learned without using future validation targets.

The best Ridge-calibration result was still far below later tree and TabM
families, but it proved that instability control could be worth more than simply
adding model capacity.

### Tree Engines As Stronger Primary Predictors

The tree-engine phase separated model-family improvement from validation
protocol changes. XGBoost, LightGBM, and CatBoost were tested under the same
rolling-fold discipline, and `time_id`/`symbol_id` were used where appropriate
for nonlinear engines rather than forced into the Ridge baseline.

The promoted tree ensemble learned per-fold simplex weights on internal OOF
calibration data. This avoided choosing ensemble weights directly on the
external validation block. It also revealed real regime variation: different
folds preferred different mixtures of Ridge, XGBoost, and LightGBM.

### Official Lags And Gateway Semantics

Official lags were not treated as just another feature engineering block. They
changed the operational model of time. In the competition-style gateway, the
model predicts the current batch while receiving responder information only for
past days. Any local reconstruction had to respect that delivery rule.

The initial lag feature variants did not beat the active baseline, but the lag
infrastructure became necessary for TabM online adaptation, gateway meta
simulation, and submission packaging.

### TabM As Primary Signal

The early neural path was weak until the implementation switched to the official
TabM formulation with independent submodels preserved along the ensemble axis.
That correction mattered because averaging submodels before the loss destroys
part of the intended TabM training signal.

The successful TabM line combined:

- official previous-day lags;
- online updates during validation;
- auxiliary responder targets;
- larger training windows;
- OOF prediction export for downstream meta layers.

TabM became a strong source of predictions, especially historically, but the
final edge came from combining it with trees and Ridge through causal
meta-learning.

### Bayesian And RLS Meta-learning

The Bayesian/gateway stage treated saved OOF predictions as expert opinions.
Instead of asking one primary model to solve the entire problem, the meta-layer
learned how to combine experts as regimes changed over time.

The strict gateway simulation was the critical audit step. The update for
`date_id=D` used only responder lags from `D-1` joined to cached predictions
from `D-1`. The current day's target was never used before predicting the
current day. That ordering is what made the gateway scores meaningfully closer
to a submission contract than ordinary offline stacking.

Dynamic RLS then added forgetting factors. The preserved conservative setting,
`alpha=10000` and `forgetting_factor=0.995`, traded a little local Stage 3 score
for historical stability and simpler operational reasoning.

### Strong OOF And Prediction-space Geometry

The strong OOF framework converted the project from isolated model training into
a controlled search over prediction-space transformations. It asked whether
expert predictions contained nonlinear or contextual structure that a linear
blend was missing.

Three target-free transform families mattered:

- scalar nonlinear transforms such as `signed_square` and `cube`;
- pairwise products between expert predictions;
- same-batch statistics such as rank, mean, demeaned value, standard deviation,
  and z-score.

The batch prediction features were the most important. They used only current
batch predictions, not current targets, and were therefore causally observable
at prediction time if the same experts are available in the gateway.

### Residual-tail Logic

Residual-tail experiments asked a narrower question: can a correction help only
where the base model is most likely to make large weighted errors? The best
historical candidate applied a residual correction only when both `weight` and
`abs(base_prediction)` exceeded previous-fold 0.95 quantile thresholds.

This was not promoted as the lowest-risk runtime candidate because tail
thresholds, residual coefficients, and correction rules need careful export into
online inference. It was preserved because it produced the best full historical
score and showed that tail-local modeling remained useful.

### Raw Batch Features And Primary Batch-aware Models

The raw preprocessing stage tested whether the same cross-sectional insight
could move from prediction space back into raw feature space. Directly adding
raw preprocessed features into the strong OOF stack failed. A primary TabM with
batch-rank style features produced a valid weak signal, but it did not improve
the full strong stack after integration.

The final `batch_deepset` probe tested a deeper set-aware architecture directly
over observable batches. Its low score killed that implementation path. The
failure did not disprove all possible set-aware architectures, but it did
disprove the current implementation strongly enough to stop local score search.

### Packaging And Operational Evidence

The Kaggle packaging step was intentionally tied to the conservative dynamic RLS
candidate, not to the highest OOF reference. The conservative candidate had the
clearest runtime story:

- exported base artifacts;
- exported RLS state;
- submission entrypoint;
- previous-day lag cache;
- package directories for Kaggle upload and kernel execution.

The package producing `submission.parquet` proved runtime viability. It did not
prove leaderboard performance, because official submissions were disabled.

## Promoted, Preserved, And Rejected Families

| Family | Outcome | Reason |
| --- | --- | --- |
| Raw Ridge | Rejected as candidate, kept as baseline | Too weak and unstable, but useful for diagnostics. |
| Calibrated Ridge | Historical stepping stone | Large improvement over raw Ridge, still below nonlinear models. |
| GBDT / Ridge blend | Historical stepping stone | First meaningful nonlinear lift. |
| Tree engine ensemble | Promoted as active baseline at the time | Strong global and worst-fold improvement. |
| Online linear models | Rejected | Adaptation without representation was insufficient. |
| Direct recurrent models | Rejected/exploratory | Did not beat calibrated Ridge under stronger folds. |
| TabPFN | Rejected/blocked | v2 negative; v3 license blocked. |
| TabM primary models | Preserved as strong component | Strong OOF source, not final standalone solution. |
| Bayesian/gateway meta layers | Promoted structurally | Main jump over tree ensemble. |
| Dynamic RLS | Preserved operationally | Best balance of causality, portability, and robustness. |
| Strong OOF batch prediction context | Preserved as best local Stage 3 | Best Stage 3 score and strongest insight. |
| Historical residual-tail | Preserved as best historical | Highest full historical score, higher runtime complexity. |
| Raw preprocessing direct append | Rejected | Did not improve strong stack. |
| Primary batch-aware TabM | Rejected as final direction | Signal existed but did not beat the stack. |
| Batch DeepSets | Killed | Final structural smoke far below strong baselines. |

## Operational Readiness Levels

| Candidate family | Offline score strength | Runtime readiness | Main blocker |
| --- | --- | --- | --- |
| Conservative dynamic RLS | High, though not highest | Highest | Needs official submission window to verify leaderboard behavior. |
| Batch mean/std fixed blend | Best local Stage 3 | Medium | Stack coefficients and batch prediction logic must be exported. |
| Historical residual-tail | Best historical | Medium-low | Residual coefficients and tail thresholds must be exported and audited online. |
| Tree engine ensemble | Historical baseline | Medium | Weaker than later candidates. |
| Raw batch-aware primary models | Weak-to-medium | Medium | Did not improve final stack. |
| Batch DeepSets | Weak | Low | Failed smoke; not worth packaging. |

## Test And Audit Progression

The test suite expanded as more causal and runtime-sensitive code was added.
Important checkpoints recorded in the audit archive include:

| Area | Recorded test status |
| --- | --- |
| Sequence primary alpha utilities | focused tests passed; full suite recorded `202 passed` |
| Primary alpha sampling / Huber utilities | focused tests passed; full suite recorded `205 passed` |
| Batch prediction expansions | focused strong OOF tests recorded `19 passed`; full suite recorded `208 passed` |
| Raw feature preprocessing | full suite recorded `210 passed` |
| Primary preprocessing bridge | full suite recorded `212 passed` |
| Primary batch-aware TabM | focused TabM tests recorded `19 passed`; full suite recorded `216 passed` before later corrections |
| Final batch-deepset smoke | focused TabM script tests recorded `22 passed` |
| RLS strategy selection and slice audit | recorded `141` to `145` passing tests across related updates |
| Submission packaging | focused submission tests recorded `7 passed` |

These counts are historical checkpoints, not a replacement for running the
current suite:

```bash
uv run pytest -q
```

## Reproduction Guide

### Basic health check

```bash
uv sync
uv run pytest -q
```

### Early baselines

```bash
uv run python scripts/run_zero_baseline.py \
  --n-folds 5 \
  --valid-window 120 \
  --gap 0

uv run python scripts/run_ridge_sweep.py \
  --fold-type rolling \
  --n-folds 5 \
  --train-window 120 \
  --valid-window 60 \
  --alphas 10,100,1000 \
  --chunk-days 10
```

### Tree ensemble control

```bash
uv run python scripts/run_tree_engine_ensemble.py \
  --n-folds 5 \
  --train-window 120 \
  --valid-window 60 \
  --inner-oof-folds 3 \
  --inner-valid-window 20 \
  --engines xgboost,lightgbm \
  --train-sample-frac 0.10 \
  --gbdt-seeds 17,23,37 \
  --max-iter 40 \
  --n-jobs 4 \
  --chunk-days 10 \
  --output-dir reports/experiments/tree_engine_ensemble_xgb_lgb_sample10_seed_ensemble
```

### Conservative dynamic gateway RLS

```bash
uv run python scripts/run_dynamic_gateway_rls_validation.py \
  --output-dir reports/experiments/dynamic_gateway_rls_stage3 \
  --experiment-name dynamic_gateway_rls_stage3
```

Historical confirmation:

```bash
uv run python scripts/run_dynamic_gateway_rls_validation.py \
  --tabm-prediction-dir reports/experiments/competitive_tabm_official_stage3_hist_max1398_5fold_valid60_lags_online_lr1e4_4m_train700_seed37_aux8_preds/validation_predictions \
  --tree-prediction-dir reports/experiments/tree_engine_ensemble_hist_max1398_xgb_lgb_sample10_seed_ensemble_preds/validation_predictions \
  --output-dir reports/experiments/dynamic_gateway_rls_hist_max1398 \
  --experiment-name dynamic_gateway_rls_hist_max1398
```

### Preserved candidates

Use the candidate-specific documentation:

```text
best-candidates/README.md
best-candidates/batch_mean_std_fixed_blend/CODE.md
best-candidates/historical_residual_tail/CODE.md
best-candidates/conservative_dynamic_gateway_rls/CODE.md
```

Those documents list the exact OOF directories, flags, and artifact paths needed
to reproduce the preserved results.

## What Was Learned

1. Ridge and online linear models are valuable baselines, but not competitive
   primary candidates.
2. Calibration and clipping matter, but their gains saturate quickly.
3. Tree engines and TabM provide the strongest primary prediction components.
4. Causal meta-learning over OOF predictions was the main leap from `~0.009` to
   `~0.014`.
5. Batch/cross-sectional context in prediction space is real and transferable.
6. Raw batch preprocessing contains weak signal, but the tested primary-model
   integrations did not beat the strong OOF/RLS stack.
7. Residual-tail corrections can improve historical validation, but increase
   runtime and regime risk.
8. Results above `0.02` appeared only in probes, folds, slices, or leaky
   diagnostics; they were not global validated results.
9. Continuing to open knobs over the same saved OOFs would raise data-snooping
   risk more than it would likely create a real score jump.

## Stop Condition

The final set-aware primary model failed the pre-defined standard for reopening
the score search:

```text
primary_batch_deepset_smoke_400k_seed37_v1
global_r2=0.003538596379
```

Because it did not show a plausible path toward `~0.016` before scaling, the
scientific decision was to stop local score optimization, preserve the best
references, and document the research line.

Future work should resume only with a material change in information, such as:

- new allowed causal data;
- a new official evaluation opportunity;
- an independent bug or leakage-reversal finding;
- a genuinely different primary architecture that beats strong baselines under
  temporal folds before expensive scaling.
