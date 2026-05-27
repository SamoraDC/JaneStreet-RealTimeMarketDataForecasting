# Scientific Methodology

This document explains the scientific methodology behind the Jane Street
research project: how hypotheses were formed, how tests were designed, how
temporal validation was enforced, how look-ahead bias was avoided, and how
results were promoted or rejected.

It is intentionally deeper than a reproduction guide. The goal is to make the
research process auditable, not just the final scores.

## Research Objective

The competition task is to predict `responder_6` for a stream of market-like
rows. The main local objective is the weighted zero-mean R2:

```text
R2 = 1 - sum_i w_i * (y_i - p_i)^2 / sum_i w_i * y_i^2
```

where:

- `y_i` is the realized target;
- `p_i` is the prediction;
- `w_i` is the competition weight;
- the zero predictor has `R2=0.0`.

The scientific objective was not only to maximize local score. It was to find
candidate families that were:

- temporally causal;
- reproducible;
- robust across folds and historical regimes;
- compatible with the Kaggle gateway runtime;
- honest about uncertainty and data-snooping risk.

## Core Scientific Principles

### 1. Time Is The Primary Split

Financial data is nonstationary. Random row splits are not valid evidence for
this kind of task because they mix regimes and leak future distributional
information into the training set. All promoted results use temporal validation
over `date_id`.

### 2. Controls Matter More Than Raw Scores

Every candidate had to be compared against the correct active control for its
stage. A score that beats an old baseline but loses to the current active
baseline is not a promoted result.

Examples:

- calibrated Ridge replaced raw Ridge as the linear control;
- the tree engine ensemble replaced the older sklearn GBDT blend;
- dynamic RLS and strong OOF candidates had to beat or explain themselves
  relative to the gateway/RLS family, not just the original tree ensemble.

### 3. Promoted Claims Need Full Validation

Smoke tests and probes were used to reduce cost, but they were not treated as
final evidence. A candidate could be explored after a smoke, but it needed a
larger temporal validation before promotion.

### 4. Diagnostics Are Not Score Claims

Slice diagnostics, daily oracles, in-sample scores, and leaky upper bounds were
allowed as instruments for understanding failures. They were not valid
performance claims.

### 5. Kill Rules Prevent Validation-set Overfitting

Many families were killed even when they had isolated positive slices. This was
deliberate. In a high-dimensional research space, continuing to tune a weak
family after repeated failure usually increases data-snooping risk more than it
increases real signal.

## Validation Regimes

The project used several validation regimes. They were not interchangeable.

| Regime | Purpose | Promotion strength |
| --- | --- | --- |
| Smoke | Check that a pipeline runs and has a plausible signal. | Low |
| Probe | Test a narrow idea cheaply. | Low |
| Stage 0 / Stage 1 | Small temporal screens, often fewer folds or fewer rows. | Low-to-medium |
| Stage 2 | More serious partial validation. | Medium |
| Stage 3 | Main full local validation protocol. | High |
| Historical `max_date_id=1398` | Earlier-regime confirmation. | High when paired with Stage 3 |
| Gateway/runtime package | Operational feasibility. | Runtime evidence, not leaderboard evidence |

The final preserved candidates were chosen because they had strong evidence in
Stage 3, historical validation, or operational packaging.

## Rolling Windows

### Definition

The main rolling validation protocol uses fixed-length training and validation
windows over `date_id`.

For each fold:

```text
valid_start = first_valid_start + fold_index * valid_window
valid_end   = valid_start + valid_window - 1
train_end   = valid_start - gap - 1
train_start = train_end - train_window + 1
```

The key invariant is:

```text
train_end < valid_start
```

This prevents direct future leakage.

### Why Rolling Windows Were Used

Rolling windows were used because the data is regime-dependent. A model trained
on all historical dates may not represent how a deployable model behaves when it
must adapt to recent regimes. A fixed recent training window creates a stricter
test of local temporal adaptation.

The common Stage 3 protocol used:

```text
n_folds=5
valid_window=60
```

This means the external validation covers five sequential 60-day blocks, or
roughly 300 validation days in total.

### What Rolling Windows Do Not Prove

Rolling windows do not perfectly reproduce the hidden leaderboard. They are an
offline approximation of a streaming future. They can test temporal causality
and relative candidate quality, but they cannot guarantee official leaderboard
ranking.

## Walk-forward Validation

Walk-forward validation means that each validation block is predicted using only
models, parameters, and meta-parameters learned from earlier data.

This rule applied at several levels:

- primary model training;
- blend weight fitting;
- calibration and clipping rules;
- residual corrections;
- tail thresholds;
- gateway/RLS state updates.

For stack or residual layers, the first fold often has no earlier fold to learn
from. In that case, the valid behavior is an identity fallback or a fixed prior,
not a model fit on the fold being scored.

## Expanding Windows

Expanding folds were also supported. In an expanding fold:

```text
train_start = min_date_id
train_end   = valid_start - gap - 1
```

The training set grows over time. Expanding windows are useful when the model
should learn from all available history. Rolling windows are stricter when
recent regime relevance is more important than long-history coverage.

## Gaps

The fold generator supports a temporal `gap`:

```text
train_end = valid_start - gap - 1
```

A gap can reduce contamination when adjacent dates are too correlated or when
features have delayed availability. Many validated experiments used `gap=0`
because the key operational safeguards were enforced elsewhere:

- official lags came from previous `date_id`;
- gateway/RLS updates used previous-day responder lags;
- batch features were target-free and computed only from the current observable
  batch;
- stack and residual layers used previous-fold fitting.

`gap=0` is valid only when the feature and update logic still respects causal
availability.

## Nested OOF Validation

Several models used internal OOF calibration inside an outer temporal fold.

The outer fold answers:

```text
Does the final candidate predict unseen future dates?
```

The inner OOF split answers:

```text
How should we fit blend weights, calibrators, or model selectors without using
the outer validation target?
```

This was used for:

- Ridge/GBDT blends;
- tree engine ensemble weights;
- simplex blending;
- calibration layers;
- some gate and selector variants.

The rule is simple: the outer validation target must not be used to fit the
parameter that is evaluated on that same outer validation block.

## Out-of-fold Prediction Artifacts

OOF prediction artifacts are central to this repository.

An OOF artifact contains predictions for validation rows where each prediction
was generated by a model that did not train on those rows. These artifacts make
it possible to test meta-models without retraining every primary model.

OOF artifacts were used for:

- tree/TabM/Ridge expert combination;
- Bayesian/gateway meta layers;
- dynamic RLS;
- strong OOF stacks;
- batch prediction expansions;
- residual-tail experiments.

The key scientific requirement is that each OOF column must remain aligned with
the fold, `date_id`, `time_id`, and `symbol_id` that generated it. Joining OOFs
incorrectly can silently create invalid scores.

## Anti-lookahead Bias Protocol

Look-ahead bias occurs when a model uses information that would not have been
available at prediction time. This project guarded against several forms.

### Target Leakage

Target leakage means using `responder_6` or any direct derivative of the current
target while predicting the current row.

Forbidden examples:

- using current validation `responder_6` to fit a calibrator;
- using residuals from the current fold to choose a residual correction;
- computing target-based groups on the validation fold and scoring the same
  fold.

Allowed examples:

- computing target-based diagnostics after scoring;
- fitting residual corrections on earlier folds and applying them to later
  folds;
- using previous-day responder lags when they match the gateway contract.

### Responder Leakage

Responder leakage is a specific target leakage risk in this competition because
lagged responders are part of the API. A responder is allowed only when the API
would have delivered it.

The local rule was:

```text
current date prediction may use responder lags from previous dates,
not current-date responders.
```

### Feature Look-ahead

Feature look-ahead can happen even without targets. Examples:

- using a future row from the same symbol to build a rolling statistic;
- computing a global rank over the full dataset;
- using validation distribution quantiles to transform validation features.

Allowed feature operations must be based on:

- current row values;
- current observable batch values;
- past rows only;
- training-window statistics fit before validation.

### Selection Leakage

Selection leakage happens when many candidates are tried and the best validation
one is reported as if it were pre-registered.

The project handled this by:

- reporting grids rather than hiding losing candidates;
- requiring historical confirmation for selected rules when possible;
- labeling daily oracles and leaky upper bounds as diagnostics;
- preserving conservative candidates when aggressive local choices were less
  stable.

### Post-hoc Slice Leakage

A bad slice can explain a fold failure, but removing it from the score after the
fact is invalid. Slice diagnostics were used to generate new hypotheses, not to
edit the validation set.

The `rw_02` Ridge failure is the main example: the problematic
`date_id=1489,symbol_id=25` slice explained much of the loss, but the valid
score still included it.

## Gateway Simulation

The competition runtime predicts in batches through a gateway-style API. The
local simulation followed this logic:

1. At the start of `date_id=D`, receive lagged responders for previous dates.
2. Join those lagged responders to cached predictions/features from previous
   dates.
3. Update the online meta-model or RLS state using only that past information.
4. Predict the current batch.
5. Cache current batch predictions/features for a future update.

The current target for `D` is not available before predicting `D`.

The gateway/RLS audits checked:

```text
bad_updates=0
all_strictly_past=true
```

These checks do not prove leaderboard performance, but they are essential for
operational causality.

## Dynamic RLS Methodology

The dynamic RLS meta-layer models expert predictions as a feature vector:

```text
x_i = [tabm_prediction, tree_prediction, xgboost_prediction,
       lightgbm_prediction, ridge_calibrated_prediction, ...]
```

The state is represented by a precision-like matrix and right-hand side:

```text
P_t
b_t
beta_t = solve(P_t, b_t)
```

With forgetting factor `lambda`, the causal update is:

```text
P_t = lambda * P_{t-1} + X_{past}' W_{past} X_{past}
b_t = lambda * b_{t-1} + X_{past}' W_{past} y_{past}
```

Prediction uses the current expert vector:

```text
p_i = x_i beta_t
```

The preserved conservative candidate used:

```text
feature_set=experts
ridge_alpha=10000
forgetting_factor=0.995
```

This setting was preserved because it balanced local performance, historical
confirmation, and runtime simplicity.

## Batch Feature Methodology

Batch features were allowed only when they were computable from the current
observable batch and did not use current targets.

For a prediction column `p` inside a `date_id,time_id` batch:

```text
batch_mean(p)   = mean of p inside the current batch
batch_demean(p) = p - batch_mean(p)
batch_std(p)    = standard deviation of p inside the current batch
batch_zscore(p) = batch_demean(p) / guarded_batch_std(p)
batch_rank(p)   = rank of p inside the current batch
```

These features are causal if all input predictions are available before the
submission for that batch. They are not causal if they depend on targets,
future batches, or full-dataset ranks.

The best local Stage 3 candidate came from applying this idea to expert
predictions, not directly to raw targets.

## Residual-tail Methodology

Residual-tail experiments tried to apply corrections only where expected
weighted error was largest.

A typical residual-tail rule has three parts:

1. A base prediction, such as conservative RLS with risk shrinkage.
2. A residual correction trained on earlier folds.
3. A tail mask whose thresholds are also fit on earlier folds.

The best historical candidate used a `weight_and_abs` mask:

```text
apply correction if:
  weight is in the previous-fold high tail
  and abs(base_prediction) is in the previous-fold high tail
```

This is more complex than dynamic RLS because it requires exporting:

- residual coefficients;
- tail thresholds;
- tail mask semantics;
- exact base prediction logic.

That is why it was preserved as the best historical reference, not as the lowest
operational-risk candidate.

## Test Design

The project used several kinds of tests.

### Unit Tests

Unit tests covered deterministic code paths:

- metric math;
- fold boundaries;
- calibration behavior;
- feature construction;
- lag reconstruction;
- submission artifact loading;
- submission inference behavior.

### Script Tests

Script-level tests verified that CLI entrypoints parsed arguments, wrote
expected artifacts, and enforced important constraints.

### Artifact Audits

Experiment artifacts recorded:

- candidate summaries;
- fold scores;
- parameter files;
- daily gateway audits;
- report JSON payloads;
- leakage and causality status.

### Row-level Reproduction

For some sensitive operations, aggregate results were reproduced at row level to
ensure that a reported improvement was not only an aggregation artifact.

## Promotion Criteria

A candidate could be promoted or preserved when it satisfied most of:

- better `global_r2` than the correct active baseline;
- non-catastrophic `min_fold_r2`;
- consistent fold behavior or a clear explanation for fold concentration;
- clean target-leakage and fold-causality checks;
- reproducible command and artifacts;
- no reliance on validation-only selection;
- plausible gateway/runtime path.

The threshold was stricter for final candidates than for intermediate research
bridges.

## Kill Criteria

A family was killed when one or more of the following happened:

- it lost to the active baseline under comparable folds;
- improvement appeared only in one fold or one slice;
- it required current targets or validation targets to choose parameters;
- it created worse worst-fold behavior without a compensating reason;
- it became a hyperparameter search over the same validation artifacts;
- scaling cost was high while smoke evidence was weak;
- it could not plausibly be exported into the gateway runtime.

## Handling Multiple Comparisons

The project tested many ideas. That creates multiple-comparison risk: some
variant may win by chance.

Mitigations included:

- preserving all major negative results;
- separating probes from full validations;
- using historical confirmation when possible;
- preferring simple frozen rules over highly selected rules;
- labeling oracles as leaky diagnostics;
- maintaining conservative operational references beside higher-scoring
  experimental references.

## Why The Search Was Stopped

The search was stopped because the final structural hypothesis, a set-aware
`batch_deepset` primary model, failed a five-fold smoke:

```text
global_r2=0.003538596379
min_fold_r2=0.000237110623
```

At that point, the validated score had saturated near:

```text
Stage 3 best:      global_r2=0.014424968604
Historical best:  global_r2=0.015630171202
```

Continuing to tune the same saved OOFs would likely increase data-snooping risk
more than it would create real alpha. The scientific decision was to preserve
the best references and document the research line.

## Reproduction Checklist

Before treating a reproduced result as valid, check:

1. The same raw data and OOF prediction directories are available.
2. The same candidate name is generated.
3. The same validation regime is used.
4. Fold rows and weight sums match.
5. Fold-level R2 values match within expected numerical tolerance.
6. The final global weighted zero-mean R2 matches.
7. Leakage/audit fields remain clean.
8. Any difference in code, data, or validation protocol is explicitly stated.

## Publication Standard

Public documentation should not imply that local validation equals official
leaderboard performance. The correct public claim is:

```text
These are causal local and historical validation results, reproducible from the
full project checkout when the required data and OOF artifacts are available.
```

The Kaggle package producing `submission.parquet` is runtime evidence. It is not
an official leaderboard result because submissions were disabled at the time the
package was prepared.
