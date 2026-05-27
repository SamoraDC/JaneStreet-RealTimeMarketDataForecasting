# Best Candidates Reproducibility Guide

This directory preserves the three best reference candidates from the current
Jane Street research line. Each candidate has its own documentation, generated
artifacts, and code snapshot so the result can be audited independently from the
rest of the experiment history.

## Candidates

| Directory | Preserved reference | Reported score |
| --- | --- | --- |
| `batch_mean_std_fixed_blend/` | `strong_oof_subset_s23aux8_s17_gateway_batch_mean_std_stage3_narrow_v1/fixed_blend_0_w0p75_fixed_blend` | Stage 3 `global_r2=0.014424968604` |
| `historical_residual_tail/` | `strong_oof_hist_max1398_gateway_residual_tail_modes_v1/gateway_risk_conservative_rls_abs_pred_s100_prediction_residual_weight_and_abs_q0p95_residual_tail` | Historical `global_r2=0.015630171202` |
| `conservative_dynamic_gateway_rls/` | `dynamic_gateway_rls_experts_alpha10000_f0p995` | Stage 3 `global_r2=0.013836465`; historical `global_r2=0.015425344` |

## Reproducibility Status

These candidates are reproducible from a full repository checkout, but this
directory is not a standalone data bundle.

The versioned files in `best-candidates/` are enough to:

- inspect the preserved reports, parameter files, CSV summaries, and audit
  payloads;
- review the code that generated each candidate family;
- compare the preserved metrics against the documented candidate names;
- understand the assumptions, validation setup, and runtime limitations.

Full regeneration of the scores requires additional local inputs that are not
committed to Git:

- the original Jane Street Kaggle data under `data/raw/`;
- the saved OOF prediction directories referenced by each candidate's
  `CODE.md`;
- the project Python environment installed with `uv`;
- enough local compute to rebuild the referenced base predictions when those OOF
  directories are missing.

## Required Setup

Run all commands from the repository root, not from inside `best-candidates/`.

```bash
uv sync
uv run pytest -q
```

Expected repository-side inputs:

```text
data/raw/
reports/experiments/
multi-models/reports/
artifacts/
best-candidates/
```

The raw Kaggle data is intentionally not included in this repository. It must be
downloaded separately according to the competition license and placed under the
same local layout expected by `src/janestreet/paths.py`.

## Directory Contract

Each candidate directory follows the same top-level contract:

```text
README.md      Detailed method explanation.
ARTIFACTS.md   Inventory of preserved outputs and what each file means.
CODE.md        Code map and reproduction commands.
code/          Snapshot of the relevant source files.
artifacts/     Preserved candidate outputs, reports, and audit files.
validation/    Validation outputs when applicable.
```

The `code/` folders are review snapshots. The documented commands in `CODE.md`
use root-relative project paths because the OOF inputs, raw data, and generated
reports are organized at repository level.

## Reproduction Modes

### 1. Audit-only reproduction

Use this when the raw Kaggle data or OOF prediction directories are unavailable.

What can be checked:

- candidate names and reported scores;
- fold-level summaries;
- parameter files;
- audit JSON payloads;
- generated report text;
- package contents for the conservative Kaggle runtime.

This mode verifies that the preserved result is internally documented and
traceable, but it does not recompute the score from raw predictions.

### 2. Local score regeneration

Use this when the full local experiment workspace is available.

General flow:

```bash
uv sync
uv run pytest -q
```

Then run the command listed in the candidate's `CODE.md`.

The two strong OOF candidates depend on saved OOF prediction directories:

- `batch_mean_std_fixed_blend/CODE.md`
- `historical_residual_tail/CODE.md`

The conservative dynamic RLS candidate has separate commands for Stage 3
validation, historical validation, RLS artifact export, and Kaggle package
build:

- `conservative_dynamic_gateway_rls/CODE.md`

### 3. Runtime package reproduction

Only `conservative_dynamic_gateway_rls/` currently includes a Kaggle-style
runtime package:

```text
conservative_dynamic_gateway_rls/kaggle_upload/
conservative_dynamic_gateway_rls/kaggle_kernel/
```

The two strong OOF references are research/reference candidates, not complete
Kaggle runtime packages. To make them submissible, their OOF-only logic would
need to be exported into the online `predict(test, lags)` runtime, including
their stack coefficients, thresholds, batch features, residual-tail rules, and
fixed blend logic.

## Candidate-specific Inputs

### `batch_mean_std_fixed_blend/`

Requires:

- Stage 3 TabM OOF predictions for seed 23 auxiliary 8;
- Stage 3 TabM OOF predictions for seed 17;
- tree and baseline prediction sources expected by the strong OOF pipeline;
- the root `multi-models/run_strong_oof_experiment.py` command shown in
  `CODE.md`.

Primary preserved score:

```text
global_r2=0.014424968604
```

### `historical_residual_tail/`

Requires:

- historical `max_date_id=1398` TabM OOF predictions;
- historical `max_date_id=1398` tree ensemble OOF predictions;
- residual-tail mode generation enabled through the strong OOF CLI.

Primary preserved score:

```text
global_r2=0.015630171202
```

### `conservative_dynamic_gateway_rls/`

Requires:

- Stage 3 OOF predictions for local validation;
- historical `max_date_id=1398` OOF predictions for historical confirmation;
- export of `meta_rls_experts_alpha10000_f0p995`;
- optional Kaggle package build if reproducing the runtime artifacts.

Primary preserved scores:

```text
stage3_global_r2=0.013836465
historical_global_r2=0.015425344
```

## Validation Standard

For this repository, a result should be treated as reproduced only when the
following match the preserved documentation:

- the candidate name;
- the validation regime;
- the OOF input directories;
- the feature flags and hyperparameters;
- the fold-level scores;
- the final weighted zero-mean global R2.

Small formatting differences in generated reports are acceptable. Metric or
fold differences are not acceptable unless they are explained by an intentional
change in code, data, or validation regime.

## Known Limitations

- `best-candidates/` preserves the candidate evidence, but does not include the
  licensed Kaggle raw data.
- OOF prediction directories can be large and are expected to live in the normal
  project report paths, not inside this directory.
- The strong OOF candidates are not directly submissible to Kaggle yet.
- The conservative dynamic RLS candidate is the operationally closest candidate
  because it includes exported runtime artifacts and package directories.
- If root source files change after this preservation step, use the candidate
  `code/` snapshots to audit what implementation was intended for each result.
