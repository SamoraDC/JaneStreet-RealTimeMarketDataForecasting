# Raw Data Geometry & Operational Time Experiments

This directory is isolated for experiments that study whether `time_id` should be treated as clock time, trading time, event time, or an artificial row grid.

Rules:

- Do not write large transformed parquet datasets here.
- Start with aggregates and sampled model runs.
- Keep final predictions on the original Jane Street rows.
- Treat target/responder-derived clocks as diagnostics only, not online features.
- Compare any promoted feature set against `tree_engine_ensemble_xgb_lgb_sample10_seed_ensemble` (`global_r2=0.008946915`).

Initial commands:

```bash
uv run python scripts/run_time_geometry_audit.py --recent-days 180 --n-features 24 --output-dir experiments/time_geometry/reports/audit_recent180
uv run python scripts/run_clock_property_comparison.py --recent-days 180 --n-features 32 --bucket-count 20 --output-dir experiments/time_geometry/reports/clock_property_recent180
uv run python scripts/run_operational_time_feature_baseline.py --engine lightgbm --feature-set clock --n-folds 2 --valid-window 20 --train-sample-frac 0.05 --output-dir experiments/time_geometry/reports/lightgbm_clock_control_2fold
uv run python scripts/run_operational_time_feature_baseline.py --engine lightgbm --feature-set operational --n-folds 2 --valid-window 20 --train-sample-frac 0.05 --output-dir experiments/time_geometry/reports/lightgbm_operational_2fold
uv run python scripts/run_clock_tournament.py --n-folds 5 --train-window 120 --valid-window 20 --inner-oof-folds 2 --inner-valid-window 20 --engines xgboost,lightgbm --train-sample-frac 0.05 --gbdt-seeds 17 --max-iter 40 --learning-rate 0.03 --max-leaf-nodes 31 --n-jobs 2 --chunk-days 10 --clock-bucket-count 20 --min-group-rows 2000 --output-dir experiments/time_geometry/reports/clock_tournament_stage2_5fold_sample05
uv run python scripts/run_clock_tournament.py --n-folds 5 --train-window 120 --valid-window 60 --inner-oof-folds 3 --inner-valid-window 20 --engines xgboost,lightgbm --train-sample-frac 0.10 --gbdt-seeds 17,23,37 --max-iter 80 --learning-rate 0.03 --max-leaf-nodes 31 --n-jobs 2 --chunk-days 10 --clock-bucket-count 20 --clock-candidates row_activity,batch_missing --min-group-rows 2000 --output-dir experiments/time_geometry/reports/clock_tournament_stage3_5fold60_sample10_seed_ensemble_iter80
uv run python scripts/analyze_clock_tail_switch.py --input experiments/time_geometry/reports/clock_tournament_stage3_5fold60_sample10_seed_ensemble_iter80/clock_tournament_weight_bucket_by_fold.csv --output-dir experiments/time_geometry/reports/clock_tail_switch_stage3_iter80_q90_q100 --candidate row_activity:clock_simplex --candidate batch_missing:clock_simplex --tail-buckets q90_q99,q99_q100
uv run python scripts/run_clock_tournament.py --n-folds 5 --train-window 120 --valid-window 60 --inner-oof-folds 3 --inner-valid-window 20 --engines xgboost,lightgbm --train-sample-frac 0.10 --gbdt-seeds 17,23,37 --max-iter 80 --learning-rate 0.03 --max-leaf-nodes 31 --n-jobs 2 --chunk-days 10 --clock-bucket-count 20 --clock-candidates row_activity,batch_missing --min-group-rows 2000 --output-dir experiments/time_geometry/reports/clock_tournament_stage3_5fold60_sample10_seed_ensemble_iter80_tail_rowlevel
uv run python scripts/run_frozen_tail_holdout.py --fold-name pre_stage3_holdout --train-start 1219 --train-end 1338 --valid-start 1339 --valid-end 1398 --inner-oof-folds 3 --inner-valid-window 20 --engines xgboost,lightgbm --train-sample-frac 0.10 --gbdt-seeds 17,23,37 --max-iter 80 --learning-rate 0.03 --max-leaf-nodes 31 --n-jobs 2 --chunk-days 10 --clock-bucket-count 20 --min-group-rows 2000 --output-dir experiments/time_geometry/reports/frozen_tail_holdout_pre_stage3
```

Current sample evidence:

- Recent 180 days have a perfectly regular `time_id` grid: 968 buckets per day and date-symbol-time completeness `1.0`.
- Target energy is intraday-position dependent: `time_id_target_energy_cv=0.569375`, with `time_id=0..4` highest in the recent audit.
- Alternative activity clocks have better diagnostic balance than raw clock buckets: `batch_activity_rank_bucket` has `energy_cv=0.177924` versus `clock_time_bucket=0.548222`; `row_activity_rank_bucket` and `symbol_activity_ewm_rank_bucket` are also near `0.20`.
- Better clock geometry is not yet better alpha. The model ablation below did not justify promotion to the production ensemble.
- LightGBM operational clocks worsened the 2-fold sample (`0.002346` vs clock-control `0.002547`).
- XGBoost operational clocks gave a tiny 5% sample gain (`0.002583` vs `0.002358`), but at 10% sampling were effectively tied on global and worse on `min_r2` (`0.004439`, `min=0.002737` vs clock-control `0.004438`, `min=0.002912`).
- Clock tournament Stage 2: as a gating layer over Ridge/XGBoost/LightGBM predictions, `row_activity + clock_simplex` beat the local 5-fold control (`global_r2=0.003720451` vs `0.003527437`), with a small `min_r2` improvement. `batch_missing + clock_simplex` was second (`global_r2=0.003646137`) and had a larger `min_r2` improvement. Raw `clock_time + clock_simplex` underperformed the control.
- Clock tournament Stage 3: against the active 10% seed ensemble, no clock improved the global score. The reproduced control scored `global_r2=0.008946910`; best clock variant was `row_activity + clock_predabs_shrink` at `0.008886691`. `row_activity + clock_simplex` improved `q99_q100` (`0.008485` vs `0.008072`) but degraded global score.
- Tail-switch analysis from additive weight-bucket aggregates found a useful constrained bridge: keep the base ensemble outside high-weight buckets and use `batch_missing + clock_simplex` only on `q90_q99,q99_q100`. This scored `global_r2=0.009010271`, `min_fold_r2=0.004536404`, above the base control.
- Row-level confirmation in `run_clock_tournament.py` reproduced the same result: `batch_missing + clock_simplex_tail_q90_q100` scored `global_r2=0.009010271128322644`, `mean_r2=0.008865908294896729`, and `min_r2=0.004536403701327352`.
- Frozen pre-discovery holdout (`train=1219..1338`, `valid=1339..1398`) kept the tail rule positive: `batch_missing_tail_q90_q100=0.006551490` vs base `0.006280330`. The direct `batch_missing_clock_simplex` was higher in this earlier regime (`0.007021577`) but failed in later Stage 3, so it remains rejected as global gating.
- Gateway observability audit confirmed `batch_missing_frac` is exactly computable from the served `(date_id,time_id)` batch: `58,080` batches and `max_abs_diff=0.0` versus the tournament values.
- First regime/confidence auxiliary (`tail_advantage`, fitted on calibration by `clock_bucket`) did not improve the fixed tail rule on the holdout: both scored `0.006551490`. Keep the utility as infrastructure, not as a promoted strategy.
- Decision: do not promote simple operational clocks as direct model features or global gating. Promote `batch_missing + clock_simplex` only as a high-weight tail-control candidate for a faithful online pipeline simulation.
